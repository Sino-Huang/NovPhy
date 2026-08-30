"""Pair-expert training and shared-versus-expert utility-map analysis."""
from __future__ import annotations

import math
from typing import Final

from world_model.model import PredictionPair, identity
from world_model.training.cohort_v2_evaluation import CohortV2EvaluationResult
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    MACRO_PAIRS,
    CohortV2MacroConfig,
    CohortV2MacroError,
    CohortV2MacroPairScorer,
    CohortV2MacroTrainer,
    CohortV2MacroTrainingData,
)


PAIR_EXPERT_OBJECTIVE_SCHEMA: Final = "cohort_v2_pair_expert_objective_v1"
PAIR_MAP_COMPARISON_SCHEMA: Final = "cohort_v2_pair_expert_map_comparison_v1"


def pair_label(pair: PredictionPair) -> str:
    return f"h{pair.delta}/{pair.abstraction}"


class CohortV2PairExpertTrainingData(CohortV2MacroTrainingData):
    """One pair's exact minibatches from the shared model's balanced schedule."""

    def __init__(
        self,
        reader,
        config: CohortV2MacroConfig,
        pair: PredictionPair,
        *,
        shared_step_count: int,
    ) -> None:
        if pair not in MACRO_PAIRS:
            raise CohortV2MacroError("pair expert targets a pair outside the central grid")
        if type(shared_step_count) is not int or shared_step_count <= 0:
            raise CohortV2MacroError("shared step count must be positive")
        super().__init__(reader, config)
        self.pair = pair
        self.shared_step_count = shared_step_count
        self.shared_steps = tuple(
            step
            for step in range(shared_step_count)
            if super(CohortV2PairExpertTrainingData, self).schedule_at(step) == pair
        )
        if len(self.shared_steps) != config.steps:
            raise CohortV2MacroError(
                "pair-expert steps must equal its updates in the shared balanced schedule"
            )

    def schedule_at(self, step: int) -> PredictionPair:
        if type(step) is not int or not 0 <= step < len(self.shared_steps):
            raise CohortV2MacroError("pair-expert step is outside its declared schedule")
        return self.pair

    def batch_at(self, pair: PredictionPair, step: int):
        if pair != self.pair:
            raise CohortV2MacroError("pair expert cannot draw another pair's minibatch")
        return super().batch_at(pair, self.shared_steps[step])


class CohortV2PairExpertTrainer(CohortV2MacroTrainer):
    """Train one independently parameterized predictor on one declared pair."""

    data: CohortV2PairExpertTrainingData

    def __init__(
        self, data: CohortV2PairExpertTrainingData, config: CohortV2MacroConfig
    ) -> None:
        super().__init__(data, config)

    def _learning_rate(self, step: int) -> float:
        shared_step = self.data.shared_steps[step]
        progress = shared_step / max(1, self.data.shared_step_count - 1)
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


class CohortV2PairExpertScorer:
    """Route each grid pair to its independently trained predictor."""

    capabilities = MACRO_CAPABILITIES

    def __init__(
        self,
        scorers: tuple[tuple[PredictionPair, CohortV2MacroPairScorer], ...],
    ) -> None:
        if tuple(pair for pair, _ in scorers) != MACRO_PAIRS:
            raise CohortV2MacroError("pair-expert scorers must cover the ordered central grid")
        if any(scorer.capabilities != MACRO_CAPABILITIES for _, scorer in scorers):
            raise CohortV2MacroError("pair-expert scorer capabilities differ")
        self.scorers = dict(scorers)
        bindings = tuple(
            (pair.identity, scorer.checkpoint_identity, scorer.objective_identity)
            for pair, scorer in scorers
        )
        self.checkpoint_identity = identity((
            "cohort-v2-pair-expert-bundle-v1",
            tuple((pair_identity, checkpoint) for pair_identity, checkpoint, _ in bindings),
        ))
        self.objective_identity = identity((PAIR_EXPERT_OBJECTIVE_SCHEMA, bindings))

    def objective(self, window, pair: PredictionPair) -> float:
        return self.scorers[pair].objective(window, pair)

    def objective_batch(self, windows, pair: PredictionPair) -> tuple[float, ...]:
        return self.scorers[pair].objective_batch(windows, pair)


def _state_index(evaluation: CohortV2EvaluationResult):
    return {state.state_id: state for state in evaluation.states}


def compare_preferred_pair_maps(
    shared: CohortV2EvaluationResult,
    experts: CohortV2EvaluationResult,
    *,
    exposure_role: str = "model_selection",
) -> dict[str, object]:
    """Quantify held-out preferred-pair changes and per-pair objective gains."""
    if (
        shared.release_identity != experts.release_identity
        or shared.capability_declaration_identity != experts.capability_declaration_identity
        or shared.partition_identity != experts.partition_identity
        or shared.grid.identity != experts.grid.identity
        or shared.state_set_identity != experts.state_set_identity
    ):
        raise CohortV2MacroError("shared and pair-expert evaluations have different data bindings")
    shared_by_state = _state_index(shared)
    expert_by_state = _state_index(experts)
    if shared_by_state.keys() != expert_by_state.keys():
        raise CohortV2MacroError("shared and pair-expert evaluations cover different states")

    selected_counts = {
        "shared": {pair_label(pair): 0 for pair in MACRO_PAIRS},
        "pair_experts": {pair_label(pair): 0 for pair in MACRO_PAIRS},
    }
    transitions: dict[tuple[str, str], int] = {}
    scenario_counts: dict[str, list[int]] = {}
    objective_gains: dict[PredictionPair, list[float]] = {
        pair: [] for pair in MACRO_PAIRS
    }
    changed = 0
    state_count = 0
    for state_id, shared_state in shared_by_state.items():
        expert_state = expert_by_state[state_id]
        if shared_state.exposure_role != expert_state.exposure_role:
            raise CohortV2MacroError("shared and pair-expert state roles differ")
        if shared_state.exposure_role != exposure_role:
            continue
        if shared_state.scenario_lineage_identity != expert_state.scenario_lineage_identity:
            raise CohortV2MacroError("shared and pair-expert scenario lineages differ")
        if shared_state.selected_pair is None or expert_state.selected_pair is None:
            raise CohortV2MacroError("held-out state has no available preferred pair")
        state_count += 1
        shared_label = pair_label(shared_state.selected_pair)
        expert_label = pair_label(expert_state.selected_pair)
        selected_counts["shared"][shared_label] += 1
        selected_counts["pair_experts"][expert_label] += 1
        transitions[(shared_label, expert_label)] = (
            transitions.get((shared_label, expert_label), 0) + 1
        )
        scenario = scenario_counts.setdefault(
            shared_state.scenario_lineage_identity, [0, 0]
        )
        scenario[0] += 1
        if shared_state.selected_pair != expert_state.selected_pair:
            changed += 1
            scenario[1] += 1

        for shared_outcome, expert_outcome in zip(
            shared_state.outcomes, expert_state.outcomes, strict=True
        ):
            if (
                shared_outcome.pair != expert_outcome.pair
                or shared_outcome.target_frame_record_identity
                != expert_outcome.target_frame_record_identity
                or shared_outcome.available != expert_outcome.available
            ):
                raise CohortV2MacroError(
                    "shared and pair-expert outcome eligibility differs"
                )
            if shared_outcome.available:
                objective_gains[shared_outcome.pair].append(
                    float(shared_outcome.objective) - float(expert_outcome.objective)
                )
    if not state_count:
        raise CohortV2MacroError(f"evaluation has no {exposure_role} states")

    per_pair = []
    for pair in MACRO_PAIRS:
        gains = objective_gains[pair]
        label = pair_label(pair)
        per_pair.append({
            "available_state_count": len(gains),
            "mean_shared_minus_expert_objective": (
                sum(gains) / len(gains) if gains else None
            ),
            "pair": label,
            "pair_expert_preferred_count": selected_counts["pair_experts"][label],
            "preferred_count_change": (
                selected_counts["pair_experts"][label]
                - selected_counts["shared"][label]
            ),
            "shared_preferred_count": selected_counts["shared"][label],
        })
    return {
        "changed_preferred_pair_count": changed,
        "changed_preferred_pair_rate": changed / state_count,
        "exposure_role": exposure_role,
        "interpretation": (
            "positive shared-minus-expert objective indicates a pair-specific gain "
            "after removing shared-training interference"
        ),
        "per_pair": per_pair,
        "preferred_pair_transitions": [
            {"count": count, "pair_expert": target, "shared": source}
            for (source, target), count in sorted(transitions.items())
        ],
        "scenario_lineages": [
            {
                "changed_preferred_pair_count": counts[1],
                "changed_preferred_pair_rate": counts[1] / counts[0],
                "scenario_lineage_identity": lineage,
                "state_count": counts[0],
            }
            for lineage, counts in sorted(scenario_counts.items())
        ],
        "schema": PAIR_MAP_COMPARISON_SCHEMA,
        "state_count": state_count,
    }


__all__ = [
    "CohortV2PairExpertScorer",
    "CohortV2PairExpertTrainer",
    "CohortV2PairExpertTrainingData",
    "PAIR_EXPERT_OBJECTIVE_SCHEMA",
    "PAIR_MAP_COMPARISON_SCHEMA",
    "compare_preferred_pair_maps",
    "pair_label",
]
