"""Broad action probes and deterministic ensemble ranking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch

from world_model.model import Abstraction, PredictionPair
from world_model.planning.gameplay import SlingshotAction, SlingshotActionBounds
from world_model.training.lineage_scaling import ActionCandidate, ActionRankingState


BROAD_ACTION_DESIGN_ID = "issue-66-broad-action-design-v1:2x3x2-full-bounds"
RANKING_FAILURE_COST = 1_000_000_000.0
DISAGREEMENT_PENALTY_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


class ActionRankingProbeError(ValueError):
    """The declared action-ranking diagnostic is inconsistent."""


@dataclass(frozen=True, slots=True)
class BroadActionCandidate:
    identity: str
    ordinal: int
    action_stratum: str
    action: SlingshotAction


def broad_action_candidates(
    source_state_identity: str,
    bounds: SlingshotActionBounds,
) -> tuple[BroadActionCandidate, ...]:
    """Bind the fixed 2x3x2 full-bound design to one source decision state."""

    if not source_state_identity or type(bounds) is not SlingshotActionBounds:
        raise ActionRankingProbeError("broad action design requires a source and bounds")
    drag_values = (
        ("near", bounds.drag_x[1]),
        ("middle", round((bounds.drag_x[0] + bounds.drag_x[1]) / 2)),
        ("far", bounds.drag_x[0]),
    )
    arc_values = (("low", bounds.drag_y[0]), ("high", bounds.drag_y[1]))
    tap_values = (
        ("early", bounds.tap_time_ms[0]),
        ("late", bounds.tap_time_ms[1]),
    )
    candidates = []
    for arc_name, drag_y in arc_values:
        for drag_name, drag_x in drag_values:
            for tap_name, tap_time_ms in tap_values:
                ordinal = len(candidates) + 1
                stratum = (
                    f"arc_{arc_name}__drag_{drag_name}__tap_{tap_name}"
                )
                candidates.append(BroadActionCandidate(
                    identity=(
                        f"issue-66-broad-candidate-v1:{source_state_identity}:"
                        f"c{ordinal:02d}"
                    ),
                    ordinal=ordinal,
                    action_stratum=stratum,
                    action=SlingshotAction(drag_x, drag_y, tap_time_ms),
                ))
    result = tuple(candidates)
    if (
        len(result) != 12
        or len({item.action_stratum for item in result}) != 12
        or len({item.action for item in result}) != 12
        or any(not bounds.contains(item.action) for item in result)
    ):
        raise ActionRankingProbeError("broad action strata are not distinct and legal")
    return result


@dataclass(frozen=True, slots=True)
class RankingDiversityReport:
    state_count: int
    candidate_count: int
    all_tied_state_count: int
    pig_removal_discordant_state_count: int
    block_only_discordant_state_count: int
    best_action_tie_sizes: tuple[int, ...]
    candidate_failure_count: int
    state_failure_count: int


def _goal_counts(realized_cost: float) -> tuple[int, int]:
    rounded = round(realized_cost)
    if realized_cost < 0.0 or not math.isclose(realized_cost, rounded, abs_tol=1e-6):
        raise ActionRankingProbeError("realized goal cost is not an integer count cost")
    return rounded // 1000, rounded % 1000


def summarize_ranking_diversity(
    states: tuple[ActionRankingState, ...],
    *,
    failure_cost: float = RANKING_FAILURE_COST,
) -> RankingDiversityReport:
    """Describe realized outcome diversity before any model comparison."""

    if not states or not math.isfinite(failure_cost):
        raise ActionRankingProbeError("ranking diversity inputs are invalid")
    all_tied = 0
    pig_discordant = 0
    block_only_discordant = 0
    failures = 0
    failed_states = 0
    tie_sizes = []
    candidate_count = 0
    for state in states:
        candidate_count += len(state.candidates)
        valid_costs = []
        state_failures = 0
        for candidate in state.candidates:
            cost = float(candidate.realized_cost)
            if cost >= failure_cost:
                failures += 1
                state_failures += 1
            else:
                valid_costs.append(cost)
        if state_failures:
            failed_states += 1
        if not valid_costs:
            tie_sizes.append(0)
            continue
        best = min(valid_costs)
        tie_sizes.append(sum(cost == best for cost in valid_costs))
        outcomes = tuple(_goal_counts(cost) for cost in valid_costs)
        pig_counts = {pigs for pigs, _blocks in outcomes}
        block_counts = {blocks for _pigs, blocks in outcomes}
        if not state_failures and len(set(outcomes)) == 1:
            all_tied += 1
        elif len(pig_counts) > 1:
            pig_discordant += 1
        elif len(block_counts) > 1:
            block_only_discordant += 1
    return RankingDiversityReport(
        state_count=len(states),
        candidate_count=candidate_count,
        all_tied_state_count=all_tied,
        pig_removal_discordant_state_count=pig_discordant,
        block_only_discordant_state_count=block_only_discordant,
        best_action_tie_sizes=tuple(tie_sizes),
        candidate_failure_count=failures,
        state_failure_count=failed_states,
    )


@dataclass(frozen=True, slots=True)
class EnsembleCheckpointBinding:
    checkpoint_identity: str
    protocol_identity: str
    carrier_identity: str
    scale_name: str
    carrier: str
    seed: int
    available_horizons: tuple[int, ...]

    @property
    def member_id(self) -> str:
        return f"{self.scale_name}-{self.carrier}-seed-{self.seed}"


def validate_ensemble_checkpoint_bindings(
    bindings: tuple[EnsembleCheckpointBinding, ...],
) -> tuple[EnsembleCheckpointBinding, ...]:
    """Require three distinct h15 seeds from one matched training cell."""

    if (
        type(bindings) is not tuple
        or len(bindings) != 3
        or len({item.member_id for item in bindings}) != 3
        or len({item.seed for item in bindings}) != 3
        or any(15 not in item.available_horizons for item in bindings)
        or len({
            (
                item.protocol_identity,
                item.carrier_identity,
                item.scale_name,
                item.carrier,
            )
            for item in bindings
        }) != 1
    ):
        raise ActionRankingProbeError(
            "ensemble checkpoints must be three distinct matched h15 seeds"
        )
    return tuple(sorted(bindings, key=lambda item: item.member_id))


@dataclass(frozen=True, slots=True)
class MemberCandidatePrediction:
    candidate_identity: str
    predicted_cost: float | None
    failure: str | None


@dataclass(frozen=True, slots=True)
class MemberStatePrediction:
    state_identity: str
    candidate_set_identity: str
    candidates: tuple[MemberCandidatePrediction, ...]


@dataclass(frozen=True, slots=True)
class MemberRankingPrediction:
    checkpoint: EnsembleCheckpointBinding
    states: tuple[MemberStatePrediction, ...]
    execution_failures: tuple[str, ...]
    model_evaluations: int
    wall_seconds: float


def score_action_ranking_member(
    model: torch.nn.Module,
    checkpoint: EnsembleCheckpointBinding,
    states: tuple[ActionRankingState, ...],
    *,
    horizon: int = 15,
    recursive_steps: int = 15,
    predicted_cost: Callable[
        [ActionRankingState, ActionCandidate, torch.Tensor], float
    ],
    progress: Callable[[str], None] | None = None,
) -> MemberRankingPrediction:
    """Score one ensemble member while retaining every candidate prediction."""

    if (
        not states
        or horizon <= 0
        or recursive_steps <= 0
        or horizon not in checkpoint.available_horizons
        or any(
            state.carrier.value != checkpoint.carrier
            or state.carrier_identity != checkpoint.carrier_identity
            for state in states
        )
    ):
        raise ActionRankingProbeError("member ranking inputs are invalid")
    device = next(model.parameters()).device
    pair = PredictionPair(horizon, Abstraction.CONTINUOUS)
    results = []
    failures = []
    model_evaluations = 0
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for state_index, state in enumerate(states, start=1):
            candidate_results = []
            for candidate_index, candidate in enumerate(state.candidates, start=1):
                if progress is not None:
                    progress(
                        f"model={checkpoint.seed} state={state_index}/{len(states)} "
                        f"candidate={candidate_index}/{len(state.candidates)}"
                    )
                try:
                    predicted = state.context.to(device)
                    for _step in range(recursive_steps):
                        model_evaluations += 1
                        predicted = model.carrier(
                            predicted.unsqueeze(0),
                            candidate.action.to(device).unsqueeze(0),
                            pair,
                        )[0]
                    if not bool(torch.isfinite(predicted).all()):
                        raise RuntimeError("candidate prediction is nonfinite")
                    cost = float(
                        predicted_cost(
                            state,
                            candidate,
                            predicted.detach().cpu(),
                        )
                    )
                    if not math.isfinite(cost):
                        raise RuntimeError("candidate predicted cost is nonfinite")
                    candidate_results.append(MemberCandidatePrediction(
                        candidate.identity, cost, None
                    ))
                except Exception as error:
                    failure = (
                        f"model={checkpoint.seed} state={state.identity} "
                        f"candidate={candidate.identity}:"
                        f"{type(error).__name__}: {error}"
                    )
                    failures.append(failure)
                    candidate_results.append(MemberCandidatePrediction(
                        candidate.identity, None, failure
                    ))
            results.append(MemberStatePrediction(
                state.identity,
                state.candidate_set_identity,
                tuple(candidate_results),
            ))
    return MemberRankingPrediction(
        checkpoint=checkpoint,
        states=tuple(results),
        execution_failures=tuple(failures),
        model_evaluations=model_evaluations,
        wall_seconds=time.monotonic() - started,
    )


def pessimistic_ensemble_cost(
    member_costs: tuple[float, ...],
    disagreement_penalty: float,
) -> tuple[float, float, float]:
    """Return population mean, population deviation, and pessimistic cost."""

    if (
        not member_costs
        or any(not math.isfinite(value) for value in member_costs)
        or not math.isfinite(disagreement_penalty)
        or disagreement_penalty < 0.0
    ):
        raise ActionRankingProbeError("ensemble cost inputs are invalid")
    mean = math.fsum(member_costs) / len(member_costs)
    disagreement = math.sqrt(
        math.fsum((value - mean) ** 2 for value in member_costs)
        / len(member_costs)
    )
    return mean, disagreement, mean + disagreement_penalty * disagreement


@dataclass(frozen=True, slots=True)
class EnsembleCandidateScore:
    candidate_identity: str
    member_costs: tuple[tuple[str, float], ...]
    ensemble_mean_cost: float
    model_disagreement: float
    pessimistic_cost: float


@dataclass(frozen=True, slots=True)
class EnsembleStateResult:
    state_identity: str
    candidate_set_identity: str
    candidate_scores: tuple[EnsembleCandidateScore, ...]
    member_selected_candidates: tuple[tuple[str, str], ...]
    ensemble_mean_selected_candidate: str
    uncertainty_penalized_selected_candidate: str
    best_realized_candidate: str
    single_model_top_action_regrets: tuple[tuple[str, float], ...]
    ensemble_mean_top_action_regret: float
    uncertainty_penalized_top_action_regret: float


@dataclass(frozen=True, slots=True)
class EnsembleRankingEvaluation:
    checkpoint_identities: tuple[str, ...]
    disagreement_penalty: float
    requested_state_count: int
    evaluated_state_count: int
    single_model_mean_top_action_regrets: tuple[tuple[str, float | None], ...]
    ensemble_mean_top_action_regret: float | None
    uncertainty_penalized_mean_top_action_regret: float | None
    states: tuple[EnsembleStateResult, ...]
    execution_failures: tuple[str, ...]


def aggregate_ensemble_ranking(
    states: tuple[ActionRankingState, ...],
    members: tuple[MemberRankingPrediction, ...],
    *,
    disagreement_penalty: float,
) -> EnsembleRankingEvaluation:
    """Aggregate three member predictions on one exactly matched candidate set."""

    ordered_bindings = validate_ensemble_checkpoint_bindings(
        tuple(item.checkpoint for item in members)
    )
    by_identity = {item.checkpoint.member_id: item for item in members}
    ordered = tuple(by_identity[item.member_id] for item in ordered_bindings)
    if len(by_identity) != 3 or any(len(item.states) != len(states) for item in ordered):
        raise ActionRankingProbeError("ensemble member state inventory differs")
    state_results = []
    failures = [
        failure for member in ordered for failure in member.execution_failures
    ]
    per_member_regrets = {
        item.checkpoint.member_id: [] for item in ordered
    }
    ensemble_regrets = []
    pessimistic_regrets = []
    for state_index, state in enumerate(states):
        member_states = tuple(item.states[state_index] for item in ordered)
        expected_candidates = tuple(item.identity for item in state.candidates)
        if any(
            item.state_identity != state.identity
            or item.candidate_set_identity != state.candidate_set_identity
            or tuple(value.candidate_identity for value in item.candidates)
            != expected_candidates
            for item in member_states
        ):
            raise ActionRankingProbeError("ensemble member candidate inventory differs")
        if any(
            value.predicted_cost is None
            for item in member_states
            for value in item.candidates
        ):
            failures.append(f"state={state.identity}:incomplete_member_prediction")
            continue
        candidate_scores = []
        for candidate_index, candidate in enumerate(state.candidates):
            costs = tuple(
                float(item.candidates[candidate_index].predicted_cost)
                for item in member_states
            )
            mean, disagreement, pessimistic = pessimistic_ensemble_cost(
                costs, disagreement_penalty
            )
            candidate_scores.append(EnsembleCandidateScore(
                candidate_identity=candidate.identity,
                member_costs=tuple(
                    (member.checkpoint.member_id, cost)
                    for member, cost in zip(ordered, costs, strict=True)
                ),
                ensemble_mean_cost=mean,
                model_disagreement=disagreement,
                pessimistic_cost=pessimistic,
            ))
        best_index = min(
            range(len(state.candidates)),
            key=lambda index: state.candidates[index].realized_cost,
        )
        mean_index = min(
            range(len(candidate_scores)),
            key=lambda index: candidate_scores[index].ensemble_mean_cost,
        )
        pessimistic_index = min(
            range(len(candidate_scores)),
            key=lambda index: candidate_scores[index].pessimistic_cost,
        )
        member_selected = []
        member_regrets = []
        best_cost = float(state.candidates[best_index].realized_cost)
        for member, member_state in zip(ordered, member_states, strict=True):
            selected_index = min(
                range(len(member_state.candidates)),
                key=lambda index: float(
                    member_state.candidates[index].predicted_cost
                ),
            )
            identity = member.checkpoint.member_id
            regret = float(
                state.candidates[selected_index].realized_cost - best_cost
            )
            member_selected.append((identity, state.candidates[selected_index].identity))
            member_regrets.append((identity, regret))
            per_member_regrets[identity].append(regret)
        ensemble_regret = float(
            state.candidates[mean_index].realized_cost - best_cost
        )
        pessimistic_regret = float(
            state.candidates[pessimistic_index].realized_cost - best_cost
        )
        ensemble_regrets.append(ensemble_regret)
        pessimistic_regrets.append(pessimistic_regret)
        state_results.append(EnsembleStateResult(
            state_identity=state.identity,
            candidate_set_identity=state.candidate_set_identity,
            candidate_scores=tuple(candidate_scores),
            member_selected_candidates=tuple(member_selected),
            ensemble_mean_selected_candidate=state.candidates[mean_index].identity,
            uncertainty_penalized_selected_candidate=(
                state.candidates[pessimistic_index].identity
            ),
            best_realized_candidate=state.candidates[best_index].identity,
            single_model_top_action_regrets=tuple(member_regrets),
            ensemble_mean_top_action_regret=ensemble_regret,
            uncertainty_penalized_top_action_regret=pessimistic_regret,
        ))
    return EnsembleRankingEvaluation(
        checkpoint_identities=tuple(item.checkpoint.member_id for item in ordered),
        disagreement_penalty=disagreement_penalty,
        requested_state_count=len(states),
        evaluated_state_count=len(state_results),
        single_model_mean_top_action_regrets=tuple(
            (
                identity,
                None if not values else math.fsum(values) / len(values),
            )
            for identity, values in per_member_regrets.items()
        ),
        ensemble_mean_top_action_regret=(
            None
            if not ensemble_regrets
            else math.fsum(ensemble_regrets) / len(ensemble_regrets)
        ),
        uncertainty_penalized_mean_top_action_regret=(
            None
            if not pessimistic_regrets
            else math.fsum(pessimistic_regrets) / len(pessimistic_regrets)
        ),
        states=tuple(state_results),
        execution_failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class DisagreementPenaltyCalibration:
    selected_penalty: float
    calibration_regrets: tuple[tuple[float, float | None], ...]


def calibrate_disagreement_penalty(
    states: tuple[ActionRankingState, ...],
    members: tuple[MemberRankingPrediction, ...],
    *,
    penalty_grid: tuple[float, ...] = DISAGREEMENT_PENALTY_GRID,
) -> DisagreementPenaltyCalibration:
    """Select the lowest-regret penalty on calibration states, ties to smaller."""

    if (
        not penalty_grid
        or len(set(penalty_grid)) != len(penalty_grid)
        or any(not math.isfinite(value) or value < 0.0 for value in penalty_grid)
    ):
        raise ActionRankingProbeError("disagreement penalty grid is invalid")
    rows = tuple(
        (
            penalty,
            aggregate_ensemble_ranking(
                states,
                members,
                disagreement_penalty=penalty,
            ).uncertainty_penalized_mean_top_action_regret,
        )
        for penalty in sorted(penalty_grid)
    )
    available = tuple((penalty, regret) for penalty, regret in rows if regret is not None)
    if not available:
        raise ActionRankingProbeError("calibration produced no complete ranking states")
    selected = min(available, key=lambda item: (float(item[1]), item[0]))[0]
    return DisagreementPenaltyCalibration(selected, rows)
