from __future__ import annotations

import math
import unittest

from world_model.data.cohort_v2 import CAPABILITY_DECLARATION_IDENTITY
from world_model.model import PredictionPair
from world_model.training import (
    CohortV2EvaluationResult,
    CohortV2PairGrid,
    CohortV2PairOutcome,
    CohortV2StateEvaluation,
)
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    MACRO_PAIRS,
    CohortV2MacroConfig,
    CohortV2MacroError,
    CohortV2MacroTrainingData,
)
from world_model.training.cohort_v2_pair_experts import (
    CohortV2PairExpertScorer,
    CohortV2PairExpertTrainer,
    CohortV2PairExpertTrainingData,
    compare_preferred_pair_maps,
    pair_label,
)


class _Scorer:
    capabilities = MACRO_CAPABILITIES

    def __init__(self, pair: PredictionPair) -> None:
        self.pair = pair
        self.checkpoint_identity = f"checkpoint:{pair_label(pair)}"
        self.objective_identity = f"objective:{pair_label(pair)}"

    def objective(self, window, pair: PredictionPair) -> float:
        if pair != self.pair:
            raise AssertionError("misrouted")
        return float(pair.delta)

    def objective_batch(self, windows, pair: PredictionPair) -> tuple[float, ...]:
        return tuple(self.objective(window, pair) for window in windows)


def _outcomes(preferred: PredictionPair, *, expert: bool):
    outcomes = []
    for index, pair in enumerate(MACRO_PAIRS):
        objective = float(index + 2)
        if pair == preferred:
            objective = 0.5
        if expert:
            objective -= 0.1
        outcomes.append(CohortV2PairOutcome(
            pair=pair,
            requested_horizon=pair.delta,
            effective_horizon=pair.delta,
            target_frame_record_identity=f"target:h{pair.delta}",
            objective=objective,
            unavailable_reasons=(),
        ))
    return tuple(outcomes)


def _evaluation(*, expert: bool) -> CohortV2EvaluationResult:
    preferred = MACRO_PAIRS[1] if expert else MACRO_PAIRS[0]
    state = CohortV2StateEvaluation(
        state_id="state:held-out",
        exposure_role="model_selection",
        attempt_id="attempt:held-out",
        scenario_lineage_identity="lineage:held-out",
        context_position=0,
        context_fixed_step=0,
        frame_record_count=16,
        outcomes=_outcomes(preferred, expert=expert),
        selected_pair=preferred,
        tied_pairs=(preferred,),
    )
    return CohortV2EvaluationResult(
        release_identity="release:v5",
        capability_declaration_identity=CAPABILITY_DECLARATION_IDENTITY,
        partition_identity="partition:v5",
        checkpoint_identity="checkpoint:experts" if expert else "checkpoint:shared",
        checkpoint_capabilities=tuple(sorted(MACRO_CAPABILITIES)),
        objective_identity="objective:experts" if expert else "objective:shared",
        grid=CohortV2PairGrid(),
        state_set_identity="state-set:v5",
        states=(state,),
    )


class CohortV2PairExpertTests(unittest.TestCase):
    def test_expert_schedule_selects_the_same_global_pair_slots(self) -> None:
        config = CohortV2MacroConfig(
            steps=2,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )
        base = object.__new__(CohortV2MacroTrainingData)
        base.config = config
        pair = MACRO_PAIRS[4]
        expected = tuple(
            step for step in range(18) if base.schedule_at(step) == pair
        )
        data = object.__new__(CohortV2PairExpertTrainingData)
        data.config = config
        data.pair = pair
        data.shared_step_count = 18
        data.shared_steps = expected

        self.assertEqual(len(expected), 2)
        self.assertEqual(data.schedule_at(0), pair)
        self.assertEqual(data.schedule_at(1), pair)
        trainer = object.__new__(CohortV2PairExpertTrainer)
        trainer.data = data
        trainer.config = config
        self.assertAlmostEqual(
            trainer._learning_rate(1),
            config.learning_rate
            * 0.5
            * (1.0 + math.cos(math.pi * expected[1] / 17)),
        )

    def test_composite_scorer_routes_every_pair_to_its_expert(self) -> None:
        scorer = CohortV2PairExpertScorer(tuple(
            (pair, _Scorer(pair)) for pair in MACRO_PAIRS
        ))

        for pair in MACRO_PAIRS:
            with self.subTest(pair=pair):
                self.assertEqual(scorer.objective(object(), pair), float(pair.delta))
        self.assertEqual(scorer.capabilities, MACRO_CAPABILITIES)

    def test_composite_scorer_requires_the_complete_ordered_grid(self) -> None:
        with self.assertRaisesRegex(CohortV2MacroError, "ordered central grid"):
            CohortV2PairExpertScorer(tuple(
                (pair, _Scorer(pair)) for pair in reversed(MACRO_PAIRS)
            ))

    def test_map_comparison_reports_held_out_pair_changes_and_objective_gains(self) -> None:
        comparison = compare_preferred_pair_maps(
            _evaluation(expert=False), _evaluation(expert=True)
        )

        self.assertEqual(comparison["state_count"], 1)
        self.assertEqual(comparison["changed_preferred_pair_count"], 1)
        self.assertEqual(comparison["changed_preferred_pair_rate"], 1.0)
        self.assertEqual(
            comparison["scenario_lineages"][0]["scenario_lineage_identity"],
            "lineage:held-out",
        )
        unchanged_pair = comparison["per_pair"][2]
        self.assertAlmostEqual(
            unchanged_pair["mean_shared_minus_expert_objective"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
