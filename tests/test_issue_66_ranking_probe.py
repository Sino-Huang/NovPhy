from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.run_issue_66_ranking_probe import _score_checkpoint_isolated
from world_model.data.successor_cohort import ACTION_BOUNDS
from world_model.model import DualOutputPredictor, PredictorConfig
from world_model.planning.gameplay import (
    CEMConfig,
    CEMPlanner,
    CandidateEvaluation,
    DeterministicEnsembleCandidateEvaluator,
    GameplayCostBreakdown,
    PlanningObservation,
    SlingshotAction,
    SlingshotActionBounds,
)
from world_model.training.action_ranking_probe import (
    ActionRankingProbeError,
    EnsembleCheckpointBinding,
    MemberCandidatePrediction,
    MemberRankingPrediction,
    MemberStatePrediction,
    aggregate_ensemble_ranking,
    broad_action_candidates,
    pessimistic_ensemble_cost,
    summarize_ranking_diversity,
    validate_ensemble_checkpoint_bindings,
)
from world_model.training.lineage_scaling import (
    ActionCandidate,
    ActionRankingState,
    CarrierKind,
    TrainingCell,
    save_action_ranking_bundle,
)


def _bounds() -> SlingshotActionBounds:
    return SlingshotActionBounds(
        tuple(ACTION_BOUNDS["drag_x"]),
        tuple(ACTION_BOUNDS["drag_y"]),
        tuple(ACTION_BOUNDS["tap_time_ms"]),
        ACTION_BOUNDS["release_time_ms"],
    )


def _state(identity: str, costs: tuple[float, ...]) -> ActionRankingState:
    bounds = _bounds()
    actions = broad_action_candidates(identity, bounds)[: len(costs)]
    candidates = tuple(
        ActionCandidate(
            item.identity,
            torch.tensor((
                item.action.drag_x / 480.0,
                item.action.drag_y / 480.0,
                bounds.release_time_ms / 1000.0,
                item.action.tap_time_ms / 1000.0,
                1.0,
            )),
            cost,
            item.action,
        )
        for item, cost in zip(actions, costs, strict=True)
    )
    return ActionRankingState(
        identity=identity,
        scenario_lineage_identity=f"lineage-{identity}",
        trajectory_identity=f"trajectory-{identity}",
        decision_transition_identity=f"transition-{identity}",
        exposure_role="calibration",
        carrier=CarrierKind.DEPLOYMENT,
        carrier_identity="deployment-carrier",
        context=torch.zeros(197),
        action_bounds=bounds,
        frame_height=480,
        candidates=candidates,
        cost_target=torch.zeros(197),
    )


def _binding(index: int) -> EnsembleCheckpointBinding:
    return EnsembleCheckpointBinding(
        checkpoint_identity=f"checkpoint-{index}",
        protocol_identity="protocol",
        carrier_identity="deployment-carrier",
        scale_name="full",
        carrier="deployment",
        seed=100 + index,
        available_horizons=(1, 15),
    )


def _member(
    binding: EnsembleCheckpointBinding,
    state: ActionRankingState,
    costs: tuple[float | None, ...],
) -> MemberRankingPrediction:
    candidates = tuple(
        MemberCandidatePrediction(
            candidate.identity,
            cost,
            None if cost is not None else "model failure",
        )
        for candidate, cost in zip(state.candidates, costs, strict=True)
    )
    failures = () if all(cost is not None for cost in costs) else ("model failure",)
    return MemberRankingPrediction(
        checkpoint=binding,
        states=(MemberStatePrediction(
            state.identity, state.candidate_set_identity, candidates
        ),),
        execution_failures=failures,
        model_evaluations=len(costs),
        wall_seconds=0.1,
    )


class Issue66ActionRankingProbeTests(unittest.TestCase):
    def test_broad_candidates_are_deterministic_source_bound_and_physically_distinct(self) -> None:
        bounds = _bounds()

        first = broad_action_candidates("state-a", bounds)
        repeated = broad_action_candidates("state-a", bounds)
        other_source = broad_action_candidates("state-b", bounds)

        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({item.action_stratum for item in first}), 12)
        self.assertTrue(all(bounds.contains(item.action) for item in first))
        self.assertEqual(
            {item.action.drag_x for item in first}, {-160, -110, -60, -10}
        )
        self.assertEqual({item.action.drag_y for item in first}, {-80, 0, 80})
        self.assertEqual({item.action.tap_time_ms for item in first}, {0})
        self.assertEqual(
            len({(item.action.drag_x, item.action.drag_y) for item in first}),
            12,
        )
        self.assertEqual(
            [item.action for item in first], [item.action for item in other_source]
        )
        self.assertNotEqual(
            [item.identity for item in first],
            [item.identity for item in other_source],
        )
        self.assertTrue(all("sha256" not in item.identity for item in first))

    def test_diversity_reports_ties_pigs_blocks_and_failures(self) -> None:
        states = (
            _state("tied", (1002.0, 1002.0, 1002.0)),
            _state("pig", (1002.0, 2.0, 2.0)),
            _state("block", (1002.0, 1001.0, 1001.0)),
            _state("progress", (1002.9, 1002.2, 1002.2)),
            _state("failure", (1002.0, 1002.0, 1_000_000_000.0)),
        )

        report = summarize_ranking_diversity(states)

        self.assertEqual(report.state_count, 5)
        self.assertEqual(report.candidate_count, 15)
        self.assertEqual(report.all_tied_state_count, 1)
        self.assertEqual(report.pig_removal_discordant_state_count, 1)
        self.assertEqual(report.block_only_discordant_state_count, 1)
        self.assertEqual(report.progress_only_discordant_state_count, 1)
        self.assertEqual(report.best_action_tie_sizes, (3, 2, 2, 2, 2))
        self.assertEqual(report.candidate_failure_count, 1)
        self.assertEqual(report.state_failure_count, 1)

    def test_identical_members_have_zero_disagreement_and_report_all_regrets(self) -> None:
        state = _state("ranking", (0.0, 10.0, 20.0))
        members = tuple(
            _member(_binding(index), state, (3.0, 1.0, 2.0))
            for index in range(1, 4)
        )

        result = aggregate_ensemble_ranking(
            states=(state,),
            members=members,
            disagreement_penalty=2.0,
        )

        self.assertEqual(result.evaluated_state_count, 1)
        self.assertEqual(
            result.single_model_mean_top_action_regrets,
            (
                ("full-deployment-seed-101", 10.0),
                ("full-deployment-seed-102", 10.0),
                ("full-deployment-seed-103", 10.0),
            ),
        )
        self.assertEqual(result.ensemble_mean_top_action_regret, 10.0)
        self.assertEqual(result.uncertainty_penalized_mean_top_action_regret, 10.0)
        self.assertTrue(all(
            score.model_disagreement == 0.0
            for score in result.states[0].candidate_scores
        ))

    def test_ensemble_is_checkpoint_order_invariant(self) -> None:
        state = _state("ranking", (0.0, 10.0, 20.0))
        members = (
            _member(_binding(1), state, (3.0, 1.0, 2.0)),
            _member(_binding(2), state, (2.0, 1.5, 1.0)),
            _member(_binding(3), state, (2.5, 1.0, 3.0)),
        )

        forward = aggregate_ensemble_ranking(
            (state,), members, disagreement_penalty=1.0
        )
        reverse = aggregate_ensemble_ranking(
            (state,), tuple(reversed(members)), disagreement_penalty=1.0
        )

        self.assertEqual(forward, reverse)

    def test_recursive_checkpoint_identities_do_not_enter_retained_results(self) -> None:
        state = _state("ranking", (0.0, 10.0))
        recursive_metadata = "recursive-checkpoint-metadata:" + "x" * 100_000
        members = tuple(
            _member(
                replace(
                    _binding(index),
                    checkpoint_identity=f"{recursive_metadata}:{index}",
                ),
                state,
                (1.0, 2.0),
            )
            for index in range(1, 4)
        )

        result = aggregate_ensemble_ranking(
            (state,), members, disagreement_penalty=1.0
        )

        self.assertNotIn(recursive_metadata, repr(result))

    def test_checkpoint_worker_returns_only_compact_metadata(self) -> None:
        recursive_metadata = "recursive-checkpoint-metadata:" + "x" * 100_000
        cell = TrainingCell("full", CarrierKind.DEPLOYMENT, 20260901)
        config = PredictorConfig(
            latent_dim=197,
            action_dim=5,
            hidden_dim=16,
            depth=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ranking = root / "ranking-bundles"
            calibration = _state("calibration", (0.0, 10.0))
            model_selection = replace(
                _state("model-selection", (0.0, 10.0)),
                exposure_role="model_selection",
            )
            save_action_ranking_bundle(
                ranking / "calibration-deployment.pt", (calibration,)
            )
            save_action_ranking_bundle(
                ranking / "model_selection-deployment.pt", (model_selection,)
            )
            checkpoint = root / "checkpoint.pt"
            model = DualOutputPredictor(config)
            torch.save({
                "schema": "lineage_scaled_continuous_checkpoint_v1",
                "metadata": {
                    "identity": recursive_metadata,
                    "protocol_identity": recursive_metadata,
                    "cell": {
                        "scale_name": cell.scale_name,
                        "carrier": cell.carrier.value,
                        "seed": cell.seed,
                    },
                    "carrier_identity": "deployment-carrier",
                    "predictor_config": asdict(config),
                    "available_horizon_counts": ((1, 1), (15, 1)),
                },
                "model_state": model.state_dict(),
            }, checkpoint)

            binding, returned_config, predictions, peak_rss_mib = (
                _score_checkpoint_isolated(
                    checkpoint,
                    cell,
                    root,
                    "deployment-carrier",
                    None,
                    1,
                    "cpu",
                )
            )

        self.assertEqual(returned_config, config)
        self.assertEqual(binding.member_id, "full-deployment-seed-20260901")
        self.assertNotIn(recursive_metadata, repr((binding, predictions)))
        self.assertGreater(peak_rss_mib, 0.0)

    def test_disagreement_never_reduces_pessimistic_cost(self) -> None:
        same = pessimistic_ensemble_cost((2.0, 2.0, 2.0), 2.0)
        spread = pessimistic_ensemble_cost((0.0, 2.0, 4.0), 2.0)

        self.assertEqual(same, (2.0, 0.0, 2.0))
        self.assertGreater(spread[1], same[1])
        self.assertGreater(spread[2], spread[0])

    def test_member_failure_is_counted_and_excludes_state_from_all_comparisons(self) -> None:
        state = _state("ranking", (0.0, 10.0))
        members = (
            _member(_binding(1), state, (1.0, 2.0)),
            _member(_binding(2), state, (1.0, None)),
            _member(_binding(3), state, (1.0, 2.0)),
        )

        result = aggregate_ensemble_ranking(
            (state,), members, disagreement_penalty=1.0
        )

        self.assertEqual(result.requested_state_count, 1)
        self.assertEqual(result.evaluated_state_count, 0)
        self.assertIsNone(result.ensemble_mean_top_action_regret)
        self.assertGreaterEqual(len(result.execution_failures), 2)

    def test_checkpoint_mismatch_is_rejected(self) -> None:
        bindings = (_binding(1), _binding(2), replace(_binding(3), carrier="source"))

        with self.assertRaises(ActionRankingProbeError):
            validate_ensemble_checkpoint_bindings(bindings)


class _FixedEvaluator:
    def __init__(self, cost: float, physical: float, rollout: float) -> None:
        self.cost = cost
        self.physical = physical
        self.rollout = rollout

    def evaluate(self, observation, actions):
        breakdown = GameplayCostBreakdown(
            total=self.cost,
            goal_progress_cost=0.0,
            terminal_cost=0.0,
            legal_action_cost=0.0,
            physical_cost=self.physical,
            rollout_cost=self.rollout,
            compute_cost=0.0,
            structure_unstable_cost=0.0,
            structure_unstable_affects_cost=False,
        )
        return CandidateEvaluation(
            actions=actions,
            total_cost=self.cost,
            model_rollout_count=1,
            cost_breakdown=breakdown,
            model_compute=2.0,
        )


class _FailingEvaluator:
    def evaluate(self, observation, actions):
        raise RuntimeError("member failed")


class Issue66GameplayEnsembleTests(unittest.TestCase):
    def test_ensemble_evaluator_preserves_member_penalties_behind_candidate_seam(self) -> None:
        evaluator = DeterministicEnsembleCandidateEvaluator(
            (
                ("member-c", _FixedEvaluator(7.0, 0.7, 0.07)),
                ("member-a", _FixedEvaluator(3.0, 0.3, 0.03)),
                ("member-b", _FixedEvaluator(5.0, 0.5, 0.05)),
            ),
            disagreement_penalty=1.0,
        )
        action = SlingshotAction(-80, 20, 300)

        result = evaluator.evaluate(
            PlanningObservation("state", torch.zeros(15), (), (0, 0)),
            (action,),
        )

        self.assertIsNone(result.failure)
        self.assertEqual(result.model_rollout_count, 3)
        self.assertEqual(result.model_compute, 6.0)
        assert result.ensemble_cost is not None
        self.assertEqual(
            result.ensemble_cost.member_costs,
            (("member-a", 3.0), ("member-b", 5.0), ("member-c", 7.0)),
        )
        self.assertEqual(
            result.ensemble_cost.member_physical_costs,
            (("member-a", 0.3), ("member-b", 0.5), ("member-c", 0.7)),
        )
        self.assertEqual(
            result.ensemble_cost.member_rollout_costs,
            (("member-a", 0.03), ("member-b", 0.05), ("member-c", 0.07)),
        )
        self.assertGreater(result.total_cost, 5.0)
        self.assertTrue(math.isfinite(result.total_cost))

    def test_ensemble_evaluator_attributes_member_failure(self) -> None:
        evaluator = DeterministicEnsembleCandidateEvaluator(
            (
                ("member-a", _FixedEvaluator(3.0, 0.3, 0.03)),
                ("member-b", _FailingEvaluator()),
                ("member-c", _FixedEvaluator(7.0, 0.7, 0.07)),
            ),
            disagreement_penalty=1.0,
        )

        result = evaluator.evaluate(
            PlanningObservation("state", torch.zeros(15), (), (0, 0)),
            (SlingshotAction(-80, 20, 300),),
        )

        self.assertEqual(result.failure, "ensemble_member_failure")
        self.assertTrue(math.isinf(result.total_cost))
        assert result.ensemble_cost is not None
        self.assertEqual(
            result.ensemble_cost.member_failures,
            (("member-b", "RuntimeError: member failed"),),
        )

    def test_cem_accepts_the_ensemble_without_planner_changes(self) -> None:
        bounds = _bounds()
        evaluator = DeterministicEnsembleCandidateEvaluator(
            tuple(
                (f"member-{index}", _FixedEvaluator(float(index), 0.0, 0.0))
                for index in range(1, 4)
            ),
            disagreement_penalty=1.0,
        )
        planner = CEMPlanner(
            CEMConfig(4, 2, 1, 1, 66),
            bounds,
            evaluator,
        )

        result = planner.plan(
            PlanningObservation("state", torch.zeros(15), (), (0, 0))
        )

        self.assertEqual(result.candidate_count, 4)
        self.assertIsNotNone(result.selected_evaluation.ensemble_cost)
