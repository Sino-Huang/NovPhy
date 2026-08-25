from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from world_model.model import Abstraction
from world_model.training import (
    CohortV2ComputeCalibration,
    CohortV2ExecutionProfile,
    CohortV2ExhaustiveEvaluator,
    CohortV2TrajectoryCostSpec,
    generate_cohort_v2_policy_baselines,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_policy_baselines,
    write_cohort_v2_policy_baselines,
)

from tests.test_cohort_v2_exhaustive_evaluation import _readers


@dataclass(frozen=True)
class _BaselineScorer:
    checkpoint_identity: str = "checkpoint:baseline-fixture"
    objective_identity: str = "objective:baseline-fixture"
    capabilities: frozenset[str] = frozenset({
        "event_endpoint.terminal",
        "transition.continuous",
        "transition.micro",
        "transition.macro",
    })

    def objective(self, window, pair) -> float:
        del window
        matrix = {
            1: {
                Abstraction.CONTINUOUS: 4.0,
                Abstraction.MICRO: 3.0,
                Abstraction.MACRO: 8.0,
            },
            5: {
                Abstraction.CONTINUOUS: 2.0,
                Abstraction.MICRO: 4.0,
                Abstraction.MACRO: 7.0,
            },
            15: {
                Abstraction.CONTINUOUS: 6.0,
                Abstraction.MICRO: 5.0,
                Abstraction.MACRO: 1.0,
            },
        }
        return matrix[pair.delta][pair.abstraction]


def _calibration() -> CohortV2ComputeCalibration:
    return CohortV2ComputeCalibration(
        authority="fixture:baseline-costs",
        unit="multiply_accumulate",
        controller_per_decision=0.0,
        continuous_adapter_per_decision=0.0,
        micro_adapter_per_decision=0.0,
        macro_adapter_per_decision=0.0,
        micro_graph_base_per_decision=0.0,
        micro_graph_per_entity=0.0,
        micro_graph_per_contact=0.0,
        micro_graph_per_support=0.0,
        transition_per_decision=0.0,
        continuous_readout_per_decision=0.0,
        micro_readout_per_decision=0.0,
        macro_readout_per_decision=0.0,
        shared_initial_perception_per_rollout=0.0,
    )


def _inputs():
    readers = _readers()
    evaluation = CohortV2ExhaustiveEvaluator(_BaselineScorer()).evaluate(readers)
    measurement = measure_cohort_v2_evaluation(
        evaluation,
        readers,
        _calibration(),
        CohortV2ExecutionProfile(
            controller_executed=False,
            shared_perception_executed=False,
        ),
    )
    return evaluation, measurement


class CohortV2PolicyBaselineTests(unittest.TestCase):
    def test_baseline_families_share_scope_and_independent_axes_use_uniform_margins(self):
        evaluation, measurement = _inputs()
        result = generate_cohort_v2_policy_baselines(
            evaluation,
            measurement,
            CohortV2TrajectoryCostSpec(0.0, 0.0, 1.0),
            trajectory_label_artifact_identity="labels:fixture",
            derivation_index_identity="derivations:fixture",
        )

        self.assertEqual(
            result.selected_configurations["fixed_pair"],
            {"abstraction": "macro", "requested_horizon": 15},
        )
        self.assertEqual(
            result.selected_configurations["temporal_only_fixed_abstraction"],
            "macro",
        )
        self.assertEqual(
            result.selected_configurations[
                "description_only_fixed_requested_horizon"
            ],
            15,
        )
        state_sets = {
            policy_id: {
                decision.state_id
                for decision in result.decisions
                if decision.policy_id == policy_id
            }
            for policy_id in {decision.policy_id for decision in result.decisions}
        }
        self.assertTrue(next(iter(state_sets.values())))
        self.assertEqual(len({frozenset(states) for states in state_sets.values()}), 1)
        independent = tuple(
            decision
            for decision in result.decisions
            if decision.policy_id == "uniformly_marginalized_independent_axes"
        )
        self.assertTrue(independent)
        self.assertTrue(
            all(
                decision.selected_pair.identity == (15, "continuous")
                for decision in independent
            )
        )
        self.assertEqual(
            {score.state_count for score in result.scores},
            {len(independent) // 3},
        )

    def test_artifacts_are_deterministic_provenance_bound_and_tamper_evident(self):
        evaluation, measurement = _inputs()
        spec = CohortV2TrajectoryCostSpec(1.0, 1.0, 1.0)
        kwargs = {
            "trajectory_label_artifact_identity": "labels:fixture",
            "derivation_index_identity": "derivations:fixture",
            "implementation_revision": "implementation:fixture",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_cohort_v2_policy_baselines(
                root, evaluation, measurement, spec, **kwargs
            )
            first_bytes = tuple(
                (root / name).read_bytes()
                for name in (
                    "manifest.json",
                    "baseline_decisions.jsonl",
                    "scores.json",
                    "frontier.json",
                )
            )
            second = write_cohort_v2_policy_baselines(
                root, evaluation, measurement, spec, **kwargs
            )
            second_bytes = tuple(
                (root / name).read_bytes()
                for name in (
                    "manifest.json",
                    "baseline_decisions.jsonl",
                    "scores.json",
                    "frontier.json",
                )
            )
            manifest = json.loads(first_bytes[0])

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                validate_cohort_v2_policy_baselines(
                    root, evaluation, measurement, spec, **kwargs
                ),
                second,
            )
            self.assertEqual(manifest["release_identity"], evaluation.release_identity)
            self.assertEqual(
                manifest["capability_declaration_identity"],
                evaluation.capability_declaration_identity,
            )
            self.assertEqual(manifest["partition_identity"], evaluation.partition_identity)
            self.assertEqual(manifest["grid_identity"], evaluation.grid.identity)
            self.assertEqual(
                manifest["derivation_index_identity"], "derivations:fixture"
            )
            self.assertEqual(
                manifest["role_permissions"]["model_selection"],
                ["configuration_selection"],
            )
            self.assertFalse(manifest["final_evaluation_consumed"])
            self.assertFalse(manifest["independent_axes_are_output_factorization"])

            scores = (root / "scores.json").read_bytes()
            (root / "scores.json").write_bytes(scores + b"\n")
            with self.assertRaises(ValueError):
                validate_cohort_v2_policy_baselines(
                    root, evaluation, measurement, spec, **kwargs
                )


if __name__ == "__main__":
    unittest.main()
