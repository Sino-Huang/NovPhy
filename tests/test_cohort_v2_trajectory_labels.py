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
    generate_cohort_v2_myopic_ablation_labels,
    generate_cohort_v2_trajectory_labels,
    measure_cohort_v2_evaluation,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_trajectory_labels,
)

from tests.test_cohort_v2_exhaustive_evaluation import _readers


def _calibration() -> CohortV2ComputeCalibration:
    return CohortV2ComputeCalibration(
        authority="fixture:trajectory-label-costs",
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


def _measure(scorer, *, readers=None):
    readers = _readers() if readers is None else readers
    evaluation = CohortV2ExhaustiveEvaluator(scorer).evaluate(readers)
    measurement = measure_cohort_v2_evaluation(
        evaluation,
        readers,
        _calibration(),
        CohortV2ExecutionProfile(
            controller_executed=False,
            shared_perception_executed=False,
        ),
    )
    return readers, evaluation, measurement


def _label_at(result, role: str, position: int):
    return next(
        label
        for label in result.labels
        if label.exposure_role == role and label.context_position == position
    )


@dataclass(frozen=True)
class _TrajectoryScorer:
    checkpoint_identity: str = "checkpoint:trajectory-fixture"
    objective_identity: str = "objective:duration-weighted-fixture"
    capabilities: frozenset[str] = frozenset({"transition.continuous"})

    def objective(self, window, pair) -> float:
        costs = {
            0: {1: 0.10, 5: 0.25, 15: 0.40},
            1: {1: 0.30, 5: 0.50, 15: 0.60},
            2: {1: 0.30, 5: 0.30, 15: 0.30},
        }
        return costs[window.context_position][pair.delta]


@dataclass(frozen=True)
class _EqualSegmentationScorer:
    checkpoint_identity: str = "checkpoint:segmentation-fixture"
    objective_identity: str = "objective:duration-weighted-equal-segments"
    capabilities: frozenset[str] = frozenset({"transition.continuous"})

    def objective(self, window, pair) -> float:
        if pair.delta == 1:
            return 1.0 / 3.0
        if window.context_position == 0 and pair.delta == 5:
            return 1.0
        return 2.0


@dataclass(frozen=True)
class _UnavailableMicroScorer:
    checkpoint_identity: str = "checkpoint:unavailable-micro-fixture"
    objective_identity: str = "objective:unavailable-micro-fixture"
    capabilities: frozenset[str] = frozenset(
        {"transition.continuous", "transition.micro", "transition.macro"}
    )

    def objective(self, window, pair) -> float:
        del window
        return {
            Abstraction.CONTINUOUS: 1.0,
            Abstraction.MICRO: 0.0,
            Abstraction.MACRO: 2.0,
        }[pair.abstraction]


class CohortV2TrajectoryLabelTests(unittest.TestCase):
    def test_default_teacher_is_non_myopic_and_respects_the_next_decision_point(self):
        _readers_, evaluation, measurement = _measure(_TrajectoryScorer())
        spec = CohortV2TrajectoryCostSpec(
            physical_violation_weight=0.0,
            compute_weight=0.0,
            compute_reference=1.0,
        )

        trajectory = generate_cohort_v2_trajectory_labels(
            evaluation, measurement, spec
        )
        myopic = generate_cohort_v2_myopic_ablation_labels(
            evaluation, measurement, spec
        )

        trajectory_first = _label_at(trajectory, "training", 0)
        myopic_first = _label_at(myopic, "training", 0)
        self.assertEqual(trajectory.teacher, "trajectory_optimal")
        self.assertEqual(myopic.teacher, "myopic_ablation")
        self.assertEqual(trajectory_first.selected_pair.identity, (5, "continuous"))
        self.assertEqual(trajectory_first.effective_horizon, 3)
        self.assertEqual(trajectory_first.next_context_position, 3)
        self.assertEqual(myopic_first.selected_pair.identity, (1, "continuous"))
        self.assertLess(myopic_first.segment_cost, trajectory_first.segment_cost)
        self.assertGreater(myopic_first.cost_to_go, trajectory_first.cost_to_go)

    def test_duration_weighted_pair_costs_make_equal_rollout_segmentations_tie(self):
        _readers_, evaluation, measurement = _measure(_EqualSegmentationScorer())
        result = generate_cohort_v2_trajectory_labels(
            evaluation,
            measurement,
            CohortV2TrajectoryCostSpec(0.0, 0.0, 1.0),
        )

        first = _label_at(result, "training", 0)
        self.assertEqual(first.cost_to_go, 1.0)
        self.assertEqual(
            tuple(pair.identity for pair in first.tied_pairs),
            ((1, "continuous"), (5, "continuous")),
        )
        self.assertEqual(first.selected_pair.identity, (1, "continuous"))

    def test_unavailable_symbolic_targets_are_inadmissible_not_high_cost(self):
        readers = _readers(contact_available=False)
        _readers_, evaluation, measurement = _measure(
            _UnavailableMicroScorer(), readers=readers
        )

        result = generate_cohort_v2_trajectory_labels(
            evaluation,
            measurement,
            CohortV2TrajectoryCostSpec(0.0, 0.0, 1.0),
        )

        self.assertTrue(
            all(
                label.selected_pair.abstraction is Abstraction.CONTINUOUS
                and all(
                    pair.abstraction is not Abstraction.MICRO
                    for pair in label.tied_pairs
                )
                for label in result.labels
            )
        )

    def test_artifact_is_deterministic_and_binds_all_controller_label_provenance(self):
        _readers_, evaluation, measurement = _measure(_TrajectoryScorer())
        spec = CohortV2TrajectoryCostSpec(1.5, 0.25, 100.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_cohort_v2_trajectory_labels(
                root,
                evaluation,
                measurement,
                spec,
                implementation_revision="implementation:fixture",
            )
            first_bytes = tuple(
                (root / name).read_bytes()
                for name in ("manifest.json", "controller_labels.jsonl")
            )
            second = write_cohort_v2_trajectory_labels(
                root,
                evaluation,
                measurement,
                spec,
                implementation_revision="implementation:fixture",
            )
            second_bytes = tuple(
                (root / name).read_bytes()
                for name in ("manifest.json", "controller_labels.jsonl")
            )
            manifest = json.loads(first_bytes[0])

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                validate_cohort_v2_trajectory_labels(
                    root,
                    evaluation,
                    measurement,
                    spec,
                    implementation_revision="implementation:fixture",
                ),
                second,
            )
            self.assertEqual(manifest["release_identity"], evaluation.release_identity)
            self.assertEqual(manifest["checkpoint_identity"], evaluation.checkpoint_identity)
            self.assertEqual(manifest["objective_identity"], evaluation.objective_identity)
            self.assertEqual(
                manifest["implementation_revision"], "implementation:fixture"
            )
            self.assertEqual(
                manifest["capability_declaration_identity"],
                evaluation.capability_declaration_identity,
            )
            self.assertEqual(
                manifest["compute_calibration_identity"],
                measurement.compute_calibration_identity,
            )
            self.assertEqual(
                manifest["exposure_roles"],
                ["training", "calibration", "model_selection"],
            )
            self.assertEqual(
                manifest["scenario_lineage_identities"],
                ["lineage:training", "lineage:calibration", "lineage:model_selection"],
            )

            records = (root / "controller_labels.jsonl").read_bytes()
            (root / "controller_labels.jsonl").write_bytes(
                records.replace(b'"cost_to_go":0.25', b'"cost_to_go":9.25', 1)
            )
            with self.assertRaises(ValueError):
                validate_cohort_v2_trajectory_labels(
                    root,
                    evaluation,
                    measurement,
                    spec,
                    implementation_revision="implementation:fixture",
                )


if __name__ == "__main__":
    unittest.main()
