from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts.run_issue_63_matched_experiment import (
    _analyze,
    _bounds,
    _candidate_actions,
    _declared_training_scales,
    _realized_goal_cost,
)
from tests.test_lineage_scaled_retraining import _protocol
from world_model.training.lineage_scaling import CarrierKind, TrainingCell


def _score(value: float) -> dict:
    return {
        "recursive": {
            "1": {"error_auc": value, "normalized_error_auc": value},
            "15": {"error_auc": value, "normalized_error_auc": value},
        },
        "ranking": {
            "mean_top_action_regret": value,
            "execution_failures": [],
        },
        "physical": {"predicted": {"carrier_bound_excess": 0.0}},
        "failures": {"nonfinite": 0, "execution": []},
    }


class Issue63MatchedExperimentTests(unittest.TestCase):
    def test_reads_the_issue_62_v4_nested_training_scale_schema(self) -> None:
        scales = _declared_training_scales({
            "nested_training_scales": [
                {
                    "name": "training_6",
                    "lineage_count": 6,
                    "slot_identities": [f"slot:{index}" for index in range(6)],
                },
                {
                    "name": "training_200",
                    "lineage_count": 200,
                    "slot_identities": [f"slot:{index}" for index in range(200)],
                },
                {
                    "name": "training_1000",
                    "lineage_count": 1000,
                    "slot_identities": [f"slot:{index}" for index in range(1000)],
                },
                {
                    "name": "training_3000",
                    "lineage_count": 3000,
                    "slot_identities": [f"slot:{index}" for index in range(3000)],
                },
            ]
        })

        self.assertEqual(
            tuple(item["lineage_count"] for item in scales),
            (6, 200, 1000, 3000),
        )

    def test_candidate_freeze_keeps_the_observed_action_and_four_distinct_legal_perturbations(self) -> None:
        observed = {
            "engine_relative_action": {
                "drag_delta_canvas_pixels": [-100, 10],
                "tap_time_milliseconds": 300,
            }
        }

        actions = _candidate_actions(observed)

        self.assertEqual(
            (actions[0].drag_x, actions[0].drag_y, actions[0].tap_time_ms),
            (-100, 10, 300),
        )
        self.assertEqual(len(actions), 5)
        self.assertEqual(
            len({(item.drag_x, item.drag_y, item.tap_time_ms) for item in actions}),
            5,
        )
        self.assertTrue(all(_bounds().contains(item) for item in actions))

    def test_realized_goal_cost_prioritizes_pig_removal_then_collapse(self) -> None:
        frame = SimpleNamespace(engine_state={
            "entities": (
                {"scenario_object_id": "pig:0000", "lifecycle": "active"},
                {"scenario_object_id": "block:0000", "lifecycle": "active"},
                {"scenario_object_id": "block:0001", "lifecycle": "destroyed"},
            )
        })

        self.assertEqual(_realized_goal_cost(frame), 1001.0)

    def test_advancement_requires_all_three_frozen_deployment_effects(self) -> None:
        protocol = _protocol()
        scores = {}
        for cell in protocol.cells:
            value = 1.0
            if cell.scale_name == "full":
                value = 0.9 if cell.carrier is CarrierKind.SOURCE else 0.8
            scores[cell] = _score(value)

        supported = _analyze(protocol, scores)
        scores[TrainingCell("full", CarrierKind.DEPLOYMENT, 11)] = _score(1.2)
        scores[TrainingCell("full", CarrierKind.DEPLOYMENT, 12)] = _score(1.2)
        scores[TrainingCell("full", CarrierKind.DEPLOYMENT, 13)] = _score(1.2)
        unsupported = _analyze(protocol, scores)

        self.assertEqual(supported["decision"], "supported")
        self.assertEqual(
            unsupported["decision"], "not_supported_by_this_experiment"
        )


if __name__ == "__main__":
    unittest.main()
