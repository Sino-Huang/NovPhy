from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

from scripts.run_issue_63_matched_experiment import (
    DEFAULT_SUMMARY,
    _analyze,
    _bounds,
    _candidate_actions,
    _cell_score_identity,
    _declared_training_scales,
    _parser,
    _paths,
    _realized_goal_cost,
    _results_identity,
    _ranking_branch_slot,
    _score_matches,
    _score_path,
    _supersede_legacy_long_filename_failures,
    _training_wall_seconds,
    _write_json,
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
    def test_published_results_identity_is_compact_and_ordinal(self) -> None:
        results = {
            "analysis": {"decision": "not_supported_by_this_experiment"},
            "design_identity": "x" * 1_000_000,
            "decision_freeze_identity": "y" * 1_000_000,
            "ranking_collection_correction_identity": None,
            "selected_deployment_configuration": None,
        }

        result_identity = _results_identity(results)

        self.assertEqual(
            result_identity,
            "issue-63-matched-experiment-results-v2:"
            "not_supported_by_this_experiment:none",
        )
        self.assertNotIn("sha256", result_identity)
        self.assertLess(len(result_identity), 120)
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                _paths(Path(temporary))["results"].name,
                "results-v2.json",
            )
        self.assertEqual(
            DEFAULT_SUMMARY.name,
            "matched-carrier-scaling-summary-v2.json",
        )

    def test_training_wall_seconds_reads_only_the_canonical_report_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.training.json"
            _write_json(path, {
                "schema": "issue_63_training_compute_report_v1",
                "protocol_identity": "x" * 10_000,
                "wall_seconds": 52.25,
            })

            self.assertEqual(_training_wall_seconds(path), 52.25)

    def test_cell_scores_use_compact_ordinal_identity_and_v2_path(self) -> None:
        cell = TrainingCell("training_3000", CarrierKind.DEPLOYMENT, 13)
        score = {
            "schema": "issue_63_matched_cell_score_v2",
            "evaluation_role": "calibration",
            "cell": {
                "scale": cell.scale_name,
                "carrier": cell.carrier.value,
                "seed": cell.seed,
            },
        }

        score_identity = _cell_score_identity(score)

        self.assertEqual(
            score_identity,
            "issue-63-matched-cell-score-v2:calibration:"
            "training_3000:deployment:seed-13",
        )
        self.assertNotIn("sha256", score_identity)
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            path = _score_path(paths, "calibration", cell)
            score.update({
                "identity": score_identity,
                "protocol_reference": "protocol.json",
                "checkpoint_reference": (
                    "checkpoints/training_3000/deployment/seed-13.pt"
                ),
                "compute": {
                    "training_report_reference": (
                        "checkpoints/training_3000/deployment/"
                        "seed-13.training.json"
                    ),
                },
                "final_evaluation_opened": False,
            })

            self.assertEqual(path.name, "seed-13.score-v2.json")
            self.assertTrue(_score_matches(score, paths, "calibration", cell))

    def test_scoring_defaults_to_four_bounded_workers(self) -> None:
        parser = _parser()

        defaults = parser.parse_args(["--score-calibration"])
        explicit = parser.parse_args(
            ["--score-calibration", "--score-workers", "2"]
        )

        self.assertEqual(defaults.score_workers, 4)
        self.assertEqual(explicit.score_workers, 2)

    def test_pre_execution_long_filename_failures_are_preserved_and_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failure_path = (
                root
                / "ranking-collection/branches/legacy-state"
                / "legacy-candidate.failure.json"
            )
            _write_json(failure_path, {
                "schema": "issue_63_ranking_candidate_failure_v1",
                "state_identity": "state:legacy",
                "candidate_identity": "candidate:legacy",
                "error_type": "OSError",
                "error": "[Errno 36] File name too long: legacy.xml",
                "failed_run_treatment": "worst_cost",
                "final_evaluation_opened": False,
            })

            correction = _supersede_legacy_long_filename_failures(
                {"root": root}
            )

            self.assertIsNotNone(correction)
            assert correction is not None
            self.assertEqual(correction["failure_count"], 1)
            self.assertFalse(correction["candidate_outcomes_observed"])
            self.assertTrue(failure_path.is_file())
            self.assertTrue(
                (
                    root
                    / "ranking-collection/superseded-pre-execution-failures-v2.json"
                ).is_file()
            )

    def test_ranking_branch_slot_identity_fits_the_runtime_level_filename(self) -> None:
        state_identity = (
            'issue-63-ranking-state-v1:["calibration",'
            '"issue-62-decision-trajectory-v1:sha256:' + "a" * 64 + '",0]'
        )
        candidate_identity = (
            'issue-63-ranking-candidate-v1:["'
            + state_identity.replace('"', '\\"')
            + '",4,-159,15,901]'
        )

        branch = _ranking_branch_slot(
            {"slot_identity": "original", "planned_actions": []},
            {"identity": state_identity, "exposure_role": "calibration"},
            {
                "identity": candidate_identity,
                "drag_x": -159,
                "drag_y": 15,
                "tap_time_ms": 901,
            },
            state_ordinal=11,
            candidate_ordinal=4,
        )

        installed_name = f"issue-53-{branch['slot_identity']}.xml"
        self.assertLessEqual(len(installed_name.encode("utf-8")), 255)
        self.assertEqual(
            branch["slot_identity"], "issue-63-rank-cal-s011-c04-slot-v3"
        )

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
