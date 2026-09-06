from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_issue_69_corrected_experiment as experiment


def scores(count=24):
    result = {}
    for prefix in ("original", "teacher-forced-h15", *experiment.CORRECTIONS):
        corrected = prefix in experiment.CORRECTIONS
        for seed in experiment.SEEDS:
            name = f"{prefix}-{seed}"
            rows = []
            for i in range(count):
                rows.append({"state": f"state-{i}", "lineage": f"lineage-{i}",
                    "candidate_ids": [f"c{j}" for j in range(12)],
                    "realized_costs": [0.0] + [1.0] * 11,
                    "candidate_costs": ([0.0]+[1.0]*11) if corrected else ([1.0]+[0.0]*11),
                    "candidate_failures": [], "local_failures": [],
                    "h15": {"error_auc": 1.0 if corrected else 10.0,
                        "execution_failures": [], "nonfinite_failures": 0,
                        "mean_absolute_carrier_bound_excess": 0.0,
                        "step_curve": [{"step": j, "mean_mse": 0.01} for j in (1, 2)]},
                    "physical_squared_bound_excess": 0.0,
                    "ranking_seconds": 0.1, "ranking_model_evaluations": 180})
            result[name] = {"cell": {"kind": "single", "checkpoint": f"/{name}.pt"}, "rows": rows}
    for prefix in experiment.CORRECTIONS:
        result[f"{prefix}-ensemble"] = deepcopy(result[f"{prefix}-{experiment.SEEDS[0]}"])
        result[f"{prefix}-ensemble"]["cell"] = {"kind": "ensemble"}
    return result


class CorrectedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.scores = scores()
        self.frozen = experiment.freeze_payload({"contract": experiment.contract()}, self.scores)

    def test_supported_requires_all_frozen_conditions(self):
        result = experiment.report(self.scores, self.frozen)
        self.assertEqual(result["decision"], "supported")
        selected = self.scores[self.frozen["selected_system"]]["rows"]
        selected[-1]["h15"]["step_curve"][-1]["mean_mse"] = 10.0
        result = experiment.report(self.scores, self.frozen)
        self.assertEqual(result["decision"], "not_supported_by_this_experiment")
        self.assertIsNone(result["selected_configuration"])
        self.assertFalse(result["issue_64_authorized"])
        self.assertFalse(result["gates"]["absolute_step_mse"])

    def test_failed_predictions_on_tied_states_are_not_removed(self):
        self.assertEqual(experiment.regret([None]*12, [1.0]*12), (1.0, None))
        selected = self.scores[self.frozen["selected_system"]]["rows"]
        selected[0]["candidate_costs"][0] = None
        selected[0]["candidate_failures"] = ["nonfinite"]
        result = experiment.report(self.scores, self.frozen)
        self.assertEqual(result["state_count"], 24)
        self.assertFalse(result["gates"]["no_failures"])
        self.assertFalse(result["issue_64_authorized"])

    def test_retained_replay_failure_can_be_selected_and_costs_worst_regret(self):
        regret, selected = experiment.regret([0.0, 1.0], [1e9, 1001.0])
        self.assertEqual(selected, 0)
        self.assertEqual(regret, 1.0)

    def test_ensemble_failure_does_not_drop_state_or_candidate(self):
        self.scores[f"{experiment.CORRECTIONS[0]}-{experiment.SEEDS[0]}"]["rows"][0]["candidate_costs"][0] = None
        system = experiment.systems(self.scores, 1.0)[f"{experiment.CORRECTIONS[0]}-pessimistic"]
        self.assertEqual(len(system["rows"]), 24)
        self.assertEqual(len(system["rows"][0]["candidate_costs"]), 12)
        self.assertEqual(system["regrets"][0], 1.0)

    def test_calibration_selection_is_independent_of_dictionary_order(self):
        other = experiment.freeze_payload({"contract": experiment.contract()}, dict(reversed(list(self.scores.items()))))
        self.assertEqual(self.frozen, other)

    def test_simultaneous_bounds_preserve_difference_direction(self):
        self.assertAlmostEqual(experiment.lower_bound([0.3]*200), 0.3)
        self.assertLess(experiment.lower_bound([-0.3]*200), 0.0)
        self.assertIsNone(experiment.lower_bound([float("nan")]))

    def test_model_selection_without_freeze_never_loads_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            args = type("Args", (), {"output": Path(directory)})()
            with patch.object(experiment, "load_plan", return_value={}), patch.object(experiment, "role_data") as load:
                with self.assertRaises(experiment.LineageScalingError):
                    experiment.score(args, "model_selection")
                load.assert_not_called()

    def test_freeze_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"decision.json"
            experiment.write(target, self.frozen)
            experiment.write(target, self.frozen)
            changed = {**self.frozen, "penalty": 99.0}
            with self.assertRaises(experiment.LineageScalingError):
                experiment.write(target, changed)

    def test_publish_validate_round_trip_and_detect_changed_summary(self):
        selection = deepcopy(self.scores)
        for cell in selection.values():
            for row in cell["rows"]:
                row["lineage"] = "model-selection-" + row["lineage"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = type("Args", (), {"output": root, "summary": root/"public/result.json"})()
            experiment.write(root/"frozen-decision.json", self.frozen)
            experiment.write(root/"model-selection-access.json", {
                "frozen_decision": self.frozen, "final_evaluation_opened": False})
            with patch.object(experiment, "load_plan", return_value={"contract": experiment.contract()}), patch.object(
                experiment, "load_scores", side_effect=lambda _args, role: self.scores if role == "calibration" else selection
            ), redirect_stdout(io.StringIO()):
                experiment.publish(args)
                experiment.publish(args, validate=True)
                with patch.object(experiment, "report", return_value={**experiment.read(args.summary), "decision": "tampered"}):
                    with self.assertRaises(experiment.LineageScalingError):
                        experiment.publish(args, validate=True)

    def test_no_write_dry_run_exercises_all_systems(self):
        with patch.object(experiment, "write", side_effect=AssertionError("write")), patch("torch.save", side_effect=AssertionError("save")):
            output = io.StringIO()
            with redirect_stdout(output):
                experiment.dry_run()
        self.assertIn("cells=14 systems=16", output.getvalue())
        self.assertIn("files_written=false", output.getvalue())


if __name__ == "__main__":
    unittest.main()
