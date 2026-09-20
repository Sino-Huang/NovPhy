import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_77_n1_diagnostic as diag
from scripts.run_issue_77_n1_diagnostic import (
    HORIZONS, PRIOR_ORDINAL, SYSTEMS, TIMES,
)
from world_model.model import Abstraction, PredictionPair
from world_model.training.matched_dynamics import ContinuousDynamics


def fixture_heldout_campaign(root):
    members, branches = [], []
    held = {7, 8, 14, 16}
    for ordinal in range(1, 17):
        family = "type010103" if ordinal <= 8 else "type010105"
        if ordinal % 5 == 0:
            role, partition, study = "training", "controller", "controller_only"
        elif ordinal in held:
            role, partition, study = "calibration", None, "held_out_evaluation"
        else:
            role, partition, study = "training", "predictor", "predictor_train"
        identity = f"issue-77-n1-{ordinal:03d}"
        members.append({"identity": identity, "ordinal": ordinal, "generator_family": family,
                        "exposure_role": role, "fit_partition": partition, "study_role": study})
        for action in range(13):
            branches.append({"identity": f"{identity}-a{action:02d}",
                             "source_member_identity": identity, "candidate_ordinal": action,
                             "action": {"drag_x": -80, "drag_y": 10,
                                        "tap_time_ms": 0, "release_time_ms": 1000}})
    plan = {"identity": "issue-77-n1-v1", "members": members, "branches": branches}
    coverage = {"schema": "issue_77_n1_coverage_v1", "coverage_complete": True,
                "status_counts": {"admissible": 200, "failed": 8},
                "branches": {}}
    for branch in branches:
        status = "admissible"
        if branch["source_member_identity"] == "issue-77-n1-007" and branch["candidate_ordinal"] >= 9:
            status = "failed"
        if branch["source_member_identity"] == "issue-77-n1-016":
            status = "failed"
        coverage["branches"][branch["identity"]] = {"status": status}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "coverage.json").write_text(json.dumps(coverage))
    return plan, coverage


def fixture_result(member, branch):
    return {"member_identity": branch["identity"], "complete": True,
            "segments": [{"capture_id": "capture-v2:test", "identity": "segment:test",
                          "native_root": "/nonexistent",
                          "source_bindings": {},
                          "action": {"drag_release": [branch["action"]["drag_x"], branch["action"]["drag_y"]],
                                     "release_time": branch["action"]["release_time_ms"],
                                     "tap_time": branch["action"]["tap_time_ms"]},
                          "observation_manifest": "observation-trace-manifest-v1:sha256:test",
                          "summary": {"first_fixed_step": 30000, "last_fixed_step": 60000,
                                      "frame_count": 601, "censored": True,
                                      "terminal_evidence": None}}]}


class Issue77N1DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_help_resolves_defaults_without_data_access(self):
        with patch("sys.argv", ["diag", "--help"]), self.assertRaises(SystemExit) as ended:
            diag.main()
        self.assertEqual(ended.exception.code, 0)

    def test_needed_offsets_stay_inside_the_observed_window(self):
        offsets = diag.needed_offsets()
        self.assertEqual(offsets[0], 0)
        self.assertLessEqual(offsets[-1], 600)
        self.assertEqual(offsets, sorted(set(offsets)))
        for t in TIMES:
            self.assertIn(t, offsets)
            for h in HORIZONS:
                self.assertIn(t - h, offsets)

    def test_select_samples_binds_heldout_lineages_and_absorbs_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, coverage = fixture_heldout_campaign(root)
            results = root / "results"
            results.mkdir()
            for branch in plan["branches"]:
                if coverage["branches"][branch["identity"]]["status"] != "admissible":
                    continue
                if not branch["source_member_identity"].endswith(("-007", "-008", "-014", "-016")):
                    continue
                value = fixture_result(branch["source_member_identity"], branch)
                (results / f"{branch['identity']}.json").write_text(json.dumps(value))
            with patch.object(diag, "CAMPAIGN", root):
                samples = diag.select_samples(coverage, plan)
        identities = [s["state"]["identity"] for s in samples]
        self.assertEqual(identities, ["issue-77-n1-007", "issue-77-n1-008",
                                      "issue-77-n1-014", "issue-77-n1-016"])
        by_identity = {s["state"]["identity"]: s for s in samples}
        self.assertEqual(len(by_identity["issue-77-n1-007"]["sources"]), 9)
        self.assertEqual(len(by_identity["issue-77-n1-007"]["dropped_candidates"]), 4)
        self.assertEqual(by_identity["issue-77-n1-016"]["sources"], [])
        self.assertIn("unsupported", by_identity["issue-77-n1-016"])
        ordinals = [s["ordinal"] for s in by_identity["issue-77-n1-008"]["sources"]]
        self.assertEqual(ordinals, list(range(13)))
        for sample in samples:
            for source in sample["sources"]:
                self.assertEqual(source["first_fixed_step"], 30000)
                self.assertEqual(source["last_offset"], 600)
                self.assertTrue(source["censored"])

    def test_task_metrics_ranks_by_realized_cost_and_flags_failures(self):
        realized = [2., 0., 1., 2.]
        metrics = diag.task_metrics([.9, .8, .7, .6], realized)
        self.assertEqual(metrics["selected"], 3)
        self.assertEqual(metrics["regret"], 1.)
        self.assertFalse(metrics["top1"])
        self.assertTrue(metrics["top3"])
        metrics = diag.task_metrics([.9, .1, .8, .7][:2], realized[:2])
        self.assertEqual(metrics["selected"], 1)
        self.assertEqual(metrics["regret"], 0.)
        self.assertTrue(metrics["top1"])
        failed = diag.task_metrics([None, .5], realized[:2])
        self.assertTrue(failed["prediction_failure"])
        self.assertEqual(failed["regret"], 1.)
        tied = diag.task_metrics([.5, .5], [1., 1.])
        self.assertTrue(tied["all_tied"])
        self.assertFalse(tied["informative"])

    def test_fixed_curve_and_check_curve_roundtrip(self):
        model = ContinuousDynamics(16)
        context = torch.zeros(236)
        action = {"drag_x": -80, "drag_y": 10, "tap_time_ms": 0, "release_time_ms": 1000}
        pair = PredictionPair(15, Abstraction.CONTINUOUS)
        with patch.object(diag, "TIMES", (15, 30, 60)), patch.object(diag, "TASK_TIME", 60):
            curve = diag.fixed_curve(model, context, action, pair)
            self.assertIsNone(curve["failure"])
            self.assertEqual(curve["completed_steps"], 60)
            self.assertEqual(set(curve["outputs"]), {"15", "30", "60"})
            plan = {"identity": "test", "contract": {"vocabulary":
                    ["bird:0000"] + [f"block:{i:04d}" for i in range(5)] + ["pig:0000"] +
                    [f"platform:{i:04d}" for i in range(6)] +
                    ["slingshot:0000", "world:landscape:0000", "world:landscape:0001"]}}
            sample = {"state": {"identity": "issue-77-n1-007"}}
            source = {"identity": "issue-77-n1-007-a00", "ordinal": 0, "action": action}
            target = {"carriers": {str(t): [0.] * 236 for t in (15, 30, 60)},
                      "source": source}
            target_state = {"context": [0.] * 236, "candidates": [target]}
            with patch.object(diag, "SYSTEMS", {"continuous_h15": ("continuous", pair)}):
                record = diag.fixed_record(argparse.Namespace(), plan, sample, target_state,
                                           1, "continuous_h15", source, model)
                diag.check_curve(record, plan, sample, 1, "continuous_h15", source, model)
                broken = dict(record, cost=record["cost"] + 1 if record["cost"] is not None else 1)
                with self.assertRaises(ValueError):
                    diag.check_curve(broken, plan, sample, 1, "continuous_h15", source, model)

    def test_action_tensor_uses_true_n1_hold(self):
        tensor = diag.n1_action_tensor({"drag_x": -80, "drag_y": 10,
                                        "tap_time_ms": 0, "release_time_ms": 1000}, "cpu")
        self.assertEqual(tensor.shape, (1, 5))
        self.assertAlmostEqual(float(tensor[0, 0]), -80 / 480)
        self.assertAlmostEqual(float(tensor[0, 2]), 1.0)
        self.assertEqual(float(tensor[0, 4]), 1.0)


if __name__ == "__main__":
    unittest.main()
