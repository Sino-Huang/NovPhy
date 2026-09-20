import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_issue_77_n1_train as runner
from tests.test_issue_71_hybrid_readiness import batch
from world_model.training.matched_dynamics import ContinuousDynamics, MatchedController


def fixture_campaign(root):
    """Minimal issue-77 N1 campaign artifacts exercising the pool derivation."""
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
            branches.append({"identity": f"{identity}-a{action:02d}", "source_member_identity": identity,
                             "candidate_ordinal": action,
                             "action": {"drag_x": -80, "drag_y": 10,
                                        "tap_time_ms": 0, "release_time_ms": 1000}})
    plan = {"identity": "issue-77-n1-v1", "members": members, "branches": branches}
    lineages = {m["identity"]: {"generator_family": m["generator_family"], "admissible": 13,
                                "inadmissible": 0, "coverage_complete": m["exposure_role"] == "training"}
                for m in members}
    for m in members:
        if m["exposure_role"] != "training":
            lineages[m["identity"]]["coverage_complete"] = False
    coverage = {"schema": "issue_77_n1_coverage_v1", "coverage_complete": True, "lineages": lineages,
                "branches": {b["identity"]: {"status": "admissible"} for b in branches}}
    (root / "plan.json").write_text(json.dumps(plan))
    (root / "coverage.json").write_text(json.dumps(coverage))
    return plan, coverage


class Issue77N1DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_help_resolves_defaults_without_data_access(self):
        with patch("sys.argv", ["runner", "--help"]), self.assertRaises(SystemExit) as ended:
            runner.main()
        self.assertEqual(ended.exception.code, 0)

    def test_training_role_pools_follow_the_declared_split(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_campaign(Path(directory))
            with patch.object(runner, "CAMPAIGN", Path(directory)):
                pools = runner.n1_lineages()
        self.assertEqual(len(pools["predictor"]), 9)
        self.assertEqual(len(pools["controller"]), 3)
        self.assertEqual(sorted(r["ordinal"] for r in pools["records"].values()),
                         [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15])
        held = {7, 8, 14, 16}
        self.assertFalse(any(record["ordinal"] in held or record["ordinal"] % 5 == 0
                             for record in (pools["records"][i] for i in pools["predictor"])))
        self.assertTrue(all(record["ordinal"] % 5 == 0
                            for record in (pools["records"][i] for i in pools["controller"])))
        self.assertTrue(all(len(record["branches"]) == 13 for record in pools["records"].values()))
    def test_pool_derivation_absorbs_typed_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, coverage = fixture_campaign(root)
            branches = {}
            for member in plan["members"]:
                for branch in plan["branches"]:
                    if branch["source_member_identity"] != member["identity"]:
                        continue
                    status = "admissible"
                    if member["identity"] == "issue-77-n1-002" and branch["candidate_ordinal"] >= 5:
                        status = "failed"
                    if member["identity"] == "issue-77-n1-010":
                        status = "failed"
                    branches[branch["identity"]] = {"status": status}
            coverage["branches"] = branches
            (root / "coverage.json").write_text(json.dumps(coverage))
            with patch.object(runner, "CAMPAIGN", root):
                pools = runner.n1_lineages()
            self.assertEqual(len(pools["records"]["issue-77-n1-002"]["branches"]), 5)
            self.assertEqual(pools["records"]["issue-77-n1-002"]["dropped_branches"], 8)
            self.assertNotIn("issue-77-n1-010", pools["records"])
            self.assertEqual([r["identity"] for r in pools["dropped"]], ["issue-77-n1-010"])
            self.assertEqual(len(pools["predictor"]), 9)
            self.assertEqual(len(pools["controller"]), 2)
            coverage["coverage_complete"] = False
            (root / "coverage.json").write_text(json.dumps(coverage))
            with patch.object(runner, "CAMPAIGN", root), \
                    self.assertRaisesRegex(ValueError, "coverage"):
                runner.n1_lineages()

    def test_fitting_group_cycles_mixed_pool_identically_for_both_arms(self):
        plan = {"n1_lineages": {"predictor": [f"issue-77-n1-{o:03d}" for o in (1, 2, 3, 4, 6, 7, 8, 9, 11)],
                                "controller": [], "records": {}}}
        pool = runner.fitting_pool(plan)
        self.assertEqual(len(pool), 2409)
        self.assertEqual(pool[0], 1)
        self.assertEqual(pool[-1], "issue-77-n1-011")
        groups = {runner.fitting_lineage_group(plan, step) for step in range(18)}
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group) == 3 for group in groups))
        seen = [runner.fitting_lineage_group(plan, step) for step in range(0, 9000, 9)]
        flat = [entry for group in seen for entry in group]
        self.assertEqual(set(flat), set(pool))

    def test_label_paths_are_unique_across_frozen_and_n1_entries(self):
        root = Path("labels")
        frozen = runner.label_path(root, 0, 5)
        n1 = runner.label_path(root, 0, "issue-77-n1-005")
        self.assertNotEqual(frozen, n1)
        self.assertEqual(frozen.name, "lineage-0005.pt")
        self.assertEqual(n1.name, "lineage-n1-005.pt")

    def test_exact_optimizer_and_sampler_resume(self):
        data = {k: v for k, v in batch().items() if k in ("z", "action", "length")}
        plan = {"identity": "test", "contract": {}, "capacity": {"continuous_width": 16},
                "source_release_identity": "test", "controller": {},
                "n1_lineages": {"predictor": [], "controller": [], "records": {}},
                "training": {"steps": 18, "batch_size": 2, "learning_rate": .0001,
                             "weight_decay": .0001, "grad_clip": 1.}}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "shard", return_value=data):
            args = argparse.Namespace(output=Path(directory) / "resume", device="cpu")
            runner.train_cell(args, plan, {}, 74, "continuous", stop_after=9)
            runner.train_cell(args, plan, {}, 74, "continuous")
            resumed, _ = runner.load_predictor(args, plan, 74, "continuous")
            args.output = Path(directory) / "full"
            runner.train_cell(args, plan, {}, 74, "continuous")
            full, saved = runner.load_predictor(args, plan, 74, "continuous")
            self.assertTrue(all(torch.equal(v, full.state_dict()[k])
                                for k, v in resumed.state_dict().items()))
            changed = dict(saved)
            changed["binding"] = dict(saved["binding"], arm="hybrid")
            with self.assertRaises(ValueError):
                runner.check_binding(changed, saved["binding"])

    def test_controller_interruption_resume_matches_uninterrupted(self):
        data = batch(1)
        labels = {"z": data["z"][0, :60], "action": data["action"].expand(60, -1),
                  "remaining": torch.arange(60, 0, -1), "labels": torch.zeros(60, dtype=torch.long)}
        plan = {"identity": "test", "contract": {}, "capacity": {},
                "source_release_identity": "test", "training": {},
                "n1_lineages": {"predictor": [], "controller": [], "records": {}},
                "controller": {"steps_per_round": 120, "batch_size": 4, "learning_rate": .001}}
        atomic = runner.old.repair.atomic_torch

        def interrupt(path, value):
            atomic(path, value)
            if path.name == "controller-progress-0.pt" and value["step"] == 60:
                raise RuntimeError("simulated interruption")
        source = {"records": [{} for _ in range(5)]}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "load_predictor", return_value=(ContinuousDynamics(16), {})), \
                patch.object(runner, "label_lineage", return_value=labels):
            args = argparse.Namespace(output=Path(directory) / "resume", device="cpu")
            with patch.object(runner.old.repair, "atomic_torch", side_effect=interrupt), \
                    self.assertRaisesRegex(RuntimeError, "simulated"):
                runner.train_control(args, plan, source, 74, "continuous", indices=(5,))
            runner.train_control(args, plan, source, 74, "continuous", indices=(5,))
            resumed, _ = runner.load_control(args, plan, 74, "continuous")
            args.output = Path(directory) / "full"
            runner.train_control(args, plan, source, 74, "continuous", indices=(5,))
            full, _ = runner.load_control(args, plan, 74, "continuous")
            self.assertTrue(all(torch.equal(v, full.state_dict()[k])
                                for k, v in resumed.state_dict().items()))

    def test_controller_pool_extends_frozen_indices(self):
        plan = {"n1_lineages": {"predictor": [], "records": {},
                                "controller": ["issue-77-n1-005", "issue-77-n1-010", "issue-77-n1-015"]}}
        pool = runner.controller_pool(plan)
        self.assertEqual(len(pool), 603)
        self.assertEqual(pool[:3], (5, 10, 15))
        self.assertEqual(pool[-3:], ("issue-77-n1-005", "issue-77-n1-010", "issue-77-n1-015"))


if __name__ == "__main__":
    unittest.main()
