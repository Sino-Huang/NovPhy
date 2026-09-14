from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_issue_76_wall_continuation as wall


class WallContinuationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plan = {"identity": "execution", "inventory": {"assignments": [{"identity": "new"}],
            "limits": {"shot_seconds": 180, "attempt_seconds": 420, "native_step_seconds": .0004,
                       "native_steps_per_shot_max": 30000, "rgb_frames_per_shot_max": 601,
                       "workers": 8, "collection_wall_seconds": 172800}, "data_readiness": {"threshold": .9}}}
        self.assignment = {"identity": "new", "source_member_identity": "source", "action": {"drag_x": -80}}
        self.budget = {"running": False, "stopped": True, "stop_reason": "supervisor_KeyboardInterrupt: ",
                       "active_seconds": 4147., "artifact_bytes": 26000, "completed_attempt_bytes": {"old": 123},
                       "display_amendment_identity": "display"}
        self.amendment = {"identity": wall.IDENTITY, "execution_plan_identity": "execution",
            "display_amendment_identity": "display", "boundary_budget": self.budget,
            "remaining_member_identities": ["new"], "wall_time_overrides": dict(wall.OVERRIDES),
            "prior_records": {"old": {"result": {"failure": "retained deadline"}, "receipt": {"worker_exitcode": 0}}}}
        wall.base.files.write(self.root / "capture-budget.json", self.budget)
        for folder, key in (("results", "result"), ("supervision", "receipt")):
            wall.base.files.write(self.root / folder / "old.json", self.amendment["prior_records"]["old"][key])

    def test_only_two_wall_limits_change_without_mutating_original(self):
        original = deepcopy(self.plan)
        effective = wall.effective_plan(self.plan)
        expected = deepcopy(original)
        expected["inventory"]["limits"].update(shot_seconds=360, attempt_seconds=600)
        self.assertEqual(effective, expected)
        self.assertEqual(self.plan, original)

    def test_prior_and_foreign_assignments_cannot_be_replayed(self):
        for identity in ("old", "foreign"):
            with self.assertRaisesRegex(ValueError, "cannot replay"):
                wall.launch_record(self.amendment, dict(self.assignment, identity=identity), 0)

    def test_worker_keeps_display_correction_and_binds_wall_limits(self):
        limits = wall.effective_plan(self.plan)["inventory"]["limits"]
        with patch.object(wall, "load", return_value=self.amendment), patch.object(wall.display, "worker") as worker:
            wall.worker(self.root, {}, self.assignment, limits, 2)
            worker.assert_called_once_with(self.root, {}, self.assignment, limits, 2)
            record = wall.base.files.read(self.root / "wall-launches/new.json")
            self.assertEqual(record["wall_time_overrides"], {"shot_seconds": 360, "attempt_seconds": 600})
            self.assertFalse((self.root / "attempts/new").exists())
            with self.assertRaisesRegex(ValueError, "allowance differs"):
                wall.worker(self.root, {}, self.assignment, self.plan["inventory"]["limits"], 2)

    def test_validator_rejects_missing_binding_and_changed_prior_evidence(self):
        old = self.amendment["prior_records"]["old"]
        wall.validate_launch(self.root, self.amendment, {"identity": "old"}, old["result"], old["receipt"])
        with self.assertRaises(ValueError):
            wall.validate_launch(self.root, self.amendment, {"identity": "old"}, {}, old["receipt"])
        receipt = {"worker_slot": 2, "wall_seconds": 500}
        with self.assertRaises(FileNotFoundError):
            wall.validate_launch(self.root, self.amendment, self.assignment, {}, receipt)
        wall.base.files.write(self.root / "wall-launches/new.json", wall.launch_record(self.amendment, self.assignment, 2))
        wall.validate_launch(self.root, self.amendment, self.assignment, {}, receipt)
        with self.assertRaisesRegex(ValueError, "exceeded"):
            wall.validate_launch(self.root, self.amendment, self.assignment, {}, dict(receipt, wall_seconds=600))

    def run_with(self, action):
        with patch.object(wall, "load", return_value=self.amendment), patch.object(
                wall.display, "load", return_value={"identity": "display"}), patch.object(
                wall.base, "load_plan", return_value=self.plan), patch.object(
                wall.base, "resume_inventory"), patch.object(wall.base, "run", side_effect=action):
            return wall.run(self.root, self.root / "public")

    def test_cumulative_budget_and_old_failures_survive_transition(self):
        previous = wall.base.resources.worker
        def run(plan, root, output, smoke):
            self.assertEqual(plan, wall.effective_plan(self.plan))
            value = wall.base.files.read(root / "capture-budget.json")
            for key in ("active_seconds", "artifact_bytes", "completed_attempt_bytes", "display_amendment_identity"):
                self.assertEqual(value[key], self.budget[key])
            self.assertFalse(value["stopped"])
            self.assertEqual(value["wall_amendment_identity"], wall.IDENTITY)
            self.assertIs(wall.base.resources.worker, wall.worker)
            self.assertEqual(wall.base.files.read(root / "results/old.json")["failure"], "retained deadline")
        self.run_with(run)
        self.assertIs(wall.base.resources.worker, previous)
        with self.assertRaisesRegex(ValueError, "boundary budget changed"):
            self.run_with(run)

    def test_changed_prior_result_blocks_before_budget_reset(self):
        wall.base.files.write(self.root / "results/old.json", {})
        with self.assertRaisesRegex(ValueError, "earlier result"):
            self.run_with(lambda *args: self.fail("must not run"))
        self.assertEqual(wall.base.files.read(self.root / "capture-budget.json"), self.budget)
