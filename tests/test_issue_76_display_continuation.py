from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_issue_76_display_continuation as continuation


class DisplayContinuationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assignment = {"identity": "new", "action": {"drag_x": -80}, "source_member_identity": "source"}
        self.budget = {"running": False, "stopped": True, "stop_reason": "supervisor_KeyboardInterrupt: ",
                       "active_seconds": 2958., "artifact_bytes": 1000, "completed_attempt_bytes": {"old": 500}}
        self.amendment = {"identity": continuation.IDENTITY, "execution_plan_identity": "execution",
            "remaining_member_identities": ["new"], "boundary_budget": self.budget,
            "prior_records": {"old": {"result": {"complete": False, "failure": "retained"}, "receipt": {"worker_exitcode": 0}}}}
        continuation.base.files.write(self.root / "capture-budget.json", self.budget)
        for folder, key in (("results", "result"), ("supervision", "receipt")):
            continuation.base.files.write(self.root / folder / "old.json", self.amendment["prior_records"]["old"][key])

    def test_original_assignments_cannot_be_launched_again(self):
        for identity in ("old", "foreign"):
            with self.assertRaisesRegex(ValueError, "cannot replay"):
                continuation.launch_record(self.amendment, dict(self.assignment, identity=identity), 0)

    def test_worker_records_change_without_precreating_native_attempt(self):
        previous = continuation.base.resources.live.start_display
        def original(root, source, assignment, limits, slot):
            self.assertIs(continuation.base.resources.live.start_display, continuation.start_display)
            self.assertFalse((root / "attempts/new").exists())
            record = continuation.base.files.read(root / "display-launches/new.json")
            self.assertEqual(record["amendment_identity"], continuation.IDENTITY)
            self.assertEqual(record["action"], self.assignment["action"])
            raise RuntimeError("simulated worker failure")
        with patch.object(continuation, "load", return_value=self.amendment), patch.object(
                continuation, "original_worker", side_effect=original):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                continuation.worker(self.root, {}, self.assignment, {}, 2)
        self.assertIs(continuation.base.resources.live.start_display, previous)

    def run_with(self, action):
        with patch.object(continuation, "load", return_value=self.amendment), patch.object(
                continuation.base, "load_plan", return_value={"identity": "execution"}), patch.object(
                continuation.base, "resume_inventory"), patch.object(continuation.base, "run", side_effect=action):
            return continuation.run(self.root, self.root / "public")

    def test_transition_preserves_budget_and_failure_then_refuses_restart(self):
        previous = continuation.base.resources.worker
        def run(plan, root, output, smoke):
            value = continuation.base.files.read(root / "capture-budget.json")
            self.assertEqual(value["active_seconds"], 2958.)
            self.assertEqual(value["artifact_bytes"], 1000)
            self.assertEqual(value["completed_attempt_bytes"], {"old": 500})
            self.assertFalse(value["stopped"])
            self.assertIs(continuation.base.resources.worker, continuation.worker)
            self.assertEqual(continuation.base.files.read(root / "results/old.json")["failure"], "retained")
            return value
        self.run_with(run)
        self.assertIs(continuation.base.resources.worker, previous)
        with self.assertRaisesRegex(ValueError, "boundary budget changed"):
            self.run_with(run)

    def test_changed_old_result_blocks_before_reset(self):
        continuation.base.files.write(self.root / "results/old.json", {"complete": True})
        with self.assertRaisesRegex(ValueError, "earlier capture"):
            self.run_with(lambda *args: self.fail("must not run"))
        self.assertEqual(continuation.base.files.read(self.root / "capture-budget.json"), self.budget)
        self.assertFalse((self.root / "display-continuation-start.json").exists())

    def test_live_or_other_stop_cannot_prepare_amendment(self):
        for change in ({"running": True}, {"stop_reason": "collection_wall_limit"}):
            budget = {**self.budget, "active_members": [], **change}
            continuation.base.files.write(self.root / "capture-budget.json", budget)
            with patch.object(continuation.base, "load_plan", return_value={}):
                with self.assertRaises(ValueError):
                    continuation.prepare(self.root, self.root / "public")
            self.assertFalse((self.root / continuation.FILENAME).exists())

    def test_validation_distinguishes_preserved_and_corrected_assignments(self):
        old = self.amendment["prior_records"]["old"]
        continuation.validate_launch(self.root, self.amendment, {"identity": "old"}, old["result"], old["receipt"])
        with self.assertRaisesRegex(ValueError, "pre-amendment"):
            continuation.validate_launch(self.root, self.amendment, {"identity": "old"}, {}, old["receipt"])
        with self.assertRaises(FileNotFoundError):
            continuation.validate_launch(self.root, self.amendment, self.assignment, {}, {"worker_slot": 2})
        record = continuation.launch_record(self.amendment, self.assignment, 2)
        continuation.base.files.write(self.root / "display-launches/new.json", record)
        continuation.validate_launch(self.root, self.amendment, self.assignment, {}, {"worker_slot": 2})
        with self.assertRaisesRegex(ValueError, "exact display-amendment"):
            continuation.validate_launch(self.root, self.amendment, self.assignment, {}, {"worker_slot": 3})
