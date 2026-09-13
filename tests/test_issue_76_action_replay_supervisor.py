from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import run_issue_76_action_replay as replay


class ReplaySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.limits = dict(collection_wall_seconds=14400, smoke_wall_seconds=1200,
                           cpu_rss_mib=4096, artifact_bytes=1000000, rgb_frames_per_shot_max=601, attempt_seconds=420)
        self.budget = dict(active_seconds=0., smoke_active_seconds=0., peak_cpu_rss_mib=0., artifact_bytes=0)

    def test_every_frozen_resource_limit_stops(self):
        self.assertIsNone(replay.stop_reason(self.limits, self.budget, 0, smoke=True))
        for field, value, reason in (("active_seconds", 14400, "collection_wall_limit"),
                                     ("smoke_active_seconds", 1200, "smoke_wall_limit"),
                                     ("peak_cpu_rss_mib", 4097, "aggregate_memory_limit"),
                                     ("artifact_bytes", 1000001, "artifact_limit")):
            self.assertEqual(replay.stop_reason(self.limits, dict(self.budget, **{field: value}), 0, smoke=True), reason)
        self.assertEqual(replay.stop_reason(self.limits, self.budget, 420, smoke=False), "attempt_wall_limit")
        self.assertEqual(replay.stop_reason(self.limits, self.budget, 0, smoke=True, frames=602), "capture_or_frame_limit")
        self.assertEqual(replay.stop_reason(self.limits, self.budget, 0, smoke=True, captures=2), "capture_or_frame_limit")

    def test_smoke_order_and_full_collection_gate(self):
        members = [{"identity": "a"}, {"identity": "b"}, {"identity": "c"}]
        plan = {"identity": "execution", "inventory": {"members": members, "smoke_member_identities": ["a", "b"]}}
        self.assertEqual(replay.selected_members(plan, Path("/unused"), True), members[:2])
        with patch.object(replay.files, "read", return_value={"validated": False}):
            with self.assertRaises(ValueError):
                replay.selected_members(plan, Path("/unused"), False)

    def test_interrupted_attempt_is_retained_without_starting_worker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            member = dict(identity="a", base_cluster="base", exposure_role="training", engine_seed=1)
            plan = {"inventory": {"members": [member], "smoke_member_identities": ["a"], "limits": self.limits}}
            (root / "player").mkdir()
            (root / "player/9001.x86_64").touch()
            (root / "attempts/a").mkdir(parents=True)
            with patch.object(replay.multiprocessing, "get_context") as spawn:
                replay.run(plan, root, smoke=True)
                spawn.assert_not_called()
            value = replay.files.read(root / "results/a.json")
            self.assertFalse(value["complete"])
            self.assertEqual(value["failure"], "interrupted_attempt_no_retry")
            self.assertFalse(replay.files.read(root / "capture-budget.json")["running"])

    def test_unclean_supervisor_ledger_cannot_reset_allowance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"inventory": {"members": [], "smoke_member_identities": [], "limits": self.limits}}
            replay.files.write(root / "capture-budget.json", {"running": True, "stopped": False})
            with self.assertRaises(ValueError):
                replay.run(plan, root, smoke=True)

    def test_late_complete_result_does_not_erase_supervisor_deadline_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            member = dict(identity="a", ordinal=1, base_cluster="base", exposure_role="training", engine_seed=1)
            plan = {"identity": "execution", "readiness_action": {}, "inventory": {
                "members": [member], "smoke_member_identities": ["a"], "limits": self.limits}}
            (root / "player").mkdir()
            (root / "player/9001.x86_64").touch()
            with patch.object(replay.multiprocessing, "get_context") as context, \
                    patch.object(replay, "stop_reason", side_effect=[None, "attempt_wall_limit"]):
                process = context.return_value.Process.return_value
                process.pid, process.exitcode = 123, 0
                process.is_alive.return_value = False
                process.start.side_effect = lambda: replay.files.write(root / "results/a.json", {"member_identity": "a", "complete": True})
                replay.run(plan, root, smoke=True)
            self.assertTrue(replay.files.read(root / "results/a.json")["complete"])
            receipt = replay.files.read(root / "supervision/a.json")
            self.assertFalse(receipt["execution_within_limits"])
            self.assertEqual(receipt["stop_reason"], "attempt_wall_limit")


if __name__ == "__main__":
    unittest.main()
