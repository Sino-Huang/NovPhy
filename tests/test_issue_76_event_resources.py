from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from scripts import issue_76_event_resources as resources


class EventResourcesTests(unittest.TestCase):
    def test_slots_have_disjoint_ports_and_busy_port_fails_without_search(self):
        ports = [port for slot in range(8) for port in resources.worker_ports(slot)]
        self.assertEqual(len(set(ports)), 24)
        with self.assertRaises(ValueError):
            resources.worker_ports(8)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaises(OSError):
                resources.check_ports_available((sock.getsockname()[1],))

    def test_completed_attempts_are_never_rescanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root, publication = Path(directory) / "root", Path(directory) / "public"
            for path, count in ((root / "player/data", 2), (root / "inventory.json", 3),
                                (root / "attempts/done/raw", 11), (root / "attempts/live/raw", 7),
                                (publication / "inventory.json", 5)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * count)
            ledger = resources.ArtifactLedger(root, publication)
            self.assertEqual(ledger.seal("done"), 11)
            scan = resources.tree_bytes
            def active_only(path):
                if Path(path) == root / "attempts/done":
                    self.fail("sealed trace was rescanned")
                return scan(path)
            with patch.object(resources, "tree_bytes", side_effect=active_only):
                self.assertEqual(ledger.snapshot(["live"]), 28)
                restored = resources.ArtifactLedger(root, publication, completed=ledger.completed)
                self.assertEqual(restored.snapshot(["live"]), 28)
            with self.assertRaises(ValueError):
                ledger.seal("done")
            with self.assertRaises(ValueError):
                ledger.snapshot(["done"])

    def test_atomic_publish_discards_partial_sum_and_rescans_once(self):
        with patch.object(resources, "_tree_bytes", side_effect=[FileNotFoundError("staging"), 23]) as scan:
            self.assertEqual(resources.tree_bytes(Path("attempt")), 23)
            self.assertEqual(scan.call_count, 2)
        with patch.object(resources, "_tree_bytes", side_effect=FileNotFoundError("staging")) as scan:
            with self.assertRaises(FileNotFoundError):
                resources.tree_bytes(Path("attempt"))
            self.assertEqual(scan.call_count, 2)
        with patch.object(resources, "_tree_bytes", side_effect=PermissionError("unreadable")) as scan:
            with self.assertRaises(PermissionError):
                resources.tree_bytes(Path("attempt"))
            self.assertEqual(scan.call_count, 1)

    def test_spawned_worker_preserves_each_role_and_restores_allocator(self):
        original = resources.live.capture.old.capture.free_port
        assignment = {"identity": "new", "ordinal": 1, "source_member_identity": "source",
                      "candidate_ordinal": 0, "original_action_reference": True,
                      "action": {"drag_x": -80, "drag_y": 10, "release_time_ms": 600, "tap_time_ms": 0}}
        for role in resources.inventory.ROLES:
            source = {"identity": "source", "exposure_role": role, "scenario": {"retained": True}}
            def play(root, member, limits, policy):
                self.assertEqual(member["exposure_role"], role)
                self.assertEqual(member["scenario"], source["scenario"])
                self.assertEqual(tuple(resources.live.capture.old.capture.free_port() for _ in range(3)), resources.worker_ports(2))
                return {"complete": True}
            with patch.object(resources, "check_ports_available"), patch.object(resources.live, "play_episode", side_effect=play):
                self.assertEqual(resources.worker(Path("root"), source, assignment, {}, 2), {"complete": True})
            self.assertIs(resources.live.capture.old.capture.free_port, original)
        with patch.object(resources, "check_ports_available"), patch.object(resources.live, "play_episode", side_effect=RuntimeError("capture")):
            with self.assertRaises(RuntimeError):
                resources.worker(Path("root"), source, assignment, {}, 2)
        self.assertIs(resources.live.capture.old.capture.free_port, original)

    def test_global_and_attempt_limits_stay_separate(self):
        limits = {"collection_wall_seconds": 100, "smoke_wall_seconds": 20,
                  "aggregate_cpu_rss_mib": 32, "artifact_bytes": 50, "attempt_seconds": 10,
                  "worker_cpu_rss_mib": 4, "rgb_frames_per_shot_max": 601}
        args = dict(active_seconds=1, smoke_seconds=1, rss_mib=1, artifact_bytes=1, smoke=True)
        self.assertIsNone(resources.global_stop(limits, **args))
        for key, value, reason in (("active_seconds", 100, "collection_wall_limit"),
                                   ("smoke_seconds", 20, "smoke_wall_limit"),
                                   ("rss_mib", 33, "aggregate_memory_limit"),
                                   ("artifact_bytes", 51, "artifact_limit")):
            self.assertEqual(resources.global_stop(limits, **{**args, key: value}), reason)
        args = dict(wall_seconds=1, rss_mib=1, captures=1, frames=601)
        self.assertIsNone(resources.attempt_stop(limits, **args))
        for key, value, reason in (("wall_seconds", 10, "attempt_wall_limit"),
                                   ("rss_mib", 5, "worker_memory_limit"),
                                   ("captures", 2, "capture_or_frame_limit"),
                                   ("frames", 602, "capture_or_frame_limit")):
            self.assertEqual(resources.attempt_stop(limits, **{**args, key: value}), reason)
