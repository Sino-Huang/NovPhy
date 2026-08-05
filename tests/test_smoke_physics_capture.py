from __future__ import annotations

from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock

from scripts.smoke_physics_capture import (
    archive_details,
    canonical_root_from_git,
    capture_finalized_action,
    CapturedRequest,
    free_port,
    perform_known_action,
    protected_receipt,
    require_action_events,
    start_display,
    tree_digest,
)
from src.webui.bridge import PhysicsCaptureV1


ROOT = Path(__file__).resolve().parents[1]


class SmokePhysicsCaptureTests(unittest.TestCase):
    def test_canonical_root_comes_from_git_common_directory(self) -> None:
        canonical_root = canonical_root_from_git(ROOT)
        self.assertTrue((canonical_root / ".git").is_dir())
        self.assertNotEqual(canonical_root, ROOT)

    def test_known_action_uses_request_62_slingshot_before_public_recorded_shoot(self) -> None:
        class Bridge:
            def __init__(self) -> None:
                self.recorded_shots: list[tuple[int, int, int, int, int]] = []

            def get_symbolic_state_without_screenshot(self) -> list[dict]:
                return [{"features": [{"properties": {"label": "Slingshot"}, "geometry": {"type": "Polygon", "coordinates": [[[100, 200], [120, 200], [120, 260], [100, 260]]]}}]}]

            def shoot_and_record_ground_truth(self, x: int, y: int, tap_time: int = 0, release_time: int = 0, frequency: int = 1) -> int:
                self.recorded_shots.append((x, y, tap_time, release_time, frequency))
                return 1

        bridge = Bridge()
        receipt = perform_known_action(bridge)
        self.assertEqual(bridge.recorded_shots, [(59, 247, 0, 1000, 1)])
        self.assertEqual(receipt["response"], 1)

    @mock.patch("scripts.smoke_physics_capture.time.sleep")
    @mock.patch("scripts.smoke_physics_capture.connect_with_retry")
    def test_action_capture_retries_only_until_recorder_is_finalized(self, connect: mock.Mock, _sleep: mock.Mock) -> None:
        from src.webui.bridge import PhysicsCaptureV1Failure

        pending = mock.Mock()
        pending.get_physics_capture_v1.side_effect = PhysicsCaptureV1Failure(4, "no finalized recorder batch")
        finalized = mock.Mock()
        finalized.get_physics_capture_v1.return_value = PhysicsCaptureV1(b"png", MappingProxyType({}), ())
        connect.side_effect = [pending, finalized]

        capture = capture_finalized_action(2004, deadline_seconds=1.0)

        self.assertEqual(capture.png, b"png")
        pending.disconnect.assert_called_once_with()
        finalized.disconnect.assert_called_once_with()

    def test_action_capture_requires_authoritative_launch_event(self) -> None:
        with self.assertRaisesRegex(Exception, "bird_launched"):
            require_action_events(())
        event_types = require_action_events((MappingProxyType({"event_type": "bird_launched"}),))
        self.assertEqual(event_types, ("bird_launched",))

    def test_captured_request_thaws_nested_bridge_json(self) -> None:
        capture = PhysicsCaptureV1(b"png", MappingProxyType({"coordinates": MappingProxyType({"world": "unity"})}), ())
        thawed = CapturedRequest(capture).get_physics_capture_v1()
        self.assertEqual(thawed.state, {"coordinates": {"world": "unity"}})

    @mock.patch("scripts.smoke_physics_capture.time.sleep")
    @mock.patch("scripts.smoke_physics_capture.subprocess.Popen")
    def test_private_display_uses_glx_capable_xvnc(self, popen: mock.Mock, _sleep: mock.Mock) -> None:
        process = popen.return_value
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary:
            start_display(Path(temporary) / "display.log")
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "Xvnc")
        self.assertIn("-SecurityTypes", command)
        self.assertIn("-rfbport", command)

    def test_tree_digest_is_stable_and_detects_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "nested" / "file.bin"
            target.parent.mkdir()
            target.write_bytes(b"one")
            first = tree_digest(root)
            self.assertEqual(first, tree_digest(root))
            target.write_bytes(b"two")
            self.assertNotEqual(first, tree_digest(root))

    def test_active_data_receipt_detects_nested_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "episode" / "frames" / "frame.png"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"one")
            first = protected_receipt("active_data", root)
            nested.write_bytes(b"two")
            self.assertNotEqual(first, protected_receipt("active_data", root))

    def test_missing_root_has_explicit_digest(self) -> None:
        self.assertEqual(tree_digest(ROOT / "does-not-exist-for-physics-smoke"), "ABSENT")

    def test_stage_archive_provenance_is_verified_in_a_clone(self) -> None:
        stage = ROOT / "sciencebirdsgames" / "physics-v1"
        with tempfile.TemporaryDirectory() as temporary:
            _, archive_sha, player_sha, protocol_sha = archive_details(stage, Path(temporary) / "clone")
        self.assertEqual(len(archive_sha), 64)
        self.assertEqual(len(player_sha), 64)
        self.assertEqual(len(protocol_sha), 64)

    def test_free_port_returns_bindable_port(self) -> None:
        port = free_port()
        self.assertGreater(port, 0)


if __name__ == "__main__":
    unittest.main()
