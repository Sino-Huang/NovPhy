import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import issue_76_live_episode as live
from src.webui.bridge import ObservationCaptureEngine, _freeze
from tests.test_observation_trace import engine_capture, source_bindings


class LiveEpisodeTests(unittest.TestCase):
    def test_typed_snapshot_persists_nested_frozen_metadata_and_exposes_only_rgb_time(self):
        frame = engine_capture()
        png = frame.pop("canonical_png")
        endpoint = Mock()
        endpoint.get_observation_capture.return_value = ObservationCaptureEngine(png, _freeze(frame))
        policy = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                live.capture.old.capture, "_observation_bindings", return_value=source_bindings()):
            identity = live.observe_snapshot(endpoint, Path(directory) / "observation", None,
                                             "test", "calibration", policy)
            policy.observe.assert_called_once_with(png, .4)
            saved = live.files.read(Path(directory) / "observation" / live.MANIFEST_NAME)
            self.assertEqual(identity, saved["identity"])
            self.assertIsInstance(saved["frame_records"][0]["capture_metadata"]["camera"]["position_world"], list)

    def test_accepted_segment_replays_only_agent_images_with_one_action_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                (root / f"{index}.png").write_bytes(bytes([index]))
            live.files.write(root / live.MANIFEST_NAME, {"frame_records": [
                {"agent_observation": {"relative_path": f"{i}.png"}, "fixed_time_seconds": t,
                 "oracle_entities": "must not reach the policy"}
                for i, t in enumerate((2., 2.02, 2.04))]})
            policy = Mock()
            action = {"drag_x": -80, "drag_y": 0, "release_time_ms": 600, "tap_time_ms": 0}
            live.consume_executed_segment(root, action, policy)
            self.assertEqual(policy.method_calls, [
                unittest.mock.call.observe(b"\x00", 2.), unittest.mock.call.executed(action, 2.),
                unittest.mock.call.observe(b"\x01", 2.02), unittest.mock.call.observe(b"\x02", 2.04)])

    def run_episode(self, terminal, censored=False, capture_error=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        member = dict(identity="test", base_cluster="test", exposure_role="calibration",
                      engine_seed=123, maximum_shots=3, template={}, scenario={})
        action = dict(drag_x=-80, drag_y=0, release_time_ms=600, tap_time_ms=0)
        bridge = Mock()
        bridge.get_current_level.return_value = 1
        bridge.get_game_state.return_value = SimpleNamespace(name="PLAYING")
        policy = Mock()
        policy.choose.side_effect = lambda: {"action": action.copy()}
        policy.evidence.return_value = {"fixture": True}
        engine = SimpleNamespace(novphy_log_file=SimpleNamespace(name="engine.log"))
        segment = dict(summary={"censored": censored}, capture_id="capture", native_root="none")
        patches = [
            patch.object(live.shutil, "copytree"), patch.object(live.files, "install_level"),
            patch.object(live.files, "materialize", return_value=(None, SimpleNamespace(to_dict=lambda: {}))),
            patch.object(live, "start_display", return_value=(":999", Mock())),
            patch.object(live.capture.old.capture, "start_engine", return_value=engine),
            patch.object(live.capture.old.capture, "free_port", side_effect=[22001, 22002, 22003]),
            patch.object(live.capture.old.capture, "connect_with_retry", return_value=bridge),
            patch.object(live.capture.old.capture, "prepare_for_play"),
            patch.object(live.capture.old.capture, "stop_started_engine"),
            patch.object(live.capture.old.capture, "terminate"),
            patch.object(live, "ScienceBirdsBridge"), patch.object(live.capture.old, "prepare_action"),
            patch.object(live, "observe_snapshot", return_value="rgb-only"),
            patch.object(live.capture, "capture_segment", return_value=segment, side_effect=capture_error),
            patch.object(live, "terminal_evidence", return_value=terminal),
            patch.object(live, "consume_executed_segment"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        result = live.play_episode(root, member,
            dict(attempt_seconds=30, shot_seconds=10, readiness_action=action), policy)
        return result, bridge, policy

    def test_native_clear_stops_before_a_later_interface_query(self):
        result, bridge, policy = self.run_episode({"reason": "level_clear"})
        self.assertTrue(result["gameplay_success"])
        self.assertEqual(result["attempted_shots"], [1])
        self.assertEqual(result["unattempted_shots"], [2, 3])
        bridge.get_game_state.assert_called_once()
        policy.choose.assert_called_once()

    def test_censor_is_a_failure_even_if_a_clear_is_also_reported(self):
        result, _, _ = self.run_episode({"reason": "level_clear"}, censored=True)
        self.assertFalse(result["gameplay_success"])
        self.assertEqual(result["gameplay_failure_penalty"], 1)
        self.assertEqual(result["attempted_shots"], [1])

    def test_unaccepted_capture_never_updates_the_executed_history(self):
        result, _, policy = self.run_episode(None, capture_error=ValueError("shot not accepted"))
        self.assertFalse(result["complete"])
        self.assertFalse(result["gameplay_success"])
        live.consume_executed_segment.assert_not_called()
        policy.executed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
