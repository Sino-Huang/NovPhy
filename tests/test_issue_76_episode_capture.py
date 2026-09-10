from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import issue_76_episode_capture as episode
from scripts import run_issue_76_canonical_smoke as smoke


class EpisodeCaptureTests(unittest.TestCase):
    def test_actuator_preparation_never_uses_speed_50_or_ground_truth_shot(self):
        bridge = SimpleNamespace(set_speed=Mock(), fully_zoom_out=Mock(),
                                 get_symbolic_state_without_screenshot=Mock(), shoot=Mock(return_value=1))
        anchor = {"gameX": 100., "gameY": 200., "canvasX": 100., "canvasY": 280., "pixelsPerWorldUnit": 20.}
        with patch.object(episode, "_stable_observation", return_value=(anchor, 2)):
            prepared = episode.prepare_action(bridge, smoke.ACTIONS[0])
        self.assertEqual([c.args[0] for c in bridge.set_speed.call_args_list], [1])
        self.assertFalse(prepared.record_ground_truth)
        self.assertEqual(prepared.socket_command["gameX"], 20)
        self.assertEqual(prepared.socket_command["gameY"], 190)
        self.assertEqual(prepared.execute(), 1)
        self.assertTrue(bridge.shoot.call_args.kwargs["fast"])
        self.assertEqual(bridge.shoot.call_args.kwargs["release_time"], 600)

    def test_previous_shot_response_cannot_be_adopted_as_new_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            aligned = root / "aligned"
            aligned.mkdir()
            (aligned / "old").mkdir()
            def execute():
                (aligned / "new").mkdir()
                return 1
            prepared = SimpleNamespace(execute=execute)
            physics = SimpleNamespace(get_physics_capture_v2=lambda: SimpleNamespace(record={"capture_id": "old"}))
            with patch.object(episode, "prepare_action", return_value=prepared), patch.object(episode, "persist_segment") as persist:
                with self.assertRaisesRegex(ValueError, "stale or different"):
                    episode.capture_segment(None, physics, aligned, root / "shot", None, "episode:shot-2", {})
                persist.assert_not_called()

    def test_transport_failure_does_not_read_or_publish_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            physics = Mock()
            with patch.object(episode, "prepare_action", return_value=SimpleNamespace(execute=lambda: 0)):
                with self.assertRaisesRegex(ValueError, "did not accept"):
                    episode.capture_segment(None, physics, Path(temp) / "aligned", Path(temp) / "shot", None, "shot", {})
            physics.get_physics_capture_v2.assert_not_called()

    def test_interruption_keeps_completed_segments_and_unused_shot_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode.expansion.write(root / "progress.json", {"attempted_shots": [1, 2]})
            episode.expansion.write(root / "shot-1.json", {"identity": "first"})
            member = {"identity": "episode", "actions": list(smoke.ACTIONS)}
            result = smoke.interrupted_result(root, member, "watchdog")
            self.assertEqual(result["segments"], [{"identity": "first"}])
            self.assertEqual(result["attempted_shots"], [1, 2])
            self.assertEqual(result["unattempted_shots"], [3])
            self.assertFalse(result["complete"])

    def test_completed_and_interrupted_episodes_are_not_replayed_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            done = {"identity": "done", "actions": list(smoke.ACTIONS)}
            interrupted = {"identity": "interrupted", "actions": list(smoke.ACTIONS)}
            episode.expansion.write(smoke.result_path(output, done), {"complete": True})
            (output / "attempts/interrupted").mkdir(parents=True)
            plan = {"limits": smoke.limits(), "members": [done, interrupted]}
            with patch.object(smoke.multiprocessing, "get_context", side_effect=AssertionError("unexpected replay")):
                smoke.run(output, plan)
            result = episode.expansion.read(smoke.result_path(output, interrupted))
            self.assertEqual(result["failure"], "interrupted_single_attempt_retained_no_retry")

    def test_c2_has_eight_shots_not_eight_three_shot_episodes(self):
        self.assertEqual(sum(c[-1] for c in smoke.CELLS), 8)
        self.assertEqual(len(smoke.CELLS), 4)
        self.assertEqual(smoke.CELLS[0][:3], smoke.CELLS[1][:3])
        self.assertEqual(smoke.limits()["technical_retries"], 0)
        self.assertEqual(smoke.limits()["models_run"], [])


if __name__ == "__main__":
    unittest.main()
