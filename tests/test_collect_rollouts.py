import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts.collect_rollouts import collect_rollouts, main, stop_owned_engine, write_action_plan


class FakeBridge:
    def __init__(self):
        self.shots = []
        self.frame_index = 0

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        self.shots.append((x, y, tap_time, fast, release_time))
        return 1

    def configure(self, agent_id, mode):
        self.configured = (agent_id, mode)
        return (0, 0, 1)

    def set_speed(self, speed):
        self.speed = speed
        return 1

    def disconnect(self):
        self.disconnected = True

    def screenshot(self):
        class Screenshot:
            width = 4
            height = 3

            def __init__(self, frame_index):
                value = 50 + frame_index
                self.rgb = bytes(channel for pixel in range(4 * 3) for channel in (value + pixel, 20, 30))

        screenshot = Screenshot(self.frame_index)
        self.frame_index += 1
        return screenshot

    def get_game_state(self):
        from src.webui.bridge import GameState

        return GameState.PLAYING

    def get_current_score(self):
        return 200 + self.frame_index


class CollectRolloutsTest(unittest.TestCase):
    def test_collect_rollouts_writes_manifest_and_per_shot_metadata(self):
        actions = [
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
                "holdTime": 120,
            },
            {
                "action_type": "drag_hold_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-80, -20],
                "tapTime": 0,
            },
        ]
        now = [5.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            manifest = collect_rollouts(
                FakeBridge(),
                Path(tmp),
                actions,
                target_fps=2,
                duration_seconds=1.0,
                frame_height=480,
                max_frames=2,
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(manifest["rollout_count"], 2)
            self.assertEqual(manifest["rollouts"][0]["shoot_response"], 1)
            self.assertEqual(manifest["rollouts"][0]["frame_count"], 2)
            self.assertTrue((Path(tmp) / "shot_001" / "metadata.json").is_file())
            saved_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["rollout_count"], 2)

    def test_write_action_plan_dry_run_writes_actions_without_bridge(self):
        with TemporaryDirectory() as tmp:
            path = write_action_plan(Path(tmp), count=3)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["action_count"], 3)
            self.assertEqual(len(payload["actions"]), 3)
            self.assertEqual(payload["actions"][0]["action_type"], "drag_hold_release")

    def test_main_auto_starts_engine_after_connection_refusal(self):
        class FakeProcess:
            pid = 1234

            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True

        process = FakeProcess()
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp:
            args = [
                "collect_rollouts.py",
                "--output-dir",
                tmp,
                "--count",
                "1",
                "--fps",
                "1",
                "--duration",
                "0.01",
                "--no-prepare",
            ]

            with (
                patch("sys.argv", args),
                patch("scripts.collect_rollouts.connect_with_retry", side_effect=[RuntimeError("connection refused"), bridge]) as connect,
                patch("scripts.collect_rollouts.start_engine", return_value=process) as start,
            ):
                main()

        self.assertEqual(connect.call_count, 2)
        start.assert_called_once()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)
        self.assertTrue(bridge.disconnected)

    def test_stop_owned_engine_kills_process_that_ignores_terminate(self):
        class SlowProcess:
            pid = 4321

            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                if not self.killed:
                    raise TimeoutError("still running")

            def kill(self):
                self.killed = True

        process = SlowProcess()

        stop_owned_engine(process)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_main_reports_auto_start_failure_without_traceback(self):
        args = ["collect_rollouts.py", "--output-dir", "data/rollout-plan-debug", "--count", "3"]
        stderr = io.StringIO()

        with (
            patch("sys.argv", args),
            patch("scripts.collect_rollouts.connect_with_retry", side_effect=RuntimeError("connection refused")),
            patch("scripts.collect_rollouts.start_engine", side_effect=FileNotFoundError("missing jar")),
        ):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("could not start the local engine", stderr.getvalue())
        self.assertIn("--dry-run", stderr.getvalue())

    def test_main_preflights_output_dir_before_starting_engine(self):
        args = ["collect_rollouts.py", "--output-dir", "/data/collect_rollouts_debug", "--count", "1"]
        stderr = io.StringIO()

        with (
            patch("sys.argv", args),
            patch("scripts.collect_rollouts.ensure_output_dir", side_effect=PermissionError("denied")) as ensure,
            patch("scripts.collect_rollouts.start_engine") as start,
            patch("scripts.collect_rollouts.connect_with_retry") as connect,
        ):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        ensure.assert_called_once()
        start.assert_not_called()
        connect.assert_not_called()
        self.assertIn("Cannot write output directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
