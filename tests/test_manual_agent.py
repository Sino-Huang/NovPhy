import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.manual_agent import (
    capture_pixel_rollout,
    deduplicate_similar_actions,
    generate_diverse_drag_release_actions,
    image_is_uniform,
    prepare_for_play,
    render_symbolic_frame,
    repl,
    save_frame,
)


class FakeBridge:
    def __init__(self):
        self.shots = []

    def shoot(self, x, y, tap_time=0, fast=False, release_time=0):
        self.shots.append((x, y, tap_time, fast, release_time))
        return 1

    def screenshot(self):
        class Screenshot:
            width = 8
            height = 6
            rgb = bytes([205, 205, 205] * (8 * 6))

        return Screenshot()

    def get_symbolic_state_without_screenshot(self):
        return [
            {
                "features": [
                    {
                        "geometry": {},
                        "type": "Feature",
                        "properties": {"id": "-166", "label": "Ground", "yindex": 4, "colormap": []},
                    },
                    {
                        "geometry": {"coordinates": [[[1, 4], [1, 1], [2, 1], [2, 4]]], "type": "Polygon"},
                        "type": "Feature",
                        "properties": {"id": "-302", "label": "Slingshot", "colormap": []},
                    },
                    {
                        "geometry": {"coordinates": [[[5, 3], [6, 2], [5, 1], [4, 2]]], "type": "Polygon"},
                        "type": "Feature",
                        "properties": {"id": "-398", "label": "pig_basic_big_1", "colormap": []},
                    },
                ]
            }
        ]


class ManualAgentTest(unittest.TestCase):
    def test_generate_diverse_actions_spreads_angle_strength_hold_and_tap(self):
        actions = generate_diverse_drag_release_actions(
            drag_start=(300, 220),
            count=8,
            strengths=(40, 90),
            angles_degrees=(-60, -15, 30, 60),
            tap_times=(0, 70),
            hold_times=(0, 120),
        )

        self.assertEqual(len(actions), 8)
        self.assertEqual({action["action_type"] for action in actions}, {"drag_hold_release"})
        self.assertEqual({tuple(action["drag_start"]) for action in actions}, {(300, 220)})
        self.assertGreaterEqual(len({tuple(action["drag_release"]) for action in actions}), 4)
        self.assertEqual({action["tapTime"] for action in actions}, {0, 70})
        self.assertEqual({action.get("holdTime", 0) for action in actions}, {0, 120})

        unique_actions = deduplicate_similar_actions(actions + actions, strength_bin=10, angle_bin_degrees=10)
        self.assertEqual(unique_actions, actions)

    def test_capture_pixel_rollout_records_timestamped_frames_at_target_fps(self):
        class RolloutBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.frame_index = 0

            def screenshot(self):
                class Screenshot:
                    width = 4
                    height = 3

                    def __init__(self, frame_index):
                        value = 40 + frame_index
                        self.rgb = bytes(channel for pixel in range(4 * 3) for channel in (value + pixel, 10, 20))

                screenshot = Screenshot(self.frame_index)
                self.frame_index += 1
                return screenshot

            def get_game_state(self):
                from src.webui.bridge import GameState

                return GameState.PLAYING

            def get_current_score(self):
                return 100 + self.frame_index

        now = [10.0]

        def clock():
            return now[0]

        def sleeper(seconds):
            now[0] += seconds

        with TemporaryDirectory() as tmp:
            metadata = capture_pixel_rollout(
                RolloutBridge(),
                Path(tmp),
                target_fps=4,
                duration_seconds=1.0,
                max_frames=4,
                clock=clock,
                sleeper=sleeper,
            )

            self.assertEqual(metadata["target_fps"], 4)
            self.assertEqual(metadata["frame_count"], 4)
            self.assertEqual([round(frame["t"], 2) for frame in metadata["frames"]], [0.0, 0.25, 0.5, 0.75])
            self.assertEqual(len(metadata["state_samples"]), 4)
            self.assertTrue((Path(tmp) / "frames" / "frame_000000.png").is_file())
            saved_metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["frame_count"], 4)

    def test_prepare_for_play_recovers_from_lost_with_next_level(self):
        class PrepareBridge:
            def __init__(self):
                self.states = ["LOST", "PLAYING"]
                self.novelty_calls = 0
                self.next_calls = 0
                self.zoom_out_calls = 0

            def get_game_state(self):
                from src.webui.bridge import GameState

                state = self.states.pop(0)
                return getattr(GameState, state)

            def get_novelty_info(self):
                self.novelty_calls += 1
                return -1

            def load_next_available_level(self):
                self.next_calls += 1
                return 1

            def fully_zoom_out(self):
                self.zoom_out_calls += 1
                return 1

        bridge = PrepareBridge()
        state = prepare_for_play(bridge, timeout=2, poll_delay=0)

        self.assertEqual(state.name, "PLAYING")
        self.assertEqual(bridge.novelty_calls, 1)
        self.assertEqual(bridge.next_calls, 1)
        self.assertEqual(bridge.zoom_out_calls, 1)

    def test_image_is_uniform_detects_solid_frame(self):
        bridge = FakeBridge()
        image = render_symbolic_frame(bridge.get_symbolic_state_without_screenshot(), 8, 6)

        self.assertFalse(image_is_uniform(image))

    def test_save_frame_rejects_uniform_screenshot_without_symbolic_fallback(self):
        bridge = FakeBridge()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            with self.assertRaisesRegex(RuntimeError, "uniform Science Birds screenshot"):
                save_frame(bridge, path)

            self.assertFalse(path.exists())

    def test_drag_hold_release_sends_relative_release(self):
        bridge = FakeBridge()
        commands = iter([
            "drag 300 220",
            "hold 120",
            "release 250 180 70 fast",
            "quit",
        ])

        with patch("builtins.input", lambda _: next(commands)), redirect_stdout(io.StringIO()):
            repl(bridge, frame_height=480)

        self.assertEqual(bridge.shots, [(250, 299, 70, True, 120)])

    def test_release_requires_drag_start(self):
        bridge = FakeBridge()
        commands = iter(["release 250 180", "quit"])
        output = io.StringIO()

        with patch("builtins.input", lambda _: next(commands)), redirect_stdout(output):
            repl(bridge, frame_height=480)

        self.assertEqual(bridge.shots, [])
        self.assertIn("drag", output.getvalue())


if __name__ == "__main__":
    unittest.main()
