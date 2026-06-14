import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.manual_agent import image_is_uniform, prepare_for_play, render_symbolic_frame, repl, save_frame


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
