import base64
import json
import subprocess
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.webui.bridge import GameState, Screenshot
from src.webui.server import AppState, create_handler


class FakeBridge:
    connected = True

    def __init__(self):
        self.shots = []
        self.loaded = []
        self.loaded_next = []
        self.restarted = 0
        self.configured = 0
        self.speeds = []
        self.ready_calls = 0
        self.novelty_calls = 0
        self.zoomed_out = 0
        self.states = [GameState.PLAYING]
        self.number_of_levels = 20

    def configure(self, agent_id, mode):
        self.configured += 1
        return (0, 0, 0)

    def set_speed(self, speed):
        self.speeds.append(speed)
        return 1

    def screenshot(self):
        return Screenshot(width=2, height=1, rgb=bytes([1, 2, 3, 4, 5, 6]))

    def get_game_state(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    def get_current_score(self):
        return 123

    def get_current_level(self):
        return 2

    def get_number_of_levels(self):
        return self.number_of_levels

    def shoot(self, x, y, tap_time=0, fast=False):
        self.shots.append((x, y, tap_time, fast))
        return 1

    def load_level(self, level):
        self.loaded.append(level)
        return 1

    def load_next_available_level(self):
        self.loaded_next.append(True)
        return 1

    def restart_level(self):
        self.restarted += 1
        return 1

    def fully_zoom_out(self):
        self.zoomed_out += 1
        return 1

    def ready_for_new_set(self):
        self.ready_calls += 1
        return (9000, 60000, 20, 5, 1, 0, 0)

    def get_novelty_info(self):
        self.novelty_calls += 1
        return 0


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.app = AppState(root=Path("/tmp/nonexistent-webui-root"))
        self.app.bridge = FakeBridge()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, payload=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = None if payload is None else json.dumps(payload)
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, data

    def test_status_reports_connected_fake_bridge(self):
        status, data = self.request("GET", "/api/status")

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertTrue(data["connected"])

    def test_frame_returns_base64_rgb_and_metadata(self):
        status, data = self.request("GET", "/api/frame")

        self.assertEqual(status, 200)
        self.assertEqual(data["width"], 2)
        self.assertEqual(data["height"], 1)
        self.assertEqual(base64.b64decode(data["rgbBase64"]), bytes([1, 2, 3, 4, 5, 6]))
        self.assertEqual(data["state"]["name"], "PLAYING")
        self.assertEqual(data["score"], 123)
        self.assertEqual(data["numberOfLevels"], 20)

    def test_frame_falls_back_to_configured_level_count_when_socket_reports_zero(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_dir = root / "sciencebirdsgames" / "Linux"
            game_dir.mkdir(parents=True)
            (game_dir / "config.xml").write_text(
                """
                <evaluation>
                  <trials>
                    <trial>
                      <game_level_set>
                        <game_levels level_path="level-1.xml" />
                        <game_levels level_path="level-2.xml" />
                      </game_level_set>
                    </trial>
                  </trials>
                </evaluation>
                """,
                encoding="utf-8",
            )
            app = AppState(root=root)
            app.bridge = FakeBridge()
            app.bridge.number_of_levels = 0
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                connection.request("GET", "/api/frame", headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                data = json.loads(response.read().decode("utf-8"))
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(response.status, 200)
        self.assertEqual(data["numberOfLevels"], 2)

    def test_shot_validates_and_calls_bridge(self):
        status, data = self.request("POST", "/api/shot", {"x": 10, "y": 20, "tapTime": 30, "fast": True})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(self.app.bridge.shots, [(10, 459, 30, True)])

    def test_shot_uses_latest_frame_height_for_bridge_y_conversion(self):
        self.request("GET", "/api/frame")

        status, data = self.request("POST", "/api/shot", {"x": 1, "y": 0})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(self.app.bridge.shots, [(1, 0, 0, False)])

    def test_async_shot_returns_before_bridge_ack(self):
        started = threading.Event()
        release = threading.Event()

        class SlowBridge(FakeBridge):
            def shoot(self, x, y, tap_time=0, fast=False):
                started.set()
                release.wait(timeout=2)
                return super().shoot(x, y, tap_time=tap_time, fast=fast)

        self.app.bridge = SlowBridge()

        before = time.monotonic()
        status, data = self.request("POST", "/api/shot", {"x": 10, "y": 20, "fast": True, "async": True})
        elapsed = time.monotonic() - before

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertTrue(data["scheduled"])
        self.assertLess(elapsed, 1)
        self.assertTrue(started.wait(timeout=1))
        self.assertEqual(self.app.bridge.shots, [])
        release.set()

        deadline = time.monotonic() + 2
        while not self.app.bridge.shots and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.app.bridge.shots, [(10, 459, 0, True)])

    def test_agent_action_translates_relative_drag_release_without_shooting(self):
        payload = {
            "action": {
                "action_type": "drag_release",
                "coordinate_frame": "slingshot_relative",
                "drag_start": [300, 220],
                "drag_release": [-50, 40],
                "tapTime": 70,
            },
            "fast": True,
        }

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["shot"], {"x": 250, "y": 180, "tapTime": 70, "fast": True})
        self.assertEqual(data["action"]["drag_start"], [300, 220])
        self.assertEqual(data["action"]["drag_release"], [-50, 40])
        self.assertEqual(self.app.bridge.shots, [])

    def test_agent_action_translates_absolute_release_without_shooting(self):
        payload = {
            "action": {
                "action_type": "drag_release",
                "coordinate_frame": "absolute",
                "drag_start": [300, 220],
                "release": [250, 180],
                "tap_time": 65,
            }
        }

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["shot"], {"x": 250, "y": 180, "tapTime": 65, "fast": False})
        self.assertEqual(data["action"]["drag_release"], [-50, 40])
        self.assertEqual(self.app.bridge.shots, [])

    def test_agent_action_requires_drag_start_for_relative_frame(self):
        status, data = self.request("POST", "/api/agent-action", {"action": {"drag_release": [-50, 40]}})

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("drag_start", data["error"])

    def test_agent_action_rejects_bad_coordinate_frame(self):
        payload = {"action": {"drag_start": [300, 220], "release": [250, 180], "coordinate_frame": "screen"}}

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("coordinate_frame", data["error"])

    def test_agent_action_rejects_bad_action_type(self):
        payload = {"action": {"action_type": "teleport", "drag_start": [300, 220], "drag_release": [-50, 40]}}

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("action_type", data["error"])

    def test_agent_action_rejects_non_numeric_coordinates_as_bad_request(self):
        payload = {"action": {"drag_start": [300, 220], "drag_release": ["left", 40]}}

        status, data = self.request("POST", "/api/agent-action", payload)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("integer", data["error"])

    def test_load_level_rejects_invalid_level(self):
        status, data = self.request("POST", "/api/load-level", {"level": 0})

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("level", data["error"])

    def test_load_level_uses_next_available_protocol(self):
        status, data = self.request("POST", "/api/load-level", {"level": 1})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["response"], 1)
        self.assertEqual(self.app.bridge.loaded, [])
        self.assertEqual(self.app.bridge.loaded_next, [True])

    def test_restart_uses_next_available_protocol(self):
        status, data = self.request("POST", "/api/restart", {})

        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["response"], 1)
        self.assertEqual(self.app.bridge.restarted, 0)
        self.assertEqual(self.app.bridge.loaded_next, [True])

    def test_connect_accepts_new_training_set_with_ready_for_new_set(self):
        self.app.bridge.states = [GameState.NEWTRAININGSET, GameState.PLAYING]

        status = self.app.connect_bridge(configure=True)

        self.assertTrue(status["connected"])
        self.assertEqual(self.app.bridge.ready_calls, 1)
        self.assertEqual(self.app.bridge.loaded, [])

    def test_connect_loads_current_level_from_menu_states(self):
        self.app.bridge.states = [GameState.MAIN_MENU, GameState.LOADING, GameState.PLAYING]

        status = self.app.connect_bridge(configure=True)

        self.assertTrue(status["connected"])
        self.assertEqual(self.app.bridge.novelty_calls, 1)
        self.assertEqual(self.app.bridge.loaded, [])
        self.assertEqual(self.app.bridge.loaded_next, [True])
        self.assertGreaterEqual(self.app.bridge.zoomed_out, 1)

    def test_connect_waits_through_loading_until_playing(self):
        self.app.bridge.states = [GameState.LOADING, GameState.LOADING, GameState.PLAYING]

        status = self.app.connect_bridge(configure=True)

        self.assertTrue(status["connected"])
        self.assertEqual(self.app.bridge.loaded, [])

    def test_start_game_launches_java_in_process_group(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_dir = root / "sciencebirdsgames" / "Linux"
            game_dir.mkdir(parents=True)
            for name in ("game_playing_interface.jar", "9001.x86_64", "config.xml"):
                (game_dir / name).write_text("ok", encoding="utf-8")
            (game_dir / "9001_Data").mkdir()
            app = AppState(root=root)
            app.connect_bridge = lambda configure=True: app.status()

            with patch("src.webui.server.subprocess.Popen") as popen:
                popen.return_value.poll.return_value = None
                app.start_game()

            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.DEVNULL)

    def test_stop_terminates_started_process_group(self):
        process = type("Process", (), {})()
        process.pid = 12345
        process.wait_calls = 0
        process.poll = lambda: None

        def wait(timeout=None):
            process.wait_calls += 1
            return 0

        process.wait = wait
        app = AppState(root=Path("/tmp/nonexistent-webui-root"), game_process=process)

        with patch("src.webui.server.os.getpgid", return_value=12345), patch("src.webui.server.os.killpg") as killpg:
            status = app.stop()

        self.assertFalse(status["gameProcessRunning"])
        self.assertEqual(killpg.call_count, 1)


if __name__ == "__main__":
    unittest.main()
