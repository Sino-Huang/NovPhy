from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from urllib.parse import urlparse

from .bridge import GameState, PlayingMode, ScienceBirdsBridge


SETUP_COMMAND = (
    "python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux "
    "--novelty-level novelty_level_0 --level-type type010101 --max-levels 20"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


@dataclass
class AppState:
    root: Path = field(default_factory=repo_root)
    game_version: str = "Linux"
    game_host: str = "127.0.0.1"
    game_port: int = 2004
    agent_id: int = 28888
    speed: int = 50
    game_headless: bool = False
    start_level: int = 1
    readiness_timeout: float = 30.0
    readiness_poll_delay: float = 0.5
    bridge: ScienceBirdsBridge | None = None
    bridge_lock: threading.Lock = field(default_factory=threading.Lock)
    game_process: subprocess.Popen | None = None
    frame_height: int = 480

    @property
    def game_dir(self) -> Path:
        return self.root / "sciencebirdsgames" / self.game_version

    def preflight_errors(self) -> list[str]:
        required = [
            self.game_dir / "game_playing_interface.jar",
            self.game_dir / "9001.x86_64",
            self.game_dir / "9001_Data",
            self.game_dir / "config.xml",
        ]
        return [f"Missing {path}" for path in required if not path.exists()]

    def status(self) -> dict[str, Any]:
        running = self.game_process is not None and self.game_process.poll() is None
        return {
            "ok": True,
            "connected": self.bridge is not None and self.bridge.connected,
            "gameProcessRunning": running,
            "gameDir": str(self.game_dir),
            "gameHost": self.game_host,
            "gamePort": self.game_port,
            "preflightErrors": self.preflight_errors(),
        }

    def configured_level_count(self) -> int | None:
        config_path = self.game_dir / "config.xml"
        if not config_path.is_file():
            return None
        try:
            root = ET.parse(config_path).getroot()
        except ET.ParseError:
            return None
        return len(root.findall(".//game_levels"))

    def start_game(self) -> dict[str, Any]:
        errors = self.preflight_errors()
        if errors:
            raise RuntimeError("; ".join(errors) + f". Run: {SETUP_COMMAND}")

        if self.game_process is None or self.game_process.poll() is not None:
            command = ["java", "-jar", "./game_playing_interface.jar"]
            if self.game_headless:
                command.append("--headless")
            command.append("--dev")
            self.game_process = subprocess.Popen(
                command,
                cwd=self.game_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )

        deadline = time.time() + 15
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self.connect_bridge(configure=True)
                return self.status()
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"Game process started, but WebUI could not connect to {self.game_host}:{self.game_port}: {last_error}")

    def connect_bridge(self, configure: bool = True) -> dict[str, Any]:
        with self.bridge_lock:
            if self.bridge is None or not self.bridge.connected:
                self.bridge = ScienceBirdsBridge(self.game_host, self.game_port)
                self.bridge.connect()
            if configure:
                self.bridge.configure(self.agent_id, PlayingMode.TRAINING)
                self.bridge.set_speed(self.speed)
                self._prepare_game_for_play(self.bridge)
        return self.status()

    def _prepare_game_for_play(self, bridge: ScienceBirdsBridge) -> None:
        deadline = time.time() + self.readiness_timeout
        new_set_states = {
            GameState.NEWTRAININGSET,
            GameState.RESUMETRAINING,
            GameState.NEWTRIAL,
            GameState.NEWTESTSET,
        }
        menu_states = {
            GameState.MAIN_MENU,
            GameState.EPISODE_MENU,
            GameState.LEVEL_SELECTION,
        }

        while time.time() < deadline:
            state = bridge.get_game_state()
            if state == GameState.PLAYING:
                bridge.fully_zoom_out()
                return
            if state in new_set_states:
                bridge.ready_for_new_set()
            elif state in menu_states:
                bridge.get_novelty_info()
                bridge.load_next_available_level()
            elif state == GameState.LOADING:
                pass
            else:
                raise RuntimeError(f"Science Birds did not enter a playable state; current state is {state.name}")
            time.sleep(self.readiness_poll_delay)

        raise TimeoutError("Science Birds did not reach PLAYING before the readiness timeout")

    def stop(self) -> dict[str, Any]:
        with self.bridge_lock:
            if self.bridge is not None:
                self.bridge.disconnect()
                self.bridge = None
        if self.game_process is not None:
            try:
                os.killpg(os.getpgid(self.game_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.game_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.game_process.pid), signal.SIGKILL)
                self.game_process.wait(timeout=5)
            self.game_process = None
        return self.status()


def create_handler(app: AppState):
    class WebUIHandler(BaseHTTPRequestHandler):
        server_version = "NovPhyWebUI/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/status":
                self._send_json(app.status())
                return
            if path == "/api/frame":
                self._handle_frame()
                return
            self._serve_static(path)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/api/start":
                    self._send_json(app.start_game())
                elif path == "/api/connect":
                    self._send_json(app.connect_bridge(configure=True))
                elif path == "/api/stop":
                    self._send_json(app.stop())
                elif path == "/api/shot":
                    self._handle_shot()
                elif path == "/api/agent-action":
                    self._handle_agent_action()
                elif path == "/api/load-level":
                    self._handle_load_level()
                elif path == "/api/restart":
                    self._bridge_action(lambda bridge: bridge.load_next_available_level())
                elif path == "/api/zoom-out":
                    self._bridge_action(lambda bridge: bridge.fully_zoom_out())
                elif path == "/api/zoom-in":
                    self._bridge_action(lambda bridge: bridge.fully_zoom_in())
                else:
                    self._send_json({"ok": False, "error": "Unknown API endpoint"}, status=404)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}", file=sys.stderr)

        def _handle_frame(self) -> None:
            with app.bridge_lock:
                bridge = self._require_bridge()
                screenshot = bridge.screenshot()
                number_of_levels = self._number_of_levels_payload(bridge)
                payload = {
                    "ok": True,
                    "width": screenshot.width,
                    "height": screenshot.height,
                    "rgbBase64": base64.b64encode(screenshot.rgb).decode("ascii"),
                    "state": self._state_payload(bridge),
                    "score": self._safe_call(lambda: bridge.get_current_score()),
                    "currentLevel": self._safe_call(lambda: bridge.get_current_level()),
                    "numberOfLevels": number_of_levels,
                }
                app.frame_height = screenshot.height
            self._send_json(payload)

        def _handle_shot(self) -> None:
            payload = self._read_json()
            x = self._required_int(payload, "x")
            y = self._required_int(payload, "y")
            tap_time = int(payload.get("tapTime", 0))
            fast = bool(payload.get("fast", False))
            run_async = bool(payload.get("async", False))
            bridge_y = max(0, app.frame_height - 1 - y)
            action = lambda bridge: bridge.shoot(x, bridge_y, tap_time=tap_time, fast=fast)
            if run_async:
                self._bridge_action_async(action)
                self._send_json({"ok": True, "scheduled": True})
                return
            self._bridge_action(action)

        def _handle_agent_action(self) -> None:
            payload = self._read_json()
            action = payload.get("action")
            if not isinstance(action, dict):
                raise ValueError("action must be an object")

            normalized = self._normalize_agent_action(action)
            fast = bool(payload.get("fast", False))
            shot = {
                "x": normalized["shot_x"],
                "y": normalized["shot_y"],
                "tapTime": normalized["tapTime"],
                "fast": fast,
            }
            self._send_json({"ok": True, "action": normalized["action"], "shot": shot})

        def _normalize_agent_action(self, action: dict[str, Any]) -> dict[str, Any]:
            action_type = action.get("action_type", "drag_release")
            if action_type != "drag_release":
                raise ValueError("action_type must be drag_release")
            coordinate_frame = action.get("coordinate_frame", "slingshot_relative")
            if coordinate_frame not in {"slingshot_relative", "absolute"}:
                raise ValueError("coordinate_frame must be slingshot_relative or absolute")
            release = action.get("drag_release", action.get("release"))
            if release is None:
                raise ValueError("drag_release or release is required")
            if not isinstance(release, list | tuple) or len(release) < 2:
                raise ValueError("release must contain x and y values")

            tap_time = self._agent_action_int(action.get("tapTime", action.get("tap_time", 0)))
            drag_start = action.get("drag_start")
            if coordinate_frame == "slingshot_relative":
                if not isinstance(drag_start, list | tuple) or len(drag_start) < 2:
                    raise ValueError("drag_start is required for slingshot_relative actions")
                start_x = self._agent_action_int(drag_start[0])
                start_y = self._agent_action_int(drag_start[1])
                dx = self._agent_action_int(release[0])
                dy = self._agent_action_int(release[1])
                shot_x = start_x + dx
                shot_y = start_y - dy
            else:
                if drag_start is None:
                    drag_start = [release[0], release[1]]
                if not isinstance(drag_start, list | tuple) or len(drag_start) < 2:
                    raise ValueError("drag_start must contain x and y values")
                start_x = self._agent_action_int(drag_start[0])
                start_y = self._agent_action_int(drag_start[1])
                shot_x = self._agent_action_int(release[0])
                shot_y = self._agent_action_int(release[1])
                dx = shot_x - start_x
                dy = start_y - shot_y

            return {
                "action": {
                    "action_type": action_type,
                    "coordinate_frame": "slingshot_relative",
                    "drag_start": [start_x, start_y],
                    "drag_release": [dx, dy],
                    "tapTime": tap_time,
                },
                "shot_x": shot_x,
                "shot_y": shot_y,
                "tapTime": tap_time,
            }

        def _agent_action_int(self, value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("agent action coordinates and tap time must be integers") from exc

        def _handle_load_level(self) -> None:
            payload = self._read_json()
            level = self._required_int(payload, "level")
            if level < 1:
                raise ValueError("level must be >= 1")
            self._bridge_action(lambda bridge: bridge.load_next_available_level())

        def _bridge_action(self, action) -> None:
            with app.bridge_lock:
                response = action(self._require_bridge())
            self._send_json({"ok": True, "response": response})

        def _bridge_action_async(self, action) -> None:
            def run() -> None:
                try:
                    with app.bridge_lock:
                        action(self._require_bridge())
                except Exception as exc:
                    print(f"Async bridge action failed: {exc}", file=sys.stderr)

            threading.Thread(target=run, daemon=True).start()

        def _require_bridge(self):
            if app.bridge is None or not app.bridge.connected:
                raise RuntimeError("Not connected to Science Birds. Start or connect first.")
            return app.bridge

        def _state_payload(self, bridge) -> dict[str, Any]:
            value = self._safe_call(lambda: bridge.get_game_state())
            if isinstance(value, GameState):
                return {"value": int(value), "name": value.name}
            if isinstance(value, int):
                return {"value": value, "name": GameState(value).name if value in set(item.value for item in GameState) else "UNKNOWN"}
            return {"value": None, "name": None}

        def _number_of_levels_payload(self, bridge) -> int | None:
            value = self._safe_call(lambda: bridge.get_number_of_levels())
            if isinstance(value, int) and value > 0:
                return value
            return app.configured_level_count()

        def _safe_call(self, call):
            try:
                return call()
            except Exception:
                return None

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON payload must be an object")
            return payload

        def _required_int(self, payload: dict[str, Any], key: str) -> int:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")
            try:
                return int(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer") from exc

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (static_dir() / relative).resolve()
            root = static_dir().resolve()
            if root not in target.parents and target != root:
                self._send_json({"ok": False, "error": "Invalid path"}, status=400)
                return
            if not target.is_file():
                self._send_json({"ok": False, "error": "Not found"}, status=404)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return WebUIHandler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local NovPhy Science Birds WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=8766, help="Web server port")
    parser.add_argument("--game-host", default="127.0.0.1", help="Science Birds socket host")
    parser.add_argument("--game-port", type=int, default=2004, help="Science Birds socket port")
    parser.add_argument("--speed", type=int, default=50, help="Simulation speed set after connect")
    parser.add_argument("--game-headless", action="store_true", default=os.environ.get("NOVPHY_WEBUI_GAME_HEADLESS") == "1")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app = AppState(game_host=args.game_host, game_port=args.game_port, speed=args.speed, game_headless=args.game_headless)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
    print(f"NovPhy WebUI: http://{args.host}:{args.port}/")
    print(f"Science Birds target: {app.game_host}:{app.game_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()


if __name__ == "__main__":
    main()
