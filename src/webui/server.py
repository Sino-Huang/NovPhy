from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET
from urllib.parse import parse_qs, urlparse

from scripts.collect_rollouts import precise_slingshot_reference_point_from_symbolic_state
from scripts.verify_physics_player import safe_unpack
from scripts.slingshot_readiness import prepare_screen_shot

from .bridge import (
    GameState,
    PhysicsCaptureV2Failure,
    PlayingMode,
    ScienceBirdsBridge,
)
from .issue_53_review import Issue53ReviewSession
from .physics_v2_review import PhysicsV2ReviewSession, REVIEW_GOAL_LEVELS


SETUP_COMMAND = (
    "python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux "
    "--novelty-level novelty_level_0 --level-type type010101 --max-levels 20"
)
PHYSICS_PLAYER_STAGE_SCHEMA = "novphy_physics_player_stage_v1"
PHYSICS_PLAYER_UNITY_VERSION = "2019.4.41f2"
PHYSICS_V2_CAPTURE_SCHEMA = "physics_capture_v2_engine_v1"
PHYSICS_CAPTURE_PROTOCOL_VERSION = 1
PHYSICS_PLAYER_REQUIRED_FILES = (
    "game_playing_interface.jar",
    "9001.x86_64",
    "9001-player.x86_64",
    "config.xml",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


@dataclass
class AppState:
    root: Path = field(default_factory=repo_root)
    game_dir_override: Path | None = None
    game_version: str = "Linux"
    game_host: str = "127.0.0.1"
    game_port: int = 2004
    agent_id: int = 28888
    speed: int = 50
    game_headless: bool = False
    start_level: int = 1
    readiness_timeout: float = 60.0
    readiness_poll_delay: float = 0.5
    bridge: ScienceBirdsBridge | None = None
    bridge_lock: threading.Lock = field(default_factory=threading.Lock)
    game_process: subprocess.Popen | None = None
    frame_height: int = 480
    physics_v2_review: bool = False
    physics_host: str = "127.0.0.1"
    physics_port: int = 2005
    physics_bridge: ScienceBirdsBridge | None = None
    review_output_root: Path | None = None
    review_probe_plan_path: Path | None = None
    review_session: PhysicsV2ReviewSession | None = None
    review_capture_timeout: float = 60.0
    review_stage: Path | None = None
    review_runtime_dir: Path | None = None
    review_runtime_temporary: tempfile.TemporaryDirectory | None = None
    engine_game_port: int = 29001
    explicit_game_ports: bool = False
    review_reset_callback: Any | None = None
    review_goal: str | None = None
    issue_53_review_root: Path | None = None
    issue_53_review_session: Issue53ReviewSession | None = None
    manual_action_log: Path | None = None
    manual_action_context: Mapping[str, Any] = field(default_factory=dict)
    manual_action_log_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def game_dir(self) -> Path:
        if self.game_dir_override is not None:
            return Path(self.game_dir_override)
        if self.review_runtime_dir is not None:
            return self.review_runtime_dir
        return self.root / "sciencebirdsgames" / self.game_version

    def record_manual_action(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.manual_action_log is None:
            return
        record = {
            "schema": "webui_manual_action_record_v1",
            "recorded_at_unix_seconds": time.time(),
            "event": event,
            "context": dict(self.manual_action_context),
            **dict(payload),
        }
        path = Path(self.manual_action_log)
        with self.manual_action_log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def preflight_errors(self) -> list[str]:
        if self.physics_v2_review and self.review_runtime_dir is None:
            stage = self.review_stage or self.root / "sciencebirdsgames" / "physics-v2"
            required_stage = [
                stage / "novphy-physics-player-2019.4.41f2.tar.gz",
                stage / "probe-plan.json",
                stage / "review-levels" / "training.xml",
                stage / "review-levels" / "support-ready.xml",
                stage / "review-manifests" / "training.json",
                stage / "review-manifests" / "support-ready.json",
            ]
            return [f"Missing {path}" for path in required_stage if not path.exists()]
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
            "physicsV2Review": self.physics_v2_review,
            "physicsV2ReviewSession": None if self.review_session is None else self.review_session.snapshot(),
            "issue53Review": self.issue_53_review_session is not None,
            "manualActionLog": (
                None if self.manual_action_log is None else str(self.manual_action_log)
            ),
        }

    def prepare_physics_v2_review_runtime(self) -> Path:
        if not self.physics_v2_review:
            raise ValueError("physics-v2 review mode is not enabled")
        if self.review_runtime_dir is not None:
            return self.review_runtime_dir
        stage = self.review_stage or self.root / "sciencebirdsgames" / "physics-v2"
        archive = stage / "novphy-physics-player-2019.4.41f2.tar.gz"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing {archive}")
        temporary = tempfile.TemporaryDirectory(prefix="novphy_webui_physics_v2_")
        runtime = Path(temporary.name)
        try:
            safe_unpack(archive, runtime)
            self._validate_physics_v2_review_runtime(runtime)
            self._install_public_review_levels(runtime)
        except Exception:
            temporary.cleanup()
            raise
        self.review_runtime_temporary = temporary
        self.review_runtime_dir = runtime
        return runtime

    def _validate_physics_v2_review_runtime(self, runtime: Path) -> None:
        provenance_path = runtime / "provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("physics-v2 review archive provenance.json is missing or malformed") from error
        if not isinstance(provenance, dict) or provenance.get("schema_version") != PHYSICS_PLAYER_STAGE_SCHEMA:
            raise ValueError("physics-v2 review archive provenance schema is unsupported")
        unity = provenance.get("unity")
        capture = provenance.get("capture")
        files = provenance.get("files")
        if not isinstance(unity, dict) or unity.get("version") != PHYSICS_PLAYER_UNITY_VERSION:
            raise ValueError("physics-v2 review archive Unity version is unsupported")
        if (
            not isinstance(capture, dict)
            or capture.get("schema_version") != PHYSICS_V2_CAPTURE_SCHEMA
            or capture.get("protocol_version") != PHYSICS_CAPTURE_PROTOCOL_VERSION
        ):
            raise ValueError("physics-v2 review archive capture provenance is unsupported")
        if not isinstance(files, dict):
            raise ValueError("physics-v2 review archive file inventory is malformed")
        if any(
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            for relative, size in files.items()
        ):
            raise ValueError("physics-v2 review archive file inventory is malformed")
        actual_files: dict[str, int] = {}
        for path in sorted(runtime.rglob("*")):
            if path == provenance_path:
                continue
            if path.is_symlink():
                raise ValueError("physics-v2 review archive contains a non-regular file")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("physics-v2 review archive contains a non-regular file")
            actual_files[path.relative_to(runtime).as_posix()] = path.stat().st_size
        if files != actual_files:
            raise ValueError("physics-v2 review archive file inventory does not match extracted files")
        missing = [
            relative
            for relative in PHYSICS_PLAYER_REQUIRED_FILES
            if relative not in actual_files
        ]
        if missing or not (runtime / "9001_Data").is_dir():
            raise ValueError("physics-v2 review archive required files are incomplete")

    def _install_public_review_levels(self, runtime: Path) -> None:
        stage = self.review_stage or self.root / "sciencebirdsgames" / "physics-v2"
        source_root = stage / "review-levels"
        sources = (
            source_root / "training.xml",
            source_root / "support-ready.xml",
        )
        if any(not source.is_file() for source in sources):
            raise FileNotFoundError("source-bound #44 review XML artifacts are incomplete")
        selected_index = REVIEW_GOAL_LEVELS.get(self.review_goal or "", 1) - 1
        sources = (
            (sources[selected_index],)
            + sources[:selected_index]
            + sources[selected_index + 1:]
        )
        # Unity both derives the asset-bundle path from "Levels" and indexes the
        # level path recorded by its StreamingAssets scanner during selection.
        target_root = (
            runtime
            / "9001_Data"
            / "StreamingAssets"
            / "Levels"
            / "novelty_level_0"
            / "type2"
            / "Levels"
        )
        target_root.mkdir(parents=True, exist_ok=True)
        for source in sources:
            shutil.copyfile(source, target_root / source.name)
        configured_root = target_root.relative_to(runtime).as_posix()
        evaluation = ET.Element("evaluation")
        ET.SubElement(
            evaluation,
            "novelty_detection_measurement",
            {"step": "1", "measure_in_training": "False", "measure_in_testing": "False"},
        )
        trials = ET.SubElement(evaluation, "trials")
        trial = ET.SubElement(trials, "trial", {
            "id": "0",
            "number_of_executions": "1",
            "checkpoint_time_limit": "9999999",
            "checkpoint_interaction_limit": "9999999",
            "notify_novelty": "False",
        })
        level_set = ET.SubElement(trial, "game_level_set", {
            "mode": "training",
            "time_limit": "9999999",
            "total_interaction_limit": "9999999",
            "attempt_limit_per_level": "20",
            "allow_level_selection": "True",
        })
        for source in sources:
            ET.SubElement(level_set, "game_levels", {"level_path": f"{configured_root}/{source.name}"})
        ET.indent(evaluation, space="  ")
        ET.ElementTree(evaluation).write(runtime / "config.xml", encoding="utf-8", xml_declaration=True)

    def _review_root(self) -> Path:
        return self.review_output_root or self.root / ".local-artifacts" / "issue-44-webui-review"

    def _review_plan(self) -> Path:
        stage = self.review_stage or self.root / "sciencebirdsgames" / "physics-v2"
        return self.review_probe_plan_path or stage / "probe-plan.json"

    def stage_physics_v2_review(self, goal: str, action: dict[str, Any]) -> dict[str, Any]:
        if not self.physics_v2_review:
            raise ValueError("physics-v2 review mode is not enabled")
        if self.review_session is not None and self.review_session.state in {"exploring", "replaying", "frozen"}:
            raise ValueError("the current physics-v2 review session must finish before staging another action")
        self.review_session = PhysicsV2ReviewSession(
            self._review_root(),
            probe_plan_path=self._review_plan(),
        )
        return self.review_session.stage(goal, action)

    def load_physics_v2_review_goal(self, goal: str) -> int:
        if not self.physics_v2_review:
            raise ValueError("physics-v2 review mode is not enabled")
        if goal not in REVIEW_GOAL_LEVELS:
            raise ValueError(f"unknown physics-v2 review goal: {goal}")
        level = REVIEW_GOAL_LEVELS[goal]
        with self.bridge_lock:
            if self.bridge is None or not self.bridge.connected:
                raise RuntimeError("Not connected to Science Birds. Start or connect first.")
        self._restart_physics_v2_review_engine(goal)
        return level

    def freeze_physics_v2_review(self) -> dict[str, Any]:
        if self.review_session is None:
            raise ValueError("no physics-v2 review session is active")
        return self.review_session.freeze_replay()

    def _physics_v2_engine_record(self) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.review_capture_timeout
        while True:
            bridge = self.physics_bridge
            if bridge is None:
                bridge = ScienceBirdsBridge(self.physics_host, self.physics_port, timeout=10.0)
            if not bridge.connected:
                bridge.connect()
            try:
                capture = bridge.get_physics_capture_v2()
                record = getattr(capture, "record", None)
                if not isinstance(record, Mapping):
                    raise RuntimeError("request 71 returned no engine capture record")
                return record
            except PhysicsCaptureV2Failure as error:
                if error.code != 3 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)

    def run_physics_v2_review(self, *, replay: bool) -> dict[str, Any]:
        if self.review_session is None:
            raise ValueError("no physics-v2 review session is active")
        transition = self.review_session.begin_replay() if replay else self.review_session.begin_exploration()
        shot = transition["socket_command"]
        try:
            if replay:
                if self.review_reset_callback is not None:
                    self.review_reset_callback(self.review_session.goal)
                else:
                    self._reset_physics_v2_review_engine()
            with self.bridge_lock:
                if self.bridge is None or not self.bridge.connected:
                    raise RuntimeError("Not connected to Science Birds. Start or connect first.")
                action = dict(self.review_session.action or {})
                retained_anchor = action.get("slingshot_reference") if replay else None
                prepared = prepare_screen_shot(
                    self.bridge,
                    action,
                    frame_height=int(action.get("frame_height", self.frame_height)),
                    execution_speed=1,
                    frozen_socket_command=shot if replay else None,
                    retained_anchor=retained_anchor if isinstance(retained_anchor, Mapping) else None,
                    fast=True,
                )
                if not replay:
                    self.review_session.bind_prepared_exploration(prepared)
                else:
                    self.review_session.bind_prepared_replay(prepared)
                prepared.execute()
            record = self._physics_v2_engine_record()
            if replay:
                return self.review_session.complete_replay(record)
            return self.review_session.complete_exploration(record)
        except Exception as error:
            self.review_session.fail_active_capture(error)
            raise

    def _reset_physics_v2_review_engine(self) -> None:
        if self.review_session is None or self.review_session.goal is None:
            raise ValueError("confirmatory replay has no bound review goal")
        self._restart_physics_v2_review_engine(self.review_session.goal)

    def _restart_physics_v2_review_engine(self, goal: str) -> None:
        self.review_goal = goal
        if self.review_reset_callback is not None:
            self.review_reset_callback(goal)
            return
        self.stop()
        self.start_game()

    def configured_level_count(self) -> int | None:
        config_path = self.game_dir / "config.xml"
        if not config_path.is_file():
            return None
        try:
            root = ET.parse(config_path).getroot()
        except ET.ParseError:
            return None
        return len(root.findall(".//game_levels"))

    def configured_level_paths(self) -> list[str]:
        config_path = self.game_dir / "config.xml"
        if not config_path.is_file():
            return []
        try:
            root = ET.parse(config_path).getroot()
        except ET.ParseError:
            return []
        paths = []
        for node in root.findall(".//game_levels"):
            level_path = (node.get("level_path") or node.text or "").strip()
            if level_path:
                paths.append(level_path)
        return paths

    def trajectory_world_width(self, current_level: int | None = None) -> float | None:
        level_paths = self.configured_level_paths()
        if not level_paths:
            return None
        index = 0
        if current_level is not None and 1 <= current_level <= len(level_paths):
            index = current_level - 1
        level_path = self.game_dir / level_paths[index]
        if not level_path.is_file():
            return None
        try:
            root = ET.parse(level_path).getroot()
        except ET.ParseError:
            return None
        camera = root.find(".//Camera")
        for value in (camera.get("maxWidth") if camera is not None else None, camera.get("minWidth") if camera is not None else None, root.get("width")):
            if value is None:
                continue
            try:
                width = float(value)
            except ValueError:
                continue
            if width > 0:
                return width
        return None

    def start_game(self) -> dict[str, Any]:
        if self.physics_v2_review:
            self.prepare_physics_v2_review_runtime()
        errors = self.preflight_errors()
        if errors:
            raise RuntimeError("; ".join(errors) + f". Run: {SETUP_COMMAND}")

        if self.game_process is None or self.game_process.poll() is not None:
            command = ["java", "-jar", "./game_playing_interface.jar"]
            if self.game_headless:
                command.append("--headless")
            if self.physics_v2_review or self.explicit_game_ports:
                command.extend([
                    "--agent-port", str(self.game_port),
                    "--game-start-port", str(self.engine_game_port),
                    "--physics-port", str(self.physics_port),
                ])
            command.append("--dev")
            environment = os.environ.copy()
            if self.physics_v2_review or self.explicit_game_ports:
                environment["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(
                    self.physics_port
                )
            if self.physics_v2_review:
                environment["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
            self.game_process = subprocess.Popen(
                command,
                cwd=self.game_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                env=environment,
            )

        # Review mode boots the packaged Unity player, which can take tens of
        # seconds on a remote/software-GL display before the jar ACKs configure.
        deadline = time.time() + (
            60 if self.physics_v2_review or self.explicit_game_ports else 15
        )
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
        if self.physics_bridge is not None:
            if hasattr(self.physics_bridge, "disconnect"):
                self.physics_bridge.disconnect()
            self.physics_bridge = None
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
        if self.review_runtime_temporary is not None:
            self.review_runtime_temporary.cleanup()
            self.review_runtime_temporary = None
            self.review_runtime_dir = None
        return self.status()


def create_handler(app: AppState):
    class WebUIHandler(BaseHTTPRequestHandler):
        server_version = "NovPhyWebUI/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            issue_53_route = self._issue_53_item_route(path)
            if path == "/api/issue-53-review":
                self._send_json({"ok": True, "review": self._require_issue_53_review().snapshot()})
                return
            if issue_53_route is not None:
                index, action = issue_53_route
                try:
                    review = self._require_issue_53_review()
                    if action == "steps":
                        query = parse_qs(urlparse(self.path).query)
                        start = int(query.get("start", [0])[0])
                        count = int(query.get("count", [100])[0])
                        self._send_json({"ok": True, **review.fixed_steps(index, start=start, count=count)})
                    elif action == "video":
                        video = review.replay_video(index)
                        self._send_file(
                            video,
                            mimetypes.guess_type(video.name)[0]
                            or "application/octet-stream",
                        )
                    elif action == "detail":
                        self._send_json({"ok": True, "detail": review.item_detail(index)})
                    else:
                        self._send_json({"ok": False, "error": "Unknown issue-53 review endpoint"}, status=404)
                except PermissionError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=403)
                except ValueError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)
                return
            if path == "/api/status":
                self._send_json(app.status())
                return
            if path == "/api/frame":
                self._handle_frame()
                return
            if path == "/api/physics-v2-review":
                session = None if app.review_session is None else app.review_session.snapshot()
                self._send_json({"ok": True, "enabled": app.physics_v2_review, "session": session})
                return
            if path == "/api/physics-v2-review/steps":
                query = parse_qs(urlparse(self.path).query)
                start = int(query.get("start", [0])[0])
                count = int(query.get("count", [100])[0])
                if app.review_session is None:
                    self._send_json({"ok": True, "start": start, "count": 0, "total": 0, "steps": []})
                else:
                    self._send_json({"ok": True, **app.review_session.fixed_steps(start=start, count=count)})
                return
            self._serve_static(path)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                issue_53_route = self._issue_53_item_route(path)
                if path == "/api/issue-53-review/authorize":
                    payload = self._read_json()
                    identity = payload.get("authorizationIdentity")
                    if not isinstance(identity, str):
                        raise ValueError("Final review authorization identity is required")
                    review = self._require_issue_53_review()
                    self._send_json({"ok": True, "review": review.authorize_final_access(identity)})
                elif issue_53_route is not None:
                    index, action = issue_53_route
                    review = self._require_issue_53_review()
                    if action == "open":
                        self._send_json({"ok": True, "detail": review.open_trace(index)})
                    elif action == "replay":
                        self._send_json({"ok": True, "detail": review.run_replay(index)})
                    elif action == "decision":
                        payload = self._read_json()
                        self._send_json({
                            "ok": True,
                            "review": review.record_decision(
                                index,
                                decision=payload.get("decision"),
                                notes=payload.get("notes"),
                                reviewer=payload.get("reviewer"),
                            ),
                        })
                    else:
                        self._send_json({"ok": False, "error": "Unknown issue-53 review endpoint"}, status=404)
                elif path == "/api/start":
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
                elif path == "/api/physics-v2-review/stage":
                    payload = self._read_json()
                    goal = payload.get("goal")
                    action = payload.get("action")
                    if not isinstance(goal, str) or not isinstance(action, dict):
                        raise ValueError("physics-v2 review stage requires goal and action")
                    self._send_json({"ok": True, "session": app.stage_physics_v2_review(goal, action)})
                elif path == "/api/physics-v2-review/load-goal":
                    payload = self._read_json()
                    goal = payload.get("goal")
                    if not isinstance(goal, str):
                        raise ValueError("physics-v2 review goal is required")
                    self._send_json({"ok": True, "level": app.load_physics_v2_review_goal(goal)})
                elif path == "/api/physics-v2-review/explore":
                    self._send_json({"ok": True, "session": app.run_physics_v2_review(replay=False)})
                elif path == "/api/physics-v2-review/freeze":
                    self._send_json({"ok": True, "session": app.freeze_physics_v2_review()})
                elif path == "/api/physics-v2-review/replay":
                    self._send_json({"ok": True, "session": app.run_physics_v2_review(replay=True)})
                else:
                    self._send_json({"ok": False, "error": "Unknown API endpoint"}, status=404)
            except PermissionError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=403)
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
                current_level = self._safe_call(lambda: bridge.get_current_level())
                app.frame_height = screenshot.height
                payload = {
                    "ok": True,
                    "width": screenshot.width,
                    "height": screenshot.height,
                    "rgbBase64": base64.b64encode(screenshot.rgb).decode("ascii"),
                    "state": self._state_payload(bridge),
                    "score": self._safe_call(lambda: bridge.get_current_score()),
                    "currentLevel": current_level,
                    "numberOfLevels": number_of_levels,
                    "trajectoryWorldWidth": app.trajectory_world_width(
                        current_level if isinstance(current_level, int) else None
                    ),
                    "trajectorySlingCenter": self._safe_call(
                        lambda: precise_slingshot_reference_point_from_symbolic_state(
                            bridge.get_symbolic_state_without_screenshot(),
                            screenshot.height,
                        )
                    ),
                }
            self._send_json(payload)

        def _handle_shot(self) -> None:
            payload = self._read_json()
            x = self._required_int(payload, "x")
            y = self._required_int(payload, "y")
            tap_time = int(payload.get("tapTime", 0))
            release_time = int(payload.get("releaseTime", payload.get("release_time", 0)))
            fast = bool(payload.get("fast", False))
            run_async = bool(payload.get("async", False))
            shot_action = {
                "action_type": "drag_hold_release",
                "coordinate_frame": "absolute",
                "release": [x, y],
                "tapTime": tap_time,
                "releaseTime": release_time,
            }

            def action(bridge):
                try:
                    response = prepare_screen_shot(
                        bridge,
                        shot_action,
                        frame_height=app.frame_height,
                        execution_speed=app.speed,
                        fast=fast,
                    ).execute()
                except Exception as error:
                    app.record_manual_action("shot_failed", {
                        "shot": dict(payload),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    raise
                state = self._safe_call(lambda: bridge.get_game_state())
                app.record_manual_action("shot_executed", {
                    "shot": dict(payload),
                    "game_state_after": (
                        state.name if isinstance(state, GameState) else state
                    ),
                    "score_after": self._safe_call(
                        lambda: bridge.get_current_score()
                    ),
                })
                return response
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
            if normalized["releaseTime"]:
                shot["releaseTime"] = normalized["releaseTime"]
            app.record_manual_action("agent_action_validated", {
                "action": normalized["action"],
                "shot": shot,
            })
            self._send_json({"ok": True, "action": normalized["action"], "shot": shot})

        def _normalize_agent_action(self, action: dict[str, Any]) -> dict[str, Any]:
            action_type = action.get("action_type", "drag_release")
            if action_type not in {"drag_release", "drag_hold_release"}:
                raise ValueError("action_type must be drag_release or drag_hold_release")
            coordinate_frame = action.get("coordinate_frame", "slingshot_relative")
            if coordinate_frame not in {"slingshot_relative", "absolute"}:
                raise ValueError("coordinate_frame must be slingshot_relative or absolute")
            release = action.get("drag_release", action.get("release"))
            if release is None:
                raise ValueError("drag_release or release is required")
            if not isinstance(release, list | tuple) or len(release) < 2:
                raise ValueError("release must contain x and y values")

            tap_time = self._agent_action_int(action.get("tapTime", action.get("tap_time", 0)))
            release_time = self._agent_action_int(action.get("holdTime", action.get("releaseTime", action.get("release_time", 0))))
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

            normalized_action = {
                "action_type": action_type,
                "coordinate_frame": "slingshot_relative",
                "drag_start": [start_x, start_y],
                "drag_release": [dx, dy],
                "tapTime": tap_time,
            }
            if release_time:
                normalized_action["holdTime"] = release_time

            return {
                "action": normalized_action,
                "shot_x": shot_x,
                "shot_y": shot_y,
                "tapTime": tap_time,
                "releaseTime": release_time,
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

        def _require_issue_53_review(self) -> Issue53ReviewSession:
            if app.issue_53_review_session is None:
                raise ValueError("Issue-53 review mode is not enabled")
            return app.issue_53_review_session

        def _issue_53_item_route(self, path: str) -> tuple[int, str] | None:
            parts = path.strip("/").split("/")
            if len(parts) < 4 or parts[:3] != ["api", "issue-53-review", "items"]:
                return None
            try:
                index = int(parts[3])
            except ValueError:
                return None
            action = "detail" if len(parts) == 4 else parts[4]
            return index, action

        def _required_int(self, payload: dict[str, Any], key: str) -> int:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")
            try:
                return int(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer") from exc

        def _serve_static(self, path: str) -> None:
            if path in ("/", ""):
                relative = (
                    "issue53.html"
                    if app.issue_53_review_session is not None
                    else "index.html"
                )
            else:
                relative = path.lstrip("/")
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

        def _send_file(self, path: Path, content_type: str) -> None:
            size = path.stat().st_size
            start = 0
            end = size - 1
            range_header = self.headers.get("Range")
            status = 200
            if range_header and range_header.startswith("bytes="):
                bounds = range_header.removeprefix("bytes=").split("-", 1)
                start = int(bounds[0] or 0)
                end = min(int(bounds[1]) if bounds[1] else end, end)
                if start < 0 or start > end:
                    self.send_error(416)
                    return
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))

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
    parser.add_argument("--physics-port", type=int, default=2005, help="Direct request-71 socket port")
    parser.add_argument("--speed", type=int, default=50, help="Simulation speed set after connect")
    parser.add_argument("--game-headless", action="store_true", default=os.environ.get("NOVPHY_WEBUI_GAME_HEADLESS") == "1")
    parser.add_argument("--physics-v2-review", action="store_true", help="Enable guided diagnostic and confirmatory request-71 capture")
    parser.add_argument(
        "--issue-53-review-root",
        type=Path,
        help="Enable retained issue-53 mismatch review from this production runtime",
    )
    parser.add_argument("--review-output-dir", type=Path, help="Local physics-v2 review session directory")
    parser.add_argument("--physics-v2-stage", type=Path, help="Verified packaged physics-v2 stage")
    parser.add_argument("--engine-game-port", type=int, default=29001, help="Unity startup port used by the packaged interface")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.physics_v2_review and args.issue_53_review_root is not None:
        raise SystemExit("--physics-v2-review and --issue-53-review-root are separate modes")
    if args.physics_v2_review:
        # The packaged physics-v2 player binds its physics-capture listener on
        # its compiled-in default port 2004: the jar accepts --physics-port but
        # does not relay it to the player it spawns (AIBirdsConnection falls
        # back to 2004 when no --physics-port CLI arg reaches the player).
        # Keep the jar's agent listener off 2004 and aim the request-71 bridge
        # at the port the player actually binds.
        if args.game_port == 2004:
            args.game_port = 29002
        args.physics_port = 2004
    app = AppState(
        game_host=args.game_host,
        game_port=args.game_port,
        speed=args.speed,
        game_headless=args.game_headless,
        physics_v2_review=args.physics_v2_review,
        physics_port=args.physics_port,
        review_output_root=args.review_output_dir,
        review_stage=args.physics_v2_stage,
        engine_game_port=args.engine_game_port,
        issue_53_review_root=args.issue_53_review_root,
    )
    if args.issue_53_review_root is not None:
        review_output = args.review_output_dir or (
            repo_root() / ".local-artifacts/issue-53-human-review-v2"
        )
        app.issue_53_review_session = Issue53ReviewSession(
            args.issue_53_review_root,
            review_output,
            repository_root=repo_root(),
            speed=args.speed,
        )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
    print(f"NovPhy WebUI: http://{args.host}:{args.port}/")
    print(f"Science Birds target: {app.game_host}:{app.game_port}")
    if app.physics_v2_review:
        print(f"Physics-v2 review: enabled (request 71 at {app.physics_host}:{app.physics_port})")
    if app.issue_53_review_session is not None:
        print(f"Issue-53 retained-evidence review: {app.issue_53_review_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()


if __name__ == "__main__":
    main()
