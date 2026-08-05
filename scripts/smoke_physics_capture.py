#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Run one isolated request-70 physics capture and accept it only when valid."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Final, TypeAlias

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_rollouts import (
    action_to_shot,
    anchor_action_to_slingshot_reference,
    capture_physics_rollout,
    current_slingshot_reference,
)
from scripts.manual_agent import connect_with_retry, prepare_for_play
from scripts.rollout_artifacts import validate_physics_shot_artifact
from scripts.rollout_validation_types import PhysicsArtifactError
from scripts.smoke_protection import (
    ProtectionError,
    canonical_root_from_git,
    protected_receipt,
    protected_roots,
    tree_digest,
)
from scripts.verify_physics_player import safe_unpack, verify_payload
from src.webui.bridge import JsonValue as BridgeJsonValue, PhysicsCaptureV1, PhysicsCaptureV1Failure, PlayingMode, RequestCode, ScienceBirdsBridge

CAPTURE_READ_TIMEOUT_SECONDS: Final = 120.0
FRAME_HEIGHT_PIXELS: Final = 480
WIRE_STATE_FIELDS: Final = frozenset(("schema_version", "capture_id", "sequence", "render_frame", "render_time", "fixed_step", "fixed_time", "coordinates", "nodes", "raw_contacts", "support_edges"))
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SmokeError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class CapturedRequest:
    capture: PhysicsCaptureV1

    def get_physics_capture_v1(self) -> PhysicsCaptureV1:
        state = {key: mutable_json(value) for key, value in self.capture.state.items()}
        events = tuple({key: mutable_json(value) for key, value in event.items()} for event in self.capture.events)
        return PhysicsCaptureV1(self.capture.png, state, events)


def mutable_json(value: BridgeJsonValue) -> JsonValue:
    """Convert bridge-frozen JSON containers into writer-compatible containers."""
    if isinstance(value, Mapping):
        return {key: mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [mutable_json(item) for item in value]
    return value


def perform_known_action(bridge: ScienceBirdsBridge) -> JsonObject:
    reference = current_slingshot_reference(bridge, FRAME_HEIGHT_PIXELS)
    if reference is None:
        raise SmokeError("known level has no request-62 slingshot reference")
    action = anchor_action_to_slingshot_reference(
        {"coordinate_frame": "slingshot_relative", "drag_start": [0, 0], "drag_release": [-50, 40], "tapTime": 0, "holdTime": 1000},
        reference,
    )
    shot = action_to_shot(action, frame_height=FRAME_HEIGHT_PIXELS)
    ground_truth_count = bridge.shoot_and_record_ground_truth(
        shot["x"],
        shot["y"],
        tap_time=shot["tapTime"],
        release_time=shot["releaseTime"],
    )
    if ground_truth_count < 1:
        raise SmokeError("known recorded gameplay action returned no ground truth")
    return {"response": 1, "request_code": int(RequestCode.GT_SHOOT), "ground_truth_count": ground_truth_count, "slingshot_reference": reference, "socket_x": shot["x"], "socket_y": shot["y"], "tap_time": shot["tapTime"], "release_time": shot["releaseTime"]}


def capture_finalized_action(physics_port: int, *, deadline_seconds: float = 30.0) -> PhysicsCaptureV1:
    deadline = time.monotonic() + deadline_seconds
    while True:
        physics = connect_with_retry("127.0.0.1", physics_port, timeout=CAPTURE_READ_TIMEOUT_SECONDS, deadline_seconds=30.0)
        try:
            return physics.get_physics_capture_v1()
        except PhysicsCaptureV1Failure as error:
            if error.code != 4 or time.monotonic() >= deadline:
                raise
        finally:
            physics.disconnect()
        time.sleep(0.25)


def require_action_events(events: tuple[Mapping[str, BridgeJsonValue], ...]) -> tuple[str, ...]:
    event_types = tuple(str(event.get("event_type", "")) for event in events)
    if "bird_launched" not in event_types:
        raise SmokeError("request-70 capture missing authoritative bird_launched event")
    return event_types


def archive_details(stage: Path, clone: Path) -> tuple[Path, str, str, str]:
    """Unpack and verify a staged archive, returning archive/player/protocol hashes."""
    receipt = (stage / "archive.sha256").read_text(encoding="ascii").strip().split()
    if len(receipt) != 2:
        raise SmokeError("malformed archive.sha256")
    archive = stage / receipt[1]
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha != receipt[0]:
        raise SmokeError("archive SHA-256 mismatch")
    safe_unpack(archive, clone)
    verify_payload(clone)
    manifest = json.loads((clone / "provenance.json").read_text(encoding="utf-8"))
    player_sha = manifest["files"]["9001-player.x86_64"]
    protocol_sha = hashlib.sha256((clone / "game_playing_interface.jar").read_bytes()).hexdigest()
    return archive, archive_sha, player_sha, protocol_sha


def free_port() -> int:
    """Reserve an unused TCP port number, releasing it before the engine starts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def terminate(process: subprocess.Popen[bytes] | None) -> str:
    """Stop a process group and return a receipt string."""
    if process is None:
        return "not-started"
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    return f"pid={process.pid}:exit={process.returncode}"


def start_display(log_path: Path) -> tuple[str, subprocess.Popen[bytes]]:
    """Start a private Xvfb display for this run."""
    display = f":{190 + (os.getpid() % 50)}"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            ["Xvnc", display, "-geometry", "1024x768", "-depth", "24", "-SecurityTypes", "None", "-rfbport", "0"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    time.sleep(0.25)
    if process.poll() is not None:
        raise SmokeError("Xvnc failed to start")
    return display, process


def run_smoke(args: argparse.Namespace) -> tuple[JsonObject, int]:
    """Execute the smoke scenario and write its report before returning."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    protected = protected_roots(args.canonical_root)
    before = {name: protected_receipt(name, path) for name, path in protected.items()}
    report: JsonObject = {"status": "rejected", "accepted_shot": None, "canonical_root": str(args.canonical_root), "protected_before": before, "protected_receipt_mode": {"canonical_project": "content_sha256", "production_player": "content_sha256", "active_data": "complete_nested_find_manifest_sha256"}}
    engine: subprocess.Popen[bytes] | None = None
    xvfb: subprocess.Popen[bytes] | None = None
    temporary_path = ""
    try:
        with tempfile.TemporaryDirectory(prefix="novphy-physics-smoke-") as temporary:
            temporary_path = temporary
            clone = Path(temporary) / "player"
            report["phase"] = "verify-stage"
            _, archive_sha, player_sha, protocol_sha = archive_details(args.stage, clone)
            report["phase"] = "start-display"
            display, xvfb = start_display(args.output_dir / "xvfb.log")
            agent_port = args.agent_port or free_port()
            game_port = args.game_port or free_port()
            engine_log = args.output_dir / "engine.log"
            report.update({"phase": "start-engine", "ports": {"agent": agent_port, "game": game_port, "physics": args.physics_port}, "display": display})
            engine = subprocess.Popen(
                ["java", "-jar", "./game_playing_interface.jar", "--agent-port", str(agent_port), "--game-start-port", str(game_port), "--dev"],
                cwd=clone, env={**os.environ, "DISPLAY": display}, stdout=engine_log.open("wb"), stderr=subprocess.STDOUT, start_new_session=True,
            )
            report["phase"] = "connect-agent"
            bridge = connect_with_retry("127.0.0.1", agent_port, timeout=10.0, deadline_seconds=90.0)
            try:
                report["phase"] = "configure-agent"
                bridge.configure(agent_id=28701, mode=PlayingMode.TRAINING)
                report["phase"] = "load-known-level"
                prepare_for_play(bridge, timeout=120.0, poll_delay=1.0)
                report["phase"] = "perform-action"
                report["action"] = perform_known_action(bridge)
                if args.inject_request_failure:
                    raise SmokeError("injected request-70 failure")
                shot = args.output_dir / "shot_001.tmp"
                report["phase"] = "connect-request-70"
                report["phase"] = "request-70"
                capture = capture_finalized_action(args.physics_port)
                state_fields = sorted(capture.state)
                missing_fields = sorted(WIRE_STATE_FIELDS - capture.state.keys())
                event_types = require_action_events(capture.events)
                report["wire_capture"] = {"state_fields": state_fields, "missing_state_fields": missing_fields, "event_count": len(capture.events), "event_types": list(event_types), "render_frame": capture.state["render_frame"]}
                if missing_fields:
                    raise SmokeError("request-70 state missing contract fields: " + ", ".join(missing_fields))
                capture_physics_rollout(CapturedRequest(capture), shot, target_fps=1.0, duration_seconds=1.0, max_frames=1, player_sha256=player_sha, protocol_sha256=protocol_sha, archive_sha256=archive_sha)
            finally:
                bridge.disconnect()
            if args.inject_frame_mismatch:
                state_path = shot / "physics_state.jsonl"
                lines = state_path.read_text(encoding="utf-8").splitlines()
                record = json.loads(lines[1])
                record["rgb_frame"]["render_frame"] += 1
                lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
                state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if args.inject_missing_sidecar:
                (shot / "physics_events.jsonl").unlink()
            report["phase"] = "validate-artifact"
            summary = validate_physics_shot_artifact(shot)
            final_shot = args.output_dir / "shot_001"
            shot.replace(final_shot)
            report.update({"status": "accepted", "phase": "complete", "accepted_shot": str(final_shot), "artifact": {"states": summary.state_count, "events": summary.event_count}, "provenance": {"archive_sha256": archive_sha, "player_sha256": player_sha, "protocol_sha256": protocol_sha}})
            return report, 0
    except (OSError, ValueError, KeyError, RuntimeError, SmokeError, ProtectionError, PhysicsArtifactError, tarfile.TarError, TimeoutError, TypeError) as error:
        report["error"] = str(error)
        shot = args.output_dir / "shot_001.tmp"
        if shot.exists():
            quarantine = args.output_dir / "invalid_attempts" / "shot_001"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            shot.replace(quarantine)
            report["quarantine"] = str(quarantine)
        return report, 1
    finally:
        report["cleanup"] = {"engine": terminate(engine), "xvfb": terminate(xvfb), "temporary_clone": temporary_path}
        after = {name: protected_receipt(name, path) for name, path in protected.items()}
        report["protected_after"] = after
        report["protected_unchanged"] = before == after
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / ".omo/evidence/world-model-physics-instrumentation/task-8-smoke.json")
    parser.add_argument("--agent-port", type=int)
    parser.add_argument("--game-port", type=int)
    parser.add_argument("--physics-port", type=int, default=2004)
    parser.add_argument("--canonical-root", type=Path, default=canonical_root_from_git(ROOT))
    parser.add_argument("--inject-frame-mismatch", action="store_true")
    parser.add_argument("--inject-missing-sidecar", action="store_true")
    parser.add_argument("--inject-request-failure", action="store_true")
    args = parser.parse_args()
    report, code = run_smoke(args)
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
