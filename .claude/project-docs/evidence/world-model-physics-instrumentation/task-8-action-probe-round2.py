#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from scripts.collect_rollouts import action_to_shot, anchor_action_to_slingshot_reference, current_slingshot_reference
from scripts.manual_agent import connect_with_retry, prepare_for_play
from scripts.smoke_physics_capture import archive_details, free_port, mutable_json, start_display, terminate
from src.webui.bridge import PlayingMode


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "sciencebirdsgames/physics-v1"
PROBE_CASE = os.environ.get("NOVPHY_PROBE_CASE", "fast_immediate")
OUTPUT = ROOT / ".omo/evidence/world-model-physics-instrumentation/task-8-action-probe-round2" / PROBE_CASE


def request70() -> dict:
    physics = connect_with_retry("127.0.0.1", 2004, timeout=30.0, deadline_seconds=30.0)
    try:
        capture = physics.get_physics_capture_v1()
        return {
            "state": mutable_json(capture.state),
            "events": [mutable_json(event) for event in capture.events],
        }
    finally:
        physics.disconnect()


def action_receipt(bridge, *, fast: bool, record: bool = False) -> dict:
    reference = current_slingshot_reference(bridge, 480)
    if reference is None:
        raise RuntimeError("missing slingshot reference")
    action = anchor_action_to_slingshot_reference(
        {
            "coordinate_frame": "slingshot_relative",
            "drag_start": [0, 0],
            "drag_release": [-50, 40],
            "tapTime": 0,
            "holdTime": 1000,
        },
        reference,
    )
    shot = action_to_shot(action, frame_height=480)
    if record:
        bridge._send(38, "iiiii", shot["x"], shot["y"], shot["releaseTime"], shot["tapTime"], 1)
        ground_truth_count = bridge._read("I")[0]
        for _ in range(ground_truth_count):
            bridge._read_ground_truth()
        response = ground_truth_count
    else:
        response = bridge.shoot(
            shot["x"],
            shot["y"],
            tap_time=shot["tapTime"],
            fast=fast,
            release_time=shot["releaseTime"],
        )
    return {"fast": fast, "record": record, "reference": reference, "shot": shot, "response": response}


def run_case(bridge, *, name: str, ready_wait: float, fast: bool, capture_wait: float, record: bool = False) -> dict:
    prepare_for_play(bridge, timeout=120.0, poll_delay=0.25)
    time.sleep(ready_wait)
    symbolic_before = mutable_json(bridge.get_symbolic_state_without_screenshot())
    physics_before = request70()
    action = action_receipt(bridge, fast=fast, record=record)
    deadline = time.monotonic() + capture_wait
    state_samples = []
    while time.monotonic() < deadline:
        state = bridge.get_game_state().name
        state_samples.append(state)
        if state in {"WON", "LOST"}:
            break
        time.sleep(0.5)
    physics_after = request70()
    return {
        "name": name,
        "ready_wait_seconds": ready_wait,
        "capture_wait_seconds": capture_wait,
        "game_state_after": bridge.get_game_state().name,
        "request62_before": symbolic_before,
        "request70_before": physics_before,
        "action": action,
        "state_samples": state_samples,
        "request70_after": physics_after,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    engine = None
    display_process = None
    report = {"archive_sha256": None, "cases": [], "cleanup": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="novphy-task8-action-probe-") as temporary:
            clone = Path(temporary) / "player"
            _, archive_sha, _, _ = archive_details(STAGE, clone)
            report["archive_sha256"] = archive_sha
            display, display_process = start_display(OUTPUT / "xvnc.log")
            agent_port = free_port()
            game_port = free_port()
            with (OUTPUT / "engine.log").open("wb") as log:
                engine = subprocess.Popen(
                    ["java", "-jar", "./game_playing_interface.jar", "--agent-port", str(agent_port), "--game-start-port", str(game_port), "--dev"],
                    cwd=clone,
                    env={**os.environ, "DISPLAY": display},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            bridge = connect_with_retry("127.0.0.1", agent_port, timeout=10.0, deadline_seconds=90.0)
            try:
                bridge.configure(agent_id=28702, mode=PlayingMode.TRAINING)
                prepare_for_play(bridge, timeout=120.0, poll_delay=0.25)
                cases = {
                    "fast_immediate": {"name": "fast_immediate", "ready_wait": 0.0, "fast": True, "capture_wait": 0.0},
                    "fast_after_readiness_wait": {"name": "fast_after_readiness_wait", "ready_wait": 5.0, "fast": True, "capture_wait": 0.0},
                    "safe_immediate": {"name": "safe_immediate", "ready_wait": 0.0, "fast": False, "capture_wait": 0.0},
                    "fast_delayed_capture": {"name": "fast_delayed_capture", "ready_wait": 0.0, "fast": True, "capture_wait": 2.0},
                    "gt_shoot": {"name": "gt_shoot", "ready_wait": 0.0, "fast": False, "capture_wait": 0.0, "record": True},
                    "gt_shoot_terminal": {"name": "gt_shoot_terminal", "ready_wait": 0.0, "fast": False, "capture_wait": 90.0, "record": True},
                }
                selected = cases[PROBE_CASE]
                report["cases"].append(run_case(bridge, **selected))
            finally:
                bridge.disconnect()
        return 0
    finally:
        report["cleanup"] = {"engine": terminate(engine), "xvnc": terminate(display_process)}
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
