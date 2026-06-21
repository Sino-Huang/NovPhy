#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manual_agent import (  # noqa: E402
    capture_pixel_rollout,
    connect_with_retry,
    generate_diverse_drag_release_actions,
    prepare_for_play,
    start_engine,
)
from src.webui.bridge import PlayingMode  # noqa: E402


def _action_int(value) -> int:
    return int(value)


def action_to_shot(action: dict, *, frame_height: int) -> dict:
    release = action.get("drag_release", action.get("release"))
    if not isinstance(release, list | tuple) or len(release) < 2:
        raise ValueError("drag_release or release must contain x and y values")

    coordinate_frame = action.get("coordinate_frame", "slingshot_relative")
    if coordinate_frame == "slingshot_relative":
        drag_start = action.get("drag_start")
        if not isinstance(drag_start, list | tuple) or len(drag_start) < 2:
            raise ValueError("drag_start is required for slingshot_relative actions")
        shot_x = _action_int(drag_start[0]) + _action_int(release[0])
        shot_y = _action_int(drag_start[1]) - _action_int(release[1])
    elif coordinate_frame == "absolute":
        shot_x = _action_int(release[0])
        shot_y = _action_int(release[1])
    else:
        raise ValueError("coordinate_frame must be slingshot_relative or absolute")

    return {
        "x": shot_x,
        "y": max(0, frame_height - 1 - shot_y),
        "tapTime": _action_int(action.get("tapTime", action.get("tap_time", 0))),
        "releaseTime": _action_int(action.get("holdTime", action.get("releaseTime", action.get("release_time", 0)))),
    }


def write_action_plan(output_dir: Path, *, count: int, drag_start: tuple[int, int] = (300, 220)) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = generate_diverse_drag_release_actions(drag_start=drag_start, count=count)
    path = output_dir / "action_plan.json"
    path.write_text(json.dumps({"action_count": len(actions), "actions": actions}, indent=2), encoding="utf-8")
    return path


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def collect_rollouts(
    bridge,
    output_dir: Path,
    actions: list[dict],
    *,
    target_fps: float,
    duration_seconds: float,
    frame_height: int = 480,
    fast: bool = True,
    max_frames: int | None = None,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = []
    for index, action in enumerate(actions, start=1):
        shot = action_to_shot(action, frame_height=frame_height)
        response = bridge.shoot(
            shot["x"],
            shot["y"],
            tap_time=shot["tapTime"],
            fast=fast,
            release_time=shot["releaseTime"],
        )
        shot_dir = output_dir / f"shot_{index:03d}"
        metadata = capture_pixel_rollout(
            bridge,
            shot_dir,
            target_fps=target_fps,
            duration_seconds=duration_seconds,
            max_frames=max_frames,
            action=action,
            clock=clock,
            sleeper=sleeper,
        )
        rollouts.append(
            {
                "name": shot_dir.name,
                "action": action,
                "shot": shot,
                "shoot_response": response,
                "frame_count": metadata["frame_count"],
                "metadata_path": str(shot_dir / "metadata.json"),
            }
        )

    manifest = {
        "capture_source": "scripts.collect_rollouts",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect diverse high-FPS Science Birds pixel rollouts")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2004)
    parser.add_argument("--agent-id", type=int, default=28888)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--read-timeout", type=float, default=300)
    parser.add_argument("--prepare-timeout", type=float, default=60)
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--start-engine", action="store_true")
    parser.add_argument("--game-headless", action="store_true", help="Only used with --start-engine")
    parser.add_argument("--safe", action="store_true", help="Use safe shot instead of fast shot")
    parser.add_argument("--dry-run", action="store_true", help="Only write action_plan.json; do not connect to Java")
    return parser


def connect_or_start_engine(args) -> tuple[object, object | None]:
    try:
        return connect_with_retry(args.host, args.port, timeout=args.read_timeout, deadline_seconds=args.connect_timeout), None
    except RuntimeError as first_error:
        try:
            engine_process = start_engine(ROOT / "sciencebirdsgames" / "Linux", args.game_headless)
        except (OSError, FileNotFoundError) as exc:
            message = (
                f"Could not connect to Science Birds at {args.host}:{args.port}, and could not start the local engine.\n"
                f"Connection details: {first_error}\n"
                f"Startup details: {exc}\n"
                "For an action plan only, run: scripts/collect_rollouts.py --dry-run --output-dir <dir> --count <n>"
            )
            print(message, file=sys.stderr)
            raise SystemExit(2) from None
        print(f"Started engine pid={engine_process.pid}")
        try:
            bridge = connect_with_retry(args.host, args.port, timeout=args.read_timeout, deadline_seconds=args.connect_timeout)
        except RuntimeError as exc:
            stop_owned_engine(engine_process)
            message = (
                f"Started Science Birds engine pid={engine_process.pid}, but could not connect to {args.host}:{args.port}.\n"
                f"Initial connection details: {first_error}\n"
                f"Retry details: {exc}\n"
                "For an action plan only, run: scripts/collect_rollouts.py --dry-run --output-dir <dir> --count <n>"
            )
            print(message, file=sys.stderr)
            raise SystemExit(2) from None
        return bridge, engine_process


def stop_owned_engine(engine_process) -> None:
    if engine_process is None or engine_process.poll() is not None:
        return
    engine_process.terminate()
    try:
        engine_process.wait(timeout=5)
    except (subprocess.TimeoutExpired, TimeoutError):
        engine_process.kill()
        engine_process.wait(timeout=5)


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        path = write_action_plan(args.output_dir, count=args.count)
        print(json.dumps({"action_plan": str(path)}, indent=2))
        return

    try:
        ensure_output_dir(args.output_dir)
    except OSError as exc:
        print(f"Cannot write output directory {args.output_dir}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if args.start_engine:
        engine_process = start_engine(ROOT / "sciencebirdsgames" / "Linux", args.game_headless)
        print(f"Started engine pid={engine_process.pid}")
        bridge = connect_with_retry(args.host, args.port, timeout=args.read_timeout, deadline_seconds=args.connect_timeout)
    else:
        bridge, engine_process = connect_or_start_engine(args)
    try:
        print(f"configure -> {bridge.configure(args.agent_id, PlayingMode.TRAINING)}")
        print(f"speed -> {bridge.set_speed(args.speed)}")
        if not args.no_prepare:
            print(f"ready -> {prepare_for_play(bridge, timeout=args.prepare_timeout, poll_delay=0.5).name}")
        actions = generate_diverse_drag_release_actions(count=args.count)
        manifest = collect_rollouts(
            bridge,
            args.output_dir,
            actions,
            target_fps=args.fps,
            duration_seconds=args.duration,
            frame_height=args.frame_height,
            fast=not args.safe,
        )
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "rollout_count": manifest["rollout_count"]}, indent=2))
    finally:
        bridge.disconnect()
        stop_owned_engine(engine_process)


if __name__ == "__main__":
    main()
# python scripts/collect_rollouts.py --output-dir data/collect_rollouts_debug --count 1 --fps 1 --duration 0.1 --connect-timeout 20 --prepare-timeout 20
