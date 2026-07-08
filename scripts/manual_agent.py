#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.webui.bridge import GameState, PlayingMode, ScienceBirdsBridge  # noqa: E402


HELP = """
Commands:
  help                         Show this help
  state                        Print game state
  score                        Print current score
  level                        Print current level index
  count                        Print number of configured levels
  prepare                      Drive menus/new-set state until PLAYING
  next                         Load next available level
  zoom out | zoom in           Send zoom command
  speed N                      Set simulation speed, 1-50 is typical
  drag X Y                     Start a drag at bottom-left game coordinates
  hold [MS]                    Hold the current drag before release
  release X Y [tap] [fast|safe]
                               Release the current drag at bottom-left game coordinates
  shoot X Y [tap] [fast|safe]  Shoot using bottom-left game coordinates
  rawshoot X Y [tap] [fast|safe]
                               Shoot using raw engine/socket coordinates
  frame [PATH]                 Save one screenshot as a binary PPM file
  rollout DIR [fps] [seconds]  Capture timestamped pixel frames into DIR
  diverse [N]                  Print diverse drag-release action dictionaries
  quit                         Disconnect and exit

Notes:
  - This agent only polls screenshots when you run `frame` or `rollout`.
  - `shoot` treats X/Y as game coordinates with origin at bottom-left.
  - `drag`, `hold`, and `release` model the agent action space; the TCP protocol still sends the final release shot.
  - Keep this prompt open while trying the native Unity window with your mouse.
""".strip()


def connect_with_retry(host: str, port: int, timeout: float, deadline_seconds: float) -> ScienceBirdsBridge:
    deadline = time.time() + deadline_seconds
    last_error: OSError | None = None
    while time.time() < deadline:
        bridge = ScienceBirdsBridge(host, port, timeout=timeout)
        try:
            bridge.connect()
            return bridge
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Could not connect to Science Birds at {host}:{port}: {last_error}")


def start_engine(game_dir: Path, headless: bool, *, agent_port: int | None = None, game_port: int | None = None) -> subprocess.Popen:
    jar_path = game_dir / "game_playing_interface.jar"
    if not jar_path.is_file():
        raise FileNotFoundError(f"Missing {jar_path}")
    command = ["java", "-jar", "./game_playing_interface.jar"]
    if headless:
        command.append("--headless")
    if agent_port is not None:
        command.extend(["--agent-port", str(agent_port)])
    if game_port is not None:
        command.extend(["--game-start-port", str(game_port)])
    command.append("--dev")
    log_path = Path(f"/tmp/novphy_game_engine_{int(time.time() * 1000)}.log")
    log_file = log_path.open("ab")
    process = subprocess.Popen(command, cwd=game_dir, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    process.novphy_log_file = log_file
    process.novphy_process_group = True
    return process


def stop_started_engine(engine_process: subprocess.Popen | None) -> None:
    if engine_process is None or engine_process.poll() is not None:
        return
    pgid = None
    if getattr(engine_process, "novphy_process_group", False):
        try:
            pgid = os.getpgid(engine_process.pid)
        except (OSError, ProcessLookupError):
            pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        engine_process.terminate()
    try:
        engine_process.wait(timeout=5)
    except (subprocess.TimeoutExpired, TimeoutError):
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            engine_process.kill()
        engine_process.wait(timeout=5)


def prepare_for_play(
    bridge: ScienceBirdsBridge,
    timeout: float,
    poll_delay: float,
    *,
    transition_retry_seconds: float = 5.0,
    new_set_ready_attempts: int = 3,
    clock=time.time,
    sleeper=time.sleep,
) -> GameState:
    deadline = clock() + timeout
    last_transition_state: GameState | None = None
    last_transition_time: float | None = None
    transition_attempt_count = 0
    transition_retry_seconds = max(0.0, float(transition_retry_seconds))
    new_set_ready_attempts = max(1, int(new_set_ready_attempts))
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
        GameState.WON,
        GameState.LOST,
        GameState.EVALUATION_TERMINATED,
    }

    while clock() < deadline:
        now = clock()
        state = bridge.get_game_state()
        if state == GameState.PLAYING:
            bridge.fully_zoom_out()
            return state
        if state != last_transition_state:
            transition_attempt_count = 0
        if state in new_set_states:
            should_retry = (
                state != last_transition_state
                or last_transition_time is None
                or now - last_transition_time >= transition_retry_seconds
            )
            if should_retry:
                if transition_attempt_count >= new_set_ready_attempts:
                    print(f"State {state.name}: load_next_available_level")
                    bridge.get_novelty_info()
                    bridge.load_next_available_level()
                    transition_attempt_count = 0
                else:
                    print(f"State {state.name}: ready_for_new_set")
                    bridge.ready_for_new_set()
                    transition_attempt_count += 1
                last_transition_state = state
                last_transition_time = now
            else:
                print(f"State {state.name}: waiting")
        elif state in menu_states:
            should_retry = (
                state != last_transition_state
                or last_transition_time is None
                or now - last_transition_time >= transition_retry_seconds
            )
            if should_retry:
                print(f"State {state.name}: load_next_available_level")
                bridge.get_novelty_info()
                bridge.load_next_available_level()
                last_transition_state = state
                last_transition_time = now
                transition_attempt_count = 0
            else:
                print(f"State {state.name}: waiting")
        elif state == GameState.LOADING:
            last_transition_state = None
            last_transition_time = None
            transition_attempt_count = 0
            print("State LOADING: waiting")
        else:
            last_transition_state = None
            last_transition_time = None
            transition_attempt_count = 0
            print(f"State {state.name}: waiting")
        sleeper(poll_delay)
    raise TimeoutError("Science Birds did not reach PLAYING before timeout")


def screenshot_to_image(screenshot) -> Image.Image:
    return Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)


def image_is_uniform(image: Image.Image) -> bool:
    extrema = image.getextrema()
    return all(channel_min == channel_max for channel_min, channel_max in extrema)


def feature_color(label: str) -> tuple[int, int, int]:
    lowered = label.lower()
    if lowered == "slingshot":
        return (139, 69, 19)
    if lowered == "ground":
        return (90, 90, 90)
    if "pig" in lowered:
        return (60, 180, 75)
    if "bird" in lowered:
        return (220, 60, 60)
    if "wood" in lowered:
        return (160, 110, 60)
    if "stone" in lowered:
        return (130, 130, 130)
    if "ice" in lowered:
        return (120, 200, 255)
    if "tnt" in lowered:
        return (220, 40, 40)
    if "platform" in lowered:
        return (100, 100, 100)
    return (235, 235, 235)


def render_symbolic_frame(ground_truth, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    features = []
    if isinstance(ground_truth, list) and ground_truth:
        first = ground_truth[0]
        if isinstance(first, dict) and isinstance(first.get("features"), list):
            features = first["features"]

    ground_y = None
    polygons = []
    for feature in features:
        properties = feature.get("properties") or {}
        label = str(properties.get("label") or "")
        geometry = feature.get("geometry") or {}
        if label == "Ground":
            ground_y = properties.get("yindex")
            continue
        if geometry.get("type") != "Polygon":
            continue
        coordinates = geometry.get("coordinates") or []
        if not coordinates:
            continue
        polygon = coordinates[0]
        if len(polygon) < 3:
            continue
        polygons.append((label, polygon))

    if isinstance(ground_y, int | float):
        y = int(ground_y)
        draw.rectangle([(0, y), (width, height)], fill=(70, 70, 74))

    for label, polygon in polygons:
        fill = feature_color(label)
        outline = tuple(max(0, channel - 30) for channel in fill)
        points = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(points, fill=fill, outline=outline)

    return image


def save_frame(bridge: ScienceBirdsBridge, path: Path) -> None:
    screenshot = bridge.screenshot()
    image = screenshot_to_image(screenshot)
    if image_is_uniform(image):
        raise RuntimeError(
            "uniform Science Birds screenshot; refusing to save symbolic fallback as a rollout frame"
        )
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        image.save(path, format="PPM")
    else:
        image.save(path, format="PNG")
    print(f"Saved {path} ({screenshot.width}x{screenshot.height}, source=screenshot)")


def _frame_stats(path: Path, image: Image.Image, t: float) -> dict:
    return {
        "path": str(path),
        "t": round(float(t), 6),
        "width": image.width,
        "height": image.height,
        "unique_colors": len(image.getcolors(maxcolors=image.width * image.height + 1) or []),
        "uniform": image_is_uniform(image),
    }


def capture_pixel_rollout(
    bridge: ScienceBirdsBridge,
    output_dir: Path,
    *,
    target_fps: float = 30.0,
    duration_seconds: float = 5.0,
    max_frames: int | None = None,
    action: dict | None = None,
    pre_shot_image=None,
    pre_shot_sample: dict | None = None,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> dict:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / float(target_fps)
    total_frames = max_frames if max_frames is not None else max(1, int(math.ceil(duration_seconds * target_fps)))
    started_at = clock()
    frames = []
    state_samples = []

    for frame_index in range(total_frames):
        target_time = started_at + frame_index * interval
        now = clock()
        if now < target_time:
            sleeper(target_time - now)
        elapsed = clock() - started_at
        if elapsed > duration_seconds and frame_index > 0:
            break

        screenshot = bridge.screenshot()
        image = screenshot_to_image(screenshot)
        if image_is_uniform(image):
            raise RuntimeError("uniform Science Birds screenshot; refusing to save rollout frame")
        frame_path = frames_dir / f"frame_{frame_index:06d}.png"
        image.save(frame_path, format="PNG")
        frames.append(_frame_stats(frame_path, image, elapsed))

        sample = {"index": frame_index, "t": round(float(elapsed), 6)}
        try:
            sample["state"] = bridge.get_game_state().name
        except Exception as exc:
            sample["state_error"] = str(exc)
        try:
            sample["score"] = bridge.get_current_score()
        except Exception as exc:
            sample["score_error"] = str(exc)
        state_samples.append(sample)

    metadata = {
        "capture_source": "manual-agent-pixel-screenshot",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "frame_count": len(frames),
        "frames_dir": str(frames_dir),
        "frames": frames,
        "state_samples": state_samples,
    }
    if pre_shot_sample is not None:
        metadata["pre_shot_sample"] = pre_shot_sample
    if action is not None:
        metadata["action"] = action
        metadata["action_signature"] = action_diversity_signature(action)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def generate_diverse_drag_release_actions(
    *,
    drag_start: tuple[int, int] = (300, 220),
    count: int = 16,
    strengths: tuple[int, ...] = (80, 105, 130, 155),
    angles_degrees: tuple[int, ...] = (5, 20, 35, 50, 65, 80),
    tap_times: tuple[int, ...] = (0, 45, 70, 90),
    hold_times: tuple[int, ...] = (1000, 1400),
) -> list[dict]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if not strengths or not angles_degrees or not tap_times or not hold_times:
        raise ValueError("strengths, angles_degrees, tap_times, and hold_times must be non-empty")

    actions = []
    for index in range(count):
        angle = angles_degrees[index % len(angles_degrees)]
        strength = strengths[(index // len(angles_degrees)) % len(strengths)]
        tap_time = tap_times[index % len(tap_times)]
        hold_time = hold_times[(index // len(tap_times)) % len(hold_times)]
        radians = math.radians(angle)
        dx = -int(round(math.cos(radians) * strength))
        dy = int(round(math.sin(radians) * strength))
        action = {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_start": [int(drag_start[0]), int(drag_start[1])],
            "drag_release": [dx, dy],
            "tapTime": int(tap_time),
        }
        action["holdTime"] = int(hold_time)
        actions.append(action)
    return deduplicate_similar_actions(actions)


def action_diversity_signature(
    action: dict,
    *,
    strength_bin: int = 10,
    angle_bin_degrees: int = 10,
) -> dict:
    release = action.get("drag_release", action.get("release"))
    if not isinstance(release, list | tuple) or len(release) < 2:
        raise ValueError("drag_release or release must contain x and y values")
    dx = float(release[0])
    dy = float(release[1])
    strength = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, -dx if dx <= 0 else dx))
    return {
        "strength_bin": int(round(strength / strength_bin) * strength_bin),
        "angle_bin": int(round(angle / angle_bin_degrees) * angle_bin_degrees),
        "tapTime": int(action.get("tapTime", action.get("tap_time", 0))),
        "holdTime": int(action.get("holdTime", action.get("releaseTime", action.get("release_time", 0)))),
    }


def deduplicate_similar_actions(
    actions: list[dict],
    *,
    strength_bin: int = 10,
    angle_bin_degrees: int = 10,
) -> list[dict]:
    deduplicated = []
    seen = set()
    for action in actions:
        signature = action_diversity_signature(
            action,
            strength_bin=strength_bin,
            angle_bin_degrees=angle_bin_degrees,
        )
        key = tuple(sorted(signature.items()))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(action)
    return deduplicated


def parse_shot(parts: list[str], *, raw: bool, frame_height: int) -> tuple[int, int, int, bool]:
    if len(parts) < 3:
        raise ValueError("usage: shoot X Y [tap] [fast|safe]")
    x = int(parts[1])
    y = int(parts[2])
    tap_time = 0
    fast = True
    for token in parts[3:]:
        lowered = token.lower()
        if lowered == "fast":
            fast = True
        elif lowered == "safe":
            fast = False
        else:
            tap_time = int(token)
    bridge_y = y if raw else max(0, frame_height - 1 - y)
    return x, bridge_y, tap_time, fast


def parse_game_point(parts: list[str], usage: str) -> tuple[int, int]:
    if len(parts) < 3:
        raise ValueError(usage)
    return int(parts[1]), int(parts[2])


def repl(bridge: ScienceBirdsBridge, frame_height: int) -> None:
    print(HELP)
    drag_start: tuple[int, int] | None = None
    hold_time = 0
    while True:
        try:
            line = input("sciencebirds> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()
        try:
            if command in {"quit", "exit", "q"}:
                return
            if command in {"help", "h", "?"}:
                print(HELP)
            elif command == "state":
                print(bridge.get_game_state().name)
            elif command == "score":
                print(bridge.get_current_score())
            elif command == "level":
                print(bridge.get_current_level())
            elif command == "count":
                print(bridge.get_number_of_levels())
            elif command == "prepare":
                print(f"Reached {prepare_for_play(bridge, timeout=60, poll_delay=0.5).name}")
            elif command == "next":
                print(bridge.load_next_available_level())
            elif command == "zoom" and len(parts) == 2 and parts[1].lower() == "out":
                print(bridge.fully_zoom_out())
            elif command == "zoom" and len(parts) == 2 and parts[1].lower() == "in":
                print(bridge.fully_zoom_in())
            elif command == "speed" and len(parts) == 2:
                print(bridge.set_speed(int(parts[1])))
            elif command == "drag":
                drag_start = parse_game_point(parts, "usage: drag X Y")
                hold_time = 0
                print({"action_type": "drag", "drag_start": list(drag_start)})
            elif command == "hold":
                if drag_start is None:
                    raise ValueError("drag must be started before hold")
                hold_time = int(parts[1]) if len(parts) > 1 else 0
                print({"action_type": "hold", "drag_start": list(drag_start), "holdTime": hold_time})
            elif command == "release":
                if drag_start is None:
                    raise ValueError("drag must be started before release")
                x, y, tap_time, fast = parse_shot(parts, raw=False, frame_height=frame_height)
                dx = int(parts[1]) - drag_start[0]
                dy = drag_start[1] - int(parts[2])
                action = {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_start": list(drag_start),
                    "drag_release": [dx, dy],
                    "holdTime": hold_time,
                    "tapTime": tap_time,
                }
                print(action)
                print(bridge.shoot(x, y, tap_time=tap_time, fast=fast, release_time=hold_time))
                drag_start = None
                hold_time = 0
            elif command in {"shoot", "rawshoot"}:
                x, y, tap_time, fast = parse_shot(parts, raw=command == "rawshoot", frame_height=frame_height)
                print(bridge.shoot(x, y, tap_time=tap_time, fast=fast))
            elif command == "frame":
                save_frame(bridge, Path(parts[1] if len(parts) > 1 else "sciencebirds-frame.ppm"))
            elif command == "rollout":
                if len(parts) < 2:
                    raise ValueError("usage: rollout DIR [fps] [seconds]")
                target_fps = float(parts[2]) if len(parts) > 2 else 30.0
                duration_seconds = float(parts[3]) if len(parts) > 3 else 5.0
                metadata = capture_pixel_rollout(
                    bridge,
                    Path(parts[1]),
                    target_fps=target_fps,
                    duration_seconds=duration_seconds,
                )
                print(json.dumps({"frame_count": metadata["frame_count"], "metadata": str(Path(parts[1]) / "metadata.json")}))
            elif command == "diverse":
                count = int(parts[1]) if len(parts) > 1 else 16
                print(json.dumps(generate_diverse_drag_release_actions(count=count), indent=2))
            else:
                print("Unknown command. Type `help`.")
        except Exception as exc:
            print(f"Error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual stdin controller for a running Science Birds Java engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2004)
    parser.add_argument("--agent-id", type=int, default=28888)
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--frame-height", type=int, default=480, help="Height used to convert bottom-left shot Y to socket Y")
    parser.add_argument("--connect-timeout", type=float, default=30)
    parser.add_argument("--read-timeout", type=float, default=300)
    parser.add_argument("--prepare-timeout", type=float, default=60)
    parser.add_argument("--no-prepare", action="store_true", help="Connect/configure only; do not auto-load a level")
    parser.add_argument("--start-engine", action="store_true", help="Start game_playing_interface.jar before connecting")
    parser.add_argument("--game-headless", action="store_true", help="Only used with --start-engine")
    parser.add_argument("--print-diverse-actions", action="store_true", help="Print deterministic diverse drag-release actions and exit")
    parser.add_argument("--diverse-action-count", type=int, default=16, help="Number of actions for --print-diverse-actions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.print_diverse_actions:
        print(json.dumps(generate_diverse_drag_release_actions(count=args.diverse_action_count), indent=2))
        return

    engine_process: subprocess.Popen | None = None
    if args.start_engine:
        engine_process = start_engine(ROOT / "sciencebirdsgames" / "Linux", args.game_headless)
        print(f"Started engine pid={engine_process.pid}")

    try:
        bridge = connect_with_retry(args.host, args.port, timeout=args.read_timeout, deadline_seconds=args.connect_timeout)
    except RuntimeError:
        stop_started_engine(engine_process)
        raise
    try:
        print("Connected")
        print(f"configure -> {bridge.configure(args.agent_id, PlayingMode.TRAINING)}")
        print(f"speed -> {bridge.set_speed(args.speed)}")
        if not args.no_prepare:
            state = prepare_for_play(bridge, timeout=args.prepare_timeout, poll_delay=0.5)
            print(f"Ready: {state.name}")
        repl(bridge, frame_height=args.frame_height)
    finally:
        try:
            bridge.disconnect()
        finally:
            stop_started_engine(engine_process)


if __name__ == "__main__":
    main()
