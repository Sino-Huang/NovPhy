#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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


DEFAULT_DESKTOP_GAME_CROP = (32, 64, 672, 544)
DEFAULT_HUMAN_HOLD_MS = 600


def _image_is_uniform(image) -> bool:
    extrema = image.getextrema()
    return all(channel_min == channel_max for channel_min, channel_max in extrema)


def _default_desktop_crop_for(image) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = DEFAULT_DESKTOP_GAME_CROP
    if image.width >= right and image.height >= bottom:
        return DEFAULT_DESKTOP_GAME_CROP
    return None


def _crop_desktop_image(image, crop: tuple[int, int, int, int] | None):
    if crop is None:
        return image
    return image.crop(crop)


def write_rollout_video(
    frames_dir: Path,
    output_path: Path,
    *,
    fps: float,
    runner=subprocess.run,
) -> dict[str, str]:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(float(fps)).rstrip("0").rstrip("."),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    runner(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"video_path": str(output_path)}


def _format_xy(values) -> str:
    if isinstance(values, list | tuple) and len(values) >= 2:
        return f"({int(values[0])},{int(values[1])})"
    return "(?,?)"


def _normal_action_release_time(action: dict) -> int:
    release_time = _action_int(action.get("holdTime", action.get("releaseTime", action.get("release_time", DEFAULT_HUMAN_HOLD_MS))))
    return release_time if release_time > 0 else DEFAULT_HUMAN_HOLD_MS


def normalize_action_to_game(action: dict) -> dict[str, Any]:
    release = action.get("drag_release", action.get("release"))
    if not isinstance(release, list | tuple) or len(release) < 2:
        raise ValueError("drag_release or release must contain x and y values")

    coordinate_frame = action.get("coordinate_frame", "slingshot_relative")
    drag_start = action.get("drag_start")
    if coordinate_frame == "slingshot_relative":
        if not isinstance(drag_start, list | tuple) or len(drag_start) < 2:
            raise ValueError("drag_start is required for slingshot_relative actions")
        start_x = _action_int(drag_start[0])
        start_y = _action_int(drag_start[1])
        dx = _action_int(release[0])
        dy = _action_int(release[1])
        shot_x = start_x + dx
        shot_y = start_y - dy
    elif coordinate_frame == "absolute":
        shot_x = _action_int(release[0])
        shot_y = _action_int(release[1])
        if drag_start is None:
            start_x = shot_x
            start_y = shot_y
        elif isinstance(drag_start, list | tuple) and len(drag_start) >= 2:
            start_x = _action_int(drag_start[0])
            start_y = _action_int(drag_start[1])
        else:
            raise ValueError("drag_start must contain x and y values")
        dx = shot_x - start_x
        dy = start_y - shot_y
    else:
        raise ValueError("coordinate_frame must be slingshot_relative or absolute")

    return {
        "action_type": action.get("action_type", "drag_hold_release"),
        "coordinate_frame": "slingshot_relative",
        "drag_start": [start_x, start_y],
        "drag_release": [dx, dy],
        "gameX": shot_x,
        "gameY": shot_y,
        "tapTime": _action_int(action.get("tapTime", action.get("tap_time", 0))),
        "releaseTime": _normal_action_release_time(action),
    }


def format_action_overlay_text(action: dict, shot: dict) -> str:
    normalized = normalize_action_to_game(action)
    coordinate_frame = action.get("coordinate_frame", "slingshot_relative")
    release_display = action.get("release") if coordinate_frame == "absolute" else normalized["drag_release"]
    launch_x = int(normalized["drag_start"][0]) - int(normalized["drag_release"][0])
    launch_y = int(normalized["drag_start"][1]) + int(normalized["drag_release"][1])
    return " ".join(
        [
            f"drag_mode={coordinate_frame}",
            f"release_mode={coordinate_frame}",
            f"drag_xy={_format_xy(normalized['drag_start'])}",
            f"pull_release_xy={_format_xy(release_display)}",
            f"launch_xy=({launch_x},{launch_y})",
            f"game_xy=({int(normalized['gameX'])},{int(normalized['gameY'])})",
            f"socket_xy=({int(shot['x'])},{int(shot['y'])})",
            f"tapTime={int(shot.get('tapTime', 0))}",
            f"releaseTime={int(shot.get('releaseTime', 0))}",
        ]
    )


def _action_guide_points(action: dict, shot: dict, image_height: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    normalized = normalize_action_to_game(action)
    start_x, start_y = normalized["drag_start"]
    return (int(start_x), image_height - 1 - int(start_y)), (int(normalized["gameX"]), image_height - 1 - int(normalized["gameY"]))


def _launch_guide_points(action: dict, image_height: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    normalized = normalize_action_to_game(action)
    start_x, start_y = normalized["drag_start"]
    dx, dy = normalized["drag_release"]
    launch_x = int(start_x) - int(dx)
    launch_y = int(start_y) + int(dy)
    return (int(start_x), image_height - 1 - int(start_y)), (launch_x, image_height - 1 - launch_y)


def _draw_overlay(image, text: str, action: dict, shot: dict):
    from PIL import ImageDraw, ImageFont

    output = image.copy()
    draw = ImageDraw.Draw(output)
    banner_height = min(max(24, output.height // 14), max(24, output.height))
    draw.rectangle((0, 0, output.width, banner_height), fill=(0, 0, 0))
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((6, 5), text, fill=(255, 255, 255), font=font)
    guide_points = _action_guide_points(action, shot, output.height)
    if guide_points is not None:
        start, end = guide_points
        draw.line((start[0], start[1], end[0], end[1]), fill=(0, 0, 0), width=7)
        draw.line((start[0], start[1], end[0], end[1]), fill=(255, 230, 0), width=5)
        radius = 5
        draw.ellipse((start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius), outline=(0, 255, 255), width=2)
        draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), outline=(255, 80, 80), width=2)
    launch_points = _launch_guide_points(action, output.height)
    if launch_points is not None:
        launch_start, launch_end = launch_points
        draw.line((launch_start[0], launch_start[1], launch_end[0], launch_end[1]), fill=(0, 0, 0), width=8)
        draw.line((launch_start[0], launch_start[1], launch_end[0], launch_end[1]), fill=(80, 255, 120), width=5)
        radius = 6
        draw.ellipse(
            (launch_end[0] - radius, launch_end[1] - radius, launch_end[0] + radius, launch_end[1] + radius),
            outline=(80, 255, 120),
            width=3,
        )
    return output


def prepare_rollout_video_frames(
    shot_dir: Path,
    frames_dir: Path,
    *,
    action: dict,
    shot: dict,
    fps: float,
    pre_shot_path: Path | None = None,
    lead_in_seconds: float = 0.75,
) -> dict:
    from PIL import Image

    video_frames_dir = shot_dir / "video_frames"
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in video_frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    overlay_text = format_action_overlay_text(action, shot)
    video_index = 0
    pre_action_frame_count = 0
    if pre_shot_path is not None and pre_shot_path.is_file():
        pre_shot_image = Image.open(pre_shot_path).convert("RGB")
        pre_action_frame_count = max(1, int(round(float(fps) * lead_in_seconds)))
        for _ in range(pre_action_frame_count):
            _draw_overlay(pre_shot_image, overlay_text, action, shot).save(video_frames_dir / f"frame_{video_index:06d}.png", format="PNG")
            video_index += 1

    for frame_path in sorted(frames_dir.glob("frame_*.png")):
        image = Image.open(frame_path).convert("RGB")
        _draw_overlay(image, overlay_text, action, shot).save(video_frames_dir / f"frame_{video_index:06d}.png", format="PNG")
        video_index += 1

    return {
        "video_frames_dir": str(video_frames_dir),
        "video_input_pattern": str(video_frames_dir / "frame_%06d.png"),
        "pre_action_frame_count": pre_action_frame_count,
        "video_frame_count": video_index,
        "video_overlay": {
            "position": "top",
            "text": overlay_text,
            "action_guide": "cyan=start yellow=pull red=release green=launch",
        },
    }


def write_action_logs(output_dir: Path, rollouts: list[dict]) -> dict[str, str]:
    trials = []
    for rollout in rollouts:
        trial = {
            "shot_name": rollout["name"],
            "action": rollout["action"],
            "shot": rollout["shot"],
            "shoot_response": rollout["shoot_response"],
            "frame_count": rollout["frame_count"],
            "metadata_path": rollout["metadata_path"],
        }
        for key in ("pre_shot_path", "video_path", "slingshot_reference"):
            if rollout.get(key) is not None:
                trial[key] = rollout[key]
        trials.append(trial)

    action_log_path = output_dir / "action_log.json"
    action_log_jsonl_path = output_dir / "action_log.jsonl"
    payload = {"episode_dir": str(output_dir), "trial_count": len(trials), "trials": trials}
    action_log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    action_log_jsonl_path.write_text("".join(json.dumps(trial) + "\n" for trial in trials), encoding="utf-8")
    return {"action_log_path": str(action_log_path), "action_log_jsonl_path": str(action_log_jsonl_path)}


def current_slingshot_reference(bridge, frame_height: int) -> dict[str, int] | None:
    try:
        symbolic_state = bridge.get_symbolic_state_without_screenshot()
    except Exception:
        return None
    return slingshot_reference_point_from_symbolic_state(symbolic_state, frame_height)


def anchor_actions_to_current_slingshot(bridge, actions: list[dict], frame_height: int) -> list[dict]:
    if not any(action.get("coordinate_frame", "slingshot_relative") == "slingshot_relative" for action in actions):
        return actions
    slingshot_reference = current_slingshot_reference(bridge, frame_height)
    if slingshot_reference is None:
        return actions
    return [
        anchor_action_to_slingshot_reference(action, slingshot_reference)
        if action.get("coordinate_frame", "slingshot_relative") == "slingshot_relative"
        else action
        for action in actions
    ]


def _frame_stats(path: Path, image, t: float) -> dict:
    return {
        "path": str(path),
        "t": round(float(t), 6),
        "width": image.width,
        "height": image.height,
        "unique_colors": len(image.getcolors(maxcolors=image.width * image.height + 1) or []),
        "uniform": _image_is_uniform(image),
    }


def _action_int(value) -> int:
    return int(value)


def _point_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, dict):
        x = point.get("x")
        y = point.get("y")
        if isinstance(x, int | float) and isinstance(y, int | float):
            return float(x), float(y)
        return None
    if isinstance(point, list | tuple) and len(point) >= 2:
        x, y = point[0], point[1]
        if isinstance(x, int | float) and isinstance(y, int | float):
            return float(x), float(y)
    return None


def slingshot_reference_point_from_symbolic_state(symbolic_state: Any, frame_height: int) -> dict[str, int] | None:
    if not isinstance(symbolic_state, list):
        return None

    for item in symbolic_state:
        vertices = None
        if not isinstance(item, dict):
            continue
        if item.get("type") == "Slingshot":
            vertices = item.get("vertices")
        else:
            features = item.get("features")
            if isinstance(features, list):
                for feature in features:
                    if not isinstance(feature, dict):
                        continue
                    properties = feature.get("properties")
                    geometry = feature.get("geometry")
                    if not isinstance(properties, dict) or not isinstance(geometry, dict):
                        continue
                    if properties.get("label") != "Slingshot" or geometry.get("type") != "Polygon":
                        continue
                    coordinates = geometry.get("coordinates")
                    if isinstance(coordinates, list) and coordinates:
                        polygon = coordinates[0]
                        if isinstance(polygon, list):
                            vertices = polygon
                            break
        if not vertices:
            continue

        points = [_point_xy(vertex) for vertex in vertices]
        points = [point for point in points if point is not None]
        if not points:
            continue

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        sling_width = max_x - min_x
        if sling_width <= 0:
            return None

        canvas_x = int(min_x + 0.45 * sling_width)
        canvas_y = int(min_y + 0.35 * sling_width)
        return {
            "gameX": canvas_x,
            "gameY": max(0, frame_height - 1 - canvas_y),
            "canvasX": canvas_x,
            "canvasY": canvas_y,
        }

    return None


def anchor_action_to_slingshot_reference(action: dict, slingshot_reference: dict[str, int]) -> dict:
    anchored = dict(action)
    if anchored.get("coordinate_frame", "slingshot_relative") == "slingshot_relative":
        anchored["drag_start"] = [int(slingshot_reference["gameX"]), int(slingshot_reference["gameY"])]
        anchored["slingshot_reference"] = dict(slingshot_reference)
    return anchored


def _image_delta_stats(previous_image, current_image) -> dict[str, float | int | list[int] | None]:
    from PIL import ImageChops, ImageStat

    difference = ImageChops.difference(previous_image, current_image)
    raw_pixels = difference.tobytes()
    changed_pixel_count = sum(1 for offset in range(0, len(raw_pixels), 3) if raw_pixels[offset : offset + 3] != b"\x00\x00\x00")
    stat = ImageStat.Stat(difference)
    mean_absolute_channel_delta = sum(stat.mean) / len(stat.mean)
    bbox = difference.getbbox()
    return {
        "changed_pixel_count": changed_pixel_count,
        "mean_absolute_channel_delta": round(float(mean_absolute_channel_delta), 6),
        "bbox": list(bbox) if bbox is not None else None,
    }


def action_to_shot(action: dict, *, frame_height: int) -> dict:
    normalized = normalize_action_to_game(action)
    shot_x = int(normalized["gameX"])
    shot_y = int(normalized["gameY"])

    return {
        "x": shot_x,
        "y": max(0, frame_height - 1 - shot_y),
        "gameX": shot_x,
        "gameY": shot_y,
        "tapTime": int(normalized["tapTime"]),
        "releaseTime": int(normalized["releaseTime"]),
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


def capture_desktop_rollout(
    bridge,
    output_dir: Path,
    *,
    target_fps: float,
    duration_seconds: float,
    max_frames: int | None = None,
    action: dict | None = None,
    pre_shot_image=None,
    pre_shot_sample: dict | None = None,
    desktop_crop: tuple[int, int, int, int] | None = DEFAULT_DESKTOP_GAME_CROP,
    grabber=None,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> dict:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    if grabber is None:
        from PIL import ImageGrab

        grabber = ImageGrab

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pre_shot_path = None
    if pre_shot_image is not None:
        desktop_crop = _default_desktop_crop_for(pre_shot_image) if desktop_crop == DEFAULT_DESKTOP_GAME_CROP else desktop_crop
        pre_shot_image = _crop_desktop_image(pre_shot_image, desktop_crop)
        if _image_is_uniform(pre_shot_image):
            raise RuntimeError("uniform desktop pre-shot screenshot; refusing to save rollout baseline")
        pre_shot_path = output_dir / "pre_shot.png"
        pre_shot_image.save(pre_shot_path, format="PNG")

    interval = 1.0 / float(target_fps)
    total_frames = max_frames if max_frames is not None else max(1, int(math.ceil(duration_seconds * target_fps)))
    started_at = clock()
    frames = []
    state_samples = []
    previous_image = None
    max_frame_delta = 0
    max_mean_absolute_channel_delta = 0.0
    max_pre_shot_delta = 0
    max_pre_shot_delta_bbox = None

    for frame_index in range(total_frames):
        target_time = started_at + frame_index * interval
        now = clock()
        if now < target_time:
            sleeper(target_time - now)
        elapsed = clock() - started_at
        if elapsed > duration_seconds and frame_index > 0:
            break

        image = grabber.grab()
        desktop_crop = _default_desktop_crop_for(image) if desktop_crop == DEFAULT_DESKTOP_GAME_CROP else desktop_crop
        image = _crop_desktop_image(image, desktop_crop)
        if _image_is_uniform(image):
            raise RuntimeError("uniform desktop screenshot; refusing to save rollout frame")
        frame_path = frames_dir / f"frame_{frame_index:06d}.png"
        image.save(frame_path, format="PNG")
        frame = _frame_stats(frame_path, image, elapsed)
        if previous_image is None:
            frame["frame_delta"] = None
        else:
            frame_delta = _image_delta_stats(previous_image, image)
            frame["frame_delta"] = frame_delta
            max_frame_delta = max(max_frame_delta, frame_delta["changed_pixel_count"])
            max_mean_absolute_channel_delta = max(max_mean_absolute_channel_delta, frame_delta["mean_absolute_channel_delta"])
        if pre_shot_image is not None:
            pre_shot_delta = _image_delta_stats(pre_shot_image, image)
            frame["pre_shot_delta"] = pre_shot_delta
            if pre_shot_delta["changed_pixel_count"] > max_pre_shot_delta:
                max_pre_shot_delta = pre_shot_delta["changed_pixel_count"]
                max_pre_shot_delta_bbox = pre_shot_delta["bbox"]
        frames.append(frame)
        previous_image = image

    if frames:
        sample = {"index": len(frames) - 1, "t": frames[-1]["t"], "phase": "post_capture"}
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
        "capture_source": "desktop-imagegrab",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "frame_count": len(frames),
        "frames_dir": str(frames_dir),
        "frames": frames,
        "state_samples": state_samples,
        "max_frame_delta": max_frame_delta,
        "max_mean_absolute_channel_delta": round(max_mean_absolute_channel_delta, 6),
    }
    if pre_shot_path is not None:
        metadata["pre_shot_path"] = str(pre_shot_path)
        metadata["max_pre_shot_delta"] = max_pre_shot_delta
        metadata["max_pre_shot_delta_bbox"] = max_pre_shot_delta_bbox
    if desktop_crop is not None:
        metadata["desktop_crop"] = list(desktop_crop)
    if pre_shot_sample is not None:
        metadata["pre_shot_sample"] = pre_shot_sample
    if action is not None:
        metadata["action"] = action

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def select_level_in_display(
    level: int,
    *,
    play_xy: tuple[int, int] = (512, 390),
    input_xy: tuple[int, int] = (495, 343),
    confirm_xy: tuple[int, int] = (492, 465),
    runner=subprocess.run,
    sleeper=time.sleep,
) -> None:
    def click(point: tuple[int, int]) -> None:
        runner(["xdotool", "mousemove", str(point[0]), str(point[1]), "click", "1"], check=True)

    click(play_xy)
    sleeper(1)
    click(input_xy)
    sleeper(0.2)
    runner(["xdotool", "type", str(int(level))], check=True)
    sleeper(0.2)
    click(confirm_xy)


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
    reset_rollout=None,
    capture_rollout=capture_pixel_rollout,
    pre_shot_grabber=None,
    start_index: int = 1,
    write_manifest: bool = True,
    video_runner=subprocess.run,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = anchor_actions_to_current_slingshot(bridge, actions, frame_height)
    rollouts = []
    for index, action in enumerate(actions, start=start_index):
        if reset_rollout is not None:
            reset_rollout(index, action)
        shot = action_to_shot(action, frame_height=frame_height)
        shot_dir = output_dir / f"shot_{index:03d}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        pre_shot_image = None
        pre_shot_sample = None
        if pre_shot_grabber is not None:
            pre_shot_image = pre_shot_grabber()
            if _image_is_uniform(pre_shot_image):
                raise RuntimeError("uniform desktop pre-shot screenshot; refusing to save rollout baseline")
            pre_shot_path = shot_dir / "pre_shot.png"
            pre_shot_image.save(pre_shot_path, format="PNG")
            pre_shot_sample = {
                "state": bridge.get_game_state().name,
                "score": bridge.get_current_score(),
            }
        response = bridge.shoot(
            shot["x"],
            shot["y"],
            tap_time=shot["tapTime"],
            fast=fast,
            release_time=shot["releaseTime"],
        )
        capture_kwargs = {
            "target_fps": target_fps,
            "duration_seconds": duration_seconds,
            "max_frames": max_frames,
            "action": action,
            "clock": clock,
            "sleeper": sleeper,
        }
        if pre_shot_image is not None:
            capture_kwargs["pre_shot_image"] = pre_shot_image
        if pre_shot_sample is not None:
            capture_kwargs["pre_shot_sample"] = pre_shot_sample
        metadata = capture_rollout(bridge, shot_dir, **capture_kwargs)
        if metadata.get("frames_dir"):
            pre_shot_path = Path(metadata["pre_shot_path"]) if metadata.get("pre_shot_path") else None
            video_frame_metadata = prepare_rollout_video_frames(
                shot_dir,
                Path(metadata["frames_dir"]),
                action=action,
                shot=shot,
                fps=target_fps,
                pre_shot_path=pre_shot_path,
            )
            metadata.update(video_frame_metadata)
            try:
                video = write_rollout_video(
                    Path(metadata["video_frames_dir"]),
                    shot_dir / "rollout.mp4",
                    fps=target_fps,
                    runner=video_runner,
                )
                metadata.update(video)
            except Exception as exc:
                metadata["video_error"] = str(exc)
        slingshot_reference = action.get("slingshot_reference")
        if slingshot_reference is not None:
            metadata["slingshot_reference"] = slingshot_reference
        (shot_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        rollouts.append(
            {
                "name": shot_dir.name,
                "action": action,
                "shot": shot,
                "shoot_response": response,
                "frame_count": metadata["frame_count"],
                "slingshot_reference": slingshot_reference,
                "metadata_path": str(shot_dir / "metadata.json"),
                **({"pre_shot_path": metadata["pre_shot_path"]} if "pre_shot_path" in metadata else {}),
                **({"video_path": metadata["video_path"]} if "video_path" in metadata else {}),
            }
        )

    manifest = {
        "capture_source": "scripts.collect_rollouts",
        "replay_mode": "same-episode-varied-trials",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }
    manifest.update(write_action_logs(output_dir, rollouts))
    if write_manifest:
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def collect_fresh_engine_rollouts(
    output_dir: Path,
    actions: list[dict],
    *,
    game_dir: Path,
    host: str,
    port: int,
    agent_id: int,
    speed: int,
    connect_timeout: float,
    read_timeout: float,
    prepare_timeout: float,
    frame_height: int,
    fast: bool,
    headless: bool,
    target_fps: float,
    duration_seconds: float,
    ui_level: int | None,
    ui_settle_seconds: float,
    engine_settle_seconds: float = 0.0,
    agent_settle_seconds: float = 0.0,
    start_engine_func=start_engine,
    connect_func=connect_with_retry,
    prepare_func=prepare_for_play,
    capture_rollout=capture_pixel_rollout,
    pre_shot_grabber=None,
    select_level_func=select_level_in_display,
    video_runner=subprocess.run,
    sleeper=time.sleep,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = []
    for index, action in enumerate(actions, start=1):
        engine_process = start_engine_func(game_dir, headless)
        print(f"Started engine pid={engine_process.pid} for rollout {index}")
        if engine_settle_seconds > 0:
            sleeper(engine_settle_seconds)
        bridge = None
        try:
            bridge = connect_func(host, port, timeout=read_timeout, deadline_seconds=connect_timeout)
            if agent_settle_seconds > 0:
                sleeper(agent_settle_seconds)
            print(f"configure -> {bridge.configure(agent_id, PlayingMode.TRAINING)}")
            print(f"speed -> {bridge.set_speed(speed)}")
            print(f"ready -> {prepare_func(bridge, timeout=prepare_timeout, poll_delay=0.5).name}")
            if ui_level is not None:
                if ui_settle_seconds > 0:
                    sleeper(ui_settle_seconds)
                select_level_func(ui_level)
                if ui_settle_seconds > 0:
                    sleeper(ui_settle_seconds)
            slingshot_reference = None
            anchored_action = action
            if action.get("coordinate_frame", "slingshot_relative") == "slingshot_relative":
                slingshot_reference = slingshot_reference_point_from_symbolic_state(
                    bridge.get_symbolic_state_without_screenshot(),
                    frame_height,
                )
                if slingshot_reference is None:
                    raise RuntimeError("could not resolve slingshot reference from symbolic state")
                anchored_action = anchor_action_to_slingshot_reference(action, slingshot_reference)
            partial = collect_rollouts(
                bridge,
                output_dir,
                [anchored_action],
                target_fps=target_fps,
                duration_seconds=duration_seconds,
                frame_height=frame_height,
                fast=fast,
                capture_rollout=capture_rollout,
                pre_shot_grabber=pre_shot_grabber,
                start_index=index,
                write_manifest=False,
                video_runner=video_runner,
            )
            partial["rollouts"][0]["slingshot_reference"] = slingshot_reference
            rollouts.extend(partial["rollouts"])
        finally:
            if bridge is not None:
                try:
                    bridge.disconnect()
                finally:
                    stop_owned_engine(engine_process)
            else:
                stop_owned_engine(engine_process)

    manifest = {
        "capture_source": getattr(capture_rollout, "__name__", "custom-capture"),
        "replay_mode": "fresh-engine-per-rollout",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }
    if ui_level is not None:
        manifest["ui_level"] = ui_level
    manifest.update(write_action_logs(output_dir, rollouts))
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
    parser.add_argument("--capture-source", choices=("protocol", "desktop"), default="protocol")
    parser.add_argument("--fresh-engine-per-rollout", action="store_true")
    parser.add_argument("--ui-level", type=int, help="Visible level number to enter with xdotool for each fresh rollout")
    parser.add_argument("--ui-settle-seconds", type=float, default=5.0)
    parser.add_argument("--engine-settle-seconds", type=float, default=20.0)
    parser.add_argument("--agent-settle-seconds", type=float, default=45.0)
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

    capture_rollout = capture_desktop_rollout if args.capture_source == "desktop" else capture_pixel_rollout
    pre_shot_grabber = None
    if args.capture_source == "desktop":
        from PIL import ImageGrab

        pre_shot_grabber = ImageGrab.grab
    actions = generate_diverse_drag_release_actions(count=args.count)

    if args.fresh_engine_per_rollout:
        manifest = collect_fresh_engine_rollouts(
            args.output_dir,
            actions,
            game_dir=ROOT / "sciencebirdsgames" / "Linux",
            host=args.host,
            port=args.port,
            agent_id=args.agent_id,
            speed=args.speed,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            prepare_timeout=args.prepare_timeout,
            frame_height=args.frame_height,
            fast=not args.safe,
            headless=args.game_headless,
            target_fps=args.fps,
            duration_seconds=args.duration,
            ui_level=args.ui_level,
            ui_settle_seconds=args.ui_settle_seconds,
            engine_settle_seconds=args.engine_settle_seconds,
            agent_settle_seconds=args.agent_settle_seconds,
            capture_rollout=capture_rollout,
            pre_shot_grabber=pre_shot_grabber,
        )
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "rollout_count": manifest["rollout_count"]}, indent=2))
        return

    if args.start_engine:
        engine_process = start_engine(ROOT / "sciencebirdsgames" / "Linux", args.game_headless)
        print(f"Started engine pid={engine_process.pid}")
        try:
            bridge = connect_with_retry(args.host, args.port, timeout=args.read_timeout, deadline_seconds=args.connect_timeout)
        except RuntimeError:
            stop_owned_engine(engine_process)
            raise
    else:
        bridge, engine_process = connect_or_start_engine(args)
    try:
        print(f"configure -> {bridge.configure(args.agent_id, PlayingMode.TRAINING)}")
        print(f"speed -> {bridge.set_speed(args.speed)}")
        if not args.no_prepare:
            print(f"ready -> {prepare_for_play(bridge, timeout=args.prepare_timeout, poll_delay=0.5).name}")
        manifest = collect_rollouts(
            bridge,
            args.output_dir,
            actions,
            target_fps=args.fps,
            duration_seconds=args.duration,
            frame_height=args.frame_height,
            fast=not args.safe,
            capture_rollout=capture_rollout,
            pre_shot_grabber=pre_shot_grabber,
        )
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "rollout_count": manifest["rollout_count"]}, indent=2))
    finally:
        try:
            bridge.disconnect()
        finally:
            stop_owned_engine(engine_process)


if __name__ == "__main__":
    main()
# python scripts/collect_rollouts.py --output-dir data/collect_rollouts_debug --count 1 --fps 1 --duration 0.1 --connect-timeout 20 --prepare-timeout 20
