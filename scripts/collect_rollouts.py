#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
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
from scripts.physics_capture_contract import load_physics_capture  # noqa: E402
from scripts.physics_capture_types import EventType  # noqa: E402
from src.webui.bridge import PhysicsCaptureV1Failure, PlayingMode, ScienceBirdsBridge  # noqa: E402
from scripts.physics_rollout_contract import (  # noqa: E402
    CaptureProvenance,
    PhysicsPersistenceError,
)
from scripts.physics_rollout_persistence import (  # noqa: E402
    install_physics_metadata,
    persist_physics_rollout,
    publish_physics_shot,
)
from scripts.physics_rollout_semantics import REQUIRED_ROLLOUT_SEMANTICS_FIELDS  # noqa: E402
from scripts.rollout_validation_types import PhysicsArtifactError, PhysicsRecoveryResult  # noqa: E402
from scripts.scenario_manifest import ScenarioManifest, load_manifest  # noqa: E402


DEFAULT_DESKTOP_GAME_CROP = (32, 64, 672, 544)
DEFAULT_GAME_VIEWPORT_SIZE = (640, 480)
DEFAULT_HUMAN_HOLD_MS = 600
PRE_DRAG_OVERLAY_TEXT = "phase=pre_drag pre_shot_baseline"
DEFAULT_PRE_SHOT_GUARD_RECOVERY_ATTEMPTS = 2
PRE_SHOT_NEW_SET_STATES = {"NEWTRAININGSET", "RESUMETRAINING", "NEWTRIAL", "NEWTESTSET"}
PRE_SHOT_MENU_STATES = {"MAIN_MENU", "EPISODE_MENU", "LEVEL_SELECTION", "WON", "LOST"}


def _image_is_uniform(image) -> bool:
    extrema = image.getextrema()
    return all(channel_min == channel_max for channel_min, channel_max in extrema)


def _default_desktop_crop_for(image) -> tuple[int, int, int, int] | None:
    detected_crop = _detect_desktop_game_crop(image)
    if detected_crop is not None:
        return detected_crop
    left, top, right, bottom = DEFAULT_DESKTOP_GAME_CROP
    if image.width >= right and image.height >= bottom:
        return DEFAULT_DESKTOP_GAME_CROP
    return None


def _detect_desktop_game_crop(image) -> tuple[int, int, int, int] | None:
    viewport_width, viewport_height = DEFAULT_GAME_VIEWPORT_SIZE
    if image.width < viewport_width or image.height < viewport_height:
        return None
    bbox = image.convert("RGB").getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None
    crop = (left, top, left + viewport_width, top + viewport_height)
    if crop[2] > image.width or crop[3] > image.height:
        return None
    return crop


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


def _interpolate_point(start: tuple[int, int], end: tuple[int, int], fraction: float) -> tuple[int, int]:
    clamped = min(1.0, max(0.0, fraction))
    return (int(round(start[0] + (end[0] - start[0]) * clamped)), int(round(start[1] + (end[1] - start[1]) * clamped)))


def _draw_overlay(image, text: str, action: dict, shot: dict, *, action_fraction: float = 1.0, show_action_guides: bool = True):
    from PIL import ImageDraw, ImageFont

    output = image.copy()
    draw = ImageDraw.Draw(output)
    banner_height = min(max(24, output.height // 14), max(24, output.height))
    draw.rectangle((0, 0, output.width, banner_height - 1), fill=(0, 0, 0))
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((6, 5), text, fill=(255, 255, 255), font=font)
    if not show_action_guides:
        return output
    guide_points = _action_guide_points(action, shot, output.height)
    if guide_points is not None:
        start, release_end = guide_points
        end = _interpolate_point(start, release_end, action_fraction)
        draw.line((start[0], start[1], end[0], end[1]), fill=(0, 0, 0), width=7)
        draw.line((start[0], start[1], end[0], end[1]), fill=(255, 230, 0), width=5)
        radius = 5
        draw.ellipse((start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius), outline=(0, 255, 255), width=2)
        draw.ellipse((end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius), outline=(255, 80, 80), width=2)
    launch_points = _launch_guide_points(action, output.height)
    if launch_points is not None:
        launch_start, full_launch_end = launch_points
        launch_end = _interpolate_point(launch_start, full_launch_end, action_fraction)
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
    video_frames_dir: Path,
    frames_dir: Path,
    *,
    action: dict,
    shot: dict,
    fps: float,
    pre_shot_path: Path | None = None,
    lead_in_seconds: float = 0.75,
) -> dict:
    from PIL import Image

    video_frames_dir.mkdir(parents=True, exist_ok=True)

    overlay_text = format_action_overlay_text(action, shot)
    video_index = 0
    pre_action_frame_count = 0
    pre_drag_frame_count = 0
    aim_hold_frame_count = 0
    if pre_shot_path is not None and pre_shot_path.is_file():
        pre_shot_image = Image.open(pre_shot_path).convert("RGB")
        pre_action_frame_count = max(1, int(round(float(fps) * lead_in_seconds)))
        pre_drag_frame_count = max(1, pre_action_frame_count // 2)
        aim_hold_frame_count = pre_action_frame_count - pre_drag_frame_count
        for _ in range(pre_drag_frame_count):
            _draw_overlay(pre_shot_image, PRE_DRAG_OVERLAY_TEXT, action, shot, show_action_guides=False).save(
                video_frames_dir / f"frame_{video_index:06d}.png", format="PNG"
            )
            video_index += 1
        for aim_index in range(aim_hold_frame_count):
            action_fraction = 0.25 + (0.75 * (aim_index + 1) / aim_hold_frame_count)
            _draw_overlay(pre_shot_image, f"phase=aim_hold {overlay_text}", action, shot, action_fraction=action_fraction).save(
                video_frames_dir / f"frame_{video_index:06d}.png", format="PNG"
            )
            video_index += 1

    for frame_path in sorted(frames_dir.glob("frame_*.png")):
        image = Image.open(frame_path).convert("RGB")
        _draw_overlay(image, overlay_text, action, shot).save(video_frames_dir / f"frame_{video_index:06d}.png", format="PNG")
        video_index += 1

    return {
        "video_frames_dir": str(video_frames_dir),
        "video_input_pattern": str(video_frames_dir / "frame_%06d.png"),
        "pre_action_frame_count": pre_action_frame_count,
        "pre_drag_frame_count": pre_drag_frame_count,
        "aim_hold_frame_count": aim_hold_frame_count,
        "video_phase_counts": {
            "pre_drag": pre_drag_frame_count,
            "aim_hold": aim_hold_frame_count,
            "rollout": video_index - pre_action_frame_count,
        },
        "video_frame_count": video_index,
        "video_overlay": {
            "position": "top",
            "text": overlay_text,
            "action_guide": "cyan=start yellow=pull red=release green=launch",
        },
    }


def write_action_logs(output_dir: Path, attempts: list[dict]) -> dict[str, str]:
    trials = []
    accepted_trials = []
    for rollout in attempts:
        trial = {
            "shot_name": rollout["name"],
            "action": rollout["action"],
            "shot": rollout["shot"],
            "shoot_response": rollout["shoot_response"],
            "frame_count": rollout["frame_count"],
            "metadata_path": rollout["metadata_path"],
        }
        for key in (
            "pre_shot_path",
            "video_path",
            "slingshot_reference",
            "fresh_engine_attempt",
            "attempt_status",
            "invalid_reason",
            "retry_attempt",
            "quarantined_path",
            "prior_invalid_attempts",
            "pre_shot_protocol_state",
            "post_shoot_protocol_state",
            "post_capture_protocol_state",
            "post_recovery_protocol_state",
            "recovery_action",
            "pre_shot_guard",
            "artifact_validation",
        ):
            if rollout.get(key) is not None:
                trial[key] = rollout[key]
        trial["accepted"] = bool(trial.get("artifact_validation", {}).get("accepted"))
        trials.append(trial)
        if trial["accepted"]:
            accepted_trials.append(trial)

    action_log_path = output_dir / "action_log.json"
    action_log_jsonl_path = output_dir / "action_log.jsonl"
    payload = {
        "episode_dir": str(output_dir),
        "attempt_count": len(trials),
        "accepted_trial_count": len(accepted_trials),
        "invalid_attempts": [trial for trial in trials if not trial["accepted"]],
        "trial_count": len(trials),
        "trials": trials,
        "accepted_trials": accepted_trials,
    }
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


def _normalize_rgb_image(image):
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _normalize_delta_images(previous_image, current_image):
    previous_image = _normalize_rgb_image(previous_image)
    current_image = _normalize_rgb_image(current_image)
    if current_image.size != previous_image.size:
        current_image = current_image.resize(previous_image.size)
    return previous_image, current_image


def _action_int(value) -> int:
    return int(value)


def _protocol_state_snapshot(bridge) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        snapshot["game_state"] = bridge.get_game_state().name
    except Exception as exc:
        snapshot["game_state_error"] = str(exc)
    try:
        snapshot["current_level"] = bridge.get_current_level()
    except Exception as exc:
        snapshot["current_level_error"] = str(exc)
    try:
        snapshot["score"] = bridge.get_current_score()
    except Exception as exc:
        snapshot["score_error"] = str(exc)
    return snapshot


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

    previous_image, current_image = _normalize_delta_images(previous_image, current_image)
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


def _artifact_missing_result(shot_dir: Path, message: str) -> dict:
    return {
        "accepted": False,
        "classification": "missing-artifact",
        "invalid_reason": "missing_artifact",
        "signals": ["missing-artifact"],
        "shot_dir": str(shot_dir),
        "message": message,
    }


def _resolve_artifact_path(path_value, shot_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, shot_dir / path, shot_dir.parent / path):
        if candidate.exists():
            return candidate
    return ROOT / path


def _numeric_metadata_value(metadata: dict, key: str) -> int:
    value = metadata.get(key, 0)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _score_from_metadata(metadata: dict) -> int | None:
    state_samples = metadata.get("state_samples")
    if not isinstance(state_samples, list):
        return None
    for sample in reversed(state_samples):
        if not isinstance(sample, dict):
            continue
        score = sample.get("score")
        if isinstance(score, int | float):
            return int(score)
    return None


def _frame_paths_from_metadata(metadata: dict, shot_dir: Path, frames_dir: Path) -> list[Path]:
    frames = metadata.get("frames")
    paths = []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict) or not frame.get("path"):
                continue
            paths.append(_resolve_artifact_path(frame["path"], shot_dir))
    if paths:
        return paths
    return sorted(frames_dir.glob("frame_*.png"))


def _menu_like_frame_evidence(image) -> dict[str, float | int | bool]:
    image = _normalize_rgb_image(image)
    total_pixels = image.width * image.height
    raw_pixels = image.tobytes()
    white_pixels = 0
    bright_saturated_pixels = 0
    for offset in range(0, len(raw_pixels), 3):
        red, green, blue = raw_pixels[offset], raw_pixels[offset + 1], raw_pixels[offset + 2]
        if red >= 200 and green >= 200 and blue >= 200:
            white_pixels += 1
        if max(red, green, blue) >= 120 and max(red, green, blue) - min(red, green, blue) >= 80:
            bright_saturated_pixels += 1
    unique_colors = len(image.getcolors(maxcolors=total_pixels + 1) or [])
    white_ratio = white_pixels / total_pixels if total_pixels else 0.0
    bright_saturated_ratio = bright_saturated_pixels / total_pixels if total_pixels else 0.0
    menu_like = white_ratio >= 0.7 and unique_colors <= 3300
    return {
        "menu_like": menu_like,
        "unique_colors": unique_colors,
        "white_ratio": round(float(white_ratio), 6),
        "bright_saturated_ratio": round(float(bright_saturated_ratio), 6),
    }


class RolloutCollectionError(RuntimeError):
    pass


class TemporaryCaptureError(RolloutCollectionError):
    """An operational capture failure explicitly declared safe to retry."""


def _collection_exception_failure_code(
    error: Exception, *, operation: str = "collection"
) -> str:
    if operation == "post_validation_publication":
        return "attempt_publication_error"
    if isinstance(error, TimeoutError):
        return "engine_start_timeout"
    if isinstance(error, (ConnectionError, OSError)):
        return "transport_unavailable"
    if isinstance(error, TemporaryCaptureError):
        return "capture_temporarily_unavailable"
    return "collection_runtime_error"


def _has_complete_strict_semantics(metadata: dict[str, Any]) -> bool:
    (
        initial_identity,
        intervention_event_id,
        termination_reason,
        termination_fixed_step,
        termination_event_id_field,
        terminal_state_fixed_step,
    ) = REQUIRED_ROLLOUT_SEMANTICS_FIELDS
    string_values = (
        metadata.get(initial_identity),
        metadata.get(intervention_event_id),
        metadata.get(termination_reason),
    )
    integer_values = (
        metadata.get(termination_fixed_step),
        metadata.get(terminal_state_fixed_step),
    )
    termination_event_id = metadata.get(termination_event_id_field)
    return (
        all(isinstance(value, str) and value for value in string_values)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in integer_values)
        and termination_event_id_field in metadata
        and (termination_event_id is None or isinstance(termination_event_id, str) and termination_event_id)
    )


class PreShotGuardError(RuntimeError):
    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        self.rollout: dict[str, Any] | None = None
        message = metadata.get("error") or metadata.get("invalid_reason") or "pre-shot guard failed"
        super().__init__(f"recovery_failed: {message}")


def _pre_shot_surface_evidence(image) -> dict[str, Any]:
    if image is None:
        return {
            "available": False,
            "valid": True,
            "classification": "not-captured",
            "invalid_reason": None,
        }
    image = _normalize_rgb_image(image)
    image = _crop_desktop_image(image, _default_desktop_crop_for(image))
    uniform = _image_is_uniform(image)
    menu_evidence = _menu_like_frame_evidence(image)
    if uniform:
        classification = "uniform-pre-shot"
        invalid_reason = "uniform_pre_shot"
        valid = False
    elif menu_evidence["menu_like"]:
        classification = "menu-like-pre-shot"
        invalid_reason = "menu_like_pre_shot"
        valid = False
    else:
        classification = "gameplay-candidate"
        invalid_reason = None
        valid = True
    return {
        "available": True,
        "valid": valid,
        "classification": classification,
        "invalid_reason": invalid_reason,
        "uniform": uniform,
        "width": image.width,
        "height": image.height,
        **menu_evidence,
    }


def _pre_shot_attempt_evidence(protocol_state: dict[str, Any], pre_shot_image) -> dict[str, Any]:
    visual_evidence = _pre_shot_surface_evidence(pre_shot_image)
    game_state = protocol_state.get("game_state")
    protocol_playing = game_state == "PLAYING"
    if not protocol_playing:
        invalid_reason = "protocol_not_playing"
    elif not visual_evidence["valid"]:
        invalid_reason = visual_evidence["invalid_reason"]
    else:
        invalid_reason = None
    return {
        "accepted": invalid_reason is None,
        "invalid_reason": invalid_reason,
        "protocol_playing": protocol_playing,
        "protocol_state": protocol_state,
        "visual_evidence": visual_evidence,
    }


def _call_pre_shot_recovery(bridge, attempt: dict[str, Any]) -> dict[str, Any]:
    protocol_state = attempt["protocol_state"]
    visual_evidence = attempt["visual_evidence"]
    game_state = protocol_state.get("game_state")
    if game_state in PRE_SHOT_NEW_SET_STATES:
        action = "ready_for_new_set"
        if not hasattr(bridge, "ready_for_new_set"):
            return {"action": action, "ok": False, "error": "bridge missing ready_for_new_set"}
        try:
            result = bridge.ready_for_new_set()
        except Exception as exc:
            return {"action": action, "ok": False, "error": str(exc)}
        return {"action": action, "ok": True, "result": result}

    visual_invalid = visual_evidence.get("invalid_reason") in {"menu_like_pre_shot", "uniform_pre_shot"}
    if game_state in PRE_SHOT_MENU_STATES or visual_invalid:
        action = "load_next_available_level"
        if not hasattr(bridge, "load_next_available_level"):
            return {"action": action, "ok": False, "error": "bridge missing load_next_available_level"}
        recovery: dict[str, Any] = {"action": action, "ok": True}
        try:
            if hasattr(bridge, "get_novelty_info"):
                recovery["novelty_info"] = bridge.get_novelty_info()
            recovery["result"] = bridge.load_next_available_level()
        except Exception as exc:
            return {"action": action, "ok": False, "error": str(exc)}
        return recovery

    if game_state == "LOADING":
        return {"action": "wait", "ok": True}
    return {"action": None, "ok": False, "error": f"no safe recovery path for {game_state}"}


def _pre_shot_sample_from_protocol_state(protocol_state: dict[str, Any]) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    if "game_state" in protocol_state:
        sample["state"] = protocol_state["game_state"]
    if "score" in protocol_state:
        sample["score"] = protocol_state["score"]
    return sample


def _save_pre_shot_attempt_image(shot_dir: Path, image, attempt_number: int) -> str | None:
    if image is None:
        return None
    attempt_path = shot_dir / f"pre_shot_guard_attempt_{attempt_number:02d}.png"
    image.save(attempt_path, format="PNG")
    return str(attempt_path)


def _run_pre_shot_guard(
    bridge,
    shot_dir: Path,
    *,
    initial_protocol_state: dict[str, Any],
    pre_shot_grabber=None,
    max_recovery_attempts: int = DEFAULT_PRE_SHOT_GUARD_RECOVERY_ATTEMPTS,
    poll_delay: float = 0.5,
    sleeper=time.sleep,
) -> dict[str, Any]:
    max_recovery_attempts = max(0, int(max_recovery_attempts))
    recovery_attempts = 0
    protocol_state = dict(initial_protocol_state)
    attempts = []
    recoveries = []
    last_successful_recovery_action = None
    last_image = None

    while True:
        attempt_number = len(attempts) + 1
        last_image = pre_shot_grabber() if pre_shot_grabber is not None else None
        if last_image is not None:
            last_image = _crop_desktop_image(_normalize_rgb_image(last_image), _default_desktop_crop_for(last_image))
        attempt = _pre_shot_attempt_evidence(protocol_state, last_image)
        attempt_path = _save_pre_shot_attempt_image(shot_dir, last_image, attempt_number)
        if attempt_path is not None:
            attempt["visual_evidence"]["path"] = attempt_path
        attempts.append(attempt)

        if attempt["accepted"]:
            if last_image is not None:
                pre_shot_path = shot_dir / "pre_shot.png"
                last_image.save(pre_shot_path, format="PNG")
                attempt["visual_evidence"]["accepted_path"] = str(pre_shot_path)
            post_guard_protocol_state = _protocol_state_snapshot(bridge)
            status = "accepted_after_recovery" if recovery_attempts else "accepted"
            guard = {
                "status": status,
                "recovery_attempts": recovery_attempts,
                "invalid_reason": None,
                "protocol_state": protocol_state,
                "post_guard_protocol_state": post_guard_protocol_state,
                "visual_evidence": attempt["visual_evidence"],
                "attempts": attempts,
                "recoveries": recoveries,
            }
            return {
                "pre_shot_image": last_image,
                "pre_shot_sample": _pre_shot_sample_from_protocol_state(post_guard_protocol_state) if last_image is not None else None,
                "post_recovery_protocol_state": post_guard_protocol_state,
                "recovery_action": last_successful_recovery_action,
                "pre_shot_guard": guard,
            }

        if recovery_attempts >= max_recovery_attempts:
            guard = {
                "status": "recovery_failed",
                "recovery_attempts": recovery_attempts,
                "invalid_reason": attempt["invalid_reason"],
                "protocol_state": protocol_state,
                "visual_evidence": attempt["visual_evidence"],
                "attempts": attempts,
                "recoveries": recoveries,
                "recovery_action": last_successful_recovery_action,
                "post_recovery_protocol_state": protocol_state,
                "error": f"pre-shot surface remained invalid: {attempt['invalid_reason']}",
            }
            raise PreShotGuardError(guard)

        recovery = _call_pre_shot_recovery(bridge, attempt)
        recoveries.append(recovery)
        if not recovery.get("ok"):
            guard = {
                "status": "recovery_failed",
                "recovery_attempts": recovery_attempts,
                "invalid_reason": attempt["invalid_reason"],
                "protocol_state": protocol_state,
                "visual_evidence": attempt["visual_evidence"],
                "attempts": attempts,
                "recoveries": recoveries,
                "recovery_action": last_successful_recovery_action,
                "post_recovery_protocol_state": protocol_state,
                "error": recovery.get("error") or "safe recovery failed",
            }
            raise PreShotGuardError(guard)
        recovery_attempts += 1
        if recovery.get("action") != "wait":
            last_successful_recovery_action = recovery.get("action")
        if poll_delay > 0:
            sleeper(poll_delay)
        protocol_state = _protocol_state_snapshot(bridge)


def _rollout_frame_evidence(frame_paths: list[Path], pre_shot_path: Path | None) -> dict:
    from PIL import Image

    previous_image = None
    max_frame_delta = 0
    max_pre_shot_delta = 0
    first_frame_evidence = None
    pre_shot_image = None
    if pre_shot_path is not None and pre_shot_path.is_file():
        pre_shot_image = Image.open(pre_shot_path)
    for index, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path)
        if index == 0:
            first_frame_evidence = _menu_like_frame_evidence(image)
        if previous_image is not None:
            max_frame_delta = max(max_frame_delta, int(_image_delta_stats(previous_image, image)["changed_pixel_count"]))
        if pre_shot_image is not None:
            max_pre_shot_delta = max(max_pre_shot_delta, int(_image_delta_stats(pre_shot_image, image)["changed_pixel_count"]))
        previous_image = image
    return {
        "observed_max_frame_delta": max_frame_delta,
        "observed_max_pre_shot_delta": max_pre_shot_delta,
        "first_frame_evidence": first_frame_evidence or {},
    }


def validate_rollout_artifact(shot_dir: Path, *, gameplay_motion_threshold: int = 100, capture_contract: str = "legacy_rgb_v1") -> dict:
    shot_dir = Path(shot_dir)
    if capture_contract not in {"legacy_rgb_v1", "physics_capture_v1"}:
        return _artifact_missing_result(shot_dir, f"unsupported capture contract: {capture_contract}")
    if not shot_dir.is_dir():
        return _artifact_missing_result(shot_dir, "missing shot directory")
    metadata_path = shot_dir / "metadata.json"
    if not metadata_path.is_file():
        return _artifact_missing_result(shot_dir, "missing metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _artifact_missing_result(shot_dir, f"unreadable metadata.json: {exc}")
    if not isinstance(metadata, dict):
        return _artifact_missing_result(shot_dir, "metadata.json must contain an object")
    if capture_contract == "physics_capture_v1":
        try:
            from scripts.rollout_artifacts import validate_physics_shot_artifact

            summary = validate_physics_shot_artifact(shot_dir)
        except (PhysicsArtifactError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return {"accepted": False, "classification": "physics-capture-invalid", "invalid_reason": "physics_capture_invalid", "retryable": False, "retry_decision": "quarantine", "message": str(exc), "shot_dir": str(shot_dir)}
        return {"accepted": True, "classification": "gameplay-valid", "invalid_reason": None, "retryable": False, "retry_decision": "accept", "message": "physics_capture_v1", "shot_dir": str(shot_dir), "physics_state_count": summary.state_count, "physics_event_count": summary.event_count}

    frames_dir_value = metadata.get("frames_dir")
    if not frames_dir_value:
        return _artifact_missing_result(shot_dir, "missing frames_dir metadata")
    frames_dir = _resolve_artifact_path(frames_dir_value, shot_dir)
    if not frames_dir.is_dir():
        return _artifact_missing_result(shot_dir, f"missing frames directory: {frames_dir}")

    frame_paths = _frame_paths_from_metadata(metadata, shot_dir, frames_dir)
    missing_frame_paths = [frame_path for frame_path in frame_paths if not frame_path.is_file()]
    if not frame_paths or missing_frame_paths:
        return _artifact_missing_result(shot_dir, f"missing frames: {len(missing_frame_paths) or 'none listed'}")
    frame_count = _numeric_metadata_value(metadata, "frame_count")
    if frame_count <= 0 or len(frame_paths) < frame_count:
        return _artifact_missing_result(shot_dir, f"missing frames: expected {frame_count}, found {len(frame_paths)}")

    pre_shot_path = _resolve_artifact_path(metadata["pre_shot_path"], shot_dir) if metadata.get("pre_shot_path") else None
    try:
        frame_evidence = _rollout_frame_evidence(frame_paths, pre_shot_path)
    except OSError as exc:
        return _artifact_missing_result(shot_dir, f"unreadable frame artifact: {exc}")

    reported_max_frame_delta = _numeric_metadata_value(metadata, "max_frame_delta")
    reported_max_pre_shot_delta = _numeric_metadata_value(metadata, "max_pre_shot_delta")
    max_frame_delta = int(frame_evidence["observed_max_frame_delta"])
    max_pre_shot_delta = int(frame_evidence["observed_max_pre_shot_delta"])
    first_frame_evidence = frame_evidence["first_frame_evidence"]
    signals = []
    if max_frame_delta == 0:
        signals.append("no-frame-motion")
    if max_frame_delta == 0 and max_pre_shot_delta == 0 and first_frame_evidence.get("menu_like"):
        signals.append("menu-detected")
    if 0 < max_frame_delta < gameplay_motion_threshold:
        signals.append("low-motion-suspicious")

    if "menu-detected" in signals:
        classification = "menu-detected"
        invalid_reason = "menu_detected"
        accepted = False
    elif "no-frame-motion" in signals:
        classification = "no-frame-motion"
        invalid_reason = "no_frame_motion"
        accepted = False
    elif "low-motion-suspicious" in signals:
        classification = "low-motion-suspicious"
        invalid_reason = "low_motion_suspicious"
        accepted = False
    else:
        classification = "gameplay-valid"
        invalid_reason = None
        accepted = True
        signals.append("gameplay-valid")

    retryable = classification == "low-motion-suspicious"
    retry_decision = "retry" if retryable else ("accept" if accepted else "quarantine")

    return {
        "accepted": accepted,
        "classification": classification,
        "invalid_reason": invalid_reason,
        "retryable": retryable,
        "retry_decision": retry_decision,
        "signals": signals,
        "shot_dir": str(shot_dir),
        "metadata_path": str(metadata_path),
        "frames_dir": str(frames_dir),
        "frame_count": frame_count,
        "frame_path_count": len(frame_paths),
        "max_frame_delta": max_frame_delta,
        "max_pre_shot_delta": max_pre_shot_delta,
        "observed_max_frame_delta": max_frame_delta,
        "observed_max_pre_shot_delta": max_pre_shot_delta,
        "reported_max_frame_delta": reported_max_frame_delta,
        "reported_max_pre_shot_delta": reported_max_pre_shot_delta,
        "capture_stop_reason": metadata.get("capture_stop_reason"),
        "score": _score_from_metadata(metadata),
        "first_frame_evidence": first_frame_evidence,
        "message": classification,
    }


def cleanup_incomplete_physics_attempts(output_dir: Path) -> tuple[str, ...]:
    """Remove incomplete temporary enriched attempts before resuming."""
    removed: list[str] = []
    for path in sorted(output_dir.glob("shot_*.tmp")):
        if path.is_symlink():
            path.unlink()
            removed.append(path.name)
        elif path.is_dir():
            shutil.rmtree(path)
            removed.append(path.name)
    return tuple(removed)


def _quarantine_completed_physics_shot(output_dir: Path, shot_dir: Path) -> Path:
    invalid_root = output_dir / "invalid_attempts"
    invalid_root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    target = invalid_root / f"{shot_dir.name}_recovered_{suffix:02d}"
    while target.exists():
        suffix += 1
        target = invalid_root / f"{shot_dir.name}_recovered_{suffix:02d}"
    shot_dir.replace(target)
    return target


def recover_physics_capture_attempts(output_dir: Path) -> PhysicsRecoveryResult:
    removed = cleanup_incomplete_physics_attempts(output_dir)
    quarantined: list[str] = []
    from scripts.rollout_artifacts import validate_physics_shot_artifact

    for shot_dir in sorted(output_dir.glob("shot_[0-9][0-9][0-9]")):
        try:
            validate_physics_shot_artifact(shot_dir)
        except PhysicsArtifactError:
            target = _quarantine_completed_physics_shot(output_dir, shot_dir)
            quarantined.append(str(target.relative_to(output_dir)))
    return PhysicsRecoveryResult(removed, tuple(quarantined))


def capture_physics_rollout(
    bridge,
    output_dir: Path,
    *,
    target_fps: float,
    duration_seconds: float,
    max_frames: int | None = None,
    state_header: dict | None = None,
    clock=time.monotonic,
    sleeper=time.sleep,
    player_sha256: str | None = None,
    protocol_sha256: str | None = None,
    archive_sha256: str | None = None,
    initial_capture=None,
    shoot=None,
    expected_initial_engine_state_identity: str | None = None,
    scenario_context: dict[str, Any] | None = None,
) -> dict:
    if not all(isinstance(value, str) for value in (player_sha256, protocol_sha256, archive_sha256)):
        raise RolloutCollectionError("physics capture requires lowercase SHA-256 player, protocol, and archive provenance")
    try:
        return persist_physics_rollout(
            bridge,
            output_dir,
            target_fps=target_fps,
            duration_seconds=duration_seconds,
            max_frames=max_frames,
            state_header=state_header,
            provenance=CaptureProvenance(player_sha256, protocol_sha256, archive_sha256),
            clock=clock,
            sleeper=sleeper,
            initial_capture=initial_capture,
            shoot=shoot,
            expected_initial_engine_state_identity=expected_initial_engine_state_identity,
            scenario_context=scenario_context,
        )
    except PhysicsPersistenceError as error:
        raise RolloutCollectionError(str(error)) from error


def _physics_contract_descriptor(player_sha256: str, protocol_sha256: str, archive_sha256: str) -> dict:
    from world_model.data.types import PHYSICS_CAPTURE_V1

    return {
        "contract_name": PHYSICS_CAPTURE_V1.contract_name,
        "contract_version": PHYSICS_CAPTURE_V1.contract_version,
        "artifact_layout_version": PHYSICS_CAPTURE_V1.artifact_layout_version,
        "player_sha256": player_sha256,
        "protocol_sha256": protocol_sha256,
        "archive_sha256": archive_sha256,
        "declared_capabilities": list(PHYSICS_CAPTURE_V1.declared_capabilities),
        "sidecar_paths": [{"relative_path": sidecar.relative_path, "capabilities": list(sidecar.capabilities)} for sidecar in PHYSICS_CAPTURE_V1.sidecar_paths],
    }


def _invalid_attempt_status(artifact_validation: dict, retryable_recovery_action: str) -> str:
    if artifact_validation.get("retryable") and retryable_recovery_action == "fresh_engine_retry":
        return "invalid_retryable"
    if artifact_validation.get("retryable") and retryable_recovery_action == "fresh_engine_attempts_exhausted":
        return "invalid_exhausted"
    return "invalid_quarantined"


def _invalid_attempt_recovery_action(artifact_validation: dict, retryable_recovery_action: str) -> str:
    if artifact_validation.get("retryable"):
        return retryable_recovery_action
    return "quarantine"


def _quarantine_target(output_dir: Path, shot_name: str, retry_attempt: int) -> Path:
    invalid_root = output_dir / "invalid_attempts"
    invalid_root.mkdir(parents=True, exist_ok=True)
    return invalid_root / f"{shot_name}_attempt_{int(retry_attempt):02d}"


def _quarantined_path_value(value, shot_dir: Path, quarantined_path: Path):
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if path.is_absolute():
        try:
            return str(quarantined_path / path.relative_to(shot_dir))
        except ValueError:
            return value
    if path.parts and path.parts[0] == shot_dir.name:
        return str(quarantined_path.joinpath(*path.parts[1:]))
    try:
        return str(quarantined_path / (shot_dir.parent / path).relative_to(shot_dir))
    except ValueError:
        return value


def _rewrite_quarantined_metadata(metadata: dict, shot_dir: Path, quarantined_path: Path) -> dict:
    rewritten = dict(metadata)
    for key in ("metadata_path", "pre_shot_path", "video_path", "frames_dir", "video_frames_dir", "video_input_pattern"):
        if key in rewritten:
            rewritten[key] = _quarantined_path_value(rewritten[key], shot_dir, quarantined_path)

    frames = rewritten.get("frames")
    if isinstance(frames, list):
        rewritten_frames = []
        for frame in frames:
            if isinstance(frame, dict):
                rewritten_frame = dict(frame)
                if "path" in rewritten_frame:
                    rewritten_frame["path"] = _quarantined_path_value(rewritten_frame["path"], shot_dir, quarantined_path)
                rewritten_frames.append(rewritten_frame)
            else:
                rewritten_frames.append(frame)
        rewritten["frames"] = rewritten_frames

    validation = rewritten.get("artifact_validation")
    if isinstance(validation, dict):
        rewritten_validation = dict(validation)
        rewritten_validation["shot_dir"] = str(quarantined_path)
        if "metadata_path" in rewritten_validation:
            rewritten_validation["metadata_path"] = str(quarantined_path / "metadata.json")
        if "frames_dir" in rewritten_validation:
            rewritten_validation["frames_dir"] = _quarantined_path_value(rewritten_validation["frames_dir"], shot_dir, quarantined_path)
        rewritten["artifact_validation"] = rewritten_validation
    return rewritten


def _copy_invalid_attempt(shot_dir: Path, quarantined_path: Path) -> dict:
    if quarantined_path.exists():
        raise RuntimeError(f"invalid attempt quarantine path already exists: {quarantined_path}")
    shutil.copytree(shot_dir, quarantined_path)
    metadata_path = quarantined_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rewritten = _rewrite_quarantined_metadata(metadata, shot_dir, quarantined_path)
    _write_metadata(metadata_path, rewritten)
    contract = "physics_capture_v1" if rewritten.get("capture_contract") == "physics_capture_v1" else "legacy_rgb_v1"
    rewritten["artifact_validation"] = validate_rollout_artifact(quarantined_path, capture_contract=contract)
    _write_metadata(metadata_path, rewritten)
    return rewritten


def _invalid_attempt_reference(rollout: dict) -> dict[str, Any]:
    reference = {
        "shot_name": rollout["name"],
        "attempt_status": rollout.get("attempt_status"),
        "accepted": False,
        "invalid_reason": rollout.get("invalid_reason"),
        "retry_attempt": rollout.get("retry_attempt"),
        "recovery_action": rollout.get("recovery_action"),
        "quarantined_path": rollout.get("quarantined_path"),
        "metadata_path": rollout.get("metadata_path"),
    }
    if rollout.get("fresh_engine_attempt") is not None:
        reference["fresh_engine_attempt"] = rollout["fresh_engine_attempt"]
    validation = rollout.get("artifact_validation")
    if isinstance(validation, dict):
        reference["artifact_validation"] = validation
    return {key: value for key, value in reference.items() if value is not None}


def _write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _finalize_attempt_metadata(
    *,
    output_dir: Path,
    shot_dir: Path,
    metadata: dict,
    artifact_validation: dict,
    retry_attempt: int,
    prior_invalid_attempts: list[dict] | None,
    retryable_recovery_action: str,
    override_recovery_action: bool = True,
) -> dict:
    metadata["artifact_validation"] = artifact_validation
    metadata["accepted"] = bool(artifact_validation.get("accepted"))
    metadata["retry_attempt"] = int(retry_attempt)
    if prior_invalid_attempts:
        metadata["prior_invalid_attempts"] = list(prior_invalid_attempts)

    if metadata["accepted"]:
        metadata["attempt_status"] = "accepted"
        return metadata

    recovery_action = _invalid_attempt_recovery_action(artifact_validation, retryable_recovery_action)
    metadata["attempt_status"] = _invalid_attempt_status(artifact_validation, retryable_recovery_action)
    metadata["invalid_reason"] = artifact_validation.get("invalid_reason")
    if override_recovery_action:
        metadata["recovery_action"] = recovery_action
    quarantined_path = _quarantine_target(output_dir, shot_dir.name, retry_attempt)
    metadata["quarantined_path"] = str(quarantined_path)
    _write_metadata(shot_dir / "metadata.json", metadata)
    metadata = _copy_invalid_attempt(shot_dir, quarantined_path)
    return metadata


def _rollout_record_from_metadata(
    shot_dir: Path,
    *,
    action: dict,
    shot: dict,
    metadata: dict,
    shoot_response=None,
    slingshot_reference=None,
) -> dict[str, Any]:
    return {
        "name": shot_dir.name,
        "action": action,
        "shot": shot,
        "shoot_response": metadata.get("shoot_response") if shoot_response is None else shoot_response,
        "frame_count": metadata["frame_count"],
        "slingshot_reference": slingshot_reference,
        "metadata_path": str(Path(metadata["quarantined_path"]) / "metadata.json")
        if metadata.get("quarantined_path")
        else str(shot_dir / "metadata.json"),
        "pre_shot_protocol_state": metadata.get("pre_shot_protocol_state"),
        "post_shoot_protocol_state": metadata.get("post_shoot_protocol_state"),
        "post_capture_protocol_state": metadata.get("post_capture_protocol_state"),
        "post_recovery_protocol_state": metadata.get("post_recovery_protocol_state"),
        "recovery_action": metadata.get("recovery_action"),
        "pre_shot_guard": metadata.get("pre_shot_guard"),
        "artifact_validation": metadata.get("artifact_validation"),
        "attempt_status": metadata.get("attempt_status"),
        "accepted": bool(metadata.get("accepted")),
        "invalid_reason": metadata.get("invalid_reason"),
        "retry_attempt": metadata.get("retry_attempt"),
        "quarantined_path": metadata.get("quarantined_path"),
        "prior_invalid_attempts": metadata.get("prior_invalid_attempts"),
        "initial_engine_state_identity": metadata.get("initial_engine_state_identity"),
        "intervention_event_id": metadata.get("intervention_event_id"),
        "termination_reason": metadata.get("termination_reason"),
        "termination_fixed_step": metadata.get("termination_fixed_step"),
        "termination_event_id": metadata.get("termination_event_id"),
        "terminal_state_fixed_step": metadata.get("terminal_state_fixed_step"),
        "expected_initial_engine_state_identity": metadata.get("expected_initial_engine_state_identity"),
        "scenario_context": metadata.get("scenario_context"),
        **({"pre_shot_path": metadata["pre_shot_path"]} if "pre_shot_path" in metadata else {}),
        **({"video_path": metadata["video_path"]} if "video_path" in metadata else {}),
    }


def _mark_guard_failure_retryable(artifact_validation: dict, retryable_recovery_action: str) -> dict:
    if retryable_recovery_action not in {"fresh_engine_retry", "fresh_engine_attempts_exhausted"}:
        return artifact_validation
    marked = dict(artifact_validation)
    marked["retryable"] = True
    marked["retry_decision"] = "retry" if retryable_recovery_action == "fresh_engine_retry" else "retry_exhausted"
    return marked


def _record_fresh_engine_attempt_metadata(
    output_dir: Path,
    rollout: dict,
    *,
    attempt: int,
    slingshot_reference: dict[str, int] | None,
    physics_capture_v1: bool,
) -> None:
    rollout["slingshot_reference"] = slingshot_reference
    rollout["fresh_engine_attempt"] = attempt
    if physics_capture_v1 and rollout.get("accepted"):
        return
    metadata_paths = [output_dir / rollout["name"] / "metadata.json", Path(rollout["metadata_path"])]
    for shot_metadata_path in dict.fromkeys(metadata_paths):
        if not shot_metadata_path.is_file():
            continue
        shot_metadata = json.loads(shot_metadata_path.read_text(encoding="utf-8"))
        shot_metadata["fresh_engine_attempt"] = attempt
        if slingshot_reference is not None:
            shot_metadata["slingshot_reference"] = slingshot_reference
        _write_metadata(shot_metadata_path, shot_metadata)


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


def write_action_plan(
    output_dir: Path,
    *,
    count: int,
    drag_start: tuple[int, int] = (300, 220),
    bidirectional_launches: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = generate_diverse_drag_release_actions(
        drag_start=drag_start,
        count=count,
        bidirectional_launches=bidirectional_launches,
    )
    path = output_dir / "action_plan.json"
    path.write_text(json.dumps({"action_count": len(actions), "actions": actions}, indent=2), encoding="utf-8")
    return path


def load_actions_from_action_log(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise ValueError("action log must contain a trials list")
    actions = []
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, dict) or "action" not in trial:
            raise ValueError(f"trial {index} is missing action")
        action = trial["action"]
        if not isinstance(action, dict):
            raise ValueError(f"trial {index} action must be an object")
        actions.append(action)
    trial_count = payload.get("trial_count")
    if trial_count is not None and int(trial_count) != len(actions):
        raise ValueError("action log trial_count does not match trials length")
    return actions


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
    shoot=None,
    max_duration_seconds: float | None = None,
    settle_seconds: float = 1.5,
    settle_pixel_threshold: int = 100,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> dict:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")
    if settle_seconds < 0:
        raise ValueError("settle_seconds must be non-negative")
    if settle_pixel_threshold < 0:
        raise ValueError("settle_pixel_threshold must be non-negative")

    if grabber is None:
        from PIL import ImageGrab

        grabber = ImageGrab

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pre_shot_path = None
    if pre_shot_image is not None:
        pre_shot_crop = _default_desktop_crop_for(pre_shot_image) if desktop_crop == DEFAULT_DESKTOP_GAME_CROP else desktop_crop
        pre_shot_image = _crop_desktop_image(pre_shot_image, pre_shot_crop)
        if _image_is_uniform(pre_shot_image):
            raise RuntimeError("uniform desktop pre-shot screenshot; refusing to save rollout baseline")
        pre_shot_path = output_dir / "pre_shot.png"
        pre_shot_image.save(pre_shot_path, format="PNG")

    interval = 1.0 / float(target_fps)
    completion_capture = shoot is not None and max_frames is None
    if completion_capture and max_duration_seconds is None:
        max_duration_seconds = max(duration_seconds * 8.0, duration_seconds + 12.0)
    total_frames = None if completion_capture else max_frames if max_frames is not None else max(1, int(math.ceil(duration_seconds * target_fps)))
    started_at = clock()
    schedule_started_at = started_at
    frames = []
    state_samples = []
    previous_image = None
    max_frame_delta = 0
    max_mean_absolute_channel_delta = 0.0
    max_pre_shot_delta = 0
    max_pre_shot_delta_bbox = None
    shoot_response = None
    shoot_frame_index = None
    post_shot_started_at = None
    settled_since = None
    stop_reason = "max_frames" if max_frames is not None else "fixed_duration"

    frame_index = 0
    while total_frames is None or frame_index < total_frames:
        if completion_capture and post_shot_started_at is not None and clock() - post_shot_started_at >= float(max_duration_seconds):
            stop_reason = "max_duration"
            break
        target_time = schedule_started_at + frame_index * interval
        now = clock()
        if now < target_time:
            sleeper(target_time - now)
        elapsed = clock() - started_at
        if not completion_capture and elapsed > duration_seconds and frame_index > 0:
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
        if shoot is not None and shoot_response is None:
            shoot_response = shoot()
            shoot_frame_index = frame_index
            post_shot_started_at = clock()
            schedule_started_at = post_shot_started_at
            settled_since = None

        if completion_capture and post_shot_started_at is not None and shoot_frame_index is not None and frame_index > shoot_frame_index:
            post_shot_elapsed = clock() - post_shot_started_at
            frame_delta = frame.get("frame_delta")
            changed_pixels = frame_delta.get("changed_pixel_count") if isinstance(frame_delta, dict) else None
            if isinstance(changed_pixels, int) and changed_pixels <= settle_pixel_threshold:
                if settled_since is None:
                    settled_since = clock()
            else:
                settled_since = None
            settled_elapsed = 0.0 if settled_since is None else clock() - settled_since
            if post_shot_elapsed >= duration_seconds and settled_elapsed >= settle_seconds:
                stop_reason = "settled"
                frame_index += 1
                break
            if post_shot_elapsed >= float(max_duration_seconds):
                stop_reason = "max_duration"
                frame_index += 1
                break
        frame_index += 1

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
        "capture_stop_reason": stop_reason,
    }
    if completion_capture:
        metadata["min_post_shot_duration_seconds"] = duration_seconds
        metadata["max_duration_seconds"] = max_duration_seconds
        metadata["settle_seconds"] = settle_seconds
        metadata["settle_pixel_threshold"] = settle_pixel_threshold
        if post_shot_started_at is not None:
            metadata["post_shot_capture_seconds"] = round(float(clock() - post_shot_started_at), 6)
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
    if shoot_response is not None:
        metadata["shoot_response"] = shoot_response
        metadata["shoot_frame_index"] = shoot_frame_index

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


class _FinalizedPhysicsBridge:
    def __init__(self, bridge, *, deadline_seconds: float, clock, sleeper) -> None:
        self._bridge = bridge
        self._deadline_seconds = deadline_seconds
        self._clock = clock
        self._sleeper = sleeper

    def get_physics_capture_v1(self):
        deadline = self._clock() + self._deadline_seconds
        while True:
            try:
                return self._bridge.get_physics_capture_v1()
            except PhysicsCaptureV1Failure as error:
                if error.code != 4 or self._clock() >= deadline:
                    raise
                self._sleeper(0.25)


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
    shoot_before_capture: bool = True,
    anchor_actions: bool = True,
    retry_attempt: int = 1,
    fresh_engine_attempt: int | None = None,
    prior_invalid_attempts: list[dict] | None = None,
    retryable_recovery_action: str = "quarantine",
    physics_capture_v1: bool = False,
    physics_bridge=None,
    physics_player_sha256: str | None = None,
    physics_protocol_sha256: str | None = None,
    physics_archive_sha256: str | None = None,
    expected_initial_engine_state_identity: str | None = None,
    scenario_context: dict[str, Any] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if physics_capture_v1:
        recover_physics_capture_attempts(output_dir)
        capture_rollout = capture_physics_rollout
    prior_invalid_attempts = list(prior_invalid_attempts or [])
    if anchor_actions:
        actions = anchor_actions_to_current_slingshot(bridge, actions, frame_height)
    rollouts = []
    for index, action in enumerate(actions, start=start_index):
        shot = action_to_shot(action, frame_height=frame_height)
        final_shot_dir = output_dir / f"shot_{index:03d}"
        shot_dir = output_dir / f"shot_{index:03d}.tmp" if physics_capture_v1 else final_shot_dir
        if physics_capture_v1 and final_shot_dir.exists():
            artifact_validation = validate_rollout_artifact(
                final_shot_dir, capture_contract="physics_capture_v1"
            )
            if artifact_validation.get("accepted"):
                metadata = json.loads((final_shot_dir / "metadata.json").read_text(encoding="utf-8"))
                requested_provenance = {
                    "player_sha256": physics_player_sha256,
                    "protocol_sha256": physics_protocol_sha256,
                    "archive_sha256": physics_archive_sha256,
                }
                identity_matches = (
                    expected_initial_engine_state_identity is None
                    or metadata.get("initial_engine_state_identity")
                    == expected_initial_engine_state_identity
                )
                scenario_matches = (
                    scenario_context is None
                    or metadata.get("scenario_context") == scenario_context
                )
                if (
                    _has_complete_strict_semantics(metadata)
                    and all(metadata.get(field) == value for field, value in requested_provenance.items())
                    and identity_matches
                    and scenario_matches
                ):
                    metadata = dict(metadata)
                    metadata["artifact_validation"] = artifact_validation
                    metadata["accepted"] = True
                    metadata.setdefault("attempt_status", "accepted")
                    rollouts.append(
                        _rollout_record_from_metadata(
                            final_shot_dir,
                            action=action,
                            shot=shot,
                            metadata=metadata,
                            slingshot_reference=action.get("slingshot_reference"),
                        )
                    )
                    continue
                _quarantine_completed_physics_shot(output_dir, final_shot_dir)
            else:
                recover_physics_capture_attempts(output_dir)
        if reset_rollout is not None:
            reset_rollout(index, action)
        if retry_attempt > 1 and shot_dir.exists():
            shutil.rmtree(shot_dir)
        shot_dir.mkdir(parents=True, exist_ok=True)
        pre_shot_image = None
        pre_shot_sample = None
        pre_shot_protocol_state = _protocol_state_snapshot(bridge)
        try:
            pre_shot_guard_result = _run_pre_shot_guard(
                bridge,
                shot_dir,
                initial_protocol_state=pre_shot_protocol_state,
                pre_shot_grabber=pre_shot_grabber,
                sleeper=sleeper,
            )
        except PreShotGuardError as exc:
            metadata = {
                "frame_count": 0,
                "pre_shot_protocol_state": pre_shot_protocol_state,
                "post_recovery_protocol_state": exc.metadata.get("post_recovery_protocol_state"),
                "post_shoot_protocol_state": None,
                "post_capture_protocol_state": _protocol_state_snapshot(bridge),
                "recovery_action": exc.metadata.get("recovery_action"),
                "pre_shot_guard": exc.metadata,
            }
            _write_metadata(shot_dir / "metadata.json", metadata)
            metadata["artifact_validation"] = _mark_guard_failure_retryable(
                validate_rollout_artifact(shot_dir),
                retryable_recovery_action,
            )
            metadata = _finalize_attempt_metadata(
                output_dir=output_dir,
                shot_dir=shot_dir,
                metadata=metadata,
                artifact_validation=metadata["artifact_validation"],
                retry_attempt=retry_attempt,
                prior_invalid_attempts=prior_invalid_attempts,
                retryable_recovery_action=retryable_recovery_action,
                override_recovery_action=False,
            )
            if metadata.get("accepted"):
                _write_metadata(shot_dir / "metadata.json", metadata)
            exc.rollout = _rollout_record_from_metadata(shot_dir, action=action, shot=shot, metadata=metadata)
            raise
        pre_shot_image = pre_shot_guard_result["pre_shot_image"]
        pre_shot_sample = pre_shot_guard_result["pre_shot_sample"]
        pre_shot_guard = pre_shot_guard_result["pre_shot_guard"]
        def shoot_once():
            if physics_capture_v1:
                return bridge.shoot_and_record_ground_truth(
                    shot["x"],
                    shot["y"],
                    tap_time=shot["tapTime"],
                    release_time=shot["releaseTime"],
                )
            return bridge.shoot(
                shot["x"],
                shot["y"],
                tap_time=shot["tapTime"],
                fast=fast,
                release_time=shot["releaseTime"],
            )

        capture_kwargs = {"target_fps": target_fps, "duration_seconds": duration_seconds, "max_frames": max_frames, "clock": clock, "sleeper": sleeper}
        if physics_capture_v1:
            capture_kwargs.update({"player_sha256": physics_player_sha256, "protocol_sha256": physics_protocol_sha256, "archive_sha256": physics_archive_sha256, "expected_initial_engine_state_identity": expected_initial_engine_state_identity, "scenario_context": scenario_context})
        else:
            capture_kwargs["action"] = action
        if not physics_capture_v1 and pre_shot_image is not None:
            capture_kwargs["pre_shot_image"] = pre_shot_image
        if not physics_capture_v1 and pre_shot_sample is not None:
            capture_kwargs["pre_shot_sample"] = pre_shot_sample
        post_recovery_protocol_state = pre_shot_guard_result["post_recovery_protocol_state"]
        post_shoot_protocol_state = None
        response = None
        capture_bridge = physics_bridge if physics_capture_v1 and physics_bridge is not None else bridge
        if physics_capture_v1:
            initial_capture = capture_bridge.get_physics_capture_v1()

            def shoot_and_snapshot():
                nonlocal post_shoot_protocol_state
                result = shoot_once()
                post_shoot_protocol_state = _protocol_state_snapshot(bridge)
                return result

            capture_bridge = _FinalizedPhysicsBridge(
                capture_bridge,
                deadline_seconds=max(30.0, duration_seconds),
                clock=clock,
                sleeper=sleeper,
            )
            capture_kwargs.update({"initial_capture": initial_capture, "shoot": shoot_and_snapshot})
        elif shoot_before_capture:
            response = shoot_once()
            post_shoot_protocol_state = _protocol_state_snapshot(bridge)
        else:
            def shoot_and_snapshot():
                nonlocal post_shoot_protocol_state
                result = shoot_once()
                post_shoot_protocol_state = _protocol_state_snapshot(bridge)
                return result

            capture_kwargs["shoot"] = shoot_and_snapshot
        metadata = capture_rollout(capture_bridge, shot_dir, **capture_kwargs)
        if response is None:
            response = metadata.get("shoot_response")
        metadata["pre_shot_protocol_state"] = pre_shot_protocol_state
        metadata["post_recovery_protocol_state"] = post_recovery_protocol_state
        if post_shoot_protocol_state is None:
            post_shoot_protocol_state = metadata.get("post_shoot_protocol_state")
        if post_shoot_protocol_state is not None:
            metadata["post_shoot_protocol_state"] = post_shoot_protocol_state
        metadata["post_capture_protocol_state"] = _protocol_state_snapshot(bridge)
        if "recovery_action" not in metadata:
            metadata["recovery_action"] = pre_shot_guard_result["recovery_action"]
        metadata["pre_shot_guard"] = pre_shot_guard
        if metadata.get("frames_dir"):
            pre_shot_path = Path(metadata["pre_shot_path"]) if metadata.get("pre_shot_path") else None
            with TemporaryDirectory(prefix="rollout-video-") as temporary_directory:
                video_frames_dir = Path(temporary_directory)
                video_frame_metadata = prepare_rollout_video_frames(
                    video_frames_dir,
                    Path(metadata["frames_dir"]),
                    action=action,
                    shot=shot,
                    fps=target_fps,
                    pre_shot_path=pre_shot_path,
                )
                metadata.update(
                    {
                        key: value
                        for key, value in video_frame_metadata.items()
                        if key not in {"video_frames_dir", "video_input_pattern"}
                    }
                )
                temporary_video_path = shot_dir / ".rollout.tmp.mp4"
                final_video_path = shot_dir / "rollout.mp4"
                metadata.pop("video_path", None)
                try:
                    write_rollout_video(
                        video_frames_dir,
                        temporary_video_path,
                        fps=target_fps,
                        runner=video_runner,
                    )
                    temporary_video_path.replace(final_video_path)
                    metadata["video_path"] = str(final_video_path)
                except Exception as exc:
                    temporary_video_path.unlink(missing_ok=True)
                    metadata["video_error"] = (
                        str(exc)
                        .replace(str(video_frames_dir), "<temporary-video-frames>")
                        .replace(str(temporary_video_path), "<temporary-rollout.mp4>")
                    )
        slingshot_reference = action.get("slingshot_reference")
        if slingshot_reference is not None:
            metadata["slingshot_reference"] = slingshot_reference
        if fresh_engine_attempt is not None:
            metadata["fresh_engine_attempt"] = fresh_engine_attempt
        if not physics_capture_v1:
            _write_metadata(shot_dir / "metadata.json", metadata)
        artifact_validation = validate_rollout_artifact(shot_dir, capture_contract="physics_capture_v1" if physics_capture_v1 else "legacy_rgb_v1")
        metadata = _finalize_attempt_metadata(
            output_dir=output_dir,
            shot_dir=shot_dir,
            metadata=metadata,
            artifact_validation=artifact_validation,
            retry_attempt=retry_attempt,
            prior_invalid_attempts=prior_invalid_attempts,
            retryable_recovery_action=retryable_recovery_action,
        )
        if metadata.get("accepted"):
            if physics_capture_v1:
                metadata = _rewrite_quarantined_metadata(metadata, shot_dir, final_shot_dir)

                def finalize_accepted_physics_shot(
                    shot_descriptor: int,
                    stable_shot_path: Path,
                ) -> None:
                    install_physics_metadata(shot_descriptor, metadata)
                    final_validation = validate_rollout_artifact(
                        stable_shot_path,
                        capture_contract="physics_capture_v1",
                    )
                    if final_validation.get("accepted") is not True:
                        raise RolloutCollectionError(
                            "finalized physics shot failed validation before publication"
                        )

                try:
                    publish_physics_shot(
                        output_dir,
                        shot_dir.name,
                        final_shot_dir.name,
                        finalize_accepted_physics_shot,
                    )
                except PhysicsPersistenceError as error:
                    raise RolloutCollectionError(str(error)) from error
                shot_dir = final_shot_dir
            else:
                _write_metadata(shot_dir / "metadata.json", metadata)
        elif physics_capture_v1 and shot_dir.exists():
            shutil.rmtree(shot_dir)
        rollouts.append(
            _rollout_record_from_metadata(
                shot_dir,
                action=action,
                shot=shot,
                metadata=metadata,
                shoot_response=response,
                slingshot_reference=slingshot_reference,
            )
        )

    accepted_rollouts = [rollout for rollout in rollouts if rollout.get("accepted")]
    invalid_attempts = [rollout for rollout in rollouts if not rollout.get("accepted")]
    manifest = {
        "capture_source": "scripts.collect_rollouts",
        "replay_mode": "same-episode-varied-trials",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "attempt_count": len(rollouts),
        "accepted_rollout_count": len(accepted_rollouts),
        "rollout_count": len(accepted_rollouts),
        "attempts": rollouts,
        "rollouts": rollouts,
        "accepted_rollouts": accepted_rollouts,
        "invalid_attempts": invalid_attempts,
    }
    if physics_capture_v1:
        if not all(isinstance(value, str) for value in (physics_player_sha256, physics_protocol_sha256, physics_archive_sha256)):
            raise RolloutCollectionError("physics capture provenance is incomplete")
        manifest.update({"capture_contract": _physics_contract_descriptor(physics_player_sha256, physics_protocol_sha256, physics_archive_sha256), "schema_version": "physics_capture_v1", "protocol_version": 1, "player_sha256": physics_player_sha256, "protocol_sha256": physics_protocol_sha256, "archive_sha256": physics_archive_sha256, "sidecar_paths": ["physics_state.jsonl", "physics_events.jsonl"], "physics_state_count": sum(int(item.get("frame_count", 0)) for item in accepted_rollouts), "physics_event_count": sum(int(item.get("physics_event_count", 0)) for item in accepted_rollouts)})
        if scenario_context is not None:
            manifest["scenario_context"] = scenario_context
    manifest.update(write_action_logs(output_dir, rollouts))
    if write_manifest:
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _scenario_generation_version(manifest: ScenarioManifest) -> str:
    generation = manifest.generation
    if generation.mode == "generated" and generation.generator_version:
        return generation.generator_version
    if generation.mode == "legacy_static" and generation.importer_identity and generation.importer_version:
        return f"{generation.importer_identity}:{generation.importer_version}"
    raise ValueError("scenario manifest has no version-bounded generator or importer")


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
    fresh_engine_attempts: int = 1,
    engine_agent_port: int | None = None,
    engine_game_port: int | None = None,
    start_engine_func=start_engine,
    connect_func=connect_with_retry,
    prepare_func=prepare_for_play,
    capture_rollout=capture_pixel_rollout,
    pre_shot_grabber=None,
    select_level_func=select_level_in_display,
    video_runner=subprocess.run,
    sleeper=time.sleep,
    shoot_before_capture: bool = True,
    anchor_actions: bool = True,
    physics_capture_v1: bool = False,
    physics_host: str = "127.0.0.1",
    physics_port: int = 2004,
    physics_player_sha256: str | None = None,
    physics_protocol_sha256: str | None = None,
    physics_archive_sha256: str | None = None,
    scenario_manifest: ScenarioManifest | None = None,
    scenario_context_override: Mapping[str, Any] | None = None,
) -> dict:
    if fresh_engine_attempts != 1:
        raise ValueError(
            "fresh_engine_attempts must be 1; frozen collection plans own retry decisions"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = []
    attempts_per_action = 1
    expected_initial_engine_state_identity: str | None = None
    scenario_context = None if scenario_manifest is None else {
        "scenario_manifest_schema": scenario_manifest.schema,
        "scenario_lineage_identity": scenario_manifest.scenario_lineage.identity,
        "declared_initial_engine_state_identity": scenario_manifest.declared_initial_engine_state.identity,
    }
    if scenario_context_override is not None:
        if scenario_context is None:
            raise ValueError("scenario_context_override requires scenario_manifest")
        scenario_context.update(dict(scenario_context_override))
    for index, action in enumerate(actions, start=1):
        prior_invalid_attempts: list[dict] = []
        for attempt in range(1, attempts_per_action + 1):
            slingshot_reference = None
            engine_port_options = {}
            if engine_agent_port is not None:
                engine_port_options["agent_port"] = engine_agent_port
            if engine_game_port is not None:
                engine_port_options["game_port"] = engine_game_port
            if physics_capture_v1:
                engine_port_options["physics_port"] = physics_port
            try:
                engine_process = start_engine_func(game_dir, headless, **engine_port_options)
            except TypeError:
                if not physics_capture_v1:
                    raise
                engine_port_options.pop("physics_port")
                engine_process = start_engine_func(game_dir, headless, **engine_port_options)
            attempt_label = f" for rollout {index}" if attempts_per_action == 1 else f" for rollout {index} attempt {attempt}/{attempts_per_action}"
            print(f"Started engine pid={engine_process.pid}{attempt_label}")
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
                anchored_action = action
                if anchor_actions and action.get("coordinate_frame", "slingshot_relative") == "slingshot_relative":
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
                    shoot_before_capture=shoot_before_capture,
                    anchor_actions=anchor_actions,
                    retry_attempt=attempt,
                    fresh_engine_attempt=attempt if physics_capture_v1 else None,
                    prior_invalid_attempts=prior_invalid_attempts,
                    retryable_recovery_action="fresh_engine_retry" if attempt < attempts_per_action else "fresh_engine_attempts_exhausted",
                    physics_capture_v1=physics_capture_v1,
                    physics_bridge=ScienceBirdsBridge(physics_host, physics_port, timeout=read_timeout) if physics_capture_v1 else None,
                    physics_player_sha256=physics_player_sha256,
                    physics_protocol_sha256=physics_protocol_sha256,
                    physics_archive_sha256=physics_archive_sha256,
                    expected_initial_engine_state_identity=expected_initial_engine_state_identity,
                    scenario_context=scenario_context,
                )
                rollout = partial["rollouts"][0]
                _record_fresh_engine_attempt_metadata(
                    output_dir,
                    rollout,
                    attempt=attempt,
                    slingshot_reference=slingshot_reference,
                    physics_capture_v1=physics_capture_v1,
                )

                rollouts.append(rollout)
                if rollout.get("accepted"):
                    observed_identity = rollout.get("initial_engine_state_identity")
                    if expected_initial_engine_state_identity is None and isinstance(observed_identity, str) and observed_identity:
                        expected_initial_engine_state_identity = observed_identity
                    break
                prior_invalid_attempts.append(_invalid_attempt_reference(rollout))
                artifact_validation = rollout.get("artifact_validation")
                retryable = isinstance(artifact_validation, dict) and artifact_validation.get("retryable")
                if retryable and attempt < attempts_per_action:
                    print(f"Rollout {index} attempt {attempt}/{attempts_per_action} invalid: {rollout.get('invalid_reason')}; retrying")
                    continue
                break
            except PreShotGuardError as exc:
                if exc.rollout is not None:
                    rollout = exc.rollout
                    _record_fresh_engine_attempt_metadata(
                        output_dir,
                        rollout,
                        attempt=attempt,
                        slingshot_reference=slingshot_reference,
                        physics_capture_v1=physics_capture_v1,
                    )
                    rollouts.append(rollout)
                    prior_invalid_attempts.append(_invalid_attempt_reference(rollout))
                    if attempt < attempts_per_action:
                        print(f"Rollout {index} attempt {attempt}/{attempts_per_action} failed: {exc}; retrying")
                        continue
                    break
                if attempt >= attempts_per_action:
                    raise
                print(f"Rollout {index} attempt {attempt}/{attempts_per_action} failed: {exc}; retrying")
            except Exception as exc:
                if attempt >= attempts_per_action:
                    raise
                print(f"Rollout {index} attempt {attempt}/{attempts_per_action} failed: {exc}; retrying")
            finally:
                if bridge is not None:
                    try:
                        bridge.disconnect()
                    finally:
                        stop_owned_engine(engine_process)
                else:
                    stop_owned_engine(engine_process)

    accepted_rollouts = [rollout for rollout in rollouts if rollout.get("accepted")]
    invalid_attempts = [rollout for rollout in rollouts if not rollout.get("accepted")]
    manifest = {
        "capture_source": getattr(capture_rollout, "__name__", "custom-capture"),
        "replay_mode": "fresh-engine-per-rollout",
        "target_fps": target_fps,
        "duration_seconds": duration_seconds,
        "fresh_engine_attempts": attempts_per_action,
        "attempt_count": len(rollouts),
        "accepted_rollout_count": len(accepted_rollouts),
        "rollout_count": len(accepted_rollouts),
        "attempts": rollouts,
        "rollouts": rollouts,
        "accepted_rollouts": accepted_rollouts,
        "invalid_attempts": invalid_attempts,
    }
    if physics_capture_v1:
        if not all(isinstance(value, str) for value in (physics_player_sha256, physics_protocol_sha256, physics_archive_sha256)):
            raise RolloutCollectionError("physics capture provenance is incomplete")
        manifest.update({"capture_contract": _physics_contract_descriptor(physics_player_sha256, physics_protocol_sha256, physics_archive_sha256), "schema_version": "physics_capture_v1", "protocol_version": 1, "player_sha256": physics_player_sha256, "protocol_sha256": physics_protocol_sha256, "archive_sha256": physics_archive_sha256, "sidecar_paths": ["physics_state.jsonl", "physics_events.jsonl"], "physics_state_count": sum(int(item.get("frame_count", 0)) for item in accepted_rollouts), "physics_event_count": sum(int(item.get("physics_event_count", 0)) for item in accepted_rollouts)})
        accepted_initial_identities = [
            rollout.get("initial_engine_state_identity") for rollout in accepted_rollouts
        ]
        verified_initial_identity = (
            accepted_initial_identities[0]
            if accepted_initial_identities
            and all(
                isinstance(identity, str)
                and identity
                and identity == accepted_initial_identities[0]
                for identity in accepted_initial_identities
            )
            else None
        )
        manifest["initial_engine_state_identity"] = verified_initial_identity
        manifest["initial_engine_state_verified"] = verified_initial_identity is not None
        if scenario_context is not None:
            manifest["scenario_context"] = scenario_context
    if ui_level is not None:
        manifest["ui_level"] = ui_level
    exhausted_attempts = [attempt for attempt in invalid_attempts if attempt.get("attempt_status") == "invalid_exhausted"]
    if exhausted_attempts:
        invalid_reasons = sorted({str(attempt.get("invalid_reason") or "unknown") for attempt in exhausted_attempts})
        manifest["collection_status"] = "retry_exhausted"
        manifest["collection_error"] = {
            "error": "fresh_engine_retries_exhausted",
            "accepted_rollout_count": len(accepted_rollouts),
            "requested_rollout_count": len(actions),
            "attempt_count": len(rollouts),
            "invalid_reasons": invalid_reasons,
        }
    manifest.update(write_action_logs(output_dir, rollouts))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if exhausted_attempts:
        raise RolloutCollectionError(
            "fresh-engine retries exhausted: "
            f"accepted {len(accepted_rollouts)}/{len(actions)} requested rollouts after {len(rollouts)} attempts; "
            f"final invalid reason(s): {', '.join(invalid_reasons)}; manifest: {manifest_path}"
        )
    return manifest


def _rewrite_staged_attempt_paths(value: Any, staging_dir: Path, destination: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_staged_attempt_paths(item, staging_dir, destination)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_staged_attempt_paths(item, staging_dir, destination) for item in value]
    if isinstance(value, str):
        try:
            return str(destination / Path(value).relative_to(staging_dir))
        except ValueError:
            return value
    return value


def _json_compatible_action(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_compatible_action(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible_action(item) for item in value]
    return value


def realized_coverage_strata(shot_dir: Path) -> tuple[str, ...]:
    """Classify a validated rollout from authoritative engine records."""
    capture = load_physics_capture(
        Path(shot_dir) / "physics_state.jsonl",
        Path(shot_dir) / "physics_events.jsonl",
    )
    event_types = {event.event_type for event in capture.events}
    strata = {
        stratum
        for event_type, stratum in (
            (EventType.COLLISION, "collision"),
            (EventType.EXPLOSION, "explosion"),
            (EventType.ENTITY_DESTROYED, "destruction"),
            (EventType.PIG_REMOVED, "pig removal"),
            (EventType.LEVEL_CLEARED, "level clear"),
            (EventType.LEVEL_FAILED, "level fail"),
        )
        if event_type in event_types
    }
    if EventType.COLLISION not in event_types:
        strata.add("no-contact/miss")
    if event_types & {EventType.STABLE_ENTERED, EventType.STABLE_EXITED}:
        strata.add("stability transitions")

    support_sets = [
        {edge.support_id for edge in state.support_edges}
        for state in capture.states
    ]
    if any(supports for supports in support_sets):
        strata.add("persistent support")
    if any(before != after for before, after in zip(support_sets, support_sets[1:])):
        strata.add("support change")
    return tuple(sorted(strata))


def collect_fresh_engine_attempt(
    output_root: Path,
    action: dict,
    *,
    attempt_id: str,
    attempt_number: int,
    expected_initial_engine_state_identity: str,
    **fresh_engine_options,
) -> dict[str, Any]:
    """Collect and atomically publish one fresh-engine attempt for a plan runtime."""
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("attempt_id must be a nonempty string")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number <= 0:
        raise ValueError("attempt_number must be a positive integer")
    if (
        not isinstance(expected_initial_engine_state_identity, str)
        or not expected_initial_engine_state_identity
    ):
        raise ValueError("expected_initial_engine_state_identity must be a nonempty string")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".attempt-{attempt_id}.tmp"
    accepted_dir = output_root / "accepted" / attempt_id
    quarantine_dir = output_root / "quarantine" / attempt_id
    if staging_dir.exists() or accepted_dir.exists() or quarantine_dir.exists():
        raise RuntimeError(f"collection attempt path already exists: {attempt_id}")
    staging_dir.mkdir()

    options = dict(fresh_engine_options)
    options.pop("fresh_engine_attempts", None)
    manifest: dict[str, Any] | None = None
    collection_error: Exception | None = None
    try:
        candidate = collect_fresh_engine_rollouts(
            staging_dir,
            [action],
            fresh_engine_attempts=1,
            **options,
        )
        if isinstance(candidate, dict):
            manifest = candidate
    except Exception as exc:
        collection_error = exc

    manifest_path = staging_dir / "manifest.json"
    if manifest is None and manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded_manifest = None
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest
    if manifest is not None and not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rollout = None
    if manifest is not None:
        rollouts = manifest.get("rollouts")
        if isinstance(rollouts, list) and len(rollouts) == 1 and isinstance(rollouts[0], dict):
            rollout = rollouts[0]

    status: str | None = None
    reason: str | None = None
    failure_code: str | None = None
    artifact_validation: dict[str, Any] | None = None
    if rollout is not None and isinstance(rollout.get("artifact_validation"), dict):
        artifact_validation = rollout["artifact_validation"]

    if rollout is None:
        if collection_error is None:
            status = "rejected"
            reason = "collection manifest does not contain exactly one rollout"
            failure_code = "missing_artifact"
        else:
            status = "failed"
            reason = str(collection_error) or collection_error.__class__.__name__
            failure_code = _collection_exception_failure_code(collection_error)
    elif not rollout.get("accepted"):
        status = "rejected"
        if artifact_validation is not None:
            failure_code = next(
                (
                    artifact_validation.get(field)
                    for field in ("failure_code", "invalid_reason", "classification")
                    if isinstance(artifact_validation.get(field), str) and artifact_validation[field]
                ),
                None,
            )
            reason = next(
                (
                    artifact_validation.get(field)
                    for field in ("message", "invalid_reason", "classification")
                    if isinstance(artifact_validation.get(field), str) and artifact_validation[field]
                ),
                None,
            )
        failure_code = failure_code or "artifact_rejected"
        reason = reason or "rollout artifact was rejected"
    elif collection_error is not None:
        status = "failed"
        reason = str(collection_error) or collection_error.__class__.__name__
        failure_code = _collection_exception_failure_code(collection_error)
    else:
        shot_name = rollout.get("name")
        shot_dir = staging_dir / shot_name if isinstance(shot_name, str) and shot_name else None
        metadata = None
        if shot_dir is not None:
            metadata_path = shot_dir / "metadata.json"
            if metadata_path.is_file():
                try:
                    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    loaded_metadata = None
                if isinstance(loaded_metadata, dict):
                    metadata = loaded_metadata

        if artifact_validation is None or artifact_validation.get("accepted") is not True:
            status = "rejected"
            failure_code = "artifact_rejected"
            reason = "accepted rollout is missing successful artifact validation"
        elif metadata is None:
            status = "rejected"
            failure_code = "missing_required_evidence"
            reason = "missing rollout metadata evidence"
        elif metadata.get("capture_contract") != "physics_capture_v1":
            status = "rejected"
            failure_code = "missing_required_evidence"
            reason = "missing collision, contact, and lifecycle evidence"
        elif not _has_complete_strict_semantics(metadata):
            status = "rejected"
            failure_code = "missing_required_evidence"
            reason = "missing terminal or identity rollout evidence"
        elif metadata.get("initial_engine_state_identity") != expected_initial_engine_state_identity:
            status = "rejected"
            failure_code = "initial_engine_state_identity_mismatch"
            reason = "observed initial engine state identity does not match the planned identity"
        else:
            try:
                coverage_strata = realized_coverage_strata(shot_dir)
                accepted_dir.parent.mkdir(parents=True, exist_ok=True)
                for path in staging_dir.rglob("*.json"):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    path.write_text(
                        json.dumps(
                            _rewrite_staged_attempt_paths(payload, staging_dir, accepted_dir),
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                os.replace(staging_dir, accepted_dir)
            except Exception as exc:
                collection_error = exc
                status = "failed"
                reason = str(exc) or exc.__class__.__name__
                failure_code = _collection_exception_failure_code(
                    exc,
                    operation="post_validation_publication",
                )
            else:
                return {
                    "status": "accepted",
                    "reason": None,
                    "failure_code": None,
                    "realized_coverage_strata": list(coverage_strata),
                    "eligible": True,
                    "artifact_path": str(accepted_dir),
                    "quarantine_path": None,
                    "failure_manifest_path": None,
                }

    assert status is not None and reason is not None and failure_code is not None
    permanent_codes = {
        "missing_artifact",
        "artifact_rejected",
        "capture_invalid",
        "missing_required_evidence",
        "initial_engine_state_identity_mismatch",
        "collection_runtime_error",
        "attempt_publication_error",
    }
    permanent = status == "rejected" or failure_code in permanent_codes
    quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
    failure_manifest_path = quarantine_dir / "failure.json"
    failure_manifest = {
        "schema": "collection_attempt_failure_v1",
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "expected_initial_engine_state_identity": expected_initial_engine_state_identity,
        "action": action,
        "status": status,
        "reason": reason,
        "failure_code": failure_code,
        "failure_class": "permanent" if permanent else "transient",
        "retryable": not permanent,
        "retry_decision": "stop" if permanent else "collection_plan",
        "quarantine_path": str(quarantine_dir),
    }
    if artifact_validation is not None:
        failure_manifest["artifact_validation"] = artifact_validation
    if collection_error is not None:
        failure_manifest["exception_type"] = collection_error.__class__.__name__
    (staging_dir / "failure.json").write_text(
        json.dumps(failure_manifest, indent=2),
        encoding="utf-8",
    )
    os.replace(staging_dir, quarantine_dir)
    return {
        "status": status,
        "reason": reason,
        "failure_code": failure_code,
        "realized_coverage_strata": [],
        "eligible": False,
        "artifact_path": None,
        "quarantine_path": str(quarantine_dir),
        "failure_manifest_path": str(failure_manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect diverse high-FPS Science Birds pixel rollouts")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2004)
    parser.add_argument("--physics-host", default="127.0.0.1")
    parser.add_argument("--physics-port", type=int, default=2005)
    parser.add_argument("--game-dir", type=Path, default=ROOT / "sciencebirdsgames" / "Linux")
    parser.add_argument("--engine-agent-port", type=int)
    parser.add_argument("--engine-game-port", type=int)
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
    parser.add_argument("--actions-from-log", type=Path, help="Replay exact actions loaded from a previous action_log.json")
    parser.add_argument("--bidirectional-launches", action="store_true", help="Alternate generated drag-release horizontal signs")
    parser.add_argument("--capture-source", choices=("protocol", "desktop"), default="protocol")
    parser.add_argument("--physics-capture-v1", action="store_true", help="Persist synchronized request-70 physics sidecars")
    parser.add_argument("--physics-player-sha256")
    parser.add_argument("--physics-protocol-sha256")
    parser.add_argument("--physics-archive-sha256")
    parser.add_argument("--fresh-engine-per-rollout", action="store_true")
    parser.add_argument("--collection-plan", type=Path)
    parser.add_argument("--scenario-manifest", type=Path)
    parser.add_argument("--scenario-xml", type=Path)
    parser.add_argument(
        "--scenario-input",
        action="append",
        nargs=4,
        default=[],
        metavar=("SCENARIO_ID", "MANIFEST", "XML", "GAME_DIR"),
        help="Bind one collection-plan scenario to its manifest, XML, and single-level game directory",
    )
    parser.add_argument("--ui-level", type=int, help="Visible level number to enter with xdotool for each fresh rollout")
    parser.add_argument("--ui-settle-seconds", type=float, default=5.0)
    parser.add_argument("--engine-settle-seconds", type=float, default=20.0)
    parser.add_argument("--agent-settle-seconds", type=float, default=45.0)
    parser.add_argument("--fresh-engine-attempts", type=int, default=1)
    return parser


def connect_or_start_engine(args) -> tuple[object, object | None]:
    try:
        return connect_with_retry(args.host, args.port, timeout=args.read_timeout, deadline_seconds=args.connect_timeout), None
    except RuntimeError as first_error:
        game_dir = getattr(args, "game_dir", ROOT / "sciencebirdsgames" / "Linux")
        engine_agent_port = getattr(args, "engine_agent_port", None)
        engine_game_port = getattr(args, "engine_game_port", None)
        try:
            engine_port_options = {}
            if engine_agent_port is not None:
                engine_port_options["agent_port"] = engine_agent_port
            if engine_game_port is not None:
                engine_port_options["game_port"] = engine_game_port
            if getattr(args, "physics_capture_v1", False):
                engine_port_options["physics_port"] = getattr(args, "physics_port", None)
            engine_process = start_engine(game_dir, args.game_headless, **engine_port_options)
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


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        path = write_action_plan(
            args.output_dir,
            count=args.count,
            bidirectional_launches=args.bidirectional_launches,
        )
        print(json.dumps({"action_plan": str(path)}, indent=2))
        return

    if args.collection_plan is not None and not args.fresh_engine_per_rollout:
        print("--collection-plan requires --fresh-engine-per-rollout", file=sys.stderr)
        raise SystemExit(2)

    strict_physics_collection = args.physics_capture_v1 and args.fresh_engine_per_rollout
    has_scenario_manifest = args.scenario_manifest is not None
    has_scenario_xml = args.scenario_xml is not None
    if args.scenario_input and (has_scenario_manifest or has_scenario_xml or args.ui_level is not None):
        print(
            "--scenario-input cannot be combined with --scenario-manifest, --scenario-xml, or --ui-level",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if strict_physics_collection and has_scenario_manifest != has_scenario_xml:
        print(
            "--physics-capture-v1 --fresh-engine-per-rollout requires both --scenario-manifest and --scenario-xml",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if strict_physics_collection and not has_scenario_manifest and not args.scenario_input:
        print(
            "--physics-capture-v1 --fresh-engine-per-rollout requires either --scenario-input or both --scenario-manifest and --scenario-xml",
            file=sys.stderr,
        )
        raise SystemExit(2)
    scenario_manifest = None
    if has_scenario_manifest and has_scenario_xml:
        try:
            scenario_manifest = load_manifest(args.scenario_manifest, args.scenario_xml)
        except (OSError, ValueError) as error:
            print(f"Cannot load scenario manifest/XML: {error}", file=sys.stderr)
            raise SystemExit(2) from None
    scenario_inputs: dict[str, tuple[ScenarioManifest, Path]] = {}
    for scenario_id, manifest_path, xml_path, game_dir in args.scenario_input:
        if not scenario_id or scenario_id in scenario_inputs:
            print("--scenario-input scenario IDs must be nonempty and unique", file=sys.stderr)
            raise SystemExit(2)
        try:
            manifest = load_manifest(Path(manifest_path), Path(xml_path))
            scenario_game_dir = Path(game_dir)
            if not (scenario_game_dir / "game_playing_interface.jar").is_file():
                raise ValueError("GAME_DIR does not contain game_playing_interface.jar")
        except (OSError, ValueError) as error:
            print(f"Cannot load --scenario-input {scenario_id}: {error}", file=sys.stderr)
            raise SystemExit(2) from None
        scenario_inputs[scenario_id] = (manifest, scenario_game_dir)
    if args.fresh_engine_per_rollout and args.collection_plan is None:
        print("--fresh-engine-per-rollout requires --collection-plan", file=sys.stderr)
        raise SystemExit(2)
    if args.fresh_engine_per_rollout and args.fresh_engine_attempts != 1:
        print(
            "--fresh-engine-attempts must be 1; frozen collection plans own retry decisions",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        ensure_output_dir(args.output_dir)
    except OSError as exc:
        print(f"Cannot write output directory {args.output_dir}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    capture_rollout = capture_desktop_rollout if args.capture_source == "desktop" else capture_pixel_rollout
    physics_capture_v1 = bool(args.physics_capture_v1)
    if physics_capture_v1:
        capture_rollout = capture_physics_rollout
        if not all((args.physics_player_sha256, args.physics_protocol_sha256, args.physics_archive_sha256)):
            print("--physics-capture-v1 requires all three physics SHA-256 provenance options", file=sys.stderr)
            raise SystemExit(2)
    pre_shot_grabber = None
    if args.capture_source == "desktop":
        from PIL import ImageGrab

        pre_shot_grabber = ImageGrab.grab
    if args.fresh_engine_per_rollout:
        from scripts.collection_plan import execute_collection_plan, load_collection_plan

        try:
            loaded_plan = load_collection_plan(args.collection_plan)
        except (OSError, ValueError) as error:
            print(f"Cannot load frozen collection plan: {error}", file=sys.stderr)
            raise SystemExit(2) from None

        if scenario_inputs:
            planned_scenario_ids = {scenario.scenario_id for scenario in loaded_plan.plan.scenarios}
            if set(scenario_inputs) != planned_scenario_ids:
                print(
                    "--scenario-input IDs must exactly match the frozen collection plan scenarios",
                    file=sys.stderr,
                )
                raise SystemExit(2)

        def runtime(request):
            selected_manifest = scenario_manifest
            selected_game_dir = args.game_dir
            selected_ui_level = args.ui_level
            if scenario_inputs:
                selected_manifest, selected_game_dir = scenario_inputs[request.scenario_id]
                selected_ui_level = None
            scenario_context_override = None
            if selected_manifest is not None:
                scenario_context_override = {
                    "version_envelope": {
                        "player_sha256": args.physics_player_sha256,
                        "protocol_sha256": args.physics_protocol_sha256,
                        "archive_sha256": args.physics_archive_sha256,
                        "generator_version": _scenario_generation_version(selected_manifest),
                    },
                    "plan_identity": request.plan_identity,
                    "plan_version": request.plan_version,
                    "scenario_id": request.scenario_id,
                    "scenario_identity": request.scenario_identity,
                    "intervention_id": request.intervention_id,
                    "intervention_identity": request.intervention_identity,
                    "attempt_id": request.attempt_id,
                    "attempt_number": request.attempt_number,
                }
            return collect_fresh_engine_attempt(
                args.output_dir,
                _json_compatible_action(request.interface_action),
                attempt_id=request.attempt_id,
                attempt_number=request.attempt_number,
                expected_initial_engine_state_identity=request.expected_initial_engine_state_identity,
                game_dir=selected_game_dir,
                host=args.host,
                port=args.port,
                physics_host=args.physics_host,
                physics_port=args.physics_port,
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
                ui_level=selected_ui_level,
                ui_settle_seconds=args.ui_settle_seconds,
                engine_settle_seconds=args.engine_settle_seconds,
                agent_settle_seconds=args.agent_settle_seconds,
                engine_agent_port=args.engine_agent_port,
                engine_game_port=args.engine_game_port,
                capture_rollout=capture_rollout,
                pre_shot_grabber=pre_shot_grabber,
                shoot_before_capture=physics_capture_v1 or args.capture_source != "desktop",
                anchor_actions=False,
                physics_capture_v1=physics_capture_v1,
                physics_player_sha256=args.physics_player_sha256,
                physics_protocol_sha256=args.physics_protocol_sha256,
                physics_archive_sha256=args.physics_archive_sha256,
                scenario_manifest=selected_manifest,
                scenario_context_override=scenario_context_override,
            )

        report = execute_collection_plan(loaded_plan, runtime, args.output_dir)
        print(
            json.dumps(
                {
                    "collection_plan": str(args.collection_plan),
                    "accepted_count": report["accepted_count"],
                    "rejected_count": report["rejected_count"],
                    "failed_count": report["failed_count"],
                },
                indent=2,
            )
        )
        return

    actions_from_log = args.actions_from_log is not None
    actions = (
        load_actions_from_action_log(args.actions_from_log)
        if actions_from_log
        else generate_diverse_drag_release_actions(
            count=args.count,
            bidirectional_launches=args.bidirectional_launches,
        )
    )

    if args.start_engine:
        engine_process = start_engine(
            args.game_dir,
            args.game_headless,
            agent_port=args.engine_agent_port,
            game_port=args.engine_game_port,
            physics_port=args.physics_port if physics_capture_v1 else None,
        )
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
            shoot_before_capture=physics_capture_v1 or args.capture_source != "desktop",
            anchor_actions=not actions_from_log,
            physics_capture_v1=physics_capture_v1,
            physics_bridge=ScienceBirdsBridge(args.physics_host, args.physics_port, timeout=args.read_timeout) if physics_capture_v1 else None,
            physics_player_sha256=args.physics_player_sha256,
            physics_protocol_sha256=args.physics_protocol_sha256,
            physics_archive_sha256=args.physics_archive_sha256,
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
