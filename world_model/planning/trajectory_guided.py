"""Trajectory-preview-guided actions from deployment-visible pig geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from world_model.planning.gameplay import SlingshotAction, SlingshotActionBounds


TRAJECTORY_MAX_LAUNCH_SPEED = 10.0
TRAJECTORY_LAUNCH_GRAVITY = 0.48
TRAJECTORY_DRAG_RADIUS_WORLD = 1.0
TRAJECTORY_TIME_STEP_SECONDS = 0.02
TRAJECTORY_STEPS = 300


class TrajectoryGuidedAimError(ValueError):
    """The live deployment observation cannot resolve the frozen guided action."""


@dataclass(frozen=True, slots=True)
class TrajectoryGuidedAim:
    action: SlingshotAction
    target_id: str
    target_label: str
    target_canvas: tuple[float, float]
    predicted_canvas_y: float
    predicted_miss_pixels: float
    arc: str
    aim_point: str
    target_polygon_bounds: tuple[float, float, float, float]

    def evidence(self) -> dict[str, Any]:
        return {
            "schema": "trajectory_guided_direct_pig_resolution_v1",
            "target_id": self.target_id,
            "target_label": self.target_label,
            "target_canvas": list(self.target_canvas),
            "predicted_canvas_y": self.predicted_canvas_y,
            "predicted_miss_pixels": self.predicted_miss_pixels,
            "trajectory_arc": self.arc,
            "aim_point": self.aim_point,
            "target_polygon_bounds": list(self.target_polygon_bounds),
            "preview_model": {
                "maximum_launch_speed": TRAJECTORY_MAX_LAUNCH_SPEED,
                "launch_gravity_scale": TRAJECTORY_LAUNCH_GRAVITY,
                "drag_radius_world": TRAJECTORY_DRAG_RADIUS_WORLD,
                "fixed_step_seconds": TRAJECTORY_TIME_STEP_SECONDS,
                "preview_steps": TRAJECTORY_STEPS,
            },
        }


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x), float(y)
    return None


def _target(label: str, target_id: str, vertices: Any) -> dict[str, Any] | None:
    if "pig" not in label.lower() or not isinstance(vertices, list):
        return None
    points = [item for item in (_point(value) for value in vertices) if item is not None]
    if not points:
        return None
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    return {
        "id": target_id,
        "label": label,
        "center": ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
        "bounds": (min(xs), min(ys), max(xs), max(ys)),
    }


def visible_pig_targets(symbolic_state: Any) -> tuple[dict[str, Any], ...]:
    targets = []
    if not isinstance(symbolic_state, list):
        return ()
    for item in symbolic_state:
        if not isinstance(item, Mapping):
            continue
        direct = _target(
            str(item.get("type") or ""),
            str(item.get("id") or ""),
            item.get("vertices"),
        )
        if direct is not None:
            targets.append(direct)
        features = item.get("features")
        if not isinstance(features, list):
            continue
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            properties = feature.get("properties")
            geometry = feature.get("geometry")
            if not isinstance(properties, Mapping) or not isinstance(geometry, Mapping):
                continue
            coordinates = geometry.get("coordinates")
            vertices = (
                coordinates[0]
                if geometry.get("type") == "Polygon"
                and isinstance(coordinates, list)
                and coordinates
                else None
            )
            target = _target(
                str(properties.get("label") or ""),
                str(properties.get("id") or ""),
                vertices,
            )
            if target is not None:
                targets.append(target)
    return tuple(sorted(targets, key=lambda item: (item["center"], item["id"])))


def _preview_at_target_x(
    *,
    sling_x: float,
    sling_y: float,
    pixels_per_world_unit: float,
    drag_x: int,
    drag_y: int,
    target_x: float,
) -> float | None:
    pull = math.hypot(drag_x, drag_y)
    if pull == 0:
        return None
    radius = pixels_per_world_unit * TRAJECTORY_DRAG_RADIUS_WORLD
    capped_pull = min(pull, radius)
    release_x = sling_x + drag_x / pull * capped_pull
    release_y = sling_y + drag_y / pull * capped_pull
    speed = (
        capped_pull / radius
        * TRAJECTORY_MAX_LAUNCH_SPEED
        * pixels_per_world_unit
    )
    velocity_x = -drag_x / pull * speed
    velocity_y = -drag_y / pull * speed
    if velocity_x <= 0 or target_x <= release_x:
        return None
    gravity_y = (
        9.81 * TRAJECTORY_LAUNCH_GRAVITY * pixels_per_world_unit
    )
    previous_x, previous_y = release_x, release_y
    for _ in range(TRAJECTORY_STEPS):
        velocity_y += gravity_y * TRAJECTORY_TIME_STEP_SECONDS
        current_x = previous_x + velocity_x * TRAJECTORY_TIME_STEP_SECONDS
        current_y = previous_y + velocity_y * TRAJECTORY_TIME_STEP_SECONDS
        if previous_x <= target_x <= current_x:
            fraction = (target_x - previous_x) / (current_x - previous_x)
            return previous_y + fraction * (current_y - previous_y)
        previous_x, previous_y = current_x, current_y
    return None


def aim_directly_at_visible_pig(
    symbolic_state: Any,
    slingshot_reference: Mapping[str, float],
    bounds: SlingshotActionBounds,
    *,
    target_rank: int,
    arc: str,
    aim_point: str,
    tap_time_ms: int,
) -> TrajectoryGuidedAim:
    if arc != "low":
        raise TrajectoryGuidedAimError("only the direct low trajectory arc is supported")
    targets = visible_pig_targets(symbolic_state)
    if not 0 <= target_rank < len(targets):
        raise TrajectoryGuidedAimError("the frozen visible-pig target is unavailable")
    target = targets[target_rank]
    if aim_point != "visible_polygon_upper_edge":
        raise TrajectoryGuidedAimError("the frozen pig aim point is unsupported")
    target_x = target["center"][0]
    target_y = target["bounds"][1]
    sling_x = float(slingshot_reference["canvasX"])
    sling_y = float(slingshot_reference["canvasY"])
    pixels_per_world_unit = float(slingshot_reference["pixelsPerWorldUnit"])
    if pixels_per_world_unit <= 0 or target_x <= sling_x:
        raise TrajectoryGuidedAimError("the visible pig is outside direct trajectory range")
    candidates = []
    for drag_x in range(bounds.drag_x[0], bounds.drag_x[1] + 1):
        for drag_y in range(max(0, bounds.drag_y[0]), bounds.drag_y[1] + 1):
            action = SlingshotAction(drag_x, drag_y, tap_time_ms)
            if not bounds.contains(action):
                continue
            predicted_y = _preview_at_target_x(
                sling_x=sling_x,
                sling_y=sling_y,
                pixels_per_world_unit=pixels_per_world_unit,
                drag_x=drag_x,
                drag_y=drag_y,
                target_x=target_x,
            )
            if predicted_y is None:
                continue
            angle = math.atan2(drag_y, -drag_x)
            if angle > math.pi / 4:
                continue
            candidates.append(
                (
                    abs(predicted_y - target_y),
                    math.hypot(drag_x, drag_y),
                    angle,
                    action,
                    predicted_y,
                )
            )
    if not candidates:
        raise TrajectoryGuidedAimError("no legal low trajectory reaches the visible pig")
    miss, _pull, _angle, action, predicted_y = min(candidates)
    return TrajectoryGuidedAim(
        action=action,
        target_id=target["id"],
        target_label=target["label"],
        target_canvas=(target_x, target_y),
        predicted_canvas_y=predicted_y,
        predicted_miss_pixels=miss,
        arc=arc,
        aim_point=aim_point,
        target_polygon_bounds=target["bounds"],
    )
