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
    obstacle_clearance: str
    cleared_obstacle_ids: tuple[str, ...]
    bird_radius_pixels: float

    def evidence(self) -> dict[str, Any]:
        return {
            "schema": "trajectory_guided_direct_pig_resolution_v2",
            "target_id": self.target_id,
            "target_label": self.target_label,
            "target_canvas": list(self.target_canvas),
            "predicted_canvas_y": self.predicted_canvas_y,
            "predicted_miss_pixels": self.predicted_miss_pixels,
            "trajectory_arc": self.arc,
            "aim_point": self.aim_point,
            "target_polygon_bounds": list(self.target_polygon_bounds),
            "obstacle_clearance": self.obstacle_clearance,
            "cleared_obstacle_ids": list(self.cleared_obstacle_ids),
            "bird_radius_pixels": self.bird_radius_pixels,
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


def _polygon(label: str, object_id: str, vertices: Any) -> dict[str, Any] | None:
    if not isinstance(vertices, list):
        return None
    points = [item for item in (_point(value) for value in vertices) if item is not None]
    if not points:
        return None
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    return {
        "id": object_id,
        "label": label,
        "center": ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0),
        "bounds": (min(xs), min(ys), max(xs), max(ys)),
    }


def _visible_polygons(symbolic_state: Any) -> tuple[dict[str, Any], ...]:
    polygons = []
    if not isinstance(symbolic_state, list):
        return ()
    for item in symbolic_state:
        if not isinstance(item, Mapping):
            continue
        direct = _polygon(
            str(item.get("type") or ""),
            str(item.get("id") or ""),
            item.get("vertices"),
        )
        if direct is not None:
            polygons.append(direct)
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
            polygon = _polygon(
                str(properties.get("label") or ""),
                str(properties.get("id") or ""),
                vertices,
            )
            if polygon is not None:
                polygons.append(polygon)
    return tuple(polygons)


def visible_pig_targets(symbolic_state: Any) -> tuple[dict[str, Any], ...]:
    targets = [
        polygon
        for polygon in _visible_polygons(symbolic_state)
        if "pig" in polygon["label"].lower()
    ]
    return tuple(sorted(targets, key=lambda item: (item["center"], item["id"])))


def _preview_points(
    *,
    sling_x: float,
    sling_y: float,
    pixels_per_world_unit: float,
    drag_x: int,
    drag_y: int,
) -> tuple[tuple[float, float], ...]:
    pull = math.hypot(drag_x, drag_y)
    if pull == 0:
        return ()
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
    if velocity_x <= 0:
        return ()
    gravity_y = 9.81 * TRAJECTORY_LAUNCH_GRAVITY * pixels_per_world_unit
    points = [(release_x, release_y)]
    for _ in range(TRAJECTORY_STEPS):
        velocity_y += gravity_y * TRAJECTORY_TIME_STEP_SECONDS
        previous_x, previous_y = points[-1]
        points.append((
            previous_x + velocity_x * TRAJECTORY_TIME_STEP_SECONDS,
            previous_y + velocity_y * TRAJECTORY_TIME_STEP_SECONDS,
        ))
    return tuple(points)


def _preview_at_target_x(
    *,
    sling_x: float,
    sling_y: float,
    pixels_per_world_unit: float,
    drag_x: int,
    drag_y: int,
    target_x: float,
) -> float | None:
    points = _preview_points(
        sling_x=sling_x,
        sling_y=sling_y,
        pixels_per_world_unit=pixels_per_world_unit,
        drag_x=drag_x,
        drag_y=drag_y,
    )
    for (previous_x, previous_y), (current_x, current_y) in zip(
        points, points[1:]
    ):
        if previous_x <= target_x <= current_x:
            fraction = (target_x - previous_x) / (current_x - previous_x)
            return previous_y + fraction * (current_y - previous_y)
    return None


def _segment_intersects_bounds(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    lower = 0.0
    upper = 1.0
    for origin, delta, minimum, maximum in (
        (start[0], end[0] - start[0], bounds[0], bounds[2]),
        (start[1], end[1] - start[1], bounds[1], bounds[3]),
    ):
        if delta == 0:
            if origin < minimum or origin > maximum:
                return False
            continue
        entry = (minimum - origin) / delta
        exit_ = (maximum - origin) / delta
        if entry > exit_:
            entry, exit_ = exit_, entry
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if lower > upper:
            return False
    return True


def _trajectory_is_clear(
    points: tuple[tuple[float, float], ...],
    obstacles: tuple[dict[str, Any], ...],
    *,
    bird_radius: float,
    horizon_x: float,
) -> bool:
    inflated = [
        (
            obstacle["bounds"][0] - bird_radius,
            obstacle["bounds"][1] - bird_radius,
            obstacle["bounds"][2] + bird_radius,
            obstacle["bounds"][3] + bird_radius,
        )
        for obstacle in obstacles
    ]
    for start, end in zip(points, points[1:]):
        if start[0] >= horizon_x:
            break
        clipped_end = end
        if end[0] > horizon_x:
            fraction = (horizon_x - start[0]) / (end[0] - start[0])
            clipped_end = (
                horizon_x,
                start[1] + fraction * (end[1] - start[1]),
            )
        if any(
            _segment_intersects_bounds(start, clipped_end, bounds)
            for bounds in inflated
        ):
            return False
    return True


def _bird_radius(
    polygons: tuple[dict[str, Any], ...], sling_x: float, sling_y: float
) -> float:
    birds = [item for item in polygons if "bird" in item["label"].lower()]
    if not birds:
        raise TrajectoryGuidedAimError("the live bird polygon is unavailable")
    bird = min(
        birds,
        key=lambda item: math.hypot(
            item["center"][0] - sling_x, item["center"][1] - sling_y
        ),
    )
    width = bird["bounds"][2] - bird["bounds"][0]
    height = bird["bounds"][3] - bird["bounds"][1]
    radius = max(width, height) / 2.0
    if radius <= 0:
        raise TrajectoryGuidedAimError("the live bird polygon has no volume")
    return radius


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
    if arc != "lowest_clear":
        raise TrajectoryGuidedAimError(
            "only the frozen lowest-clear trajectory rule is supported"
        )
    polygons = _visible_polygons(symbolic_state)
    targets = tuple(sorted(
        (item for item in polygons if "pig" in item["label"].lower()),
        key=lambda item: (item["center"], item["id"]),
    ))
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
    bird_radius = _bird_radius(polygons, sling_x, sling_y)
    obstacle_tokens = ("block", "ice", "platform", "stone", "wood")
    horizon_x = target["bounds"][0] - bird_radius
    obstacles = tuple(
        item
        for item in polygons
        if any(token in item["label"].lower() for token in obstacle_tokens)
        and item["bounds"][2] + bird_radius >= sling_x
        and item["bounds"][0] - bird_radius <= horizon_x
    )
    hit_tolerance = (
        (target["bounds"][3] - target["bounds"][1]) / 2.0 + bird_radius
    )
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
            miss = abs(predicted_y - target_y)
            if miss > hit_tolerance:
                continue
            angle = math.atan2(drag_y, -drag_x)
            points = _preview_points(
                sling_x=sling_x,
                sling_y=sling_y,
                pixels_per_world_unit=pixels_per_world_unit,
                drag_x=drag_x,
                drag_y=drag_y,
            )
            if not _trajectory_is_clear(
                points,
                obstacles,
                bird_radius=bird_radius,
                horizon_x=horizon_x,
            ):
                continue
            candidates.append(
                (
                    0 if angle <= math.pi / 4 else 1,
                    miss,
                    math.hypot(drag_x, drag_y),
                    angle,
                    drag_x,
                    drag_y,
                    action,
                    predicted_y,
                )
            )
    if not candidates:
        raise TrajectoryGuidedAimError(
            "no legal bird-volume-clear trajectory reaches the visible pig"
        )
    (
        arc_rank,
        miss,
        _pull,
        _angle,
        _drag_x,
        _drag_y,
        action,
        predicted_y,
    ) = min(candidates)
    return TrajectoryGuidedAim(
        action=action,
        target_id=target["id"],
        target_label=target["label"],
        target_canvas=(target_x, target_y),
        predicted_canvas_y=predicted_y,
        predicted_miss_pixels=miss,
        arc="low" if arc_rank == 0 else "high",
        aim_point=aim_point,
        target_polygon_bounds=target["bounds"],
        obstacle_clearance="bird_volume_swept_clear",
        cleared_obstacle_ids=tuple(sorted({item["id"] for item in obstacles})),
        bird_radius_pixels=bird_radius,
    )
