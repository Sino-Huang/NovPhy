"""Camera-aligned preparation for screen-coordinate Science Birds shots."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, Protocol


STARTUP_SPEED = 50
DEFAULT_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_ANCHOR_TOLERANCE_PIXELS = 2.0
TRAJECTORY_SLINGSHOT_HEIGHT_WORLD = 2.055
TRAJECTORY_SLING_REFERENCE_X_FROM_MIN_WORLD = 0.335
TRAJECTORY_SLING_REFERENCE_Y_FROM_TOP_WORLD = 0.25


class SlingshotReadinessError(RuntimeError):
    """The camera projection could not be proven safe for a screen shot."""

    def __init__(self, reason: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = dict(evidence)


class SlingshotShotClient(Protocol):
    """Small common interface used by readiness and prepared-shot execution."""

    def set_simulation_speed(self, speed: int) -> Any: ...

    def fully_zoom_out(self) -> Any: ...

    def read_symbolic_state(self) -> Any: ...

    def send_prepared_shot(
        self,
        command: Mapping[str, Any],
        *,
        fast: bool,
        record_ground_truth: bool,
        ground_truth_frequency: int,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class _WebUIBridgeAdapter:
    raw: Any

    def set_simulation_speed(self, speed: int) -> Any:
        return self.raw.set_speed(speed)

    def fully_zoom_out(self) -> Any:
        return self.raw.fully_zoom_out()

    def read_symbolic_state(self) -> Any:
        return self.raw.get_symbolic_state_without_screenshot()

    def send_prepared_shot(
        self,
        command: Mapping[str, Any],
        *,
        fast: bool,
        record_ground_truth: bool,
        ground_truth_frequency: int,
    ) -> Any:
        if record_ground_truth:
            return self.raw.shoot_and_record_ground_truth(
                int(command["x"]),
                int(command["y"]),
                tap_time=int(command.get("tapTime", 0)),
                release_time=int(command.get("releaseTime", 0)),
                frequency=int(ground_truth_frequency),
            )
        return self.raw.shoot(
            int(command["x"]),
            int(command["y"]),
            tap_time=int(command.get("tapTime", 0)),
            fast=bool(fast),
            release_time=int(command.get("releaseTime", 0)),
        )


@dataclass(frozen=True, slots=True)
class _LegacyAgentClientAdapter:
    raw: Any

    def set_simulation_speed(self, speed: int) -> Any:
        return self.raw.set_game_simulation_speed(speed)

    def fully_zoom_out(self) -> Any:
        return self.raw.fully_zoom_out()

    def read_symbolic_state(self) -> Any:
        return self.raw.get_symbolic_state_without_screenshot()

    def send_prepared_shot(
        self,
        command: Mapping[str, Any],
        *,
        fast: bool,
        record_ground_truth: bool,
        ground_truth_frequency: int,
    ) -> Any:
        x = int(command["x"])
        y = int(command["y"])
        release_time = int(command.get("releaseTime", 0))
        tap_time = int(command.get("tapTime", 0))
        if record_ground_truth:
            return self.raw.shoot_and_record_ground_truth(
                x, y, release_time, tap_time, int(ground_truth_frequency)
            )
        transport = self.raw.fast_shoot if fast else self.raw.shoot
        return transport(x, y, release_time, tap_time, False)


def adapt_slingshot_client(client: Any) -> SlingshotShotClient:
    """Adapt either repository socket client to the readiness interface."""
    if all(
        hasattr(client, name)
        for name in (
            "set_simulation_speed",
            "fully_zoom_out",
            "read_symbolic_state",
            "send_prepared_shot",
        )
    ):
        return client
    if hasattr(client, "set_speed"):
        return _WebUIBridgeAdapter(client)
    if hasattr(client, "set_game_simulation_speed"):
        return _LegacyAgentClientAdapter(client)
    raise TypeError("Unsupported Science Birds socket client")


def _point_xy(point: Any) -> tuple[float, float] | None:
    if isinstance(point, Mapping):
        x, y = point.get("x"), point.get("y")
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        x, y = point[0], point[1]
    else:
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return float(x), float(y)
    return None


def _slingshot_bounds(
    symbolic_state: Any,
) -> tuple[float, float, float, float] | None:
    if not isinstance(symbolic_state, list):
        return None
    for item in symbolic_state:
        if not isinstance(item, Mapping):
            continue
        vertices = item.get("vertices") if item.get("type") == "Slingshot" else None
        features = item.get("features")
        if vertices is None and isinstance(features, list):
            for feature in features:
                if not isinstance(feature, Mapping):
                    continue
                properties = feature.get("properties")
                geometry = feature.get("geometry")
                if not isinstance(properties, Mapping) or not isinstance(geometry, Mapping):
                    continue
                if properties.get("label") != "Slingshot" or geometry.get("type") != "Polygon":
                    continue
                coordinates = geometry.get("coordinates")
                if isinstance(coordinates, list) and coordinates and isinstance(coordinates[0], list):
                    vertices = coordinates[0]
                    break
        if not vertices:
            continue
        points = [point for point in (_point_xy(value) for value in vertices) if point is not None]
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bounds = min(xs), max(xs), min(ys), max(ys)
        if bounds[1] > bounds[0] and bounds[3] > bounds[2]:
            return bounds
    return None


def slingshot_observation_from_symbolic_state(
    symbolic_state: Any, frame_height: int
) -> dict[str, float] | None:
    """Return the screen anchor and projection scale from request-62 geometry."""
    bounds = _slingshot_bounds(symbolic_state)
    if bounds is None:
        return None
    min_x, _, min_y, max_y = bounds
    pixels_per_world_unit = (max_y - min_y) / TRAJECTORY_SLINGSHOT_HEIGHT_WORLD
    canvas_x = min_x + TRAJECTORY_SLING_REFERENCE_X_FROM_MIN_WORLD * pixels_per_world_unit
    canvas_y = min_y + TRAJECTORY_SLING_REFERENCE_Y_FROM_TOP_WORLD * pixels_per_world_unit
    return {
        "gameX": canvas_x,
        "gameY": max(0.0, float(frame_height) - canvas_y),
        "canvasX": canvas_x,
        "canvasY": canvas_y,
        "pixelsPerWorldUnit": pixels_per_world_unit,
    }


def _normal_action_release_time(action: Mapping[str, Any]) -> int:
    for key in ("holdTime", "releaseTime", "release_time"):
        if key in action:
            return int(action[key])
    return 600


def _anchored_action_and_command(
    action: Mapping[str, Any], reference: Mapping[str, float], frame_height: int
) -> tuple[dict[str, Any], dict[str, int]]:
    anchored = dict(action)
    release = anchored.get("drag_release", anchored.get("release"))
    if not isinstance(release, (list, tuple)) or len(release) < 2:
        raise ValueError("drag_release or release must contain x and y values")
    coordinate_frame = anchored.get("coordinate_frame", "slingshot_relative")
    if coordinate_frame == "slingshot_relative":
        start_x = int(reference["gameX"])
        start_y = int(reference["gameY"])
        anchored["drag_start"] = [start_x, start_y]
        anchored["slingshot_reference"] = dict(reference)
        dx, dy = int(release[0]), int(release[1])
        game_x, game_y = start_x + dx, start_y - dy
    elif coordinate_frame == "absolute":
        game_x, game_y = int(release[0]), int(release[1])
    else:
        raise ValueError("coordinate_frame must be slingshot_relative or absolute")
    command = {
        "x": game_x,
        "y": max(0, int(frame_height) - 1 - game_y),
        "gameX": game_x,
        "gameY": game_y,
        "tapTime": int(anchored.get("tapTime", anchored.get("tap_time", 0))),
        "releaseTime": _normal_action_release_time(anchored),
    }
    return anchored, command


def _alignment_check(
    observed: Mapping[str, float],
    expected: Mapping[str, int | float] | None,
    tolerance: float,
) -> dict[str, Any]:
    if expected is None:
        return {"required": False, "matched": True, "tolerance_pixels": tolerance}
    deltas = {
        key: abs(float(observed[key]) - float(expected[key]))
        for key in ("canvasX", "canvasY")
        if key in expected
    }
    scale_delta = None
    if "pixelsPerWorldUnit" in expected:
        scale_delta = abs(
            float(observed["pixelsPerWorldUnit"])
            - float(expected["pixelsPerWorldUnit"])
        )
    matched = (
        set(deltas) == {"canvasX", "canvasY"}
        and all(delta <= tolerance for delta in deltas.values())
        and (scale_delta is None or scale_delta <= tolerance)
    )
    return {
        "required": True,
        "matched": matched,
        "tolerance_pixels": tolerance,
        "expected": dict(expected),
        "observed": dict(observed),
        "absolute_delta": deltas,
        "pixels_per_world_unit_delta": scale_delta,
    }


def _stable_observation(
    client: SlingshotShotClient,
    *,
    frame_height: int,
    deadline: float,
    poll_interval: float,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> tuple[dict[str, float], int]:
    previous: dict[str, float] | None = None
    observation_count = 0
    while clock() < deadline:
        current = slingshot_observation_from_symbolic_state(
            client.read_symbolic_state(), frame_height
        )
        observation_count += 1
        if current is not None and current == previous:
            return current, observation_count
        previous = current
        sleeper(poll_interval)
    reason = "slingshot_missing" if previous is None else "slingshot_stability_timeout"
    raise SlingshotReadinessError(reason, {"observation_count": observation_count})


@dataclass(frozen=True, slots=True)
class PreparedScreenShot:
    """A screen command that may be sent only after camera readiness succeeded."""

    client: SlingshotShotClient
    action: Mapping[str, Any]
    socket_command: Mapping[str, Any]
    slingshot: Mapping[str, float]
    evidence: Mapping[str, Any]
    fast: bool = False
    record_ground_truth: bool = False
    ground_truth_frequency: int = 1

    def execute(self) -> Any:
        return self.client.send_prepared_shot(
            self.socket_command,
            fast=self.fast,
            record_ground_truth=self.record_ground_truth,
            ground_truth_frequency=self.ground_truth_frequency,
        )


def prepare_screen_shot(
    client: Any,
    action: Mapping[str, Any] | Callable[[Mapping[str, float]], Mapping[str, Any]],
    *,
    frame_height: int = 480,
    execution_speed: int = 1,
    frozen_socket_command: Mapping[str, Any] | None = None,
    retained_anchor: Mapping[str, int | float] | None = None,
    fast: bool = False,
    record_ground_truth: bool = False,
    ground_truth_frequency: int = 1,
    prepare_level: Callable[[], Any] | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    anchor_tolerance: float = DEFAULT_ANCHOR_TOLERANCE_PIXELS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> PreparedScreenShot:
    """Prepare and validate one live-relative or exact frozen screen shot."""
    adapted = adapt_slingshot_client(client)
    requested_execution_speed = int(execution_speed)
    bounded_execution_speed = min(STARTUP_SPEED, max(1, requested_execution_speed))
    bounded_startup_speed = STARTUP_SPEED
    deadline = clock() + float(timeout)
    evidence: dict[str, Any] = {
        "schema": "slingshot_readiness_v1",
        "startup_speed": bounded_startup_speed,
        "requested_execution_speed": requested_execution_speed,
        "execution_speed": bounded_execution_speed,
        "poll_interval_seconds": float(poll_interval),
        "timeout_seconds": float(timeout),
        "frozen_command": frozen_socket_command is not None,
        "alignment_contract": {
            "stable_observations_required_per_phase": 2,
            "anchor_tolerance_pixels": float(anchor_tolerance),
            "frozen_commands_are_reanchored": False,
        },
    }
    try:
        adapted.set_simulation_speed(bounded_startup_speed)
        if frozen_socket_command is not None and retained_anchor is None:
            raise SlingshotReadinessError("retained_anchor_missing", evidence)
        if prepare_level is not None:
            prepare_level()
        adapted.fully_zoom_out()
        startup_observation, startup_count = _stable_observation(
            adapted,
            frame_height=frame_height,
            deadline=deadline,
            poll_interval=float(poll_interval),
            clock=clock,
            sleeper=sleeper,
        )
        startup_alignment = _alignment_check(
            startup_observation, retained_anchor, float(anchor_tolerance)
        )
        evidence["startup"] = {
            "observation_count": startup_count,
            "stable_slingshot": dict(startup_observation),
            "alignment": startup_alignment,
        }
        if not startup_alignment["matched"]:
            raise SlingshotReadinessError("retained_anchor_mismatch", evidence)

        adapted.set_simulation_speed(bounded_execution_speed)
        execution_observation, execution_count = _stable_observation(
            adapted,
            frame_height=frame_height,
            deadline=deadline,
            poll_interval=float(poll_interval),
            clock=clock,
            sleeper=sleeper,
        )
        execution_alignment = _alignment_check(
            execution_observation, retained_anchor, float(anchor_tolerance)
        )
        projection_unchanged = execution_observation == startup_observation
        evidence["execution"] = {
            "observation_count": execution_count,
            "stable_slingshot": dict(execution_observation),
            "alignment": execution_alignment,
            "unchanged_from_startup": projection_unchanged,
        }
        if not execution_alignment["matched"]:
            raise SlingshotReadinessError("retained_anchor_mismatch", evidence)
        if not projection_unchanged:
            raise SlingshotReadinessError(
                "projection_changed_after_execution_speed", evidence
            )

        resolved_action = action(execution_observation) if callable(action) else action
        if frozen_socket_command is None:
            prepared_action, command = _anchored_action_and_command(
                resolved_action, execution_observation, frame_height
            )
        else:
            prepared_action = dict(resolved_action)
            command = dict(frozen_socket_command)
            for field in ("x", "y"):
                if field not in command:
                    raise ValueError(f"frozen socket command is missing {field}")
        evidence["status"] = "ready"
        evidence["anchoring"] = (
            "retained_exact_socket_command"
            if frozen_socket_command is not None
            else "stabilized_live_slingshot"
        )
        evidence["socket_command"] = dict(command)
        return PreparedScreenShot(
            client=adapted,
            action=prepared_action,
            socket_command=command,
            slingshot=execution_observation,
            evidence=evidence,
            fast=bool(fast),
            record_ground_truth=bool(record_ground_truth),
            ground_truth_frequency=int(ground_truth_frequency),
        )
    except Exception as error:
        try:
            adapted.set_simulation_speed(bounded_execution_speed)
            evidence["execution_speed_restored_after_failure"] = True
        except Exception as restore_error:
            evidence["execution_speed_restored_after_failure"] = False
            evidence["speed_restoration_error"] = str(restore_error)
        if isinstance(error, SlingshotReadinessError):
            error.evidence.update(evidence)
            raise
        evidence["status"] = "failed"
        evidence["error"] = str(error)
        raise SlingshotReadinessError("preparation_failed", evidence) from error
