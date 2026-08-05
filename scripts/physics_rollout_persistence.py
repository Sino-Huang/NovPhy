from __future__ import annotations

from collections.abc import Callable
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import BinaryIO, Mapping, Protocol

from scripts.physics_capture_contract import PhysicsContractError
from scripts.physics_capture_parsing import _parse_event, _parse_header, _parse_state
from scripts.physics_capture_types import StateFrame
from scripts.physics_rollout_contract import (
    CaptureBounds,
    CaptureProvenance,
    JsonObject,
    JsonValue,
    MAX_EVENT_RECORDS,
    MAX_STATE_RECORDS,
    MAX_TOTAL_BYTES,
    PersistenceErrorCode,
    PhysicsCaptureBridge,
    PhysicsPersistenceError,
    StreamProgress,
)


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


def _encoded_record(record: JsonObject) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_bounded_record(
    stream: BinaryIO,
    digest: _Digest,
    encoded: bytes,
    progress: StreamProgress,
) -> StreamProgress:
    total_bytes = progress.total_bytes + len(encoded)
    if total_bytes > MAX_TOTAL_BYTES:
        raise PhysicsPersistenceError(
            PersistenceErrorCode.BYTE_LIMIT,
            f"sidecar byte limit is {MAX_TOTAL_BYTES}",
        )
    stream.write(encoded)
    digest.update(encoded)
    return StreamProgress(progress.state_count, progress.event_count, total_bytes)


def _state_header(first_state: StateFrame, coordinates: JsonValue, shot_id: str) -> JsonObject:
    return {
        "record_type": "state_header",
        "schema_version": first_state.clock.schema_version,
        "capture_id": first_state.clock.capture_id,
        "shot_id": shot_id,
        "sequence": max(0, first_state.clock.sequence - 1),
        "render_frame": max(0, first_state.clock.render_frame - 1),
        "render_time": max(0.0, first_state.clock.render_time),
        "fixed_step": max(0, first_state.clock.fixed_step - 1),
        "fixed_time": max(0.0, first_state.clock.fixed_time),
        "coordinates": coordinates,
        "capture_status": "complete",
        "state_sidecar": "physics_state.jsonl",
        "event_sidecar": "physics_events.jsonl",
        "support_rule": {
            "name": "support_v1",
            "minimum_consecutive_fixed_steps": 2,
            "minimum_abs_normal_y": 0.5,
            "minimum_vertical_center_delta": 0.0001,
            "include_triggers": False,
            "missing_contact_policy": "no_support",
            "static_entity_id_prefix": "world:static:",
        },
        "event_taxonomy": [
            "bird_launched", "collision", "explosion", "entity_destroyed", "pig_removed",
            "bird_exhausted", "stable_entered", "stable_exited", "level_cleared", "level_failed",
        ],
        "capture_limits": {
            "max_state_records": MAX_STATE_RECORDS,
            "max_event_records": MAX_EVENT_RECORDS,
            "max_total_bytes": MAX_TOTAL_BYTES,
        },
    }


def _png_dimensions(png: bytes) -> tuple[int, int]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(png)) as image:
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            return image.size
    except (OSError, ValueError) as error:
        raise PhysicsPersistenceError(
            PersistenceErrorCode.MALFORMED_CAPTURE,
            "request 70 returned an invalid PNG",
        ) from error


def _materialize_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        materialized: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PhysicsPersistenceError(
                    PersistenceErrorCode.MALFORMED_CAPTURE,
                    "request-70 JSON object key is not a string",
                )
            materialized[key] = _materialize_json(item)
        return materialized
    if isinstance(value, (list, tuple)):
        return [_materialize_json(item) for item in value]
    raise PhysicsPersistenceError(
        PersistenceErrorCode.MALFORMED_CAPTURE,
        "request-70 payload contains a non-JSON value",
    )


def _capture_record(raw: Mapping[str, JsonValue], shot_id: str, index: int, dimensions: tuple[int, int]) -> JsonObject:
    materialized = _materialize_json(raw)
    if not isinstance(materialized, dict):
        raise PhysicsPersistenceError(
            PersistenceErrorCode.MALFORMED_CAPTURE,
            "request-70 state is not a JSON object",
        )
    state = materialized
    state.update({"shot_id": shot_id, "record_type": "state"})
    rgb_value = state.get("rgb_frame")
    if not isinstance(rgb_value, dict):
        raise PhysicsPersistenceError(
            PersistenceErrorCode.MALFORMED_CAPTURE,
            "request-70 state has no rgb_frame object",
        )
    rgb = dict(rgb_value)
    rgb.update({
        "relative_path": f"frames/frame_{index:06d}.png",
        "width_pixels": dimensions[0],
        "height_pixels": dimensions[1],
    })
    state["rgb_frame"] = rgb
    return state


def persist_physics_rollout(
    bridge: PhysicsCaptureBridge,
    output_dir: Path,
    *,
    target_fps: float,
    duration_seconds: float,
    max_frames: int | None,
    state_header: Mapping[str, JsonValue] | None,
    provenance: CaptureProvenance,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> JsonObject:
    bounds = CaptureBounds.resolve(target_fps, duration_seconds, max_frames)
    provenance.validate()
    if output_dir.is_symlink():
        raise PhysicsPersistenceError(
            PersistenceErrorCode.INVALID_CONFIGURATION,
            "physics capture output directory cannot be a symlink",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_descriptor = os.open(
            output_dir,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise PhysicsPersistenceError(
            PersistenceErrorCode.INVALID_CONFIGURATION,
            "physics capture output directory is not confined",
        ) from error
    frames_dir = output_dir / "frames"
    state_descriptor: int | None = None
    event_descriptor: int | None = None
    frames_descriptor: int | None = None
    try:
        try:
            os.mkdir("frames", dir_fd=output_descriptor)
        except FileExistsError:
            pass
        frames_descriptor = os.open(
            "frames",
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=output_descriptor,
        )
        output_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW
        state_descriptor = os.open(
            "physics_state.jsonl", output_flags, 0o600, dir_fd=output_descriptor
        )
        event_descriptor = os.open(
            "physics_events.jsonl", output_flags, 0o600, dir_fd=output_descriptor
        )
    except OSError as error:
        for descriptor in (state_descriptor, event_descriptor, frames_descriptor):
            if descriptor is not None:
                os.close(descriptor)
        raise PhysicsPersistenceError(
            PersistenceErrorCode.INVALID_CONFIGURATION,
            "physics capture child artifact is a symlink or is not confined",
        ) from error
    finally:
        os.close(output_descriptor)
    shot_id = output_dir.name.removesuffix(".tmp")
    state_path = output_dir / "physics_state.jsonl"
    event_path = output_dir / "physics_events.jsonl"
    state_digest = hashlib.sha256()
    event_digest = hashlib.sha256()
    frame_entries: list[JsonObject] = []
    frame_checksums: list[JsonObject] = []
    progress = StreamProgress(0, 0, 0)
    started_at = clock()

    try:
        with os.fdopen(state_descriptor, "wb") as state_stream, os.fdopen(event_descriptor, "wb") as event_stream:
            for index in range(bounds.frame_count):
                delay = started_at + index / target_fps - clock()
                if delay > 0:
                    sleeper(delay)
                capture = bridge.get_physics_capture_v1()
                dimensions = _png_dimensions(capture.png)
                state = _capture_record(capture.state, shot_id, index, dimensions)
                parsed_state = _parse_state(state, index + 1)
                if index == 0:
                    header = dict(state_header) if state_header is not None else _state_header(
                        parsed_state, state["coordinates"], shot_id
                    )
                    header.update({
                        "shot_id": shot_id,
                        "record_type": "state_header",
                        "capture_status": "complete",
                        "state_sidecar": "physics_state.jsonl",
                        "event_sidecar": "physics_events.jsonl",
                        "capture_limits": {
                            "max_state_records": MAX_STATE_RECORDS,
                            "max_event_records": MAX_EVENT_RECORDS,
                            "max_total_bytes": MAX_TOTAL_BYTES,
                        },
                    })
                    _parse_header(header)
                    progress = _write_bounded_record(
                        state_stream, state_digest, _encoded_record(header), progress
                    )
                    progress = StreamProgress(1, progress.event_count, progress.total_bytes)

                progress = _write_bounded_record(
                    state_stream, state_digest, _encoded_record(state), progress
                )
                progress = StreamProgress(progress.state_count + 1, progress.event_count, progress.total_bytes)

                for raw_event in capture.events:
                    materialized_event = _materialize_json(raw_event)
                    if not isinstance(materialized_event, dict):
                        raise PhysicsPersistenceError(
                            PersistenceErrorCode.MALFORMED_CAPTURE,
                            "request-70 event is not a JSON object",
                        )
                    event = materialized_event
                    event.update({"shot_id": shot_id, "record_type": "event"})
                    parsed_event = _parse_event(event, progress.event_count)
                    if parsed_event.clock.sequence < progress.event_count:
                        continue
                    if parsed_event.clock.sequence > progress.event_count:
                        raise PhysicsPersistenceError(
                            PersistenceErrorCode.MALFORMED_CAPTURE,
                            "request-70 event sequence contains a gap",
                        )
                    if progress.event_count >= MAX_EVENT_RECORDS:
                        raise PhysicsPersistenceError(
                            PersistenceErrorCode.EVENT_LIMIT,
                            f"event record limit is {MAX_EVENT_RECORDS}",
                        )
                    progress = _write_bounded_record(
                        event_stream, event_digest, _encoded_record(event), progress
                    )
                    progress = StreamProgress(progress.state_count, progress.event_count + 1, progress.total_bytes)

                relative_path = f"frames/frame_{index:06d}.png"
                try:
                    frame_descriptor = os.open(
                        f"frame_{index:06d}.png",
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=frames_descriptor,
                    )
                except OSError as error:
                    raise PhysicsPersistenceError(
                        PersistenceErrorCode.INVALID_CONFIGURATION,
                        "physics capture frame artifact is a symlink or is not confined",
                    ) from error
                with os.fdopen(frame_descriptor, "wb") as frame_stream:
                    frame_stream.write(capture.png)
                    frame_stream.flush()
                    os.fsync(frame_stream.fileno())
                frame_entries.append({"path": relative_path})
                frame_checksums.append({
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(capture.png).hexdigest(),
                })

            state_stream.flush()
            event_stream.flush()
            os.fsync(state_stream.fileno())
            os.fsync(event_stream.fileno())
    except PhysicsContractError as error:
        raise PhysicsPersistenceError(
            PersistenceErrorCode.MALFORMED_CAPTURE,
            str(error),
        ) from error
    finally:
        os.close(frames_descriptor)

    metadata: JsonObject = {
        "capture_contract": "physics_capture_v1",
        "schema_version": "physics_capture_v1",
        "protocol_version": 1,
        "player_sha256": provenance.player_sha256,
        "protocol_sha256": provenance.protocol_sha256,
        "archive_sha256": provenance.archive_sha256,
        "frame_count": bounds.frame_count,
        "frames_dir": "frames",
        "frames": frame_entries,
        "frame_checksums": frame_checksums,
        "physics_state_path": "physics_state.jsonl",
        "physics_events_path": "physics_events.jsonl",
        "physics_state_count": bounds.frame_count,
        "physics_event_count": progress.event_count,
        "physics_state_sha256": state_digest.hexdigest(),
        "physics_events_sha256": event_digest.hexdigest(),
        "sidecars_closed": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
