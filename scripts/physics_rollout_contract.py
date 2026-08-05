from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
import math
from typing import Final, Mapping, Protocol, TypeAlias


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
MAX_STATE_RECORDS: Final = 64
MAX_FRAME_RECORDS: Final = MAX_STATE_RECORDS - 1
MAX_EVENT_RECORDS: Final = 64
MAX_TOTAL_BYTES: Final = 1_048_576


class PhysicsCapturePacket(Protocol):
    @property
    def png(self) -> bytes: ...

    @property
    def state(self) -> Mapping[str, JsonValue]: ...

    @property
    def events(self) -> tuple[Mapping[str, JsonValue], ...]: ...


class PhysicsCaptureBridge(Protocol):
    def get_physics_capture_v1(self) -> PhysicsCapturePacket: ...


@unique
class PersistenceErrorCode(StrEnum):
    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_PROVENANCE = "invalid_provenance"
    MALFORMED_CAPTURE = "malformed_capture"
    STATE_LIMIT = "state_limit"
    EVENT_LIMIT = "event_limit"
    BYTE_LIMIT = "byte_limit"


@dataclass(slots=True)
class PhysicsPersistenceError(RuntimeError):
    code: PersistenceErrorCode
    detail: str

    def __str__(self) -> str:
        return f"{self.code.value}: {self.detail}"


@dataclass(frozen=True, slots=True)
class CaptureProvenance:
    player_sha256: str
    protocol_sha256: str
    archive_sha256: str

    def validate(self) -> None:
        for name, value in (
            ("player_sha256", self.player_sha256),
            ("protocol_sha256", self.protocol_sha256),
            ("archive_sha256", self.archive_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise PhysicsPersistenceError(
                    PersistenceErrorCode.INVALID_PROVENANCE,
                    f"{name} must be a lowercase SHA-256 digest",
                )


@dataclass(frozen=True, slots=True)
class CaptureBounds:
    frame_count: int

    @classmethod
    def resolve(
        cls,
        target_fps: float,
        duration_seconds: float,
        max_frames: int | None,
    ) -> CaptureBounds:
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise PhysicsPersistenceError(
                PersistenceErrorCode.INVALID_CONFIGURATION,
                "target_fps must be finite and positive",
            )
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise PhysicsPersistenceError(
                PersistenceErrorCode.INVALID_CONFIGURATION,
                "duration_seconds must be finite and positive",
            )
        if max_frames is None:
            requested_frames = target_fps * duration_seconds
            if not math.isfinite(requested_frames):
                raise PhysicsPersistenceError(
                    PersistenceErrorCode.INVALID_CONFIGURATION,
                    "fps and duration produce an unbounded frame count",
                )
            frame_count = math.ceil(requested_frames)
        else:
            frame_count = max_frames
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
            raise PhysicsPersistenceError(
                PersistenceErrorCode.INVALID_CONFIGURATION,
                "max_frames must be a positive integer",
            )
        if frame_count > MAX_FRAME_RECORDS:
            raise PhysicsPersistenceError(
                PersistenceErrorCode.STATE_LIMIT,
                f"state record limit is {MAX_STATE_RECORDS} including the header; at most {MAX_FRAME_RECORDS} frames are allowed",
            )
        return cls(frame_count)


@dataclass(frozen=True, slots=True)
class StreamProgress:
    state_count: int
    event_count: int
    total_bytes: int
