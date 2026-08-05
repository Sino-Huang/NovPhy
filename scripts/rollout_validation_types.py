"""Immutable public outcomes for canonical rollout validation."""
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, assert_never

from world_model.data.types import CaptureContractDescriptor, FrameRecord, ShotAction


@unique
class EpisodeRejectionCode(StrEnum):
    MISSING_ARTIFACT = "missing_artifact"
    SYMLINK_ARTIFACT = "symlink_artifact"
    ESCAPING_ARTIFACT = "escaping_artifact"
    UNREADABLE_ARTIFACT = "unreadable_artifact"
    MALFORMED_JSON = "malformed_json"
    INVALID_EPISODE_CONTRACT = "invalid_episode_contract"
    INVALID_ATTEMPT_LOG = "invalid_attempt_log"
    INVALID_ACTION_LOG = "invalid_action_log"
    INVALID_SHOT_ARTIFACT = "invalid_shot_artifact"
    NONCONTIGUOUS_FRAMES = "noncontiguous_frames"
    INVALID_FRAME_METADATA = "invalid_frame_metadata"
    LEVEL_FIVE_ACTION_POLICY = "level_five_action_policy"
    MALFORMED_CAPTURE_CONTRACT = "malformed_capture_contract"
    UNKNOWN_CAPTURE_CONTRACT = "unknown_capture_contract"
    UNSUPPORTED_CAPTURE_CONTRACT = "unsupported_capture_contract"


@unique
class EpisodeValidationMode(StrEnum):
    MATERIALIZED = "materialized"
    CANONICAL_SUMMARY = "canonical_summary"

    @property
    def materializes_frames(self) -> bool:
        match self:
            case EpisodeValidationMode.MATERIALIZED:
                return True
            case EpisodeValidationMode.CANONICAL_SUMMARY:
                return False
            case unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class EpisodeValidationContract:
    count: int
    fps: float
    duration_seconds: float
    level_five: bool = False


@dataclass(frozen=True, slots=True)
class ValidatedShot:
    name: str
    relative_path: str
    frames: tuple[FrameRecord, ...]
    frame_count: int
    release_x: float
    action: ShotAction | None


@dataclass(frozen=True, slots=True)
class ValidatedEpisode:
    directory: Path
    shots: tuple[ValidatedShot, ...]
    capture_contract: CaptureContractDescriptor


@dataclass(frozen=True, slots=True)
class EpisodeAccepted:
    episode: ValidatedEpisode


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode: ValidatedEpisode
    canonical_acceptance_available: Literal[True] = True


class EpisodeRejected(NamedTuple):
    code: EpisodeRejectionCode
    artifact: str


EpisodeValidationResult: TypeAlias = EpisodeAccepted | EpisodeSummary | EpisodeRejected


@dataclass(frozen=True, slots=True)
class PhysicsArtifactSummary:
    state_count: int
    event_count: int
    frame_sha256: tuple[str, ...]
    state_sha256: str
    event_sha256: str


@dataclass(frozen=True, slots=True)
class PhysicsArtifactError(Exception):
    artifact: str
    reason: str

    def __str__(self) -> str:
        return f"invalid physics artifact {self.artifact}: {self.reason}"


@dataclass(frozen=True, slots=True)
class PhysicsRecoveryResult:
    removed_temporary: tuple[str, ...]
    quarantined: tuple[str, ...]


def reject(code: EpisodeRejectionCode, artifact: Path | str) -> EpisodeRejected:
    return EpisodeRejected(code, str(artifact))
