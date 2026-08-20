from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import PurePosixPath
from typing import Final


@dataclass(frozen=True, slots=True)
class ContractValueError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"invalid {self.field}: {self.reason}"


def _require_nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ContractValueError(field, "must be nonempty")


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ContractValueError(field, "must be a contained POSIX relative path")


@unique
class SplitName(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SidecarPath:
    relative_path: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path, "sidecar relative_path")
        if type(self.capabilities) is not tuple:
            raise ContractValueError("sidecar capabilities", "must be an immutable tuple")
        if not self.capabilities:
            raise ContractValueError("sidecar capabilities", "must be nonempty")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ContractValueError("sidecar capabilities", "must be unique")
        for capability in self.capabilities:
            _require_nonempty(capability, "sidecar capability")


@dataclass(frozen=True, slots=True)
class CaptureContractDescriptor:
    contract_name: str
    contract_version: str
    artifact_layout_version: str
    player_provenance: str | None
    protocol_provenance: str | None
    declared_capabilities: tuple[str, ...] = ()
    sidecar_paths: tuple[SidecarPath, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.contract_name, "contract_name")
        _require_nonempty(self.contract_version, "contract_version")
        _require_nonempty(self.artifact_layout_version, "artifact_layout_version")
        if type(self.declared_capabilities) is not tuple:
            raise ContractValueError("declared_capabilities", "must be an immutable tuple")
        if type(self.sidecar_paths) is not tuple:
            raise ContractValueError("sidecar_paths", "must be an immutable tuple")
        if len(set(self.declared_capabilities)) != len(self.declared_capabilities):
            raise ContractValueError("declared_capabilities", "must be unique")
        for capability in self.declared_capabilities:
            _require_nonempty(capability, "declared capability")
        declared = set(self.declared_capabilities)
        represented: set[str] = set()
        for sidecar in self.sidecar_paths:
            for capability in sidecar.capabilities:
                if capability not in declared:
                    raise ContractValueError(
                        "sidecar capability",
                        f"{capability!r} is not declared",
                    )
                if capability in represented:
                    raise ContractValueError(
                        "sidecar capability",
                        f"{capability!r} has multiple paths",
                    )
                represented.add(capability)


@dataclass(frozen=True, slots=True)
class ShotAction:
    values: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple:
            raise ContractValueError("shot action", "must be an immutable tuple")
        if len(self.values) != 5:
            raise ContractValueError("shot action", "must contain exactly five values")
        if not all(type(value) in (int, float) for value in self.values):
            raise ContractValueError("shot action", "values must be numeric")

    @property
    def drag_start_x(self) -> float:
        return self.values[0]

    @property
    def drag_start_y(self) -> float:
        return self.values[1]

    @property
    def drag_release_x(self) -> float:
        return self.values[2]

    @property
    def drag_release_y(self) -> float:
        return self.values[3]

    @property
    def hold_time(self) -> float:
        return self.values[4]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    index: int
    relative_path: str
    timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ContractValueError("frame index", "must be nonnegative")
        _validate_relative_path(self.relative_path, "frame relative_path")


@dataclass(frozen=True, slots=True)
class ShotRecord:
    name: str
    relative_path: str
    action: ShotAction
    frames: tuple[FrameRecord, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "shot name")
        _validate_relative_path(self.relative_path, "shot relative_path")
        if type(self.frames) is not tuple:
            raise ContractValueError("shot frames", "must be an immutable tuple")
        if not self.frames:
            raise ContractValueError("shot frames", "must be nonempty")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    name: str
    split: SplitName
    relative_path: str
    shots: tuple[ShotRecord, ...]
    capture_contract: CaptureContractDescriptor
    source_level_key: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "episode name")
        _validate_relative_path(self.relative_path, "episode relative_path")
        if type(self.shots) is not tuple:
            raise ContractValueError("episode shots", "must be an immutable tuple")
        if not self.shots:
            raise ContractValueError("episode shots", "must be nonempty")


@dataclass(frozen=True, slots=True)
class TemporalWindowRequest:
    prediction_steps: int
    stride_frames: int

    def __post_init__(self) -> None:
        if type(self.prediction_steps) is not int:
            raise ContractValueError("prediction_steps", "must be an integer")
        if type(self.stride_frames) is not int:
            raise ContractValueError("stride_frames", "must be an integer")
        if self.prediction_steps <= 0:
            raise ContractValueError("prediction_steps", "must be positive")
        if self.stride_frames <= 0:
            raise ContractValueError("stride_frames", "must be positive")

    @property
    def horizon_frames(self) -> int:
        return self.prediction_steps * self.stride_frames


@dataclass(frozen=True, slots=True)
class ReproducibilityMetadata:
    catalog_identity: str
    seed: int
    epoch: int
    draw_count: int

    def __post_init__(self) -> None:
        _require_nonempty(self.catalog_identity, "catalog_identity")
        if self.epoch < 0:
            raise ContractValueError("epoch", "must be nonnegative")
        if self.draw_count <= 0:
            raise ContractValueError("draw_count", "must be positive")


PHYSICS_CAPTURE_V1_CAPABILITIES: Final = (
    "scene_nodes",
    "raw_contacts",
    "derived_support",
    "kinematics",
    "macro_events",
)

LEGACY_RGB_V1: Final = CaptureContractDescriptor(
    contract_name="legacy_rgb_v1",
    contract_version="1",
    artifact_layout_version="collector_v1",
    player_provenance=None,
    protocol_provenance="desktop-imagegrab",
)

PHYSICS_CAPTURE_V1: Final = CaptureContractDescriptor(
    contract_name="physics_capture_v1",
    contract_version="1",
    artifact_layout_version="collector_v1",
    player_provenance=None,
    protocol_provenance=None,
    declared_capabilities=PHYSICS_CAPTURE_V1_CAPABILITIES,
    sidecar_paths=(
        SidecarPath(
            relative_path="physics_state.jsonl",
            capabilities=("scene_nodes", "raw_contacts", "derived_support", "kinematics"),
        ),
        SidecarPath(
            relative_path="physics_events.jsonl",
            capabilities=("macro_events",),
        ),
    ),
)


def infer_capture_contract(
    explicit_descriptor: CaptureContractDescriptor | None,
) -> CaptureContractDescriptor:
    if explicit_descriptor is None:
        return LEGACY_RGB_V1
    return explicit_descriptor
