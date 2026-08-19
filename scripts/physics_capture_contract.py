from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

from scripts.physics_capture_types import CoordinateDeclaration, PhysicsCapture


SCHEMA_VERSION: Final = "physics_capture_v1"
STATE_SIDECAR: Final = "physics_state.jsonl"
EVENT_SIDECAR: Final = "physics_events.jsonl"
VIOLATION_EVIDENCE_SIDECAR: Final = "physics_violation_engine_evidence_v1.jsonl"
EXPECTED_COORDINATES: Final = CoordinateDeclaration(
    world_space="unity_world_2d",
    world_origin="scene_defined",
    world_x_axis="right",
    world_y_axis="up",
    world_length_unit="unity_unit",
    screen_space="rgb_pixel_2d",
    screen_origin="top_left",
    screen_x_axis="right",
    screen_y_axis="down",
    screen_length_unit="pixel",
    time_unit="second",
    angle_unit="degree",
    mass_unit="unity_mass_unit",
    velocity_unit="unity_unit/second",
    angular_velocity_unit="degree/second",
    kinetic_energy_unit="unity_mass_unit*unity_unit^2/second^2",
    impulse_unit="unity_mass_unit*unity_unit/second",
)


@unique
class ContractErrorCode(StrEnum):
    MALFORMED_JSON = "malformed_json"
    EXPECTED_OBJECT = "expected_object"
    MISSING_FIELD = "missing_field"
    UNKNOWN_FIELD = "unknown_field"
    WRONG_TYPE = "wrong_type"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    INVALID_VALUE = "invalid_value"
    CAPTURE_MISMATCH = "capture_mismatch"
    SEQUENCE_ORDER = "sequence_order"
    CLOCK_ORDER = "clock_order"
    RENDER_FRAME_MISMATCH = "render_frame_mismatch"
    DETERMINISTIC_ORDER = "deterministic_order"
    NONPERSISTENT_SUPPORT = "nonpersistent_support"
    INVALID_EVENT = "invalid_event"


@dataclass(frozen=True, slots=True)
class PhysicsContractError(Exception):
    code: ContractErrorCode
    location: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code.value} at {self.location}: {self.detail}"


def contract_error(code: ContractErrorCode, location: str, detail: str) -> PhysicsContractError:
    return PhysicsContractError(code, location, detail)


def load_physics_capture(
    state_path: Path,
    event_path: Path,
    evidence_path: Path | None = None,
) -> PhysicsCapture:
    from scripts.physics_capture_parsing import parse_physics_sidecars
    from scripts.physics_capture_validation import validate_physics_capture

    if evidence_path is None:
        sibling = state_path.parent / VIOLATION_EVIDENCE_SIDECAR
        if sibling.is_file():
            evidence_path = sibling
    capture = parse_physics_sidecars(state_path, event_path, evidence_path)
    # The v1 header's max_total_bytes bound belongs to the retained state/event
    # pair.  The independently bounded evidence sidecar must not consume that
    # budget or change recorder retention semantics.
    validate_physics_capture(capture, state_path.stat().st_size + event_path.stat().st_size)
    return capture
