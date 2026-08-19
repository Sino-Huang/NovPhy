from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import NewType


CaptureId = NewType("CaptureId", str)
ShotId = NewType("ShotId", str)
EntityId = NewType("EntityId", str)
ContactId = NewType("ContactId", str)
SupportId = NewType("SupportId", str)
EventId = NewType("EventId", str)


@unique
class EventType(StrEnum):
    BIRD_LAUNCHED = "bird_launched"
    COLLISION = "collision"
    EXPLOSION = "explosion"
    ENTITY_DESTROYED = "entity_destroyed"
    PIG_REMOVED = "pig_removed"
    BIRD_EXHAUSTED = "bird_exhausted"
    STABLE_ENTERED = "stable_entered"
    STABLE_EXITED = "stable_exited"
    LEVEL_CLEARED = "level_cleared"
    LEVEL_FAILED = "level_failed"


@unique
class EvidenceIncompleteReason(StrEnum):
    NO_FIXED_STEP_SAMPLES = "no_fixed_step_samples"
    FIXED_STEP_GAP = "fixed_step_gap"
    CONTACT_SAMPLE_OVERFLOW = "contact_sample_overflow"
    ENTITY_SAMPLE_OVERFLOW = "entity_sample_overflow"
    SAMPLING_FAILURE = "sampling_failure"


@dataclass(frozen=True, slots=True)
class Vector2:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CoordinateDeclaration:
    world_space: str
    world_origin: str
    world_x_axis: str
    world_y_axis: str
    world_length_unit: str
    screen_space: str
    screen_origin: str
    screen_x_axis: str
    screen_y_axis: str
    screen_length_unit: str
    time_unit: str
    angle_unit: str
    mass_unit: str
    velocity_unit: str
    angular_velocity_unit: str
    kinetic_energy_unit: str
    impulse_unit: str


@dataclass(frozen=True, slots=True)
class RecordClock:
    schema_version: str
    capture_id: CaptureId
    shot_id: ShotId
    sequence: int
    render_frame: int
    render_time: float
    fixed_step: int
    fixed_time: float
    coordinates: CoordinateDeclaration


@dataclass(frozen=True, slots=True)
class CaptureLimits:
    max_state_records: int
    max_event_records: int
    max_total_bytes: int


@dataclass(frozen=True, slots=True)
class SupportRule:
    name: str
    minimum_consecutive_fixed_steps: int
    minimum_abs_normal_y: float
    minimum_vertical_center_delta: float
    include_triggers: bool
    missing_contact_policy: str
    static_entity_id_prefix: str


@dataclass(frozen=True, slots=True)
class WorldPose:
    position: Vector2
    rotation_degrees: float


@dataclass(frozen=True, slots=True)
class PhysicsBody:
    present: bool
    velocity: Vector2 | None
    angular_velocity_degrees_per_second: float | None
    mass_unity_units: float | None
    kinetic_energy_unity_units: float | None


@dataclass(frozen=True, slots=True)
class SceneNode:
    entity_id: EntityId
    unity_instance_id: int
    object_class: str
    object_type: str
    screen_polygon: tuple[Vector2, ...]
    world_pose: WorldPose
    life: float | None
    body: PhysicsBody


@dataclass(frozen=True, slots=True)
class RgbFrame:
    relative_path: str
    render_frame: int
    width_pixels: int
    height_pixels: int
    source: str


@dataclass(frozen=True, slots=True)
class RawContact:
    contact_id: ContactId
    entity_a_id: EntityId
    entity_b_id: EntityId
    collider_a_id: int
    collider_b_id: int
    point: Vector2
    normal_a_to_b: Vector2
    separation: float
    relative_velocity_a_to_b: Vector2
    normal_impulse: float | None
    tangent_impulse: float | None
    is_trigger: bool


@dataclass(frozen=True, slots=True)
class SupportEdge:
    support_id: SupportId
    rule_version: str
    supporter_id: EntityId
    supported_id: EntityId
    evidence_contact_ids: tuple[ContactId, ContactId]
    evidence_fixed_steps: tuple[int, int]


@dataclass(frozen=True, slots=True)
class StateHeader:
    clock: RecordClock
    capture_status: str
    state_sidecar: str
    event_sidecar: str
    support_rule: SupportRule
    event_taxonomy: tuple[EventType, ...]
    limits: CaptureLimits


@dataclass(frozen=True, slots=True)
class StateFrame:
    clock: RecordClock
    rgb_frame: RgbFrame
    nodes: tuple[SceneNode, ...]
    raw_contacts: tuple[RawContact, ...]
    support_edges: tuple[SupportEdge, ...]


@dataclass(frozen=True, slots=True)
class EventRecord:
    clock: RecordClock
    event_id: EventId
    event_type: EventType
    participants: tuple[EntityId, ...]
    payload_json: str


@dataclass(frozen=True, slots=True)
class EvidenceFixedStepCoverage:
    first_fixed_step: int | None
    last_fixed_step: int | None
    sample_count: int
    complete: bool
    incomplete_reason: EvidenceIncompleteReason | None


@dataclass(frozen=True, slots=True)
class MinimumContactSeparation:
    observed: bool
    separation: float | None
    contact_id: ContactId | None
    fixed_step: int | None


@dataclass(frozen=True, slots=True)
class EvidenceSupportEdge:
    support_id: SupportId
    supporter_id: EntityId
    evidence_contact_ids: tuple[ContactId, ContactId]
    evidence_fixed_steps: tuple[int, int]


@dataclass(frozen=True, slots=True)
class EvidenceSupport:
    present: bool
    edges: tuple[EvidenceSupportEdge, ...]


@dataclass(frozen=True, slots=True)
class EvidenceTraceEntity:
    entity_id: EntityId
    observed: bool
    present: bool
    world_position: Vector2 | None
    body_type: str | None
    simulated: bool | None
    gravity_scale: float | None
    support_v1: EvidenceSupport


@dataclass(frozen=True, slots=True)
class EvidenceTraceSample:
    fixed_step: int
    physics2d_gravity: Vector2
    entities: tuple[EvidenceTraceEntity, ...]


@dataclass(frozen=True, slots=True)
class EvidenceTerminalTrace:
    max_fixed_steps: int
    max_entities_per_step: int
    first_fixed_step: int | None
    last_fixed_step: int | None
    truncated: bool
    truncation_reason: str | None
    failure_reason: EvidenceIncompleteReason | None
    samples: tuple[EvidenceTraceSample, ...]


@dataclass(frozen=True, slots=True)
class PhysicsViolationEngineEvidence:
    schema_version: str
    capture_id: CaptureId
    shot_id: str
    sequence: int
    coverage: EvidenceFixedStepCoverage
    minimum_contact_separation: MinimumContactSeparation
    terminal_trace: EvidenceTerminalTrace


@dataclass(frozen=True, slots=True)
class PhysicsCapture:
    header: StateHeader
    states: tuple[StateFrame, ...]
    events: tuple[EventRecord, ...]
    violation_evidence: tuple[PhysicsViolationEngineEvidence, ...] = ()
