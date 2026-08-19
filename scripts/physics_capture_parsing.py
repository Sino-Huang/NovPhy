from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import TypeAlias

from scripts.physics_capture_contract import (
    ContractErrorCode,
    EXPECTED_COORDINATES,
    SCHEMA_VERSION,
    contract_error,
)
from scripts.physics_capture_types import (
    CaptureId,
    CaptureLimits,
    ContactId,
    CoordinateDeclaration,
    EntityId,
    EvidenceFixedStepCoverage,
    EvidenceIncompleteReason,
    EvidenceSupport,
    EvidenceSupportEdge,
    EvidenceTerminalTrace,
    EvidenceTraceEntity,
    EvidenceTraceSample,
    EventId,
    EventRecord,
    EventType,
    PhysicsBody,
    PhysicsCapture,
    PhysicsViolationEngineEvidence,
    RawContact,
    RecordClock,
    RgbFrame,
    SceneNode,
    ShotId,
    StateFrame,
    StateHeader,
    SupportEdge,
    SupportId,
    SupportRule,
    MinimumContactSeparation,
    Vector2,
    WorldPose,
)
from scripts.physics_rollout_contract import MAX_EVENT_RECORDS, MAX_STATE_RECORDS, MAX_TOTAL_BYTES


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
COMMON_FIELDS = frozenset(("schema_version", "capture_id", "shot_id", "sequence", "render_frame", "render_time", "fixed_step", "fixed_time", "coordinates"))
HEADER_FIELDS = COMMON_FIELDS | frozenset(("record_type", "capture_status", "state_sidecar", "event_sidecar", "support_rule", "event_taxonomy", "capture_limits"))
STATE_FIELDS = COMMON_FIELDS | frozenset(("record_type", "rgb_frame", "nodes", "raw_contacts", "support_edges"))
EVENT_FIELDS = COMMON_FIELDS | frozenset(("record_type", "event_id", "event_type", "participants", "payload"))
ENTITY_ID_PATTERN = re.compile(r"^(?:-?[0-9]+:[0-9]+|world:static:-?[0-9]+)$")
EVIDENCE_CONTACT_ID_PATTERN = re.compile(
    r"^contact:([0-9]+):"
    r"(?:-?[0-9]+:[0-9]+|world:static:-?[0-9]+):-?[0-9]+\|"
    r"(?:-?[0-9]+:[0-9]+|world:static:-?[0-9]+):-?[0-9]+:[0-9]+$"
)
EVIDENCE_SCHEMA_VERSION = "physics_violation_engine_evidence_v1"
EVIDENCE_FIELDS = frozenset(("schema_version", "capture_id", "shot_id", "sequence", "fixed_step_coverage", "minimum_contact_separation", "terminal_trace"))


def _read_jsonl(path: Path, *, max_records: int, allow_empty: bool = False) -> tuple[JsonObject, ...]:
    if path.stat().st_size > MAX_TOTAL_BYTES:
        raise contract_error(ContractErrorCode.INVALID_VALUE, str(path), "sidecar byte limit exceeded")
    records: list[JsonObject] = []
    try:
        with path.open("rb") as sidecar:
            line_number = 0
            bytes_read = 0
            while line := sidecar.readline(MAX_TOTAL_BYTES + 1):
                line_number += 1
                bytes_read += len(line)
                if bytes_read > MAX_TOTAL_BYTES:
                    raise contract_error(ContractErrorCode.INVALID_VALUE, str(path), "sidecar byte limit exceeded")
                if line_number > max_records:
                    raise contract_error(ContractErrorCode.INVALID_VALUE, str(path), "record limit exceeded")
                try:
                    value: JsonValue = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as error:
                    raise contract_error(ContractErrorCode.MALFORMED_JSON, f"{path}:{line_number}", error.msg) from error
                if not isinstance(value, dict):
                    raise contract_error(ContractErrorCode.EXPECTED_OBJECT, f"{path}:{line_number}", "record must be an object")
                records.append(value)
    except UnicodeDecodeError as error:
        raise contract_error(ContractErrorCode.MALFORMED_JSON, str(path), "sidecar is not UTF-8") from error
    if not records and not allow_empty:
        raise contract_error(ContractErrorCode.INVALID_VALUE, str(path), "sidecar must not be empty")
    return tuple(records)


def _value(record: JsonObject, field: str, location: str) -> JsonValue:
    if field not in record:
        raise contract_error(ContractErrorCode.MISSING_FIELD, location, field)
    return record[field]


def _string(record: JsonObject, field: str, location: str) -> str:
    value = _value(record, field, location)
    if not isinstance(value, str):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _integer(record: JsonObject, field: str, location: str) -> int:
    value = _value(record, field, location)
    if isinstance(value, bool) or not isinstance(value, int):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _number(record: JsonObject, field: str, location: str) -> float:
    value = _value(record, field, location)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return float(value)


def _boolean(record: JsonObject, field: str, location: str) -> bool:
    value = _value(record, field, location)
    if not isinstance(value, bool):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _object(record: JsonObject, field: str, location: str) -> JsonObject:
    value = _value(record, field, location)
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _array(record: JsonObject, field: str, location: str) -> list[JsonValue]:
    value = _value(record, field, location)
    if not isinstance(value, list):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _optional_number(record: JsonObject, field: str, location: str) -> float | None:
    value = _value(record, field, location)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return float(value)


def _optional_integer(record: JsonObject, field: str, location: str) -> int | None:
    value = _value(record, field, location)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _optional_string(record: JsonObject, field: str, location: str) -> str | None:
    value = _value(record, field, location)
    if value is None:
        return None
    if not isinstance(value, str):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, field)
    return value


def _exact_fields(record: JsonObject, fields: frozenset[str], location: str) -> None:
    unknown = record.keys() - fields
    if unknown:
        raise contract_error(ContractErrorCode.UNKNOWN_FIELD, location, min(unknown))
    missing = fields - record.keys()
    if missing:
        raise contract_error(ContractErrorCode.MISSING_FIELD, location, min(missing))


def _vector(record: JsonObject, location: str) -> Vector2:
    _exact_fields(record, frozenset(("x", "y")), location)
    return Vector2(_number(record, "x", location), _number(record, "y", location))


def _vector_value(value: JsonValue, location: str) -> Vector2:
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "vector")
    return _vector(value, location)


def _coordinates(record: JsonObject, location: str) -> CoordinateDeclaration:
    coordinates = _object(record, "coordinates", location)
    expected = EXPECTED_COORDINATES.__dataclass_fields__.keys()
    _exact_fields(coordinates, frozenset(expected), f"{location}.coordinates")
    parsed = CoordinateDeclaration(**{field: _string(coordinates, field, location) for field in expected})
    if parsed != EXPECTED_COORDINATES:
        raise contract_error(ContractErrorCode.INVALID_VALUE, f"{location}.coordinates", "coordinate declaration differs from v1")
    return parsed


def _clock(record: JsonObject, location: str) -> RecordClock:
    version = _string(record, "schema_version", location)
    if version != SCHEMA_VERSION:
        raise contract_error(ContractErrorCode.UNSUPPORTED_SCHEMA, location, version)
    sequence = _integer(record, "sequence", location)
    render_frame = _integer(record, "render_frame", location)
    fixed_step = _integer(record, "fixed_step", location)
    render_time = _number(record, "render_time", location)
    fixed_time = _number(record, "fixed_time", location)
    capture_id = _string(record, "capture_id", location)
    shot_id = _string(record, "shot_id", location)
    if min(sequence, render_frame, fixed_step, render_time, fixed_time) < 0 or not capture_id or re.fullmatch(r"shot_[0-9]{3,}", shot_id) is None:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "clock counters must be nonnegative")
    return RecordClock(version, CaptureId(capture_id), ShotId(shot_id), sequence, render_frame, render_time, fixed_step, fixed_time, _coordinates(record, location))


def _parse_header(record: JsonObject) -> StateHeader:
    location = "physics_state.jsonl:1"
    _exact_fields(record, HEADER_FIELDS, location)
    if _string(record, "record_type", location) != "state_header":
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "first state record must be state_header")
    taxonomy_values = _array(record, "event_taxonomy", location)
    try:
        taxonomy = tuple(EventType(value) for value in taxonomy_values if isinstance(value, str))
    except ValueError as error:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "unknown event taxonomy") from error
    if len(taxonomy) != len(taxonomy_values) or taxonomy != tuple(EventType):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "event taxonomy differs from v1")
    limits_raw = _object(record, "capture_limits", location)
    _exact_fields(limits_raw, frozenset(("max_state_records", "max_event_records", "max_total_bytes")), f"{location}.capture_limits")
    limits = CaptureLimits(_integer(limits_raw, "max_state_records", location), _integer(limits_raw, "max_event_records", location), _integer(limits_raw, "max_total_bytes", location))
    if min(limits.max_state_records, limits.max_event_records, limits.max_total_bytes) <= 0:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "capture limits must be positive")
    rule_raw = _object(record, "support_rule", location)
    _exact_fields(rule_raw, frozenset(("name", "minimum_consecutive_fixed_steps", "minimum_abs_normal_y", "minimum_vertical_center_delta", "include_triggers", "missing_contact_policy", "static_entity_id_prefix")), f"{location}.support_rule")
    rule = SupportRule(_string(rule_raw, "name", location), _integer(rule_raw, "minimum_consecutive_fixed_steps", location), _number(rule_raw, "minimum_abs_normal_y", location), _number(rule_raw, "minimum_vertical_center_delta", location), _boolean(rule_raw, "include_triggers", location), _string(rule_raw, "missing_contact_policy", location), _string(rule_raw, "static_entity_id_prefix", location))
    if rule != SupportRule("support_v1", 2, 0.5, 0.0001, False, "no_support", "world:static:"):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "support rule differs from v1")
    return StateHeader(_clock(record, location), _string(record, "capture_status", location), _string(record, "state_sidecar", location), _string(record, "event_sidecar", location), rule, taxonomy, limits)


def _parse_body(record: JsonObject, location: str) -> PhysicsBody:
    _exact_fields(record, frozenset(("present", "velocity", "angular_velocity_degrees_per_second", "mass_unity_units", "kinetic_energy_unity_units")), location)
    present = _boolean(record, "present", location)
    velocity_raw = _value(record, "velocity", location)
    velocity = None if velocity_raw is None else _vector_value(velocity_raw, f"{location}.velocity")
    body = PhysicsBody(present, velocity, _optional_number(record, "angular_velocity_degrees_per_second", location), _optional_number(record, "mass_unity_units", location), _optional_number(record, "kinetic_energy_unity_units", location))
    values_present = all(value is not None for value in (body.velocity, body.angular_velocity_degrees_per_second, body.mass_unity_units, body.kinetic_energy_unity_units))
    if present != values_present:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "body fields must all be present or all null")
    if present and body.velocity is not None and body.mass_unity_units is not None and body.kinetic_energy_unity_units is not None:
        expected = 0.5 * body.mass_unity_units * (body.velocity.x**2 + body.velocity.y**2)
        if not math.isclose(body.kinetic_energy_unity_units, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise contract_error(ContractErrorCode.INVALID_VALUE, location, "kinetic energy does not match 0.5*m*|v|^2")
    return body


def _parse_node(value: JsonValue, location: str) -> SceneNode:
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "node")
    _exact_fields(value, frozenset(("entity_id", "unity_instance_id", "object_class", "object_type", "screen_polygon", "world_pose", "life", "body")), location)
    entity_id = _string(value, "entity_id", location)
    if ENTITY_ID_PATTERN.fullmatch(entity_id) is None:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "entity_id")
    polygon = tuple(_vector_value(point, f"{location}.screen_polygon") for point in _array(value, "screen_polygon", location))
    if len(polygon) < 3:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "screen polygon needs at least three points")
    pose = _object(value, "world_pose", location)
    _exact_fields(pose, frozenset(("position", "rotation_degrees")), f"{location}.world_pose")
    return SceneNode(EntityId(entity_id), _integer(value, "unity_instance_id", location), _string(value, "object_class", location), _string(value, "object_type", location), polygon, WorldPose(_vector(_object(pose, "position", location), location), _number(pose, "rotation_degrees", location)), _optional_number(value, "life", location), _parse_body(_object(value, "body", location), f"{location}.body"))


def _parse_contact(value: JsonValue, location: str) -> RawContact:
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "contact")
    _exact_fields(value, frozenset(("contact_id", "entity_a_id", "entity_b_id", "collider_a_id", "collider_b_id", "point", "normal_a_to_b", "separation", "relative_velocity_a_to_b", "normal_impulse", "tangent_impulse", "is_trigger")), location)
    contact = RawContact(ContactId(_string(value, "contact_id", location)), EntityId(_string(value, "entity_a_id", location)), EntityId(_string(value, "entity_b_id", location)), _integer(value, "collider_a_id", location), _integer(value, "collider_b_id", location), _vector(_object(value, "point", location), location), _vector(_object(value, "normal_a_to_b", location), location), _number(value, "separation", location), _vector(_object(value, "relative_velocity_a_to_b", location), location), _optional_number(value, "normal_impulse", location), _optional_number(value, "tangent_impulse", location), _boolean(value, "is_trigger", location))
    if contact.entity_a_id >= contact.entity_b_id or contact.is_trigger:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "contacts must be canonical and non-trigger")
    return contact


def _parse_support(value: JsonValue, location: str) -> SupportEdge:
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "support")
    _exact_fields(value, frozenset(("support_id", "rule_version", "supporter_id", "supported_id", "evidence_contact_ids", "evidence_fixed_steps")), location)
    contacts = _array(value, "evidence_contact_ids", location)
    steps = _array(value, "evidence_fixed_steps", location)
    if len(contacts) != 2 or not all(isinstance(item, str) for item in contacts) or len(steps) != 2 or not all(isinstance(item, int) and not isinstance(item, bool) for item in steps):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "support evidence")
    supporter = EntityId(_string(value, "supporter_id", location))
    supported = EntityId(_string(value, "supported_id", location))
    support_id = SupportId(_string(value, "support_id", location))
    if support_id != f"support:{supporter}->{supported}":
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "support_id")
    return SupportEdge(support_id, _string(value, "rule_version", location), supporter, supported, (ContactId(contacts[0]), ContactId(contacts[1])), (steps[0], steps[1]))


def _parse_state(record: JsonObject, index: int) -> StateFrame:
    location = f"physics_state.jsonl:{index + 1}"
    _exact_fields(record, STATE_FIELDS, location)
    if _string(record, "record_type", location) != "state":
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "expected state")
    rgb_raw = _object(record, "rgb_frame", location)
    _exact_fields(rgb_raw, frozenset(("relative_path", "render_frame", "width_pixels", "height_pixels", "source")), f"{location}.rgb_frame")
    rgb = RgbFrame(_string(rgb_raw, "relative_path", location), _integer(rgb_raw, "render_frame", location), _integer(rgb_raw, "width_pixels", location), _integer(rgb_raw, "height_pixels", location), _string(rgb_raw, "source", location))
    clock = _clock(record, location)
    if rgb.render_frame != clock.render_frame or rgb.source != "synchronized_endpoint":
        raise contract_error(ContractErrorCode.RENDER_FRAME_MISMATCH, location, "RGB and state render_frame/source mismatch")
    nodes = tuple(_parse_node(value, f"{location}.nodes") for value in _array(record, "nodes", location))
    contacts = tuple(_parse_contact(value, f"{location}.raw_contacts") for value in _array(record, "raw_contacts", location))
    supports = tuple(_parse_support(value, f"{location}.support_edges") for value in _array(record, "support_edges", location))
    if tuple(node.entity_id for node in nodes) != tuple(sorted(node.entity_id for node in nodes)):
        raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, location, "nodes")
    if contacts != tuple(sorted(contacts, key=lambda item: (item.entity_a_id, item.entity_b_id, item.collider_a_id, item.collider_b_id, item.point.x, item.point.y, item.contact_id))):
        raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, location, "raw_contacts")
    if supports != tuple(sorted(supports, key=lambda item: (item.supporter_id, item.supported_id, item.support_id))):
        raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, location, "support_edges")
    return StateFrame(clock, rgb, nodes, contacts, supports)


def _parse_event(record: JsonObject, index: int) -> EventRecord:
    location = f"physics_events.jsonl:{index + 1}"
    _exact_fields(record, EVENT_FIELDS, location)
    if _string(record, "record_type", location) != "event":
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "expected event")
    try:
        event_type = EventType(_string(record, "event_type", location))
    except ValueError as error:
        raise contract_error(ContractErrorCode.INVALID_EVENT, location, "unknown event type") from error
    raw_participants = _array(record, "participants", location)
    if not all(isinstance(item, str) for item in raw_participants):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "participants")
    participants = tuple(EntityId(item) for item in raw_participants)
    if participants != tuple(sorted(set(participants))):
        raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, location, "participants")
    payload = _object(record, "payload", location)
    return EventRecord(_clock(record, location), EventId(_string(record, "event_id", location)), event_type, participants, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _evidence_reason(value: str | None, location: str) -> EvidenceIncompleteReason | None:
    if value is None:
        return None
    try:
        return EvidenceIncompleteReason(value)
    except ValueError as error:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "incomplete reason") from error


def _evidence_contact_step(value: str, location: str) -> int:
    match = EVIDENCE_CONTACT_ID_PATTERN.fullmatch(value)
    if match is None:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "contact_id")
    return int(match.group(1))


def _parse_evidence_support(value: JsonValue, location: str) -> EvidenceSupport:
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "support_v1")
    _exact_fields(value, frozenset(("present", "edges")), location)
    edges: list[EvidenceSupportEdge] = []
    for index, raw in enumerate(_array(value, "edges", location)):
        edge_location = f"{location}.edges[{index}]"
        if not isinstance(raw, dict):
            raise contract_error(ContractErrorCode.WRONG_TYPE, edge_location, "edge")
        _exact_fields(raw, frozenset(("support_id", "supporter_id", "evidence_contact_ids", "evidence_fixed_steps")), edge_location)
        contacts = _array(raw, "evidence_contact_ids", edge_location)
        steps = _array(raw, "evidence_fixed_steps", edge_location)
        if len(contacts) != 2 or not all(isinstance(item, str) and item for item in contacts):
            raise contract_error(ContractErrorCode.WRONG_TYPE, edge_location, "evidence_contact_ids")
        if len(steps) != 2 or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in steps):
            raise contract_error(ContractErrorCode.WRONG_TYPE, edge_location, "evidence_fixed_steps")
        if steps[1] != steps[0] + 1:
            raise contract_error(ContractErrorCode.INVALID_VALUE, edge_location, "support evidence steps")
        if tuple(_evidence_contact_step(item, edge_location) for item in contacts) != tuple(steps):
            raise contract_error(ContractErrorCode.INVALID_VALUE, edge_location, "support contact steps")
        supporter = _string(raw, "supporter_id", edge_location)
        if ENTITY_ID_PATTERN.fullmatch(supporter) is None:
            raise contract_error(ContractErrorCode.INVALID_VALUE, edge_location, "supporter_id")
        support_id = _string(raw, "support_id", edge_location)
        edges.append(EvidenceSupportEdge(SupportId(support_id), EntityId(supporter),
            (ContactId(contacts[0]), ContactId(contacts[1])), (steps[0], steps[1])))
    present = _boolean(value, "present", location)
    if present != bool(edges):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "support presence differs from evidence")
    support_ids = tuple(edge.support_id for edge in edges)
    if support_ids != tuple(sorted(set(support_ids))):
        raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, location, "support edges")
    return EvidenceSupport(present, tuple(edges))


def _parse_evidence_entity(value: JsonValue, location: str) -> EvidenceTraceEntity:
    if not isinstance(value, dict):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "entity")
    _exact_fields(value, frozenset(("entity_id", "observed", "present", "world_position", "body_type", "simulated", "gravity_scale", "support_v1")), location)
    entity_id = _string(value, "entity_id", location)
    if ENTITY_ID_PATTERN.fullmatch(entity_id) is None:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "entity_id")
    observed = _boolean(value, "observed", location)
    present = _boolean(value, "present", location)
    position_raw = _value(value, "world_position", location)
    position = None if position_raw is None else _vector_value(position_raw, f"{location}.world_position")
    body_type = _optional_string(value, "body_type", location)
    simulated_raw = _value(value, "simulated", location)
    if simulated_raw is not None and not isinstance(simulated_raw, bool):
        raise contract_error(ContractErrorCode.WRONG_TYPE, location, "simulated")
    gravity_scale = _optional_number(value, "gravity_scale", location)
    if not observed or present != all(item is not None for item in (position, body_type, simulated_raw, gravity_scale)):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "observed/present body facts are inconsistent")
    if body_type not in (None, "dynamic", "kinematic", "static"):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "body_type")
    support = _parse_evidence_support(
        _value(value, "support_v1", location), f"{location}.support_v1"
    )
    if not present and support.present:
        raise contract_error(
            ContractErrorCode.INVALID_VALUE, location, "absent entity has support evidence"
        )
    return EvidenceTraceEntity(EntityId(entity_id), observed, present, position, body_type,
        simulated_raw, gravity_scale, support)


def _parse_evidence(record: JsonObject, index: int) -> PhysicsViolationEngineEvidence:
    location = f"physics_violation_engine_evidence_v1.jsonl:{index + 1}"
    _exact_fields(record, EVIDENCE_FIELDS, location)
    version = _string(record, "schema_version", location)
    if version != EVIDENCE_SCHEMA_VERSION:
        raise contract_error(ContractErrorCode.UNSUPPORTED_SCHEMA, location, version)
    capture_id = _string(record, "capture_id", location)
    shot_id = _string(record, "shot_id", location)
    sequence = _integer(record, "sequence", location)
    if not capture_id or not shot_id or sequence < 0:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "evidence identity")

    coverage_raw = _object(record, "fixed_step_coverage", location)
    _exact_fields(coverage_raw, frozenset(("first_fixed_step", "last_fixed_step", "sample_count", "complete", "incomplete_reason")), f"{location}.fixed_step_coverage")
    first = _optional_integer(coverage_raw, "first_fixed_step", location)
    last = _optional_integer(coverage_raw, "last_fixed_step", location)
    sample_count = _integer(coverage_raw, "sample_count", location)
    complete = _boolean(coverage_raw, "complete", location)
    reason = _evidence_reason(_optional_string(coverage_raw, "incomplete_reason", location), location)
    if sample_count < 0 or (sample_count == 0) != (first is None and last is None):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "fixed-step coverage bounds")
    if sample_count and (first is None or last is None or min(first, last) < 0 or first > last):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "fixed-step coverage bounds")
    if complete != (reason is None):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "fixed-step completeness")
    if sample_count == 0:
        if reason is not EvidenceIncompleteReason.NO_FIXED_STEP_SAMPLES:
            raise contract_error(ContractErrorCode.INVALID_VALUE, location, "empty fixed-step coverage reason")
    else:
        assert first is not None and last is not None
        covered_span = last - first + 1
        if sample_count > covered_span or (complete and sample_count != covered_span):
            raise contract_error(ContractErrorCode.INVALID_VALUE, location, "fixed-step completeness")
    coverage = EvidenceFixedStepCoverage(first, last, sample_count, complete, reason)

    minimum_raw = _object(record, "minimum_contact_separation", location)
    _exact_fields(minimum_raw, frozenset(("observed", "separation", "contact_id", "fixed_step")), f"{location}.minimum_contact_separation")
    minimum_observed = _boolean(minimum_raw, "observed", location)
    separation = _optional_number(minimum_raw, "separation", location)
    contact_id = _optional_string(minimum_raw, "contact_id", location)
    minimum_step = _optional_integer(minimum_raw, "fixed_step", location)
    if minimum_observed != all(item is not None for item in (separation, contact_id, minimum_step)):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "minimum contact presence")
    if contact_id == "":
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "minimum contact_id")
    if minimum_step is not None and minimum_step < 0:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "minimum contact fixed_step")
    if contact_id is not None and _evidence_contact_step(contact_id, location) != minimum_step:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "minimum contact fixed_step")
    minimum = MinimumContactSeparation(minimum_observed, separation,
        None if contact_id is None else ContactId(contact_id), minimum_step)

    trace_raw = _object(record, "terminal_trace", location)
    _exact_fields(trace_raw, frozenset(("max_fixed_steps", "max_entities_per_step", "first_fixed_step", "last_fixed_step", "truncated", "truncation_reason", "failure_reason", "samples")), f"{location}.terminal_trace")
    max_steps = _integer(trace_raw, "max_fixed_steps", location)
    max_entities = _integer(trace_raw, "max_entities_per_step", location)
    if (max_steps, max_entities) != (8, 128):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "terminal trace bounds differ from v1")
    samples: list[EvidenceTraceSample] = []
    for sample_index, raw_sample in enumerate(_array(trace_raw, "samples", location)):
        sample_location = f"{location}.terminal_trace.samples[{sample_index}]"
        if not isinstance(raw_sample, dict):
            raise contract_error(ContractErrorCode.WRONG_TYPE, sample_location, "sample")
        _exact_fields(raw_sample, frozenset(("fixed_step", "physics2d_gravity", "entities")), sample_location)
        entities = tuple(
            _parse_evidence_entity(item, f"{sample_location}.entities[{entity_index}]")
            for entity_index, item in enumerate(_array(raw_sample, "entities", sample_location))
        )
        entity_ids = tuple(item.entity_id for item in entities)
        if len(entities) > max_entities or entity_ids != tuple(sorted(set(entity_ids))):
            raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, sample_location, "entities")
        step = _integer(raw_sample, "fixed_step", sample_location)
        if step < 0:
            raise contract_error(ContractErrorCode.INVALID_VALUE, sample_location, "fixed_step")
        samples.append(EvidenceTraceSample(step, _vector(_object(raw_sample, "physics2d_gravity", sample_location), sample_location), entities))
    if len(samples) > max_steps or any(current.fixed_step != previous.fixed_step + 1 for previous, current in zip(samples, samples[1:])):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "terminal trace is not bounded consecutive history")
    trace_first = _optional_integer(trace_raw, "first_fixed_step", location)
    trace_last = _optional_integer(trace_raw, "last_fixed_step", location)
    expected_bounds = (None, None) if not samples else (samples[0].fixed_step, samples[-1].fixed_step)
    if (trace_first, trace_last) != expected_bounds:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "terminal trace bounds")
    truncated = _boolean(trace_raw, "truncated", location)
    truncation_reason = _optional_string(trace_raw, "truncation_reason", location)
    if (truncated, truncation_reason) not in ((False, None), (True, "terminal_trace_bound")):
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "terminal trace truncation")
    failure_reason = _evidence_reason(_optional_string(trace_raw, "failure_reason", location), location)
    if failure_reason != reason:
        raise contract_error(ContractErrorCode.INVALID_VALUE, location, "trace failure differs from coverage")
    trace = EvidenceTerminalTrace(max_steps, max_entities, trace_first, trace_last, truncated,
        truncation_reason, failure_reason, tuple(samples))
    return PhysicsViolationEngineEvidence(version, CaptureId(capture_id), shot_id, sequence, coverage, minimum, trace)


def parse_physics_sidecars(
    state_path: Path,
    event_path: Path,
    evidence_path: Path | None = None,
) -> PhysicsCapture:
    if state_path.stat().st_size + event_path.stat().st_size > MAX_TOTAL_BYTES:
        raise contract_error(ContractErrorCode.INVALID_VALUE, "sidecars", "sidecar byte limit exceeded")
    state_records = _read_jsonl(state_path, max_records=MAX_STATE_RECORDS)
    event_records = _read_jsonl(event_path, max_records=MAX_EVENT_RECORDS, allow_empty=True)
    evidence_records = () if evidence_path is None else _read_jsonl(
        evidence_path, max_records=MAX_STATE_RECORDS
    )
    return PhysicsCapture(_parse_header(state_records[0]), tuple(_parse_state(record, index) for index, record in enumerate(state_records[1:], start=1)), tuple(_parse_event(record, index) for index, record in enumerate(event_records)), tuple(_parse_evidence(record, index) for index, record in enumerate(evidence_records)))
