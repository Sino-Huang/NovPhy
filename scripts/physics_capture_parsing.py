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
    EventId,
    EventRecord,
    EventType,
    PhysicsBody,
    PhysicsCapture,
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
    Vector2,
    WorldPose,
)


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
COMMON_FIELDS = frozenset(("schema_version", "capture_id", "shot_id", "sequence", "render_frame", "render_time", "fixed_step", "fixed_time", "coordinates"))
HEADER_FIELDS = COMMON_FIELDS | frozenset(("record_type", "capture_status", "state_sidecar", "event_sidecar", "support_rule", "event_taxonomy", "capture_limits"))
STATE_FIELDS = COMMON_FIELDS | frozenset(("record_type", "rgb_frame", "nodes", "raw_contacts", "support_edges"))
EVENT_FIELDS = COMMON_FIELDS | frozenset(("record_type", "event_id", "event_type", "participants", "payload"))
ENTITY_ID_PATTERN = re.compile(r"^(?:-?[0-9]+:[0-9]+|world:static:-?[0-9]+)$")


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    records: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value: JsonValue = json.loads(line)
        except json.JSONDecodeError as error:
            raise contract_error(ContractErrorCode.MALFORMED_JSON, f"{path}:{line_number}", error.msg) from error
        if not isinstance(value, dict):
            raise contract_error(ContractErrorCode.EXPECTED_OBJECT, f"{path}:{line_number}", "record must be an object")
        records.append(value)
    if not records:
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


def parse_physics_sidecars(state_path: Path, event_path: Path) -> PhysicsCapture:
    state_records = _read_jsonl(state_path)
    event_records = _read_jsonl(event_path)
    return PhysicsCapture(_parse_header(state_records[0]), tuple(_parse_state(record, index) for index, record in enumerate(state_records[1:], start=1)), tuple(_parse_event(record, index) for index, record in enumerate(event_records)))
