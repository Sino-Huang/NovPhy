"""Parser and fail-closed validator for the prospective `physics_capture_v2`."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from scripts.physics_capture_v2_types import PhysicsCaptureV2


SCHEMA_VERSION = "physics_capture_v2"
ENGINE_SCHEMA_VERSION = "physics_capture_v2_engine_v1"
SIDECAR = "physics_capture_v2.json"
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_FIXED_STEP_SAMPLES = 100_000
MAX_CAUSAL_ENTITIES = 2_048
MAX_COLLIDERS = 8_192
MAX_CONTACTS_PER_STEP = 32_768
MAX_FRAME_RECORDS = 100_000
MAX_EVENTS = 100_000
EVENT_PARTICIPANT_COUNTS = {
    "bird_launched": 1,
    "collision": 2,
    "entity_destroyed": 1,
    "entity_death": 1,
    "pig_removed": 1,
    "tnt_explosion": 1,
    "bird_exhaustion": 0,
    "stable_entered": 0,
    "stable_exited": 0,
    "level_clear": 0,
    "level_fail": 0,
    "terminal": 0,
}
_BINDINGS = frozenset((
    "scenario_template_id",
    "level_instance_id",
    "scenario_lineage_id",
    "rollout_id",
    "intervention_id",
))


class PhysicsCaptureV2Error(ValueError):
    """A v2 capture cannot be admitted under its declared evidence contract."""


def _fail(location: str, detail: str) -> NoReturn:
    raise PhysicsCaptureV2Error(f"{location}: {detail}")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(location, "expected object")
    return value


def _sequence(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, list):
        _fail(location, "expected array")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(location, "expected nonempty string")
    return value


def _integer(value: object, location: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        _fail(location, "expected" + (" positive" if positive else "") + " integer")
    return value


def _number(value: object, location: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(location, "expected finite number")
    return float(value)


def _vector(value: object, location: str) -> tuple[float, float]:
    values = _sequence(value, location)
    if len(values) != 2:
        _fail(location, "expected two-dimensional vector")
    return (_number(values[0], f"{location}[0]"), _number(values[1], f"{location}[1]"))


def _require_fields(record: Mapping[str, object], required: frozenset[str], location: str) -> None:
    missing = required - set(record)
    extra = set(record) - required
    if missing:
        _fail(location, f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        _fail(location, f"unknown fields: {', '.join(sorted(extra))}")


def _finite_tree(value: object, location: str) -> None:
    if type(value) in (int, float):
        _number(value, location)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _string(key, f"{location} key")
            _finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{location}[{index}]")
    elif value is None or isinstance(value, (str, bool)):
        return
    else:
        _fail(location, "unsupported value")


def _materialize_json(value: object, location: str) -> Any:
    if isinstance(value, Mapping):
        return {
            _string(key, f"{location} key"): _materialize_json(child, f"{location}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _materialize_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int, float)):
        if type(value) in (int, float):
            _number(value, location)
        return value
    _fail(location, "unsupported JSON value")


def _parse_body(value: object, location: str) -> None:
    body = _mapping(value, location)
    _require_fields(body, frozenset((
        "body_type", "simulated", "gravity_scale", "gravity_applicable", "position",
        "rotation_degrees", "velocity", "angular_velocity_degrees_per_second",
    )), location)
    body_type = _string(body["body_type"], f"{location}.body_type")
    if body_type not in {"dynamic", "kinematic", "static"}:
        _fail(f"{location}.body_type", "unknown body type")
    if type(body["simulated"]) is not bool:
        _fail(f"{location}.simulated", "expected boolean")
    if type(body["gravity_applicable"]) is not bool:
        _fail(f"{location}.gravity_applicable", "expected boolean")
    gravity_scale = _number(body["gravity_scale"], f"{location}.gravity_scale")
    expected_gravity_applicable = body_type == "dynamic" and body["simulated"] and gravity_scale != 0.0
    if body["gravity_applicable"] != expected_gravity_applicable:
        _fail(f"{location}.gravity_applicable", "does not match body type, simulation state, and gravity scale")
    _vector(body["position"], f"{location}.position")
    _number(body["rotation_degrees"], f"{location}.rotation_degrees")
    _vector(body["velocity"], f"{location}.velocity")
    _number(
        body["angular_velocity_degrees_per_second"],
        f"{location}.angular_velocity_degrees_per_second",
    )


def _parse_collider_shape(value: object, location: str) -> None:
    shape = _mapping(value, location)
    if not shape:
        _fail(location, "geometry is absent")
    kind = _string(shape.get("kind"), f"{location}.kind")
    if kind == "circle":
        _require_fields(shape, frozenset(("kind", "center", "radius")), location)
        _vector(shape["center"], f"{location}.center")
        if _number(shape["radius"], f"{location}.radius") <= 0:
            _fail(f"{location}.radius", "must be positive")
    elif kind == "box":
        _require_fields(shape, frozenset(("kind", "center", "size", "angle_degrees")), location)
        _vector(shape["center"], f"{location}.center")
        size = _vector(shape["size"], f"{location}.size")
        if min(size) <= 0:
            _fail(f"{location}.size", "must be positive")
        _number(shape["angle_degrees"], f"{location}.angle_degrees")
    elif kind == "polygon":
        _require_fields(shape, frozenset(("kind", "paths")), location)
        paths = _sequence(shape["paths"], f"{location}.paths")
        if not paths:
            _fail(f"{location}.paths", "must be nonempty")
        for path_index, raw_path in enumerate(paths):
            path = _sequence(raw_path, f"{location}.paths[{path_index}]")
            if len(path) < 3:
                _fail(f"{location}.paths[{path_index}]", "polygon path needs at least three points")
            for point_index, point in enumerate(path):
                _vector(point, f"{location}.paths[{path_index}][{point_index}]")
    elif kind == "edge":
        _require_fields(shape, frozenset(("kind", "points")), location)
        points = _sequence(shape["points"], f"{location}.points")
        if len(points) < 2:
            _fail(f"{location}.points", "edge needs at least two points")
        for point_index, point in enumerate(points):
            _vector(point, f"{location}.points[{point_index}]")
    elif kind == "capsule":
        _require_fields(shape, frozenset(("kind", "center", "size", "direction", "angle_degrees")), location)
        _vector(shape["center"], f"{location}.center")
        size = _vector(shape["size"], f"{location}.size")
        if min(size) <= 0:
            _fail(f"{location}.size", "must be positive")
        if _string(shape["direction"], f"{location}.direction") not in {"horizontal", "vertical"}:
            _fail(f"{location}.direction", "must be horizontal or vertical")
        _number(shape["angle_degrees"], f"{location}.angle_degrees")
    else:
        _fail(f"{location}.kind", "expected a supported Unity Collider2D shape")


def parse_physics_capture_v2(record: object) -> PhysicsCaptureV2:
    root = _mapping(record, "capture")
    _require_fields(root, frozenset((
        "schema_version", "capture_id", "shot_id", "source_bindings",
        "configured_fixed_step_capture_stride", "pre_intervention_fixed_step",
        "coordinate_convention", "causal_entities",
        "colliders", "fixed_step_samples", "minimum_contact_separation", "frame_records",
        "events", "terminal_evidence",
    )), "capture")
    if root["schema_version"] != SCHEMA_VERSION:
        _fail("capture.schema_version", f"expected {SCHEMA_VERSION}")
    capture_id = _string(root["capture_id"], "capture.capture_id")
    shot_id = _string(root["shot_id"], "capture.shot_id")
    stride = _integer(root["configured_fixed_step_capture_stride"], "capture.configured_fixed_step_capture_stride", positive=True)
    pre_intervention_step = _integer(root["pre_intervention_fixed_step"], "capture.pre_intervention_fixed_step")
    if pre_intervention_step < 0:
        _fail("capture.pre_intervention_fixed_step", "must be nonnegative")

    bindings = _mapping(root["source_bindings"], "capture.source_bindings")
    _require_fields(bindings, _BINDINGS, "capture.source_bindings")
    parsed_bindings = {key: _string(bindings[key], f"capture.source_bindings.{key}") for key in _BINDINGS}

    coordinates = _mapping(root["coordinate_convention"], "capture.coordinate_convention")
    _require_fields(coordinates, frozenset(("world_space", "world_x_axis", "world_y_axis", "world_length_unit")), "capture.coordinate_convention")
    for key in coordinates:
        _string(coordinates[key], f"capture.coordinate_convention.{key}")

    causal_entities = _sequence(root["causal_entities"], "capture.causal_entities")
    if len(causal_entities) > MAX_CAUSAL_ENTITIES:
        _fail("capture.causal_entities", "causal-entity bound exceeded")
    entity_ids = tuple(_string(value, f"capture.causal_entities[{index}]") for index, value in enumerate(causal_entities))
    if not entity_ids or len(set(entity_ids)) != len(entity_ids):
        _fail("capture.causal_entities", "must be nonempty and unique")
    if entity_ids != tuple(sorted(entity_ids)):
        _fail("capture.causal_entities", "must use deterministic order")
    entity_set = set(entity_ids)

    colliders = _sequence(root["colliders"], "capture.colliders")
    if len(colliders) > MAX_COLLIDERS:
        _fail("capture.colliders", "collider bound exceeded")
    collider_ids: set[str] = set()
    collider_order: list[str] = []
    collider_catalog: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(colliders):
        collider = _mapping(value, f"capture.colliders[{index}]")
        _require_fields(collider, frozenset(("collider_id", "entity_id", "geometry_source")), f"capture.colliders[{index}]")
        collider_id = _string(collider["collider_id"], f"capture.colliders[{index}].collider_id")
        if collider_id in collider_ids:
            _fail("capture.colliders", "collider IDs must be unique")
        collider_ids.add(collider_id)
        collider_order.append(collider_id)
        collider_catalog[collider_id] = collider
        if _string(collider["entity_id"], f"capture.colliders[{index}].entity_id") not in entity_set:
            _fail(f"capture.colliders[{index}].entity_id", "unresolved causal entity")
        if collider["geometry_source"] != "unity_collider_2d":
            _fail(f"capture.colliders[{index}].geometry_source", "geometry must be Unity-authored")
    if collider_order != sorted(collider_order):
        _fail("capture.colliders", "must use deterministic order")

    samples = _sequence(root["fixed_step_samples"], "capture.fixed_step_samples")
    if len(samples) > MAX_FIXED_STEP_SAMPLES:
        _fail("capture.fixed_step_samples", "fixed-step sample bound exceeded")
    steps: list[int] = []
    observed_contacts: list[tuple[float, int, str]] = []
    for index, value in enumerate(samples):
        sample = _mapping(value, f"capture.fixed_step_samples[{index}]")
        _require_fields(sample, frozenset((
            "fixed_step", "complete_raw_non_trigger_contacts", "world", "entities", "colliders",
            "contacts", "supports",
        )), f"capture.fixed_step_samples[{index}]")
        step = _integer(sample["fixed_step"], f"capture.fixed_step_samples[{index}].fixed_step")
        if step < 0:
            _fail(f"capture.fixed_step_samples[{index}].fixed_step", "must be nonnegative")
        steps.append(step)
        if type(sample["complete_raw_non_trigger_contacts"]) is not bool or not sample["complete_raw_non_trigger_contacts"]:
            _fail(f"capture.fixed_step_samples[{index}].complete_raw_non_trigger_contacts", "contact enumeration is incomplete")
        world = _mapping(sample["world"], f"capture.fixed_step_samples[{index}].world")
        _require_fields(world, frozenset(("world_id", "gravity_vector")), f"capture.fixed_step_samples[{index}].world")
        _string(world["world_id"], f"capture.fixed_step_samples[{index}].world.world_id")
        _vector(world["gravity_vector"], f"capture.fixed_step_samples[{index}].world.gravity_vector")

        sample_colliders = _sequence(sample["colliders"], f"capture.fixed_step_samples[{index}].colliders")
        if len(sample_colliders) > MAX_COLLIDERS:
            _fail(f"capture.fixed_step_samples[{index}].colliders", "collider bound exceeded")
        sample_collider_ids: list[str] = []
        for collider_index, collider_value in enumerate(sample_colliders):
            location = f"capture.fixed_step_samples[{index}].colliders[{collider_index}]"
            collider = _mapping(collider_value, location)
            _require_fields(collider, frozenset(("collider_id", "entity_id", "geometry_source", "enabled", "is_trigger", "shape")), location)
            collider_id = _string(collider["collider_id"], f"{location}.collider_id")
            entity_id = _string(collider["entity_id"], f"{location}.entity_id")
            if collider_id not in collider_ids:
                _fail(f"{location}.collider_id", "is absent from the root collider catalog")
            catalog = collider_catalog[collider_id]
            if catalog["entity_id"] != entity_id or collider["geometry_source"] != "unity_collider_2d":
                _fail(location, "collider catalog identity is stale")
            if type(collider["enabled"]) is not bool or type(collider["is_trigger"]) is not bool:
                _fail(location, "collider enabled and trigger state must be boolean")
            _parse_collider_shape(collider["shape"], f"{location}.shape")
            sample_collider_ids.append(collider_id)
        if sample_collider_ids != sorted(set(sample_collider_ids)):
            _fail(f"capture.fixed_step_samples[{index}].colliders", "must be unique and use deterministic order")
        sample_collider_set = set(sample_collider_ids)

        entities = _sequence(sample["entities"], f"capture.fixed_step_samples[{index}].entities")
        seen_entities: set[str] = set()
        declared_context: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
        for entity_index, entity_value in enumerate(entities):
            entity = _mapping(entity_value, f"capture.fixed_step_samples[{index}].entities[{entity_index}]")
            _require_fields(entity, frozenset((
                "entity_id", "scenario_object_id", "lifecycle", "body_present", "body",
                "contact_ids", "supported_by_entity_ids", "supports_entity_ids",
            )), f"capture.fixed_step_samples[{index}].entities[{entity_index}]")
            entity_id = _string(entity["entity_id"], f"capture.fixed_step_samples[{index}].entities[{entity_index}].entity_id")
            if entity_id not in entity_set:
                _fail(f"capture.fixed_step_samples[{index}].entities[{entity_index}].entity_id", "unresolved causal entity")
            seen_entities.add(entity_id)
            _string(entity["scenario_object_id"], f"capture.fixed_step_samples[{index}].entities[{entity_index}].scenario_object_id")
            if _string(entity["lifecycle"], f"capture.fixed_step_samples[{index}].entities[{entity_index}].lifecycle") not in {"active", "inactive", "destroyed"}:
                _fail(f"capture.fixed_step_samples[{index}].entities[{entity_index}].lifecycle", "unknown lifecycle")
            if type(entity["body_present"]) is not bool:
                _fail(f"capture.fixed_step_samples[{index}].entities[{entity_index}].body_present", "expected boolean")
            if entity["body_present"] != (entity["body"] is not None):
                _fail(f"capture.fixed_step_samples[{index}].entities[{entity_index}]", "body presence does not match body record")
            if entity["body"] is not None:
                _parse_body(entity["body"], f"capture.fixed_step_samples[{index}].entities[{entity_index}].body")
            context_values: list[tuple[str, ...]] = []
            for context_field in ("contact_ids", "supported_by_entity_ids", "supports_entity_ids"):
                context = tuple(
                    _string(item, f"capture.fixed_step_samples[{index}].entities[{entity_index}].{context_field}")
                    for item in _sequence(
                        entity[context_field],
                        f"capture.fixed_step_samples[{index}].entities[{entity_index}].{context_field}",
                    )
                )
                if context != tuple(sorted(set(context))):
                    _fail(
                        f"capture.fixed_step_samples[{index}].entities[{entity_index}].{context_field}",
                        "must be unique and deterministic",
                    )
                if context_field != "contact_ids" and any(item not in entity_set for item in context):
                    _fail(
                        f"capture.fixed_step_samples[{index}].entities[{entity_index}].{context_field}",
                        "contains an unresolved causal entity",
                    )
                context_values.append(context)
            declared_context[entity_id] = (
                context_values[0], context_values[1], context_values[2]
            )
        if seen_entities != entity_set:
            _fail(f"capture.fixed_step_samples[{index}].entities", "causal-entity lifecycle evidence is incomplete")
        if tuple(entity["entity_id"] for entity in entities) != tuple(sorted(entity_set)):
            _fail(f"capture.fixed_step_samples[{index}].entities", "must use deterministic order")

        contact_ids: set[str] = set()
        contacts = _sequence(sample["contacts"], f"capture.fixed_step_samples[{index}].contacts")
        if len(contacts) > MAX_CONTACTS_PER_STEP:
            _fail(f"capture.fixed_step_samples[{index}].contacts", "per-step contact bound exceeded")
        for contact_index, contact_value in enumerate(contacts):
            contact = _mapping(contact_value, f"capture.fixed_step_samples[{index}].contacts[{contact_index}]")
            _require_fields(contact, frozenset(("contact_id", "entity_a_id", "entity_b_id", "collider_a_id", "collider_b_id", "point", "normal_a_to_b", "separation")), f"capture.fixed_step_samples[{index}].contacts[{contact_index}]")
            contact_id = _string(contact["contact_id"], f"capture.fixed_step_samples[{index}].contacts[{contact_index}].contact_id")
            if contact_id in contact_ids:
                _fail(f"capture.fixed_step_samples[{index}].contacts", "contact IDs must be unique within a fixed step")
            contact_ids.add(contact_id)
            for key in ("entity_a_id", "entity_b_id"):
                if _string(contact[key], f"capture.fixed_step_samples[{index}].contacts[{contact_index}].{key}") not in entity_set:
                    _fail(f"capture.fixed_step_samples[{index}].contacts[{contact_index}].{key}", "unresolved causal entity")
            for key in ("collider_a_id", "collider_b_id"):
                if _string(contact[key], f"capture.fixed_step_samples[{index}].contacts[{contact_index}].{key}") not in sample_collider_set:
                    _fail(f"capture.fixed_step_samples[{index}].contacts[{contact_index}].{key}", "unresolved collider geometry at the same fixed step")
            _vector(contact["point"], f"capture.fixed_step_samples[{index}].contacts[{contact_index}].point")
            _vector(contact["normal_a_to_b"], f"capture.fixed_step_samples[{index}].contacts[{contact_index}].normal_a_to_b")
            separation = _number(contact["separation"], f"capture.fixed_step_samples[{index}].contacts[{contact_index}].separation")
            observed_contacts.append((separation, step, contact_id))
        if tuple(contact["contact_id"] for contact in contacts) != tuple(sorted(contact_ids)):
            _fail(f"capture.fixed_step_samples[{index}].contacts", "must use deterministic order")
        supports = _sequence(sample["supports"], f"capture.fixed_step_samples[{index}].supports")
        expected_contacts = {entity_id: set() for entity_id in entity_set}
        for contact_value in _sequence(sample["contacts"], f"capture.fixed_step_samples[{index}].contacts"):
            contact = _mapping(contact_value, f"capture.fixed_step_samples[{index}].contacts")
            expected_contacts[contact["entity_a_id"]].add(contact["contact_id"])
            expected_contacts[contact["entity_b_id"]].add(contact["contact_id"])
        expected_supported_by = {entity_id: set() for entity_id in entity_set}
        expected_supports = {entity_id: set() for entity_id in entity_set}
        for support_index, support_value in enumerate(supports):
            support = _mapping(support_value, f"capture.fixed_step_samples[{index}].supports[{support_index}]")
            _require_fields(support, frozenset(("supporter_entity_id", "supported_entity_id", "contact_ids")), f"capture.fixed_step_samples[{index}].supports[{support_index}]")
            for key in ("supporter_entity_id", "supported_entity_id"):
                if _string(support[key], f"capture.fixed_step_samples[{index}].supports[{support_index}].{key}") not in entity_set:
                    _fail(f"capture.fixed_step_samples[{index}].supports[{support_index}].{key}", "unresolved causal entity")
            for contact_id in _sequence(support["contact_ids"], f"capture.fixed_step_samples[{index}].supports[{support_index}].contact_ids"):
                if _string(contact_id, f"capture.fixed_step_samples[{index}].supports[{support_index}].contact_ids") not in contact_ids:
                    _fail(f"capture.fixed_step_samples[{index}].supports[{support_index}].contact_ids", "unresolved same-step contact")
            supporter = support["supporter_entity_id"]
            supported = support["supported_entity_id"]
            expected_supported_by[supported].add(supporter)
            expected_supports[supporter].add(supported)
        support_order = [
            (support["supporter_entity_id"], support["supported_entity_id"], tuple(support["contact_ids"]))
            for support in supports
        ]
        if support_order != sorted(support_order):
            _fail(f"capture.fixed_step_samples[{index}].supports", "must use deterministic order")
        for entity_id in entity_set:
            expected_context = (
                tuple(sorted(expected_contacts[entity_id])),
                tuple(sorted(expected_supported_by[entity_id])),
                tuple(sorted(expected_supports[entity_id])),
            )
            if declared_context[entity_id] != expected_context:
                _fail(
                    f"capture.fixed_step_samples[{index}].entities",
                    "entity support/contact context is incomplete",
                )

    if not steps or any(current != previous + 1 for previous, current in zip(steps, steps[1:])):
        _fail("capture.fixed_step_samples", "fixed-step contact coverage has a gap")
    if steps[0] != pre_intervention_step:
        _fail("capture.pre_intervention_fixed_step", "does not match the first pre-intervention sample")

    minimum = _mapping(root["minimum_contact_separation"], "capture.minimum_contact_separation")
    _require_fields(
        minimum,
        frozenset(("observed", "separation", "contact_id", "fixed_step")),
        "capture.minimum_contact_separation",
    )
    if type(minimum["observed"]) is not bool:
        _fail("capture.minimum_contact_separation.observed", "expected boolean")
    if observed_contacts:
        separation, minimum_step, minimum_contact_id = min(observed_contacts)
        expected_minimum = {
            "observed": True,
            "separation": separation,
            "contact_id": minimum_contact_id,
            "fixed_step": minimum_step,
        }
    else:
        expected_minimum = {
            "observed": False,
            "separation": None,
            "contact_id": None,
            "fixed_step": None,
        }
    if dict(minimum) != expected_minimum:
        _fail("capture.minimum_contact_separation", "does not match the recomputed minimum")

    frame_records = _sequence(root["frame_records"], "capture.frame_records")
    if len(frame_records) > MAX_FRAME_RECORDS:
        _fail("capture.frame_records", "frame-record bound exceeded")
    frame_steps: list[int] = []
    forced_terminal_flags: list[bool] = []
    for index, value in enumerate(frame_records):
        frame = _mapping(value, f"capture.frame_records[{index}]")
        _require_fields(frame, frozenset(("fixed_step", "state_id", "forced_terminal")), f"capture.frame_records[{index}]")
        frame_steps.append(_integer(frame["fixed_step"], f"capture.frame_records[{index}].fixed_step"))
        _string(frame["state_id"], f"capture.frame_records[{index}].state_id")
        if type(frame["forced_terminal"]) is not bool:
            _fail(f"capture.frame_records[{index}].forced_terminal", "expected boolean")
        forced_terminal_flags.append(frame["forced_terminal"])
    if len(frame_steps) < 2:
        _fail("capture.frame_records", "retained fixed steps do not validate configured capture stride")
    if any(forced_terminal_flags[:-1]):
        _fail("capture.frame_records", "only the final record may be forced for off-grid termination")
    scheduled_steps = [step for step, forced in zip(frame_steps, forced_terminal_flags) if not forced]
    if not scheduled_steps or any(
        current - previous != stride
        for previous, current in zip(scheduled_steps, scheduled_steps[1:])
    ):
        _fail("capture.frame_records", "retained fixed steps do not validate configured capture stride")
    if forced_terminal_flags[-1]:
        terminal_offset = frame_steps[-1] - scheduled_steps[-1]
        if terminal_offset <= 0 or terminal_offset >= stride:
            _fail("capture.frame_records", "forced terminal record is not off the configured stride grid")
    if frame_steps[0] != steps[0] or frame_steps[-1] != steps[-1] or any(step not in set(steps) for step in frame_steps):
        _fail("capture.frame_records", "frames are not covered by retained fixed-step evidence")

    event_ids: set[str] = set()
    event_steps: dict[str, int] = {}
    event_types: dict[str, str] = {}
    event_order: list[tuple[int, str]] = []
    events = _sequence(root["events"], "capture.events")
    if len(events) > MAX_EVENTS:
        _fail("capture.events", "event bound exceeded")
    for index, value in enumerate(events):
        event = _mapping(value, f"capture.events[{index}]")
        _require_fields(event, frozenset(("event_id", "event_type", "fixed_step", "participants", "payload")), f"capture.events[{index}]")
        event_id = _string(event["event_id"], f"capture.events[{index}].event_id")
        if event_id in event_ids:
            _fail("capture.events", "event IDs must be unique")
        event_ids.add(event_id)
        event_type = _string(event["event_type"], f"capture.events[{index}].event_type")
        if event_type not in EVENT_PARTICIPANT_COUNTS:
            _fail(f"capture.events[{index}].event_type", "unknown event type")
        event_step = _integer(event["fixed_step"], f"capture.events[{index}].fixed_step")
        if event_step not in set(steps):
            _fail(f"capture.events[{index}].fixed_step", "event step lacks retained physical evidence")
        event_steps[event_id] = event_step
        event_types[event_id] = event_type
        event_order.append((event_step, event_id))
        participants = tuple(
            _string(participant, f"capture.events[{index}].participants")
            for participant in _sequence(event["participants"], f"capture.events[{index}].participants")
        )
        if len(participants) != EVENT_PARTICIPANT_COUNTS[event_type]:
            _fail(f"capture.events[{index}].participants", "event participant count is incomplete")
        if participants != tuple(sorted(set(participants))):
            _fail(f"capture.events[{index}].participants", "must be unique and deterministic")
        for participant in participants:
            if _string(participant, f"capture.events[{index}].participants") not in entity_set:
                _fail(f"capture.events[{index}].participants", "unresolved causal entity")
        _finite_tree(_mapping(event["payload"], f"capture.events[{index}].payload"), f"capture.events[{index}].payload")
    if event_order != sorted(event_order):
        _fail("capture.events", "must use deterministic order")

    terminal = _mapping(root["terminal_evidence"], "capture.terminal_evidence")
    _require_fields(terminal, frozenset(("reason", "fixed_step", "event_id")), "capture.terminal_evidence")
    terminal_reason = _string(terminal["reason"], "capture.terminal_evidence.reason")
    terminal_step = _integer(terminal["fixed_step"], "capture.terminal_evidence.fixed_step")
    if terminal_step not in set(steps):
        _fail("capture.terminal_evidence.fixed_step", "termination lacks retained physical evidence")
    terminal_event_id = _string(terminal["event_id"], "capture.terminal_evidence.event_id")
    if terminal_event_id not in event_ids:
        _fail("capture.terminal_evidence.event_id", "unresolved terminal event")
    if event_steps[terminal_event_id] != terminal_step:
        _fail("capture.terminal_evidence", "terminal event and evidence fixed steps differ")
    if event_types[terminal_event_id] != terminal_reason:
        _fail("capture.terminal_evidence.reason", "terminal reason does not match its event type")
    if frame_steps[-1] != terminal_step:
        _fail("capture.frame_records", "final retained frame does not cover termination")

    return PhysicsCaptureV2(capture_id, shot_id, stride, parsed_bindings, root)


def bind_physics_capture_v2_engine(
    engine_record: Mapping[str, object],
    source_bindings: Mapping[str, object],
) -> PhysicsCaptureV2:
    """Add collector-owned frozen-plan identities to Unity-authored evidence."""
    engine = _mapping(engine_record, "engine_capture")
    if engine.get("schema_version") != ENGINE_SCHEMA_VERSION:
        _fail("engine_capture.schema_version", f"expected {ENGINE_SCHEMA_VERSION}")
    if "source_bindings" in engine:
        _fail("engine_capture.source_bindings", "must be added by the collector")
    final_record = _materialize_json(engine, "engine_capture")
    final_record["schema_version"] = SCHEMA_VERSION
    final_record["source_bindings"] = _materialize_json(source_bindings, "source_bindings")
    return parse_physics_capture_v2(final_record)


def normalized_initial_engine_state_identity(capture: PhysicsCaptureV2) -> str:
    """Identify the first engine state without rollout- or request-specific IDs."""
    record = capture.record
    initial_sample = _materialize_json(record["fixed_step_samples"][0], "initial_sample")
    initial_sample.pop("fixed_step")
    contact_identities = {
        contact["contact_id"]: f"initial-contact:{index:04d}"
        for index, contact in enumerate(initial_sample["contacts"])
    }
    for contact in initial_sample["contacts"]:
        contact["contact_id"] = contact_identities[contact["contact_id"]]
    for entity in initial_sample["entities"]:
        entity["contact_ids"] = [
            contact_identities[contact_id]
            for contact_id in entity["contact_ids"]
        ]
    for support in initial_sample["supports"]:
        support["contact_ids"] = [
            contact_identities[contact_id]
            for contact_id in support["contact_ids"]
        ]
    payload = {
        "schema": "normalized_initial_engine_state_v1",
        "coordinate_convention": _materialize_json(record["coordinate_convention"], "coordinate_convention"),
        "causal_entities": _materialize_json(record["causal_entities"], "causal_entities"),
        "collider_catalog": _materialize_json(record["colliders"], "collider_catalog"),
        "sample": initial_sample,
    }
    semantic_keys = json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"normalized-initial-engine-state-v1:{semantic_keys}"


def load_physics_capture_v2(path: Path) -> PhysicsCaptureV2:
    """Load exactly one v2 sidecar; legacy sidecars are intentionally unsupported."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PhysicsCaptureV2Error(f"{path}: cannot read capture") from exc
    if size > MAX_CAPTURE_BYTES:
        raise PhysicsCaptureV2Error(f"{path}: capture exceeds the byte bound")
    try:
        record: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicsCaptureV2Error(f"{path}: malformed JSON") from exc
    return parse_physics_capture_v2(record)
