from __future__ import annotations

from dataclasses import asdict
import json
from typing import Final, TypeAlias

from scripts.physics_capture_types import EventRecord, EventType, PhysicsCapture, RawContact


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
REQUIRED_ROLLOUT_SEMANTICS_FIELDS: Final = (
    "initial_engine_state_identity",
    "intervention_event_id",
    "termination_reason",
    "termination_fixed_step",
    "termination_event_id",
    "terminal_state_fixed_step",
)
SEMANTICS_METADATA_FIELDS: Final = frozenset((
    "initial_engine_state_identity",
    "intervention_event_id",
    "termination_reason",
    "termination_fixed_step",
    "termination_event_id",
    "terminal_state_fixed_step",
    "expected_initial_engine_state_identity",
    "scenario_context",
))
_TERMINAL_EVENT_TYPES: Final = frozenset((
    EventType.LEVEL_CLEARED,
    EventType.LEVEL_FAILED,
    EventType.BIRD_EXHAUSTED,
    EventType.STABLE_ENTERED,
))


class PhysicsRolloutSemanticsError(ValueError):
    pass


def initial_engine_state_identity(capture: PhysicsCapture) -> str:
    if not capture.states:
        raise PhysicsRolloutSemanticsError("a rollout needs an initial state")
    state = capture.states[0]
    engine_content = {
        "coordinates": asdict(state.clock.coordinates),
        "nodes": [asdict(node) for node in state.nodes],
        "raw_contacts": [asdict(contact) for contact in state.raw_contacts],
        "support_edges": [asdict(edge) for edge in state.support_edges],
    }
    semantic_keys = json.dumps(engine_content, sort_keys=True, separators=(",", ":"))
    return f"normalized-initial-engine-state-v1:{semantic_keys}"


def _payload(event: EventRecord) -> JsonObject:
    try:
        value: JsonValue = json.loads(event.payload_json)
    except json.JSONDecodeError as error:
        raise PhysicsRolloutSemanticsError(f"event {event.event_id} payload is malformed") from error
    if not isinstance(value, dict):
        raise PhysicsRolloutSemanticsError(f"event {event.event_id} payload is not an object")
    return value


def _contact_has_canonical_id(contact: RawContact, contact_id: str, fixed_step: int) -> bool:
    prefix = (
        f"contact:{fixed_step}:{contact.entity_a_id}|{contact.collider_a_id}:"
        f"{contact.entity_b_id}|{contact.collider_b_id}:"
    )
    point_index = contact_id.removeprefix(prefix)
    return contact_id.startswith(prefix) and point_index.isdecimal()


def _validate_collision_evidence(capture: PhysicsCapture) -> None:
    retained_contacts: dict[str, list[RawContact]] = {}
    for state in capture.states:
        for contact in state.raw_contacts:
            retained_contacts.setdefault(str(contact.contact_id), []).append(contact)

    for event in capture.events:
        if event.event_type is not EventType.COLLISION:
            continue
        payload = _payload(event)
        contact_ids = payload.get("contact_ids")
        if (
            not isinstance(contact_ids, list)
            or not contact_ids
            or not all(isinstance(contact_id, str) for contact_id in contact_ids)
        ):
            raise PhysicsRolloutSemanticsError(
                f"collision event {event.event_id} has no contact_ids evidence"
            )
        participants = frozenset(str(participant) for participant in event.participants)
        if len(participants) != 2:
            raise PhysicsRolloutSemanticsError(
                f"collision event {event.event_id} participants are not an unordered pair"
            )
        for contact_id in contact_ids:
            contacts = retained_contacts.get(contact_id)
            if not contacts:
                raise PhysicsRolloutSemanticsError(
                    f"collision event {event.event_id} contact {contact_id} is not retained"
                )
            if not any(
                _contact_has_canonical_id(contact, contact_id, event.clock.fixed_step)
                and frozenset((str(contact.entity_a_id), str(contact.entity_b_id))) == participants
                for contact in contacts
            ):
                raise PhysicsRolloutSemanticsError(
                    f"collision event {event.event_id} contact evidence does not match its step and participants"
                )


def _first_event(events: tuple[EventRecord, ...]) -> EventRecord:
    return min(events, key=lambda event: (event.clock.sequence, str(event.event_id)))


def _termination(capture: PhysicsCapture) -> tuple[str, int, str | None, int]:
    terminal_state_fixed_step = capture.states[-1].clock.fixed_step
    for event in capture.events:
        if event.event_type in _TERMINAL_EVENT_TYPES and event.clock.fixed_step > terminal_state_fixed_step:
            raise PhysicsRolloutSemanticsError(
                f"terminal event {event.event_id} occurs after the final state"
            )

    level_events = tuple(
        event for event in capture.events
        if event.event_type in (EventType.LEVEL_CLEARED, EventType.LEVEL_FAILED)
    )
    if level_events:
        event = _first_event(level_events)
        return str(event.event_type), event.clock.fixed_step, str(event.event_id), terminal_state_fixed_step

    exhausted_events = tuple(
        event for event in capture.events if event.event_type is EventType.BIRD_EXHAUSTED
    )
    if exhausted_events:
        event = _first_event(exhausted_events)
        return str(event.event_type), event.clock.fixed_step, str(event.event_id), terminal_state_fixed_step

    if capture.events and capture.events[-1].event_type is EventType.STABLE_ENTERED:
        event = capture.events[-1]
        return "post_intervention_stable", event.clock.fixed_step, str(event.event_id), terminal_state_fixed_step

    return "rollout_ceiling", terminal_state_fixed_step, None, terminal_state_fixed_step


def validate_physics_rollout_semantics(
    capture: PhysicsCapture,
    *,
    expected_initial_engine_state_identity: str | None = None,
) -> JsonObject:
    if len(capture.states) < 2:
        raise PhysicsRolloutSemanticsError("a shot needs at least two retained states")
    launches = tuple(event for event in capture.events if event.event_type is EventType.BIRD_LAUNCHED)
    if len(launches) != 1:
        raise PhysicsRolloutSemanticsError("a shot needs exactly one bird_launched event")
    if launches[0].clock.fixed_step < capture.states[0].clock.fixed_step:
        raise PhysicsRolloutSemanticsError("bird_launched occurs before the first retained state")
    _validate_collision_evidence(capture)
    identity = initial_engine_state_identity(capture)
    if expected_initial_engine_state_identity is not None and not isinstance(expected_initial_engine_state_identity, str):
        raise PhysicsRolloutSemanticsError("expected initial engine state identity is not a string")
    termination_reason, termination_fixed_step, termination_event_id, terminal_state_fixed_step = _termination(capture)
    return {
        "initial_engine_state_identity": identity,
        "intervention_event_id": str(launches[0].event_id),
        "termination_reason": termination_reason,
        "termination_fixed_step": termination_fixed_step,
        "termination_event_id": termination_event_id,
        "terminal_state_fixed_step": terminal_state_fixed_step,
    }
