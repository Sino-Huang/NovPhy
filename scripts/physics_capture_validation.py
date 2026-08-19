from __future__ import annotations

import json
import math
from typing import TypeAlias, assert_never

from scripts.physics_capture_contract import (
    ContractErrorCode,
    EVENT_SIDECAR,
    STATE_SIDECAR,
    contract_error,
)
from scripts.physics_capture_types import (
    EventId,
    EventRecord,
    EventType,
    PhysicsCapture,
    PhysicsViolationEngineEvidence,
    StateFrame,
)


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _validate_streams(capture: PhysicsCapture, total_bytes: int) -> None:
    header = capture.header
    if header.capture_status != "complete" or header.state_sidecar != STATE_SIDECAR or header.event_sidecar != EVENT_SIDECAR:
        raise contract_error(ContractErrorCode.INVALID_VALUE, "physics_state.jsonl:1", "header layout/status differs from v1")
    all_clocks = (header.clock,) + tuple(state.clock for state in capture.states) + tuple(event.clock for event in capture.events)
    identities = {(clock.capture_id, clock.shot_id, clock.coordinates) for clock in all_clocks}
    if len(identities) != 1:
        raise contract_error(ContractErrorCode.CAPTURE_MISMATCH, "sidecars", "identity or coordinate declaration mismatch")
    state_sequences = (header.clock.sequence,) + tuple(state.clock.sequence for state in capture.states)
    event_sequences = tuple(event.clock.sequence for event in capture.events)
    if state_sequences != tuple(sorted(set(state_sequences))) or event_sequences != tuple(sorted(set(event_sequences))):
        raise contract_error(ContractErrorCode.SEQUENCE_ORDER, "sidecars", "sequence must be strictly increasing within each sidecar")
    state_clock_keys = tuple((state.clock.render_frame, state.clock.render_time, state.clock.fixed_step, state.clock.fixed_time) for state in capture.states)
    if state_clock_keys != tuple(sorted(state_clock_keys)):
        raise contract_error(ContractErrorCode.CLOCK_ORDER, "physics_state.jsonl", "state clocks must be monotonic")
    if len(capture.states) + 1 > header.limits.max_state_records or len(capture.events) > header.limits.max_event_records or total_bytes > header.limits.max_total_bytes:
        raise contract_error(ContractErrorCode.INVALID_VALUE, "sidecars", "record limit exceeded")


def _validate_support(states: tuple[StateFrame, ...]) -> None:
    contacts_by_step = {(state.clock.fixed_step, contact.contact_id): (contact, state) for state in states for contact in state.raw_contacts}
    for state in states:
        for support in state.support_edges:
            if support.rule_version != "support_v1" or support.evidence_fixed_steps != (state.clock.fixed_step - 1, state.clock.fixed_step):
                raise contract_error(ContractErrorCode.NONPERSISTENT_SUPPORT, str(support.support_id), "evidence steps are not consecutive")
            for pair in zip(support.evidence_fixed_steps, support.evidence_contact_ids, strict=True):
                item = contacts_by_step.get(pair)
                if item is None:
                    raise contract_error(ContractErrorCode.NONPERSISTENT_SUPPORT, str(support.support_id), "retained contact evidence is missing")
                contact, evidence_state = item
                if {contact.entity_a_id, contact.entity_b_id} != {support.supporter_id, support.supported_id} or abs(contact.normal_a_to_b.y) < 0.5:
                    raise contract_error(ContractErrorCode.NONPERSISTENT_SUPPORT, str(support.support_id), "contact pair/normal fails support_v1")
                centers = {node.entity_id: node.world_pose.position.y for node in evidence_state.nodes}
                if support.supporter_id not in centers or support.supported_id not in centers or centers[support.supported_id] - centers[support.supporter_id] < 0.0001:
                    raise contract_error(ContractErrorCode.NONPERSISTENT_SUPPORT, str(support.support_id), "vertical centre ordering fails support_v1")


def _payload(event: EventRecord, fields: frozenset[str]) -> JsonObject:
    value: JsonValue = json.loads(event.payload_json)
    if not isinstance(value, dict) or value.keys() != fields:
        raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "payload fields differ from taxonomy")
    return value


def _payload_number(payload: JsonObject, field: str, event: EventRecord) -> float:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), field)
    return float(value)


def _validate_event_payload(event: EventRecord) -> None:
    match event.event_type:
        case EventType.BIRD_LAUNCHED:
            velocity = _payload(event, frozenset(("launch_velocity",)))["launch_velocity"]
            if not isinstance(velocity, dict) or velocity.keys() != {"x", "y"}:
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "launch_velocity")
            _payload_number(velocity, "x", event)
            _payload_number(velocity, "y", event)
        case EventType.COLLISION:
            payload = _payload(event, frozenset(("contact_ids", "relative_speed")))
            contacts = payload["contact_ids"]
            if not isinstance(contacts, list) or not contacts or not all(isinstance(item, str) for item in contacts) or contacts != sorted(set(contacts)) or _payload_number(payload, "relative_speed", event) < 0:
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "collision payload")
        case EventType.EXPLOSION:
            if _payload_number(_payload(event, frozenset(("radius_unity_units",))), "radius_unity_units", event) < 0:
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "explosion radius")
        case EventType.ENTITY_DESTROYED | EventType.PIG_REMOVED | EventType.LEVEL_FAILED:
            reason = _payload(event, frozenset(("reason",)))["reason"]
            if not isinstance(reason, str) or not reason:
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "reason")
        case EventType.BIRD_EXHAUSTED:
            if _payload(event, frozenset(("birds_remaining",)))["birds_remaining"] != 0:
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "birds_remaining")
        case EventType.STABLE_ENTERED | EventType.STABLE_EXITED:
            debounce = _payload(event, frozenset(("debounce_fixed_steps",)))["debounce_fixed_steps"]
            if isinstance(debounce, bool) or not isinstance(debounce, int) or debounce < 1:
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "debounce_fixed_steps")
        case EventType.LEVEL_CLEARED:
            score = _payload(event, frozenset(("score",)))["score"]
            if isinstance(score, bool) or not isinstance(score, int):
                raise contract_error(ContractErrorCode.INVALID_EVENT, str(event.event_id), "score")
        case unreachable:
            assert_never(unreachable)


def _validate_events(events: tuple[EventRecord, ...]) -> None:
    for event in events:
        _validate_event_payload(event)
    rank = {event_type: index for index, event_type in enumerate(EventType)}
    keys = tuple((event.clock.fixed_step, event.clock.render_frame, rank[event.event_type], event.participants, event.event_id) for event in events)
    if keys != tuple(sorted(keys)):
        raise contract_error(ContractErrorCode.DETERMINISTIC_ORDER, "physics_events.jsonl", "event order differs from v1")
    expected_ids = tuple(EventId(f"event:{index:08d}") for index in range(len(events)))
    if tuple(event.event_id for event in events) != expected_ids or tuple(event.clock.sequence for event in events) != tuple(range(len(events))):
        raise contract_error(ContractErrorCode.SEQUENCE_ORDER, "physics_events.jsonl", "event IDs must follow assigned sequence")
    collisions = [(event.clock.fixed_step, event.participants) for event in events if event.event_type is EventType.COLLISION]
    if len(collisions) != len(set(collisions)):
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "duplicate collision pair in fixed step")
    one_shot_types = (EventType.BIRD_LAUNCHED, EventType.BIRD_EXHAUSTED, EventType.LEVEL_CLEARED, EventType.LEVEL_FAILED)
    if any(sum(event.event_type is event_type for event in events) > 1 for event_type in one_shot_types):
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "one-shot event repeated")
    if sum(event.event_type in (EventType.LEVEL_CLEARED, EventType.LEVEL_FAILED) for event in events) > 1:
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "clear/fail are mutually exclusive")
    entity_types = (EventType.EXPLOSION, EventType.ENTITY_DESTROYED, EventType.PIG_REMOVED)
    entity_events = [(event.event_type, event.participants) for event in events if event.event_type in entity_types]
    if any(len(participants) != 1 for _, participants in entity_events) or len(entity_events) != len(set(entity_events)):
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "entity lifecycle event repeated")
    pair_types = (EventType.BIRD_LAUNCHED, EventType.COLLISION)
    if any(len(event.participants) != (2 if event.event_type is EventType.COLLISION else 1) for event in events if event.event_type in pair_types):
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "event participant cardinality")
    participant_free = (EventType.BIRD_EXHAUSTED, EventType.STABLE_ENTERED, EventType.STABLE_EXITED, EventType.LEVEL_CLEARED, EventType.LEVEL_FAILED)
    if any(event.participants for event in events if event.event_type in participant_free):
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "event participants must be empty")
    stable_events = tuple(event.event_type for event in events if event.event_type in (EventType.STABLE_ENTERED, EventType.STABLE_EXITED))
    if any(previous is current for previous, current in zip(stable_events, stable_events[1:])):
        raise contract_error(ContractErrorCode.INVALID_EVENT, "physics_events.jsonl", "stable transitions must alternate")


def _validate_violation_evidence(
    capture: PhysicsCapture,
    evidence: tuple[PhysicsViolationEngineEvidence, ...],
) -> None:
    if not evidence:
        return
    state_sequences = {state.clock.sequence for state in capture.states}
    capture_id = capture.header.clock.capture_id
    identities = {(item.capture_id, item.shot_id) for item in evidence}
    if len(identities) != 1 or next(iter(identities))[0] != capture_id:
        raise contract_error(ContractErrorCode.CAPTURE_MISMATCH, "violation evidence", "engine evidence identity mismatch")
    sequences = tuple(item.sequence for item in evidence)
    if sequences != tuple(sorted(set(sequences))) or any(sequence not in state_sequences for sequence in sequences):
        raise contract_error(ContractErrorCode.SEQUENCE_ORDER, "violation evidence", "evidence sequence must identify one retained state")
    for item in evidence:
        coverage = item.coverage
        trace = item.terminal_trace
        if trace.samples:
            if coverage.first_fixed_step is None or coverage.last_fixed_step is None:
                raise contract_error(ContractErrorCode.INVALID_VALUE, "violation evidence", "trace has no coverage")
            if trace.samples[0].fixed_step < coverage.first_fixed_step or trace.samples[-1].fixed_step > coverage.last_fixed_step:
                raise contract_error(ContractErrorCode.INVALID_VALUE, "violation evidence", "trace lies outside coverage")
        minimum = item.minimum_contact_separation
        if minimum.observed and (
            coverage.first_fixed_step is None
            or minimum.fixed_step is None
            or not coverage.first_fixed_step <= minimum.fixed_step <= coverage.last_fixed_step
        ):
            raise contract_error(ContractErrorCode.INVALID_VALUE, "violation evidence", "minimum contact lies outside coverage")
        for sample in trace.samples:
            for entity in sample.entities:
                for edge in entity.support_v1.edges:
                    expected_id = f"support:{edge.supporter_id}->{entity.entity_id}"
                    if edge.support_id != expected_id or edge.evidence_fixed_steps != (sample.fixed_step - 1, sample.fixed_step):
                        raise contract_error(ContractErrorCode.NONPERSISTENT_SUPPORT, str(edge.support_id), "engine support_v1 evidence is not authoritative for the sample")
    for previous, current in zip(evidence, evidence[1:]):
        if current.coverage.sample_count < previous.coverage.sample_count:
            raise contract_error(ContractErrorCode.CLOCK_ORDER, "violation evidence", "coverage regressed")
        if previous.coverage.first_fixed_step != current.coverage.first_fixed_step:
            raise contract_error(ContractErrorCode.CLOCK_ORDER, "violation evidence", "coverage start changed")
        if (previous.coverage.last_fixed_step is not None and current.coverage.last_fixed_step is not None
                and current.coverage.last_fixed_step < previous.coverage.last_fixed_step):
            raise contract_error(ContractErrorCode.CLOCK_ORDER, "violation evidence", "coverage end regressed")
        if previous.minimum_contact_separation.observed and (
            not current.minimum_contact_separation.observed
            or current.minimum_contact_separation.separation > previous.minimum_contact_separation.separation
        ):
            raise contract_error(ContractErrorCode.INVALID_VALUE, "violation evidence", "capture-wide minimum regressed")


def validate_physics_capture(capture: PhysicsCapture, total_bytes: int) -> None:
    _validate_streams(capture, total_bytes)
    _validate_support(capture.states)
    _validate_events(capture.events)
    _validate_violation_evidence(capture, capture.violation_evidence)
