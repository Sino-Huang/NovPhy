"""Deterministic relational supervision over validated ``physics_capture_v1``.

This artifact is deliberately source-bound and sparse.  Contact truths contain
only unordered pairs present in non-trigger ``RawContact`` records.  Support
labels are emitted for the current/previous source candidates, but a negative
label requires the immediately preceding fixed step, both same-pair contacts,
both entity lifecycles, and usable source geometry.  No RGB, node appearance,
future state, or unrelated event is consulted to manufacture a relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
from typing import Final, TypeAlias

from scripts.physics_capture_contract import (
    EVENT_SIDECAR,
    STATE_SIDECAR,
    PhysicsContractError,
    load_physics_capture,
)
from scripts.physics_capture_types import (
    ContactId,
    EntityId,
    PhysicsCapture,
    RawContact,
    SceneNode,
    StateFrame,
    SupportEdge,
    EventType,
)


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
EntityPair: TypeAlias = tuple[str, str]

RELATIONAL_SUPERVISION_SCHEMA_VERSION: Final = "physics_relational_supervision_v1"
RELATIONAL_SUPERVISION_SIDECAR: Final = "physics_relational_supervision.jsonl"
RELATIONAL_LABEL_SCHEMA_VERSION: Final = RELATIONAL_SUPERVISION_SCHEMA_VERSION
RELATIONAL_LABEL_SIDECAR: Final = RELATIONAL_SUPERVISION_SIDECAR
CAPTURE_SCHEMA_VERSION: Final = "physics_capture_v1"
DERIVATION_SPEC_VERSION: Final = "relational_supervision_derivation_v1"

EVENT_CLOCK_JSON: Final = {
    "occurrence_authority": "fixed_step",
    "render_frame_role": "provenance_only",
}


@unique
class RelationalAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE_NO_PREDECESSOR = "unavailable_no_predecessor"
    UNAVAILABLE_INSUFFICIENT_PREDECESSOR = "unavailable_insufficient_predecessor"
    UNAVAILABLE_INSUFFICIENT_LIFECYCLE_EVIDENCE = "unavailable_insufficient_lifecycle_evidence"
    UNAVAILABLE_INSUFFICIENT_CONTACT_EVIDENCE = "unavailable_insufficient_contact_evidence"
    UNAVAILABLE_INSUFFICIENT_GEOMETRY_EVIDENCE = "unavailable_insufficient_geometry_evidence"
    UNAVAILABLE_MISSING_OR_INCONSISTENT_POSITIVE_SUPPORT_DERIVATION = (
        "unavailable_missing_or_inconsistent_positive_support_derivation"
    )
    UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION = "unavailable_no_declared_physical_regime_derivation"
    UNAVAILABLE_NOT_DERIVABLE = "unavailable_not_derivable"


# A short compatibility name is useful at the public seam and mirrors the macro
# artifact's vocabulary without sharing its implementation.
Availability = RelationalAvailability


@dataclass(frozen=True, slots=True)
class RelationalSupervisionError(ValueError):
    location: str
    detail: str

    def __str__(self) -> str:
        return f"invalid relational supervision at {self.location}: {self.detail}"


RelationalLabelError = RelationalSupervisionError


def _pair(left: str | EntityId, right: str | EntityId) -> EntityPair:
    values = (str(left), str(right))
    if values[0] == values[1]:
        raise ValueError("a relational pair must contain two distinct entities")
    return tuple(sorted(values))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RelationalStateIdentity:
    capture_id: str
    shot_id: str
    state_sequence: int
    render_frame: int
    fixed_step: int
    rgb_relative_path: str

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "state_sequence": self.state_sequence,
            "render_frame": self.render_frame,
            "fixed_step": self.fixed_step,
            "rgb_relative_path": self.rgb_relative_path,
        }


@dataclass(frozen=True, slots=True)
class ContactCitation:
    capture_id: str
    shot_id: str
    state_sequence: int
    fixed_step: int
    contact_id: str

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "state_sequence": self.state_sequence,
            "fixed_step": self.fixed_step,
            "contact_id": self.contact_id,
        }


@dataclass(frozen=True, slots=True)
class ContactTruth:
    """One canonical unordered contact truth; presence is true, never false."""

    pair: EntityPair
    evidence: tuple[ContactCitation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", _pair(*self.pair))
        if not self.evidence:
            raise ValueError("contact truth requires source evidence")

    @property
    def entity_a_id(self) -> str:
        return self.pair[0]

    @property
    def entity_b_id(self) -> str:
        return self.pair[1]

    @property
    def value(self) -> bool:
        return True

    def to_json(self) -> JsonObject:
        return {
            "entity_a_id": self.pair[0],
            "entity_b_id": self.pair[1],
            "evidence": [citation.to_json() for citation in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class SupportLabel:
    """A directed support truth for one unordered candidate pair."""

    pair: EntityPair
    value: bool | None
    availability: RelationalAvailability
    evidence: tuple[ContactCitation, ...]
    supporter_id: str | None = None
    supported_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", _pair(*self.pair))
        if self.value is None and self.availability is RelationalAvailability.AVAILABLE:
            raise ValueError("available support labels require a boolean value")
        if self.value is not None and self.availability is not RelationalAvailability.AVAILABLE:
            raise ValueError("unavailable support labels require a null value")
        if (self.supporter_id is None) != (self.supported_id is None):
            raise ValueError("support direction must be complete or absent")
        if self.supporter_id is not None and _pair(self.supporter_id, self.supported_id) != self.pair:
            raise ValueError("support direction does not match pair")

    def to_json(self) -> JsonObject:
        return {
            "entity_a_id": self.pair[0],
            "entity_b_id": self.pair[1],
            "supporter_id": self.supporter_id,
            "supported_id": self.supported_id,
            "value": self.value,
            "availability": self.availability.value,
            "evidence": [citation.to_json() for citation in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class PhysicalRegimeEligibility:
    """Unavailable until a versioned physical-regime derivation is declared."""

    value: bool | None
    availability: RelationalAvailability
    evidence: tuple[ContactCitation, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.value is not None
            or self.availability
            is not RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION
            or self.evidence
        ):
            raise ValueError("physical-regime eligibility must remain explicitly unavailable")

    def to_json(self) -> JsonObject:
        return {
            "value": self.value,
            "availability": self.availability.value,
            "evidence": [citation.to_json() for citation in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ModelRelativeMicroRelationUsefulness:
    """Model-relative usefulness is intentionally outside this source contract."""

    value: bool | None
    availability: RelationalAvailability
    evidence: tuple[ContactCitation, ...] = ()

    def __post_init__(self) -> None:
        if self.value is not None or self.availability is not RelationalAvailability.UNAVAILABLE_NOT_DERIVABLE:
            raise ValueError("model-relative usefulness must remain explicitly unavailable")

    def to_json(self) -> JsonObject:
        return {
            "value": None,
            "availability": self.availability.value,
            "evidence": [],
        }


@dataclass(frozen=True, slots=True)
class RelationalFrameLabel:
    identity: RelationalStateIdentity
    contacts: tuple[ContactTruth, ...]
    supports: tuple[SupportLabel, ...]
    physical_regime_eligibility: PhysicalRegimeEligibility
    model_relative_micro_relation_usefulness: ModelRelativeMicroRelationUsefulness

    @property
    def render_frame(self) -> int:
        return self.identity.render_frame

    @property
    def fixed_step(self) -> int:
        return self.identity.fixed_step

    @property
    def contact_truths(self) -> tuple[ContactTruth, ...]:
        return self.contacts

    @property
    def support_labels(self) -> tuple[SupportLabel, ...]:
        return self.supports

    def contact_truth(self, pair: tuple[str, str]) -> ContactTruth:
        wanted = _pair(*pair)
        for contact in self.contacts:
            if contact.pair == wanted:
                return contact
        raise KeyError(wanted)

    def support_label(self, pair: tuple[str, str]) -> SupportLabel:
        wanted = _pair(*pair)
        for label in self.supports:
            if label.pair == wanted:
                return label
        raise KeyError(wanted)

    def to_json(self) -> JsonObject:
        return {
            "record_type": "frame_label",
            **self.identity.to_json(),
            "contacts": [contact.to_json() for contact in self.contacts],
            "supports": [support.to_json() for support in self.supports],
            "physical_regime_eligibility": self.physical_regime_eligibility.to_json(),
            "model_relative_micro_relation_usefulness": self.model_relative_micro_relation_usefulness.to_json(),
        }


@dataclass(frozen=True, slots=True)
class RelationalSupervision:
    capture_id: str
    shot_id: str
    state_sha256: str
    events_sha256: str
    event_count: int
    frames: tuple[RelationalFrameLabel, ...]
    schema_version: str = RELATIONAL_SUPERVISION_SCHEMA_VERSION

    def header_json(self) -> JsonObject:
        return {
            "record_type": "relational_supervision_header",
            "schema_version": self.schema_version,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "derivation_spec_version": DERIVATION_SPEC_VERSION,
            "derivation_spec_digest": derivation_spec_digest(),
            "event_clock": dict(EVENT_CLOCK_JSON),
            "sources": {
                "physics_state_path": STATE_SIDECAR,
                "physics_state_sha256": self.state_sha256,
                "physics_events_path": EVENT_SIDECAR,
                "physics_events_sha256": self.events_sha256,
            },
            "state_count": len(self.frames),
            "event_count": self.event_count,
            "frame_label_count": len(self.frames),
        }

    def to_jsonl(self) -> str:
        records = [self.header_json(), *(frame.to_json() for frame in self.frames)]
        return "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for record in records
        )


# Public aliases use both the artifact name and the shorter label vocabulary.
RelationalLabels: TypeAlias = RelationalSupervision
RelationalLabel: TypeAlias = SupportLabel
ContactRelationLabel: TypeAlias = ContactTruth
SupportRelationLabel: TypeAlias = SupportLabel
PhysicalRegimeEligibilityLabel: TypeAlias = PhysicalRegimeEligibility
ModelRelativeMicroRelationUsefulnessLabel: TypeAlias = ModelRelativeMicroRelationUsefulness


def derivation_spec_json() -> JsonObject:
    return {
        "version": DERIVATION_SPEC_VERSION,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "contact_truth": "current_state_non_trigger_raw_contacts_only_unordered_pairs",
        "support_truth": "positive_support_edge_or_negative_complete_previous_fixed_step_same_pair_evidence",
        "support_unavailable": [
            availability.value
            for availability in RelationalAvailability
            if availability is not RelationalAvailability.AVAILABLE
            and availability is not RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION
            and availability is not RelationalAvailability.UNAVAILABLE_NOT_DERIVABLE
        ],
        "event_clock": dict(EVENT_CLOCK_JSON),
        "future_state_access": "none",
        "rgb_or_node_appearance_access": "none",
        "physical_regime_eligibility": "unavailable_no_declared_physical_regime_derivation",
        "model_relative_micro_relation_usefulness": "unavailable_not_derivable",
    }


def derivation_spec_digest() -> str:
    encoded = json.dumps(derivation_spec_json(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _citation(capture: PhysicsCapture, state: StateFrame, contact: RawContact) -> ContactCitation:
    return ContactCitation(
        capture_id=str(capture.header.clock.capture_id),
        shot_id=str(capture.header.clock.shot_id),
        state_sequence=state.clock.sequence,
        fixed_step=_contact_fixed_step(contact),
        contact_id=str(contact.contact_id),
    )


def _contact_fixed_step(contact: RawContact) -> int:
    match = re.fullmatch(r"contact:([0-9]+):.+:[0-9]+", str(contact.contact_id))
    if match is None:
        raise RelationalSupervisionError(
            str(contact.contact_id),
            "contact id does not encode its fixed step",
        )
    return int(match.group(1))


def _retained_contacts(state: StateFrame, pair: EntityPair | None = None) -> tuple[RawContact, ...]:
    values = tuple(
        contact
        for contact in state.raw_contacts
        if not contact.is_trigger
        and (pair is None or _pair(contact.entity_a_id, contact.entity_b_id) == pair)
    )
    return tuple(sorted(values, key=lambda contact: str(contact.contact_id)))


def _contacts(state: StateFrame, pair: EntityPair | None = None) -> tuple[RawContact, ...]:
    return tuple(
        contact
        for contact in _retained_contacts(state, pair)
        if _contact_fixed_step(contact) == state.clock.fixed_step
    )


def _pairs(state: StateFrame) -> set[EntityPair]:
    return {_pair(contact.entity_a_id, contact.entity_b_id) for contact in _contacts(state)}


def _support_pairs(state: StateFrame) -> set[EntityPair]:
    return {_pair(edge.supporter_id, edge.supported_id) for edge in state.support_edges}


def _node_pairs(state: StateFrame) -> set[EntityPair]:
    entity_ids = sorted({str(node.entity_id) for node in state.nodes})
    return {
        _pair(left, right)
        for index, left in enumerate(entity_ids)
        for right in entity_ids[index + 1 :]
    }


def _identity(capture: PhysicsCapture, state: StateFrame) -> RelationalStateIdentity:
    return RelationalStateIdentity(
        capture_id=str(capture.header.clock.capture_id),
        shot_id=str(capture.header.clock.shot_id),
        state_sequence=state.clock.sequence,
        render_frame=state.clock.render_frame,
        fixed_step=state.clock.fixed_step,
        rgb_relative_path=state.rgb_frame.relative_path,
    )


def _unavailable(pair: EntityPair, availability: RelationalAvailability) -> SupportLabel:
    return SupportLabel(pair, None, availability, ())


def _vertical_ordering(
    nodes: dict[str, SceneNode],
    pair: EntityPair,
    minimum_delta: float,
) -> tuple[bool, tuple[str, str] | None]:
    try:
        first_y = nodes[pair[0]].world_pose.position.y
        second_y = nodes[pair[1]].world_pose.position.y
        delta = second_y - first_y
    except (AttributeError, KeyError, TypeError):
        return False, None
    if not all(math.isfinite(value) for value in (first_y, second_y, delta)):
        return False, None
    if abs(delta) < minimum_delta:
        return True, None
    return True, (pair[1], pair[0]) if delta < 0 else (pair[0], pair[1])


def _has_support_v1_contact_normals(
    previous_contacts: tuple[RawContact, ...],
    current_contacts: tuple[RawContact, ...],
    minimum_abs_normal_y: float,
) -> bool:
    previous_by_colliders: dict[tuple[int, int], list[RawContact]] = {}
    current_by_colliders: dict[tuple[int, int], list[RawContact]] = {}
    for contact in previous_contacts:
        previous_by_colliders.setdefault(
            (contact.collider_a_id, contact.collider_b_id), []
        ).append(contact)
    for contact in current_contacts:
        current_by_colliders.setdefault(
            (contact.collider_a_id, contact.collider_b_id), []
        ).append(contact)

    return any(
        abs(previous_contact.normal_a_to_b.y) >= minimum_abs_normal_y
        and abs(current_contact.normal_a_to_b.y) >= minimum_abs_normal_y
        for colliders in previous_by_colliders.keys() & current_by_colliders.keys()
        for previous_contact in previous_by_colliders[colliders]
        for current_contact in current_by_colliders[colliders]
    )


def _support_label(
    capture: PhysicsCapture,
    index: int,
    state: StateFrame,
    pair: EntityPair,
) -> SupportLabel:
    current_edges = [
        edge for edge in state.support_edges if _pair(edge.supporter_id, edge.supported_id) == pair
    ]
    if current_edges:
        edge = current_edges[0]
        by_step_and_id: dict[tuple[int, str], tuple[StateFrame, RawContact]] = {}
        for candidate_state in capture.states[: index + 1]:
            for contact in _retained_contacts(candidate_state):
                key = (_contact_fixed_step(contact), str(contact.contact_id))
                by_step_and_id.setdefault(key, (candidate_state, contact))
        evidence: list[ContactCitation] = []
        for fixed_step, contact_id in zip(edge.evidence_fixed_steps, edge.evidence_contact_ids, strict=True):
            source = by_step_and_id.get((fixed_step, str(contact_id)))
            if source is None or _pair(source[1].entity_a_id, source[1].entity_b_id) != pair:
                return _unavailable(pair, RelationalAvailability.UNAVAILABLE_INSUFFICIENT_CONTACT_EVIDENCE)
            evidence.append(_citation(capture, source[0], source[1]))
        return SupportLabel(
            pair,
            True,
            RelationalAvailability.AVAILABLE,
            tuple(evidence),
            str(edge.supporter_id),
            str(edge.supported_id),
        )

    if index == 0:
        return _unavailable(pair, RelationalAvailability.UNAVAILABLE_NO_PREDECESSOR)
    previous = capture.states[index - 1]
    if previous.clock.fixed_step + 1 != state.clock.fixed_step:
        return _unavailable(pair, RelationalAvailability.UNAVAILABLE_INSUFFICIENT_PREDECESSOR)

    previous_contacts = _contacts(previous, pair)
    current_contacts = _contacts(state, pair)
    if not previous_contacts or not current_contacts:
        return _unavailable(pair, RelationalAvailability.UNAVAILABLE_INSUFFICIENT_CONTACT_EVIDENCE)

    previous_nodes = {str(node.entity_id): node for node in previous.nodes}
    current_nodes = {str(node.entity_id): node for node in state.nodes}
    lifecycle_events = (EventType.ENTITY_DESTROYED, EventType.PIG_REMOVED, EventType.EXPLOSION)
    if any(
        event.event_type in lifecycle_events
        and previous.clock.fixed_step < event.clock.fixed_step <= state.clock.fixed_step
        and any(str(participant) in pair for participant in event.participants)
        for event in capture.events
    ):
        return _unavailable(pair, RelationalAvailability.UNAVAILABLE_INSUFFICIENT_LIFECYCLE_EVIDENCE)

    minimum_delta = capture.header.support_rule.minimum_vertical_center_delta
    previous_geometry_available, previous_ordering = _vertical_ordering(
        previous_nodes, pair, minimum_delta
    )
    current_geometry_available, current_ordering = _vertical_ordering(
        current_nodes, pair, minimum_delta
    )
    if not previous_geometry_available or not current_geometry_available:
        return _unavailable(pair, RelationalAvailability.UNAVAILABLE_INSUFFICIENT_GEOMETRY_EVIDENCE)
    evidence = tuple(
        [_citation(capture, previous, contact) for contact in previous_contacts]
        + [_citation(capture, state, contact) for contact in current_contacts]
    )
    direction = current_ordering if previous_ordering == current_ordering else None
    if direction is not None and _has_support_v1_contact_normals(
        previous_contacts,
        current_contacts,
        capture.header.support_rule.minimum_abs_normal_y,
    ):
        return SupportLabel(
            pair,
            None,
            RelationalAvailability.UNAVAILABLE_MISSING_OR_INCONSISTENT_POSITIVE_SUPPORT_DERIVATION,
            evidence,
            *direction,
        )
    return SupportLabel(
        pair,
        False,
        RelationalAvailability.AVAILABLE,
        evidence,
        *(direction or (None, None)),
    )


def derive_relational_supervision(
    capture: PhysicsCapture,
    *,
    state_sha256: str = "",
    events_sha256: str = "",
) -> RelationalSupervision:
    """Derive labels using only the current and immediately previous source states."""
    frames: list[RelationalFrameLabel] = []
    for index, state in enumerate(capture.states):
        current_contacts = _contacts(state)
        contacts_by_pair: dict[EntityPair, list[RawContact]] = {}
        for contact in current_contacts:
            contacts_by_pair.setdefault(_pair(contact.entity_a_id, contact.entity_b_id), []).append(contact)
        contacts = tuple(
            ContactTruth(
                pair,
                tuple(_citation(capture, state, contact) for contact in contacts_by_pair[pair]),
            )
            for pair in sorted(contacts_by_pair)
        )

        candidate_pairs = _node_pairs(state) | _pairs(state) | _support_pairs(state)
        if index:
            previous = capture.states[index - 1]
            candidate_pairs |= _node_pairs(previous) | _pairs(previous) | _support_pairs(previous)
        supports = tuple(
            _support_label(capture, index, state, pair) for pair in sorted(candidate_pairs)
        )
        physical = PhysicalRegimeEligibility(
            None,
            RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION,
        )
        micro = ModelRelativeMicroRelationUsefulness(
            None,
            RelationalAvailability.UNAVAILABLE_NOT_DERIVABLE,
        )
        frames.append(
            RelationalFrameLabel(_identity(capture, state), contacts, supports, physical, micro)
        )

    return RelationalSupervision(
        capture_id=str(capture.header.clock.capture_id),
        shot_id=str(capture.header.clock.shot_id),
        state_sha256=state_sha256,
        events_sha256=events_sha256,
        event_count=len(capture.events),
        frames=tuple(frames),
    )


def derive_relational_supervision_for_shot(shot_dir: Path) -> RelationalSupervision:
    state_path = shot_dir / STATE_SIDECAR
    event_path = shot_dir / EVENT_SIDECAR
    if not state_path.is_file() or not event_path.is_file():
        raise RelationalSupervisionError(str(shot_dir), "physics capture sidecars are missing")
    try:
        loaded = load_physics_capture(state_path, event_path)
    except (OSError, PhysicsContractError) as error:
        raise RelationalSupervisionError(str(shot_dir), "physics capture sidecars are invalid") from error
    return derive_relational_supervision(
        loaded,
        state_sha256=_sha256_file(state_path),
        events_sha256=_sha256_file(event_path),
    )


def write_relational_supervision_file(labels: RelationalSupervision, destination: Path) -> Path:
    for field, digest in (
        ("state_sha256", labels.state_sha256),
        ("events_sha256", labels.events_sha256),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RelationalSupervisionError(
                str(destination),
                f"{field} must be a lowercase SHA-256 digest",
            )
    temporary = destination.parent / f".{RELATIONAL_SUPERVISION_SIDECAR}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(labels.to_jsonl())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RelationalSupervisionError(str(destination), "sidecar could not be written") from error
    return destination


def write_relational_supervision(shot_dir: Path) -> Path:
    labels = derive_relational_supervision_for_shot(shot_dir)
    return write_relational_supervision_file(labels, shot_dir / RELATIONAL_SUPERVISION_SIDECAR)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _parse_record(line: str, location: str) -> JsonObject:
    try:
        value = json.loads(line, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise RelationalSupervisionError(location, "record is not valid finite JSON") from error
    if not isinstance(value, dict):
        raise RelationalSupervisionError(location, "record must be an object")
    return value


def _expect_fields(record: JsonObject, fields: frozenset[str], location: str) -> None:
    if not isinstance(record, dict):
        raise RelationalSupervisionError(location, "record must be an object")
    unknown = record.keys() - fields
    missing = fields - record.keys()
    if unknown:
        raise RelationalSupervisionError(location, f"unknown field: {min(unknown)}")
    if missing:
        raise RelationalSupervisionError(location, f"missing field: {min(missing)}")


def _string(record: JsonObject, field: str, location: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value:
        raise RelationalSupervisionError(location, f"{field} must be a nonempty string")
    return value


def _integer(record: JsonObject, field: str, location: str) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RelationalSupervisionError(location, f"{field} must be a nonnegative integer")
    return value


def _digest(record: JsonObject, field: str, location: str) -> str:
    value = _string(record, field, location)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RelationalSupervisionError(location, f"{field} must be a lowercase SHA-256 digest")
    return value


IDENTITY_FIELDS: Final = frozenset(("capture_id", "shot_id", "state_sequence", "render_frame", "fixed_step", "rgb_relative_path"))
CITATION_FIELDS: Final = frozenset(("capture_id", "shot_id", "state_sequence", "fixed_step", "contact_id"))
CONTACT_FIELDS: Final = frozenset(("entity_a_id", "entity_b_id", "evidence"))
SUPPORT_FIELDS: Final = frozenset(("entity_a_id", "entity_b_id", "supporter_id", "supported_id", "value", "availability", "evidence"))
ELIGIBILITY_FIELDS: Final = frozenset(("value", "availability", "evidence"))
FRAME_FIELDS: Final = frozenset(("record_type", *IDENTITY_FIELDS, "contacts", "supports", "physical_regime_eligibility", "model_relative_micro_relation_usefulness"))
HEADER_FIELDS: Final = frozenset(("record_type", "schema_version", "capture_schema_version", "capture_id", "shot_id", "derivation_spec_version", "derivation_spec_digest", "event_clock", "sources", "state_count", "event_count", "frame_label_count"))
SOURCE_FIELDS: Final = frozenset(("physics_state_path", "physics_state_sha256", "physics_events_path", "physics_events_sha256"))
EVENT_CLOCK_FIELDS: Final = frozenset(("occurrence_authority", "render_frame_role"))


def _citation_from_json(value: JsonValue, location: str) -> ContactCitation:
    if not isinstance(value, dict):
        raise RelationalSupervisionError(location, "citation must be an object")
    _expect_fields(value, CITATION_FIELDS, location)
    return ContactCitation(
        _string(value, "capture_id", location),
        _string(value, "shot_id", location),
        _integer(value, "state_sequence", location),
        _integer(value, "fixed_step", location),
        _string(value, "contact_id", location),
    )


def _evidence(value: JsonValue, location: str) -> tuple[ContactCitation, ...]:
    if not isinstance(value, list):
        raise RelationalSupervisionError(location, "evidence must be an array")
    citations = tuple(_citation_from_json(item, location) for item in value)
    keys = tuple((item.fixed_step, item.state_sequence, item.contact_id) for item in citations)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise RelationalSupervisionError(location, "evidence is not in canonical order")
    return citations


def _identity_from_json(record: JsonObject, location: str) -> RelationalStateIdentity:
    missing = IDENTITY_FIELDS - record.keys()
    if missing:
        raise RelationalSupervisionError(location, f"missing field: {min(missing)}")
    return RelationalStateIdentity(
        _string(record, "capture_id", location),
        _string(record, "shot_id", location),
        _integer(record, "state_sequence", location),
        _integer(record, "render_frame", location),
        _integer(record, "fixed_step", location),
        _string(record, "rgb_relative_path", location),
    )


def _pair_fields(record: JsonObject, location: str) -> EntityPair:
    try:
        pair = _pair(_string(record, "entity_a_id", location), _string(record, "entity_b_id", location))
    except ValueError as error:
        raise RelationalSupervisionError(location, "entity pair must contain two distinct entities") from error
    if (record["entity_a_id"], record["entity_b_id"]) != pair:
        raise RelationalSupervisionError(location, "entity pair is not canonical unordered order")
    return pair


def _contact_from_json(record: JsonObject, location: str) -> ContactTruth:
    _expect_fields(record, CONTACT_FIELDS, location)
    pair = _pair_fields(record, location)
    evidence = _evidence(record["evidence"], location)
    try:
        return ContactTruth(pair, evidence)
    except ValueError as error:
        raise RelationalSupervisionError(location, str(error)) from error


def _support_from_json(record: JsonObject, location: str) -> SupportLabel:
    _expect_fields(record, SUPPORT_FIELDS, location)
    pair = _pair_fields(record, location)
    raw_value = record["value"]
    if raw_value is not None and not isinstance(raw_value, bool):
        raise RelationalSupervisionError(location, "support value must be true, false, or null")
    try:
        availability = RelationalAvailability(record["availability"])
    except (TypeError, ValueError) as error:
        raise RelationalSupervisionError(location, "unknown support availability") from error
    evidence = _evidence(record["evidence"], location)
    supporter = record["supporter_id"]
    supported = record["supported_id"]
    if supporter is not None and (not isinstance(supporter, str) or not supporter):
        raise RelationalSupervisionError(location, "supporter_id must be a nonempty string or null")
    if supported is not None and (not isinstance(supported, str) or not supported):
        raise RelationalSupervisionError(location, "supported_id must be a nonempty string or null")
    try:
        return SupportLabel(pair, raw_value, availability, evidence, supporter, supported)
    except ValueError as error:
        raise RelationalSupervisionError(location, str(error)) from error


def _eligibility_from_json(value: JsonValue, location: str, *, model_relative: bool) -> PhysicalRegimeEligibility | ModelRelativeMicroRelationUsefulness:
    if not isinstance(value, dict):
        raise RelationalSupervisionError(location, "eligibility must be an object")
    _expect_fields(value, ELIGIBILITY_FIELDS, location)
    raw_value = value["value"]
    if raw_value is not None and not isinstance(raw_value, bool):
        raise RelationalSupervisionError(location, "eligibility value must be true, false, or null")
    try:
        availability = RelationalAvailability(value["availability"])
    except (TypeError, ValueError) as error:
        raise RelationalSupervisionError(location, "unknown eligibility availability") from error
    evidence = _evidence(value["evidence"], location)
    try:
        if model_relative:
            return ModelRelativeMicroRelationUsefulness(raw_value, availability, evidence)
        return PhysicalRegimeEligibility(raw_value, availability, evidence)
    except ValueError as error:
        raise RelationalSupervisionError(location, str(error)) from error


def _frame_from_json(record: JsonObject, location: str) -> RelationalFrameLabel:
    _expect_fields(record, FRAME_FIELDS, location)
    if record["record_type"] != "frame_label":
        raise RelationalSupervisionError(location, "expected frame_label")
    identity = _identity_from_json(record, location)
    raw_contacts = record["contacts"]
    raw_supports = record["supports"]
    if not isinstance(raw_contacts, list) or not isinstance(raw_supports, list):
        raise RelationalSupervisionError(location, "contacts and supports must be arrays")
    contacts = tuple(_contact_from_json(value, f"{location}.contacts") for value in raw_contacts)
    supports = tuple(_support_from_json(value, f"{location}.supports") for value in raw_supports)
    if contacts != tuple(sorted(contacts, key=lambda item: item.pair)):
        raise RelationalSupervisionError(location, "contacts differ from canonical pair order")
    if supports != tuple(sorted(supports, key=lambda item: item.pair)):
        raise RelationalSupervisionError(location, "supports differ from canonical pair order")
    physical = _eligibility_from_json(record["physical_regime_eligibility"], f"{location}.physical_regime_eligibility", model_relative=False)
    micro = _eligibility_from_json(record["model_relative_micro_relation_usefulness"], f"{location}.model_relative_micro_relation_usefulness", model_relative=True)
    assert isinstance(physical, PhysicalRegimeEligibility)
    assert isinstance(micro, ModelRelativeMicroRelationUsefulness)
    return RelationalFrameLabel(identity, contacts, supports, physical, micro)


def read_relational_supervision(path: Path) -> RelationalSupervision:
    location = str(path)
    if not path.is_file():
        raise RelationalSupervisionError(location, "relational supervision sidecar is missing")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RelationalSupervisionError(location, "sidecar is unreadable or not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise RelationalSupervisionError(location, "sidecar must use LF line endings and end with a newline")
    lines = text[:-1].split("\n")
    if len(lines) < 2:
        raise RelationalSupervisionError(location, "sidecar requires a header and at least one frame")
    records = tuple(_parse_record(line, f"{location}:{index}") for index, line in enumerate(lines, start=1))
    header = records[0]
    _expect_fields(header, HEADER_FIELDS, f"{location}:1")
    if header["record_type"] != "relational_supervision_header":
        raise RelationalSupervisionError(f"{location}:1", "first record must be relational_supervision_header")
    if header["schema_version"] != RELATIONAL_SUPERVISION_SCHEMA_VERSION or header["capture_schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise RelationalSupervisionError(f"{location}:1", "unsupported schema version")
    if header["derivation_spec_version"] != DERIVATION_SPEC_VERSION or header["derivation_spec_digest"] != derivation_spec_digest():
        raise RelationalSupervisionError(f"{location}:1", "derivation specification differs from this module")
    clock = header["event_clock"]
    if not isinstance(clock, dict):
        raise RelationalSupervisionError(f"{location}:1", "event_clock must be an object")
    _expect_fields(clock, EVENT_CLOCK_FIELDS, f"{location}:1.event_clock")
    if clock != EVENT_CLOCK_JSON:
        raise RelationalSupervisionError(f"{location}:1", "event clock differs from fixed-step contract")
    sources = header["sources"]
    if not isinstance(sources, dict):
        raise RelationalSupervisionError(f"{location}:1", "sources must be an object")
    _expect_fields(sources, SOURCE_FIELDS, f"{location}:1.sources")
    if sources["physics_state_path"] != STATE_SIDECAR or sources["physics_events_path"] != EVENT_SIDECAR:
        raise RelationalSupervisionError(f"{location}:1", "source paths differ from physics_capture_v1")
    state_count = _integer(header, "state_count", f"{location}:1")
    event_count = _integer(header, "event_count", f"{location}:1")
    frame_count = _integer(header, "frame_label_count", f"{location}:1")
    if state_count != frame_count or frame_count != len(records) - 1:
        raise RelationalSupervisionError(f"{location}:1", "header frame counts disagree with records")
    frames = tuple(_frame_from_json(record, f"{location}:{index}") for index, record in enumerate(records[1:], start=2))
    if any(previous.identity.state_sequence >= current.identity.state_sequence for previous, current in zip(frames, frames[1:])):
        raise RelationalSupervisionError(location, "frames differ from accepted state order")
    return RelationalSupervision(
        capture_id=_string(header, "capture_id", f"{location}:1"),
        shot_id=_string(header, "shot_id", f"{location}:1"),
        state_sha256=_digest(sources, "physics_state_sha256", f"{location}:1.sources"),
        events_sha256=_digest(sources, "physics_events_sha256", f"{location}:1.sources"),
        event_count=event_count,
        frames=frames,
    )


def validate_relational_supervision(shot_dir: Path, label_path: Path | None = None) -> RelationalSupervision:
    path = label_path if label_path is not None else shot_dir / RELATIONAL_SUPERVISION_SIDECAR
    stored = read_relational_supervision(path)
    expected = derive_relational_supervision_for_shot(shot_dir)
    if stored.capture_id != expected.capture_id or stored.shot_id != expected.shot_id:
        raise RelationalSupervisionError(str(path), "capture identity differs from source")
    if stored.state_sha256 != expected.state_sha256 or stored.events_sha256 != expected.events_sha256:
        raise RelationalSupervisionError(str(path), "source sidecar digests are stale")
    try:
        actual = path.read_bytes()
    except OSError as error:
        raise RelationalSupervisionError(str(path), "sidecar is unreadable") from error
    if actual != expected.to_jsonl().encode("utf-8"):
        raise RelationalSupervisionError(str(path), "stored labels disagree with fresh derivation")
    return stored


# Short aliases match the existing derived/macro naming and make the public seam
# tolerant of callers using either "labels" or "supervision" terminology.
derive_relational_labels = derive_relational_supervision
derive_relational_labels_for_shot = derive_relational_supervision_for_shot
read_relational_labels = read_relational_supervision
validate_relational_labels = validate_relational_supervision
write_relational_labels = write_relational_supervision
write_relational_label_file = write_relational_supervision_file
