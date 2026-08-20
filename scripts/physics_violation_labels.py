"""Fail-closed physical-violation labels over validated ``physics_capture_v1``.

No frozen NovPhy level/pilot plan currently defines these labels. Engine evidence
is retained as capture context, but cannot make a physical-violation label available.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Final, TypeAlias

from scripts.physics_capture_contract import (
    EVENT_SIDECAR,
    EXPECTED_COORDINATES,
    STATE_SIDECAR,
    load_physics_capture,
)
from scripts.physics_capture_types import PhysicsCapture
from scripts.physics_relational_supervision import RelationalAvailability


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
LabelRecord: TypeAlias = tuple[str, str | None, bool | None, str, tuple[object, ...]]

PHYSICAL_VIOLATION_SCHEMA_VERSION: Final = "physics_violation_labels_v1"
PHYSICAL_VIOLATION_DERIVATION_VERSION: Final = "physical_violation_derivation_v1"
PHYSICAL_VIOLATION_SIDECAR: Final = "physics_violation_labels.jsonl"
CAPTURE_SCHEMA_VERSION: Final = "physics_capture_v1"

EXCESS_PENETRATION_LABEL: Final = "excess_penetration"
UNSUPPORTED_STATIONARY_BODY_LABEL: Final = "unsupported_stationary_or_floating_body"
ILLEGAL_CONTACT_LABEL: Final = "illegal_contact"
SUPPORTED_PHYSICAL_VIOLATION_LABELS: Final = (
    EXCESS_PENETRATION_LABEL,
    UNSUPPORTED_STATIONARY_BODY_LABEL,
)

@dataclass(frozen=True, slots=True)
class PhysicalViolationError(ValueError):
    location: str
    detail: str

    def __str__(self) -> str:
        return f"invalid physical-violation labels at {self.location}: {self.detail}"


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _canonical_record(record: JsonObject) -> bytes:
    try:
        return (
            json.dumps(
                record,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise PhysicalViolationError("record", "record must contain finite JSON data") from error


@dataclass(frozen=True, slots=True)
class PhysicalViolationLabel:
    label_name: str
    entity_id: str | None
    value: bool | None
    availability: RelationalAvailability
    evidence: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.label_name not in SUPPORTED_PHYSICAL_VIOLATION_LABELS:
            if self.label_name == ILLEGAL_CONTACT_LABEL:
                raise ValueError("illegal_contact is not available in the v1 vocabulary")
            raise ValueError(f"unsupported physical-violation label: {self.label_name}")
        if self.label_name == EXCESS_PENETRATION_LABEL and self.entity_id is not None:
            raise ValueError("excess_penetration is capture-scoped")
        if self.label_name == UNSUPPORTED_STATIONARY_BODY_LABEL:
            _nonempty(self.entity_id, "unsupported-body entity_id")
        if self.availability is not RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION:
            raise ValueError("physical-violation labels are unavailable without an accepted level/pilot plan")
        if self.evidence:
            raise ValueError("unavailable labels require empty evidence")
        if self.value is not None:
            raise ValueError("unavailable labels require a null value")

    def to_json(self) -> JsonObject:
        return {
            "record_type": "violation_label",
            "label_name": self.label_name,
            "entity_id": self.entity_id,
            "value": self.value,
            "availability": self.availability.value,
            "evidence": [],
        }


@dataclass(frozen=True, slots=True)
class PhysicalViolationLabels:
    capture_id: str
    shot_id: str
    labels: tuple[PhysicalViolationLabel, ...]
    _labels_snapshot: tuple[PhysicalViolationLabel, ...] = field(init=False, repr=False, compare=False)
    _label_records: tuple[LabelRecord, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        records = self._validate()
        object.__setattr__(self, "_labels_snapshot", self.labels)
        object.__setattr__(self, "_label_records", records)

    def _validate(self) -> tuple[LabelRecord, ...]:
        _nonempty(self.capture_id, "capture_id")
        _nonempty(self.shot_id, "shot_id")
        if any(type(label) is not PhysicalViolationLabel for label in self.labels):
            raise ValueError("artifact labels must be exact PhysicalViolationLabel instances")
        for label in self.labels:
            PhysicalViolationLabel.__post_init__(label)
        expected = tuple(sorted(self.labels, key=_label_key))
        if self.labels != expected:
            raise ValueError("physical-violation labels differ from canonical order")
        penetration = [label for label in self.labels if label.label_name == EXCESS_PENETRATION_LABEL]
        if len(penetration) != 1:
            raise ValueError("artifact requires exactly one excess_penetration label")
        subjects = [
            label.entity_id
            for label in self.labels
            if label.label_name == UNSUPPORTED_STATIONARY_BODY_LABEL
        ]
        if len(subjects) != len(set(subjects)):
            raise ValueError("unsupported-body subjects must be unique")
        return tuple(_label_record(label) for label in self.labels)

    def _revalidate(self) -> tuple[LabelRecord, ...]:
        records = self._validate()
        if self.labels is not self._labels_snapshot:
            raise ValueError("artifact labels were replaced after construction")
        if records != self._label_records:
            raise ValueError("artifact label members changed after construction")
        return self._label_records

    def header_json(self) -> JsonObject:
        self._revalidate()
        return self._header_json()

    def _header_json(self) -> JsonObject:
        return {
            "record_type": "physical_violation_header",
            "schema_version": PHYSICAL_VIOLATION_SCHEMA_VERSION,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
            "derivation_version": PHYSICAL_VIOLATION_DERIVATION_VERSION,
            "capture_id": self.capture_id,
            "shot_id": self.shot_id,
            "coordinate_convention": asdict(EXPECTED_COORDINATES),
            "supported_labels": list(SUPPORTED_PHYSICAL_VIOLATION_LABELS),
            "label_count": len(self.labels),
            "sources": {
                "physics_state_path": STATE_SIDECAR,
                "physics_events_path": EVENT_SIDECAR,
            },
        }

    def to_bytes(self) -> bytes:
        label_records = self._revalidate()
        records = (self._header_json(), *(_label_json(record) for record in label_records))
        return b"".join(_canonical_record(record) for record in records)

    def to_jsonl(self) -> str:
        return self.to_bytes().decode("ascii")


def _label_key(label: PhysicalViolationLabel) -> tuple[str, str]:
    return label.label_name, label.entity_id or ""


def _label_record(label: PhysicalViolationLabel) -> LabelRecord:
    return (
        label.label_name,
        label.entity_id,
        label.value,
        label.availability.value,
        label.evidence,
    )


def _label_json(record: LabelRecord) -> JsonObject:
    label_name, entity_id, value, availability, evidence = record
    if evidence:
        raise ValueError("canonical fail-closed label records require empty evidence")
    return {
        "record_type": "violation_label",
        "label_name": label_name,
        "entity_id": entity_id,
        "value": value,
        "availability": availability,
        "evidence": [],
    }


def _capture_identity(capture: PhysicsCapture) -> tuple[str, str]:
    return str(capture.header.clock.capture_id), str(capture.header.clock.shot_id)


def _validate_capture_values(capture: PhysicsCapture) -> None:
    capture_id, shot_id = _capture_identity(capture)
    if not capture_id or not shot_id:
        raise PhysicalViolationError("capture", "capture and shot identity must be nonempty")
    sequences: set[int] = set()
    for state in capture.states:
        if state.clock.coordinates != EXPECTED_COORDINATES:
            raise PhysicalViolationError(f"state {state.clock.sequence}", "coordinate declaration is invalid")
        if state.clock.sequence in sequences:
            raise PhysicalViolationError("capture", "state sequences must be unique")
        sequences.add(state.clock.sequence)
        for contact in state.raw_contacts:
            values = [
                contact.separation,
                contact.point.x,
                contact.point.y,
                contact.normal_a_to_b.x,
                contact.normal_a_to_b.y,
                contact.relative_velocity_a_to_b.x,
                contact.relative_velocity_a_to_b.y,
            ]
            values.extend(
                value
                for value in (contact.normal_impulse, contact.tangent_impulse)
                if value is not None
            )
            if any(not math.isfinite(value) for value in values):
                raise PhysicalViolationError(str(contact.contact_id), "raw-contact geometry must be finite")
        for node in state.nodes:
            values = [node.world_pose.position.x, node.world_pose.position.y, node.world_pose.rotation_degrees]
            values.extend(value for point in node.screen_polygon for value in (point.x, point.y))
            if node.body.velocity is not None:
                values.extend((node.body.velocity.x, node.body.velocity.y))
            if node.body.angular_velocity_degrees_per_second is not None:
                values.append(node.body.angular_velocity_degrees_per_second)
            values.extend(
                value
                for value in (
                    node.life,
                    node.body.mass_unity_units,
                    node.body.kinetic_energy_unity_units,
                )
                if value is not None
            )
            if any(not math.isfinite(value) for value in values):
                raise PhysicalViolationError(str(node.entity_id), "body and geometry values must be finite")
        if any(edge.rule_version != "support_v1" for edge in state.support_edges):
            raise PhysicalViolationError(f"state {state.clock.sequence}", "only support_v1 is authoritative")


def _unavailable(
    label_name: str,
    entity_id: str | None,
) -> PhysicalViolationLabel:
    return PhysicalViolationLabel(
        label_name,
        entity_id,
        None,
        RelationalAvailability.UNAVAILABLE_NO_DECLARED_PHYSICAL_REGIME_DERIVATION,
    )


def derive_excess_penetration(capture: PhysicsCapture) -> PhysicalViolationLabel:
    """Return the unavailable capture-scoped label until a level/pilot plan exists."""
    _validate_capture_values(capture)
    return _unavailable(
        EXCESS_PENETRATION_LABEL,
        None,
    )


def derive_unsupported_stationary_body(
    capture: PhysicsCapture,
    entity_id: str,
) -> PhysicalViolationLabel:
    """Return one unavailable entity label until a level/pilot plan exists."""
    _validate_capture_values(capture)
    _nonempty(entity_id, "entity_id")
    return _unavailable(
        UNSUPPORTED_STATIONARY_BODY_LABEL,
        entity_id,
    )


def derive_physical_violation_labels(
    capture: PhysicsCapture,
) -> PhysicalViolationLabels:
    """Derive the complete closed-vocabulary artifact from a loaded capture."""
    _validate_capture_values(capture)
    entities = sorted(
        {
            str(node.entity_id)
            for state in capture.states
            for node in state.nodes
            if node.body.present
        }
    )
    labels = [_unavailable(EXCESS_PENETRATION_LABEL, None)]
    labels.extend(
        _unavailable(UNSUPPORTED_STATIONARY_BODY_LABEL, entity_id)
        for entity_id in entities
    )
    capture_id, shot_id = _capture_identity(capture)
    return PhysicalViolationLabels(
        capture_id,
        shot_id,
        tuple(sorted(labels, key=_label_key)),
    )


def derive_physical_violation_labels_for_shot(shot_dir: Path) -> PhysicalViolationLabels:
    state_path = Path(shot_dir) / STATE_SIDECAR
    event_path = Path(shot_dir) / EVENT_SIDECAR
    capture = load_physics_capture(state_path, event_path)
    return derive_physical_violation_labels(capture)


def _expect_fields(record: Mapping[str, Any], expected: set[str], location: str) -> None:
    if set(record) != expected:
        raise PhysicalViolationError(location, "record is incomplete or contains unknown fields")


def _parse_label(record: JsonObject, location: str) -> PhysicalViolationLabel:
    _expect_fields(
        record,
        {"record_type", "label_name", "entity_id", "value", "availability", "evidence"},
        location,
    )
    if record["record_type"] != "violation_label":
        raise PhysicalViolationError(location, "record_type must be violation_label")
    evidence = record["evidence"]
    if not isinstance(evidence, list):
        raise PhysicalViolationError(location, "evidence must be an array")
    if evidence:
        raise PhysicalViolationError(location, "fail-closed labels require empty evidence")
    try:
        availability = RelationalAvailability(record["availability"])
        return PhysicalViolationLabel(
            label_name=record["label_name"],  # type: ignore[arg-type]
            entity_id=record["entity_id"],  # type: ignore[arg-type]
            value=record["value"],  # type: ignore[arg-type]
            availability=availability,
        )
    except (TypeError, ValueError) as error:
        raise PhysicalViolationError(location, str(error)) from error


def read_physical_violation_labels(path: Path) -> PhysicalViolationLabels:
    location = str(path)
    try:
        raw = Path(path).read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise PhysicalViolationError(location, "sidecar is missing, unreadable, or non-ASCII") from error
    if not text.endswith("\n") or "\r" in text:
        raise PhysicalViolationError(location, "sidecar must use LF and end with a newline")
    try:
        records = [json.loads(line) for line in text[:-1].split("\n")]
    except json.JSONDecodeError as error:
        raise PhysicalViolationError(location, "sidecar contains malformed JSON") from error
    if len(records) < 2 or any(not isinstance(record, dict) for record in records):
        raise PhysicalViolationError(location, "sidecar requires an object header and label records")
    header = records[0]
    _expect_fields(
        header,
        {
            "record_type",
            "schema_version",
            "capture_schema_version",
            "derivation_version",
            "capture_id",
            "shot_id",
            "coordinate_convention",
            "supported_labels",
            "label_count",
            "sources",
        },
        f"{location}:1",
    )
    if (
        header["record_type"] != "physical_violation_header"
        or header["schema_version"] != PHYSICAL_VIOLATION_SCHEMA_VERSION
        or header["capture_schema_version"] != CAPTURE_SCHEMA_VERSION
        or header["derivation_version"] != PHYSICAL_VIOLATION_DERIVATION_VERSION
    ):
        raise PhysicalViolationError(f"{location}:1", "unsupported physical-violation contract")
    if header["supported_labels"] != list(SUPPORTED_PHYSICAL_VIOLATION_LABELS):
        raise PhysicalViolationError(f"{location}:1", "supported label vocabulary differs from v1")
    if header["coordinate_convention"] != asdict(EXPECTED_COORDINATES):
        raise PhysicalViolationError(f"{location}:1", "coordinate convention differs from physics_capture_v1")
    sources = header["sources"]
    if not isinstance(sources, dict):
        raise PhysicalViolationError(f"{location}:1.sources", "sources must be an object")
    _expect_fields(
        sources,
        {
            "physics_state_path",
            "physics_events_path",
        },
        f"{location}:1.sources",
    )
    if sources["physics_state_path"] != STATE_SIDECAR or sources["physics_events_path"] != EVENT_SIDECAR:
        raise PhysicalViolationError(f"{location}:1.sources", "source paths differ from physics_capture_v1")
    labels = tuple(
        _parse_label(record, f"{location}:{index}")
        for index, record in enumerate(records[1:], start=2)
    )
    if (
        isinstance(header["label_count"], bool)
        or not isinstance(header["label_count"], int)
        or header["label_count"] != len(labels)
    ):
        raise PhysicalViolationError(f"{location}:1", "label_count disagrees with records")
    try:
        artifact = PhysicalViolationLabels(
            capture_id=_nonempty(header["capture_id"], "capture_id"),
            shot_id=_nonempty(header["shot_id"], "shot_id"),
            labels=labels,
        )
    except ValueError as error:
        raise PhysicalViolationError(location, str(error)) from error
    if raw != artifact.to_bytes():
        raise PhysicalViolationError(location, "sidecar is not canonical deterministic JSONL")
    return artifact


def write_physical_violation_labels(
    shot_dir: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    artifact = derive_physical_violation_labels_for_shot(shot_dir)
    target = output_path if output_path is not None else Path(shot_dir) / PHYSICAL_VIOLATION_SIDECAR
    temporary = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            stream.write(artifact.to_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise PhysicalViolationError(str(target), "sidecar could not be written") from error
    return target


def validate_physical_violation_labels(
    shot_dir: Path,
    *,
    label_path: Path | None = None,
) -> PhysicalViolationLabels:
    path = label_path if label_path is not None else Path(shot_dir) / PHYSICAL_VIOLATION_SIDECAR
    stored = read_physical_violation_labels(path)
    expected = derive_physical_violation_labels_for_shot(shot_dir)
    if stored.capture_id != expected.capture_id or stored.shot_id != expected.shot_id:
        raise PhysicalViolationError(str(path), "capture identity differs from source")
    if stored.to_bytes() != expected.to_bytes():
        raise PhysicalViolationError(str(path), "stored labels differ from canonical re-derivation")
    return stored


__all__ = [
    "EXCESS_PENETRATION_LABEL",
    "ILLEGAL_CONTACT_LABEL",
    "PHYSICAL_VIOLATION_DERIVATION_VERSION",
    "PHYSICAL_VIOLATION_SCHEMA_VERSION",
    "PHYSICAL_VIOLATION_SIDECAR",
    "SUPPORTED_PHYSICAL_VIOLATION_LABELS",
    "UNSUPPORTED_STATIONARY_BODY_LABEL",
    "PhysicalViolationError",
    "PhysicalViolationLabel",
    "PhysicalViolationLabels",
    "derive_excess_penetration",
    "derive_physical_violation_labels",
    "derive_physical_violation_labels_for_shot",
    "derive_unsupported_stationary_body",
    "read_physical_violation_labels",
    "validate_physical_violation_labels",
    "write_physical_violation_labels",
]
