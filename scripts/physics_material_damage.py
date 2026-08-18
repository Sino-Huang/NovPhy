from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum, unique
import hashlib
import json
import os
from pathlib import Path
from typing import Final, TypeAlias

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR, load_physics_capture
from scripts.physics_capture_types import EventType, PhysicsCapture, StateFrame
from scripts.physics_macro_labels import SemanticStatus


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Source: TypeAlias = PhysicsCapture | Path

MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION: Final = "physics_material_damage_mapping_v1"
MATERIAL_DAMAGE_DERIVATION_SCHEMA_VERSION: Final = "physics_material_damage_derivation_v3"
MATERIAL_DAMAGE_SIDECAR: Final = "physics_material_damage.json"
DAMAGE_LIFECYCLE_MAPPING_VERSION: Final = "damage_lifecycle_mapping_v1"
PHYSICS_CAPTURE_SCHEMA_VERSION: Final = "physics_capture_v1"
MATERIAL_UNAVAILABLE_LABEL: Final = "unavailable"
DAMAGE_LABEL: Final = "damage"
NO_DAMAGE_LABEL: Final = "no_damage"
SUPPORTED_VERSION_ENVELOPE: Final = (
    ("capture_schema_version", PHYSICS_CAPTURE_SCHEMA_VERSION),
    ("mapping_version", DAMAGE_LIFECYCLE_MAPPING_VERSION),
)
MAPPING_SOURCE_FACTS: Final = (
    "physics_capture_v1.states[].nodes[].life",
    "physics_capture_v1.events[].payload.reason[event_type=entity_destroyed]",
)
MAX_VALIDATION_COHORT_RECORDS: Final = 4096
_RECEIPT_TOKEN = object()


class MaterialDamageContractError(ValueError):
    """Raised for malformed mapping, source, or validation-receipt input."""


class MaterialDamageValidationError(MaterialDamageContractError):
    """Raised when a stored derivation fails source-bound validation."""

    def __init__(self, location: str, detail: str) -> None:
        self.location = location
        self.detail = detail
        super().__init__(f"invalid material/damage sidecar at {location}: {detail}")


@unique
class Availability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD = "unavailable_missing_engine_material_field"
    UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE = (
        "unavailable_insufficient_damage_lifecycle_evidence"
    )
    UNAVAILABLE_UNSUPPORTED_MAPPING = "unavailable_unsupported_mapping"


@dataclass(frozen=True, slots=True)
class DamageLifecycleMapping:
    """Immutable mapping configuration; verification belongs to a receipt."""

    def to_json(self) -> JsonObject:
        return {
            "mapping_version": DAMAGE_LIFECYCLE_MAPPING_VERSION,
            "schema_version": MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
            "source_facts": list(MAPPING_SOURCE_FACTS),
            "version_envelope": {key: value for key, value in SUPPORTED_VERSION_ENVELOPE},
        }

    @property
    def mapping_version(self) -> str:
        return DAMAGE_LIFECYCLE_MAPPING_VERSION

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_json())).hexdigest()


SUPPORTED_DAMAGE_LIFECYCLE_MAPPING: Final = DamageLifecycleMapping()


@dataclass(frozen=True, slots=True)
class DamageLifeDecreaseWitness:
    capture_id: str
    shot_id: str
    entity_id: str
    previous_fixed_step: int
    previous_life: float
    current_fixed_step: int
    current_life: float
    state_sha256: str
    events_sha256: str

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "current_fixed_step": self.current_fixed_step,
            "current_life": self.current_life,
            "entity_id": self.entity_id,
            "events_sha256": self.events_sha256,
            "previous_fixed_step": self.previous_fixed_step,
            "previous_life": self.previous_life,
            "shot_id": self.shot_id,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class DamageDestructionWitness:
    capture_id: str
    shot_id: str
    entity_id: str
    event_id: str
    event_fixed_step: int
    state_sha256: str
    events_sha256: str

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "entity_id": self.entity_id,
            "event_fixed_step": self.event_fixed_step,
            "event_id": self.event_id,
            "events_sha256": self.events_sha256,
            "shot_id": self.shot_id,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class DamageSourceRecord:
    capture_id: str
    shot_id: str
    state_sha256: str
    events_sha256: str
    life_decrease_witnesses: tuple[DamageLifeDecreaseWitness, ...]
    destruction_witnesses: tuple[DamageDestructionWitness, ...]

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "events_sha256": self.events_sha256,
            "life_decrease_witnesses": [witness.to_json() for witness in self.life_decrease_witnesses],
            "destruction_witnesses": [witness.to_json() for witness in self.destruction_witnesses],
            "shot_id": self.shot_id,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class DamageLifecycleValidationReceipt:
    mapping_version: str
    mapping_digest: str
    source_cohort_identity: str
    cohort_context: tuple[tuple[str, str], ...]
    source_records: tuple[DamageSourceRecord, ...]
    status: SemanticStatus
    _factory_marker: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("DamageLifecycleValidationReceipt must be issued by its public factory")

    @classmethod
    def _from_factory(
        cls,
        mapping_version: str,
        mapping_digest: str,
        source_cohort_identity: str,
        cohort_context: tuple[tuple[str, str], ...],
        source_records: tuple[DamageSourceRecord, ...],
        status: SemanticStatus,
    ) -> "DamageLifecycleValidationReceipt":
        receipt = object.__new__(cls)
        object.__setattr__(receipt, "mapping_version", mapping_version)
        object.__setattr__(receipt, "mapping_digest", mapping_digest)
        object.__setattr__(receipt, "source_cohort_identity", source_cohort_identity)
        object.__setattr__(receipt, "cohort_context", cohort_context)
        object.__setattr__(receipt, "source_records", source_records)
        object.__setattr__(receipt, "status", status)
        object.__setattr__(receipt, "_factory_marker", _RECEIPT_TOKEN)
        return receipt

    def to_json(self) -> JsonObject:
        return {
            "cohort_context": {key: value for key, value in self.cohort_context},
            "mapping_digest": self.mapping_digest,
            "mapping_version": self.mapping_version,
            "source_cohort_identity": self.source_cohort_identity,
            "source_records": [record.to_json() for record in self.source_records],
            "status": self.status.value,
        }


def canonical_json_bytes(value: JsonValue) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise MaterialDamageContractError("value must contain only finite JSON data") from error
    return (encoded + "\n").encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MaterialDamageContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MaterialDamageContractError(f"{field} must be a non-empty canonical string")
    return value


def _canonical_context(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise MaterialDamageContractError("cohort_context must be a string mapping")
    pairs = tuple(sorted((_nonempty(key, "cohort context key"), _nonempty(item, "cohort context value")) for key, item in value.items()))
    if len({key for key, _ in pairs}) != len(pairs):
        raise MaterialDamageContractError("cohort_context keys must be unique")
    return pairs


def _canonical_source_records(records: Sequence[DamageSourceRecord]) -> tuple[DamageSourceRecord, ...]:
    if not isinstance(records, tuple) or len(records) > MAX_VALIDATION_COHORT_RECORDS:
        raise MaterialDamageContractError("source cohort must be a bounded tuple")
    by_source: dict[tuple[str, str], DamageSourceRecord] = {}
    for record in records:
        _nonempty(record.capture_id, "source capture_id")
        _nonempty(record.shot_id, "source shot_id")
        _require_digest(record.state_sha256, "source state_sha256")
        _require_digest(record.events_sha256, "source events_sha256")
        if tuple(sorted(record.life_decrease_witnesses, key=lambda item: (item.current_fixed_step, item.entity_id))) != record.life_decrease_witnesses:
            raise MaterialDamageContractError("life decrease witnesses must be in canonical order")
        if tuple(sorted(record.destruction_witnesses, key=lambda item: (item.event_fixed_step, item.entity_id, item.event_id))) != record.destruction_witnesses:
            raise MaterialDamageContractError("destruction witnesses must be in canonical order")
        for witness in record.life_decrease_witnesses:
            if (
                witness.capture_id != record.capture_id
                or witness.shot_id != record.shot_id
                or witness.previous_fixed_step >= witness.current_fixed_step
                or witness.current_life >= witness.previous_life
                or witness.state_sha256 != record.state_sha256
                or witness.events_sha256 != record.events_sha256
            ):
                raise MaterialDamageContractError("life decrease witness is not bound to its source record")
        for witness in record.destruction_witnesses:
            if (
                witness.capture_id != record.capture_id
                or witness.shot_id != record.shot_id
                or witness.state_sha256 != record.state_sha256
                or witness.events_sha256 != record.events_sha256
            ):
                raise MaterialDamageContractError("destruction witness is not bound to its source record")
        key = (record.capture_id, record.shot_id)
        previous = by_source.get(key)
        if previous is not None and previous != record:
            raise MaterialDamageContractError("source capture/shot identity has conflicting digests")
        by_source[key] = record
    return tuple(by_source[key] for key in sorted(by_source))


def _source_record_from_json(value: Mapping[str, object]) -> DamageSourceRecord:
    def text(name: str) -> str:
        return _nonempty(value.get(name), f"source record {name}")

    def integer(name: str) -> int:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int):
            raise MaterialDamageContractError(f"source record {name} must be an integer")
        return item

    def number(name: str) -> float:
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MaterialDamageContractError(f"source record {name} must be numeric")
        return float(item)

    life_witnesses: list[DamageLifeDecreaseWitness] = []
    for raw in value.get("life_decrease_witnesses", []):
        if not isinstance(raw, Mapping):
            raise MaterialDamageContractError("life decrease witness must be an object")
        life_witnesses.append(DamageLifeDecreaseWitness(
            text_from(raw, "capture_id"), text_from(raw, "shot_id"), text_from(raw, "entity_id"),
            integer_from(raw, "previous_fixed_step"), number_from(raw, "previous_life"),
            integer_from(raw, "current_fixed_step"), number_from(raw, "current_life"),
            digest_from(raw, "state_sha256"), digest_from(raw, "events_sha256"),
        ))
    destruction_witnesses: list[DamageDestructionWitness] = []
    for raw in value.get("destruction_witnesses", []):
        if not isinstance(raw, Mapping):
            raise MaterialDamageContractError("destruction witness must be an object")
        destruction_witnesses.append(DamageDestructionWitness(
            text_from(raw, "capture_id"), text_from(raw, "shot_id"), text_from(raw, "entity_id"),
            text_from(raw, "event_id"), integer_from(raw, "event_fixed_step"),
            digest_from(raw, "state_sha256"), digest_from(raw, "events_sha256"),
        ))
    return DamageSourceRecord(
        text("capture_id"), text("shot_id"), _require_digest(value.get("state_sha256"), "source state_sha256"),
        _require_digest(value.get("events_sha256"), "source events_sha256"),
        tuple(life_witnesses), tuple(destruction_witnesses),
    )


def text_from(value: Mapping[str, object], name: str) -> str:
    return _nonempty(value.get(name), f"witness {name}")


def integer_from(value: Mapping[str, object], name: str) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise MaterialDamageContractError(f"witness {name} must be an integer")
    return item


def number_from(value: Mapping[str, object], name: str) -> float:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise MaterialDamageContractError(f"witness {name} must be numeric")
    return float(item)


def digest_from(value: Mapping[str, object], name: str) -> str:
    return _require_digest(value.get(name), f"witness {name}")


def source_cohort_identity_for_records(
    records: Sequence[DamageSourceRecord],
    *,
    cohort_context: Mapping[str, str] | None = None,
) -> str:
    canonical_records = _canonical_source_records(tuple(records))
    context = _canonical_context(cohort_context)
    payload = {
        "cohort_context": {key: value for key, value in context},
        "mapping_digest": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest,
        "mapping_version": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version,
        "source_records": [record.to_json() for record in canonical_records],
    }
    return f"damage-source-cohort-v1:sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def validate_damage_lifecycle_receipt_descriptor(value: Mapping[str, object]) -> dict[str, object]:
    """Validate a serialized receipt descriptor without reading source artifacts."""
    required = {
        "mapping_version", "mapping_digest", "source_cohort_identity", "cohort_context",
        "source_records", "status",
    }
    if set(value) != required:
        raise MaterialDamageContractError("receipt descriptor fields are incomplete or unknown")
    if value["mapping_version"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version:
        raise MaterialDamageContractError("receipt mapping version is stale")
    if value["mapping_digest"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest:
        raise MaterialDamageContractError("receipt mapping digest is stale")
    if not isinstance(value["cohort_context"], Mapping):
        raise MaterialDamageContractError("receipt cohort_context must be an object")
    context = _canonical_context(value["cohort_context"])
    raw_records = value["source_records"]
    if not isinstance(raw_records, list):
        raise MaterialDamageContractError("receipt source_records must be a list")
    records = _canonical_source_records(tuple(_source_record_from_json(item) for item in raw_records if isinstance(item, Mapping)))
    if len(records) != len(raw_records):
        raise MaterialDamageContractError("receipt source_records contains malformed entries")
    identity = source_cohort_identity_for_records(records, cohort_context=dict(context))
    if value["source_cohort_identity"] != identity:
        raise MaterialDamageContractError("receipt source-cohort identity is stale")
    expected_status = (
        SemanticStatus.ENGINE_VERIFIED
        if any(record.life_decrease_witnesses for record in records)
        and any(record.destruction_witnesses for record in records)
        else SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION
    )
    if value["status"] != expected_status.value:
        raise MaterialDamageContractError("receipt status is inconsistent with concrete witnesses")
    return {
        "mapping_version": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version,
        "mapping_digest": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest,
        "source_cohort_identity": identity,
        "cohort_context": dict(context),
        "source_records": [record.to_json() for record in records],
        "status": expected_status.value,
    }


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _event_payload(event: object) -> JsonObject:
    try:
        payload = json.loads(event.payload_json)  # type: ignore[union-attr]
    except (AttributeError, json.JSONDecodeError) as error:
        raise MaterialDamageContractError("validated event payload is not JSON") from error
    if not isinstance(payload, dict):
        raise MaterialDamageContractError("validated event payload must be an object")
    return payload


def _damage_event(event: object) -> bool:
    return (
        event.event_type is EventType.ENTITY_DESTROYED  # type: ignore[union-attr]
        and _event_payload(event).get("reason") == "damage"
    )


def _lifecycle_witnesses(
    capture: PhysicsCapture,
    state_sha256: str,
    events_sha256: str,
) -> tuple[tuple[DamageLifeDecreaseWitness, ...], tuple[DamageDestructionWitness, ...]]:
    previous: dict[str, tuple[int, float]] = {}
    life_witnesses: list[DamageLifeDecreaseWitness] = []
    for state in sorted(capture.states, key=lambda item: item.clock.fixed_step):
        for node in state.nodes:
            entity_id = str(node.entity_id)
            if node.life is not None and entity_id in previous and node.life < previous[entity_id][1]:
                previous_step, previous_life = previous[entity_id]
                life_witnesses.append(
                    DamageLifeDecreaseWitness(
                        str(state.clock.capture_id), str(state.clock.shot_id), entity_id,
                        previous_step, previous_life, state.clock.fixed_step, node.life,
                        state_sha256, events_sha256,
                    )
                )
            if node.life is not None:
                previous[entity_id] = (state.clock.fixed_step, node.life)
    destruction_witnesses = tuple(
        DamageDestructionWitness(
            str(capture.header.clock.capture_id), str(capture.header.clock.shot_id),
            str(participant), str(event.event_id), event.clock.fixed_step,
            state_sha256, events_sha256,
        )
        for event in capture.events if _damage_event(event)
        for participant in event.participants
    )
    return (
        tuple(sorted(life_witnesses, key=lambda item: (item.current_fixed_step, item.entity_id))),
        tuple(sorted(destruction_witnesses, key=lambda item: (item.event_fixed_step, item.entity_id, item.event_id))),
    )


def _source(value: Source) -> tuple[PhysicsCapture, DamageSourceRecord]:
    if isinstance(value, PhysicsCapture):
        capture = value
        if not capture.states:
            raise MaterialDamageContractError("source must be a non-empty PhysicsCapture")
        state_bytes = canonical_json_bytes({"header": asdict(capture.header), "states": [asdict(state) for state in capture.states]})
        event_bytes = canonical_json_bytes({"events": [asdict(event) for event in capture.events]})
    elif isinstance(value, Path):
        state_path = value / STATE_SIDECAR
        event_path = value / EVENT_SIDECAR
        if not state_path.is_file() or not event_path.is_file():
            raise MaterialDamageContractError("validated shot sidecars are missing")
        try:
            capture = load_physics_capture(state_path, event_path)
            state_bytes = state_path.read_bytes()
            event_bytes = event_path.read_bytes()
        except (OSError, ValueError) as error:
            raise MaterialDamageContractError("validated shot sidecars could not be loaded") from error
    else:
        raise MaterialDamageContractError("source must be PhysicsCapture or Path")
    life_witnesses, destruction_witnesses = _lifecycle_witnesses(
        capture, _digest_bytes(state_bytes), _digest_bytes(event_bytes)
    )
    record = DamageSourceRecord(
        capture_id=str(capture.header.clock.capture_id),
        shot_id=str(capture.header.clock.shot_id),
        state_sha256=_digest_bytes(state_bytes),
        events_sha256=_digest_bytes(event_bytes),
        life_decrease_witnesses=life_witnesses,
        destruction_witnesses=destruction_witnesses,
    )
    return capture, record


def build_damage_lifecycle_validation_receipt(
    sources: Sequence[Source],
    *,
    cohort_context: Mapping[str, str] | None = None,
) -> DamageLifecycleValidationReceipt:
    """Inspect one bounded full source cohort and issue its immutable receipt."""
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise MaterialDamageContractError("sources must be a bounded sequence")
    if len(sources) > MAX_VALIDATION_COHORT_RECORDS:
        raise MaterialDamageContractError("source cohort exceeds the bounded record limit")
    loaded = tuple(_source(source) for source in sources)
    records = _canonical_source_records(
        tuple(sorted((record for _, record in loaded), key=lambda item: (item.capture_id, item.shot_id)))
    )
    context = _canonical_context(cohort_context)
    identity = source_cohort_identity_for_records(records, cohort_context=dict(context))
    status = (
        SemanticStatus.ENGINE_VERIFIED
        if any(record.life_decrease_witnesses for record in records)
        and any(record.destruction_witnesses for record in records)
        else SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION
    )
    return DamageLifecycleValidationReceipt._from_factory(
        SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version,
        SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest,
        identity,
        context,
        records,
        status,
    )


def _receipt_valid(receipt: object) -> bool:
    if not isinstance(receipt, DamageLifecycleValidationReceipt) or receipt._factory_marker is not _RECEIPT_TOKEN:
        return False
    if receipt.mapping_version != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version:
        return False
    if receipt.mapping_digest != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest:
        return False
    try:
        records = _canonical_source_records(receipt.source_records)
        context = _canonical_context(dict(receipt.cohort_context))
    except MaterialDamageContractError:
        return False
    if context != receipt.cohort_context:
        return False
    if receipt.source_cohort_identity != source_cohort_identity_for_records(records, cohort_context=dict(context)):
        return False
    expected_status = (
        SemanticStatus.ENGINE_VERIFIED
        if any(record.life_decrease_witnesses for record in records)
        and any(record.destruction_witnesses for record in records)
        else SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION
    )
    return receipt.status is expected_status


def _receipt_matches(receipt: object, source_record: DamageSourceRecord) -> bool:
    return _receipt_valid(receipt) and source_record in receipt.source_records  # type: ignore[union-attr]


@dataclass(frozen=True, slots=True)
class MaterialDamageRecord:
    capture_id: str
    shot_id: str
    render_frame: int
    fixed_step: int
    entity_id: str
    material_label: str
    material_availability: Availability
    damage_label: str
    damage_availability: Availability
    damage_evidence_event_ids: tuple[str, ...]
    state_sha256: str
    events_sha256: str
    mapping_version: str | None
    mapping_digest: str | None
    source_cohort_identity: str | None
    validation_status: SemanticStatus | None

    @property
    def frame_scoped_key(self) -> tuple[str, str, int, int, str]:
        return (self.capture_id, self.shot_id, self.render_frame, self.fixed_step, self.entity_id)

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "damage_availability": self.damage_availability.value,
            "damage_evidence_event_ids": list(self.damage_evidence_event_ids),
            "damage_label": self.damage_label,
            "entity_id": self.entity_id,
            "events_sha256": self.events_sha256,
            "fixed_step": self.fixed_step,
            "mapping_digest": self.mapping_digest,
            "mapping_version": self.mapping_version,
            "material_availability": self.material_availability.value,
            "material_label": self.material_label,
            "render_frame": self.render_frame,
            "shot_id": self.shot_id,
            "source_cohort_identity": self.source_cohort_identity,
            "state_sha256": self.state_sha256,
            "validation_status": self.validation_status.value if self.validation_status else None,
        }


@dataclass(frozen=True, slots=True)
class MaterialDamageArtifact:
    capture_id: str
    shot_id: str
    state_sha256: str
    events_sha256: str
    mapping_version: str | None
    mapping_digest: str | None
    source_cohort_identity: str | None
    cohort_context: tuple[tuple[str, str], ...]
    source_records: tuple[DamageSourceRecord, ...]
    validation_status: SemanticStatus | None
    records: tuple[MaterialDamageRecord, ...]

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "cohort_context": {key: value for key, value in self.cohort_context},
            "events_sha256": self.events_sha256,
            "mapping_digest": self.mapping_digest,
            "mapping_version": self.mapping_version,
            "record_count": len(self.records),
            "records": [record.to_json() for record in self.records],
            "schema_version": MATERIAL_DAMAGE_DERIVATION_SCHEMA_VERSION,
            "shot_id": self.shot_id,
            "source_cohort_identity": self.source_cohort_identity,
            "source_records": [record.to_json() for record in self.source_records],
            "state_sha256": self.state_sha256,
            "validation_status": self.validation_status.value if self.validation_status else None,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_json())


def _damage_event_projection(
    capture: PhysicsCapture,
) -> tuple[dict[tuple[int, str], tuple[str, ...]], tuple[tuple[int, int, str, tuple[str, ...]], ...]]:
    states = tuple(sorted(capture.states, key=lambda item: item.clock.fixed_step))
    attached: dict[tuple[int, str], list[str]] = {}
    scoped: list[tuple[int, int, str, tuple[str, ...]]] = []
    for event in capture.events:
        if not _damage_event(event):
            continue
        for participant in event.participants:
            entity_id = str(participant)
            target = next(
                (
                    state
                    for state in states
                    if state.clock.fixed_step >= event.clock.fixed_step
                    and any(str(node.entity_id) == entity_id for node in state.nodes)
                ),
                None,
            )
            if target is None:
                scoped.append((event.clock.render_frame, event.clock.fixed_step, entity_id, (str(event.event_id),)))
            else:
                attached.setdefault((target.clock.fixed_step, entity_id), []).append(str(event.event_id))
    return (
        {key: tuple(sorted(value)) for key, value in attached.items()},
        tuple(sorted(scoped, key=lambda item: (item[1], item[0], item[2]))),
    )


def _record(
    *,
    capture_id: str,
    shot_id: str,
    render_frame: int,
    fixed_step: int,
    entity_id: str,
    damage_label: str,
    damage_availability: Availability,
    evidence: tuple[str, ...],
    state_record: DamageSourceRecord,
    receipt: DamageLifecycleValidationReceipt | None,
    mapping_supported: bool,
) -> MaterialDamageRecord:
    return MaterialDamageRecord(
        capture_id=capture_id,
        shot_id=shot_id,
        render_frame=render_frame,
        fixed_step=fixed_step,
        entity_id=entity_id,
        material_label=MATERIAL_UNAVAILABLE_LABEL,
        material_availability=Availability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD,
        damage_label=damage_label,
        damage_availability=damage_availability,
        damage_evidence_event_ids=evidence,
        state_sha256=state_record.state_sha256,
        events_sha256=state_record.events_sha256,
        mapping_version=SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version if mapping_supported else None,
        mapping_digest=SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest if mapping_supported else None,
        source_cohort_identity=receipt.source_cohort_identity if receipt else None,
        validation_status=receipt.status if receipt else None,
    )


def _derive(
    capture: PhysicsCapture,
    source_record: DamageSourceRecord,
    receipt: DamageLifecycleValidationReceipt | None,
    mapping_supported: bool,
) -> MaterialDamageArtifact:
    event_projection, event_scoped = _damage_event_projection(capture)
    verified = mapping_supported and receipt is not None and receipt.status is SemanticStatus.ENGINE_VERIFIED
    unavailable = (
        Availability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE
        if mapping_supported
        else Availability.UNAVAILABLE_UNSUPPORTED_MAPPING
    )
    records: list[MaterialDamageRecord] = []
    previous: dict[str, tuple[int, float | None]] = {}
    states = tuple(sorted(capture.states, key=lambda item: item.clock.fixed_step))
    for state in states:
        for node in state.nodes:
            entity_id = str(node.entity_id)
            evidence = event_projection.get((state.clock.fixed_step, entity_id), ())
            predecessor = previous.get(entity_id)
            if not verified:
                label = MATERIAL_UNAVAILABLE_LABEL
                availability = unavailable
            elif evidence or (
                predecessor is not None
                and node.life is not None
                and predecessor[1] is not None
                and node.life < predecessor[1]
            ):
                label = DAMAGE_LABEL
                availability = Availability.AVAILABLE
            elif predecessor is not None and node.life is not None and predecessor[1] == node.life:
                label = NO_DAMAGE_LABEL
                availability = Availability.AVAILABLE
            else:
                label = MATERIAL_UNAVAILABLE_LABEL
                availability = Availability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE
            records.append(
                _record(
                    capture_id=str(state.clock.capture_id),
                    shot_id=str(state.clock.shot_id),
                    render_frame=state.clock.render_frame,
                    fixed_step=state.clock.fixed_step,
                    entity_id=entity_id,
                    damage_label=label,
                    damage_availability=availability,
                    evidence=evidence,
                    state_record=source_record,
                    receipt=receipt if _receipt_matches(receipt, source_record) else None,
                    mapping_supported=mapping_supported,
                )
            )
            if node.life is not None:
                previous[entity_id] = (state.clock.fixed_step, node.life)
    for render_frame, fixed_step, entity_id, evidence in event_scoped:
        records.append(
            _record(
                capture_id=str(capture.header.clock.capture_id),
                shot_id=str(capture.header.clock.shot_id),
                render_frame=render_frame,
                fixed_step=fixed_step,
                entity_id=entity_id,
                damage_label=DAMAGE_LABEL if verified else MATERIAL_UNAVAILABLE_LABEL,
                damage_availability=Availability.AVAILABLE if verified else unavailable,
                evidence=evidence,
                state_record=source_record,
                receipt=receipt if _receipt_matches(receipt, source_record) else None,
                mapping_supported=mapping_supported,
            )
        )
    valid_receipt = receipt if _receipt_matches(receipt, source_record) else None
    return MaterialDamageArtifact(
        capture_id=source_record.capture_id,
        shot_id=source_record.shot_id,
        state_sha256=source_record.state_sha256,
        events_sha256=source_record.events_sha256,
        mapping_version=SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version if mapping_supported else None,
        mapping_digest=SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest if mapping_supported else None,
        source_cohort_identity=valid_receipt.source_cohort_identity if valid_receipt else None,
        cohort_context=valid_receipt.cohort_context if valid_receipt else (),
        source_records=valid_receipt.source_records if valid_receipt else (),
        validation_status=valid_receipt.status if valid_receipt else None,
        records=tuple(sorted(records, key=lambda record: (record.fixed_step, record.render_frame, record.entity_id))),
    )


def derive_material_damage(
    source: Source,
    *,
    receipt: DamageLifecycleValidationReceipt | None = None,
    mapping: DamageLifecycleMapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
) -> MaterialDamageArtifact:
    """Derive source-grounded labels; absent or invalid receipts fail closed."""
    capture, source_record = _source(source)
    mapping_supported = mapping is SUPPORTED_DAMAGE_LIFECYCLE_MAPPING
    return _derive(
        capture,
        source_record,
        receipt if _receipt_matches(receipt, source_record) else None,
        mapping_supported,
    )


def _require_receipt_for_source(
    source: Source,
    receipt: DamageLifecycleValidationReceipt | None,
) -> tuple[PhysicsCapture, DamageSourceRecord]:
    capture, source_record = _source(source)
    if not _receipt_matches(receipt, source_record):
        raise MaterialDamageValidationError("receipt", "missing, invalid, or source-mismatched validation receipt")
    return capture, source_record


def write_material_damage_sidecar(
    path: Path,
    source: Source,
    *,
    receipt: DamageLifecycleValidationReceipt | None,
    mapping: DamageLifecycleMapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
) -> MaterialDamageArtifact:
    if mapping is not SUPPORTED_DAMAGE_LIFECYCLE_MAPPING:
        raise MaterialDamageValidationError(str(path), "unsupported mapping cannot be written")
    capture, source_record = _require_receipt_for_source(source, receipt)
    artifact = _derive(capture, source_record, receipt, True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            stream.write(artifact.to_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise MaterialDamageValidationError(str(path), "sidecar could not be written") from error
    return artifact


def validate_material_damage_sidecar(
    path: Path,
    source: Source,
    *,
    receipt: DamageLifecycleValidationReceipt | None,
    mapping: DamageLifecycleMapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
) -> MaterialDamageArtifact:
    if not path.is_file():
        raise MaterialDamageValidationError(str(path), "sidecar is missing")
    if mapping is not SUPPORTED_DAMAGE_LIFECYCLE_MAPPING:
        raise MaterialDamageValidationError(str(path), "unsupported mapping cannot validate")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
            raise MaterialDamageValidationError(str(path), "sidecar is not canonical JSON")
        capture, source_record = _require_receipt_for_source(source, receipt)
        expected = _derive(capture, source_record, receipt, True)
    except MaterialDamageValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, MaterialDamageContractError) as error:
        raise MaterialDamageValidationError(str(path), "source or sidecar is malformed") from error
    if raw != expected.to_bytes():
        raise MaterialDamageValidationError(str(path), "stored artifact differs from canonical re-derivation")
    return expected


__all__ = [
    "Availability",
    "DAMAGE_LIFECYCLE_MAPPING_VERSION",
    "DamageLifecycleMapping",
    "DamageLifecycleValidationReceipt",
    "DamageSourceRecord",
    "MAPPING_SOURCE_FACTS",
    "MATERIAL_DAMAGE_DERIVATION_SCHEMA_VERSION",
    "MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION",
    "MATERIAL_DAMAGE_SIDECAR",
    "MATERIAL_UNAVAILABLE_LABEL",
    "MaterialDamageArtifact",
    "MaterialDamageContractError",
    "MaterialDamageRecord",
    "MaterialDamageValidationError",
    "NO_DAMAGE_LABEL",
    "SUPPORTED_DAMAGE_LIFECYCLE_MAPPING",
    "SUPPORTED_VERSION_ENVELOPE",
    "build_damage_lifecycle_validation_receipt",
    "canonical_json_bytes",
    "derive_material_damage",
    "source_cohort_identity_for_records",
    "validate_damage_lifecycle_receipt_descriptor",
    "validate_material_damage_sidecar",
    "write_material_damage_sidecar",
]
