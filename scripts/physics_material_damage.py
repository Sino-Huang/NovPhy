from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum, unique
import hashlib
import json
import os
from pathlib import Path
from typing import Final, TypeAlias

from scripts.physics_capture_contract import EVENT_SIDECAR, STATE_SIDECAR, load_physics_capture
from scripts.physics_capture_types import EventType, PhysicsCapture
from scripts.physics_macro_labels import SemanticStatus


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION: Final = "physics_material_damage_mapping_v1"
MATERIAL_DAMAGE_DERIVATION_SCHEMA_VERSION: Final = "physics_material_damage_derivation_v2"
MATERIAL_DAMAGE_SIDECAR: Final = "physics_material_damage.json"
DAMAGE_LIFECYCLE_MAPPING_VERSION: Final = "damage_lifecycle_mapping_v1"
PHYSICS_CAPTURE_SCHEMA_VERSION: Final = "physics_capture_v1"
MATERIAL_UNAVAILABLE_LABEL: Final = "unavailable"
DAMAGE_LABEL: Final = "damage"
NO_DAMAGE_LABEL: Final = "no_damage"
SOURCE_COHORT_IDENTITY: Final = "engine-contract:physics_capture_v1"
SUPPORTED_VERSION_ENVELOPE: Final = (
    ("capture_schema_version", PHYSICS_CAPTURE_SCHEMA_VERSION),
    ("mapping_version", DAMAGE_LIFECYCLE_MAPPING_VERSION),
)
MAPPING_SOURCE_FACTS: Final = (
    "SceneNode.life",
    "EventRecord(event_type=entity_destroyed,payload.reason=damage)",
)


class MaterialDamageContractError(ValueError):
    """Raised for unsupported mapping, source, or version-contract input."""


class MaterialDamageValidationError(MaterialDamageContractError):
    """Raised when a stored material/damage artifact fails closed validation."""

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
    UNAVAILABLE_VERSION_ENVELOPE_MISMATCH = "unavailable_version_envelope_mismatch"


@dataclass(frozen=True, slots=True)
class DamageLifecycleMapping:
    """The sole supported mapping; its status and source facts are implementation-fixed."""

    def to_json(self) -> JsonObject:
        return {
            "mapping_version": DAMAGE_LIFECYCLE_MAPPING_VERSION,
            "representative_validation_status": SemanticStatus.ENGINE_VERIFIED.value,
            "schema_version": MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
            "source_cohort_identity": SOURCE_COHORT_IDENTITY,
            "source_facts": list(MAPPING_SOURCE_FACTS),
            "version_envelope": {key: value for key, value in SUPPORTED_VERSION_ENVELOPE},
        }

    @property
    def mapping_version(self) -> str:
        return DAMAGE_LIFECYCLE_MAPPING_VERSION

    @property
    def source_cohort_identity(self) -> str:
        return SOURCE_COHORT_IDENTITY

    @property
    def representative_validation_status(self) -> SemanticStatus:
        return SemanticStatus.ENGINE_VERIFIED

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_json())).hexdigest()


SUPPORTED_DAMAGE_LIFECYCLE_MAPPING: Final = DamageLifecycleMapping()


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


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MaterialDamageContractError(f"{field} must be a non-empty canonical string")
    return value


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_mapping(mapping: object) -> DamageLifecycleMapping:
    if mapping is not SUPPORTED_DAMAGE_LIFECYCLE_MAPPING:
        raise MaterialDamageContractError("only the built-in damage lifecycle mapping is supported")
    return SUPPORTED_DAMAGE_LIFECYCLE_MAPPING


def _canonical_envelope(value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or not value:
        raise MaterialDamageContractError("version_envelope must be a non-empty tuple")
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not item[0]
            or not item[1]
        ):
            raise MaterialDamageContractError("version_envelope entries must be string pairs")
    result = tuple(sorted(value))
    if len({key for key, _ in result}) != len(result):
        raise MaterialDamageContractError("version_envelope keys must be unique")
    return result


def _require_envelope(envelope: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    canonical = _canonical_envelope(envelope)
    if canonical != SUPPORTED_VERSION_ENVELOPE:
        raise MaterialDamageContractError("version envelope is not supported by the built-in mapping")
    return canonical


def _capture_digest(capture: PhysicsCapture, field: str) -> str:
    if field == "state":
        value: JsonValue = {
            "header": asdict(capture.header),
            "states": [asdict(state) for state in capture.states],
        }
    else:
        value = {"events": [asdict(event) for event in capture.events]}
    return _digest_bytes(canonical_json_bytes(value))


def _source_from_capture(capture: PhysicsCapture) -> tuple[PhysicsCapture, str, str]:
    if not isinstance(capture, PhysicsCapture) or not capture.states:
        raise MaterialDamageContractError("source must be a non-empty PhysicsCapture")
    return capture, _capture_digest(capture, "state"), _capture_digest(capture, "event")


def _source_from_shot(shot_dir: Path) -> tuple[PhysicsCapture, str, str]:
    state_path = shot_dir / STATE_SIDECAR
    event_path = shot_dir / EVENT_SIDECAR
    if not state_path.is_file() or not event_path.is_file():
        raise MaterialDamageContractError("validated shot sidecars are missing")
    try:
        capture = load_physics_capture(state_path, event_path)
        return capture, _digest_bytes(state_path.read_bytes()), _digest_bytes(event_path.read_bytes())
    except (OSError, ValueError) as error:
        raise MaterialDamageContractError("validated shot sidecars could not be loaded") from error


def _source(value: PhysicsCapture | Path) -> tuple[PhysicsCapture, str, str]:
    if isinstance(value, PhysicsCapture):
        return _source_from_capture(value)
    if isinstance(value, Path):
        return _source_from_shot(value)
    raise MaterialDamageContractError("source must be PhysicsCapture or validated shot directory")


def _event_payload(event: object) -> JsonObject:
    try:
        payload = json.loads(event.payload_json)  # type: ignore[union-attr]
    except (AttributeError, json.JSONDecodeError) as error:
        raise MaterialDamageContractError("validated event payload is not JSON") from error
    if not isinstance(payload, dict):
        raise MaterialDamageContractError("validated event payload must be an object")
    return payload


def _damage_events(capture: PhysicsCapture) -> dict[tuple[int, int, str], tuple[str, ...]]:
    result: dict[tuple[int, int, str], list[str]] = {}
    for event in capture.events:
        if event.event_type is not EventType.ENTITY_DESTROYED:
            continue
        if _event_payload(event).get("reason") != "damage":
            continue
        key = (event.clock.render_frame, event.clock.fixed_step, "")
        for participant in event.participants:
            result.setdefault((key[0], key[1], str(participant)), []).append(str(event.event_id))
    return {key: tuple(sorted(value)) for key, value in result.items()}


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
    mapping_version: str
    mapping_digest: str
    source_cohort_identity: str

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
        }


@dataclass(frozen=True, slots=True)
class MaterialDamageArtifact:
    capture_id: str
    shot_id: str
    state_sha256: str
    events_sha256: str
    mapping_version: str
    mapping_digest: str
    version_envelope: tuple[tuple[str, str], ...]
    source_cohort_identity: str
    records: tuple[MaterialDamageRecord, ...]

    def to_json(self) -> JsonObject:
        return {
            "capture_id": self.capture_id,
            "events_sha256": self.events_sha256,
            "mapping_digest": self.mapping_digest,
            "mapping_version": self.mapping_version,
            "record_count": len(self.records),
            "records": [record.to_json() for record in self.records],
            "schema_version": MATERIAL_DAMAGE_DERIVATION_SCHEMA_VERSION,
            "shot_id": self.shot_id,
            "source_cohort_identity": self.source_cohort_identity,
            "state_sha256": self.state_sha256,
            "version_envelope": {key: value for key, value in self.version_envelope},
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_json())


def _derive(
    capture: PhysicsCapture,
    mapping: DamageLifecycleMapping,
    state_sha256: str,
    events_sha256: str,
    version_envelope: tuple[tuple[str, str], ...],
    source_cohort_identity: str,
) -> MaterialDamageArtifact:
    previous_life: dict[str, float] = {}
    damage_events = _damage_events(capture)
    records: list[MaterialDamageRecord] = []
    seen_keys: set[tuple[int, int, str]] = set()
    envelope_supported = version_envelope == SUPPORTED_VERSION_ENVELOPE
    for state in capture.states:
        frame = state.clock.render_frame
        fixed_step = state.clock.fixed_step
        for node in state.nodes:
            entity_id = str(node.entity_id)
            seen_keys.add((frame, fixed_step, entity_id))
            evidence = damage_events.get((frame, fixed_step, entity_id), ())
            life_decreased = (
                node.life is not None
                and entity_id in previous_life
                and node.life < previous_life[entity_id]
            )
            if not envelope_supported:
                damage_label = MATERIAL_UNAVAILABLE_LABEL
                damage_availability = Availability.UNAVAILABLE_VERSION_ENVELOPE_MISMATCH
            elif evidence or life_decreased:
                damage_label = DAMAGE_LABEL
                damage_availability = Availability.AVAILABLE
            elif node.life is not None:
                damage_label = NO_DAMAGE_LABEL
                damage_availability = Availability.AVAILABLE
            else:
                damage_label = MATERIAL_UNAVAILABLE_LABEL
                damage_availability = Availability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE
            if node.life is not None:
                previous_life[entity_id] = node.life
            records.append(
                MaterialDamageRecord(
                    capture_id=str(state.clock.capture_id),
                    shot_id=str(state.clock.shot_id),
                    render_frame=frame,
                    fixed_step=fixed_step,
                    entity_id=entity_id,
                    material_label=MATERIAL_UNAVAILABLE_LABEL,
                    material_availability=Availability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD,
                    damage_label=damage_label,
                    damage_availability=damage_availability,
                    damage_evidence_event_ids=evidence,
                    state_sha256=state_sha256,
                    events_sha256=events_sha256,
                    mapping_version=mapping.mapping_version,
                    mapping_digest=mapping.digest,
                    source_cohort_identity=source_cohort_identity,
                )
            )
    if envelope_supported:
        for (frame, fixed_step, entity_id), evidence in damage_events.items():
            if (frame, fixed_step, entity_id) in seen_keys:
                continue
            records.append(
                MaterialDamageRecord(
                    capture_id=str(capture.header.clock.capture_id),
                    shot_id=str(capture.header.clock.shot_id),
                    render_frame=frame,
                    fixed_step=fixed_step,
                    entity_id=entity_id,
                    material_label=MATERIAL_UNAVAILABLE_LABEL,
                    material_availability=Availability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD,
                    damage_label=DAMAGE_LABEL,
                    damage_availability=Availability.AVAILABLE,
                    damage_evidence_event_ids=evidence,
                    state_sha256=state_sha256,
                    events_sha256=events_sha256,
                    mapping_version=mapping.mapping_version,
                    mapping_digest=mapping.digest,
                    source_cohort_identity=source_cohort_identity,
                )
            )
    return MaterialDamageArtifact(
        capture_id=str(capture.header.clock.capture_id),
        shot_id=str(capture.header.clock.shot_id),
        state_sha256=state_sha256,
        events_sha256=events_sha256,
        mapping_version=mapping.mapping_version,
        mapping_digest=mapping.digest,
        version_envelope=version_envelope,
        source_cohort_identity=source_cohort_identity,
        records=tuple(sorted(records, key=lambda record: record.frame_scoped_key)),
    )


def derive_material_damage(
    source: PhysicsCapture | Path,
    *,
    mapping: DamageLifecycleMapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
    version_envelope: tuple[tuple[str, str], ...] = SUPPORTED_VERSION_ENVELOPE,
    source_cohort_identity: str = SOURCE_COHORT_IDENTITY,
) -> MaterialDamageArtifact:
    """Derive lifecycle labels from a full capture or validated shot sidecars."""
    supported = _require_mapping(mapping)
    envelope = _canonical_envelope(version_envelope)
    _nonempty(source_cohort_identity, "source_cohort_identity")
    capture, state_sha256, events_sha256 = _source(source)
    return _derive(capture, supported, state_sha256, events_sha256, envelope, source_cohort_identity)


def write_material_damage_sidecar(
    path: Path,
    source: PhysicsCapture | Path,
    *,
    mapping: DamageLifecycleMapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
    version_envelope: tuple[tuple[str, str], ...] = SUPPORTED_VERSION_ENVELOPE,
    source_cohort_identity: str = SOURCE_COHORT_IDENTITY,
) -> MaterialDamageArtifact:
    """Write only a supported, envelope-bound canonical artifact."""
    envelope = _require_envelope(version_envelope)
    artifact = derive_material_damage(
        source,
        mapping=mapping,
        version_envelope=envelope,
        source_cohort_identity=source_cohort_identity,
    )
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
    source: PhysicsCapture | Path,
    *,
    mapping: DamageLifecycleMapping = SUPPORTED_DAMAGE_LIFECYCLE_MAPPING,
    version_envelope: tuple[tuple[str, str], ...] = SUPPORTED_VERSION_ENVELOPE,
    source_cohort_identity: str = SOURCE_COHORT_IDENTITY,
) -> MaterialDamageArtifact:
    if not path.is_file():
        raise MaterialDamageValidationError(str(path), "sidecar is missing")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
            raise MaterialDamageValidationError(str(path), "sidecar is not canonical JSON")
        expected = derive_material_damage(
            source,
            mapping=mapping,
            version_envelope=version_envelope,
            source_cohort_identity=source_cohort_identity,
        )
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
    "SOURCE_COHORT_IDENTITY",
    "SUPPORTED_DAMAGE_LIFECYCLE_MAPPING",
    "SUPPORTED_VERSION_ENVELOPE",
    "canonical_json_bytes",
    "derive_material_damage",
    "validate_material_damage_sidecar",
    "write_material_damage_sidecar",
]
