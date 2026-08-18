"""Capability-gated representative-pilot collection and assessment."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Final, Literal

from scripts.collection_plan import (
    CollectionPlanRuntime,
    LoadedCollectionPlan,
    PLAN_COPY_FILENAME,
    assert_plan_unchanged,
    execute_collection_plan,
    load_collection_plan,
)
from scripts.cohort_partition import CohortPartitionManifest, audit_cohort_partition_manifest
from scripts.physics_artifact_validation import validate_physics_shot_artifact
from scripts.physics_capture_contract import load_physics_capture
from scripts.physics_macro_labels import (
    Availability,
    DERIVATION_SPEC_VERSION,
    MacroPredicate,
    SemanticStatus,
    derivation_spec_digest,
    derivation_spec_json,
    derive_macro_labels_for_shot,
)
from scripts.physics_material_damage import (
    MaterialDamageContractError,
    MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
    MATERIAL_UNAVAILABLE_LABEL,
    MAPPING_SOURCE_FACTS,
    Availability as MaterialDamageAvailability,
    build_damage_lifecycle_validation_receipt,
    derive_material_damage,
    validate_damage_lifecycle_receipt_descriptor,
)
from scripts.physics_relational_supervision import (
    validate_relational_supervision,
    write_relational_supervision,
)
from scripts.scenario_manifest import ScenarioManifest, load_manifest, verify_replay


PILOT_REPORT_SCHEMA: Final = "representative_pilot_report_v3"
PILOT_REPORT_VERSION: Final = 3
PILOT_REPORT_IDENTITY_NAMESPACE: Final = "representative-pilot-report-v3"
PILOT_REPORT_FILENAME: Final = "representative_pilot_report.json"
MACRO_SEMANTICS_SCHEMA: Final = "representative_macro_semantics_v1"
MATERIAL_DAMAGE_SEMANTICS_SCHEMA: Final = "representative_material_damage_semantics_v1"
MATERIAL_MISSING_ENGINE_FIELD_REASON: Final = (
    "physics_capture_v1 does not export a material field"
)
MATERIAL_DAMAGE_PENDING_REASON: Final = (
    "no representative engine lifecycle evidence verifies the damage mapping"
)
MATERIAL_DAMAGE_COHORT_NAMESPACE: Final = "material-damage-source-cohort-v1"
TARGET_MACRO_PREDICATES: Final = (
    MacroPredicate.CASCADE_ACTIVE.value,
    MacroPredicate.COLLAPSED.value,
    MacroPredicate.PIGS_CLEARED.value,
)
MACRO_SEMANTICS_PENDING_REASON: Final = (
    "no authorized non-fixture representative engine evidence is recorded; "
    "fixture-derived labels are diagnostic only and do not validate semantics"
)

CAPABILITY_SCENARIO_LINEAGE: Final = "scenario_lineage"
CAPABILITY_INTERVENTION_REPRESENTATION: Final = "intervention_representation"
CAPABILITY_BOUNDED_NEGATIVE_EVIDENCE: Final = "bounded_negative_evidence"
CAPABILITY_COVERAGE_STRATA: Final = "coverage_strata"
CAPABILITY_ATOMIC_PHYSICS_ARTIFACT: Final = "atomic_physics_artifact"
CAPABILITY_CAUSAL_ENTITIES: Final = "causal_entities"
CAPABILITY_RAW_CONTACTS: Final = "raw_contacts"
CAPABILITY_DERIVED_SUPPORT: Final = "derived_support"
CAPABILITY_KINEMATICS: Final = "kinematics"
CAPABILITY_MACRO_EVENTS: Final = "macro_events"
CAPABILITY_CANONICAL_OBSERVATIONS: Final = "canonical_observations"
CAPABILITY_AGENT_OBSERVATIONS: Final = "agent_observations"
CAPABILITY_INITIAL_STATE_IDENTITY: Final = "initial_engine_state_identity"
CAPABILITY_RELATIONAL_SUPERVISION: Final = "relational_supervision"
CAPABILITY_DETERMINISTIC_REPLAY: Final = "deterministic_replay"
CAPABILITY_INSTANCE_HELD_OUT_PARTITION: Final = "instance_held_out_partition"
CAPABILITY_TEMPLATE_HELD_OUT_PARTITION: Final = "template_held_out_partition"
CAPABILITY_PHYSICAL_REGIME_GATE: Final = "physical_regime_gate"
CAPABILITY_MATERIAL_DAMAGE_MAPPING: Final = "material_damage_mapping"
CAPABILITY_REPRESENTATIVE_MACRO_SEMANTICS: Final = "representative_macro_semantics"
CAPABILITY_FIXED_STEP_STRIDE_AUTHORITY: Final = "fixed_step_stride_authority"
CAPABILITY_COHORT_RELEASE: Final = "cohort_release"

DEFAULT_REQUIRED_CAPABILITIES: Final = (
    CAPABILITY_SCENARIO_LINEAGE,
    CAPABILITY_INTERVENTION_REPRESENTATION,
    CAPABILITY_BOUNDED_NEGATIVE_EVIDENCE,
    CAPABILITY_COVERAGE_STRATA,
    CAPABILITY_ATOMIC_PHYSICS_ARTIFACT,
    CAPABILITY_CAUSAL_ENTITIES,
    CAPABILITY_RAW_CONTACTS,
    CAPABILITY_DERIVED_SUPPORT,
    CAPABILITY_KINEMATICS,
    CAPABILITY_MACRO_EVENTS,
    CAPABILITY_CANONICAL_OBSERVATIONS,
    CAPABILITY_AGENT_OBSERVATIONS,
    CAPABILITY_INITIAL_STATE_IDENTITY,
    CAPABILITY_RELATIONAL_SUPERVISION,
    CAPABILITY_DETERMINISTIC_REPLAY,
    CAPABILITY_INSTANCE_HELD_OUT_PARTITION,
    CAPABILITY_TEMPLATE_HELD_OUT_PARTITION,
    CAPABILITY_PHYSICAL_REGIME_GATE,
    CAPABILITY_MATERIAL_DAMAGE_MAPPING,
    CAPABILITY_REPRESENTATIVE_MACRO_SEMANTICS,
    CAPABILITY_FIXED_STEP_STRIDE_AUTHORITY,
    CAPABILITY_COHORT_RELEASE,
)

KNOWN_UNSUPPORTED_CAPABILITIES: Final = MappingProxyType({
    CAPABILITY_AGENT_OBSERVATIONS: "current validators do not establish access-separated agent observations",
    CAPABILITY_CANONICAL_OBSERVATIONS: "current validators do not establish access-separated canonical observations",
    CAPABILITY_FIXED_STEP_STRIDE_AUTHORITY: "no accepted fixed-step-stride authority is represented by physics_capture_v1",
    CAPABILITY_PHYSICAL_REGIME_GATE: "no accepted versioned engine-derived physical-regime gate exists",
    CAPABILITY_MATERIAL_DAMAGE_MAPPING: (
        "material is unavailable because physics_capture_v1 exports no material field; "
        "the engine-verified damage lifecycle mapping cannot satisfy the combined material/damage capability"
    ),
    CAPABILITY_REPRESENTATIVE_MACRO_SEMANTICS: (
        "one or more target macro predicates is pending or lacks complete authorized "
        "representative evidence"
    ),
    CAPABILITY_COHORT_RELEASE: "representative pilot evidence is not an immutable cohort release",
})

KNOWN_UNAVAILABLE_LABELS: Final = MappingProxyType({
    "illegal_contact": "the legal-contact ontology is unavailable",
    "material": "no accepted engine-verified material mapping exists",
    "physical_regime_gate": "no accepted versioned engine-derived physical-regime gate exists",
    "micro_relation_usefulness": "frozen-model held-out evidence is unavailable",
})

_COLLECTION_REPORT_FIELDS: Final = frozenset((
    "schema",
    "plan_identity",
    "plan_version",
    "attempt_ledger",
    "accepted_count",
    "rejected_count",
    "failed_count",
    "quarantined_attempts",
    "quarantined_count",
    "planned_slots",
    "realized_coverage_stratum_counts",
    "unmet_slots",
    "realized_coverage_shortfalls",
))
_VERSION_ENVELOPE_FIELDS: Final = frozenset((
    "player_sha256",
    "protocol_sha256",
    "archive_sha256",
    "generator_version",
))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_string_mapping(value: Any, name: str) -> dict[str, str]:
    mapping = _require_mapping(value, name)
    normalized = {
        _require_nonempty_string(key, f"{name} key"): _require_nonempty_string(item, f"{name}[{key}]")
        for key, item in mapping.items()
    }
    return dict(sorted(normalized.items()))


def _validated_version_envelope(value: Any, name: str = "version_envelope") -> dict[str, str]:
    envelope = _require_string_mapping(value, name)
    if set(envelope) != _VERSION_ENVELOPE_FIELDS:
        raise ValueError(f"{name} must contain exactly {', '.join(sorted(_VERSION_ENVELOPE_FIELDS))}")
    for field in ("player_sha256", "protocol_sha256", "archive_sha256"):
        digest = envelope[field]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name}.{field} must be a lowercase SHA-256 digest")
    return envelope


def _require_json_value(value: Any, name: str) -> None:
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from error


def _identity(payload: Mapping[str, Any]) -> str:
    identity_payload = dict(payload)
    identity_payload.pop("identity", None)
    return f"{PILOT_REPORT_IDENTITY_NAMESPACE}:sha256:{sha256(_canonical_json(identity_payload)).hexdigest()}"


def _require_digest(value: Any, name: str) -> str:
    digest = _require_nonempty_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validated_macro_semantics(
    value: Any,
    attempts: Mapping[str, Any],
) -> dict[str, Any]:
    section = _require_mapping(value, "Pilot report macro_semantics")
    required = {"schema", "derivation_spec_version", "derivation_spec_digest", "predicates"}
    if set(section) != required:
        raise ValueError("Pilot report macro_semantics is incomplete or contains unknown fields")
    if section["schema"] != MACRO_SEMANTICS_SCHEMA:
        raise ValueError("Unsupported pilot report macro_semantics schema")
    if section["derivation_spec_version"] != DERIVATION_SPEC_VERSION:
        raise ValueError("Pilot report macro_semantics derivation version is stale")
    if section["derivation_spec_digest"] != derivation_spec_digest():
        raise ValueError("Pilot report macro_semantics derivation digest is stale")

    pilot_evidence = _require_mapping(
        attempts.get("pilot_evidence"), "Pilot report attempts.pilot_evidence"
    )
    accepted_attempt_ids = pilot_evidence.get("accepted_attempt_ids")
    if not isinstance(accepted_attempt_ids, list):
        raise ValueError("Pilot report accepted_attempt_ids must be a list")
    expected_attempt_ids = [
        _require_nonempty_string(item, "Pilot report accepted attempt_id")
        for item in accepted_attempt_ids
    ]
    if len(expected_attempt_ids) != len(set(expected_attempt_ids)):
        raise ValueError("Pilot report accepted_attempt_ids must be unique")
    if pilot_evidence.get("accepted_count") != len(expected_attempt_ids):
        raise ValueError("Pilot report accepted_count does not match accepted_attempt_ids")
    atomic = attempts.get("atomic_validation")
    if not isinstance(atomic, list):
        raise ValueError("Pilot report attempts.atomic_validation must be a list")
    atomic_by_attempt: dict[str, Mapping[str, Any]] = {}
    for raw_item in atomic:
        item = _require_mapping(raw_item, "Pilot report atomic validation record")
        if item.get("accepted") is not True:
            continue
        attempt_id = _require_nonempty_string(
            item.get("attempt_id"), "Pilot report accepted attempt_id"
        )
        if attempt_id in atomic_by_attempt:
            raise ValueError("Pilot report atomic validation accepted attempt IDs must be unique")
        atomic_by_attempt[attempt_id] = item
    if set(expected_attempt_ids) != set(atomic_by_attempt):
        raise ValueError("Pilot report accepted_attempt_ids do not match atomic pilot acceptance")

    predicates = _require_mapping(section["predicates"], "Pilot report macro_semantics predicates")
    if set(predicates) != set(TARGET_MACRO_PREDICATES):
        raise ValueError("Pilot report macro_semantics must contain exactly the target predicates")
    spec_predicates = _require_mapping(
        derivation_spec_json()["pending_predicates"],
        "Canonical pending macro predicate specifications",
    )
    evidence_fields = {
        "attempt_id",
        "capture_id",
        "shot_id",
        "physics_state_sha256",
        "physics_events_sha256",
        "derivation_spec_version",
        "derivation_spec_digest",
        "macro_label_artifact_sha256",
        "value_summary",
        "availability_summary",
    }
    predicate_fields = {
        "status",
        "definition",
        "prerequisites",
        "unavailable_cases",
        "failure_cases",
        "pending_reason",
        "evidence",
    }
    for name in TARGET_MACRO_PREDICATES:
        predicate = _require_mapping(predicates[name], f"Pilot report macro_semantics predicate {name}")
        if set(predicate) != predicate_fields:
            raise ValueError(f"Pilot report macro_semantics predicate {name} has invalid fields")
        if predicate["status"] != SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value:
            raise ValueError(f"Pilot report macro_semantics predicate {name} must remain pending")
        canonical = _require_mapping(spec_predicates[name], f"Canonical macro predicate {name}")
        for field in ("definition", "prerequisites", "unavailable_cases", "failure_cases"):
            if predicate[field] != canonical[field]:
                raise ValueError(f"Pilot report macro_semantics predicate {name} {field} is stale")
        if predicate["pending_reason"] != MACRO_SEMANTICS_PENDING_REASON:
            raise ValueError(f"Pilot report macro_semantics predicate {name} pending reason is invalid")
        evidence = predicate["evidence"]
        if not isinstance(evidence, list):
            raise ValueError(f"Pilot report macro_semantics predicate {name} evidence must be a list")
        observed_attempt_ids: list[str] = []
        for index, raw_row in enumerate(evidence):
            row = _require_mapping(raw_row, f"Pilot report macro_semantics {name} evidence[{index}]")
            if set(row) != evidence_fields:
                raise ValueError(f"Pilot report macro_semantics {name} evidence has invalid fields")
            attempt_id = _require_nonempty_string(
                row["attempt_id"], f"Pilot report macro_semantics {name} attempt_id"
            )
            observed_attempt_ids.append(attempt_id)
            _require_nonempty_string(row["capture_id"], f"Pilot report macro_semantics {name} capture_id")
            _require_nonempty_string(row["shot_id"], f"Pilot report macro_semantics {name} shot_id")
            for field in (
                "physics_state_sha256",
                "physics_events_sha256",
                "derivation_spec_digest",
                "macro_label_artifact_sha256",
            ):
                _require_digest(row[field], f"Pilot report macro_semantics {name} {field}")
            if row["derivation_spec_version"] != DERIVATION_SPEC_VERSION:
                raise ValueError(f"Pilot report macro_semantics {name} evidence derivation version is stale")
            if row["derivation_spec_digest"] != derivation_spec_digest():
                raise ValueError(f"Pilot report macro_semantics {name} evidence derivation digest is stale")
            artifact = atomic_by_attempt.get(attempt_id)
            if artifact is None:
                raise ValueError(
                    f"Pilot report macro_semantics {name} evidence lacks atomic validation"
                )
            stored_evidence = _require_mapping(
                artifact.get("macro_semantics_evidence"),
                f"Pilot report atomic validation {attempt_id} macro_semantics_evidence",
            )
            stored_row = _require_mapping(
                stored_evidence.get(name),
                f"Pilot report atomic validation {attempt_id} {name} evidence",
            )
            if row != stored_row:
                raise ValueError(
                    f"Pilot report macro_semantics {name} evidence differs from atomic validation"
                )
            deterministic = _require_mapping(
                artifact.get("deterministic_artifact_semantics"),
                f"Pilot report atomic validation {attempt_id} deterministic_artifact_semantics",
            )
            if (
                row["physics_state_sha256"] != deterministic.get("state_sha256")
                or row["physics_events_sha256"] != deterministic.get("event_sha256")
            ):
                raise ValueError(
                    f"Pilot report macro_semantics {name} source digests differ from atomic validation"
                )
            if row["capture_id"] != artifact.get("capture_id") or row["shot_id"] != artifact.get("shot_id"):
                raise ValueError(
                    f"Pilot report macro_semantics {name} capture/shot differs from atomic validation"
                )
            value_summary = _require_mapping(row["value_summary"], "Macro value_summary")
            availability_summary = _require_mapping(
                row["availability_summary"], "Macro availability_summary"
            )
            if set(value_summary) != {"true", "false", "null"}:
                raise ValueError("Macro value_summary must contain true, false, and null")
            if set(availability_summary) != {availability.value for availability in Availability}:
                raise ValueError("Macro availability_summary must contain every availability value")
            counts = (*value_summary.values(), *availability_summary.values())
            if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
                raise ValueError("Macro summaries must contain nonnegative integer counts")
            if sum(value_summary.values()) <= 0 or sum(value_summary.values()) != sum(availability_summary.values()):
                raise ValueError("Macro summaries must describe the same nonempty frame-record inventory")
        if len(observed_attempt_ids) != len(set(observed_attempt_ids)):
            raise ValueError(f"Pilot report macro_semantics predicate {name} has duplicate evidence")
        if set(observed_attempt_ids) != set(expected_attempt_ids):
            raise ValueError(
                f"Pilot report macro_semantics predicate {name} does not bind every pilot-accepted artifact"
            )
    return _thaw(section)


def _material_source_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    source_records: list[dict[str, str]] = []
    for raw_item in records:
        item = _require_mapping(raw_item, "Pilot report atomic validation record")
        if item.get("accepted") is not True:
            continue
        attempt_id = _require_nonempty_string(item.get("attempt_id"), "Pilot report material attempt_id")
        capture_id = _require_nonempty_string(item.get("capture_id"), "Pilot report material capture_id")
        shot_id = _require_nonempty_string(item.get("shot_id"), "Pilot report material shot_id")
        deterministic = _require_mapping(
            item.get("deterministic_artifact_semantics"),
            f"Pilot report atomic validation {attempt_id} deterministic_artifact_semantics",
        )
        source_records.append({
            "attempt_id": attempt_id,
            "capture_id": capture_id,
            "shot_id": shot_id,
            "state_sha256": _require_digest(
                deterministic.get("state_sha256"),
                f"Pilot report material {attempt_id} state_sha256",
            ),
            "events_sha256": _require_digest(
                deterministic.get("event_sha256"),
                f"Pilot report material {attempt_id} event_sha256",
            ),
        })
    source_records.sort(key=lambda item: item["attempt_id"])
    if len({item["attempt_id"] for item in source_records}) != len(source_records):
        raise ValueError("Pilot report material accepted attempt IDs must be unique")
    return source_records


def _material_damage_source_cohort_identity(
    plan_identity: str,
    version_envelope: Mapping[str, str],
    source_records: Sequence[Mapping[str, str]],
) -> str:
    payload = {
        "plan_identity": plan_identity,
        "report_version": PILOT_REPORT_VERSION,
        "source_records": [dict(record) for record in source_records],
        "version_envelope": dict(version_envelope),
    }
    return (
        f"{MATERIAL_DAMAGE_COHORT_NAMESPACE}:sha256:"
        f"{sha256(_canonical_json(payload)).hexdigest()}"
    )


def _material_damage_evidence_row(
    attempt_id: str,
    derived: Any,
    source_cohort_identity: str,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "capture_id": derived.capture_id,
        "shot_id": derived.shot_id,
        "physics_state_sha256": derived.state_sha256,
        "physics_events_sha256": derived.events_sha256,
        "derived_artifact_sha256": sha256(derived.to_bytes()).hexdigest(),
        "record_count": len(derived.records),
        "mapping_version": derived.mapping_version,
        "mapping_digest": derived.mapping_digest,
        "source_cohort_identity": source_cohort_identity,
    }


def _material_damage_semantics(
    plan_identity: str,
    version_envelope: Mapping[str, str],
    validation: list[dict[str, Any]],
) -> dict[str, Any]:
    source_records = _material_source_records(validation)
    source_cohort_identity = _material_damage_source_cohort_identity(
        plan_identity,
        version_envelope,
        source_records,
    )
    evidence: list[dict[str, Any]] = []
    by_attempt = {record["attempt_id"]: record for record in source_records}
    for item in validation:
        if not item.get("accepted"):
            continue
        attempt_id = _require_nonempty_string(item.get("attempt_id"), "Pilot material attempt_id")
        source = by_attempt[attempt_id]
        derived = derive_material_damage(
            Path(_require_nonempty_string(item.get("artifact_path"), "Pilot material artifact_path")),
            source_cohort_identity=source_cohort_identity,
        )
        if (
            derived.capture_id != source["capture_id"]
            or derived.shot_id != source["shot_id"]
            or derived.state_sha256 != source["state_sha256"]
            or derived.events_sha256 != source["events_sha256"]
        ):
            raise ValueError(f"material/damage source provenance differs for accepted attempt {attempt_id}")
        row = _material_damage_evidence_row(attempt_id, derived, source_cohort_identity)
        item["material_damage_evidence"] = row
        evidence.append(row)
    evidence.sort(key=lambda row: row["attempt_id"])
    return {
        "schema": MATERIAL_DAMAGE_SEMANTICS_SCHEMA,
        "source_cohort_identity": source_cohort_identity,
        "material": {
            "availability": MaterialDamageAvailability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD.value,
            "label": MATERIAL_UNAVAILABLE_LABEL,
            "reason": MATERIAL_MISSING_ENGINE_FIELD_REASON,
            "status": "unavailable",
        },
        "damage": {
            "availability": MaterialDamageAvailability.AVAILABLE.value,
            "mapping_schema_version": MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
            "mapping_version": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version,
            "mapping_digest": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest,
            "source_facts": list(MAPPING_SOURCE_FACTS),
            "status": SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.representative_validation_status.value,
            "evidence": evidence,
        },
    }


def _validated_material_damage_semantics(
    value: Any,
    attempts: Mapping[str, Any],
    plan_identity: str,
    version_envelope: Mapping[str, str],
) -> dict[str, Any]:
    section = _require_mapping(value, "Pilot report material_damage_semantics")
    if set(section) != {"schema", "source_cohort_identity", "material", "damage"}:
        raise ValueError("Pilot report material_damage_semantics is incomplete or contains unknown fields")
    if section["schema"] != MATERIAL_DAMAGE_SEMANTICS_SCHEMA:
        raise ValueError("Unsupported pilot report material_damage_semantics schema")
    source_cohort_identity = _require_nonempty_string(
        section["source_cohort_identity"],
        "Pilot report material source_cohort_identity",
    )
    atomic = attempts.get("atomic_validation")
    if not isinstance(atomic, list):
        raise ValueError("Pilot report attempts.atomic_validation must be a list")
    source_records = _material_source_records(atomic)
    expected_cohort = _material_damage_source_cohort_identity(
        plan_identity,
        version_envelope,
        source_records,
    )
    if source_cohort_identity != expected_cohort:
        raise ValueError("Pilot report material source-cohort identity is stale")

    material = _require_mapping(section["material"], "Pilot report material semantics material")
    if set(material) != {"availability", "label", "reason", "status"}:
        raise ValueError("Pilot report material semantics material fields are invalid")
    if material != {
        "availability": MaterialDamageAvailability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD.value,
        "label": MATERIAL_UNAVAILABLE_LABEL,
        "reason": MATERIAL_MISSING_ENGINE_FIELD_REASON,
        "status": "unavailable",
    }:
        raise ValueError("Pilot report material semantics cannot promote unavailable material")

    damage = _require_mapping(section["damage"], "Pilot report material semantics damage")
    damage_fields = {
        "availability",
        "mapping_schema_version",
        "mapping_version",
        "mapping_digest",
        "source_facts",
        "status",
        "evidence",
    }
    if set(damage) != damage_fields:
        raise ValueError("Pilot report material semantics damage fields are invalid")
    if damage["availability"] != MaterialDamageAvailability.AVAILABLE.value:
        raise ValueError("Pilot report damage mapping must remain available")
    if damage["mapping_schema_version"] != MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION:
        raise ValueError("Pilot report damage mapping schema version is stale")
    if damage["mapping_version"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version:
        raise ValueError("Pilot report damage mapping version is stale")
    if damage["mapping_digest"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest:
        raise ValueError("Pilot report damage mapping digest is stale")
    if damage["source_facts"] != list(MAPPING_SOURCE_FACTS):
        raise ValueError("Pilot report damage mapping source facts are stale")
    if damage["status"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.representative_validation_status.value:
        raise ValueError("Pilot report damage mapping status is invalid")
    evidence = damage["evidence"]
    if not isinstance(evidence, list):
        raise ValueError("Pilot report material damage evidence must be a list")
    evidence_fields = {
        "attempt_id",
        "capture_id",
        "shot_id",
        "physics_state_sha256",
        "physics_events_sha256",
        "derived_artifact_sha256",
        "record_count",
        "mapping_version",
        "mapping_digest",
        "source_cohort_identity",
    }
    atomic_by_attempt = {
        record["attempt_id"]: record
        for record in atomic
        if isinstance(record, Mapping) and record.get("accepted") is True
    }
    expected_attempts = set(atomic_by_attempt)
    observed_attempts: set[str] = set()
    for index, raw_row in enumerate(evidence):
        row = _require_mapping(raw_row, f"Pilot report material damage evidence[{index}]")
        if set(row) != evidence_fields:
            raise ValueError("Pilot report material damage evidence fields are invalid")
        attempt_id = _require_nonempty_string(row["attempt_id"], "Pilot material damage attempt_id")
        if attempt_id in observed_attempts:
            raise ValueError("Pilot report material damage evidence has duplicate attempts")
        observed_attempts.add(attempt_id)
        atomic_record = atomic_by_attempt.get(attempt_id)
        if atomic_record is None:
            raise ValueError("Pilot report material damage evidence lacks atomic validation")
        for field in ("capture_id", "shot_id", "source_cohort_identity"):
            _require_nonempty_string(row[field], f"Pilot material damage {field}")
        for field in (
            "physics_state_sha256",
            "physics_events_sha256",
            "derived_artifact_sha256",
            "mapping_digest",
        ):
            _require_digest(row[field], f"Pilot material damage {field}")
        if isinstance(row["record_count"], bool) or not isinstance(row["record_count"], int) or row["record_count"] < 0:
            raise ValueError("Pilot material damage record_count must be nonnegative")
        if row["mapping_version"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.mapping_version:
            raise ValueError("Pilot material damage mapping version is stale")
        if row["mapping_digest"] != SUPPORTED_DAMAGE_LIFECYCLE_MAPPING.digest:
            raise ValueError("Pilot material damage mapping digest is stale")
        if row["source_cohort_identity"] != source_cohort_identity:
            raise ValueError("Pilot material damage evidence source cohort differs")
        stored = _require_mapping(
            atomic_record.get("material_damage_evidence"),
            f"Pilot report atomic validation {attempt_id} material_damage_evidence",
        )
        if dict(stored) != dict(row):
            raise ValueError("Pilot material damage evidence differs from atomic validation")
        deterministic = _require_mapping(
            atomic_record.get("deterministic_artifact_semantics"),
            f"Pilot report atomic validation {attempt_id} deterministic_artifact_semantics",
        )
        if (
            row["capture_id"] != atomic_record.get("capture_id")
            or row["shot_id"] != atomic_record.get("shot_id")
            or row["physics_state_sha256"] != deterministic.get("state_sha256")
            or row["physics_events_sha256"] != deterministic.get("event_sha256")
        ):
            raise ValueError("Pilot material damage evidence source differs from atomic validation")
    if observed_attempts != expected_attempts:
        raise ValueError("Pilot material damage evidence does not bind every accepted atomic artifact")
    return _thaw(section)


# v3 receipt-bound implementation.  The preceding macro seam remains above; these
# names are the only material/damage seam used by report construction and parsing.
def _pilot_damage_context(plan_identity: str, version_envelope: Mapping[str, str]) -> dict[str, str]:
    return {
        "plan_identity": plan_identity,
        "report_version": str(PILOT_REPORT_VERSION),
        "version_envelope": json.dumps(dict(version_envelope), sort_keys=True, separators=(",", ":")),
    }


def _accepted_atomic_records(attempts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    atomic = attempts.get("atomic_validation")
    if not isinstance(atomic, list):
        raise ValueError("Pilot report attempts.atomic_validation must be a list")
    records = [
        _require_mapping(item, "Pilot report atomic validation record")
        for item in atomic
        if isinstance(item, Mapping) and item.get("accepted") is True
    ]
    if len({item.get("attempt_id") for item in records}) != len(records):
        raise ValueError("Pilot report accepted atomic attempt IDs must be unique")
    return records


def _receipt_damage_row(attempt_id: str, artifact: Any) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "capture_id": artifact.capture_id,
        "shot_id": artifact.shot_id,
        "physics_state_sha256": artifact.state_sha256,
        "physics_events_sha256": artifact.events_sha256,
        "derived_artifact_sha256": sha256(artifact.to_bytes()).hexdigest(),
        "record_count": len(artifact.records),
        "mapping_version": artifact.mapping_version,
        "mapping_digest": artifact.mapping_digest,
        "source_cohort_identity": artifact.source_cohort_identity,
        "receipt_status": artifact.validation_status.value if artifact.validation_status else None,
        "receipt_cohort_context": dict(artifact.cohort_context),
        "receipt_source_records": [record.to_json() for record in artifact.source_records],
    }


def _material_damage_semantics(
    plan_identity: str,
    version_envelope: Mapping[str, str],
    validation: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [item for item in validation if item.get("accepted")]
    sources = tuple(
        Path(_require_nonempty_string(item.get("artifact_path"), "Pilot material artifact_path"))
        for item in accepted
    )
    receipt = build_damage_lifecycle_validation_receipt(
        sources,
        cohort_context=_pilot_damage_context(plan_identity, version_envelope),
    )
    evidence: list[dict[str, Any]] = []
    for item, source in zip(accepted, sources):
        attempt_id = _require_nonempty_string(item.get("attempt_id"), "Pilot material attempt_id")
        artifact = derive_material_damage(source, receipt=receipt)
        deterministic = _require_mapping(
            item.get("deterministic_artifact_semantics"),
            f"Pilot report atomic validation {attempt_id} deterministic_artifact_semantics",
        )
        if (
            item.get("capture_id") != artifact.capture_id
            or item.get("shot_id") != artifact.shot_id
            or deterministic.get("state_sha256") != artifact.state_sha256
            or deterministic.get("event_sha256") != artifact.events_sha256
        ):
            raise ValueError("Pilot material damage source differs from atomic validation")
        row = _receipt_damage_row(attempt_id, artifact)
        item["material_damage_evidence"] = row
        evidence.append(row)
    evidence.sort(key=lambda row: row["attempt_id"])
    return {
        "schema": MATERIAL_DAMAGE_SEMANTICS_SCHEMA,
        "source_cohort_identity": receipt.source_cohort_identity,
        "material": {
            "availability": MaterialDamageAvailability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD.value,
            "label": MATERIAL_UNAVAILABLE_LABEL,
            "reason": MATERIAL_MISSING_ENGINE_FIELD_REASON,
            "status": "unavailable",
        },
        "damage": {
            "availability": (
                MaterialDamageAvailability.AVAILABLE.value
                if receipt.status is SemanticStatus.ENGINE_VERIFIED
                else MaterialDamageAvailability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE.value
            ),
            "mapping_schema_version": MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION,
            "mapping_version": receipt.mapping_version,
            "mapping_digest": receipt.mapping_digest,
            "source_facts": list(MAPPING_SOURCE_FACTS),
            "status": receipt.status.value,
            "source_cohort_identity": receipt.source_cohort_identity,
            "cohort_context": dict(receipt.cohort_context),
            "source_records": [record.to_json() for record in receipt.source_records],
            "evidence": evidence,
        },
    }


def _validated_material_damage_semantics(
    value: Any,
    attempts: Mapping[str, Any],
    plan_identity: str,
    version_envelope: Mapping[str, str],
) -> dict[str, Any]:
    section = _require_mapping(value, "Pilot report material_damage_semantics")
    if set(section) != {"schema", "source_cohort_identity", "material", "damage"}:
        raise ValueError("Pilot report material_damage_semantics is incomplete or contains unknown fields")
    if section["schema"] != MATERIAL_DAMAGE_SEMANTICS_SCHEMA:
        raise ValueError("Unsupported pilot report material_damage_semantics schema")
    accepted = _accepted_atomic_records(attempts)
    material = _require_mapping(section["material"], "Pilot report material semantics material")
    expected_material = {
        "availability": MaterialDamageAvailability.UNAVAILABLE_MISSING_ENGINE_MATERIAL_FIELD.value,
        "label": MATERIAL_UNAVAILABLE_LABEL,
        "reason": MATERIAL_MISSING_ENGINE_FIELD_REASON,
        "status": "unavailable",
    }
    if dict(material) != expected_material:
        raise ValueError("Pilot report material semantics cannot promote unavailable material")
    damage = _require_mapping(section["damage"], "Pilot report material semantics damage")
    expected_fields = {
        "availability", "mapping_schema_version", "mapping_version", "mapping_digest",
        "source_facts", "status", "source_cohort_identity", "cohort_context", "source_records", "evidence",
    }
    if set(damage) != expected_fields:
        raise ValueError("Pilot report material semantics damage fields are invalid")
    damage_source_records = damage.get("source_records")
    descriptor_input = {
        "mapping_version": damage.get("mapping_version"),
        "mapping_digest": damage.get("mapping_digest"),
        "source_cohort_identity": damage.get("source_cohort_identity"),
        "cohort_context": damage.get("cohort_context"),
        "source_records": damage_source_records,
        "status": damage.get("status"),
    }
    try:
        descriptor = validate_damage_lifecycle_receipt_descriptor(descriptor_input)
    except (TypeError, MaterialDamageContractError) as error:
        raise ValueError("Pilot report damage validation receipt is stale or altered") from error
    if section["source_cohort_identity"] != descriptor["source_cohort_identity"]:
        raise ValueError("Pilot report material source-cohort identity is stale")
    expected_availability = (
        MaterialDamageAvailability.AVAILABLE.value
        if descriptor["status"] == SemanticStatus.ENGINE_VERIFIED.value
        else MaterialDamageAvailability.UNAVAILABLE_INSUFFICIENT_DAMAGE_LIFECYCLE_EVIDENCE.value
    )
    if (
        damage["availability"] != expected_availability
        or damage["mapping_schema_version"] != MATERIAL_DAMAGE_MAPPING_SCHEMA_VERSION
        or damage["mapping_version"] != descriptor["mapping_version"]
        or damage["mapping_digest"] != descriptor["mapping_digest"]
        or damage["source_facts"] != list(MAPPING_SOURCE_FACTS)
        or damage["status"] != descriptor["status"]
        or damage["source_cohort_identity"] != descriptor["source_cohort_identity"]
        or damage["cohort_context"] != descriptor["cohort_context"]
        or damage["source_records"] != descriptor["source_records"]
    ):
        raise ValueError("Pilot report damage validation receipt is stale or altered")
    evidence = damage["evidence"]
    if not isinstance(evidence, list):
        raise ValueError("Pilot report material damage evidence must be a list")
    expected_rows: dict[str, dict[str, Any]] = {}
    expected_source_records = descriptor["source_records"]
    for item in accepted:
        attempt_id = _require_nonempty_string(item.get("attempt_id"), "Pilot material attempt_id")
        stored = _require_mapping(
            item.get("material_damage_evidence"),
            f"Pilot report atomic validation {attempt_id} material_damage_evidence",
        )
        deterministic = _require_mapping(
            item.get("deterministic_artifact_semantics"),
            f"Pilot report atomic validation {attempt_id} deterministic_artifact_semantics",
        )
        if stored.get("capture_id") != item.get("capture_id") or stored.get("shot_id") != item.get("shot_id"):
            raise ValueError("Pilot material damage atomic evidence identity differs")
        if stored.get("physics_state_sha256") != deterministic.get("state_sha256") or stored.get("physics_events_sha256") != deterministic.get("event_sha256"):
            raise ValueError("Pilot material damage atomic evidence source differs")
        if stored.get("receipt_source_records") != expected_source_records:
            raise ValueError("Pilot material damage atomic receipt source records differ")
        if stored.get("receipt_cohort_context") != descriptor["cohort_context"] or stored.get("receipt_status") != descriptor["status"]:
            raise ValueError("Pilot material damage atomic receipt descriptor differs")
    if len(evidence) != len(accepted) or {row.get("attempt_id") for row in evidence} != {item.get("attempt_id") for item in accepted}:
        raise ValueError("Pilot material damage evidence does not bind every accepted artifact")
    atomic_evidence = {
        item["attempt_id"]: _require_mapping(item.get("material_damage_evidence"), "atomic material damage evidence")
        for item in accepted
    }
    for index, row in enumerate(evidence):
        item = _require_mapping(row, f"Pilot material damage evidence[{index}]")
        attempt_id = _require_nonempty_string(item.get("attempt_id"), "Pilot material damage attempt_id")
        if dict(item) != dict(atomic_evidence[attempt_id]):
            raise ValueError("Pilot material damage evidence differs from atomic validation")
    return _thaw(section)


@dataclass(frozen=True, slots=True)
class ReplayInput:
    """One declared replay and its independently persisted artifact evidence."""

    manifest: ScenarioManifest
    xml_content: bytes
    reference: str
    version_envelope: Mapping[str, str] = field(default_factory=dict)
    scenario_id: str = ""
    intervention_identity: str = ""
    artifact_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PilotPartitionAudit:
    """One supplied partition plus its complete admitted inventories."""

    manifest: CohortPartitionManifest
    admitted_scenario_lineage_identities: tuple[str, ...]
    admitted_provenance_records: tuple[Mapping[str, Any], ...]
    reference: str


@dataclass(frozen=True, slots=True)
class PilotReport:
    """Immutable, identity-bound report for a representative pilot."""

    schema: Literal["representative_pilot_report_v3"]
    report_version: int
    identity: str
    version_envelope: Mapping[str, str]
    plan_identity: str
    plan_version: int
    scenarios: tuple[Mapping[str, Any], ...]
    attempts: Mapping[str, Any]
    coverage: Mapping[str, Any]
    replays: tuple[Mapping[str, Any], ...]
    initial_state_identities: tuple[Mapping[str, Any], ...]
    partition_audits: tuple[Mapping[str, Any], ...]
    supervision: tuple[Mapping[str, Any], ...]
    macro_semantics: Mapping[str, Any]
    material_damage_semantics: Mapping[str, Any]
    available_capabilities: tuple[str, ...]
    unavailable_capabilities: tuple[Mapping[str, str], ...]
    unavailable_labels: Mapping[str, str]
    permanent_or_systematic_exporter_defects: tuple[Mapping[str, Any], ...]
    pilot_status: Literal["accepted", "rejected"]
    acceptance_decision: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "report_version": self.report_version,
            "identity": self.identity,
            "version_envelope": _thaw(self.version_envelope),
            "plan_identity": self.plan_identity,
            "plan_version": self.plan_version,
            "scenarios": _thaw(self.scenarios),
            "attempts": _thaw(self.attempts),
            "coverage": _thaw(self.coverage),
            "replays": _thaw(self.replays),
            "initial_state_identities": _thaw(self.initial_state_identities),
            "partition_audits": _thaw(self.partition_audits),
            "supervision": _thaw(self.supervision),
            "macro_semantics": _thaw(self.macro_semantics),
            "material_damage_semantics": _thaw(self.material_damage_semantics),
            "available_capabilities": list(self.available_capabilities),
            "unavailable_capabilities": _thaw(self.unavailable_capabilities),
            "unavailable_labels": _thaw(self.unavailable_labels),
            "permanent_or_systematic_exporter_defects": _thaw(self.permanent_or_systematic_exporter_defects),
            "pilot_status": self.pilot_status,
            "acceptance_decision": _thaw(self.acceptance_decision),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PilotReport":
        required = {
            "schema",
            "report_version",
            "identity",
            "version_envelope",
            "plan_identity",
            "plan_version",
            "scenarios",
            "attempts",
            "coverage",
            "replays",
            "initial_state_identities",
            "partition_audits",
            "supervision",
            "macro_semantics",
            "material_damage_semantics",
            "available_capabilities",
            "unavailable_capabilities",
            "unavailable_labels",
            "permanent_or_systematic_exporter_defects",
            "pilot_status",
            "acceptance_decision",
        }
        raw = _require_mapping(data, "Pilot report")
        if set(raw) != required:
            raise ValueError("Pilot report is incomplete or contains unknown fields")
        if raw["schema"] != PILOT_REPORT_SCHEMA:
            raise ValueError(f"Unsupported pilot report schema: {raw['schema']}")
        if raw["report_version"] != PILOT_REPORT_VERSION:
            raise ValueError(f"Unsupported pilot report version: {raw['report_version']}")
        identity = _require_nonempty_string(raw["identity"], "Pilot report identity")
        plan_identity = _require_nonempty_string(raw["plan_identity"], "Pilot report plan_identity")
        plan_version = raw["plan_version"]
        if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version <= 0:
            raise ValueError("Pilot report plan_version must be a positive integer")
        version_envelope = _validated_version_envelope(raw["version_envelope"], "Pilot report version_envelope")
        list_fields = (
            "scenarios",
            "replays",
            "initial_state_identities",
            "partition_audits",
            "supervision",
            "available_capabilities",
            "unavailable_capabilities",
            "permanent_or_systematic_exporter_defects",
        )
        if any(not isinstance(raw[field], list) for field in list_fields):
            raise ValueError("Pilot report list field is malformed")
        if not all(isinstance(capability, str) and capability for capability in raw["available_capabilities"]):
            raise ValueError("Pilot report available_capabilities must contain nonempty strings")
        if len(raw["available_capabilities"]) != len(set(raw["available_capabilities"])):
            raise ValueError("Pilot report available_capabilities must be unique")
        for field in ("attempts", "coverage", "acceptance_decision"):
            _require_mapping(raw[field], f"Pilot report {field}")
        attempts = _require_mapping(raw["attempts"], "Pilot report attempts")
        macro_semantics = _validated_macro_semantics(raw["macro_semantics"], attempts)
        material_damage_semantics = _validated_material_damage_semantics(
            raw["material_damage_semantics"],
            attempts,
            plan_identity,
            version_envelope,
        )
        unavailable_capabilities: list[dict[str, str]] = []
        for index, item in enumerate(raw["unavailable_capabilities"]):
            unavailable = _require_mapping(item, f"Pilot report unavailable_capabilities[{index}]")
            if set(unavailable) != {"capability", "reason"}:
                raise ValueError("Pilot report unavailable_capabilities record has invalid fields")
            unavailable_capabilities.append({
                "capability": _require_nonempty_string(
                    unavailable["capability"], "Pilot report unavailable capability"
                ),
                "reason": _require_nonempty_string(
                    unavailable["reason"], "Pilot report unavailable capability reason"
                ),
            })
        unavailable_names = [item["capability"] for item in unavailable_capabilities]
        if len(unavailable_names) != len(set(unavailable_names)):
            raise ValueError("Pilot report unavailable_capabilities must be unique")
        if set(raw["available_capabilities"]) & set(unavailable_names):
            raise ValueError("Pilot report available and unavailable capabilities must be disjoint")
        if CAPABILITY_REPRESENTATIVE_MACRO_SEMANTICS in raw["available_capabilities"]:
            raise ValueError("Pilot report representative macro semantics cannot be available while pending")
        if CAPABILITY_REPRESENTATIVE_MACRO_SEMANTICS not in unavailable_names:
            raise ValueError("Pilot report representative macro semantics must be explicitly unavailable")
        unavailable_labels = _require_string_mapping(raw["unavailable_labels"], "Pilot report unavailable_labels")
        if unavailable_labels.get("material") != KNOWN_UNAVAILABLE_LABELS["material"]:
            raise ValueError("Pilot report material label must remain unavailable")
        damage_status = material_damage_semantics["damage"]["status"]
        if damage_status == SemanticStatus.ENGINE_VERIFIED.value:
            if "damage" in unavailable_labels:
                raise ValueError("verified damage mapping cannot remain globally unavailable")
        elif unavailable_labels.get("damage") != MATERIAL_DAMAGE_PENDING_REASON:
            raise ValueError("unverified damage mapping must be explicitly unavailable")
        if raw["pilot_status"] not in {"accepted", "rejected"}:
            raise ValueError("Pilot report pilot_status is invalid")
        _require_json_value(raw, "Pilot report")
        if identity != _identity(raw):
            raise ValueError("Pilot report identity is stale")
        return cls(
            PILOT_REPORT_SCHEMA,
            PILOT_REPORT_VERSION,
            identity,
            _freeze(version_envelope),
            plan_identity,
            plan_version,
            tuple(_freeze(item) for item in raw["scenarios"]),
            _freeze(raw["attempts"]),
            _freeze(raw["coverage"]),
            tuple(_freeze(item) for item in raw["replays"]),
            tuple(_freeze(item) for item in raw["initial_state_identities"]),
            tuple(_freeze(item) for item in raw["partition_audits"]),
            tuple(_freeze(item) for item in raw["supervision"]),
            _freeze(macro_semantics),
            _freeze(material_damage_semantics),
            tuple(raw["available_capabilities"]),
            tuple(_freeze(item) for item in unavailable_capabilities),
            _freeze(unavailable_labels),
            tuple(_freeze(item) for item in raw["permanent_or_systematic_exporter_defects"]),
            raw["pilot_status"],
            _freeze(raw["acceptance_decision"]),
        )


def write_pilot_report(report: PilotReport, path: Path) -> Path:
    """Atomically write a canonical representative-pilot report."""
    if not isinstance(report, PilotReport):
        raise ValueError("write_pilot_report requires a PilotReport")
    payload = PilotReport.from_dict(report.to_dict()).to_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_pilot_report(path: Path) -> PilotReport:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load pilot report {target}: {error}") from error
    return PilotReport.from_dict(_require_mapping(data, "Pilot report"))


def _normalize_required_capabilities(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("required_capabilities must be a sequence")
    capabilities = tuple(sorted(_require_nonempty_string(item, "required capability") for item in value))
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("required_capabilities must be unique")
    return capabilities


def _normalize_string_sequence(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    normalized = tuple(sorted(_require_nonempty_string(item, name) for item in value))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique")
    return normalized


def _validated_collection_report(loaded: LoadedCollectionPlan, report: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _require_mapping(report, "Collection report")
    if set(raw) != _COLLECTION_REPORT_FIELDS:
        raise ValueError("Collection report is incomplete or contains unknown fields")
    if raw["schema"] != "collection_plan_execution_report_v2":
        raise ValueError("Collection report has an unsupported schema")
    if raw["plan_identity"] != loaded.plan.identity or raw["plan_version"] != loaded.plan.plan_version:
        raise ValueError("Collection report does not match the loaded plan")
    for field in ("attempt_ledger", "quarantined_attempts", "planned_slots", "unmet_slots", "realized_coverage_shortfalls"):
        if not isinstance(raw[field], list):
            raise ValueError(f"Collection report {field} must be a list")
    for field in ("accepted_count", "rejected_count", "failed_count", "quarantined_count"):
        if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 0:
            raise ValueError(f"Collection report {field} must be a nonnegative integer")
    if not isinstance(raw["realized_coverage_stratum_counts"], Mapping):
        raise ValueError("Collection report realized_coverage_stratum_counts must be an object")
    ledger = raw["attempt_ledger"]
    if not all(isinstance(entry, Mapping) for entry in ledger):
        raise ValueError("Collection report attempt_ledger contains a malformed attempt")
    attempt_ids = [entry.get("attempt_id") for entry in ledger]
    if any(not isinstance(attempt_id, str) or not attempt_id for attempt_id in attempt_ids):
        raise ValueError("Collection report attempt IDs must be nonempty strings")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("Collection report attempt IDs must be unique")
    scenarios = {scenario.scenario_id: scenario for scenario in loaded.plan.scenarios}
    for entry in ledger:
        scenario = scenarios.get(entry.get("scenario_id"))
        if scenario is None:
            raise ValueError("Collection report names a scenario outside the loaded plan")
        interventions = {intervention.id: intervention for intervention in scenario.interventions}
        intervention = interventions.get(entry.get("intervention_id"))
        if (
            intervention is None
            or entry.get("plan_identity") != loaded.plan.identity
            or entry.get("scenario_identity") != scenario.identity
            or entry.get("intervention_identity") != intervention.identity
            or entry.get("intervention_ordinal") != intervention.ordinal
        ):
            raise ValueError("Collection report attempt does not match the loaded plan request")
        if entry.get("status") == "accepted" and (
            entry.get("artifact_disposition") != "accepted"
            or not isinstance(entry.get("artifact_path"), str)
            or not entry["artifact_path"]
        ):
            raise ValueError("Collection report accepted attempt has no accepted artifact")
    counts = {status: sum(entry.get("status") == status for entry in ledger) for status in ("accepted", "rejected", "failed")}
    if any(raw[f"{status}_count"] != counts[status] for status in counts):
        raise ValueError("Collection report attempt counts disagree with its ledger")
    if raw["quarantined_count"] != len(raw["quarantined_attempts"]):
        raise ValueError("Collection report quarantined count disagrees with its attempts")
    _require_json_value(raw, "Collection report")
    return raw


def _scenario_coverage(loaded: LoadedCollectionPlan) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario.scenario_id,
            "exposure_role": scenario.exposure_role,
            "benchmark_condition_identity": scenario.scenario_manifest_projection["benchmark_condition_identity"],
            "scenario_template_identity": scenario.scenario_manifest_projection["scenario_template_identity"],
            "level_instance_identity": scenario.scenario_manifest_projection["level_instance_identity"],
            "scenario_lineage_identity": scenario.scenario_manifest_projection["scenario_lineage_identity"],
            "expected_initial_engine_state_identity": scenario.expected_initial_engine_state_identity,
            "coverage_strata": _thaw(scenario.coverage_strata),
        }
        for scenario in loaded.plan.scenarios
    ]


def _artifact_path(
    entry: Mapping[str, Any],
    overrides: Mapping[str, Path] | None,
) -> Path:
    attempt_id = _require_nonempty_string(entry.get("attempt_id"), "Collection attempt_id")
    if overrides is not None and attempt_id in overrides:
        return Path(overrides[attempt_id])
    return Path(_require_nonempty_string(entry.get("artifact_path"), "Collection artifact_path"))


def _accepted_entries(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        entry for entry in report["attempt_ledger"]
        if entry.get("status") == "accepted" and entry.get("artifact_disposition") == "accepted"
    ]


def _request_binding(loaded: LoadedCollectionPlan, entry: Mapping[str, Any]) -> dict[str, Any]:
    scenario = next(
        scenario for scenario in loaded.plan.scenarios
        if scenario.scenario_id == entry["scenario_id"]
    )
    intervention = next(
        intervention for intervention in scenario.interventions
        if intervention.id == entry["intervention_id"]
    )
    binding: dict[str, Any] = {
        "plan_identity": loaded.plan.identity,
        "plan_version": loaded.plan.plan_version,
        "scenario_id": scenario.scenario_id,
        "scenario_identity": scenario.identity,
        "intervention_id": intervention.id,
        "intervention_identity": intervention.identity,
        "attempt_id": _require_nonempty_string(entry.get("attempt_id"), "Collection attempt_id"),
    }
    for field in ("attempt_number",):
        value = entry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Collection {field} must be a positive integer")
        binding[field] = value
    return binding


def _exclude_artifact(
    record: dict[str, Any],
    path: Path,
    attempt_id: str,
    quarantine_root: Path | None,
) -> None:
    record["pilot_disposition"] = "excluded"
    if quarantine_root is None:
        return
    destination_parent = quarantine_root / sha256(attempt_id.encode("utf-8")).hexdigest()
    destination = destination_parent / path.name
    destination_parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_path = path.resolve(strict=True)
        assessment_root = quarantine_root.parent.resolve(strict=True)
        resolved_path.relative_to(assessment_root)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("artifact is not a movable directory")
        if destination.exists():
            raise ValueError("pilot quarantine destination already exists")
        os.replace(path, destination)
    except (OSError, ValueError) as error:
        record["pilot_disposition"] = "quarantine_failed"
        record["quarantine_failure_reason"] = str(error) or error.__class__.__name__
    else:
        record["pilot_disposition"] = "quarantined"
        record["quarantine_path"] = str(destination)
    failure_manifest = destination_parent / "pilot_assessment_failure_manifest.json"
    manifest = {
        "schema": "pilot_assessment_failure_manifest_v1",
        "attempt_id": attempt_id,
        "reason": _require_nonempty_string(record.get("reason"), "Pilot exclusion reason"),
        "original_artifact_path": str(path),
        "pilot_disposition": record["pilot_disposition"],
    }
    content = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination_parent, delete=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, failure_manifest)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ValueError(f"Cannot write pilot assessment failure manifest {failure_manifest}") from error
    record["failure_manifest_path"] = str(failure_manifest)


def _macro_evidence_rows(labels: Any, attempt_id: str) -> dict[str, dict[str, Any]]:
    artifact_digest = sha256(labels.to_jsonl().encode("utf-8")).hexdigest()
    rows: dict[str, dict[str, Any]] = {}
    for name in TARGET_MACRO_PREDICATES:
        predicate = MacroPredicate(name)
        value_summary = {"true": 0, "false": 0, "null": 0}
        availability_summary = {availability.value: 0 for availability in Availability}
        for frame_record in labels.frames:
            label = frame_record.predicate(predicate)
            value_key = "null" if label.value is None else str(label.value).lower()
            value_summary[value_key] += 1
            availability_summary[label.availability.value] += 1
        rows[name] = {
            "attempt_id": attempt_id,
            "capture_id": labels.capture_id,
            "shot_id": labels.shot_id,
            "physics_state_sha256": labels.state_sha256,
            "physics_events_sha256": labels.events_sha256,
            "derivation_spec_version": DERIVATION_SPEC_VERSION,
            "derivation_spec_digest": derivation_spec_digest(),
            "macro_label_artifact_sha256": artifact_digest,
            "value_summary": value_summary,
            "availability_summary": availability_summary,
        }
    return rows


def _assessment_artifacts(
    loaded: LoadedCollectionPlan,
    collection_report: Mapping[str, Any],
    artifact_paths: Mapping[str, Path] | None,
    quarantine_root: Path | None,
    version_envelope: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_by_scenario = {
        scenario.scenario_id: scenario.expected_initial_engine_state_identity
        for scenario in loaded.plan.scenarios
    }
    validation: list[dict[str, Any]] = []
    initial_identities: list[dict[str, Any]] = []
    supervision: list[dict[str, Any]] = []
    defects: list[dict[str, Any]] = []
    for entry in _accepted_entries(collection_report):
        attempt_id = _require_nonempty_string(entry.get("attempt_id"), "Collection attempt_id")
        scenario_id = _require_nonempty_string(entry.get("scenario_id"), "Collection scenario_id")
        expected = expected_by_scenario.get(scenario_id)
        if expected is None:
            raise ValueError("Collection report names a scenario outside the loaded plan")
        path = _artifact_path(entry, artifact_paths)
        artifact_record: dict[str, Any] = {
            "attempt_id": attempt_id,
            "artifact_path": str(path),
            "scenario_id": scenario_id,
            "intervention_identity": entry.get("intervention_identity"),
            "accepted": False,
        }
        validation.append(artifact_record)
        try:
            summary = validate_physics_shot_artifact(path)
        except Exception as error:
            artifact_record["reason"] = str(error) or error.__class__.__name__
            initial_identities.append({
                "attempt_id": attempt_id,
                "planned": expected,
                "observed": None,
                "recorded_expected": None,
                "matched": False,
                "reason": "artifact validation failed",
            })
            supervision.append({
                "attempt_id": attempt_id,
                "attempted": False,
                "accepted": False,
                "reason": "artifact validation failed",
            })
            defects.append({
                "scope": "attempt",
                "code": "atomic_validation_failed",
                "attempt_id": attempt_id,
                "detail": artifact_record["reason"],
            })
            _exclude_artifact(artifact_record, path, attempt_id, quarantine_root)
            continue
        artifact_record.update({
            "state_count": summary.state_count,
            "event_count": summary.event_count,
        })
        macro_failure: str | None = None
        try:
            macro_labels = derive_macro_labels_for_shot(path)
        except Exception as error:
            macro_failure = f"macro label derivation failed: {str(error) or error.__class__.__name__}"
        else:
            if (
                macro_labels.state_sha256 != summary.state_sha256
                or macro_labels.events_sha256 != summary.event_sha256
            ):
                macro_failure = "macro label source digests differ from atomic validation"
        if macro_failure is not None:
            artifact_record["reason"] = macro_failure
            initial_identities.append({
                "attempt_id": attempt_id,
                "planned": expected,
                "observed": None,
                "recorded_expected": None,
                "matched": False,
                "reason": "macro label derivation failed",
            })
            supervision.append({
                "attempt_id": attempt_id,
                "attempted": False,
                "accepted": False,
                "reason": "macro label derivation failed",
            })
            defects.append({
                "scope": "attempt",
                "code": "macro_label_source_validation_failed",
                "attempt_id": attempt_id,
                "detail": macro_failure,
            })
            _exclude_artifact(artifact_record, path, attempt_id, quarantine_root)
            continue
        macro_evidence = _macro_evidence_rows(macro_labels, attempt_id)
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            metadata = None
            metadata_error = str(error)
        else:
            metadata_error = None
        observed = metadata.get("initial_engine_state_identity") if isinstance(metadata, Mapping) else None
        recorded_expected = metadata.get("expected_initial_engine_state_identity") if isinstance(metadata, Mapping) else None
        scenario_context = metadata.get("scenario_context") if isinstance(metadata, Mapping) else None
        expected_binding = _request_binding(loaded, entry)
        request_bound = isinstance(scenario_context, Mapping) and all(
            scenario_context.get(field) == value for field, value in expected_binding.items()
        )
        envelope_bound = (
            isinstance(metadata, Mapping)
            and all(metadata.get(field) == value for field, value in version_envelope.items() if field != "generator_version")
            and isinstance(scenario_context, Mapping)
            and scenario_context.get("version_envelope") == dict(version_envelope)
        )
        if not request_bound:
            artifact_record["reason"] = "artifact scenario context does not match the plan request"
            defects.append({
                "scope": "attempt",
                "code": "artifact_request_binding_mismatch",
                "attempt_id": attempt_id,
                "detail": artifact_record["reason"],
            })
        if not envelope_bound:
            artifact_record["reason"] = "source artifact version envelope mismatch"
            defects.append({
                "scope": "attempt",
                "code": "artifact_version_envelope_mismatch",
                "attempt_id": attempt_id,
                "detail": artifact_record["reason"],
            })
        if isinstance(metadata, Mapping):
            artifact_record["deterministic_artifact_semantics"] = {
                "state_count": summary.state_count,
                "event_count": summary.event_count,
                "frame_sha256": list(summary.frame_sha256),
                "state_sha256": summary.state_sha256,
                "event_sha256": summary.event_sha256,
                "initial_engine_state_identity": metadata.get("initial_engine_state_identity"),
                "intervention_event_id": metadata.get("intervention_event_id"),
                "termination_reason": metadata.get("termination_reason"),
                "termination_fixed_step": metadata.get("termination_fixed_step"),
                "termination_event_id": metadata.get("termination_event_id"),
                "terminal_state_fixed_step": metadata.get("terminal_state_fixed_step"),
            }
        matched = observed == expected and recorded_expected == expected
        identity_record = {
            "attempt_id": attempt_id,
            "planned": expected,
            "observed": observed if isinstance(observed, str) else None,
            "recorded_expected": recorded_expected if isinstance(recorded_expected, str) else None,
            "matched": matched,
            "reason": None if matched else (metadata_error or "planned and observed initial engine state identities differ"),
        }
        initial_identities.append(identity_record)
        if not matched:
            defects.append({
                "scope": "attempt",
                "code": "initial_engine_state_identity_mismatch",
                "attempt_id": attempt_id,
                "detail": identity_record["reason"],
            })
        if not request_bound or not matched or not envelope_bound:
            if not envelope_bound and request_bound and matched:
                exclusion_reason = "source artifact version envelope failed"
            elif matched:
                exclusion_reason = "artifact request binding failed"
            elif request_bound:
                exclusion_reason = "initial engine state identity failed"
                artifact_record["reason"] = identity_record["reason"]
            else:
                exclusion_reason = "artifact request binding and initial engine state identity failed"
            supervision.append({
                "attempt_id": attempt_id,
                "attempted": False,
                "accepted": False,
                "reason": exclusion_reason,
            })
            _exclude_artifact(artifact_record, path, attempt_id, quarantine_root)
            continue
        label_path: Path | None = None
        try:
            label_path = write_relational_supervision(path)
            labels = validate_relational_supervision(path, label_path)
        except Exception as error:
            if label_path is not None:
                label_path.unlink(missing_ok=True)
            supervision_record = {
                "attempt_id": attempt_id,
                "attempted": True,
                "accepted": False,
                "reason": str(error) or error.__class__.__name__,
            }
            artifact_record["reason"] = supervision_record["reason"]
            defects.append({
                "scope": "attempt",
                "code": "relational_supervision_invalid",
                "attempt_id": attempt_id,
                "detail": supervision_record["reason"],
            })
            _exclude_artifact(artifact_record, path, attempt_id, quarantine_root)
        else:
            artifact_record["accepted"] = True
            artifact_record["pilot_disposition"] = "accepted"
            artifact_record["capture_id"] = labels.capture_id
            artifact_record["shot_id"] = labels.shot_id
            artifact_record["macro_semantics_evidence"] = macro_evidence
            supervision_record = {
                "attempt_id": attempt_id,
                "attempted": True,
                "accepted": True,
                "label_path": str(label_path),
                "capture_id": labels.capture_id,
                "shot_id": labels.shot_id,
                "frame_count": len(labels.frames),
                "event_count": labels.event_count,
            }
        supervision.append(supervision_record)
    return validation, initial_identities, supervision, defects


def _macro_semantics(
    validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe pending target semantics from assessment-time accepted evidence."""
    rows_by_predicate: dict[str, list[dict[str, Any]]] = {
        name: [] for name in TARGET_MACRO_PREDICATES
    }
    for artifact in validation:
        if not artifact.get("accepted"):
            continue
        stored = _require_mapping(
            artifact.get("macro_semantics_evidence"),
            "Accepted artifact macro_semantics_evidence",
        )
        if set(stored) != set(TARGET_MACRO_PREDICATES):
            raise ValueError("Accepted artifact macro semantics evidence is incomplete")
        for name in TARGET_MACRO_PREDICATES:
            rows_by_predicate[name].append(
                _thaw(_require_mapping(stored[name], f"Accepted artifact {name} evidence"))
            )

    canonical_predicates = _require_mapping(
        derivation_spec_json()["pending_predicates"],
        "Canonical pending macro predicate specifications",
    )
    return {
        "schema": MACRO_SEMANTICS_SCHEMA,
        "derivation_spec_version": DERIVATION_SPEC_VERSION,
        "derivation_spec_digest": derivation_spec_digest(),
        "predicates": {
            name: {
                "status": SemanticStatus.HYPOTHESIS_PENDING_REPRESENTATIVE_VALIDATION.value,
                "definition": canonical_predicates[name]["definition"],
                "prerequisites": canonical_predicates[name]["prerequisites"],
                "unavailable_cases": canonical_predicates[name]["unavailable_cases"],
                "failure_cases": canonical_predicates[name]["failure_cases"],
                "pending_reason": MACRO_SEMANTICS_PENDING_REASON,
                "evidence": rows_by_predicate[name],
            }
            for name in TARGET_MACRO_PREDICATES
        },
    }


def _assess_replays(
    loaded: LoadedCollectionPlan,
    version_envelope: Mapping[str, str],
    inputs: Sequence[ReplayInput],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    scenarios = {scenario.scenario_id: scenario for scenario in loaded.plan.scenarios}
    replay_bindings = [(item.scenario_id, item.intervention_identity) for item in inputs if isinstance(item, ReplayInput)]
    if len(replay_bindings) != len(set(replay_bindings)):
        raise ValueError("duplicate replay binding")
    for item in inputs:
        if not isinstance(item, ReplayInput):
            raise ValueError("replay_inputs must contain ReplayInput values")
        reference = _require_nonempty_string(item.reference, "Replay reference")
        reasons: list[str] = []
        try:
            declared_envelope = _validated_version_envelope(item.version_envelope, "Replay version_envelope")
        except ValueError as error:
            declared_envelope = {}
            reasons.append(str(error))
        if declared_envelope != dict(version_envelope):
            reasons.append("replay version envelope mismatch")
        scenario = scenarios.get(item.scenario_id)
        if scenario is None:
            reasons.append("replay scenario mismatch")
        else:
            if item.manifest.scenario_lineage.identity != scenario.scenario_manifest_projection["scenario_lineage_identity"]:
                reasons.append("replay scenario lineage mismatch")
            if item.manifest.generation.generator_version != version_envelope["generator_version"]:
                reasons.append("replay manifest generator version mismatch")
            interventions = {intervention.identity: intervention for intervention in scenario.interventions}
            intervention = interventions.get(item.intervention_identity)
            if intervention is None:
                reasons.append("replay planned intervention identity mismatch")
        try:
            verify_replay(item.manifest, item.xml_content)
        except Exception as error:
            reasons.append(str(error) or error.__class__.__name__)
        sources = [
            artifact for artifact in source_artifacts
            if artifact.get("accepted")
            and artifact.get("scenario_id") == item.scenario_id
            and artifact.get("intervention_identity") == item.intervention_identity
        ]
        if len(sources) != 1:
            reasons.append("replay source artifact binding mismatch")
            source_semantics = None
        else:
            source_semantics = sources[0].get("deterministic_artifact_semantics")
        replay_semantics: Mapping[str, Any] | None = None
        if item.artifact_path is None:
            reasons.append("replay artifact is unavailable")
        else:
            replay_path = Path(item.artifact_path)
            try:
                summary = validate_physics_shot_artifact(replay_path)
                metadata = _require_mapping(
                    json.loads((replay_path / "metadata.json").read_text(encoding="utf-8")),
                    "Replay metadata",
                )
            except Exception as error:
                reasons.append(str(error) or error.__class__.__name__)
            else:
                replay_context = metadata.get("scenario_context")
                expected_context = {
                    "version_envelope": declared_envelope,
                    "plan_identity": loaded.plan.identity,
                    "plan_version": loaded.plan.plan_version,
                    "scenario_id": item.scenario_id,
                    "scenario_identity": scenario.identity if scenario is not None else None,
                    "intervention_id": intervention.id if scenario is not None and intervention is not None else None,
                    "intervention_identity": item.intervention_identity,
                }
                if not isinstance(replay_context, Mapping) or any(
                    replay_context.get(key) != value for key, value in expected_context.items()
                ):
                    reasons.append("replay artifact binding mismatch")
                if any(
                    metadata.get(field) != value
                    for field, value in version_envelope.items()
                    if field != "generator_version"
                ):
                    reasons.append("replay artifact provenance envelope mismatch")
                replay_semantics = {
                    "state_count": summary.state_count,
                    "event_count": summary.event_count,
                    "frame_sha256": list(summary.frame_sha256),
                    "state_sha256": summary.state_sha256,
                    "event_sha256": summary.event_sha256,
                    "initial_engine_state_identity": metadata.get("initial_engine_state_identity"),
                    "intervention_event_id": metadata.get("intervention_event_id"),
                    "termination_reason": metadata.get("termination_reason"),
                    "termination_fixed_step": metadata.get("termination_fixed_step"),
                    "termination_event_id": metadata.get("termination_event_id"),
                    "terminal_state_fixed_step": metadata.get("terminal_state_fixed_step"),
                }
                if source_semantics != replay_semantics:
                    reasons.append("replay deterministic artifact semantics mismatch")
        record: dict[str, Any] = {
            "reference": reference,
            "scenario_id": item.scenario_id,
            "intervention_identity": item.intervention_identity,
            "version_envelope": declared_envelope,
            "passed": not reasons,
        }
        if reasons:
            record["reason"] = "; ".join(reasons)
        else:
            record.update({
                "scenario_lineage_identity": item.manifest.scenario_lineage.identity,
                "observed_initial_engine_state_identity": replay_semantics["initial_engine_state_identity"] if replay_semantics else None,
                "deterministic_artifact_semantics": _thaw(replay_semantics),
            })
        result.append(record)
    return result


def _assess_partitions(
    loaded: LoadedCollectionPlan,
    inputs: Sequence[PilotPartitionAudit],
    validation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    split_regimes = [
        item.manifest.split_regime for item in inputs
        if isinstance(item, PilotPartitionAudit)
    ]
    if len(split_regimes) != len(set(split_regimes)):
        raise ValueError("duplicate partition audit split regime")
    lineage_by_scenario = {
        scenario.scenario_id: str(scenario.scenario_manifest_projection["scenario_lineage_identity"])
        for scenario in loaded.plan.scenarios
    }
    role_by_lineage = {
        str(scenario.scenario_manifest_projection["scenario_lineage_identity"]): scenario.exposure_role
        for scenario in loaded.plan.scenarios
    }
    projection_by_lineage = {
        str(scenario.scenario_manifest_projection["scenario_lineage_identity"]): _thaw(
            scenario.scenario_manifest_projection
        )
        for scenario in loaded.plan.scenarios
    }
    pilot_lineages = tuple(sorted(lineage_by_scenario.values()))
    artifact_bindings = [
        {
            "attempt_id": item["attempt_id"],
            "scenario_id": item["scenario_id"],
            "scenario_lineage_identity": lineage_by_scenario[item["scenario_id"]],
        }
        for item in validation
        if item.get("accepted") and item.get("scenario_id") in lineage_by_scenario
    ]
    evidenced_lineages = {item["scenario_lineage_identity"] for item in artifact_bindings}
    for item in inputs:
        if not isinstance(item, PilotPartitionAudit):
            raise ValueError("partition_audits must contain PilotPartitionAudit values")
        reference = _require_nonempty_string(item.reference, "Partition audit reference")
        reasons: list[str] = []
        manifest_lineages = tuple(sorted(
            str(entry.scenario_manifest_projection["scenario_lineage_identity"])
            for entry in item.manifest.entries
        ))
        if manifest_lineages != pilot_lineages:
            reasons.append("partition does not match the pilot scenario lineage inventory")
        manifest_roles = {
            str(entry.scenario_manifest_projection["scenario_lineage_identity"]): entry.exposure_role
            for entry in item.manifest.entries
        }
        if manifest_roles != role_by_lineage:
            reasons.append("partition exposure roles do not match the loaded pilot plan")
        manifest_projections = {
            str(entry.scenario_manifest_projection["scenario_lineage_identity"]): _thaw(
                entry.scenario_manifest_projection
            )
            for entry in item.manifest.entries
        }
        if manifest_projections != projection_by_lineage:
            reasons.append("partition complete scenario manifest projection differs from the frozen plan")
        if evidenced_lineages != set(pilot_lineages):
            reasons.append("partition lacks accepted artifact evidence for every pilot scenario lineage")
        try:
            audit_cohort_partition_manifest(
                item.manifest,
                admitted_scenario_lineage_identities=item.admitted_scenario_lineage_identities,
                admitted_provenance_records=item.admitted_provenance_records,
            )
        except Exception as error:
            reasons.append(str(error) or error.__class__.__name__)
        record = {
            "reference": reference,
            "split_regime": getattr(item.manifest, "split_regime", None),
            "partition_identity": getattr(item.manifest, "identity", None),
            "pilot_scenario_lineage_identities": list(pilot_lineages),
            "accepted_artifact_bindings": artifact_bindings,
            "passed": not reasons,
        }
        if reasons:
            record["reason"] = "; ".join(reasons)
        result.append(record)
    return result


def _audit_coverage(
    loaded: LoadedCollectionPlan,
    collection_report: Mapping[str, Any],
    validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    strata = tuple(next(iter(loaded.plan.scenarios)).coverage_strata)
    collection_counts = {stratum: 0 for stratum in strata}
    pilot_counts = {stratum: 0 for stratum in strata}
    ledger_violations: list[str] = []
    scenarios = {scenario.scenario_id: scenario for scenario in loaded.plan.scenarios}
    expected_slots = sorted(
        (
            scenario.scenario_id,
            scenario.identity,
            intervention.id,
            intervention.identity,
            intervention.ordinal,
            intervention.intended_coverage_stratum,
            intervention.source,
        )
        for scenario in loaded.plan.scenarios
        for intervention in scenario.interventions
    )
    actual_slots: list[tuple[Any, ...]] = []
    slot_fields = {
        "scenario_id", "scenario_identity", "intervention_id", "intervention_identity",
        "ordinal", "intended_coverage_stratum", "source", "disposition", "terminal_status",
        "attempt_ids",
    }
    for slot in collection_report["planned_slots"]:
        if (
            not isinstance(slot, Mapping)
            or set(slot) != slot_fields
            or any(
                not isinstance(slot.get(field), str) or not slot.get(field)
                for field in (
                    "scenario_id", "scenario_identity", "intervention_id", "intervention_identity",
                    "intended_coverage_stratum", "source", "disposition", "terminal_status",
                )
            )
            or isinstance(slot.get("ordinal"), bool)
            or not isinstance(slot.get("ordinal"), int)
            or slot["ordinal"] <= 0
            or not isinstance(slot.get("attempt_ids"), list)
            or any(not isinstance(attempt_id, str) or not attempt_id for attempt_id in slot["attempt_ids"])
            or len(slot["attempt_ids"]) != len(set(slot["attempt_ids"]))
        ):
            ledger_violations.append("planned slot inventory contains a malformed slot")
            continue
        actual_slots.append((
            slot["scenario_id"],
            slot["scenario_identity"],
            slot["intervention_id"],
            slot["intervention_identity"],
            slot["ordinal"],
            slot["intended_coverage_stratum"],
            slot["source"],
        ))
        expected_attempt_ids = [
            entry["attempt_id"] for entry in collection_report["attempt_ledger"]
            if entry.get("scenario_id") == slot["scenario_id"]
            and entry.get("intervention_id") == slot["intervention_id"]
        ]
        if slot["attempt_ids"] != expected_attempt_ids:
            ledger_violations.append("planned slot attempt inventory differs from the ledger")
    if sorted(actual_slots) != expected_slots:
        ledger_violations.append("planned slot inventory differs from the frozen collection plan")
    accepted_entries = _accepted_entries(collection_report)
    pilot_attempt_ids = {item["attempt_id"] for item in validation if item.get("accepted")}
    valid_entries: list[Mapping[str, Any]] = []
    malformed_realization_attempt_ids: set[str] = set()
    for entry in collection_report["attempt_ledger"]:
        realized = entry.get("realized_coverage_strata")
        malformed = (
            not isinstance(realized, list)
            or any(not isinstance(stratum, str) or stratum not in collection_counts for stratum in realized)
            or len(realized) != len(set(realized))
            or (entry.get("status") != "accepted" and bool(realized))
        )
        if malformed:
            ledger_violations.append("ledger entry has malformed or duplicate realized coverage")
            malformed_realization_attempt_ids.add(str(entry.get("attempt_id")))
    for entry in accepted_entries:
        if str(entry.get("attempt_id")) in malformed_realization_attempt_ids:
            continue
        scenario = scenarios.get(entry.get("scenario_id"))
        if scenario is None:
            ledger_violations.append("accepted ledger entry names an unknown scenario")
            continue
        interventions = {intervention.id: intervention for intervention in scenario.interventions}
        intervention = interventions.get(entry.get("intervention_id"))
        if (
            intervention is None
            or entry.get("plan_identity") != loaded.plan.identity
            or entry.get("scenario_identity") != scenario.identity
            or entry.get("intervention_identity") != intervention.identity
            or entry.get("intervention_ordinal") != intervention.ordinal
        ):
            ledger_violations.append("accepted ledger entry does not match its frozen plan request")
            continue
        realized = entry.get("realized_coverage_strata")
        assert isinstance(realized, list)
        valid_entries.append(entry)
        for stratum in realized:
            collection_counts[stratum] += 1
            if entry.get("attempt_id") in pilot_attempt_ids:
                pilot_counts[stratum] += 1

    reported_counts = collection_report["realized_coverage_stratum_counts"]
    ledger_counts_match = isinstance(reported_counts, Mapping) and dict(reported_counts) == collection_counts
    if not ledger_counts_match:
        ledger_violations.append("reported realized counts differ from accepted ledger entries")

    gaps: list[dict[str, Any]] = []
    inapplicable: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    negative_evidence: list[dict[str, Any]] = []
    validation_by_attempt = {item["attempt_id"]: item for item in validation if item.get("accepted")}
    for scenario in loaded.plan.scenarios:
        for stratum, disposition in scenario.coverage_strata.items():
            status = disposition["status"]
            dispositions.append({
                "scenario_id": scenario.scenario_id,
                "stratum": stratum,
                "status": status,
                "detail": _thaw(disposition),
            })
            if status == "inapplicable":
                inapplicable.append({
                    "scenario_id": scenario.scenario_id,
                    "stratum": stratum,
                    "rationale": disposition["rationale"],
                })
                continue
            for intervention_id in disposition["intervention_ids"]:
                demonstrated = any(
                    entry.get("scenario_id") == scenario.scenario_id
                    and entry.get("intervention_id") == intervention_id
                    and stratum in entry.get("realized_coverage_strata", ())
                    and entry.get("attempt_id") in pilot_attempt_ids
                    for entry in valid_entries
                )
                if not demonstrated:
                    gaps.append({
                        "scenario_id": scenario.scenario_id,
                        "stratum": stratum,
                        "intervention_id": intervention_id,
                    })

        negative = scenario.negative_specification
        realized_negative = [
            entry for entry in valid_entries
            if entry.get("scenario_id") == scenario.scenario_id
            and "no-contact/miss" in entry.get("realized_coverage_strata", ())
        ]
        pilot_negative = [
            entry for entry in realized_negative
            if entry.get("attempt_id") in pilot_attempt_ids
        ]
        negative_violations: list[str] = []
        if len(pilot_negative) > negative.cap:
            negative_violations.append("realized no-contact/miss evidence exceeds the frozen cap")
        unexpected_ids = sorted({
            str(entry.get("intervention_id")) for entry in pilot_negative
            if entry.get("intervention_id") not in negative.intervention_ids
        })
        if unexpected_ids:
            negative_violations.append(
                "no-contact/miss evidence uses interventions outside the frozen negative set: "
                + ", ".join(unexpected_ids)
            )
        for entry in pilot_negative:
            attempt_id = str(entry["attempt_id"])
            artifact = validation_by_attempt.get(attempt_id)
            if artifact is None:
                negative_violations.append(f"{attempt_id} has no pilot-accepted artifact")
                continue
            artifact_path = Path(str(artifact["artifact_path"]))
            try:
                capture = load_physics_capture(
                    artifact_path / "physics_state.jsonl",
                    artifact_path / "physics_events.jsonl",
                )
            except Exception as error:
                negative_violations.append(
                    f"{attempt_id} physics sidecars are unavailable: {str(error) or error.__class__.__name__}"
                )
                continue
            non_trigger_contacts = sum(
                not contact.is_trigger for state in capture.states for contact in state.raw_contacts
            )
            if non_trigger_contacts:
                negative_violations.append(
                    f"{attempt_id} contains {non_trigger_contacts} non-trigger raw contacts"
                )
            if any(event.event_type.value == "collision" for event in capture.events):
                negative_violations.append(f"{attempt_id} contains a collision event")
        negative_evidence.append({
            "scenario_id": scenario.scenario_id,
            "cap": negative.cap,
            "intervention_ids": list(negative.intervention_ids),
            "semantic_justification": negative.semantic_justification,
            "realized_count": len(realized_negative),
            "pilot_realized_count": len(pilot_negative),
            "passed": not negative_violations,
            "violations": negative_violations,
        })

    passed = not ledger_violations and not gaps and all(item["passed"] for item in negative_evidence)
    return {
        "passed": passed,
        "representative": passed and not inapplicable,
        "ledger_counts_match": ledger_counts_match,
        "ledger_violations": ledger_violations,
        "collection_realized_counts": collection_counts,
        "pilot_realized_counts": pilot_counts,
        "coverage_dispositions": dispositions,
        "gaps": gaps,
        "inapplicable": inapplicable,
        "negative_evidence": negative_evidence,
    }


def _available_capabilities(
    loaded: LoadedCollectionPlan,
    collection_report: Mapping[str, Any],
    validation: Sequence[Mapping[str, Any]],
    initial_identities: Sequence[Mapping[str, Any]],
    supervision: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
    partition_audits: Sequence[Mapping[str, Any]],
    coverage_audit: Mapping[str, Any],
) -> set[str]:
    accepted = _accepted_entries(collection_report)
    all_artifacts_valid = bool(accepted) and len(validation) == len(accepted) and all(item["accepted"] for item in validation)
    available = {CAPABILITY_SCENARIO_LINEAGE, CAPABILITY_INTERVENTION_REPRESENTATION}
    if all_artifacts_valid:
        available.update((
            CAPABILITY_ATOMIC_PHYSICS_ARTIFACT,
            CAPABILITY_CAUSAL_ENTITIES,
            CAPABILITY_RAW_CONTACTS,
            CAPABILITY_DERIVED_SUPPORT,
            CAPABILITY_KINEMATICS,
            CAPABILITY_MACRO_EVENTS,
        ))
    if all_artifacts_valid and initial_identities and all(item["matched"] for item in initial_identities):
        available.add(CAPABILITY_INITIAL_STATE_IDENTITY)
    if supervision and all(item["accepted"] for item in supervision):
        available.add(CAPABILITY_RELATIONAL_SUPERVISION)
    expected_replay_bindings = {
        (str(item["scenario_id"]), str(item["intervention_identity"]))
        for item in validation if item.get("accepted")
    }
    passed_replay_bindings = {
        (str(item["scenario_id"]), str(item["intervention_identity"]))
        for item in replays if item.get("passed")
    }
    if expected_replay_bindings and passed_replay_bindings == expected_replay_bindings:
        available.add(CAPABILITY_DETERMINISTIC_REPLAY)
    if sum(item.get("split_regime") == "instance_held_out" and bool(item.get("passed")) for item in partition_audits) == 1:
        available.add(CAPABILITY_INSTANCE_HELD_OUT_PARTITION)
    if sum(item.get("split_regime") == "template_held_out" and bool(item.get("passed")) for item in partition_audits) == 1:
        available.add(CAPABILITY_TEMPLATE_HELD_OUT_PARTITION)
    if coverage_audit["representative"]:
        available.add(CAPABILITY_COVERAGE_STRATA)
    negative = coverage_audit["negative_evidence"]
    if negative and all(item["passed"] and item["pilot_realized_count"] > 0 for item in negative):
        available.add(CAPABILITY_BOUNDED_NEGATIVE_EVIDENCE)
    return available


def assess_representative_pilot(
    loaded_plan: LoadedCollectionPlan,
    collection_report: Mapping[str, Any],
    *,
    frozen_plan_copy: Path,
    version_envelope: Mapping[str, str],
    artifact_paths: Mapping[str, Path] | None = None,
    required_capabilities: Sequence[str] = DEFAULT_REQUIRED_CAPABILITIES,
    replay_inputs: Sequence[ReplayInput] = (),
    partition_audits: Sequence[PilotPartitionAudit] = (),
    unavailable_capabilities: Mapping[str, str] = MappingProxyType({}),
    unavailable_labels: Mapping[str, str] = MappingProxyType({}),
    systematic_exporter_defects: Sequence[str] = (),
    pilot_quarantine_root: Path | None = None,
) -> PilotReport:
    """Assess existing collection evidence without invoking a live engine."""
    if not isinstance(loaded_plan, LoadedCollectionPlan):
        raise ValueError("assess_representative_pilot requires a LoadedCollectionPlan")
    assert_plan_unchanged(loaded_plan, loaded_plan.path)
    frozen_copy = Path(frozen_plan_copy)
    try:
        if frozen_copy.is_symlink() or not frozen_copy.is_file():
            raise ValueError("Frozen collection plan copy is missing or is not a regular file")
        frozen_bytes = frozen_copy.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read frozen collection plan copy {frozen_copy}: {error}") from error
    if frozen_bytes != loaded_plan.original_bytes:
        raise ValueError("Frozen collection plan copy differs from loaded plan bytes")
    report = _validated_collection_report(loaded_plan, collection_report)
    versions = _validated_version_envelope(version_envelope)
    for scenario in loaded_plan.plan.scenarios:
        manifest_data = _require_mapping(
            scenario.scenario_manifest_projection["scenario_manifest"],
            "Planned scenario manifest",
        )
        generation = _require_mapping(manifest_data.get("generation"), "Planned scenario generation")
        if generation.get("generator_version") != versions["generator_version"]:
            raise ValueError("Planned scenario manifest generator version differs from pilot envelope")
    requested = _normalize_required_capabilities(required_capabilities)
    required = tuple(sorted(set(DEFAULT_REQUIRED_CAPABILITIES) | set(requested)))
    explicit_unavailable = _require_string_mapping(unavailable_capabilities, "unavailable_capabilities")
    labels = dict(KNOWN_UNAVAILABLE_LABELS)
    supplied_labels = _require_string_mapping(unavailable_labels, "unavailable_labels")
    if "material" in supplied_labels and supplied_labels["material"] != KNOWN_UNAVAILABLE_LABELS["material"]:
        raise ValueError("material unavailable label cannot be overridden")
    if "damage" in supplied_labels and supplied_labels["damage"] != MATERIAL_DAMAGE_PENDING_REASON:
        raise ValueError("damage unavailable label has an invalid reason")
    labels.update(supplied_labels)
    systematic_defects = _normalize_string_sequence(systematic_exporter_defects, "systematic_exporter_defects")
    overrides: dict[str, Path] | None = None
    if artifact_paths is not None:
        overrides = {
            _require_nonempty_string(key, "artifact path attempt ID"): Path(value)
            for key, value in _require_mapping(artifact_paths, "artifact_paths").items()
        }
        accepted_ids = {_require_nonempty_string(entry.get("attempt_id"), "Collection attempt_id") for entry in _accepted_entries(report)}
        if set(overrides) - accepted_ids:
            raise ValueError("artifact_paths contains an unknown accepted attempt ID")
    quarantine_root = (
        Path(pilot_quarantine_root)
        if pilot_quarantine_root is not None
        else frozen_copy.parent / "pilot_assessment_quarantine"
    )
    validation, identities, supervision, defects = _assessment_artifacts(
        loaded_plan,
        report,
        overrides,
        quarantine_root,
        versions,
    )
    macro_semantics = _macro_semantics(validation)
    material_damage_semantics = _material_damage_semantics(
        loaded_plan.plan.identity,
        versions,
        validation,
    )
    if material_damage_semantics["damage"]["status"] == SemanticStatus.ENGINE_VERIFIED.value:
        labels.pop("damage", None)
    else:
        labels["damage"] = MATERIAL_DAMAGE_PENDING_REASON
    coverage_audit = _audit_coverage(loaded_plan, report, validation)
    replays = _assess_replays(loaded_plan, versions, replay_inputs, validation)
    partitions = _assess_partitions(loaded_plan, partition_audits, validation)
    available = _available_capabilities(
        loaded_plan,
        report,
        validation,
        identities,
        supervision,
        replays,
        partitions,
        coverage_audit,
    )
    available.difference_update(explicit_unavailable)
    unavailable = dict(KNOWN_UNSUPPORTED_CAPABILITIES)
    unavailable.update(explicit_unavailable)
    available.difference_update(unavailable)
    for capability in required:
        if capability not in available and capability not in unavailable:
            unavailable[capability] = "the supplied pilot evidence does not demonstrate this capability"
    unavailable_records = [
        {"capability": capability, "reason": reason}
        for capability, reason in sorted(unavailable.items())
    ]
    all_defects = defects + [
        {"scope": "systematic", "code": "reported_exporter_defect", "detail": detail}
        for detail in systematic_defects
    ]
    reasons: list[str] = []
    missing_required = sorted(capability for capability in required if capability not in available)
    if missing_required:
        reasons.append("required capabilities are unavailable: " + ", ".join(missing_required))
    if all_defects:
        reasons.append("permanent or systematic exporter defects are present")
    if report["unmet_slots"]:
        reasons.append("planned collection slots are unmet")
    if report["realized_coverage_shortfalls"]:
        reasons.append("planned coverage strata were not realized")
    if not coverage_audit["passed"]:
        reasons.append("frozen-plan coverage audit failed")
    payload: dict[str, Any] = {
        "schema": PILOT_REPORT_SCHEMA,
        "report_version": PILOT_REPORT_VERSION,
        "identity": "",
        "version_envelope": versions,
        "plan_identity": loaded_plan.plan.identity,
        "plan_version": loaded_plan.plan.plan_version,
        "scenarios": _scenario_coverage(loaded_plan),
        "attempts": {
            "collection_accounting": {
                "planned_count": len(report["planned_slots"]),
                "accepted_count": report["accepted_count"],
                "rejected_count": report["rejected_count"],
                "failed_count": report["failed_count"],
                "quarantined_count": report["quarantined_count"],
                "ledger": report["attempt_ledger"],
                "quarantined": report["quarantined_attempts"],
            },
            "pilot_evidence": {
                "accepted_count": sum(item["accepted"] for item in validation),
                "excluded_count": sum(not item["accepted"] for item in validation),
                "accepted_attempt_ids": [item["attempt_id"] for item in validation if item["accepted"]],
                "exclusions": [item for item in validation if not item["accepted"]],
            },
            "atomic_validation": validation,
        },
        "coverage": {
            "planned_slots": report["planned_slots"],
            "realized_counts": report["realized_coverage_stratum_counts"],
            "unmet_slots": report["unmet_slots"],
            "realized_shortfalls": report["realized_coverage_shortfalls"],
            "audit": coverage_audit,
        },
        "replays": replays,
        "initial_state_identities": identities,
        "partition_audits": partitions,
        "supervision": supervision,
        "macro_semantics": macro_semantics,
        "material_damage_semantics": material_damage_semantics,
        "available_capabilities": sorted(available),
        "unavailable_capabilities": unavailable_records,
        "unavailable_labels": labels,
        "permanent_or_systematic_exporter_defects": all_defects,
        "pilot_status": "accepted" if not reasons else "rejected",
        "acceptance_decision": {
            "accepted": not reasons,
            "reasons": reasons,
            "required_capabilities": list(required),
        },
    }
    payload["identity"] = _identity(payload)
    return PilotReport.from_dict(payload)


def run_representative_pilot(
    loaded_plan: LoadedCollectionPlan,
    runtime: CollectionPlanRuntime,
    output_dir: Path,
    *,
    version_envelope: Mapping[str, str],
    required_capabilities: Sequence[str] = DEFAULT_REQUIRED_CAPABILITIES,
    replay_inputs: Sequence[ReplayInput] = (),
    partition_audits: Sequence[PilotPartitionAudit] = (),
    unavailable_capabilities: Mapping[str, str] = MappingProxyType({}),
    unavailable_labels: Mapping[str, str] = MappingProxyType({}),
    systematic_exporter_defects: Sequence[str] = (),
) -> PilotReport:
    """Execute a frozen plan, assess its accepted artifacts, and write its pilot report."""
    root = Path(output_dir)
    collection_report = execute_collection_plan(loaded_plan, runtime, root)
    report = assess_representative_pilot(
        loaded_plan,
        collection_report,
        frozen_plan_copy=root / PLAN_COPY_FILENAME,
        version_envelope=version_envelope,
        required_capabilities=required_capabilities,
        replay_inputs=replay_inputs,
        partition_audits=partition_audits,
        unavailable_capabilities=unavailable_capabilities,
        unavailable_labels=unavailable_labels,
        systematic_exporter_defects=systematic_exporter_defects,
        pilot_quarantine_root=root / "pilot_assessment_quarantine",
    )
    write_pilot_report(report, root / PILOT_REPORT_FILENAME)
    return report


def _key_value(items: Sequence[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"{option} values must use NAME=VALUE")
        key = _require_nonempty_string(key, option)
        value = _require_nonempty_string(value, option)
        if key in result:
            raise ValueError(f"{option} names must be unique")
        result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess representative-pilot evidence without a Science Birds engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    assess = subparsers.add_parser("assess", help="assess an existing collection report and accepted artifacts")
    assess.add_argument("--plan", required=True, type=Path)
    assess.add_argument("--collection-report", required=True, type=Path)
    assess.add_argument("--frozen-plan-copy", required=True, type=Path)
    assess.add_argument("--artifact", action="append", default=[], metavar="ATTEMPT_ID=PATH")
    assess.add_argument("--output", required=True, type=Path)
    assess.add_argument("--version", action="append", required=True, metavar="NAME=VALUE")
    assess.add_argument("--required-capability", action="append", default=[])
    assess.add_argument("--unavailable-capability", action="append", default=[], metavar="NAME=REASON")
    assess.add_argument("--unavailable-label", action="append", default=[], metavar="NAME=REASON")
    assess.add_argument("--systematic-exporter-defect", action="append", default=[])
    assess.add_argument(
        "--replay",
        action="append",
        nargs=5,
        default=[],
        metavar=("MANIFEST", "XML", "SCENARIO_ID", "INTERVENTION_ID", "ARTIFACT"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        loaded = load_collection_plan(args.plan)
        collection_data = json.loads(args.collection_report.read_text(encoding="utf-8"))
        versions = _key_value(args.version, "--version")
        artifacts = {key: Path(value) for key, value in _key_value(args.artifact, "--artifact").items()}
        replays = tuple(
            ReplayInput(
                load_manifest(manifest_path),
                Path(xml_path).read_bytes(),
                f"{manifest_path}:{xml_path}",
                versions,
                scenario_id,
                intervention_identity,
                Path(artifact_path),
            )
            for manifest_path, xml_path, scenario_id, intervention_identity, artifact_path in args.replay
        )
        report = assess_representative_pilot(
            loaded,
            collection_data,
            frozen_plan_copy=args.frozen_plan_copy,
            version_envelope=versions,
            artifact_paths=artifacts or None,
            required_capabilities=tuple(args.required_capability) or DEFAULT_REQUIRED_CAPABILITIES,
            replay_inputs=replays,
            unavailable_capabilities=_key_value(args.unavailable_capability, "--unavailable-capability"),
            unavailable_labels=_key_value(args.unavailable_label, "--unavailable-label"),
            systematic_exporter_defects=tuple(args.systematic_exporter_defect),
        )
        write_pilot_report(report, args.output)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=os.sys.stderr)
        return 2
    print(json.dumps({"pilot_status": report.pilot_status, "report": str(args.output)}, sort_keys=True))
    return 0 if report.pilot_status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
