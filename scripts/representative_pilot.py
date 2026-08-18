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
from scripts.physics_relational_supervision import (
    validate_relational_supervision,
    write_relational_supervision,
)
from scripts.scenario_manifest import ScenarioManifest, load_manifest, verify_replay


PILOT_REPORT_SCHEMA: Final = "representative_pilot_report_v1"
PILOT_REPORT_VERSION: Final = 1
PILOT_REPORT_IDENTITY_NAMESPACE: Final = "representative-pilot-report-v1"
PILOT_REPORT_FILENAME: Final = "representative_pilot_report.json"

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
    CAPABILITY_MATERIAL_DAMAGE_MAPPING: "no accepted engine-verified material and damage mapping exists",
    CAPABILITY_REPRESENTATIVE_MACRO_SEMANTICS: "representative validation remains pending for required macro predicates",
    CAPABILITY_COHORT_RELEASE: "representative pilot evidence is not an immutable cohort release",
})

KNOWN_UNAVAILABLE_LABELS: Final = MappingProxyType({
    "illegal_contact": "the legal-contact ontology is unavailable",
    "material": "no accepted engine-verified material mapping exists",
    "damage": "no accepted engine-verified damage mapping exists",
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

    schema: Literal["representative_pilot_report_v1"]
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
        unavailable_labels = _require_string_mapping(raw["unavailable_labels"], "Pilot report unavailable_labels")
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
            tuple(raw["available_capabilities"]),
            tuple(_freeze(item) for item in raw["unavailable_capabilities"]),
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
    labels.update(_require_string_mapping(unavailable_labels, "unavailable_labels"))
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
