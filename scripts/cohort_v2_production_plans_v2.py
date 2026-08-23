"""Stable-only cohort-v2 production-plan correction for issue #53."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from typing import Any, Final, Mapping
import xml.etree.ElementTree as ET

from scripts.build_issue_45_evidence import (
    CONSTRAINTS_WORKBOOK_REFERENCE,
    ROLES,
    _materialize,
    _record_for_family,
)
from scripts.build_issue_52_evidence import validate_issue_52_evidence
from scripts.cohort_v2_partition import (
    EXPOSURE_ROLES,
    CohortV2PartitionExposureManifest,
    create_cohort_v2_partition_exposure_manifest,
)
from scripts.cohort_v2_production_plans import (
    ROLE_ORDER,
    _assignments,
    validate_issue_52_payloads,
)
from scripts.cohort_v2_scenarios import (
    ScenarioInventoryEntry,
    create_scenario_inventory_entry,
    load_cohort_v2_scenario_manifest,
)
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    create_final_evaluation_workflow_access_manifest,
)
from scripts.observation_trace import validate_observation_trace


ROOT: Final = Path(__file__).resolve().parents[1]
PLAN_VERSION: Final = 2
COLLECTION_SCHEMA: Final = "cohort_v2_production_collection_plan_v2"
COLLECTION_IDENTITY: Final = (
    "cohort-v2-production-collection-plan-v2:issue-53:stable-only"
)
PARAMETER_SCHEMA: Final = "cohort_v2_production_parameter_plan_v2"
PARAMETER_IDENTITY: Final = (
    "cohort-v2-production-parameter-plan-v2:issue-53:stable-only"
)
BUNDLE_SCHEMA: Final = "issue_53_cohort_v2_production_plans_bundle_v2"
BUNDLE_IDENTITY: Final = "issue-53-cohort-v2-production-plans-bundle-v2:stable-only"
CORRECTION_EVIDENCE_SCHEMA: Final = "issue_53_stable_only_plan_correction_evidence_v2"
CORRECTION_EVIDENCE_IDENTITY: Final = (
    "issue-53-stable-only-plan-correction-evidence-v2:public-non-final"
)
INVENTORY_SCHEMA: Final = "cohort_v2_production_scenario_inventory_v2"
INVENTORY_IDENTITY: Final = (
    "cohort-v2-production-scenario-inventory-v2:issue-53:stable-only"
)
SEALED_AUTHORITY_IDENTITY: Final = "issue-53-plan-v2-sealed-final-authority:seed-4503"
SEALED_AUTHORITY_REFERENCE: Final = SEALED_AUTHORITY_IDENTITY
WORKFLOW_IDENTITY: Final = (
    "central-v2-final-evaluation-workflow-v2:issue-53:stable-only"
)
WORKFLOW_OPERATOR_IDENTITY: Final = (
    "novphy-operator-v2:issue-53-stable-only-final-evaluation-custodian"
)
WORKFLOW_FROZEN_AT: Final = "2026-08-23T00:00:00Z"
FINAL_SEED: Final = 4503

DEFAULT_PLAN_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-plan-v2"
DEFAULT_SEALED_AUTHORITY_ROOT: Final = (
    ROOT / ".local-artifacts/issue-53-plan-v2-sealed-final-authority"
)
V1_PLAN_ROOT: Final = ROOT / "data/runtime_evidence/issue-52"
PUBLIC_FAILED_ROOT: Final = ROOT / "data/runtime_evidence/issue-53"
V1_PARTITION_PATH: Final = (
    ROOT / "data/runtime_evidence/issue-47/partition-exposure-manifest.json"
)

PLAN_MEMBERS: Final = (
    "bundle-manifest.json",
    "collection-plan.json",
    "final-evaluation-workflow-access-manifest.json",
    "final-evaluation.sealed-projection.json",
    "partition-exposure-manifest.json",
    "plan-correction-evidence.json",
    "production-parameter-plan.json",
    "scenario-inventory.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Plan-v2 JSON artifact must be an object: {path}")
    return value


def _xml_counts(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_bytes())
    return {
        "artifact_path": path.relative_to(ROOT).as_posix(),
        "bird_count": len(root.findall(".//Bird")),
        "pig_count": len(root.findall(".//Pig")),
    }


def derive_plan_correction_evidence(
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    """Project only public non-final evidence used by the correction."""
    repository_root = Path(repository_root).resolve()
    public_root = repository_root / "data/runtime_evidence/issue-53"
    accounting_path = public_root / "production-attempt-accounting.json"
    accounting = _load_object(accounting_path)
    non_final = [
        {
            "attempt_id": item["attempt_id"],
            "exposure_role": item["exposure_role"],
            "intervention_id": item["intervention_id"],
            "terminal_reason": item["terminal_reason"],
        }
        for item in accounting["attempt_ledger"]
        if item["exposure_role"] != "final_evaluation"
    ]
    if len(non_final) != 18 or any(item["terminal_reason"] is None for item in non_final):
        raise ValueError("Plan-v2 correction requires 18 public non-final outcomes")

    xml_counts = [
        _xml_counts(path)
        for path in sorted((public_root / "scenario-authorities").glob("*.xml"))
    ]
    if len(xml_counts) != 3 or any(item["pig_count"] != 1 for item in xml_counts):
        raise ValueError("Plan-v2 correction XML bird/pig evidence is incomplete")

    diagnostics = []
    for manifest_path in sorted(
        (public_root / "primary-rollouts").glob(
            "*/observation-trace/observation_trace_manifest.json"
        )
    ):
        trace = validate_observation_trace(manifest_path.parent)
        if trace["exposure_role"] == "final_evaluation":
            raise ValueError("Plan-v2 correction diagnostics crossed the final boundary")
        frame = trace["frame_records"][0]
        capture = frame["capture_metadata"]
        camera = capture["camera"]
        viewport = capture["viewport"]
        diagnostics.append(
            {
                "artifact_path": manifest_path.relative_to(repository_root).as_posix(),
                "exposure_role": trace["exposure_role"],
                "observation_trace_identity": trace["identity"],
                "camera_identity": camera["camera_identity"],
                "projection_kind": camera["projection_kind"],
                "viewport_width_pixels": viewport["width_pixels"],
                "viewport_height_pixels": viewport["height_pixels"],
                "world_to_observation_method": capture[
                    "world_to_observation_transform"
                ]["method"],
                "agent_width_pixels": frame["agent_observation"]["width_pixels"],
                "agent_height_pixels": frame["agent_observation"]["height_pixels"],
            }
        )
    if len(diagnostics) != 18:
        raise ValueError("Plan-v2 correction requires 18 camera-aligned diagnostics")
    for item in diagnostics:
        if (
            item["camera_identity"] != "unity-main-camera"
            or item["projection_kind"] != "orthographic"
            or (item["viewport_width_pixels"], item["viewport_height_pixels"])
            != (item["agent_width_pixels"], item["agent_height_pixels"])
            or item["world_to_observation_method"]
            != "unity_world_to_clip_to_top_left_pixel_v1"
        ):
            raise ValueError("Plan-v2 correction camera diagnostics are not screen-aligned")

    termination_counts: dict[str, int] = {}
    for item in non_final:
        reason = item["terminal_reason"]
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    return {
        "schema": CORRECTION_EVIDENCE_SCHEMA,
        "identity": CORRECTION_EVIDENCE_IDENTITY,
        "exposure_boundary": "public_non_final_only",
        "source_public_accounting_path": accounting_path.relative_to(
            repository_root
        ).as_posix(),
        "non_final_attempts": non_final,
        "observed_non_final_termination_counts": dict(sorted(termination_counts.items())),
        "xml_initial_object_counts": xml_counts,
        "camera_aligned_v2_diagnostics": diagnostics,
        "reviewed_final_outcomes_used": False,
        "decision": {
            "production_scope": "stable_only",
            "quota_bearing_termination_classes": ["stable_entered"],
            "pilot_only_supported_termination_classes": ["level_clear", "level_fail"],
            "rationale": (
                "Issue 53 corrects the production termination contract to stable stopping; "
                "clear/fail remain supported pilot capabilities but carry no production quota."
            ),
        },
    }


def _inventory_entries(
    final_scenario: Any,
    v1_partition: CohortV2PartitionExposureManifest,
) -> tuple[ScenarioInventoryEntry, ...]:
    entries = []
    for source in v1_partition.entries:
        if source.exposure_role == "final_evaluation":
            continue
        entries.append(
            ScenarioInventoryEntry.from_dict(
                {
                    "exposure_role": source.exposure_role,
                    "inventory_state": source.inventory_state,
                    "scenario_manifest_identity": source.scenario_manifest_identity,
                    "benchmark_condition_identity": source.benchmark_condition_identity,
                    "scenario_template_identity": source.scenario_template_identity,
                    "level_instance_identity": source.level_instance_identity,
                    "scenario_specification_identity": source.scenario_specification_identity,
                    "scenario_lineage_identity": source.scenario_lineage_identity,
                    "declared_initial_engine_state_identity": (
                        source.declared_initial_engine_state_identity
                    ),
                    "scenario_manifest_reference": source.scenario_manifest_reference,
                }
            )
        )
    entries.append(
        create_scenario_inventory_entry(
            "final_evaluation",
            "sealed_final",
            final_scenario,
            sealed_scenario_manifest_reference=SEALED_AUTHORITY_REFERENCE,
        )
    )
    return tuple(entries)


def _inventory(entries: tuple[ScenarioInventoryEntry, ...]) -> dict[str, Any]:
    return {
        "schema": INVENTORY_SCHEMA,
        "identity": INVENTORY_IDENTITY,
        "inventory_version": 2,
        "supersedes_inventory_identity": (
            "cohort-v2-production-scenario-inventory-v1:issue-53"
        ),
        "entries": [entry.to_dict() for entry in entries],
    }


def _corrected_plans(
    v1_collection: Mapping[str, Any],
    v1_parameters: Mapping[str, Any],
    partition: CohortV2PartitionExposureManifest,
) -> dict[str, dict[str, Any]]:
    correction_record = {
        "source_record_id": "issue_53_public_non_final_correction",
        "artifact_path": (
            "data/runtime_evidence/issue-53-plan-v2/plan-correction-evidence.json"
        ),
        "artifact_identity": CORRECTION_EVIDENCE_IDENTITY,
        "record_ids": [CORRECTION_EVIDENCE_IDENTITY],
        "exposure_boundary": "no_final_evaluation_outcomes",
    }
    collection = deepcopy(v1_collection)
    collection.update(
        {
            "schema": COLLECTION_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": COLLECTION_IDENTITY,
            "supersedes_plan_identity": v1_collection["identity"],
            "production_scope": "stable_only",
        }
    )
    collection["authority"] = {
        **collection["authority"],
        "github_issue": 53,
        "partition_manifest_identity": partition.identity,
        "correction_evidence_identity": CORRECTION_EVIDENCE_IDENTITY,
    }
    intervention_ids = [item["id"] for item in collection["interventions"]]
    collection["assignments"] = _assignments(partition.to_dict(), intervention_ids)
    for intervention in collection["interventions"]:
        intervention["intended_termination_class"] = "stable_entered"
        intervention["evidence_id"] = "termination_quotas"
    collection["termination_policy"] = {
        "closed_vocabulary": [
            "level_clear",
            "level_fail",
            "stable_entered",
            "rollout_ceiling",
        ],
        "accepted_pilot_capability_classes": [
            "level_clear",
            "level_fail",
            "stable_entered",
        ],
        "required_quota_classes": ["stable_entered"],
        "non_quota_bearing_production_classes": ["level_clear", "level_fail"],
        "rollout_ceiling_disposition": (
            "allowed_safety_termination_not_quota_bearing"
        ),
        "evidence_id": "termination_quotas",
    }
    collection["quotas"]["termination_class"] = {
        "level_clear": {"quota": 0, "evidence_id": "termination_quotas"},
        "level_fail": {"quota": 0, "evidence_id": "termination_quotas"},
        "stable_entered": {"quota": 24, "evidence_id": "termination_quotas"},
    }
    collection["source_records"][
        "issue_53_public_non_final_correction"
    ] = correction_record
    for evidence in collection["evidence"].values():
        evidence["plan_version"] = PLAN_VERSION
    collection["evidence"]["termination_quotas"] = {
        "plan_version": PLAN_VERSION,
        "source_record_ids": ["issue_53_public_non_final_correction"],
        "analysis_method": (
            "Correct the production scope from the public 18-rollout non-final "
            "accounting, XML bird/pig counts, and camera-aligned v2 diagnostics."
        ),
        "observed_range_or_uncertainty": (
            "Public non-final evidence contains 18 attempts, 14 stable_entered and "
            "4 level_fail terminations, no level_clear termination, three source XML "
            "authorities, and 18 aligned 840x480 observation diagnostics; final seed "
            "4503 remains prospective."
        ),
        "decision_rule": (
            "Require stable_entered for all 24 production assignments; keep level_clear "
            "and level_fail supported but non-quota-bearing and fail closed on any "
            "non-stable production termination."
        ),
        "rationale": (
            "Stable stopping is the corrected cohort boundary; clear/fail coverage "
            "belongs to the accepted pilot rather than this production cohort."
        ),
    }

    parameters = deepcopy(v1_parameters)
    parameters.update(
        {
            "schema": PARAMETER_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": PARAMETER_IDENTITY,
            "supersedes_plan_identity": v1_parameters["identity"],
            "production_scope": "stable_only",
        }
    )
    parameters["authority"] = {
        **parameters["authority"],
        "github_issue": 53,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "correction_evidence_identity": CORRECTION_EVIDENCE_IDENTITY,
    }
    parameters["parameters"]["termination"] = {
        "closed_vocabulary": {
            "value": [
                "level_clear",
                "level_fail",
                "stable_entered",
                "rollout_ceiling",
            ],
            "unit": "termination_reason",
            "evidence_id": "termination_vocabulary",
        },
        "accepted_pilot_capability_classes": {
            "value": ["level_clear", "level_fail", "stable_entered"],
            "unit": "termination_reason",
            "evidence_id": "termination_vocabulary",
        },
        "quota_bearing_classes": {
            "value": ["stable_entered"],
            "unit": "termination_reason",
            "evidence_id": "termination_vocabulary",
        },
        "non_quota_bearing_production_classes": {
            "value": ["level_clear", "level_fail"],
            "unit": "termination_reason",
            "evidence_id": "termination_vocabulary",
        },
    }
    parameters["source_records"][
        "issue_53_public_non_final_correction"
    ] = correction_record
    for evidence in parameters["evidence"].values():
        evidence["plan_version"] = PLAN_VERSION
    parameters["evidence"]["termination_vocabulary"] = {
        "plan_version": PLAN_VERSION,
        "source_record_ids": ["issue_53_public_non_final_correction"],
        "analysis_method": (
            "Separate supported pilot terminal capability from stable-only production quota."
        ),
        "observed_range_or_uncertainty": (
            "All three engine terminal classes remain supported; only stable_entered "
            "defines admission to the corrected production cohort."
        ),
        "decision_rule": (
            "Retain the closed vocabulary and make stable_entered the sole quota-bearing class."
        ),
        "rationale": (
            "This corrects cohort scope without claiming that Unity clear/fail are unsupported."
        ),
    }
    return {
        "collection-plan.json": collection,
        "production-parameter-plan.json": parameters,
    }


def _bundle(partition_identity: str, workflow_identity: str) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "identity": BUNDLE_IDENTITY,
        "github_issue": 53,
        "supersedes_collection_plan_identity": (
            "cohort-v2-production-collection-plan-v1:issue-52"
        ),
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "scenario_inventory_identity": INVENTORY_IDENTITY,
        "partition_manifest_identity": partition_identity,
        "final_evaluation_workflow_access_manifest_identity": workflow_identity,
        "sealed_final_authority_identity": SEALED_AUTHORITY_IDENTITY,
        "sealed_final_authority_reference": SEALED_AUTHORITY_REFERENCE,
        "artifacts": [name for name in PLAN_MEMBERS if name != "bundle-manifest.json"],
        "passed": True,
    }


def _materialize_seed_4503(output_root: Path) -> tuple[Any, Any, Any]:
    role = replace(ROLES[3], seed=FINAL_SEED)
    workbook_path = ROOT / CONSTRAINTS_WORKBOOK_REFERENCE
    source_path, constraints, record = _record_for_family(
        ROOT, workbook_path.read_bytes(), role.family
    )
    materialized, scenario = _materialize(
        role,
        source_path,
        constraints,
        record,
        workbook_path,
        output_root,
    )
    return role, materialized, scenario


def build_plan_v2_payloads(final_scenario: Any) -> dict[str, dict[str, Any]]:
    validate_issue_52_evidence(V1_PLAN_ROOT, revalidate_pilot=False)
    v1_collection = _load_object(V1_PLAN_ROOT / "collection-plan.json")
    v1_parameters = _load_object(V1_PLAN_ROOT / "production-parameter-plan.json")
    validate_issue_52_payloads(
        {
            "collection-plan.json": v1_collection,
            "production-parameter-plan.json": v1_parameters,
        },
        ROOT,
    )
    v1_partition = CohortV2PartitionExposureManifest.from_dict(
        _load_object(V1_PARTITION_PATH)
    )
    entries = _inventory_entries(final_scenario, v1_partition)
    inventory = _inventory(entries)
    partition = create_cohort_v2_partition_exposure_manifest(
        partition_version=2,
        source_inventory_identity=INVENTORY_IDENTITY,
        source_inventory_review_url="https://github.com/Sino-Huang/NovPhy/issues/53",
        inventory_entries=entries,
        lineage_quotas={role: 1 for role in EXPOSURE_ROLES},
    )
    final_entry = entries[-1]
    workflow = create_final_evaluation_workflow_access_manifest(
        partition,
        workflow_version=2,
        workflow_identity=WORKFLOW_IDENTITY,
        operator_identity=WORKFLOW_OPERATOR_IDENTITY,
        frozen_at=WORKFLOW_FROZEN_AT,
        authorized_artifacts=[
            {
                "artifact_kind": "scenario_manifest",
                "artifact_identity": final_entry.scenario_manifest_identity,
                "source_scenario_lineage_identities": [
                    final_entry.scenario_lineage_identity
                ],
            }
        ],
    )
    plans = _corrected_plans(v1_collection, v1_parameters, partition)
    evidence = derive_plan_correction_evidence(ROOT)
    return {
        **plans,
        "plan-correction-evidence.json": evidence,
        "scenario-inventory.json": inventory,
        "partition-exposure-manifest.json": partition.to_dict(),
        "final-evaluation.sealed-projection.json": final_entry.to_dict(),
        "final-evaluation-workflow-access-manifest.json": workflow.to_dict(),
        "bundle-manifest.json": _bundle(partition.identity, workflow.identity),
    }


def validate_plan_v2_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    repository_root: Path = ROOT,
) -> None:
    if set(payloads) != set(PLAN_MEMBERS):
        raise ValueError("Plan-v2 immutable bundle membership is incomplete")
    collection = payloads["collection-plan.json"]
    parameters = payloads["production-parameter-plan.json"]
    evidence = payloads["plan-correction-evidence.json"]
    inventory = payloads["scenario-inventory.json"]
    final_entry = ScenarioInventoryEntry.from_dict(
        payloads["final-evaluation.sealed-projection.json"]
    )
    partition = CohortV2PartitionExposureManifest.from_dict(
        payloads["partition-exposure-manifest.json"]
    )
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
        payloads["final-evaluation-workflow-access-manifest.json"]
    )
    if evidence != derive_plan_correction_evidence(repository_root):
        raise ValueError("Plan-v2 correction evidence differs from public non-final sources")
    if (
        collection.get("schema") != COLLECTION_SCHEMA
        or collection.get("identity") != COLLECTION_IDENTITY
        or parameters.get("schema") != PARAMETER_SCHEMA
        or parameters.get("identity") != PARAMETER_IDENTITY
        or collection.get("plan_version") != PLAN_VERSION
        or parameters.get("plan_version") != PLAN_VERSION
    ):
        raise ValueError("Plan-v2 identities or schemas are stale")
    if parameters.get("authority", {}).get("collection_plan_identity") != COLLECTION_IDENTITY:
        raise ValueError("Plan-v2 parameter plan is not collection-bound")
    interventions = collection.get("interventions")
    assignments = collection.get("assignments")
    if (
        not isinstance(interventions, list)
        or len(interventions) != 6
        or [item.get("ordinal") for item in interventions] != list(range(1, 7))
        or any(item.get("intended_termination_class") != "stable_entered" for item in interventions)
        or not isinstance(assignments, list)
        or [item.get("exposure_role") for item in assignments] != list(ROLE_ORDER)
        or any(item.get("planned_rollout_quota") != 6 for item in assignments)
    ):
        raise ValueError("Plan-v2 role/stratum assignments are not exact")
    termination_quotas = collection.get("quotas", {}).get("termination_class")
    if termination_quotas != {
        "level_clear": {"quota": 0, "evidence_id": "termination_quotas"},
        "level_fail": {"quota": 0, "evidence_id": "termination_quotas"},
        "stable_entered": {"quota": 24, "evidence_id": "termination_quotas"},
    }:
        raise ValueError("Plan-v2 stable-only termination quotas are stale")
    policy = collection.get("termination_policy", {})
    if (
        policy.get("required_quota_classes") != ["stable_entered"]
        or policy.get("non_quota_bearing_production_classes")
        != ["level_clear", "level_fail"]
        or set(policy.get("closed_vocabulary", []))
        != {"level_clear", "level_fail", "stable_entered", "rollout_ceiling"}
    ):
        raise ValueError("Plan-v2 termination policy is not stable-only")
    if inventory.get("schema") != INVENTORY_SCHEMA or inventory.get("identity") != INVENTORY_IDENTITY:
        raise ValueError("Plan-v2 scenario inventory identity is stale")
    entries = tuple(ScenarioInventoryEntry.from_dict(item) for item in inventory["entries"])
    if len(entries) != 4 or entries[-1] != final_entry:
        raise ValueError("Plan-v2 scenario inventory entries are stale")
    old_partition = CohortV2PartitionExposureManifest.from_dict(
        _load_object(Path(repository_root) / "data/runtime_evidence/issue-47/partition-exposure-manifest.json")
    )
    old_non_final = old_partition.entries[:3]
    if tuple(entry.to_dict() for entry in partition.entries[:3]) != tuple(
        entry.to_dict() for entry in old_non_final
    ):
        raise ValueError("Plan-v2 changed a preserved non-final lineage")
    old_identities = {
        value
        for entry in old_partition.entries
        for value in (
            entry.scenario_manifest_identity,
            entry.scenario_lineage_identity,
            entry.level_instance_identity,
        )
    }
    if any(
        value in old_identities
        for value in (
            final_entry.scenario_manifest_identity,
            final_entry.scenario_lineage_identity,
            final_entry.level_instance_identity,
        )
    ) or not any(
        marker in final_entry.level_instance_identity
        for marker in (":4503:", "%3A4503%3A")
    ):
        raise ValueError("Plan-v2 seed-4503 final authority overlaps an existing lineage")
    if (
        collection["authority"]["partition_manifest_identity"] != partition.identity
        or workflow.partition_identity != partition.identity
        or workflow.authorization_state != "pending"
        or workflow.authorized_artifacts[0].artifact_identity
        != final_entry.scenario_manifest_identity
    ):
        raise ValueError("Plan-v2 partition, final projection, or workflow is stale")
    source_paths = [
        record.get("artifact_path", "")
        for record in collection["source_records"].values()
    ]
    if (
        evidence.get("reviewed_final_outcomes_used") is not False
        or any(
            item.get("exposure_role") == "final_evaluation"
            for item in evidence.get("non_final_attempts", [])
        )
        or any(
            token in path
            for path in source_paths
            for token in ("human-review", ".local-artifacts")
        )
    ):
        raise ValueError("Plan-v2 derivation references reviewed final outcomes")
    expected_plans = _corrected_plans(
        _load_object(Path(repository_root) / "data/runtime_evidence/issue-52/collection-plan.json"),
        _load_object(Path(repository_root) / "data/runtime_evidence/issue-52/production-parameter-plan.json"),
        partition,
    )
    if collection != expected_plans["collection-plan.json"] or parameters != expected_plans["production-parameter-plan.json"]:
        raise ValueError("Plan-v2 plans differ from their exact correction derivation")
    if payloads["bundle-manifest.json"] != _bundle(partition.identity, workflow.identity):
        raise ValueError("Plan-v2 bundle manifest is stale")


def validate_plan_v2_evidence(
    plan_root: Path = DEFAULT_PLAN_ROOT,
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(plan_root)
    payloads = {name: _load_object(root / name) for name in PLAN_MEMBERS}
    for name, payload in payloads.items():
        expected_bytes = (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if (root / name).read_bytes() != expected_bytes:
            raise ValueError(f"Plan-v2 frozen artifact bytes are noncanonical: {name}")
    validate_plan_v2_payloads(payloads, repository_root=repository_root)
    members = sorted(path.name for path in root.iterdir() if path.is_file())
    if members != sorted(PLAN_MEMBERS):
        raise ValueError("Plan-v2 directory contains undeclared members")
    return {
        "schema": "issue_53_stable_only_plan_validation_result_v2",
        "bundle_identity": BUNDLE_IDENTITY,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "planned_rollouts": 24,
        "final_seed": FINAL_SEED,
        "passed": True,
    }
