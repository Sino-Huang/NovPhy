"""Mixed-termination issue-53 successor plan derived from public v3 evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Final, Mapping

from scripts.build_issue_45_evidence import (
    CONSTRAINTS_WORKBOOK_REFERENCE,
    ROLES,
    _materialize,
    _record_for_family,
)
from scripts.cohort_v2_partition import (
    EXPOSURE_ROLES,
    CohortV2PartitionExposureManifest,
    create_cohort_v2_partition_exposure_manifest,
)
from scripts.cohort_v2_production_plans import ROLE_ORDER, _assignments
from scripts.cohort_v2_production_plans_v3 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V3,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V3,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V3,
    validate_plan_v3_evidence,
)
from scripts.cohort_v2_scenarios import (
    ScenarioInventoryEntry,
    create_scenario_inventory_entry,
)
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    create_final_evaluation_workflow_access_manifest,
)


ROOT: Final = Path(__file__).resolve().parents[1]
PLAN_VERSION: Final = 4
COLLECTION_SCHEMA: Final = "cohort_v2_production_collection_plan_v4"
COLLECTION_IDENTITY: Final = (
    "cohort-v2-production-collection-plan-v4:issue-53:mixed-termination"
)
PARAMETER_SCHEMA: Final = "cohort_v2_production_parameter_plan_v4"
PARAMETER_IDENTITY: Final = (
    "cohort-v2-production-parameter-plan-v4:issue-53:mixed-termination"
)
BUNDLE_SCHEMA: Final = "issue_53_cohort_v2_production_plans_bundle_v4"
BUNDLE_IDENTITY: Final = "issue-53-cohort-v2-production-plans-bundle-v4:mixed-termination"
EVIDENCE_SCHEMA: Final = "issue_53_mixed_termination_correction_evidence_v4"
EVIDENCE_IDENTITY: Final = (
    "issue-53-mixed-termination-correction-evidence-v4:public-non-final"
)
INVENTORY_SCHEMA: Final = "cohort_v2_production_scenario_inventory_v4"
INVENTORY_IDENTITY: Final = "cohort-v2-production-scenario-inventory-v4:issue-53"
SEALED_AUTHORITY_IDENTITY: Final = "issue-53-plan-v4-sealed-final-authority:seed-4504"
SEALED_AUTHORITY_REFERENCE: Final = SEALED_AUTHORITY_IDENTITY
WORKFLOW_IDENTITY: Final = "central-v2-final-evaluation-workflow-v4:issue-53"
WORKFLOW_OPERATOR_IDENTITY: Final = (
    "novphy-operator-v4:issue-53-mixed-termination-final-evaluation-custodian"
)
WORKFLOW_FROZEN_AT: Final = "2026-08-24T00:00:00Z"
FINAL_SEED: Final = 4504

DEFAULT_PLAN_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-plan-v4"
DEFAULT_SEALED_AUTHORITY_ROOT: Final = (
    ROOT / ".local-artifacts/issue-53-plan-v4-sealed-final-authority"
)
FAILED_PUBLIC_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-stable-only-v3"

FAIL_INTERVENTIONS: Final = frozenset(
    {"central-no-contact-miss", "central-persistent-support"}
)
FAIL_ROLES: Final = frozenset({"training", "model_selection"})

PLAN_MEMBERS: Final = (
    "bundle-manifest.json",
    "collection-plan.json",
    "final-evaluation-workflow-access-manifest.json",
    "final-evaluation.sealed-projection.json",
    "mixed-termination-correction-evidence.json",
    "partition-exposure-manifest.json",
    "production-parameter-plan.json",
    "scenario-inventory.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Plan-v4 JSON artifact must be an object: {path}")
    return value


def assignment_termination_expectations(
    role: str, intervention_ids: list[str]
) -> dict[str, str]:
    return {
        intervention_id: (
            "level_fail"
            if role in FAIL_ROLES and intervention_id in FAIL_INTERVENTIONS
            else "stable_entered"
        )
        for intervention_id in intervention_ids
    }


def derive_mixed_termination_evidence(
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    accounting_path = (
        repository_root
        / "data/runtime_evidence/issue-53-stable-only-v3/production-attempt-accounting.json"
    )
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
        raise ValueError("Plan-v4 requires 18 public non-final v3 outcomes")
    observed = {
        (item["exposure_role"], item["intervention_id"]): item["terminal_reason"]
        for item in non_final
    }
    intervention_ids = [
        "central-no-contact-miss",
        "central-collision",
        "central-persistent-support",
        "central-support-change",
        "central-destruction",
        "central-stability-transition",
    ]
    expected_non_final = {
        (role, intervention_id): termination
        for role in ROLE_ORDER[:3]
        for intervention_id, termination in assignment_termination_expectations(
            role, intervention_ids
        ).items()
    }
    if observed != expected_non_final:
        raise ValueError("Plan-v4 public non-final terminations are not the exact mixed map")
    return {
        "schema": EVIDENCE_SCHEMA,
        "identity": EVIDENCE_IDENTITY,
        "exposure_boundary": "public_non_final_only",
        "source_public_accounting_path": accounting_path.relative_to(
            repository_root
        ).as_posix(),
        "non_final_attempts": non_final,
        "observed_non_final_termination_counts": {
            "level_fail": 4,
            "stable_entered": 14,
        },
        "reviewed_or_sealed_final_outcomes_used": False,
        "decision": {
            "production_scope": "mixed_termination",
            "termination_quotas": {
                "level_clear": 0,
                "level_fail": 4,
                "stable_entered": 20,
            },
            "assignment_rule": (
                "expect level_fail only for training/model_selection x "
                "central-no-contact-miss/central-persistent-support; expect "
                "stable_entered for every other role/intervention slot"
            ),
            "final_projection_rule": (
                "prospectively transfer the six stable calibration expectations to "
                "the fresh type010102 final lineage without consulting v3 final outcomes"
            ),
            "fresh_final_seed": FINAL_SEED,
        },
    }


def _materialize_seed_4504(output_root: Path) -> tuple[Any, Any, Any]:
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


def _inventory_entries(
    final_scenario: Any,
    source_partition: CohortV2PartitionExposureManifest,
) -> tuple[ScenarioInventoryEntry, ...]:
    entries = []
    for source in source_partition.entries:
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
        "inventory_version": 4,
        "supersedes_inventory_identity": (
            "cohort-v2-production-scenario-inventory-v2:issue-53:stable-only"
        ),
        "entries": [entry.to_dict() for entry in entries],
    }


def _successor_plans(
    v3_collection: Mapping[str, Any],
    v3_parameters: Mapping[str, Any],
    partition: CohortV2PartitionExposureManifest,
) -> dict[str, dict[str, Any]]:
    correction_record = {
        "source_record_id": "issue_53_public_non_final_mixed_termination",
        "artifact_path": (
            "data/runtime_evidence/issue-53-plan-v4/"
            "mixed-termination-correction-evidence.json"
        ),
        "artifact_identity": EVIDENCE_IDENTITY,
        "record_ids": [EVIDENCE_IDENTITY],
        "exposure_boundary": "no_final_evaluation_outcomes",
    }
    collection = deepcopy(v3_collection)
    collection.update(
        {
            "schema": COLLECTION_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": COLLECTION_IDENTITY,
            "supersedes_plan_identity": COLLECTION_IDENTITY_V3,
            "production_scope": "mixed_termination",
            "termination_expectation_scope": "assignment",
        }
    )
    collection.pop("execution_correction", None)
    collection["authority"] = {
        **collection["authority"],
        "partition_manifest_identity": partition.identity,
        "correction_evidence_identity": EVIDENCE_IDENTITY,
        "failed_collection_plan_identity": COLLECTION_IDENTITY_V3,
    }
    intervention_ids = [item["id"] for item in collection["interventions"]]
    collection["assignments"] = _assignments(partition.to_dict(), intervention_ids)
    for assignment in collection["assignments"]:
        assignment["termination_expectations"] = assignment_termination_expectations(
            assignment["exposure_role"], intervention_ids
        )
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
        "required_quota_classes": ["level_fail", "stable_entered"],
        "non_quota_bearing_production_classes": ["level_clear"],
        "rollout_ceiling_disposition": (
            "allowed_safety_termination_not_quota_bearing"
        ),
        "evidence_id": "termination_quotas",
    }
    collection["quotas"]["termination_class"] = {
        "level_clear": {"quota": 0, "evidence_id": "termination_quotas"},
        "level_fail": {"quota": 4, "evidence_id": "termination_quotas"},
        "stable_entered": {"quota": 20, "evidence_id": "termination_quotas"},
    }
    collection["source_records"][
        "issue_53_public_non_final_mixed_termination"
    ] = correction_record
    for evidence in collection["evidence"].values():
        evidence["plan_version"] = PLAN_VERSION
    collection["evidence"]["termination_quotas"] = {
        "plan_version": PLAN_VERSION,
        "source_record_ids": ["issue_53_public_non_final_mixed_termination"],
        "analysis_method": (
            "Project the exact public non-final v3 role/intervention termination map "
            "and transfer only calibration expectations to the fresh type010102 final lineage."
        ),
        "observed_range_or_uncertainty": (
            "Public v3 contains 18 accepted non-final rollouts: 4 level_fail and 14 "
            "stable_entered; seed 4504 remains prospective."
        ),
        "decision_rule": (
            "Require 4 assignment-bound level_fail and 20 stable_entered terminations; "
            "do not consult sealed v3 outcomes."
        ),
        "rationale": (
            "This retains all actions and non-final lineages while matching demonstrated "
            "engine termination semantics without post-hoc final tuning."
        ),
    }

    parameters = deepcopy(v3_parameters)
    parameters.update(
        {
            "schema": PARAMETER_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": PARAMETER_IDENTITY,
            "supersedes_plan_identity": PARAMETER_IDENTITY_V3,
            "production_scope": "mixed_termination",
        }
    )
    parameters.pop("execution_correction", None)
    parameters["authority"] = {
        **parameters["authority"],
        "collection_plan_identity": COLLECTION_IDENTITY,
        "correction_evidence_identity": EVIDENCE_IDENTITY,
        "failed_parameter_plan_identity": PARAMETER_IDENTITY_V3,
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
            "value": ["level_fail", "stable_entered"],
            "unit": "termination_reason",
            "evidence_id": "termination_vocabulary",
        },
        "non_quota_bearing_production_classes": {
            "value": ["level_clear"],
            "unit": "termination_reason",
            "evidence_id": "termination_vocabulary",
        },
    }
    parameters["source_records"][
        "issue_53_public_non_final_mixed_termination"
    ] = correction_record
    for evidence in parameters["evidence"].values():
        evidence["plan_version"] = PLAN_VERSION
    parameters["evidence"]["termination_vocabulary"] = {
        "plan_version": PLAN_VERSION,
        "source_record_ids": ["issue_53_public_non_final_mixed_termination"],
        "analysis_method": (
            "Separate supported pilot capability from mixed production quotas."
        ),
        "observed_range_or_uncertainty": (
            "Public non-final v3 evidence demonstrates both level_fail and stable_entered."
        ),
        "decision_rule": (
            "Make level_fail and stable_entered quota-bearing; keep level_clear supported "
            "but non-quota-bearing."
        ),
        "rationale": "The production contract follows the demonstrated public role/action map.",
    }
    return {
        "collection-plan.json": collection,
        "production-parameter-plan.json": parameters,
    }


def build_plan_v4_payloads(final_scenario: Any) -> dict[str, dict[str, Any]]:
    validate_plan_v3_evidence(PLAN_ROOT_V3)
    v3_collection = _load_object(PLAN_ROOT_V3 / "collection-plan.json")
    v3_parameters = _load_object(PLAN_ROOT_V3 / "production-parameter-plan.json")
    source_partition = CohortV2PartitionExposureManifest.from_dict(
        _load_object(PLAN_ROOT_V3 / "partition-exposure-manifest.json")
    )
    entries = _inventory_entries(final_scenario, source_partition)
    inventory = _inventory(entries)
    partition = create_cohort_v2_partition_exposure_manifest(
        partition_version=4,
        source_inventory_identity=INVENTORY_IDENTITY,
        source_inventory_review_url="https://github.com/Sino-Huang/NovPhy/issues/53",
        inventory_entries=entries,
        lineage_quotas={role: 1 for role in EXPOSURE_ROLES},
    )
    final_entry = entries[-1]
    workflow = create_final_evaluation_workflow_access_manifest(
        partition,
        workflow_version=4,
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
    plans = _successor_plans(v3_collection, v3_parameters, partition)
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "identity": BUNDLE_IDENTITY,
        "github_issue": 53,
        "supersedes_collection_plan_identity": COLLECTION_IDENTITY_V3,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "scenario_inventory_identity": INVENTORY_IDENTITY,
        "partition_manifest_identity": partition.identity,
        "final_evaluation_workflow_access_manifest_identity": workflow.identity,
        "sealed_final_authority_identity": SEALED_AUTHORITY_IDENTITY,
        "generation_seed": FINAL_SEED,
        "artifacts": [name for name in PLAN_MEMBERS if name != "bundle-manifest.json"],
        "passed": True,
    }
    return {
        **plans,
        "mixed-termination-correction-evidence.json": (
            derive_mixed_termination_evidence(ROOT)
        ),
        "scenario-inventory.json": inventory,
        "partition-exposure-manifest.json": partition.to_dict(),
        "final-evaluation.sealed-projection.json": final_entry.to_dict(),
        "final-evaluation-workflow-access-manifest.json": workflow.to_dict(),
        "bundle-manifest.json": bundle,
    }


def validate_plan_v4_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    repository_root: Path = ROOT,
) -> None:
    repository_root = Path(repository_root).resolve()
    if set(payloads) != set(PLAN_MEMBERS):
        raise ValueError("Plan-v4 immutable bundle membership is incomplete")
    collection = payloads["collection-plan.json"]
    parameters = payloads["production-parameter-plan.json"]
    evidence = payloads["mixed-termination-correction-evidence.json"]
    partition = CohortV2PartitionExposureManifest.from_dict(
        payloads["partition-exposure-manifest.json"]
    )
    final_entry = ScenarioInventoryEntry.from_dict(
        payloads["final-evaluation.sealed-projection.json"]
    )
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
        payloads["final-evaluation-workflow-access-manifest.json"]
    )
    if evidence != derive_mixed_termination_evidence(repository_root):
        raise ValueError("Plan-v4 mixed termination evidence is stale")
    if (
        collection.get("schema") != COLLECTION_SCHEMA
        or collection.get("identity") != COLLECTION_IDENTITY
        or collection.get("plan_version") != PLAN_VERSION
        or collection.get("termination_expectation_scope") != "assignment"
        or parameters.get("schema") != PARAMETER_SCHEMA
        or parameters.get("identity") != PARAMETER_IDENTITY
        or parameters.get("plan_version") != PLAN_VERSION
        or parameters.get("authority", {}).get("collection_plan_identity")
        != COLLECTION_IDENTITY
    ):
        raise ValueError("Plan-v4 identities, schemas, or expectation scope are stale")
    assignments = collection.get("assignments", [])
    intervention_ids = [item["id"] for item in collection.get("interventions", [])]
    if (
        [item.get("exposure_role") for item in assignments] != list(ROLE_ORDER)
        or len(intervention_ids) != 6
        or any(
            item.get("termination_expectations")
            != assignment_termination_expectations(
                item["exposure_role"], intervention_ids
            )
            for item in assignments
        )
    ):
        raise ValueError("Plan-v4 assignment termination map is not exact")
    termination_counts: dict[str, int] = {}
    for assignment in assignments:
        for termination in assignment["termination_expectations"].values():
            termination_counts[termination] = termination_counts.get(termination, 0) + 1
    if termination_counts != {"level_fail": 4, "stable_entered": 20}:
        raise ValueError("Plan-v4 assignment termination counts are stale")
    if collection.get("quotas", {}).get("termination_class") != {
        "level_clear": {"quota": 0, "evidence_id": "termination_quotas"},
        "level_fail": {"quota": 4, "evidence_id": "termination_quotas"},
        "stable_entered": {"quota": 20, "evidence_id": "termination_quotas"},
    }:
        raise ValueError("Plan-v4 mixed termination quotas are stale")
    inventory = payloads["scenario-inventory.json"]
    entries = tuple(ScenarioInventoryEntry.from_dict(item) for item in inventory["entries"])
    if (
        inventory.get("schema") != INVENTORY_SCHEMA
        or inventory.get("identity") != INVENTORY_IDENTITY
        or len(entries) != 4
        or entries[-1] != final_entry
    ):
        raise ValueError("Plan-v4 scenario inventory is stale")
    old_partition = CohortV2PartitionExposureManifest.from_dict(
        _load_object(
            repository_root
            / "data/runtime_evidence/issue-53-plan-v3/partition-exposure-manifest.json"
        )
    )
    if tuple(item.to_dict() for item in partition.entries[:3]) != tuple(
        item.to_dict() for item in old_partition.entries[:3]
    ):
        raise ValueError("Plan-v4 changed a non-final lineage")
    old_final_values = {
        value
        for item in old_partition.entries
        for value in (
            item.scenario_manifest_identity,
            item.scenario_lineage_identity,
            item.level_instance_identity,
        )
    }
    if (
        final_entry.scenario_manifest_identity in old_final_values
        or final_entry.scenario_lineage_identity in old_final_values
        or final_entry.level_instance_identity in old_final_values
        or "%3A4504%3A" not in final_entry.level_instance_identity
    ):
        raise ValueError("Plan-v4 final seed is not a fresh seed-4504 lineage")
    if (
        workflow.authorization_state != "pending"
        or workflow.partition_identity != partition.identity
        or workflow.authorized_artifacts[0].artifact_identity
        != final_entry.scenario_manifest_identity
    ):
        raise ValueError("Plan-v4 final workflow or projection is stale")
    if evidence.get("reviewed_or_sealed_final_outcomes_used") is not False:
        raise ValueError("Plan-v4 derivation crossed the final outcome boundary")
    v3_collection = _load_object(
        repository_root / "data/runtime_evidence/issue-53-plan-v3/collection-plan.json"
    )
    v3_parameters = _load_object(
        repository_root
        / "data/runtime_evidence/issue-53-plan-v3/production-parameter-plan.json"
    )
    expected_plans = _successor_plans(v3_collection, v3_parameters, partition)
    if (
        collection != expected_plans["collection-plan.json"]
        or parameters != expected_plans["production-parameter-plan.json"]
    ):
        raise ValueError("Plan-v4 differs from its exact successor derivation")


def validate_plan_v4_evidence(
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
            raise ValueError(f"Plan-v4 frozen artifact bytes are noncanonical: {name}")
    validate_plan_v4_payloads(payloads, repository_root=repository_root)
    if sorted(path.name for path in root.iterdir() if path.is_file()) != sorted(
        PLAN_MEMBERS
    ):
        raise ValueError("Plan-v4 directory contains undeclared members")
    return {
        "schema": "issue_53_mixed_termination_plan_validation_result_v4",
        "bundle_identity": BUNDLE_IDENTITY,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "planned_rollouts": 24,
        "termination_quotas": {
            "level_clear": 0,
            "level_fail": 4,
            "stable_entered": 20,
        },
        "final_seed": FINAL_SEED,
        "passed": True,
    }
