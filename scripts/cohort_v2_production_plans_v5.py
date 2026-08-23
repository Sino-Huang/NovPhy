"""Workflow-time successor for the issue-53 mixed-termination plan."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Final, Mapping

from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_production_plans_v4 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V4,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V4,
    FINAL_SEED,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V4,
    SEALED_AUTHORITY_IDENTITY,
    validate_plan_v4_evidence,
)
from scripts.cohort_v2_scenarios import ScenarioInventoryEntry
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    create_final_evaluation_workflow_access_manifest,
)


ROOT: Final = Path(__file__).resolve().parents[1]
PLAN_VERSION: Final = 5
COLLECTION_SCHEMA: Final = "cohort_v2_production_collection_plan_v5"
COLLECTION_IDENTITY: Final = (
    "cohort-v2-production-collection-plan-v5:issue-53:mixed-termination:workflow-time"
)
PARAMETER_SCHEMA: Final = "cohort_v2_production_parameter_plan_v5"
PARAMETER_IDENTITY: Final = (
    "cohort-v2-production-parameter-plan-v5:issue-53:mixed-termination:workflow-time"
)
BUNDLE_SCHEMA: Final = "issue_53_cohort_v2_production_plans_bundle_v5"
BUNDLE_IDENTITY: Final = "issue-53-cohort-v2-production-plans-bundle-v5:workflow-time"
CORRECTION_SCHEMA: Final = "issue_53_workflow_freeze_time_correction_evidence_v5"
CORRECTION_IDENTITY: Final = "issue-53-workflow-freeze-time-correction-v5:utc"
WORKFLOW_IDENTITY: Final = "central-v2-final-evaluation-workflow-v5:issue-53"
WORKFLOW_OPERATOR_IDENTITY: Final = (
    "novphy-operator-v5:issue-53-mixed-termination-final-evaluation-custodian"
)
FAILED_AUTHORIZATION_OBSERVED_AT: Final = "2026-08-23T14:45:54Z"
WORKFLOW_FROZEN_AT: Final = "2026-08-23T14:46:00Z"
DEFAULT_PLAN_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-plan-v5"

PLAN_MEMBERS: Final = (
    "bundle-manifest.json",
    "collection-plan.json",
    "final-evaluation-workflow-access-manifest.json",
    "final-evaluation.sealed-projection.json",
    "mixed-termination-correction-evidence.json",
    "partition-exposure-manifest.json",
    "production-parameter-plan.json",
    "scenario-inventory.json",
    "workflow-time-correction-evidence.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Plan-v5 JSON artifact must be an object: {path}")
    return value


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)


def derive_workflow_time_correction_evidence(
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    workflow_path = (
        root
        / "data/runtime_evidence/issue-53-plan-v4/"
        "final-evaluation-workflow-access-manifest.json"
    )
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
        _load_object(workflow_path)
    )
    if (
        workflow.frozen_at != "2026-08-24T00:00:00Z"
        or _utc(FAILED_AUTHORIZATION_OBSERVED_AT) >= _utc(workflow.frozen_at)
    ):
        raise ValueError("Plan-v5 source workflow does not reproduce the UTC freeze defect")
    return {
        "schema": CORRECTION_SCHEMA,
        "identity": CORRECTION_IDENTITY,
        "source_workflow_path": workflow_path.relative_to(root).as_posix(),
        "source_workflow_identity": workflow.identity,
        "source_frozen_at": workflow.frozen_at,
        "diagnostic_utc_observed_at": FAILED_AUTHORIZATION_OBSERVED_AT,
        "failure": "Final-evaluation authorization predates workflow freeze",
        "failure_phase": "pre_authorization_before_final_access",
        "production_attempts_started": 0,
        "final_authority_accessed": False,
        "scientific_plan_changed": False,
        "corrected_frozen_at": WORKFLOW_FROZEN_AT,
    }


def _successor_plans(
    v4_collection: Mapping[str, Any],
    v4_parameters: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    correction_record = {
        "source_record_id": "issue_53_workflow_freeze_time_correction",
        "artifact_path": (
            "data/runtime_evidence/issue-53-plan-v5/"
            "workflow-time-correction-evidence.json"
        ),
        "artifact_identity": CORRECTION_IDENTITY,
        "record_ids": [CORRECTION_IDENTITY],
        "exposure_boundary": "no_final_evaluation_outcomes",
    }
    collection = deepcopy(v4_collection)
    collection.update(
        {
            "schema": COLLECTION_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": COLLECTION_IDENTITY,
            "supersedes_plan_identity": COLLECTION_IDENTITY_V4,
            "workflow_correction": "utc_freeze_precedes_authorization",
        }
    )
    collection["authority"] = {
        **collection["authority"],
        "workflow_correction_evidence_identity": CORRECTION_IDENTITY,
        "failed_collection_plan_identity": COLLECTION_IDENTITY_V4,
    }
    collection["source_records"][
        "issue_53_workflow_freeze_time_correction"
    ] = correction_record
    for item in collection["evidence"].values():
        item["plan_version"] = PLAN_VERSION

    parameters = deepcopy(v4_parameters)
    parameters.update(
        {
            "schema": PARAMETER_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": PARAMETER_IDENTITY,
            "supersedes_plan_identity": PARAMETER_IDENTITY_V4,
            "workflow_correction": "utc_freeze_precedes_authorization",
        }
    )
    parameters["authority"] = {
        **parameters["authority"],
        "collection_plan_identity": COLLECTION_IDENTITY,
        "workflow_correction_evidence_identity": CORRECTION_IDENTITY,
        "failed_parameter_plan_identity": PARAMETER_IDENTITY_V4,
    }
    parameters["source_records"][
        "issue_53_workflow_freeze_time_correction"
    ] = correction_record
    for item in parameters["evidence"].values():
        item["plan_version"] = PLAN_VERSION
    return {
        "collection-plan.json": collection,
        "production-parameter-plan.json": parameters,
    }


def build_plan_v5_payloads(
    repository_root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    root = Path(repository_root).resolve()
    validate_plan_v4_evidence(
        root / "data/runtime_evidence/issue-53-plan-v4", repository_root=root
    )
    copied = {
        name: _load_object(root / "data/runtime_evidence/issue-53-plan-v4" / name)
        for name in (
            "collection-plan.json",
            "production-parameter-plan.json",
            "mixed-termination-correction-evidence.json",
            "scenario-inventory.json",
            "partition-exposure-manifest.json",
            "final-evaluation.sealed-projection.json",
        )
    }
    partition = CohortV2PartitionExposureManifest.from_dict(
        copied["partition-exposure-manifest.json"]
    )
    final_entry = ScenarioInventoryEntry.from_dict(
        copied["final-evaluation.sealed-projection.json"]
    )
    workflow = create_final_evaluation_workflow_access_manifest(
        partition,
        workflow_version=5,
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
    plans = _successor_plans(
        copied["collection-plan.json"], copied["production-parameter-plan.json"]
    )
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "identity": BUNDLE_IDENTITY,
        "github_issue": 53,
        "supersedes_collection_plan_identity": COLLECTION_IDENTITY_V4,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "partition_manifest_identity": partition.identity,
        "final_evaluation_workflow_access_manifest_identity": workflow.identity,
        "sealed_final_authority_identity": SEALED_AUTHORITY_IDENTITY,
        "generation_seed": FINAL_SEED,
        "artifacts": [name for name in PLAN_MEMBERS if name != "bundle-manifest.json"],
        "passed": True,
    }
    return {
        **plans,
        "workflow-time-correction-evidence.json": (
            derive_workflow_time_correction_evidence(root)
        ),
        "mixed-termination-correction-evidence.json": copied[
            "mixed-termination-correction-evidence.json"
        ],
        "scenario-inventory.json": copied["scenario-inventory.json"],
        "partition-exposure-manifest.json": partition.to_dict(),
        "final-evaluation.sealed-projection.json": final_entry.to_dict(),
        "final-evaluation-workflow-access-manifest.json": workflow.to_dict(),
        "bundle-manifest.json": bundle,
    }


def validate_plan_v5_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    repository_root: Path = ROOT,
) -> None:
    if set(payloads) != set(PLAN_MEMBERS):
        raise ValueError("Plan-v5 immutable bundle membership is incomplete")
    collection = payloads["collection-plan.json"]
    parameters = payloads["production-parameter-plan.json"]
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
        payloads["final-evaluation-workflow-access-manifest.json"]
    )
    correction = payloads["workflow-time-correction-evidence.json"]
    if correction != derive_workflow_time_correction_evidence(repository_root):
        raise ValueError("Plan-v5 workflow-time correction evidence is stale")
    if (
        collection.get("schema") != COLLECTION_SCHEMA
        or collection.get("identity") != COLLECTION_IDENTITY
        or collection.get("plan_version") != PLAN_VERSION
        or parameters.get("schema") != PARAMETER_SCHEMA
        or parameters.get("identity") != PARAMETER_IDENTITY
        or parameters.get("plan_version") != PLAN_VERSION
        or parameters.get("authority", {}).get("collection_plan_identity")
        != COLLECTION_IDENTITY
    ):
        raise ValueError("Plan-v5 identities or schemas are stale")
    if (
        workflow.workflow_version != 5
        or workflow.workflow_identity != WORKFLOW_IDENTITY
        or workflow.frozen_at != WORKFLOW_FROZEN_AT
        or workflow.authorization_state != "pending"
        or _utc(workflow.frozen_at) <= _utc(FAILED_AUTHORIZATION_OBSERVED_AT)
    ):
        raise ValueError("Plan-v5 corrected pending workflow is stale")
    v4_collection = _load_object(
        Path(repository_root)
        / "data/runtime_evidence/issue-53-plan-v4/collection-plan.json"
    )
    if (
        collection["assignments"] != v4_collection["assignments"]
        or collection["interventions"] != v4_collection["interventions"]
        or collection["quotas"] != v4_collection["quotas"]
    ):
        raise ValueError("Plan-v5 changed the mixed scientific plan")
    expected = build_plan_v5_payloads(repository_root)
    if dict(payloads) != expected:
        raise ValueError("Plan-v5 differs from its exact workflow successor derivation")


def validate_plan_v5_evidence(
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
            raise ValueError(f"Plan-v5 frozen artifact bytes are noncanonical: {name}")
    validate_plan_v5_payloads(payloads, repository_root=repository_root)
    if sorted(path.name for path in root.iterdir() if path.is_file()) != sorted(
        PLAN_MEMBERS
    ):
        raise ValueError("Plan-v5 directory contains undeclared members")
    return {
        "schema": "issue_53_workflow_time_successor_plan_validation_result_v5",
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
        "workflow_frozen_at": WORKFLOW_FROZEN_AT,
        "passed": True,
    }

