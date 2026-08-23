"""Successor issue-53 plan after the pre-shot anchor-order failure."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Final, Mapping

from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_production_plans_v2 import (
    COLLECTION_IDENTITY as COLLECTION_IDENTITY_V2,
    DEFAULT_PLAN_ROOT as PLAN_ROOT_V2,
    FINAL_SEED,
    PARAMETER_IDENTITY as PARAMETER_IDENTITY_V2,
    SEALED_AUTHORITY_IDENTITY,
    validate_plan_v2_evidence,
)
from scripts.cohort_v2_scenarios import ScenarioInventoryEntry
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    create_final_evaluation_workflow_access_manifest,
)


ROOT: Final = Path(__file__).resolve().parents[1]
PLAN_VERSION: Final = 3
COLLECTION_SCHEMA: Final = "cohort_v2_production_collection_plan_v3"
COLLECTION_IDENTITY: Final = (
    "cohort-v2-production-collection-plan-v3:issue-53:stable-only:anchor-order-correction"
)
PARAMETER_SCHEMA: Final = "cohort_v2_production_parameter_plan_v3"
PARAMETER_IDENTITY: Final = (
    "cohort-v2-production-parameter-plan-v3:issue-53:stable-only:anchor-order-correction"
)
BUNDLE_SCHEMA: Final = "issue_53_cohort_v2_production_plans_bundle_v3"
BUNDLE_IDENTITY: Final = (
    "issue-53-cohort-v2-production-plans-bundle-v3:anchor-order-correction"
)
CORRECTION_SCHEMA: Final = "issue_53_executor_anchor_order_correction_evidence_v3"
CORRECTION_IDENTITY: Final = (
    "issue-53-executor-anchor-order-correction-evidence-v3:pre-shot"
)
WORKFLOW_IDENTITY: Final = (
    "central-v2-final-evaluation-workflow-v3:issue-53:anchor-order-correction"
)
WORKFLOW_OPERATOR_IDENTITY: Final = (
    "novphy-operator-v3:issue-53-stable-only-final-evaluation-custodian"
)
WORKFLOW_FROZEN_AT: Final = "2026-08-23T12:00:00Z"

DEFAULT_PLAN_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-plan-v3"
FAILED_PUBLIC_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-stable-only-v2"

PLAN_MEMBERS: Final = (
    "bundle-manifest.json",
    "collection-plan.json",
    "executor-correction-evidence.json",
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
        raise ValueError(f"Plan-v3 JSON artifact must be an object: {path}")
    return value


def derive_executor_correction_evidence(
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    accounting_path = (
        repository_root
        / "data/runtime_evidence/issue-53-stable-only-v2/production-attempt-accounting.json"
    )
    accounting = _load_object(accounting_path)
    non_final = [
        item
        for item in accounting["attempt_ledger"]
        if item["exposure_role"] != "final_evaluation"
    ]
    failures = {
        (item.get("failure_code"), item.get("reason")) for item in non_final
    }
    if (
        len(non_final) != 18
        or failures
        != {(
            "collection_runtime_error",
            "drag_start is required for slingshot_relative actions",
        )}
        or any(item.get("terminal_reason") is not None for item in non_final)
    ):
        raise ValueError("Plan-v3 requires the exact public pre-shot v2 failure evidence")
    return {
        "schema": CORRECTION_SCHEMA,
        "identity": CORRECTION_IDENTITY,
        "exposure_boundary": "public_non_final_only",
        "source_public_accounting_path": accounting_path.relative_to(
            repository_root
        ).as_posix(),
        "failed_attempt_count": 18,
        "failure_code": "collection_runtime_error",
        "failure_reason": "drag_start is required for slingshot_relative actions",
        "failure_phase": "pre_shot_action_normalization",
        "physics_outcomes_observed": False,
        "final_evaluation_outcomes_used": False,
        "regression_test": (
            "tests.test_collect_rollouts.CollectRolloutsTest."
            "test_collect_fresh_engine_rollouts_anchors_plan_relative_actions_without_drag_start"
        ),
        "executor_contract": (
            "readiness anchors live-relative plan actions before socket-command normalization"
        ),
        "successor_decision": {
            "reuse_seed_4503": True,
            "rationale": (
                "The failed determination emitted no shot and observed no physics or "
                "termination outcome; retaining the frozen final lineage avoids tuning."
            ),
            "new_attempt_identity_required": True,
            "retry_within_plan_v2": False,
        },
    }


def _successor_plans(
    v2_collection: Mapping[str, Any],
    v2_parameters: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    correction_record = {
        "source_record_id": "issue_53_executor_anchor_order_correction",
        "artifact_path": (
            "data/runtime_evidence/issue-53-plan-v3/executor-correction-evidence.json"
        ),
        "artifact_identity": CORRECTION_IDENTITY,
        "record_ids": [CORRECTION_IDENTITY],
        "exposure_boundary": "no_final_evaluation_outcomes",
    }
    collection = deepcopy(v2_collection)
    collection.update(
        {
            "schema": COLLECTION_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": COLLECTION_IDENTITY,
            "supersedes_plan_identity": COLLECTION_IDENTITY_V2,
            "execution_correction": "readiness_anchor_before_action_normalization",
        }
    )
    collection["authority"] = {
        **collection["authority"],
        "correction_evidence_identity": CORRECTION_IDENTITY,
        "failed_collection_plan_identity": COLLECTION_IDENTITY_V2,
    }
    collection["source_records"][
        "issue_53_executor_anchor_order_correction"
    ] = correction_record
    for item in collection["evidence"].values():
        item["plan_version"] = PLAN_VERSION
    collection["evidence"]["attempt_policy"] = {
        "plan_version": PLAN_VERSION,
        "source_record_ids": ["issue_53_executor_anchor_order_correction"],
        "analysis_method": (
            "Project the exact public pre-shot v2 failure and the executor regression contract."
        ),
        "observed_range_or_uncertainty": (
            "All 18 public non-final v2 attempts failed before a shot with one action-"
            "normalization error; no physics or termination outcome was observed."
        ),
        "decision_rule": (
            "Create new attempt identities under plan v3, retain one attempt per slot, "
            "and require readiness to anchor live-relative actions before normalization."
        ),
        "rationale": (
            "The successor is a new determination, not a retry within consumed plan v2."
        ),
    }

    parameters = deepcopy(v2_parameters)
    parameters.update(
        {
            "schema": PARAMETER_SCHEMA,
            "plan_version": PLAN_VERSION,
            "identity": PARAMETER_IDENTITY,
            "supersedes_plan_identity": PARAMETER_IDENTITY_V2,
            "execution_correction": "readiness_anchor_before_action_normalization",
        }
    )
    parameters["authority"] = {
        **parameters["authority"],
        "collection_plan_identity": COLLECTION_IDENTITY,
        "correction_evidence_identity": CORRECTION_IDENTITY,
        "failed_parameter_plan_identity": PARAMETER_IDENTITY_V2,
    }
    parameters["source_records"][
        "issue_53_executor_anchor_order_correction"
    ] = correction_record
    for item in parameters["evidence"].values():
        item["plan_version"] = PLAN_VERSION
    parameters["evidence"]["retry_policy"] = {
        "plan_version": PLAN_VERSION,
        "source_record_ids": ["issue_53_executor_anchor_order_correction"],
        "analysis_method": (
            "Treat the consumed v2 determination as immutable failure evidence."
        ),
        "observed_range_or_uncertainty": (
            "No v2 shot was sent; all failures occurred before physical collection."
        ),
        "decision_rule": (
            "Retain zero retries within plan v3 and issue new attempt identities."
        ),
        "rationale": (
            "Fresh identities preserve one-attempt accounting without resampling outcomes."
        ),
    }
    return {
        "collection-plan.json": collection,
        "production-parameter-plan.json": parameters,
    }


def build_plan_v3_payloads(
    repository_root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    repository_root = Path(repository_root).resolve()
    validate_plan_v2_evidence(
        repository_root / "data/runtime_evidence/issue-53-plan-v2",
        repository_root=repository_root,
    )
    v2 = {
        name: _load_object(
            repository_root / "data/runtime_evidence/issue-53-plan-v2" / name
        )
        for name in (
            "collection-plan.json",
            "production-parameter-plan.json",
            "plan-correction-evidence.json",
            "scenario-inventory.json",
            "partition-exposure-manifest.json",
            "final-evaluation.sealed-projection.json",
        )
    }
    partition = CohortV2PartitionExposureManifest.from_dict(
        v2["partition-exposure-manifest.json"]
    )
    final_entry = ScenarioInventoryEntry.from_dict(
        v2["final-evaluation.sealed-projection.json"]
    )
    workflow = create_final_evaluation_workflow_access_manifest(
        partition,
        workflow_version=3,
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
        v2["collection-plan.json"], v2["production-parameter-plan.json"]
    )
    evidence = derive_executor_correction_evidence(repository_root)
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "identity": BUNDLE_IDENTITY,
        "github_issue": 53,
        "supersedes_collection_plan_identity": COLLECTION_IDENTITY_V2,
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
        "executor-correction-evidence.json": evidence,
        "plan-correction-evidence.json": v2["plan-correction-evidence.json"],
        "scenario-inventory.json": v2["scenario-inventory.json"],
        "partition-exposure-manifest.json": partition.to_dict(),
        "final-evaluation.sealed-projection.json": final_entry.to_dict(),
        "final-evaluation-workflow-access-manifest.json": workflow.to_dict(),
        "bundle-manifest.json": bundle,
    }


def validate_plan_v3_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    repository_root: Path = ROOT,
) -> None:
    if set(payloads) != set(PLAN_MEMBERS):
        raise ValueError("Plan-v3 immutable bundle membership is incomplete")
    collection = payloads["collection-plan.json"]
    parameters = payloads["production-parameter-plan.json"]
    evidence = payloads["executor-correction-evidence.json"]
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(
        payloads["final-evaluation-workflow-access-manifest.json"]
    )
    if evidence != derive_executor_correction_evidence(repository_root):
        raise ValueError("Plan-v3 executor correction evidence is stale")
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
        raise ValueError("Plan-v3 identities or schemas are stale")
    if (
        len(collection.get("assignments", [])) != 4
        or len(collection.get("interventions", [])) != 6
        or any(
            item.get("intended_termination_class") != "stable_entered"
            for item in collection["interventions"]
        )
        or collection.get("quotas", {}).get("termination_class")
        != {
            "level_clear": {"quota": 0, "evidence_id": "termination_quotas"},
            "level_fail": {"quota": 0, "evidence_id": "termination_quotas"},
            "stable_entered": {"quota": 24, "evidence_id": "termination_quotas"},
        }
    ):
        raise ValueError("Plan-v3 changed the stable-only scientific assignments")
    v2_collection = _load_object(
        Path(repository_root)
        / "data/runtime_evidence/issue-53-plan-v2/collection-plan.json"
    )
    if [item["interface_action"] for item in collection["interventions"]] != [
        item["interface_action"] for item in v2_collection["interventions"]
    ]:
        raise ValueError("Plan-v3 changed a frozen intervention action")
    if (
        workflow.authorization_state != "pending"
        or workflow.partition_identity
        != payloads["partition-exposure-manifest.json"]["identity"]
        or workflow.workflow_identity != WORKFLOW_IDENTITY
    ):
        raise ValueError("Plan-v3 pending final access workflow is stale")
    expected = build_plan_v3_payloads(repository_root)
    if dict(payloads) != expected:
        raise ValueError("Plan-v3 differs from its exact successor derivation")


def validate_plan_v3_evidence(
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
            raise ValueError(f"Plan-v3 frozen artifact bytes are noncanonical: {name}")
    validate_plan_v3_payloads(payloads, repository_root=repository_root)
    if sorted(path.name for path in root.iterdir() if path.is_file()) != sorted(
        PLAN_MEMBERS
    ):
        raise ValueError("Plan-v3 directory contains undeclared members")
    return {
        "schema": "issue_53_anchor_order_successor_plan_validation_result_v3",
        "bundle_identity": BUNDLE_IDENTITY,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "production_parameter_plan_identity": PARAMETER_IDENTITY,
        "planned_rollouts": 24,
        "final_seed": FINAL_SEED,
        "passed": True,
    }

