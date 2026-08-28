"""Prospective capacity-correct confirmatory plan and protocol for issue #15."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Mapping

from scripts.build_issue_45_evidence import (
    CONSTRAINTS_WORKBOOK_REFERENCE,
    ROLES,
    _materialize,
    _record_for_family,
)
from scripts.cohort_v2_partition import (
    CohortV2PartitionExposureManifest,
    create_cohort_v2_partition_exposure_manifest,
)
from scripts.cohort_v2_release import production_attempt_identity
from scripts.cohort_v2_scenarios import (
    ScenarioInventoryEntry,
    create_scenario_inventory_entry,
    write_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_bytes,
)
from scripts.final_evaluation_access import (
    create_final_evaluation_workflow_access_manifest,
)
from world_model.training.grid_artifacts import canonical_json_bytes


ROOT: Final = Path(__file__).resolve().parents[1]
FINAL_SEED: Final = 4505
COLLECTION_IDENTITY: Final = "issue-15-confirmatory-collection-v2:seed-4505"
FINAL_RELEASE_IDENTITY: Final = "issue-15-confirmatory-final-release-v2:seed-4505"
SEALED_BUNDLE_IDENTITY: Final = "issue-15-confirmatory-sealed-bundle-v2:seed-4505"
INVENTORY_IDENTITY: Final = "issue-15-confirmatory-inventory-v2:seed-4505"
WORKFLOW_IDENTITY: Final = "issue-15-final-evaluation-workflow-v2:seed-4505"
WORKFLOW_OPERATOR_IDENTITY: Final = "novphy-issue-15-confirmatory-v2-custodian"
WORKFLOW_FROZEN_AT: Final = "2026-08-27T06:20:00Z"
PROTOCOL_SCHEMA: Final = "cohort_v2_prospective_statistical_protocol_v2"
PROTOCOL_PREFIX: Final = "cohort-v2-prospective-statistical-protocol-v2"
DEFAULT_ROOT: Final = Path("data/runtime_evidence/issue-15-amendment-v2")
DEFAULT_AUTHORITY_ROOT: Final = Path(
    ".local-artifacts/issue-15-amendment-v2-authority"
)
CAPACITY_REPORT: Final = Path(
    "data/runtime_evidence/issue-15/capacity-integrated-calibration-summary.json"
)
BASE_PROTOCOL: Final = Path(
    "data/runtime_evidence/issue-34/cohort-v2-prospective-statistical-protocol-v1.json"
)
SOURCE_PARTITION: Final = Path(
    "data/runtime_evidence/issue-53-plan-v5/partition-exposure-manifest.json"
)
SOURCE_COLLECTION: Final = Path(
    "data/runtime_evidence/issue-53-plan-v5/collection-plan.json"
)


class Issue15AmendmentError(ValueError):
    """The prospective replacement plan or protocol is invalid."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise Issue15AmendmentError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise Issue15AmendmentError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    os.replace(temporary, target)


def materialize_final_authority(output_root: Path):
    role = replace(ROLES[3], seed=FINAL_SEED)
    workbook = ROOT / CONSTRAINTS_WORKBOOK_REFERENCE
    source, constraints, record = _record_for_family(
        ROOT, workbook.read_bytes(), role.family
    )
    materialized, scenario = _materialize(
        role, source, constraints, record, workbook, Path(output_root)
    )
    xml_path = Path(output_root) / "final-evaluation.xml"
    manifest_path = Path(output_root) / "final-evaluation.json"
    write_immutable_cohort_v2_bytes(materialized.xml_content, xml_path)
    write_cohort_v2_scenario_manifest(scenario, manifest_path)
    return scenario, manifest_path, xml_path


def _replacement_entries(final_scenario) -> tuple[ScenarioInventoryEntry, ...]:
    source = CohortV2PartitionExposureManifest.from_dict(
        _load(ROOT / SOURCE_PARTITION)
    )
    entries = []
    for item in source.entries:
        if item.exposure_role == "final_evaluation":
            continue
        entries.append(ScenarioInventoryEntry.from_dict({
            "exposure_role": item.exposure_role,
            "inventory_state": item.inventory_state,
            "scenario_manifest_identity": item.scenario_manifest_identity,
            "benchmark_condition_identity": item.benchmark_condition_identity,
            "scenario_template_identity": item.scenario_template_identity,
            "level_instance_identity": item.level_instance_identity,
            "scenario_specification_identity": item.scenario_specification_identity,
            "scenario_lineage_identity": item.scenario_lineage_identity,
            "declared_initial_engine_state_identity": (
                item.declared_initial_engine_state_identity
            ),
            "scenario_manifest_reference": item.scenario_manifest_reference,
        }))
    entries.append(create_scenario_inventory_entry(
        "final_evaluation",
        "sealed_final",
        final_scenario,
        sealed_scenario_manifest_reference=(
            "issue-15-amendment-v2-sealed-authority:seed-4505"
        ),
    ))
    return tuple(entries)


def build_plan(final_scenario) -> dict[str, dict[str, Any]]:
    entries = _replacement_entries(final_scenario)
    partition = create_cohort_v2_partition_exposure_manifest(
        partition_version=5,
        source_inventory_identity=INVENTORY_IDENTITY,
        source_inventory_review_url="https://github.com/Sino-Huang/NovPhy/issues/15",
        inventory_entries=entries,
        lineage_quotas={
            "training": 1,
            "calibration": 1,
            "model_selection": 1,
            "final_evaluation": 1,
        },
    )
    final_entry = entries[-1]
    source_collection = _load(ROOT / SOURCE_COLLECTION)
    interventions = deepcopy(source_collection["interventions"])
    intervention_ids = [item["id"] for item in interventions]
    attempt_ids = [
        production_attempt_identity(
            "final_evaluation", intervention_id, COLLECTION_IDENTITY
        )
        for intervention_id in intervention_ids
    ]
    assignment = {
        "exposure_role": "final_evaluation",
        "dataset_partition": "central-v2-confirmatory-v2-final-evaluation",
        "scenario_manifest_identity": final_entry.scenario_manifest_identity,
        "scenario_lineage_identity": final_entry.scenario_lineage_identity,
        "level_instance_identity": final_entry.level_instance_identity,
        "scenario_template_identity": final_entry.scenario_template_identity,
        "benchmark_condition_identity": final_entry.benchmark_condition_identity,
        "intervention_ids": intervention_ids,
        "termination_expectations": {
            intervention_id: "stable_entered" for intervention_id in intervention_ids
        },
    }
    collection = {
        "schema": "issue_15_confirmatory_collection_plan_v2",
        "identity": COLLECTION_IDENTITY,
        "plan_version": 2,
        "source_public_collection_identity": source_collection["identity"],
        "source_public_roles_used_for_configuration": [
            "training", "calibration", "model_selection"
        ],
        "retired_final_bundle_identity": (
            "issue-53-final-evaluation-sealed-bundle-v5:mixed-termination"
        ),
        "new_final_seed": FINAL_SEED,
        "assignments": [assignment],
        "interventions": interventions,
        "attempt_ids": attempt_ids,
        "replacement_attempts": "forbidden",
        "outcome_conditioned_design_changes": "forbidden",
    }
    workflow = create_final_evaluation_workflow_access_manifest(
        partition,
        workflow_version=2,
        workflow_identity=WORKFLOW_IDENTITY,
        operator_identity=WORKFLOW_OPERATOR_IDENTITY,
        frozen_at=WORKFLOW_FROZEN_AT,
        authorized_artifacts=[{
            "artifact_kind": "scenario_manifest",
            "artifact_identity": final_entry.scenario_manifest_identity,
            "source_scenario_lineage_identities": [
                final_entry.scenario_lineage_identity
            ],
        }],
    )
    inventory = {
        "schema": "issue_15_confirmatory_inventory_v2",
        "identity": INVENTORY_IDENTITY,
        "entries": [item.to_dict() for item in entries],
    }
    return {
        "confirmatory-plan.json": collection,
        "partition-exposure-manifest.json": partition.to_dict(),
        "scenario-inventory.json": inventory,
        "final-evaluation.sealed-projection.json": final_entry.to_dict(),
        "final-evaluation-workflow-access-manifest.json": workflow.to_dict(),
    }


def _capacity_comparisons(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    proposals = report.get("proposals_for_issue_34")
    if not isinstance(proposals, list) or len(proposals) != 2:
        raise Issue15AmendmentError("capacity calibration must freeze two budgets")
    comparisons = []
    for index, proposal in enumerate(proposals):
        comparisons.append({
            "budget": proposal["budget"],
            "fixed_complete_rollout_replicates": 6,
            "strongest_comparator_id": proposal["strongest_comparator_id"],
            "practical_effect_threshold_absolute_endpoint_error_reduction": (
                proposal["practical_effect_threshold"]
            ),
            "physical_violation_margin": proposal["physical_violation_margin"],
            "calibration_precision_half_width": proposal["precision_half_width"],
            "gain_bootstrap_seed": 20261126 + index * 2,
            "violation_bootstrap_seed": 20261127 + index * 2,
        })
    return comparisons


def build_protocol(
    plan: Mapping[str, Mapping[str, Any]],
    *,
    implementation_commit: str,
    capacity_report: Path = CAPACITY_REPORT,
) -> dict[str, Any]:
    base = _load(ROOT / BASE_PROTOCOL)
    capacity_path = Path(capacity_report)
    if not capacity_path.is_absolute():
        capacity_path = ROOT / capacity_path
    capacity = _load(capacity_path)
    if (
        capacity.get("design") != "issue-15-capacity"
        or capacity.get("disposition", {}).get("status")
        != "sufficient_evidence_to_freeze_issue_34"
    ):
        raise Issue15AmendmentError("capacity candidate calibration is not sufficient")
    protocol = deepcopy(base)
    protocol.update({
        "schema": PROTOCOL_SCHEMA,
        "protocol_version": 2,
        "artifact_identity": None,
        "implementation_commit": implementation_commit,
        "status": "frozen_before_new_final_collection",
    })
    protocol["source_bindings"] = {
        "base_protocol": {
            "path": BASE_PROTOCOL.as_posix(),
            "sha256": _sha256(ROOT / BASE_PROTOCOL),
        },
        "capacity_calibration": {
            "path": (
                capacity_path.relative_to(ROOT).as_posix()
                if capacity_path.is_relative_to(ROOT)
                else capacity_path.as_posix()
            ),
            "sha256": _sha256(capacity_path),
            "artifact_identity": capacity["artifact_identity"],
            "implementation_commit": capacity["implementation_commit"],
        },
        "confirmatory_plan_identity": plan["confirmatory-plan.json"]["identity"],
        "partition_identity": plan["partition-exposure-manifest.json"]["identity"],
        "workflow_manifest_identity": plan[
            "final-evaluation-workflow-access-manifest.json"
        ]["identity"],
        "retired_attempt_evidence": {
            "path": (
                "data/runtime_evidence/issue-15/"
                "cohort-v2-oracle-symbol-confirmatory-summary.json"
            ),
            "scientific_status": "inconclusive_not_evaluable",
            "confirmatory_reuse": "forbidden",
        },
    }
    protocol["calibration_basis"] = {
        "source": "capacity_integrated_public_non_final_calibration",
        "capacity": {"max_entities": 15, "latent_dim": 197},
        "final_outcomes_used": False,
    }
    issue_15 = protocol["experiment_matrix"][
        "confirmatory_oracle_symbol_issue_15"
    ]
    issue_15.update({
        "candidate_configuration_id": "integrated_aggregated_joint_controller",
        "candidate_source": "issue_15_capacity15_integrated_candidate",
        "comparisons": _capacity_comparisons(capacity),
    })
    issue_15["complete_rollout_secondary"]["selection_source"] = (
        "issue_15_capacity_model_selection"
    )
    protocol["experiment_matrix"]["configuration_freeze_rule"] = (
        "The issue-15 capacity-15 predictor, controller, pair grid, costs, and "
        "budget-specific comparators are immutable. Parser checkpoints, probability "
        "calibration, and decision thresholds must be source-bound using training "
        "parameters, model-selection configuration choice, and calibration thresholds "
        "before a parser receives final-evaluation data."
    )
    proposals = capacity["proposals_for_issue_34"]
    for experiment_id in (
        "learned_feature_symbol_stress_issue_16",
        "frozen_visual_symbol_stress_issue_17",
    ):
        comparisons = protocol["experiment_matrix"][experiment_id]["comparisons"]
        for comparison, proposal in zip(comparisons, proposals, strict=True):
            comparison.update({
                "budget": proposal["budget"],
                "calibration_precision_half_width": proposal["precision_half_width"],
                "physical_violation_margin": proposal["physical_violation_margin"],
                "practical_effect_threshold_absolute_endpoint_error_reduction": (
                    proposal["practical_effect_threshold"]
                ),
                "strongest_comparator_id": proposal["strongest_comparator_id"],
            })
    protocol["replicate_and_seed_policy"].update({
        "fixed_attempt_ids": plan["confirmatory-plan.json"]["attempt_ids"],
        "fixed_replicate_count": 6,
        "new_final_scenario_seed": FINAL_SEED,
        "analysis_bootstrap_master_seed": 20261126,
    })
    protocol["final_evaluation_access"] = {
        "authorization_rule": (
            "Issue #15 may authorize this workflow only after protocol v2 and its "
            "plan are committed. Every access must use the new seed-4505 authority."
        ),
        "current_manifest_authorization_state": "pending",
        "later_consumer_identity": "issue-15-oracle-symbol-confirmatory-v2",
        "operator_identity": WORKFLOW_OPERATOR_IDENTITY,
        "required_access_record_schema": "final_evaluation_workflow_access_audit_v1",
        "sealed_bundle_identity": SEALED_BUNDLE_IDENTITY,
        "workflow_identity": WORKFLOW_IDENTITY,
    }
    protocol["exposure_audit"] = {
        "capacity_training_calibration_model_selection_roles_accessed": True,
        "capacity_final_evaluation_accessed": False,
        "old_seed_4504_partition_opened": True,
        "old_seed_4504_partition_confirmatory_reuse": False,
        "new_seed_4505_scenario_outcomes_or_metrics_accessed": False,
        "new_seed_4505_rollouts_collected": False,
        "passed": True,
    }
    protocol["statistical_analysis"]["confirmatory_decision"] = (
        "Supported only if at least one declared budget has all six valid paired "
        "rollouts, candidate and comparator are within matched-compute support, the "
        "one-sided lower gain bound is at least that budget's practical-effect "
        "threshold, and the one-sided upper violation-increase bound is at most its "
        "margin. Otherwise report not_supported_by_this_experiment; that disposition "
        "does not assert that the scientific hypothesis is generally impossible or "
        "unsupported."
    )
    protocol["required_outputs"]["decision_values"] = [
        "supported",
        "not_supported_by_this_experiment",
        "not_run_due_to_provenance_abort",
    ]
    protocol["amendment_policy"] = (
        "Version 2 prospectively replaces the capacity-invalid version-1 run. The "
        "seed-4504 partition remains inconclusive provenance and cannot be reused. "
        "No changes are permitted after seed-4505 collection begins."
    )
    protocol["rerun_commands"] = [
        "python -u -m scripts.run_issue_15_amendment --dry-run",
        "python -u -m scripts.run_issue_15_amendment --freeze "
        f"--implementation-commit {implementation_commit}",
        "python -u -m scripts.run_issue_15_amendment --validate",
    ]
    identity_payload = {**protocol, "artifact_identity": None}
    protocol["artifact_identity"] = (
        f"{PROTOCOL_PREFIX}:sha256:"
        f"{hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()}"
    )
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen_before_new_final_collection"
    ):
        raise Issue15AmendmentError("amended protocol is not prospectively frozen")
    payload = {**protocol, "artifact_identity": None}
    expected = (
        f"{PROTOCOL_PREFIX}:sha256:"
        f"{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"
    )
    if protocol.get("artifact_identity") != expected:
        raise Issue15AmendmentError("amended protocol identity is stale")
    return dict(protocol)


def write_frozen_bundle(
    plan: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    output_root: Path,
) -> None:
    target = Path(output_root)
    if target.exists():
        raise Issue15AmendmentError(f"immutable amendment output exists: {target}")
    for name, value in plan.items():
        _atomic_write(target / name, value)
    _atomic_write(target / "cohort-v2-prospective-statistical-protocol-v2.json", protocol)


def load_frozen_bundle(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    target = Path(root)
    names = (
        "confirmatory-plan.json",
        "partition-exposure-manifest.json",
        "scenario-inventory.json",
        "final-evaluation.sealed-projection.json",
        "final-evaluation-workflow-access-manifest.json",
    )
    plan = {name: _load(target / name) for name in names}
    protocol = validate_protocol(_load(
        target / "cohort-v2-prospective-statistical-protocol-v2.json"
    ))
    return plan, protocol


__all__ = [
    "COLLECTION_IDENTITY",
    "DEFAULT_AUTHORITY_ROOT",
    "DEFAULT_ROOT",
    "FINAL_RELEASE_IDENTITY",
    "FINAL_SEED",
    "Issue15AmendmentError",
    "SEALED_BUNDLE_IDENTITY",
    "build_plan",
    "build_protocol",
    "load_frozen_bundle",
    "materialize_final_authority",
    "validate_protocol",
    "write_frozen_bundle",
]
