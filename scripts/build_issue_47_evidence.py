"""Build the real cohort-v2 partition, leakage, and access evidence for issue 47."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from scripts.build_issue_45_evidence import build_issue_45_evidence
from scripts.cohort_partition import (
    audit_cohort_partition_manifest,
    create_cohort_partition_manifest,
)
from scripts.cohort_v2_partition import (
    EXPOSURE_ROLES,
    CohortV2PartitionExposureManifest,
    audit_cohort_v2_partition_exposure,
    audit_cohort_v2_workflow_influence,
    create_cohort_v2_partition_exposure_manifest,
)
from scripts.cohort_v2_scenarios import (
    ScenarioInventoryEntry,
    create_reviewed_central_v2_scenario_inventory,
    load_cohort_v2_scenario_manifest,
    validate_central_v2_scenario_inventory,
    write_immutable_cohort_v2_json,
)
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    audit_final_evaluation_workflow_access,
    create_final_evaluation_workflow_access_manifest,
)
from scripts.scenario_manifest import scenario_manifest_projection


OUTPUT_RELATIVE_PATH = Path("data/runtime_evidence/issue-47")
SOURCE_REVIEW_AUTHOR = "Sino-Huang"
SOURCE_REVIEW_URL = (
    "https://github.com/Sino-Huang/NovPhy/issues/45#issuecomment-5358450102"
)
WORKFLOW_IDENTITY = "central-v2-final-evaluation-workflow-v1"
OPERATOR_IDENTITY = "novphy-operator-v1:final-evaluation-custodian"
WORKFLOW_FROZEN_AT = "2026-08-22T07:01:00Z"

_PUBLIC_MANIFESTS = {
    "training": "training.json",
    "calibration": "calibration.json",
    "model_selection": "model-selection.json",
}


def _provenance_records(
    manifest: CohortV2PartitionExposureManifest,
) -> list[dict[str, str]]:
    records = []
    for entry in manifest.entries:
        for kind in (
            "derivation_artifact",
            "generation_seed",
            "intervention",
            "observation_configuration",
            "observation_variant",
            "replay",
            "rerun",
        ):
            records.append({
                "artifact_kind": kind,
                "artifact_identity": (
                    f"cohort-v2-{kind.replace('_', '-')}-v1:{entry.exposure_role}"
                ),
                "source_scenario_lineage_identity": entry.scenario_lineage_identity,
                "level_instance_identity": entry.level_instance_identity,
                "scenario_template_identity": entry.scenario_template_identity,
                "dataset_partition": entry.dataset_partition,
                "exposure_role": entry.exposure_role,
            })
    return records


def _generic_provenance_records(
    records: list[dict[str, str]],
) -> list[dict[str, Any]]:
    return [{
        "artifact_kind": record["artifact_kind"],
        "artifact_identity": record["artifact_identity"],
        "consumer_scenario_lineage_identity": (
            record["source_scenario_lineage_identity"]
        ),
        "source_scenario_lineage_identities": [
            record["source_scenario_lineage_identity"]
        ],
    } for record in records]


def _expected_rejection(
    mutation: str,
    operation: Callable[[], object],
) -> dict[str, Any]:
    try:
        operation()
    except ValueError as error:
        result = {"mutation": mutation, "rejected": True, "reason": str(error)}
        audit_record = getattr(error, "audit_record", None)
        if audit_record is not None:
            result["audit_record"] = audit_record
        return result
    raise ValueError(f"Issue-47 mutation was not rejected: {mutation}")


def _mutation_results(
    manifest: CohortV2PartitionExposureManifest,
    records: list[dict[str, str]],
) -> list[dict[str, Any]]:
    results = []

    missing_role = manifest.to_dict()
    missing_role["entries"].pop()
    results.append(_expected_rejection(
        "missing_role",
        lambda: CohortV2PartitionExposureManifest.from_dict(missing_role),
    ))

    duplicate_lineage = manifest.to_dict()
    duplicate_lineage["entries"][-1]["scenario_lineage_identity"] = (
        duplicate_lineage["entries"][0]["scenario_lineage_identity"]
    )
    results.append(_expected_rejection(
        "duplicate_lineage",
        lambda: CohortV2PartitionExposureManifest.from_dict(duplicate_lineage),
    ))

    reused_level = manifest.to_dict()
    reused_level["entries"][-1]["level_instance_identity"] = (
        reused_level["entries"][0]["level_instance_identity"]
    )
    results.append(_expected_rejection(
        "held_out_level_instance_reuse",
        lambda: CohortV2PartitionExposureManifest.from_dict(reused_level),
    ))

    unknown_lineage = deepcopy(records)
    unknown_lineage[0]["source_scenario_lineage_identity"] = "scenario-lineage:unknown"
    results.append(_expected_rejection(
        "unknown_lineage",
        lambda: audit_cohort_v2_partition_exposure(
            manifest,
            declared_provenance_records=unknown_lineage,
            observed_artifact_identities=[
                record["artifact_identity"] for record in unknown_lineage
            ],
        ),
    ))

    for kind in ("replay", "derivation_artifact", "observation_variant"):
        leaked = deepcopy(records)
        record = next(item for item in leaked if item["artifact_kind"] == kind)
        record["exposure_role"] = "calibration"
        mutation_name = "derivation" if kind == "derivation_artifact" else kind
        results.append(_expected_rejection(
            f"{mutation_name}_role_leak",
            lambda leaked=leaked: audit_cohort_v2_partition_exposure(
                manifest,
                declared_provenance_records=leaked,
                observed_artifact_identities=[
                    item["artifact_identity"] for item in leaked
                ],
            ),
        ))

    results.append(_expected_rejection(
        "undeclared_artifact_provenance",
        lambda: audit_cohort_v2_partition_exposure(
            manifest,
            declared_provenance_records=records[1:],
            observed_artifact_identities=[
                record["artifact_identity"] for record in records
            ],
        ),
    ))
    return results


def _load_real_issue_45_inputs(
    repository_root: Path,
    temporary_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    public_root = temporary_root / "issue-45-public"
    sealed_root = temporary_root / "issue-45-sealed"
    build_issue_45_evidence(
        repository_root=repository_root,
        public_root=public_root,
        sealed_root=sealed_root,
    )
    draft = json.loads((public_root / "inventory/draft.json").read_bytes())
    reviewed = create_reviewed_central_v2_scenario_inventory(
        draft,
        review_author=SOURCE_REVIEW_AUTHOR,
        review_url=SOURCE_REVIEW_URL,
        manifest_root=public_root / "manifests",
    )
    validate_central_v2_scenario_inventory(
        reviewed,
        manifest_root=public_root / "manifests",
    )
    scenarios = {
        role: load_cohort_v2_scenario_manifest(
            public_root / "manifests" / reference
        )
        for role, reference in _PUBLIC_MANIFESTS.items()
    }
    scenarios["final_evaluation"] = load_cohort_v2_scenario_manifest(
        sealed_root / "final-evaluation.cohort-v2-scenario.json"
    )
    return reviewed, scenarios


def _build_payloads(repository_root: Path) -> dict[str, Mapping[str, Any]]:
    repository_root = repository_root.resolve()
    with tempfile.TemporaryDirectory() as temporary:
        reviewed, scenarios = _load_real_issue_45_inputs(
            repository_root,
            Path(temporary),
        )
        inventory_entries = tuple(
            ScenarioInventoryEntry.from_dict(entry) for entry in reviewed["entries"]
        )
        manifest = create_cohort_v2_partition_exposure_manifest(
            partition_version=1,
            source_inventory_identity=str(reviewed["identity"]),
            source_inventory_review_url=SOURCE_REVIEW_URL,
            inventory_entries=inventory_entries,
            lineage_quotas={role: 1 for role in EXPOSURE_ROLES},
        )
        provenance = _provenance_records(manifest)
        observed_artifacts = [record["artifact_identity"] for record in provenance]
        public_audit = audit_cohort_v2_partition_exposure(
            manifest,
            declared_provenance_records=provenance,
            observed_artifact_identities=observed_artifacts,
        )

        generic_entries = []
        for entry in manifest.entries:
            scenario = scenarios[entry.exposure_role]
            reference = (
                entry.sealed_scenario_manifest_reference
                if entry.exposure_role == "final_evaluation"
                else entry.scenario_manifest_reference
            )
            generic_entries.append({
                "dataset_partition": entry.dataset_partition,
                "exposure_role": entry.exposure_role,
                **scenario_manifest_projection(
                    scenario.scenario_manifest,
                    reference,
                ),
            })
        generic_provenance = _generic_provenance_records(provenance)
        full_partition = create_cohort_partition_manifest(
            partition_version=1,
            split_regime="instance_held_out",
            held_out_roles=[],
            entries=generic_entries,
            provenance_records=generic_provenance,
        )
        audit_cohort_partition_manifest(
            full_partition,
            admitted_scenario_lineage_identities=[
                entry.scenario_lineage_identity for entry in manifest.entries
            ],
            admitted_provenance_records=generic_provenance,
        )

        ordinary_reports = []
        for role in EXPOSURE_ROLES[:-1]:
            entry = next(item for item in manifest.entries if item.exposure_role == role)
            artifact = next(
                record for record in provenance
                if record["artifact_kind"] == "observation_configuration"
                and record["exposure_role"] == role
            )
            for influence in entry.may_influence:
                ordinary_reports.append(audit_cohort_v2_workflow_influence(
                    manifest,
                    workflow_kind=role,
                    influence=influence,
                    declared_provenance_records=provenance,
                    observed_scenario_lineage_identities=[
                        entry.scenario_lineage_identity
                    ],
                    observed_artifact_identities=[artifact["artifact_identity"]],
                ))

        final_entry = next(
            entry for entry in manifest.entries
            if entry.exposure_role == "final_evaluation"
        )
        workflow = create_final_evaluation_workflow_access_manifest(
            manifest,
            workflow_version=1,
            workflow_identity=WORKFLOW_IDENTITY,
            operator_identity=OPERATOR_IDENTITY,
            frozen_at=WORKFLOW_FROZEN_AT,
            authorized_artifacts=[{
                "artifact_kind": "scenario_manifest",
                "artifact_identity": final_entry.scenario_manifest_identity,
                "source_scenario_lineage_identities": [
                    final_entry.scenario_lineage_identity
                ],
            }],
        )
        attempted_final_access = {
            "workflow_identity": workflow.workflow_identity,
            "operator_identity": workflow.operator_identity,
            "artifact_identity": final_entry.scenario_manifest_identity,
            "source_scenario_lineage_identities": [
                final_entry.scenario_lineage_identity
            ],
            "accessed_at": WORKFLOW_FROZEN_AT,
            "authorization_identity": None,
            "consumer_exposure_role": "final_evaluation",
        }
        preauthorization = _expected_rejection(
            "preauthorization_final_access",
            lambda: audit_final_evaluation_workflow_access(
                manifest,
                workflow,
                observed_accesses=[attempted_final_access],
            ),
        )

        leakage_report = {
            "schema": "issue_47_lineage_template_leakage_report_v1",
            "partition_identity": manifest.identity,
            "full_partition_manifest_identity": full_partition.identity,
            "full_partition_audit_passed": True,
            "real_manifest_audit": public_audit,
            "mutation_results": _mutation_results(manifest, provenance),
            "passed": True,
        }
        access_report = {
            "schema": "issue_47_representative_access_audit_v1",
            "partition_identity": manifest.identity,
            "ordinary_workflows": ordinary_reports,
            "preauthorization_final_access": preauthorization,
            "passed": True,
        }
        bundle = {
            "schema": "issue_47_cohort_v2_partition_evidence_bundle_v1",
            "identity": "issue-47-cohort-v2-partition-evidence-bundle-v1",
            "source_issue": "https://github.com/Sino-Huang/NovPhy/issues/47",
            "source_inventory_review_url": SOURCE_REVIEW_URL,
            "partition_manifest_identity": manifest.identity,
            "final_workflow_manifest_identity": workflow.identity,
            "artifacts": [
                "partition-exposure-manifest.json",
                "lineage-template-leakage-audit.json",
                "final-evaluation-workflow-access-manifest.json",
                "representative-access-audit.json",
            ],
            "limitations": [
                "Lineage quotas freeze partition membership only; production and pilot rollout quotas remain deferred to issues 51 and 52.",
                "Final-evaluation authorization is pending, so this bundle opens no final data.",
                "Template identities are audited with reuse allowed; no template-held-out claim or score is created.",
            ],
        }
        return {
            "partition-exposure-manifest.json": manifest.to_dict(),
            "lineage-template-leakage-audit.json": leakage_report,
            "final-evaluation-workflow-access-manifest.json": workflow.to_dict(),
            "representative-access-audit.json": access_report,
            "bundle-manifest.json": bundle,
        }


def _build_result(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    partition = payloads["partition-exposure-manifest.json"]
    workflow = payloads["final-evaluation-workflow-access-manifest.json"]
    return {
        "schema": "issue_47_cohort_v2_partition_build_result_v1",
        "partition_manifest_identity": partition["identity"],
        "final_workflow_manifest_identity": workflow["identity"],
        "artifact_names": sorted(payloads),
    }


def build_issue_47_evidence(
    repository_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    payloads = _build_payloads(repository_root)
    for name, payload in payloads.items():
        write_immutable_cohort_v2_json(payload, output_root / name)
    return _build_result(payloads)


def validate_issue_47_evidence(
    repository_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    expected = _build_payloads(repository_root)
    for name, payload in expected.items():
        try:
            observed = json.loads((output_root / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load issue-47 artifact {name}: {error}") from error
        if observed != payload:
            raise ValueError(f"Issue-47 artifact is stale: {name}")
    CohortV2PartitionExposureManifest.from_dict(
        expected["partition-exposure-manifest.json"]
    )
    FinalEvaluationWorkflowAccessManifest.from_dict(
        expected["final-evaluation-workflow-access-manifest.json"]
    )
    return _build_result(expected)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = build_issue_47_evidence(
        repository_root,
        repository_root / OUTPUT_RELATIVE_PATH,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
