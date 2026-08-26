"""Prospective statistical protocol frozen by issue #34."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Final, Mapping

from scripts.final_evaluation_access import FinalEvaluationWorkflowAccessManifest
from world_model.training.grid_artifacts import canonical_json_bytes


SCHEMA: Final = "cohort_v2_prospective_statistical_protocol_v1"
IDENTITY_PREFIX: Final = "cohort-v2-prospective-statistical-protocol-v1"
RELEASE_IDENTITY: Final = (
    "representative-cohort-v2-release-v5:issue-53:mixed-termination"
)
PUBLICATION_IDENTITY: Final = (
    "representative-cohort-v2-publication-v5:issue-53:mixed-termination"
)
DERIVATION_IDENTITY: Final = (
    "cohort-v2-authoritative-derivation-index-v5:issue-53:mixed-termination"
)
SEALED_BUNDLE_IDENTITY: Final = (
    "issue-53-final-evaluation-sealed-bundle-v5:mixed-termination"
)
WORKFLOW_IDENTITY: Final = "central-v2-final-evaluation-workflow-v5:issue-53"
CANDIDATE_ID: Final = "integrated_aggregated_joint_controller"

SOURCE_PATHS: Final = {
    "issue_41_calibration": Path(
        "data/runtime_evidence/issue-41/cohort-v2-calibration-summary.json"
    ),
    "issue_58_integrated_calibration": Path(
        "data/runtime_evidence/issue-58/cohort-v2-integrated-calibration-summary.json"
    ),
    "release": Path(
        "data/runtime_evidence/issue-53-mixed-termination-v5/cohort-v2-release.json"
    ),
    "publication": Path(
        "data/runtime_evidence/issue-53-mixed-termination-v5/cohort-v2-publication.json"
    ),
    "workflow": Path(
        "data/runtime_evidence/issue-53-plan-v5/"
        "final-evaluation-workflow-access-manifest.json"
    ),
    "access_probe": Path(
        "data/runtime_evidence/issue-54/cohort-v2-downstream-ingestion-evidence.json"
    ),
}


class CohortV2ProtocolError(ValueError):
    """The protocol or one of its frozen calibration sources is invalid."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CohortV2ProtocolError(f"cannot load protocol source {path}: {error}") from error
    if not isinstance(value, dict):
        raise CohortV2ProtocolError(f"protocol source is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _validate_sources(sources: Mapping[str, Mapping[str, Any]]) -> None:
    issue_41 = sources["issue_41_calibration"]
    issue_58 = sources["issue_58_integrated_calibration"]
    release = sources["release"]
    publication = sources["publication"]
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(sources["workflow"])
    probe = sources["access_probe"]

    if (
        issue_41.get("schema")
        != "cohort_v2_preconfirmatory_calibration_summary_v1"
        or issue_41.get("disposition", {}).get("status") != "insufficient_evidence"
        or issue_41.get("exposure_audit", {}).get(
            "final_evaluation_artifacts_accessed"
        )
        or issue_41.get("exposure_audit", {}).get(
            "final_evaluation_derived_artifacts_accessed"
        )
    ):
        raise CohortV2ProtocolError("issue-41 calibration boundary is not accepted")
    if (
        issue_58.get("schema")
        != "cohort_v2_integrated_model_calibration_summary_v1"
        or issue_58.get("release_identity") != RELEASE_IDENTITY
        or issue_58.get("disposition", {}).get("status")
        != "sufficient_evidence_to_freeze_issue_34"
        or issue_58.get("independent_calibration_rollouts") != 6
        or issue_58.get("exposure_audit", {}).get("final_evaluation_accessed")
    ):
        raise CohortV2ProtocolError("issue-58 integrated calibration is not accepted")
    proposals = issue_58.get("proposals_for_issue_34")
    if not isinstance(proposals, list) or len(proposals) != 2:
        raise CohortV2ProtocolError("issue-58 must propose exactly two compute budgets")
    if release.get("identity") != RELEASE_IDENTITY:
        raise CohortV2ProtocolError("cohort release identity differs")
    sealed = release.get("sealed_final_evaluation", {})
    attempts = sealed.get("attempt_ids")
    if (
        sealed.get("bundle_identity") != SEALED_BUNDLE_IDENTITY
        or not isinstance(attempts, list)
        or len(attempts) != 6
        or len(set(attempts)) != 6
    ):
        raise CohortV2ProtocolError("sealed final replicate inventory differs")
    if (
        publication.get("identity") != PUBLICATION_IDENTITY
        or publication.get("cohort_release_identity") != RELEASE_IDENTITY
        or publication.get("authoritative_derivation_index_identity")
        != DERIVATION_IDENTITY
        or publication.get("sealed_final_evaluation_bundle_identity")
        != SEALED_BUNDLE_IDENTITY
    ):
        raise CohortV2ProtocolError("cohort publication binding differs")
    if (
        workflow.workflow_identity != WORKFLOW_IDENTITY
        or workflow.workflow_version != 5
        or workflow.authorization_state != "pending"
        or len(workflow.final_evaluation_lineage_identities) != 1
    ):
        raise CohortV2ProtocolError("frozen final-evaluation workflow differs")
    receipt = probe.get("authorized_final_evaluation_probe", {})
    if (
        probe.get("identity")
        != "cohort-v2-downstream-ingestion-evidence-v1:issue-54:release-v5"
        or receipt.get("passed") is not True
        or receipt.get("observed_access_count") != 1
        or receipt.get("release_identity") != RELEASE_IDENTITY
        or receipt.get("sealed_bundle_identity") != SEALED_BUNDLE_IDENTITY
        or receipt.get("workflow_identity") != WORKFLOW_IDENTITY
    ):
        raise CohortV2ProtocolError("issue-54 access receipt differs")


def load_protocol_sources(
    repository_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = Path(repository_root).resolve()
    sources = {
        name: _load_object(root / relative)
        for name, relative in SOURCE_PATHS.items()
    }
    _validate_sources(sources)
    bindings = {
        name: {
            "path": relative.as_posix(),
            "sha256": _sha256(root / relative),
        }
        for name, relative in SOURCE_PATHS.items()
    }
    return sources, bindings


def _budget_rows(issue_58: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seed = 20260826
    for index, proposal in enumerate(issue_58["proposals_for_issue_34"]):
        rows.append({
            "budget": proposal["budget"],
            "strongest_comparator_id": proposal["strongest_comparator_id"],
            "practical_effect_threshold_absolute_endpoint_error_reduction": (
                proposal["practical_effect_threshold"]
            ),
            "physical_violation_margin": proposal["physical_violation_margin"],
            "fixed_complete_rollout_replicates": proposal["fixed_replicate_count"],
            "calibration_precision_half_width": proposal["precision_half_width"],
            "gain_bootstrap_seed": seed + index * 2,
            "violation_bootstrap_seed": seed + index * 2 + 1,
        })
    return rows


def build_protocol(repository_root: Path, *, implementation_commit: str) -> dict[str, Any]:
    if not implementation_commit:
        raise CohortV2ProtocolError("implementation commit is required")
    sources, bindings = load_protocol_sources(repository_root)
    issue_58 = sources["issue_58_integrated_calibration"]
    release = sources["release"]
    workflow = FinalEvaluationWorkflowAccessManifest.from_dict(sources["workflow"])
    budget_rows = _budget_rows(issue_58)
    attempts = list(release["sealed_final_evaluation"]["attempt_ids"])

    def stress_budgets(seed: int) -> list[dict[str, Any]]:
        return [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"gain_bootstrap_seed", "violation_bootstrap_seed"}
                },
                "endpoint_bootstrap_seed": seed + index * 2,
                "violation_bootstrap_seed": seed + index * 2 + 1,
                "endpoint_estimand": "parser_endpoint_error_minus_oracle_endpoint_error",
                "violation_estimand": "parser_violation_rate_minus_oracle_violation_rate",
            }
            for index, row in enumerate(budget_rows)
        ]

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_version": 1,
        "artifact_identity": "",
        "implementation_commit": implementation_commit,
        "status": "frozen_before_final_evaluation",
        "claim_boundary": {
            "supported": (
                "instance-held-out configurations in the same physics family and the "
                "accepted contact, directed-supports, steady-state, and "
                "structure-unstable ontology"
            ),
            "excluded": [
                "template-held-out generalization",
                "unseen physics or gravity shift",
                "material or damage semantics",
                "illegal-contact derivation",
                "planning or gameplay success",
            ],
            "gameplay_protocol_owner": "issue_57",
        },
        "source_bindings": {
            **bindings,
            "release_identity": RELEASE_IDENTITY,
            "publication_identity": PUBLICATION_IDENTITY,
            "authoritative_derivation_index_identity": DERIVATION_IDENTITY,
            "issue_58_evidence_artifact_identity": issue_58["artifact_identity"],
            "issue_58_implementation_commit": issue_58["implementation_commit"],
            "final_workflow_manifest_identity_sha256": (
                f"sha256:{hashlib.sha256(workflow.identity.encode('utf-8')).hexdigest()}"
            ),
        },
        "exposure_audit": {
            "issue_41_final_scenarios_outcomes_labels_or_derived_metrics_accessed": False,
            "issue_58_final_scenarios_outcomes_labels_or_derived_metrics_accessed": False,
            "protocol_construction_final_scenarios_outcomes_labels_or_derived_metrics_accessed": (
                False
            ),
            "protocol_construction_inputs": (
                "non-final calibration reports and public release/workflow metadata only"
            ),
            "issue_54_receipt_use": (
                "boundary validation only; it contains no final observation, label, "
                "outcome, example, or metric"
            ),
            "sealed_final_bundle_opened": False,
            "passed": True,
        },
        "final_evaluation_access": {
            "sealed_bundle_identity": SEALED_BUNDLE_IDENTITY,
            "workflow_identity": workflow.workflow_identity,
            "operator_identity": workflow.operator_identity,
            "current_manifest_authorization_state": workflow.authorization_state,
            "later_consumer_identity": "issue-15-oracle-symbol-confirmatory-v1",
            "required_access_record_schema": "final_evaluation_workflow_access_audit_v1",
            "authorization_rule": (
                "Issue #15 must create an authorized derivative of the immutable pending "
                "workflow manifest after this protocol is committed and before its first "
                "final access; every access must be recorded under the declared workflow, "
                "operator, authorization, artifact, lineage, timestamp, and final role."
            ),
        },
        "replicate_and_seed_policy": {
            "analysis_unit": "complete_final_evaluation_rollout",
            "fixed_replicate_count": 6,
            "fixed_attempt_ids": attempts,
            "coverage": "exactly one rollout in each of the six accepted central strata",
            "candidate_training_seed": 20260824,
            "controller_seed": 10,
            "controller_aggregation_rounds": 1,
            "analysis_bootstrap_replicates": 10000,
            "analysis_bootstrap_master_seed": 20260826,
            "feature_parser_seed": 20260828,
            "visual_parser_seed": 20260829,
            "rule": (
                "Seeds, replicate count, attempt inventory, and bootstrap offsets are fixed. "
                "No outcome-conditioned seed changes, replacement replicates, or optional "
                "stopping are permitted."
            ),
        },
        "calibration_basis": {
            "source": "issue_58_integrated_calibration_only",
            "compute_budgets": (
                "The two retained budgets are the distinct calibrated support points at "
                "which the integrated candidate and at least one declared comparator are "
                "eligible. Configurations above a budget are ineligible."
            ),
            "practical_effect_threshold": (
                "At each budget, max(10 percent of the strongest comparator's calibration "
                "mean endpoint error, calibration paired-gain 95 percent interval half-width)."
            ),
            "physical_violation_margin": (
                "At each budget, max(0, upper endpoint of the calibration paired-violation "
                "increase 95 percent bootstrap interval)."
            ),
            "replicate_and_precision": (
                "Six complete calibration rollouts established the fixed six-rollout design; "
                "the budget rows retain the observed paired-gain interval half-width as the "
                "declared precision target."
            ),
            "failed_run_treatment": (
                "Copied from issue #58: source/provenance failures abort, while execution "
                "failures remain failed replicates and are neither replaced nor excluded."
            ),
            "calibration_is_confirmatory": False,
        },
        "experiment_matrix": {
            "confirmatory_oracle_symbol_issue_15": {
                "candidate_configuration_id": CANDIDATE_ID,
                "candidate_source": "issue_58_integrated_candidate_and_one_round_controller",
                "comparisons": budget_rows,
                "local_primary_mode": "teacher_forced_local_successor_prediction",
                "complete_rollout_secondary": {
                    "configuration": "fixed_h15_continuous",
                    "selection_source": "issue_58_model_selection",
                    "metrics": [
                        "requested_and_effective_horizons",
                        "terminal_mse",
                        "error_auc",
                        "error_curve",
                        "total_compute",
                    ],
                    "recursive_physical_violation": "unavailable",
                },
            },
            "learned_feature_symbol_stress_issue_16": {
                "reference": "frozen_issue_15_oracle_symbol_system",
                "stressed_configuration": "training_role_feature_parser_with_frozen_semantics",
                "parser_input": "simulator_observable_deployment_features",
                "accepted_targets": [
                    "contact",
                    "directed_supports",
                    "steady-state",
                    "structure-unstable",
                ],
                "comparisons": stress_budgets(20260926),
            },
            "frozen_visual_symbol_stress_issue_17": {
                "reference": "frozen_issue_15_oracle_symbol_system",
                "secondary_reference": "frozen_issue_16_feature_parser_system",
                "stressed_configuration": "frozen_encoder_agent_observation_parser",
                "parser_input": "synchronized_agent_observation_only",
                "canonical_observation_input": "forbidden",
                "accepted_targets": [
                    "object_alignment",
                    "contact",
                    "directed_supports",
                    "steady-state",
                    "structure-unstable",
                ],
                "comparisons": stress_budgets(20261026),
            },
            "configuration_freeze_rule": (
                "The issue-58 predictor, controller, pair grid, costs, and budget-specific "
                "comparators are immutable. Parser checkpoints, probability calibration, "
                "and decision thresholds must be source-bound using training parameters, "
                "model-selection configuration choice, and calibration thresholds before "
                "the parser receives final-evaluation data."
            ),
        },
        "estimands": {
            "primary_confirmatory": {
                "endpoint_error": (
                    "Within each complete rollout, mean duration-weighted authoritative "
                    "endpoint carrier MSE over all eligible teacher-forced decision states."
                ),
                "endpoint_gain": "strongest_comparator_error_minus_candidate_error",
                "physical_violation": (
                    "Within-rollout authoritative source-endpoint incidence of any accepted "
                    "endpoint violation; recursive predicted-carrier violation is unavailable."
                ),
                "violation_increase": "candidate_rate_minus_strongest_comparator_rate",
                "compute": [
                    "policy_compute_per_simulated_frame",
                    "full_end_to_end_compute_per_simulated_frame",
                ],
            },
            "stress_primary": {
                "endpoint_degradation": "parser_endpoint_error_minus_oracle_endpoint_error",
                "violation_degradation": "parser_violation_rate_minus_oracle_violation_rate",
                "controller_degradation": [
                    "pair_disagreement_rate",
                    "horizon_disagreement_rate",
                    "description_mode_disagreement_rate",
                ],
            },
            "parser_secondary": {
                "probability_metrics": [
                    "Brier score",
                    "negative log likelihood",
                    "10-bin equal-width expected calibration error",
                ],
                "agreement_metrics": [
                    "precision",
                    "recall",
                    "F1",
                    "directed-predicate agreement",
                    "macro-event agreement",
                ],
                "coherence": [
                    "self-relation rate",
                    "supports-without-contact rate",
                ],
                "availability_rule": (
                    "Unavailable oracle targets remain unavailable and are reported in "
                    "denominators; they are never converted to negative labels."
                ),
            },
        },
        "statistical_analysis": {
            "pairing": "pair candidate and comparator within the same complete rollout",
            "effect_summary": "arithmetic mean of the six paired rollout effects",
            "bootstrap": (
                "Nonparametric paired rollout bootstrap with replacement, 10,000 samples, "
                "using the fixed row-specific seeds."
            ),
            "reported_intervals": [
                "two-sided percentile 95 percent interval for every effect",
                "one-sided percentile 97.5 percent decision bound for each budget",
            ],
            "multiple_comparisons": (
                "Each analysis family has two budgets. Bonferroni alpha=0.05/2 gives "
                "one-sided 97.5 percent bounds. Gain and violation form an "
                "intersection-union rule at a budget and receive no additional split. "
                "Secondary and per-stratum analyses are descriptive and cannot rescue a "
                "primary decision."
            ),
            "confirmatory_decision": (
                "Supported only if at least one declared budget has all six valid paired "
                "rollouts, candidate and comparator are within matched-compute support, "
                "the one-sided lower gain bound is at least that budget's practical-effect "
                "threshold, and the one-sided upper violation-increase bound is at most its "
                "margin. Otherwise the central hypothesis is unsupported."
            ),
            "stress_decision": (
                "Each parser stress test is supported as non-materially degrading the oracle "
                "system only if at least one declared budget has an upper one-sided endpoint-"
                "degradation bound no greater than its practical-effect threshold and an "
                "upper one-sided violation-degradation bound no greater than its margin. "
                "A stress result cannot change or rescue the issue-15 decision."
            ),
            "sensitivity": (
                "Repeat descriptively at 0.5x and 1.5x each practical-effect threshold and "
                "at 0x and 2x each violation margin; the frozen primary decision is unchanged."
            ),
        },
        "failures_exclusions_and_stopping": {
            "source_or_provenance_failure": (
                "abort before analysis and do not open more final data"
            ),
            "model_or_parser_execution_failure": (
                "retain as a failed replicate, do not replace or exclude it, and fail the "
                "affected configuration and budget"
            ),
            "missing_run": "no imputation; retain the missing record and fail the affected budget",
            "outliers": "no outcome-based exclusions, winsorization, or robust substitution",
            "stopping": (
                "Run the complete fixed matrix once. No interim decision, early success stop, "
                "sample-size increase, seed change, or outcome-conditioned rerun is allowed."
            ),
        },
        "required_outputs": {
            "machine_readable": [
                "one source-bound row per configuration, budget, rollout, and metric",
                "failure and missing-run records",
                "paired effects and bootstrap samples or reproducible bootstrap inputs",
                "aggregate decisions and sensitivity results",
            ],
            "tables": [
                "final replicate inventory and exposure audit",
                "per-budget confirmatory gain, violation, compute, intervals, and decision",
                "local teacher-forced versus complete-rollout metrics",
                "feature-parser predicate, controller, endpoint, and violation gaps",
                (
                    "visual-parser agreement, calibration, coherence, controller, endpoint, "
                    "and violation gaps"
                ),
                "failures, missing runs, exclusions, and sensitivity",
            ],
            "plots": [
                "per-budget paired gain and violation interval plot",
                "compute-error frontier with matched-support boundaries",
                "fixed-h15 complete-rollout error curve",
                "feature-parser reliability and degradation plots",
                "visual-parser reliability, coherence, and degradation plots",
            ],
            "binding_columns": [
                "protocol_identity",
                "release_identity",
                "partition_identity",
                "code_revision",
                "configuration_id",
                "checkpoint_identity",
                "attempt_id",
                "seed",
                "exposure_role",
            ],
            "decision_values": [
                "supported",
                "unsupported",
                "not_run_due_to_provenance_abort",
            ],
        },
        "amendment_policy": (
            "This version is frozen before issue #15. Any change requires a new version, "
            "a reason, a field-level diff, and a complete rerun of every affected analysis. "
            "An outcome-informed amendment is exploratory unless evaluated on a new untouched "
            "sealed partition; it cannot preserve the confirmatory label on the opened data."
        ),
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_statistical_protocol --dry-run",
            (
                "python -u -m scripts.run_cohort_v2_statistical_protocol "
                f"--implementation-commit {implementation_commit}"
            ),
            "python -u -m scripts.run_cohort_v2_statistical_protocol --validate",
        ],
    }
    identity_payload = {**payload, "artifact_identity": None}
    payload["artifact_identity"] = (
        f"{IDENTITY_PREFIX}:sha256:"
        f"{hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()}"
    )
    return payload


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, Mapping) or protocol.get("schema") != SCHEMA:
        raise CohortV2ProtocolError("protocol schema is invalid")
    implementation_commit = protocol.get("implementation_commit")
    if not isinstance(implementation_commit, str) or not implementation_commit:
        raise CohortV2ProtocolError("protocol implementation commit is invalid")
    identity_payload = {**protocol, "artifact_identity": None}
    expected = (
        f"{IDENTITY_PREFIX}:sha256:"
        f"{hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()}"
    )
    if protocol.get("artifact_identity") != expected:
        raise CohortV2ProtocolError("protocol artifact identity is stale")
    return dict(protocol)


def write_protocol(protocol: Mapping[str, Any], path: Path) -> Path:
    validated = validate_protocol(protocol)
    target = Path(path)
    if target.exists():
        raise CohortV2ProtocolError(f"immutable protocol already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(validated, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


def load_protocol(path: Path) -> dict[str, Any]:
    return validate_protocol(_load_object(Path(path)))
