"""Source-bound cohort-v2 production and parameter plans for issue #52."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Final

from scripts.build_issue_51_evidence import validate_issue_51_evidence
from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest


ROOT: Final = Path(__file__).resolve().parents[1]
PLAN_VERSION: Final = 1
COLLECTION_SCHEMA: Final = "cohort_v2_production_collection_plan_v1"
COLLECTION_IDENTITY: Final = "cohort-v2-production-collection-plan-v1:issue-52"
PARAMETER_SCHEMA: Final = "cohort_v2_production_parameter_plan_v1"
PARAMETER_IDENTITY: Final = "cohort-v2-production-parameter-plan-v1:issue-52"
BUNDLE_SCHEMA: Final = "issue_52_cohort_v2_production_plans_bundle_v1"
BUNDLE_IDENTITY: Final = "issue-52-cohort-v2-production-plans-bundle-v1"

PILOT_ROOT = Path("data/runtime_evidence/issue-51")
PILOT_REPORT_PATH = PILOT_ROOT / "representative-cohort-v2-pilot-report.json"
PILOT_PLAN_PATH = PILOT_ROOT / "pilot-plan.json"
PILOT_ACCOUNTING_PATH = PILOT_ROOT / "attempt-accounting.json"
PARTITION_PATH = Path("data/runtime_evidence/issue-47/partition-exposure-manifest.json")
MACRO_SPEC_PATH = Path("data/runtime_evidence/issue-49/derivation-spec.json")
MACRO_ADJUDICATION_PATH = Path(
    "data/runtime_evidence/issue-49/macro-semantics-adjudication.json"
)
VIOLATION_SPEC_PATH = Path("data/runtime_evidence/issue-50/derivation-spec.json")
VIOLATION_ADJUDICATION_PATH = Path(
    "data/runtime_evidence/issue-50/physical-violation-adjudication.json"
)
REPLAY_PLAN_PATH = Path("data/runtime_evidence/issue-48/replay-plan.json")
REPLAY_EVIDENCE_PATH = Path("data/runtime_evidence/issue-48/replay-evidence.json")
CAPABILITY_PATH = Path("docs/data_contracts/cohort_v2_capabilities_v1.json")

CAPTURE_PATHS: Final = (
    Path("data/runtime_evidence/issue-44/captures/collision.json"),
    Path("data/runtime_evidence/issue-44/captures/no-contact.json"),
    Path("data/runtime_evidence/issue-44/captures/stable-terminal.json"),
    Path("data/runtime_evidence/issue-44/captures/support-change.json"),
    Path("data/runtime_evidence/issue-44/captures/support.json"),
    Path("data/runtime_evidence/issue-50/source-probes/captures/floating-a-geometry.json"),
    Path("data/runtime_evidence/issue-50/source-probes/captures/floating-a-stationary.json"),
    Path("data/runtime_evidence/issue-50/source-probes/captures/floating-b-geometry.json"),
    Path("data/runtime_evidence/issue-50/source-probes/captures/floating-b-stationary.json"),
    PILOT_ROOT / "supplementary/captures/level-clear-geometry.json",
    PILOT_ROOT / "supplementary/captures/level-clear-targeted.json",
)

ROLE_ORDER: Final = (
    "training",
    "calibration",
    "model_selection",
    "final_evaluation",
)
CENTRAL_STRATA: Final = (
    "no-contact/miss",
    "collision",
    "persistent support",
    "support change",
    "destruction",
    "stability transitions",
)
SUPPORTED_TERMINATIONS: Final = ("level_clear", "level_fail", "stable_entered")
TERMINATION_VOCABULARY: Final = (*SUPPORTED_TERMINATIONS, "rollout_ceiling")


def _load_json(repository_root: Path, relative_path: Path) -> dict[str, Any]:
    path = repository_root / relative_path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load issue-52 source {relative_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Issue-52 source {relative_path} must be an object")
    return value


def _record(
    source_record_id: str,
    path: Path,
    identity: str,
    *record_ids: str,
) -> dict[str, Any]:
    return {
        "source_record_id": source_record_id,
        "artifact_path": path.as_posix(),
        "artifact_identity": identity,
        "record_ids": list(record_ids),
        "exposure_boundary": "no_final_evaluation_outcomes",
    }


def _evidence(
    *records: Mapping[str, Any],
    analysis_method: str,
    observed_range_or_uncertainty: str,
    decision_rule: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "plan_version": PLAN_VERSION,
        "source_record_ids": [record["source_record_id"] for record in records],
        "analysis_method": analysis_method,
        "observed_range_or_uncertainty": observed_range_or_uncertainty,
        "decision_rule": decision_rule,
        "rationale": rationale,
    }


def _load_sources(repository_root: Path, *, validate_pilot: bool) -> dict[str, Any]:
    root = repository_root.resolve()
    if validate_pilot:
        result = validate_issue_51_evidence(root / PILOT_ROOT, repository_root=root)
        if result != {
            "schema": "issue_51_representative_cohort_v2_pilot_validation_result_v1",
            "bundle_identity": "issue-51-representative-cohort-v2-pilot-bundle-v1:accepted-6",
            "pilot_report_identity": (
                "representative-cohort-v2-pilot-report-v1:accepted-determination-6"
            ),
            "representative_audit": True,
            "passed": True,
        }:
            raise ValueError("Issue-51 pilot validation did not produce the accepted result")
    sources = {
        "pilot_report": _load_json(root, PILOT_REPORT_PATH),
        "pilot_plan": _load_json(root, PILOT_PLAN_PATH),
        "pilot_accounting": _load_json(root, PILOT_ACCOUNTING_PATH),
        "partition": _load_json(root, PARTITION_PATH),
        "capabilities": _load_json(root, CAPABILITY_PATH),
        "macro_spec": _load_json(root, MACRO_SPEC_PATH),
        "macro_adjudication": _load_json(root, MACRO_ADJUDICATION_PATH),
        "violation_spec": _load_json(root, VIOLATION_SPEC_PATH),
        "violation_adjudication": _load_json(root, VIOLATION_ADJUDICATION_PATH),
        "replay_plan": _load_json(root, REPLAY_PLAN_PATH),
        "replay_evidence": _load_json(root, REPLAY_EVIDENCE_PATH),
        "captures": [_load_json(root, path) for path in CAPTURE_PATHS],
    }
    CohortV2PartitionExposureManifest.from_dict(sources["partition"])
    report = sources["pilot_report"]
    if (
        report.get("identity")
        != "representative-cohort-v2-pilot-report-v1:accepted-determination-6"
        or report.get("disposition") != "accepted"
        or report.get("representative_audit") is not True
        or report.get("passed") is not True
        or report.get("final_evaluation", {}).get("consumed") is not False
    ):
        raise ValueError("Issue-52 requires the accepted, unconsumed issue-51 pilot")
    if report.get("partition_identity") != sources["partition"].get("identity"):
        raise ValueError("Issue-51 pilot is not bound to the issue-47 partition")
    return sources


def _source_records(sources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    report = sources["pilot_report"]
    plan = sources["pilot_plan"]
    accounting = sources["pilot_accounting"]
    partition = sources["partition"]
    macro_spec = sources["macro_spec"]
    macro_adjudication = sources["macro_adjudication"]
    violation_spec = sources["violation_spec"]
    violation_adjudication = sources["violation_adjudication"]
    replay_plan = sources["replay_plan"]
    replay_evidence = sources["replay_evidence"]
    return {
        "pilot_report": _record(
            "pilot_report", PILOT_REPORT_PATH, report["identity"], report["identity"]
        ),
        "pilot_plan": _record(
            "pilot_plan", PILOT_PLAN_PATH, plan["identity"], plan["identity"]
        ),
        "pilot_accounting": _record(
            "pilot_accounting",
            PILOT_ACCOUNTING_PATH,
            accounting["identity"],
            accounting["identity"],
        ),
        "partition": _record(
            "partition", PARTITION_PATH, partition["identity"], partition["identity"]
        ),
        "macro": _record(
            "macro",
            MACRO_ADJUDICATION_PATH,
            macro_adjudication["identity"],
            macro_spec["identity"],
        ),
        "violation": _record(
            "violation",
            VIOLATION_ADJUDICATION_PATH,
            violation_adjudication["identity"],
            violation_spec["identity"],
        ),
        "replay": _record(
            "replay",
            REPLAY_EVIDENCE_PATH,
            replay_evidence["identity"],
            replay_plan["identity"],
        ),
        "captures": _record(
            "captures",
            Path("data/runtime_evidence/issue-51/bundle-manifest.json"),
            "issue-51-representative-cohort-v2-pilot-bundle-v1:accepted-6",
            *(capture["capture_id"] for capture in sources["captures"]),
        ),
    }


def _action(offset: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "action_type": "drag_hold_release",
            "coordinate_frame": "slingshot_relative",
            "drag_release": list(offset),
            "frame_height": 480,
            "tapTime": 0,
            "releaseTime": 1000,
        },
        {
            "schema": "slingshot_relative_intervention_v1",
            "drag_delta_canvas_pixels": list(offset),
            "tap_time_milliseconds": 0,
            "hold_milliseconds": 1000,
        },
    )


def _intervention(
    identifier: str,
    ordinal: int,
    stratum: str,
    source: str,
    offset: tuple[int, int],
    termination: str,
    evidence_id: str,
) -> dict[str, Any]:
    interface, engine = _action(offset)
    return {
        "id": identifier,
        "ordinal": ordinal,
        "intended_coverage_stratum": stratum,
        "intervention_source": source,
        "intended_termination_class": termination,
        "interface_action": interface,
        "engine_relative_action": engine,
        "mapping_version": "science-birds-slingshot-relative-v1",
        "evidence_id": evidence_id,
    }


def _collection_evidence(
    sources: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "partition_quotas": _evidence(
            records["pilot_report"], records["partition"],
            analysis_method="Exact projection of the four accepted issue-47 role assignments revalidated by issue 51.",
            observed_range_or_uncertainty="Four roles, four distinct lineages and level instances, two benchmark conditions and two source templates; final evaluation remained sealed and unconsumed.",
            decision_rule="Assign one frozen lineage and six planned interventions to each accepted exposure role.",
            rationale="This preserves the accepted instance-held-out exposure manifest while giving every role the same prospective central-stratum coverage.",
        ),
        "coverage_quotas": _evidence(
            records["pilot_report"], records["pilot_plan"], records["captures"],
            analysis_method="Scale the accepted six-stratum pilot requirement uniformly across all four frozen exposure roles.",
            observed_range_or_uncertainty="Each of six central strata passed with actual Unity evidence; the pilot did not estimate a stratum prevalence.",
            decision_rule="Require one planned intervention per central stratum per role; do not fill quotas from realized outcomes.",
            rationale="Twenty-four fixed attempts cover every required role/stratum pair without outcome-conditioned resampling.",
        ),
        "intervention_actions": _evidence(
            records["pilot_report"], records["captures"], records["replay"],
            analysis_method="Reuse exact accepted slingshot-relative actions from the issue-44/48/51 pilot records and preserve both interface and engine-relative forms.",
            observed_range_or_uncertainty="Accepted offsets were [77,29], [-77,29], [-74,-31], and [-77,30] pixels with tap 0 ms and hold 1000 ms; transfer to other frozen lineages remains a prospective coverage assignment, not an outcome guarantee.",
            decision_rule="Freeze those four accepted offsets; duplicate an action only as a distinct planned rollout with a distinct intended stratum.",
            rationale="The accepted actions supply geometry-aware and targeted sources without inventing benchmark-agent provenance.",
        ),
        "termination_quotas": _evidence(
            records["pilot_report"], records["pilot_plan"], records["captures"],
            analysis_method="Map each frozen action to the terminal class demonstrated by its accepted pilot source, then repeat the map across roles.",
            observed_range_or_uncertainty="Actual accepted pilot evidence demonstrated level_clear, level_fail and stable_entered; rollout_ceiling was not observed.",
            decision_rule="Require 4 level_clear, 8 level_fail and 12 stable_entered intended assignments; allow rollout_ceiling only as a non-quota safety termination.",
            rationale="All supported terminal classes remain prospectively represented without requiring an artificial ceiling termination.",
        ),
        "capability_quotas": _evidence(
            records["pilot_report"], records["captures"], records["replay"],
            analysis_method="Apply every accepted per-rollout central capture, observation, validation and derivation path to all 24 attempts, with one deterministic replay proof per lineage.",
            observed_range_or_uncertainty="Issue 51 passed every central capability and two non-final replay collections; production-scale failure rates remain uncertain.",
            decision_rule="Require 24 validated rollout paths for per-rollout capabilities and 4 replay proofs, one per frozen lineage.",
            rationale="Capability coverage is explicit and cannot be inferred from rollout counts alone.",
        ),
        "attempt_policy": _evidence(
            records["pilot_report"], records["pilot_accounting"],
            analysis_method="Inspect outcome-independent accounting and typed dispositions for all 24 pilot attempts.",
            observed_range_or_uncertainty="Nineteen accepted and five failed/quarantined attempts; zero retries. The four runtime failures were classified permanent and the mutation was an invalidation check.",
            decision_rule="Execute every planned attempt exactly once, retry no failure code, quarantine invalid attempts, and retain all unmet quotas.",
            rationale="The pilot supplies no accepted transient-retry evidence, so production cannot justify retries or quota-filling resampling.",
        ),
        "noncentral_dispositions": _evidence(
            records["pilot_report"], records["pilot_plan"],
            analysis_method="Project the accepted secondary, optional and out-of-scope dispositions without promoting them into central quotas.",
            observed_range_or_uncertainty="The accepted pilot recorded bounded negative, material, physical-regime and other secondary capabilities as not required; benchmark-agent action provenance was unavailable.",
            decision_rule="Assign no placeholder quotas and do not emit excluded predicates as false.",
            rationale="The plan remains limited to the approved central joint-controller experiment.",
        ),
    }


def _assignments(partition: Mapping[str, Any], intervention_ids: list[str]) -> list[dict[str, Any]]:
    assignments = []
    for entry in partition["entries"]:
        reference_field = (
            "sealed_scenario_manifest_reference"
            if entry["exposure_role"] == "final_evaluation"
            else "scenario_manifest_reference"
        )
        assignments.append({
            "exposure_role": entry["exposure_role"],
            "dataset_partition": entry["dataset_partition"],
            "inventory_state": entry["inventory_state"],
            "benchmark_condition_identity": entry["benchmark_condition_identity"],
            "scenario_template_identity": entry["scenario_template_identity"],
            "level_instance_identity": entry["level_instance_identity"],
            "scenario_lineage_identity": entry["scenario_lineage_identity"],
            "scenario_manifest_identity": entry["scenario_manifest_identity"],
            reference_field: entry[reference_field],
            "intervention_ids": list(intervention_ids),
            "planned_rollout_quota": len(intervention_ids),
            "quota_evidence_id": "partition_quotas",
        })
    return assignments


def _capability_quotas() -> dict[str, int]:
    per_rollout = (
        "contact",
        "supports",
        "steady-state",
        "structure-unstable",
        "excess_penetration",
        "unsupported_stationary_or_floating_body",
        "agent_observation",
        "canonical_observation_access_restriction",
        "fixed_step_capture",
        "complete_raw_contact_intervals",
        "atomic_rollout_validation",
        "typed_failure_and_quarantine_accounting",
        "source_bound_derivations",
    )
    return {**{name: 24 for name in per_rollout}, "version_bounded_deterministic_replay": 4}


def _collection_plan(sources: Mapping[str, Any]) -> dict[str, Any]:
    records = _source_records(sources)
    interventions = [
        _intervention("central-no-contact-miss", 1, "no-contact/miss", "targeted_rare", (77, 29), "level_fail", "intervention_actions"),
        _intervention("central-collision", 2, "collision", "geometry_stratified", (-77, 29), "stable_entered", "intervention_actions"),
        _intervention("central-persistent-support", 3, "persistent support", "targeted_rare", (77, 29), "level_fail", "intervention_actions"),
        _intervention("central-support-change", 4, "support change", "geometry_stratified", (-77, 29), "stable_entered", "intervention_actions"),
        _intervention("central-destruction", 5, "destruction", "targeted_rare", (-77, 30), "level_clear", "intervention_actions"),
        _intervention("central-stability-transition", 6, "stability transitions", "targeted_rare", (-74, -31), "stable_entered", "intervention_actions"),
    ]
    intervention_ids = [item["id"] for item in interventions]
    assignments = _assignments(sources["partition"], intervention_ids)
    condition_quotas: dict[str, int] = {}
    template_instance_quotas = []
    partition_quotas = []
    for assignment in assignments:
        condition = assignment["benchmark_condition_identity"]
        condition_quotas[condition] = condition_quotas.get(condition, 0) + 6
        partition_quotas.append({
            "dataset_partition": assignment["dataset_partition"],
            "scenario_lineage_identity": assignment["scenario_lineage_identity"],
            "quota": 6,
            "evidence_id": "partition_quotas",
        })
        template_instance_quotas.append({
            "exposure_role": assignment["exposure_role"],
            "scenario_template_identity": assignment["scenario_template_identity"],
            "level_instance_identity": assignment["level_instance_identity"],
            "quota": 6,
            "evidence_id": "partition_quotas",
        })
    return {
        "schema": COLLECTION_SCHEMA,
        "plan_version": PLAN_VERSION,
        "identity": COLLECTION_IDENTITY,
        "authority": {
            "github_issue": 52,
            "capability_declaration_identity": sources["capabilities"]["identity"],
            "accepted_pilot_report_identity": sources["pilot_report"]["identity"],
            "accepted_pilot_plan_identity": sources["pilot_plan"]["identity"],
            "partition_manifest_identity": sources["partition"]["identity"],
        },
        "assignments": assignments,
        "interventions": interventions,
        "intervention_source_dispositions": {
            "geometry_stratified": "included",
            "targeted_rare": "included",
            "benchmark_agent_replay": "optional_unavailable_no_provenanced_action",
        },
        "attempt_policy": {
            "ordering": "exposure_role_order_then_intervention_ordinal",
            "exposure_role_order": list(ROLE_ORDER),
            "max_attempts_per_intervention": 1,
            "transient_failure_codes": [],
            "retry_counts": {},
            "stopping_rule": "execute_all_24_planned_attempts_once",
            "outcome_independent_accounting": True,
            "quota_fill_resampling": False,
            "invalid_attempt_disposition": "atomic_quarantine_and_retain_unmet_quota",
            "evidence_id": "attempt_policy",
        },
        "termination_policy": {
            "closed_vocabulary": list(TERMINATION_VOCABULARY),
            "required_quota_classes": list(SUPPORTED_TERMINATIONS),
            "rollout_ceiling_disposition": "allowed_safety_termination_not_quota_bearing",
            "evidence_id": "termination_quotas",
        },
        "bounded_negative": {
            "status": "not_required_named_secondary_spsg_contrastive_loss_ablation",
            "cap": 0,
            "no_contact_central_stratum_is_negative_training_evidence": False,
            "evidence_id": "noncentral_dispositions",
        },
        "quotas": {
            "total_planned_rollouts": {"quota": 24, "evidence_id": "coverage_quotas"},
            "benchmark_condition": {
                key: {"quota": value, "evidence_id": "partition_quotas"}
                for key, value in sorted(condition_quotas.items())
            },
            "exposure_role": {
                role: {"quota": 6, "evidence_id": "partition_quotas"}
                for role in ROLE_ORDER
            },
            "instance_held_out_partition": partition_quotas,
            "scenario_template_level_instance": template_instance_quotas,
            "intervention_source": {
                "geometry_stratified": {"quota": 8, "evidence_id": "coverage_quotas"},
                "targeted_rare": {"quota": 16, "evidence_id": "coverage_quotas"},
            },
            "central_coverage_stratum": {
                stratum: {"quota": 4, "evidence_id": "coverage_quotas"}
                for stratum in CENTRAL_STRATA
            },
            "termination_class": {
                "level_clear": {"quota": 4, "evidence_id": "termination_quotas"},
                "level_fail": {"quota": 8, "evidence_id": "termination_quotas"},
                "stable_entered": {"quota": 12, "evidence_id": "termination_quotas"},
            },
            "required_capability_coverage": {
                name: {"quota": quota, "evidence_id": "capability_quotas"}
                for name, quota in _capability_quotas().items()
            },
        },
        "noncentral_dispositions": {
            "evidence.bounded_negative": "not_required_named_secondary",
            "label.material": "not_required_named_secondary",
            "label.damage": "not_required_out_of_scope",
            "claim.gravity_shift_generalization": "not_required_out_of_scope",
            "split.template_held_out": "not_required_out_of_scope",
            "supervision.physical_regime_gate": "not_required_named_secondary",
            "macro.cascade-active": "excluded_not_emitted_not_false",
            "macro.collapsed": "excluded_not_emitted_not_false",
            "macro.pigs-cleared": "excluded_not_emitted_not_false",
            "violation.illegal_contact": "excluded_not_emitted_not_false",
            "evidence_id": "noncentral_dispositions",
        },
        "source_records": records,
        "evidence": _collection_evidence(sources, records),
    }


def _parameter_evidence(
    sources: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    spans = [
        capture["fixed_step_samples"][-1]["fixed_step"]
        - capture["pre_intervention_fixed_step"]
        for capture in sources["captures"]
    ]
    return {
        "capture_stride": _evidence(
            records["pilot_report"], records["captures"],
            analysis_method="Read the configured fixed-step capture stride from every accepted issue-44, issue-50 and issue-51 physical pilot capture.",
            observed_range_or_uncertainty="Observed stride range [1, 1] across 11 accepted non-final Unity captures.",
            decision_rule="Freeze the sole demonstrated positive integer stride.",
            rationale="Stride 1 retains every fixed-step frame record and matches the accepted exporter and derivation evidence.",
        ),
        "stability_window": _evidence(
            records["pilot_report"], records["macro"], records["violation"],
            analysis_method="Compare the accepted macro debounce and physical-violation stability-window specifications and their boundary adjudications.",
            observed_range_or_uncertainty="Both accepted specifications use exactly two consecutive fixed steps; each passed two boundary windows.",
            decision_rule="Freeze the common accepted two-step window.",
            rationale="A different window would invalidate both accepted central derivation authorities.",
        ),
        "rollout_ceiling": _evidence(
            records["pilot_report"], records["captures"],
            analysis_method="Measure last retained fixed step minus pre-intervention fixed step for all 11 accepted physical pilot captures.",
            observed_range_or_uncertainty=f"Observed terminal spans [{min(spans)}, {max(spans)}] fixed steps; no rollout_ceiling termination was observed and production tails are uncertain.",
            decision_rule="Round twice the observed maximum span (582) upward to the next 100 fixed steps.",
            rationale="A 600-step ceiling gives an explicit conservative margin without reusing the unrelated v1 ceiling or consulting final data.",
        ),
        "support_threshold": _evidence(
            records["pilot_report"], records["captures"],
            analysis_method="Retain the engine-authoritative support direction threshold exercised by accepted persistent-support and support-change records.",
            observed_range_or_uncertainty="The accepted physics-v2 exporter used a strict vertical contact-normal component threshold of 0.5; issue 51 records no alternate threshold sensitivity.",
            decision_rule="Freeze the accepted exporter threshold; changing it requires a new plan and semantic adjudication.",
            rationale="Central supports labels must preserve the pilot-accepted engine meaning.",
        ),
        "motion_thresholds": _evidence(
            records["pilot_report"], records["macro"], records["violation"],
            analysis_method="Retain the shared numeric inputs from both accepted derivation specifications and their positive, negative and boundary adjudications.",
            observed_range_or_uncertainty="Linear speed squared threshold 0.0001 and absolute angular speed threshold 0.01 degrees/second each passed two boundary windows in the accepted evidence; no broader sensitivity interval was accepted.",
            decision_rule="Use the exact common accepted thresholds.",
            rationale="These values jointly define steady-state and unsupported stationary/floating-body labels.",
        ),
        "penetration_tolerance": _evidence(
            records["pilot_report"], records["violation"], records["captures"],
            analysis_method="Retain the accepted excess-penetration threshold after exact re-derivation of positive, negative and threshold-boundary witnesses.",
            observed_range_or_uncertainty="Accepted contact minima span approximately -0.043324 to -0.004999995 Unity units; the 0.006 tolerance passed two crossing windows and incomplete-evidence invalidation.",
            decision_rule="Freeze 0.006 Unity units and use a strict separation < -tolerance comparison.",
            rationale="This is the empirically accepted geometric threshold for the central endpoint label.",
        ),
        "replay_tolerances": _evidence(
            records["pilot_report"], records["replay"],
            analysis_method="Retain the comparison bounds from the accepted version-bounded replay determination.",
            observed_range_or_uncertainty="Four of four replay captures passed with maximum relative fixed-step delta 1 and contact-separation delta 0.001; no retries or unavailable components occurred.",
            decision_rule="Freeze those exact comparison maxima within the accepted version envelope.",
            rationale="Production replay must not silently relax the pilot-accepted deterministic comparison contract.",
        ),
        "termination_vocabulary": _evidence(
            records["pilot_report"], records["pilot_plan"], records["captures"],
            analysis_method="Take the three terminal reasons demonstrated by the accepted pilot and add only the specification-required numeric safety ceiling.",
            observed_range_or_uncertainty="level_clear, level_fail and stable_entered were observed; rollout_ceiling was not observed and is therefore non-quota-bearing.",
            decision_rule="Close the vocabulary to the three accepted classes plus rollout_ceiling; treat other exits as failures.",
            rationale="The vocabulary covers accepted engine termination and bounded execution without converting capture/operator failure into scientific termination.",
        ),
        "retry_policy": _evidence(
            records["pilot_report"], records["pilot_accounting"],
            analysis_method="Inspect all pilot retry decisions and permanent/transient classifications.",
            observed_range_or_uncertainty="Zero retries were accepted; four runtime failures were permanent and one mutation was invalidated/quarantined.",
            decision_rule="Freeze an empty transient retry map and one attempt per intervention.",
            rationale="No non-final pilot record justifies a positive transient retry count.",
        ),
        "bounded_negative": _evidence(
            records["pilot_report"], records["pilot_plan"],
            analysis_method="Apply the accepted central-v2 capability disposition for SPSG bounded-negative evidence.",
            observed_range_or_uncertainty="The accepted pilot marks evidence.bounded_negative not required for the central experiment.",
            decision_rule="Set the central cap to zero with a not-required named-secondary disposition; do not interpret the no-contact stratum as SPSG evidence.",
            rationale="This prevents an obsolete v1 negative cap from being promoted into central v2.",
        ),
        "noncentral_dispositions": _evidence(
            records["pilot_report"],
            analysis_method="Copy accepted issue-51 secondary and out-of-scope dispositions.",
            observed_range_or_uncertainty="No non-central capability was promoted or used to pass the representative audit.",
            decision_rule="Keep named-secondary and out-of-scope items quota-free and excluded predicates unavailable rather than false.",
            rationale="The production parameter plan remains bound to the approved central experiment.",
        ),
    }


def _parameter(value: Any, unit: str, evidence_id: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "evidence_id": evidence_id}


def _parameter_plan(sources: Mapping[str, Any], collection: Mapping[str, Any]) -> dict[str, Any]:
    records = _source_records(sources)
    macro = sources["macro_spec"]["numeric_inputs"]
    violation = sources["violation_spec"]["numeric_inputs"]
    replay = sources["replay_plan"]["comparison_rules"]
    return {
        "schema": PARAMETER_SCHEMA,
        "plan_version": PLAN_VERSION,
        "identity": PARAMETER_IDENTITY,
        "authority": {
            "github_issue": 52,
            "collection_plan_identity": collection["identity"],
            "accepted_pilot_report_identity": sources["pilot_report"]["identity"],
            "macro_derivation_spec_identity": sources["macro_spec"]["identity"],
            "physical_violation_derivation_spec_identity": sources["violation_spec"]["identity"],
            "replay_comparison_rules_identity": replay["identity"],
        },
        "parameters": {
            "capture": {
                "capture_stride_fixed_steps": _parameter(1, "fixed_steps", "capture_stride"),
                "stability_window_fixed_steps": _parameter(2, "fixed_steps", "stability_window"),
                "rollout_ceiling_fixed_steps": _parameter(600, "fixed_steps_after_pre_intervention_record", "rollout_ceiling"),
            },
            "geometric_tolerances": {
                "penetration_tolerance_unity_units": _parameter(violation["penetration_tolerance_unity_units"], "unity_units", "penetration_tolerance"),
                "support_minimum_vertical_contact_normal_component": _parameter(0.5, "unitless", "support_threshold"),
            },
            "motion_tolerances": {
                "linear_speed_squared_threshold": _parameter(macro["linear_speed_squared_threshold"], "unity_units_squared_per_second_squared", "motion_thresholds"),
                "angular_speed_threshold_degrees_per_second": _parameter(macro["angular_speed_threshold_degrees_per_second"], "degrees_per_second", "motion_thresholds"),
            },
            "numeric_tolerances": {
                "replay_maximum_contact_separation_delta": _parameter(replay["maximum_contact_separation_delta"], "unity_units", "replay_tolerances"),
                "replay_maximum_relative_fixed_step_delta": _parameter(replay["maximum_relative_fixed_step_delta"], "fixed_steps_relative_to_bird_launch", "replay_tolerances"),
            },
            "central_derivation_thresholds": {
                "steady_state_debounce_fixed_steps": _parameter(macro["debounce_fixed_steps"], "fixed_steps", "stability_window"),
                "floating_body_stability_window_fixed_steps": _parameter(violation["stability_window_fixed_steps"], "fixed_steps", "stability_window"),
                "steady_state_linear_speed_squared_maximum": _parameter(macro["linear_speed_squared_threshold"], "unity_units_squared_per_second_squared", "motion_thresholds"),
                "steady_state_absolute_angular_speed_maximum": _parameter(macro["angular_speed_threshold_degrees_per_second"], "degrees_per_second", "motion_thresholds"),
                "floating_body_linear_speed_squared_maximum": _parameter(violation["linear_speed_squared_threshold"], "unity_units_squared_per_second_squared", "motion_thresholds"),
                "floating_body_absolute_angular_speed_maximum": _parameter(violation["angular_speed_threshold_degrees_per_second"], "degrees_per_second", "motion_thresholds"),
                "excess_penetration_strict_threshold": _parameter(violation["penetration_tolerance_unity_units"], "unity_units", "penetration_tolerance"),
                "supports_strict_vertical_contact_normal_component": _parameter(0.5, "unitless", "support_threshold"),
            },
            "attempts": {
                "max_attempts_per_intervention": _parameter(1, "attempts", "retry_policy"),
                "transient_retry_counts": _parameter({}, "attempts_by_failure_code", "retry_policy"),
            },
            "termination": {
                "closed_vocabulary": _parameter(list(TERMINATION_VOCABULARY), "termination_reason", "termination_vocabulary"),
                "quota_bearing_classes": _parameter(list(SUPPORTED_TERMINATIONS), "termination_reason", "termination_vocabulary"),
            },
            "bounded_negative": {
                "cap": _parameter(0, "rollouts", "bounded_negative"),
                "status": _parameter("not_required_named_secondary_spsg_contrastive_loss_ablation", "disposition", "bounded_negative"),
            },
        },
        "noncentral_dispositions": deepcopy(collection["noncentral_dispositions"]),
        "source_records": records,
        "evidence": _parameter_evidence(sources, records),
    }


def _quota_values(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "quota" in value and "evidence_id" in value:
            return [value["quota"]]
        result: list[int] = []
        for item in value.values():
            result.extend(_quota_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_quota_values(item))
        return result
    return []


def _validate_evidence_references(value: Any, evidence: Mapping[str, Any]) -> None:
    if isinstance(value, Mapping):
        if "evidence_id" in value:
            evidence_id = value["evidence_id"]
            if evidence_id not in evidence:
                raise ValueError(f"Issue-52 plan references unknown evidence {evidence_id}")
        for item in value.values():
            _validate_evidence_references(item, evidence)
    elif isinstance(value, list):
        for item in value:
            _validate_evidence_references(item, evidence)


def _validate_derived_payloads(
    collection: Mapping[str, Any],
    parameters: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> None:
    if (
        collection.get("schema") != COLLECTION_SCHEMA
        or collection.get("plan_version") != PLAN_VERSION
        or collection.get("identity") != COLLECTION_IDENTITY
    ):
        raise ValueError("Issue-52 collection plan identity or version is unknown")
    if (
        parameters.get("schema") != PARAMETER_SCHEMA
        or parameters.get("plan_version") != PLAN_VERSION
        or parameters.get("identity") != PARAMETER_IDENTITY
    ):
        raise ValueError("Issue-52 parameter plan identity or version is unknown")
    if parameters.get("authority", {}).get("collection_plan_identity") != COLLECTION_IDENTITY:
        raise ValueError("Issue-52 parameter plan is not bound to its collection plan")
    expected_collection_authority = {
        "github_issue": 52,
        "capability_declaration_identity": sources["capabilities"]["identity"],
        "accepted_pilot_report_identity": sources["pilot_report"]["identity"],
        "accepted_pilot_plan_identity": sources["pilot_plan"]["identity"],
        "partition_manifest_identity": sources["partition"]["identity"],
    }
    if collection.get("authority") != expected_collection_authority:
        raise ValueError("Issue-52 collection-plan authority is stale")
    expected_parameter_authority = {
        "github_issue": 52,
        "collection_plan_identity": COLLECTION_IDENTITY,
        "accepted_pilot_report_identity": sources["pilot_report"]["identity"],
        "macro_derivation_spec_identity": sources["macro_spec"]["identity"],
        "physical_violation_derivation_spec_identity": sources["violation_spec"][
            "identity"
        ],
        "replay_comparison_rules_identity": sources["replay_plan"][
            "comparison_rules"
        ]["identity"],
    }
    if parameters.get("authority") != expected_parameter_authority:
        raise ValueError("Issue-52 parameter-plan authority is stale")

    assignments = collection.get("assignments")
    interventions = collection.get("interventions")
    if not isinstance(assignments, list) or not isinstance(interventions, list):
        raise ValueError("Issue-52 collection assignments or interventions are malformed")
    if [item.get("exposure_role") for item in assignments] != list(ROLE_ORDER):
        raise ValueError("Issue-52 collection assignments do not match the frozen roles")
    if [item.get("intended_coverage_stratum") for item in interventions] != list(CENTRAL_STRATA):
        raise ValueError("Issue-52 interventions do not cover the six central strata exactly")
    if [item.get("ordinal") for item in interventions] != list(range(1, 7)):
        raise ValueError("Issue-52 intervention ordering is not frozen")
    intervention_ids = [item.get("id") for item in interventions]
    if assignments != _assignments(sources["partition"], intervention_ids):
        raise ValueError("Issue-52 assignments differ from the frozen exposure manifest")
    for assignment in assignments:
        if assignment.get("intervention_ids") != intervention_ids or assignment.get("planned_rollout_quota") != 6:
            raise ValueError("Issue-52 assignment does not execute all frozen interventions")
    for intervention in interventions:
        if (
            intervention.get("interface_action", {}).get("drag_release")
            != intervention.get("engine_relative_action", {}).get("drag_delta_canvas_pixels")
        ):
            raise ValueError("Issue-52 interface and engine-relative actions differ")

    quotas = collection.get("quotas", {})
    total = quotas.get("total_planned_rollouts", {}).get("quota")
    if total != 24:
        raise ValueError("Issue-52 total production quota must be 24")
    additive_dimensions = (
        "benchmark_condition",
        "exposure_role",
        "instance_held_out_partition",
        "scenario_template_level_instance",
        "intervention_source",
        "central_coverage_stratum",
        "termination_class",
    )
    for dimension in additive_dimensions:
        if sum(_quota_values(quotas.get(dimension))) != total:
            raise ValueError(f"Issue-52 {dimension} quotas do not sum to the frozen total")
    if set(quotas.get("central_coverage_stratum", {})) != set(CENTRAL_STRATA):
        raise ValueError("Issue-52 central stratum quota keys are incomplete")
    if set(quotas.get("termination_class", {})) != set(SUPPORTED_TERMINATIONS):
        raise ValueError("Issue-52 termination quota keys are incomplete")

    attempt_policy = collection.get("attempt_policy", {})
    if (
        attempt_policy.get("max_attempts_per_intervention") != 1
        or attempt_policy.get("transient_failure_codes") != []
        or attempt_policy.get("retry_counts") != {}
        or attempt_policy.get("quota_fill_resampling") is not False
    ):
        raise ValueError("Issue-52 retry or outcome-independent attempt policy changed")
    if collection.get("bounded_negative", {}).get("status") != (
        "not_required_named_secondary_spsg_contrastive_loss_ablation"
    ):
        raise ValueError("Issue-52 bounded-negative disposition changed")

    parameter_values = parameters.get("parameters", {})
    capture = parameter_values.get("capture", {})
    expected_capture = {
        "capture_stride_fixed_steps": 1,
        "stability_window_fixed_steps": 2,
        "rollout_ceiling_fixed_steps": 600,
    }
    if {key: capture.get(key, {}).get("value") for key in expected_capture} != expected_capture:
        raise ValueError("Issue-52 capture parameters fall outside their source-bound decision rules")
    macro = sources["macro_spec"]["numeric_inputs"]
    violation = sources["violation_spec"]["numeric_inputs"]
    if macro["debounce_fixed_steps"] != 2 or violation["stability_window_fixed_steps"] != 2:
        raise ValueError("Accepted derivation stability-window authorities changed")
    if parameter_values.get("bounded_negative", {}).get("cap", {}).get("value") != 0:
        raise ValueError("Issue-52 bounded-negative cap must remain zero")

    for plan in (collection, parameters):
        evidence = plan.get("evidence")
        source_records = plan.get("source_records")
        if not isinstance(evidence, Mapping) or not evidence:
            raise ValueError("Issue-52 plan is missing derivation evidence")
        if not isinstance(source_records, Mapping) or not source_records:
            raise ValueError("Issue-52 plan is missing accepted source records")
        _validate_evidence_references(plan, evidence)
        for source_record_id, record in source_records.items():
            if source_record_id != record.get("source_record_id"):
                raise ValueError("Issue-52 source-record catalog identity is stale")
            if record.get("exposure_boundary") != "no_final_evaluation_outcomes":
                raise ValueError(
                    f"Issue-52 source record {source_record_id} crosses the exposure boundary"
                )
            if not record.get("record_ids"):
                raise ValueError(
                    f"Issue-52 source record {source_record_id} has no record IDs"
                )
        for evidence_id, item in evidence.items():
            if set(item) != {
                "plan_version",
                "source_record_ids",
                "analysis_method",
                "observed_range_or_uncertainty",
                "decision_rule",
                "rationale",
            }:
                raise ValueError(f"Issue-52 evidence {evidence_id} is incomplete")
            if item["plan_version"] != PLAN_VERSION or not item["source_record_ids"]:
                raise ValueError(f"Issue-52 evidence {evidence_id} is not source-bound")
            if any(
                source_record_id not in source_records
                for source_record_id in item["source_record_ids"]
            ):
                raise ValueError(
                    f"Issue-52 evidence {evidence_id} references an unknown source record"
                )

    expected_collection = _collection_plan(sources)
    if collection != expected_collection:
        raise ValueError("Issue-52 collection plan differs from its source-bound derivation")
    if parameters != _parameter_plan(sources, expected_collection):
        raise ValueError("Issue-52 parameter plan differs from its source-bound decision rules")


def derive_issue_52_payloads(
    repository_root: Path = ROOT, *, validate_pilot: bool = True
) -> dict[str, dict[str, Any]]:
    """Derive both plans exactly from the accepted issue-51 evidence boundary."""
    sources = _load_sources(Path(repository_root), validate_pilot=validate_pilot)
    collection = _collection_plan(sources)
    parameters = _parameter_plan(sources, collection)
    _validate_derived_payloads(collection, parameters, sources)
    return {
        "collection-plan.json": collection,
        "production-parameter-plan.json": parameters,
    }


def validate_issue_52_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    repository_root: Path = ROOT,
) -> None:
    """Validate plan structure and exact source-bound values without publishing."""
    if set(payloads) != {"collection-plan.json", "production-parameter-plan.json"}:
        raise ValueError("Issue-52 plan membership is incomplete")
    sources = _load_sources(Path(repository_root), validate_pilot=False)
    _validate_derived_payloads(
        payloads["collection-plan.json"],
        payloads["production-parameter-plan.json"],
        sources,
    )
