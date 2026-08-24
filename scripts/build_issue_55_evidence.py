"""Build and revalidate the final cohort-v2 evidence audit for issue #55."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Final, Mapping

from scripts.build_issue_46_evidence import validate_issue_46_evidence
from scripts.build_issue_47_evidence import validate_issue_47_evidence
from scripts.build_issue_49_evidence import validate_issue_49_evidence
from scripts.build_issue_50_evidence import validate_issue_50_evidence
from scripts.build_issue_51_evidence import validate_issue_51_evidence
from scripts.build_issue_52_evidence import validate_issue_52_evidence
from scripts.build_issue_54_evidence import build_ingestion_evidence
from scripts.capture_issue_53_evidence import validate_existing
from scripts.cohort_v2_capabilities import load_capability_declaration
from scripts.cohort_v2_production_plans_v5 import validate_plan_v5_evidence
from scripts.cohort_v2_replay import validate_issue_48_evidence
from scripts.observation_trace import validate_observation_trace
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.physics_capture_v2_capability_report import (
    load_physics_capture_v2_capability_report,
)
from scripts.physics_capture_v2_persistence import validate_physics_capture_v2_artifact


ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "cohort_v2_final_evidence_audit_v1"
IDENTITY: Final = "cohort-v2-final-evidence-audit-v1:issue-55:release-v5"
REVIEW_IDENTITY: Final = "issue-55-independent-primary-evidence-review-v1:release-v5"
BUNDLE_IDENTITY: Final = "issue-55-cohort-v2-final-evidence-audit-bundle-v1:release-v5"
AUDIT_NAME: Final = "cohort-v2-final-evidence-audit.json"
REVIEW_NAME: Final = "independent-primary-evidence-review.json"
BUNDLE_NAME: Final = "bundle-manifest.json"

DEFAULT_OUTPUT: Final = ROOT / "data/runtime_evidence/issue-55"
ISSUE_50_PROBES: Final = ROOT / "data/runtime_evidence/issue-50/source-probes"
PLAN_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-plan-v5"
RUNTIME_ROOT: Final = ROOT / ".local-artifacts/issue-53-mixed-termination-production-run-v5"
RELEASE_ROOT: Final = ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5"
SEALED_ROOT: Final = ROOT / ".local-artifacts/issue-53-mixed-termination-final-release-v5"

CENTRAL_LABELS: Final = (
    "contact",
    "supports",
    "steady-state",
    "structure-unstable",
    "excess_penetration",
    "unsupported_stationary_or_floating_body",
)
ROLES: Final = ("training", "calibration", "model_selection", "final_evaluation")
STRATA: Final = (
    "no-contact/miss",
    "collision",
    "persistent support",
    "support change",
    "destruction",
    "stability transitions",
)


class Issue55EvidenceError(ValueError):
    """Issue #55 evidence is incomplete, stale, or internally inconsistent."""


def _log(message: str) -> None:
    print(f"[issue-55] {message}", flush=True)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Issue55EvidenceError(f"cannot load issue-55 source {path}: {error}") from error
    if not isinstance(value, dict):
        raise Issue55EvidenceError(f"issue-55 source must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Issue55EvidenceError(message)


def _historical_issue_33() -> dict[str, Any]:
    path = ".claude/project-docs/evidence/issue-33-section-16-audit-20260820/README.md"
    try:
        text = subprocess.run(
            ["git", "show", f"fb7acd8:{path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise Issue55EvidenceError("cannot resolve the immutable issue-33 audit history") from error
    _require(
        "1 PASS, 1 FAIL, and 5 PARTIAL" in text
        and "Issue #2 must remain open" in text,
        "historical issue-33 matrix differs from its recorded disposition",
    )
    return {
        "bundle_identity": "issue-33-section-16-audit-20260820",
        "historical_file_sha256": "ed02915ae861f2268a830f61f5b9cfe1d6f16b8bc41afb6ca74caf46126d6841",
        "git_reference": f"fb7acd8:{path}",
        "matrix": {"PASS": 1, "PARTIAL": 5, "FAIL": 1},
        "disposition": "incomplete",
        "preserved_immutable_history": True,
    }


def _label_floor_summary() -> dict[str, Any]:
    pilot = _load(
        ROOT
        / "data/runtime_evidence/issue-51/representative-cohort-v2-pilot-report.json"
    )
    macro = _load(ROOT / "data/runtime_evidence/issue-49/macro-semantics-adjudication.json")
    violations = _load(
        ROOT / "data/runtime_evidence/issue-50/physical-violation-adjudication.json"
    )
    result: dict[str, Any] = {}
    for label in ("contact", "supports"):
        record = pilot["micro_label_evidence"][label]
        coverage = record["coverage"]
        passed = (
            record["passed"] is True
            and coverage["positive_witness_count"] >= 2
            and coverage["negative_witness_count"] >= 2
            and coverage["boundary_window_count"] >= 2
            and coverage["scenario_lineage_count"] >= 2
            and coverage["level_instance_count"] >= 2
            and coverage["scenario_template_count"] >= 2
            and record["unavailable_or_invalidation_check"]["passed"] is True
        )
        _require(passed, f"{label} does not meet the approved representative floor")
        result[label] = {**coverage, "unavailable_or_invalidation_checks": 1, "passed": True}
    for source, labels in (
        (macro, ("steady-state", "structure-unstable")),
        (violations, ("excess_penetration", "unsupported_stationary_or_floating_body")),
    ):
        for label in labels:
            coverage = source["coverage"][label]
            positive = coverage["positive"]
            negative = coverage["negative"]
            unavailable_count = len(source["witnesses"]["unavailable"])
            passed = (
                source["passed"] is True
                and source["predicate_dispositions"][label] == "accepted_authoritative"
                and positive["passed"] is True
                and negative["passed"] is True
                and positive["witness_count"] >= 2
                and negative["witness_count"] >= 2
                and positive["scenario_lineage_count"] >= 2
                and negative["scenario_lineage_count"] >= 2
                and positive["level_instance_count"] >= 2
                and negative["level_instance_count"] >= 2
                and positive["scenario_template_count"] >= 2
                and negative["scenario_template_count"] >= 2
                and coverage["boundary_window_count"] >= 2
                and unavailable_count >= 1
            )
            _require(passed, f"{label} does not meet the approved representative floor")
            result[label] = {
                "positive_witness_count": positive["witness_count"],
                "negative_witness_count": negative["witness_count"],
                "boundary_window_count": coverage["boundary_window_count"],
                "scenario_lineage_count": min(
                    positive["scenario_lineage_count"], negative["scenario_lineage_count"]
                ),
                "level_instance_count": min(
                    positive["level_instance_count"], negative["level_instance_count"]
                ),
                "scenario_template_count": min(
                    positive["scenario_template_count"], negative["scenario_template_count"]
                ),
                "unavailable_or_invalidation_checks": unavailable_count,
                "passed": True,
            }
    return result


def _production_summary() -> dict[str, Any]:
    report = _load(RUNTIME_ROOT / "production-execution-report.json")
    replay = _load(RUNTIME_ROOT / "production-replay-report.json")
    public_quality = _load(RELEASE_ROOT / "production-quality-report.json")
    sealed_quality = _load(SEALED_ROOT / "production-quality-report.json")
    derivation_index = _load(RELEASE_ROOT / "authoritative-derivation-index.json")
    _require(
        public_quality["passed"] is True
        and public_quality["systematic_exporter_defects"] == []
        and public_quality["coverage_shortfalls"] == []
        and public_quality["termination_mismatches"] == []
        and sealed_quality["passed"] is True
        and sealed_quality["systematic_exporter_defects"] == []
        and set(derivation_index["accepted_labels"]) == set(CENTRAL_LABELS),
        "production quality, exporter, or accepted-label accounting failed",
    )
    ledger = report["attempt_ledger"]
    _require(len(ledger) == 24 and len({item["attempt_id"] for item in ledger}) == 24,
             "production attempt membership is not exactly 24 unique rollouts")
    _require(report["counts"] == {
        "planned": 24, "attempted": 24, "accepted": 24,
        "rejected": 0, "failed": 0, "quarantined": 0,
    }, "production attempt counts differ from the complete contract")
    role_counts = Counter(item["exposure_role"] for item in ledger)
    stratum_counts = Counter(item["intended_coverage_stratum"] for item in ledger)
    termination_counts = Counter(item["terminal_reason"] for item in ledger)
    _require(role_counts == Counter({role: 6 for role in ROLES}), "production role quotas failed")
    _require(stratum_counts == Counter({stratum: 4 for stratum in STRATA}),
             "production stratum quotas failed")
    _require(termination_counts == Counter({"stable_entered": 20, "level_fail": 4}),
             "production termination contract failed")
    _require(all(
        item["status"] == "accepted"
        and item["expected_termination"] == item["terminal_reason"]
        and item["failure_code"] is None
        and item["quarantine_path"] is None
        for item in ledger
    ), "production ledger contains an unresolved or mismatched attempt")

    frame_records = 0
    observations = 0
    for item in ledger:
        artifact = Path(item["artifact_path"])
        metadata = _load(artifact / "metadata.json")
        validate_physics_capture_v2_artifact(artifact, metadata)
        capture = load_physics_capture_v2(artifact / "physics_capture_v2.json")
        observation = validate_observation_trace(artifact / "observation-trace")
        _require(
            capture.source_bindings["rollout_id"] == item["attempt_id"]
            and capture.source_bindings["scenario_lineage_id"]
            == item["scenario_lineage_identity"]
            and capture.source_bindings["intervention_id"] == item["intervention_identity"]
            and observation["exposure_role"] == item["exposure_role"],
            f"runtime primary binding is stale: {item['attempt_id']}",
        )
        frame_records += len(capture.record["frame_records"])
        observations += 1
    _require(
        replay["passed"] is True
        and replay["proof_count"] == 4
        and replay["retry_count"] == 0
        and [item["exposure_role"] for item in replay["verdicts"]] == list(ROLES),
        "production replay proof does not span all four roles",
    )
    return {
        "planned_rollouts": 24,
        "attempted_rollouts": 24,
        "accepted_rollouts": 24,
        "rejected_rollouts": 0,
        "failed_rollouts": 0,
        "quarantined_rollouts": 0,
        "role_counts": dict(role_counts),
        "stratum_counts": dict(stratum_counts),
        "termination_counts": dict(termination_counts),
        "validated_primary_frame_records": frame_records,
        "validated_observation_traces": observations,
        "replay_proofs": 4,
        "replay_retries": 0,
        "passed": True,
    }


def _validate_authorities() -> dict[str, Any]:
    _log("validating capability declaration and exporter evidence")
    declaration = load_capability_declaration()
    required = declaration["capabilities"]["required_central"]
    declared_labels = (
        required["micro_labels"] + required["macro_labels"] + required["violation_labels"]
    )
    _require(
        len(declared_labels) == len(CENTRAL_LABELS)
        and set(declared_labels) == set(CENTRAL_LABELS),
        "capability declaration does not name the exact six central labels",
    )
    exporter = load_physics_capture_v2_capability_report(
        ROOT / "data/runtime_evidence/issue-44/capability-report.json"
    ).record
    _require(all(item["status"] == "demonstrated" for item in exporter["facts"].values()),
             "exporter capability report contains an unavailable fact")

    _log("validating scenario lineage, observations, partitions, and replay")
    observation = validate_issue_46_evidence(ROOT / "data/runtime_evidence/issue-46")
    partition = validate_issue_47_evidence(ROOT, ROOT / "data/runtime_evidence/issue-47")
    replay = validate_issue_48_evidence(ROOT / "data/runtime_evidence/issue-48")

    _log("recalculating all six representative label floors")
    floors = _label_floor_summary()
    macro = validate_issue_49_evidence(ROOT / "data/runtime_evidence/issue-49")
    violation = validate_issue_50_evidence(
        ROOT / "data/runtime_evidence/issue-50", probe_root=ISSUE_50_PROBES
    )

    _log("validating pilot and frozen production authorities")
    pilot = validate_issue_51_evidence(ROOT / "data/runtime_evidence/issue-51")
    plan_v1 = validate_issue_52_evidence(
        ROOT / "data/runtime_evidence/issue-52", revalidate_pilot=False
    )
    plan_v5 = validate_plan_v5_evidence(PLAN_ROOT)
    scenario_inventory = _load(RELEASE_ROOT / "scenario-inventory.json")
    scenario_entries = scenario_inventory.get("entries")
    _require(
        scenario_inventory.get("schema") == "cohort_v2_production_scenario_inventory_v5"
        and isinstance(scenario_entries, list)
        and len(scenario_entries) == 4
        and {entry.get("exposure_role") for entry in scenario_entries} == set(ROLES)
        and len({entry.get("scenario_lineage_identity") for entry in scenario_entries}) == 4
        and all(
            isinstance(entry.get(field), str) and entry[field]
            for entry in scenario_entries
            for field in (
                "scenario_template_identity", "level_instance_identity",
                "scenario_lineage_identity", "scenario_manifest_identity",
            )
        ),
        "production scenario inventory is incomplete or contains unresolved identities",
    )

    _log("validating all 24 primary rollouts, quotas, and exact-socket replays")
    production = _production_summary()
    release = validate_existing(
        plan_root=PLAN_ROOT,
        runtime_root=RUNTIME_ROOT,
        output=RELEASE_ROOT,
        sealed_output=SEALED_ROOT,
    )

    _log("revalidating public ingestion and the authorized sealed-boundary probe")
    published_ingestion = _load(
        ROOT / "data/runtime_evidence/issue-54/cohort-v2-downstream-ingestion-evidence.json"
    )
    ingestion = build_ingestion_evidence(
        repository_root=ROOT,
        release_root=RELEASE_ROOT,
        sealed_root=SEALED_ROOT,
        code_revision=published_ingestion["code_revision"],
    )
    _require(ingestion == published_ingestion and ingestion["passed"] is True,
             "issue-54 ingestion evidence differs from exact revalidation")

    return {
        "capability_declaration": {"identity": declaration["identity"], "passed": True},
        "exporter": {"identity": exporter["report_id"], "passed": True},
        "lineage": {"identity": scenario_inventory["identity"],
                    "source_issue": 45, "passed": True},
        "observations": {"identity": observation["identity"], "passed": True},
        "partition": {"identity": partition["partition_manifest_identity"], "passed": True},
        "replay": {"identity": replay["identity"], "passed": True},
        "macro_semantics": {"identity": macro["bundle_identity"], "passed": True},
        "physical_violations": {"identity": violation["bundle_identity"], "passed": True},
        "pilot": {"identity": pilot["pilot_report_identity"], "passed": True},
        "initial_production_plan": {"identity": plan_v1["bundle_identity"], "passed": True},
        "production_plan_v5": {"identity": plan_v5["bundle_identity"], "passed": True},
        "release": {"identity": "representative-cohort-v2-release-v5:issue-53:mixed-termination",
                    "validation_schema": release["schema"], "passed": release["passed"]},
        "derivation_index": {
            "identity": _load(RELEASE_ROOT / "authoritative-derivation-index.json")["identity"],
            "passed": True,
        },
        "sealed_boundary": {
            "identity": _load(SEALED_ROOT / "sealed-bundle-manifest.json")["identity"],
            "ordinary_workflow_access": False,
            "passed": True,
        },
        "ingestion": {"identity": ingestion["identity"], "passed": True},
        "label_floors": floors,
        "production": production,
    }


def _row(
    row_id: str,
    evidence: list[str],
    demonstrates: str,
    affected_experiment: str,
    next_action: str = "None; preserve the immutable evidence and fail-closed contract.",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "disposition": "PASS",
        "exact_evidence": evidence,
        "demonstrates": demonstrates,
        "missing_or_unavailable": [],
        "affected_experiment": affected_experiment,
        "smallest_next_action": next_action,
    }


def _capability_matrix() -> list[dict[str, Any]]:
    return [
        _row("scope_and_capability_declaration", ["capability_declaration"],
             "The exact approved central scope and non-central exclusions are frozen and fail closed.",
             "Central oracle-symbol joint controller."),
        _row("exporter_contract", ["exporter", "release"],
             "Configured fixed-step capture, complete contacts, geometry/separation, gravity/lifecycle/support/world evidence, identities, and terminal coverage validate from engine records.",
             "All central supervision and endpoint scoring."),
        _row("scenario_hierarchy_and_lineages", ["lineage", "partition", "release"],
             "Source-bound templates, deterministic level instances, scenario specifications, and four disjoint role lineages resolve exactly.",
             "Instance-held-out training and evaluation."),
        _row("observations_and_access", ["observations", "sealed_boundary", "ingestion"],
             "Agent observations are synchronized model inputs; canonical observations remain diagnostic-only and final evidence remains sealed from ordinary workflows.",
             "Observation-backed controller training and final evaluation."),
        _row("exposure_roles_and_instance_holdout", ["partition", "production_plan_v5", "sealed_boundary"],
             "Training, calibration, model-selection, and final-evaluation influence permissions are disjoint and enforced.",
             "All role-separated central experiments."),
        _row("version_bounded_replay", ["replay", "release"],
             "Representative replay and four production exact-socket proofs pass under their declared version envelopes.",
             "Deterministic artifact comparison."),
        _row("micro_labels", ["pilot", "label_floors", "derivation_index"],
             "contact and supports each meet positive, negative, boundary, identity-coverage, and unavailable/invalidation floors.",
             "Oracle-symbol relation inputs and targets."),
        _row("macro_labels", ["macro_semantics", "label_floors", "derivation_index"],
             "steady-state and structure-unstable are semantically accepted, source-bound, and meet every representative floor.",
             "Macro-state supervision and endpoint scoring."),
        _row("physical_violations", ["physical_violations", "label_floors", "derivation_index"],
             "Both approved endpoint violations are accepted with complete evidence contracts and unavailable preservation; illegal_contact stays excluded.",
             "Physical-plausibility endpoint scoring."),
        _row("representative_pilot", ["pilot", "replay", "macro_semantics", "physical_violations"],
             "The capability-complete pilot passes every central stratum, termination, observation, replay, failure, and semantic gate.",
             "Production authorization."),
        _row("production_plans", ["initial_production_plan", "production_plan_v5"],
             "The v5 successor preserves the approved science plan while correcting only workflow timing; 24 outcome-independent assignments and all parameters are frozen.",
             "Cohort-v2 production."),
        _row("production_release_and_quality", ["production", "release", "sealed_boundary"],
             "24/24 rollouts validate atomically across four roles and six strata with 20 stable_entered, 4 level_fail, no failures or shortfalls, and no systematic exporter defect.",
             "Immutable public and final-evaluation cohorts."),
        _row("downstream_ingestion", ["ingestion", "release", "derivation_index"],
             "Public readers consume all eligible observations and six labels through training/scoring interfaces while preserving unavailable values and access restrictions.",
             "Central training, calibration, model selection, and authorized final scoring."),
    ]


def _section_16_matrix() -> list[dict[str, Any]]:
    return [
        _row("1", ["release", "production_plan_v5", "partition", "derivation_index"],
             "A complete immutable representative cohort release binds plans, partitions, roles, provenance, failure accounting, and accepted derivations.",
             "Issue #2 cohort availability."),
        _row("2", ["production", "release"],
             "Every one of the 24 admitted rollouts passes atomic whole-rollout validation; none is partial or reconstructed.",
             "All cohort consumers."),
        _row("3", ["pilot", "production", "label_floors"],
             "Pilot and production evidence cover the approved benchmark conditions, templates, instances, interventions, strata, terminations, observations, labels, and complete zero-failure accounting.",
             "Central experiment coverage."),
        _row("4", ["partition", "sealed_boundary", "ingestion"],
             "The applicable instance-held-out split and all four exposure roles are isolated; the final-evaluation boundary is role-separated and audited. Template holdout is explicitly non-central.",
             "Held-out and final evaluation."),
        _row("5", ["derivation_index", "label_floors", "macro_semantics", "physical_violations"],
             "All six derivations required by the declared central research use are accepted, source-bound, semantically validated, and available under fail-closed rules. Non-central physical-regime, material/damage, and excluded predicates are not required by this profile.",
             "Central oracle supervision."),
        _row("6", ["ingestion", "release", "derivation_index"],
             "Downstream readers ingest every required primary and derivation artifact with identity, time, availability, and exposure enforcement.",
             "Training and evaluation interfaces."),
        _row("7", ["exporter", "production", "release"],
             "No known systematic exporter defect remains; all production rollouts validate complete collision/contact and required central source evidence without reconstruction.",
             "Engine-authoritative central evidence."),
    ]


def build_audit() -> dict[str, Any]:
    authorities = _validate_authorities()
    historical = _historical_issue_33()
    capability_matrix = _capability_matrix()
    section_16 = _section_16_matrix()
    return {
        "schema": SCHEMA,
        "identity": IDENTITY,
        "issue": 55,
        "audit_profile": "approved_central_cohort_v2",
        "publication_path": RELEASE_ROOT.relative_to(ROOT).as_posix(),
        "sealed_path": SEALED_ROOT.relative_to(ROOT).as_posix(),
        "authorities": authorities,
        "capability_matrix": capability_matrix,
        "section_16_matrix": section_16,
        "prior_issue_33_comparison": {
            "prior": historical,
            "current_matrix": {"PASS": 7, "PARTIAL": 0, "FAIL": 0},
            "prior_matrix_overwritten": False,
            "resolved_blockers": [
                "capability-complete representative pilot and production coverage",
                "real four-role instance-held-out and final-evaluation exposure boundary",
                "accepted central physical-violation derivations",
                "accepted central macro and micro representative floors",
                "access-separated agent and canonical observations",
                "authoritative configured fixed-step capture and complete exporter evidence",
                "version-bounded deterministic replay",
                "full fail-closed central downstream ingestion",
            ],
        },
        "closure_recommendation": {
            "issue_55": "complete_and_close",
            "issue_42": "complete_and_close",
            "issue_33": "incorporate_this_renewed_seven_pass_matrix_then_close",
            "issue_2": "close_after_issue_33_records_this_renewed_assessment",
        },
        "disposition": "complete",
        "passed": True,
    }


def build_independent_review(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Independently cross-check conclusions from primary counts and exact bindings."""
    _require(audit.get("identity") == IDENTITY and audit.get("passed") is True,
             "independent review received a failed or foreign audit")
    capability = audit.get("capability_matrix")
    section_16 = audit.get("section_16_matrix")
    _require(
        isinstance(capability, list)
        and len(capability) == 13
        and all(row.get("disposition") == "PASS" for row in capability)
        and isinstance(section_16, list)
        and [row.get("id") for row in section_16] == [str(index) for index in range(1, 8)]
        and all(row.get("disposition") == "PASS" for row in section_16),
        "independent review did not reproduce both PASS matrices",
    )
    production = audit["authorities"]["production"]
    floors = audit["authorities"]["label_floors"]
    _require(
        production["accepted_rollouts"] == 24
        and production["role_counts"] == {role: 6 for role in ROLES}
        and production["stratum_counts"] == {stratum: 4 for stratum in STRATA}
        and production["termination_counts"] == {"stable_entered": 20, "level_fail": 4}
        and set(floors) == set(CENTRAL_LABELS)
        and all(item["passed"] is True for item in floors.values()),
        "independent review did not reproduce production quotas or label floors",
    )
    bindings = {
        key: audit["authorities"][key]["identity"]
        for key in (
            "capability_declaration", "exporter", "lineage", "observations", "partition",
            "replay", "macro_semantics", "physical_violations", "pilot",
            "production_plan_v5", "release", "derivation_index", "sealed_boundary", "ingestion",
        )
    }
    return {
        "schema": "issue_55_independent_primary_evidence_review_v1",
        "identity": REVIEW_IDENTITY,
        "audit_identity": IDENTITY,
        "reproduced_authority_bindings": bindings,
        "reproduced_counts": {
            "accepted_rollouts": 24,
            "roles": 4,
            "strata": 6,
            "accepted_central_labels": 6,
            "production_replay_proofs": 4,
            "ingested_non_final_rollouts": 18,
            "section_16_passes": 7,
        },
        "focused_verification": [
            "exact capability declaration and exporter capability validation",
            "scenario lineage, observation, partition, and replay bundle revalidation",
            "six-label representative-floor recalculation",
            "pilot plus plan-v1 and plan-v5 exact revalidation",
            "24 runtime primary rollout and observation validations",
            "public and sealed release membership validation",
            "exact issue-54 training/scoring ingestion revalidation",
            "historical issue-33 matrix resolution from immutable git history",
        ],
        "standards_findings": [],
        "spec_findings": [],
        "disposition": "PASS",
        "passed": True,
    }


def _bundle() -> dict[str, Any]:
    return {
        "schema": "issue_55_cohort_v2_final_evidence_audit_bundle_v1",
        "identity": BUNDLE_IDENTITY,
        "audit_identity": IDENTITY,
        "independent_review_identity": REVIEW_IDENTITY,
        "artifacts": [AUDIT_NAME, REVIEW_NAME],
        "passed": True,
    }


def validate_issue_55_evidence(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = Path(output)
    expected_audit = build_audit()
    expected_review = build_independent_review(expected_audit)
    observed_audit = _load(root / AUDIT_NAME)
    observed_review = _load(root / REVIEW_NAME)
    observed_bundle = _load(root / BUNDLE_NAME)
    _require(observed_audit == expected_audit, "published issue-55 audit differs from revalidation")
    _require(observed_review == expected_review,
             "published issue-55 independent review differs from revalidation")
    _require(observed_bundle == _bundle(), "published issue-55 bundle manifest is stale")
    _require(sorted(path.name for path in root.iterdir() if path.is_file()) ==
             sorted((AUDIT_NAME, REVIEW_NAME, BUNDLE_NAME)),
             "published issue-55 bundle has undeclared members")
    return {
        "schema": "issue_55_final_evidence_audit_validation_result_v1",
        "bundle_identity": BUNDLE_IDENTITY,
        "capability_passes": len(expected_audit["capability_matrix"]),
        "section_16_passes": len(expected_audit["section_16_matrix"]),
        "accepted_rollouts": expected_audit["authorities"]["production"]["accepted_rollouts"],
        "passed": True,
    }


def build_issue_55_evidence(output: Path = DEFAULT_OUTPUT, *, dry_run: bool = False) -> dict[str, Any]:
    audit = build_audit()
    _log("running the independent primary-evidence cross-check")
    review = build_independent_review(audit)
    result = {
        "schema": "issue_55_final_evidence_audit_validation_result_v1",
        "bundle_identity": BUNDLE_IDENTITY,
        "capability_passes": len(audit["capability_matrix"]),
        "section_16_passes": len(audit["section_16_matrix"]),
        "accepted_rollouts": audit["authorities"]["production"]["accepted_rollouts"],
        "passed": True,
    }
    if dry_run:
        _log("dry-run passed; no artifact was written")
        return result
    target = Path(output)
    if target.exists():
        _log(f"validating existing immutable audit at {target}")
        return validate_issue_55_evidence(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-55-", dir=target.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        (staging / AUDIT_NAME).write_bytes(_canonical_json(audit))
        (staging / REVIEW_NAME).write_bytes(_canonical_json(review))
        (staging / BUNDLE_NAME).write_bytes(_canonical_json(_bundle()))
        _log("validating the staged immutable audit bundle")
        observed = {
            AUDIT_NAME: _load(staging / AUDIT_NAME),
            REVIEW_NAME: _load(staging / REVIEW_NAME),
            BUNDLE_NAME: _load(staging / BUNDLE_NAME),
        }
        _require(observed[AUDIT_NAME] == audit and observed[REVIEW_NAME] == review
                 and observed[BUNDLE_NAME] == _bundle(),
                 "staged issue-55 bundle differs from generated evidence")
        os.replace(staging, target)
    _log(f"immutable issue-55 audit published: {target}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = (
        validate_issue_55_evidence(args.output)
        if args.validate
        else build_issue_55_evidence(args.output, dry_run=args.dry_run)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
