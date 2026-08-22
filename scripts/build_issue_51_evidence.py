#!/usr/bin/env python3
"""Build and revalidate the capability-complete cohort-v2 pilot report."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from scripts.build_issue_46_evidence import validate_issue_46_evidence
from scripts.build_issue_47_evidence import validate_issue_47_evidence
from scripts.build_issue_49_evidence import validate_issue_49_evidence
from scripts.build_issue_50_evidence import validate_issue_50_evidence
from scripts.cohort_v2_replay import validate_issue_48_evidence
from scripts.cohort_v2_scenarios import (
    load_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_json,
)
from scripts.collection_plan import load_collection_plan
from scripts.collect_rollouts import (
    classify_physics_capture_v2_coverage,
    collect_fresh_engine_attempt,
)
from scripts.physics_capture_v2 import (
    PhysicsCaptureV2Error,
    parse_physics_capture_v2,
)
from scripts.physics_capture_v2_capability_report import (
    validate_physics_capture_v2_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-51"
CAPABILITY_DECLARATION = ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"
ISSUE_ROOTS = {
    issue: ROOT / f"data/runtime_evidence/issue-{issue}"
    for issue in range(44, 51)
}
PILOT_PLAN_IDENTITY = (
    "representative-cohort-v2-pilot-plan-v2:cohort-v2-capabilities-v1:"
    "issues-44-through-51:determination-2"
)
PILOT_REPORT_IDENTITY = (
    "representative-cohort-v2-pilot-report-v1:accepted-determination-1"
)
BUNDLE_IDENTITY = "issue-51-representative-cohort-v2-pilot-bundle-v1:accepted-1"
PRIOR_DETERMINATION_IDENTITY = (
    "issue-51-representative-pilot-determination-1:failed-level-clear"
)
SUPPORTED_TERMINATIONS = ("level_clear", "level_fail", "stable_entered")
IMPLEMENTATION_PATHS = (
    "scripts/build_issue_51_evidence.py",
    "scripts/build_issue_51_pilot_plan.py",
    "scripts/capture_issue_51_evidence.py",
)


class Issue51EvidenceError(ValueError):
    """The issue #51 pilot cannot be accepted under its frozen plan."""


def _log(message: str) -> None:
    print(f"[issue-51] {message}", flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Issue51EvidenceError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise Issue51EvidenceError(f"{path} must contain a JSON object")
    return value


def _implementation_revision(repository_root: Path) -> str:
    revision = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *IMPLEMENTATION_PATHS],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not revision:
        raise Issue51EvidenceError(
            "issue-51 implementation must be committed before canonical publication"
        )
    return revision


def _execution_revision(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_tracked_worktree(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise Issue51EvidenceError(
            "canonical issue-51 publication requires a clean tracked worktree"
        )


def _load_issue_44(repository_root: Path) -> dict[str, Any]:
    root = repository_root / "data/runtime_evidence/issue-44"
    report = _load_json(root / "capability-report.json")
    validate_physics_capture_v2_capability_report(report)
    runtime = _load_json(root / "runtime-bundle-manifest.json")
    bundle = _load_json(root / "capture-bundle-manifest.json")
    if (
        runtime.get("schema") != "issue_44_physics_v2_runtime_evidence_bundle_v1"
        or bundle.get("schema") != "issue_44_physics_v2_capture_bundle_v1"
        or bundle.get("runtime_evidence_bundle_identity") != runtime.get("identity")
        or runtime.get("capability_report_path")
        != "data/runtime_evidence/issue-44/capability-report.json"
    ):
        raise Issue51EvidenceError("issue-44 runtime/capture binding is stale")
    captures: dict[str, Any] = {}
    entries = bundle.get("captures")
    if not isinstance(entries, dict) or set(entries) != {
        "no-contact", "collision", "support", "support-change", "stable-terminal"
    }:
        raise Issue51EvidenceError("issue-44 capture membership is incomplete")
    for case, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"capture_id", "path"}:
            raise Issue51EvidenceError(f"issue-44 {case} capture entry is invalid")
        path = repository_root / entry["path"]
        capture = parse_physics_capture_v2(_load_json(path))
        if capture.capture_id != entry["capture_id"]:
            raise Issue51EvidenceError(f"issue-44 {case} capture identity is stale")
        captures[case] = capture
    report_ids = {probe["capture_id"] for probe in report["probes"]}
    if report_ids != {capture.capture_id for capture in captures.values()}:
        raise Issue51EvidenceError("issue-44 capability report cites a different capture set")
    if any(fact["status"] != "demonstrated" for fact in report["facts"].values()):
        raise Issue51EvidenceError("issue-44 has an unavailable required exporter fact")
    return {"report": report, "runtime": runtime, "bundle": bundle, "captures": captures}


def _validate_component_sources(repository_root: Path) -> dict[str, Any]:
    issue44 = _load_issue_44(repository_root)
    issue46 = validate_issue_46_evidence(repository_root / "data/runtime_evidence/issue-46")
    with redirect_stdout(io.StringIO()):
        issue47 = validate_issue_47_evidence(
            repository_root,
            repository_root / "data/runtime_evidence/issue-47",
        )
    issue48 = validate_issue_48_evidence(repository_root / "data/runtime_evidence/issue-48")
    issue49 = validate_issue_49_evidence(
        repository_root / "data/runtime_evidence/issue-49",
        repository_root=repository_root,
    )
    issue50 = validate_issue_50_evidence(
        repository_root / "data/runtime_evidence/issue-50",
        probe_root=repository_root / "data/runtime_evidence/issue-50/source-probes",
    )
    return {
        "issue44": issue44,
        "issue46": issue46,
        "issue47": issue47,
        "issue48": issue48,
        "issue49": issue49,
        "issue50": issue50,
    }


def _component_identities(repository_root: Path, sources: Mapping[str, Any]) -> dict[str, str]:
    issue47_bundle = _load_json(repository_root / "data/runtime_evidence/issue-47/bundle-manifest.json")
    issue49_bundle = _load_json(repository_root / "data/runtime_evidence/issue-49/bundle-manifest.json")
    issue50_bundle = _load_json(repository_root / "data/runtime_evidence/issue-50/bundle-manifest.json")
    return {
        "physics_exporter_capability_report": sources["issue44"]["report"]["report_id"],
        "physics_capture_bundle": sources["issue44"]["bundle"]["identity"],
        "scenario_and_partition_bundle": issue47_bundle["identity"],
        "observation_bundle": sources["issue46"]["identity"],
        "replay_bundle": sources["issue48"]["identity"],
        "macro_semantics_bundle": issue49_bundle["identity"],
        "physical_violation_bundle": issue50_bundle["identity"],
    }


def build_pilot_plan(
    repository_root: Path,
    component_identities: Mapping[str, str],
    supplementary_plan_identity: str,
) -> dict[str, Any]:
    declaration = _load_json(repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json")
    if declaration.get("identity") != "cohort-v2-capabilities-v1":
        raise Issue51EvidenceError("the cohort-v2 capability declaration identity is stale")
    floors = declaration.get("evidence_floor")
    if not isinstance(floors, dict):
        raise Issue51EvidenceError("the cohort-v2 evidence floor is missing")
    return {
        "schema": "representative_cohort_v2_pilot_plan_v1",
        "identity": PILOT_PLAN_IDENTITY,
        "authority": {
            "github_issue": 51,
            "capability_declaration_identity": declaration["identity"],
            "capability_declaration_path": "docs/data_contracts/cohort_v2_capabilities_v1.json",
        },
        "prospective_quota_basis": (
            "the exact declaration floor and issue-51 strata/termination requirements; "
            "quotas are not selected from realized outcomes"
        ),
        "evidence_floor": floors,
        "coverage_stratum_quotas": {
            stratum: 1
            for stratum in declaration["capabilities"]["required_central"]["coverage_strata"]
        },
        "supported_termination_quotas": {
            termination: 1 for termination in SUPPORTED_TERMINATIONS
        },
        "component_evidence_identities": dict(sorted(component_identities.items())),
        "prior_failed_determination_identity": PRIOR_DETERMINATION_IDENTITY,
        "supplementary_level_clear_plan_identity": supplementary_plan_identity,
        "attempt_policy": {
            "outcome_independent_retention": True,
            "maximum_attempts_per_intervention": 1,
            "transient_failure_codes": [],
            "retry_policy": "no retries in the bounded pilot",
            "atomic_disposition": "accept complete rollout or quarantine complete attempt",
        },
        "final_evaluation": {
            "access": "sealed",
            "evidence_allowed": False,
            "consumed": False,
        },
    }


def _relation_sets(capture: Any, predicate: str) -> list[tuple[int, set[tuple[str, str]]]]:
    values = []
    for sample in capture.record["fixed_step_samples"]:
        if predicate == "contact":
            relations = {
                tuple(sorted((item["entity_a_id"], item["entity_b_id"])))
                for item in sample["contacts"]
            }
        else:
            relations = {
                (item["supporter_entity_id"], item["supported_entity_id"])
                for item in sample["supports"]
            }
        values.append((sample["fixed_step"], relations))
    return values


def _witness(capture: Any, step: int, relation: tuple[str, str], value: bool) -> dict[str, Any]:
    bindings = capture.source_bindings
    return {
        "capture_id": capture.capture_id,
        "fixed_step": step,
        "relation": list(relation),
        "value": value,
        "scenario_lineage_id": bindings["scenario_lineage_id"],
        "level_instance_id": bindings["level_instance_id"],
        "scenario_template_id": bindings["scenario_template_id"],
    }


def _micro_audit(captures: Mapping[str, Any], predicate: str) -> dict[str, Any]:
    by_template: dict[str, Any] = {}
    for capture in captures.values():
        by_template.setdefault(capture.source_bindings["scenario_template_id"], capture)
    positives = []
    negatives = []
    boundaries = []
    for capture in by_template.values():
        entity_ids = tuple(capture.record["causal_entities"])
        possible = (
            {
                tuple(sorted((left, right)))
                for index, left in enumerate(entity_ids)
                for right in entity_ids[index + 1:]
            }
            if predicate == "contact"
            else {(left, right) for left in entity_ids for right in entity_ids if left != right}
        )
        for step, relations in _relation_sets(capture, predicate):
            if relations and not any(item["capture_id"] == capture.capture_id for item in positives):
                positives.append(_witness(capture, step, sorted(relations)[0], True))
            absent = possible - relations
            if absent and not any(item["capture_id"] == capture.capture_id for item in negatives):
                negatives.append(_witness(capture, step, sorted(absent)[0], False))
            if positives and negatives and positives[-1]["capture_id"] == capture.capture_id and negatives[-1]["capture_id"] == capture.capture_id:
                break
    for capture in captures.values():
        relations = _relation_sets(capture, predicate)
        for (before_step, before), (after_step, after) in zip(relations, relations[1:]):
            if before != after:
                boundaries.append({
                    "capture_id": capture.capture_id,
                    "before_fixed_step": before_step,
                    "after_fixed_step": after_step,
                    "removed": [list(item) for item in sorted(before - after)],
                    "added": [list(item) for item in sorted(after - before)],
                })
                if len(boundaries) == 2:
                    break
        if len(boundaries) == 2:
            break
    mutation = deepcopy(next(iter(captures.values())).record)
    mutation["fixed_step_samples"][0]["complete_raw_non_trigger_contacts"] = False
    try:
        parse_physics_capture_v2(mutation)
    except PhysicsCaptureV2Error as error:
        unavailable = {"passed": True, "reason": str(error)}
    else:
        unavailable = {"passed": False, "reason": "incomplete contacts were accepted"}
    lineage_count = len({item["scenario_lineage_id"] for item in positives})
    level_count = len({item["level_instance_id"] for item in positives})
    template_count = len({item["scenario_template_id"] for item in positives})
    passed = (
        len(positives) >= 2
        and len(negatives) >= 2
        and min(lineage_count, level_count, template_count) >= 2
        and len(boundaries) >= 2
        and unavailable["passed"]
    )
    return {
        "positive": positives,
        "negative": negatives,
        "boundary_windows": boundaries,
        "unavailable_or_invalidation_check": unavailable,
        "coverage": {
            "positive_witness_count": len(positives),
            "negative_witness_count": len(negatives),
            "boundary_window_count": len(boundaries),
            "scenario_lineage_count": lineage_count,
            "level_instance_count": level_count,
            "scenario_template_count": template_count,
        },
        "passed": passed,
    }


def _stratum_audit(captures: Mapping[str, Any]) -> dict[str, Any]:
    no_contact = captures["no-contact"]
    launched = {
        participant
        for event in no_contact.record["events"]
        if event["event_type"] == "bird_launched"
        for participant in event["participants"]
    }
    no_contact_passed = bool(launched) and all(
        item["entity_a_id"] not in launched and item["entity_b_id"] not in launched
        for sample in no_contact.record["fixed_step_samples"]
        for item in sample["contacts"]
    )
    collision = captures["collision"]
    contact_pairs_by_step = {
        sample["fixed_step"]: {
            frozenset((item["entity_a_id"], item["entity_b_id"]))
            for item in sample["contacts"]
        }
        for sample in collision.record["fixed_step_samples"]
    }
    collision_passed = any(
        event["event_type"] == "collision"
        and frozenset(event["participants"]) in contact_pairs_by_step[event["fixed_step"]]
        for event in collision.record["events"]
    )
    support = captures["support"]
    support_sets = [relations for _, relations in _relation_sets(support, "supports")]
    persistent_passed = len(support_sets) >= 2 and bool(support_sets[0] & support_sets[1])
    changed = captures["support-change"]
    changed_sets = _relation_sets(changed, "supports")
    support_change_passed = any(
        before != after
        for (_, before), (_, after) in zip(changed_sets, changed_sets[1:])
    )
    destruction_capture = next(
        (
            capture
            for capture in captures.values()
            if any(event["event_type"] == "entity_destroyed" for event in capture.record["events"])
        ),
        None,
    )
    stability_capture = next(
        (
            capture
            for capture in captures.values()
            if {event["event_type"] for event in capture.record["events"]}
            & {"stable_entered", "stable_exited"}
        ),
        None,
    )
    result = {
        "no-contact/miss": {"passed": no_contact_passed, "capture_id": no_contact.capture_id},
        "collision": {"passed": collision_passed, "capture_id": collision.capture_id},
        "persistent support": {"passed": persistent_passed, "capture_id": support.capture_id},
        "support change": {"passed": support_change_passed, "capture_id": changed.capture_id},
        "destruction": {
            "passed": destruction_capture is not None,
            "capture_id": destruction_capture.capture_id if destruction_capture else None,
        },
        "stability transitions": {
            "passed": stability_capture is not None,
            "capture_id": stability_capture.capture_id if stability_capture else None,
        },
    }
    return result


def source_bound_quarantine_audit(capture_record: Mapping[str, Any]) -> dict[str, Any]:
    source_capture = parse_physics_capture_v2(capture_record)
    mutation = deepcopy(capture_record)
    mutation.pop("terminal_evidence", None)
    with tempfile.TemporaryDirectory(prefix="novphy-issue51-quarantine-") as temporary:
        root = Path(temporary)
        def mutated_collector(staging: Path, _actions, **_options):
            shot = staging / "shot_001"
            shot.mkdir(parents=True)
            (shot / "physics_capture_v2.json").write_text(
                json.dumps(mutation, allow_nan=False, indent=2), encoding="utf-8"
            )
            metadata = {
                "capture_contract": "physics_capture_v2",
                "physics_capture_v2_path": "physics_capture_v2.json",
                "scenario_manifest_identity": "issue51-quarantine-audit-scenario",
            }
            (shot / "metadata.json").write_text(
                json.dumps(metadata, allow_nan=False, indent=2), encoding="utf-8"
            )
            manifest = {
                "rollouts": [{
                    "name": "shot_001",
                    "accepted": True,
                    "artifact_validation": {"accepted": True},
                }]
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, allow_nan=False, indent=2), encoding="utf-8"
            )
            return manifest

        result = collect_fresh_engine_attempt(
            root,
            {"action_type": "source_bound_invalidation_audit"},
            attempt_id="issue51-terminal-evidence-mutation",
            attempt_number=1,
            expected_initial_engine_state_identity="issue51-audit-initial-state",
            physics_capture_v2=True,
            physics_v2_scenario_manifest_identity="issue51-quarantine-audit-scenario",
            collector=mutated_collector,
        )
        quarantine = Path(result["quarantine_path"])
        failure = _load_json(Path(result["failure_manifest_path"]))
        accepted = root / "accepted/issue51-terminal-evidence-mutation"
        if (
            result["status"] != "failed"
            or result["failure_code"] != "attempt_publication_error"
            or accepted.exists()
            or not (quarantine / "failure.json").is_file()
        ):
            raise Issue51EvidenceError("atomic quarantine audit did not fail closed")
    return {
        "schema": "issue_51_source_bound_atomic_quarantine_audit_v1",
        "identity": (
            "issue-51-source-bound-atomic-quarantine-audit-v1:"
            f"{source_capture.capture_id}:missing-terminal-evidence"
        ),
        **{
            key: value
            for key, value in failure.items()
            if key
            in {
                "attempt_id",
                "attempt_number",
                "status",
                "reason",
                "failure_code",
                "failure_class",
                "retryable",
                "retry_decision",
                "exception_type",
            }
        },
        "failure_manifest_schema": failure["schema"],
        "disposition": "quarantined",
        "source_capture_id": source_capture.capture_id,
        "mutation": "remove terminal_evidence",
        "eligible_for_capability_evidence": False,
        "whole_attempt_quarantined": True,
        "accepted_namespace_untouched": True,
        "passed": True,
    }


def _supplementary_sources(root: Path) -> dict[str, Any]:
    base = Path(root) / "supplementary"
    plan = load_collection_plan(base / "collection-plan.json").plan
    scenario_value = load_cohort_v2_scenario_manifest(
        base / "scenario-manifest.json",
        xml_path=base / "scenario.xml",
        template_source_path=base / "template.xml",
    )
    scenario = scenario_value.to_dict()
    runtime = _load_json(base / "runtime-authority.json")
    prior = _load_json(base / "prior-determination.json")
    if (
        prior.get("identity") != PRIOR_DETERMINATION_IDENTITY
        or prior.get("disposition") != "failed"
        or prior.get("failure_reason") != "unmet_level_clear"
        or prior.get("counts")
        != {
            "accepted": 2,
            "rejected": 0,
            "failed": 0,
            "quarantined": 0,
            "retried": 0,
        }
    ):
        raise Issue51EvidenceError("prior failed determination accounting is incomplete")
    prior_attempts = prior.get("attempts")
    if not isinstance(prior_attempts, list) or len(prior_attempts) != 2:
        raise Issue51EvidenceError("prior determination attempt accounting is incomplete")
    prior_plan_relative = prior.get("collection_plan_path")
    if (
        not isinstance(prior_plan_relative, str)
        or Path(prior_plan_relative).is_absolute()
        or ".." in Path(prior_plan_relative).parts
    ):
        raise Issue51EvidenceError("prior determination plan path is invalid")
    prior_plan = load_collection_plan(base / prior_plan_relative).plan
    if prior.get("collection_plan_identity") != prior_plan.identity:
        raise Issue51EvidenceError("prior determination plan identity is stale")
    prior_interventions = {
        intervention.id: intervention.identity
        for collection_scenario in prior_plan.scenarios
        for intervention in collection_scenario.interventions
    }
    prior_captures = {}
    for attempt in prior_attempts:
        relative = attempt.get("capture_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise Issue51EvidenceError("prior determination capture path is invalid")
        capture = parse_physics_capture_v2(_load_json(base / relative))
        if (
            attempt.get("status") != "accepted"
            or attempt.get("capture_id") != capture.capture_id
            or attempt.get("terminal_reason")
            != capture.record["terminal_evidence"]["reason"]
            or attempt.get("realized_coverage_strata")
            != list(classify_physics_capture_v2_coverage(capture))
            or attempt.get("intervention_identity")
            != prior_interventions.get(attempt.get("intervention_id"))
            or capture.source_bindings["intervention_id"]
            != attempt.get("intervention_identity")
        ):
            raise Issue51EvidenceError("prior determination attempt is stale")
        prior_captures[attempt["intervention_id"]] = capture
    if any(
        capture.record["terminal_evidence"]["reason"] == "level_clear"
        for capture in prior_captures.values()
    ):
        raise Issue51EvidenceError("prior failed determination unexpectedly contains level_clear")
    prior_shortfalls = prior.get("realized_coverage_shortfalls")
    targeted_prior = next(
        attempt
        for attempt in prior_attempts
        if attempt["intervention_id"] == "level-clear-targeted"
    )
    if (
        not isinstance(prior_shortfalls, list)
        or len(prior_shortfalls) != 1
        or prior_shortfalls[0].get("intervention_id") != "level-clear-targeted"
        or prior_shortfalls[0].get("intended_coverage_stratum") != "level clear"
        or prior_shortfalls[0].get("realized_coverage_strata")
        != targeted_prior["realized_coverage_strata"]
    ):
        raise Issue51EvidenceError("prior determination shortfall accounting is stale")
    if runtime.get("evidence_source") != "unity_runtime_non_fixture":
        raise Issue51EvidenceError("supplementary termination evidence is not actual Unity evidence")
    if runtime.get("collection_plan_identity") != plan.identity:
        raise Issue51EvidenceError("supplementary runtime authority is stale against its plan")
    expected_runtime_identity = (
        "issue-51-supplementary-runtime-v1:"
        f"{plan.identity}:{runtime.get('source_snapshot_commit')}"
    )
    if runtime.get("identity") != expected_runtime_identity:
        raise Issue51EvidenceError("supplementary runtime identity is stale")
    wrappers = scenario.get("scenario_manifest")
    if not isinstance(wrappers, dict) or scenario.get("identity") != runtime.get("scenario_manifest_identity"):
        raise Issue51EvidenceError("supplementary scenario authority is stale")
    attempts = runtime.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise Issue51EvidenceError("supplementary attempt accounting is incomplete")
    planned_interventions = {
        intervention.id: intervention.identity
        for collection_scenario in plan.scenarios
        for intervention in collection_scenario.interventions
    }
    if {attempt.get("intervention_id") for attempt in attempts} != set(planned_interventions):
        raise Issue51EvidenceError("supplementary attempts differ from the frozen plan")
    captures = {}
    for attempt in attempts:
        relative = attempt.get("capture_path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise Issue51EvidenceError("supplementary capture path is invalid")
        capture = parse_physics_capture_v2(_load_json(base / relative))
        if (
            attempt.get("status") != "accepted"
            or attempt.get("capture_id") != capture.capture_id
            or attempt.get("terminal_reason")
            != capture.record["terminal_evidence"]["reason"]
            or attempt.get("intervention_identity")
            != planned_interventions.get(attempt.get("intervention_id"))
            or capture.source_bindings["intervention_id"]
            != attempt.get("intervention_identity")
        ):
            raise Issue51EvidenceError("supplementary attempt record is stale")
        realized = list(classify_physics_capture_v2_coverage(capture))
        intended = next(
            intervention.intended_coverage_stratum
            for collection_scenario in plan.scenarios
            for intervention in collection_scenario.interventions
            if intervention.id == attempt["intervention_id"]
        )
        if (
            attempt.get("realized_coverage_strata") != realized
            or intended not in realized
        ):
            raise Issue51EvidenceError("supplementary realized coverage is incomplete")
        bindings = capture.source_bindings
        if any(
            bindings[field] != expected
            for field, expected in {
                "scenario_template_id": scenario["template_record"]["identity"],
                "level_instance_id": wrappers["level_instance"]["identity"],
                "scenario_lineage_id": wrappers["scenario_lineage"]["identity"],
            }.items()
        ):
            raise Issue51EvidenceError("supplementary capture source bindings are stale")
        captures[attempt["intervention_id"]] = capture
    clear = [
        capture
        for capture in captures.values()
        if capture.record["terminal_evidence"]["reason"] == "level_clear"
        and any(event["event_type"] == "level_clear" for event in capture.record["events"])
    ]
    if not clear:
        raise Issue51EvidenceError("supplementary probes did not demonstrate level_clear")
    targeted = next(
        attempt for attempt in attempts if attempt["intervention_id"] == "level-clear-targeted"
    )
    if targeted["terminal_reason"] != "level_clear":
        raise Issue51EvidenceError("the prospectively targeted attempt did not demonstrate level_clear")
    return {
        "plan": plan,
        "capture": clear[0],
        "captures": captures,
        "scenario": scenario,
        "runtime": runtime,
        "prior": prior,
        "prior_captures": prior_captures,
    }


def _attempt_accounting(
    repository_root: Path,
    sources: Mapping[str, Any],
    supplementary: Mapping[str, Any],
    quarantine: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = []
    for case, capture in sorted(sources["issue44"]["captures"].items()):
        attempts.append({
            "attempt_identity": f"issue-44:{case}:{capture.capture_id}",
            "source": "issue-44",
            "status": "accepted",
            "capture_id": capture.capture_id,
            "eligible": True,
        })
    issue46_bundle = _load_json(repository_root / "data/runtime_evidence/issue-46/observation-evidence-bundle.json")
    for probe in issue46_bundle["probes"]:
        attempts.append({
            "attempt_identity": f"issue-46:{probe['probe_identity']}",
            "source": "issue-46",
            "status": "accepted",
            "evidence_identity": probe["observation_trace_manifest_identity"],
            "eligible": True,
        })
    replay_ledger = _load_json(repository_root / "data/runtime_evidence/issue-48/attempt-ledger.json")
    for attempt in replay_ledger["attempts"]:
        attempts.append({
            "attempt_identity": attempt["attempt_identity"],
            "source": "issue-48",
            "status": attempt["status"],
            "capture_id": attempt["capture_id"],
            "eligible": attempt["status"] == "accepted",
        })
    issue50_runtime = _load_json(
        repository_root / "data/runtime_evidence/issue-50/source-probes/runtime-bundle-manifest.json"
    )
    for probe in issue50_runtime["probes"]:
        attempts.append({
            "attempt_identity": f"issue-50:{probe['case']}:{probe['capture_id']}",
            "source": "issue-50",
            "status": "accepted",
            "capture_id": probe["capture_id"],
            "eligible": True,
        })
    for attempt in supplementary["runtime"]["attempts"]:
        attempts.append({
            "attempt_identity": attempt["attempt_identity"],
            "source": "issue-51",
            "status": attempt["status"],
            "capture_id": attempt["capture_id"],
            "realized_coverage_strata": attempt["realized_coverage_strata"],
            "eligible": attempt["status"] == "accepted",
        })
    for attempt in supplementary["prior"]["attempts"]:
        attempts.append({
            "attempt_identity": attempt["attempt_identity"],
            "source": "issue-51-determination-1",
            "status": attempt["status"],
            "capture_id": attempt["capture_id"],
            "realized_coverage_strata": attempt["realized_coverage_strata"],
            "eligible": False,
        })
    attempts.append({
        "attempt_identity": quarantine["attempt_id"],
        "source": "issue-51-invalidation-audit",
        "status": quarantine["status"],
        "failure_code": quarantine["failure_code"],
        "eligible": False,
    })
    statuses = ("planned", "accepted", "rejected", "failed", "quarantined", "retried")
    counts = {status: 0 for status in statuses}
    counts["planned"] = len(attempts)
    for attempt in attempts:
        counts[attempt["status"]] += 1
    counts["quarantined"] = 1
    return {
        "schema": "representative_cohort_v2_pilot_attempt_accounting_v1",
        "identity": "representative-cohort-v2-pilot-attempt-accounting-v1:determination-1",
        "attempts": attempts,
        "counts": counts,
        "unavailable": [],
        "unmet": [],
        "systematic_exporter_defects": [],
        "prior_determinations": [supplementary["prior"]],
        "outcome_independent_retention": True,
        "passed": True,
    }


def _capability_audit(
    declaration: Mapping[str, Any],
    identities: Mapping[str, str],
    micro: Mapping[str, Any],
    strata: Mapping[str, Any],
    terminations: Mapping[str, Any],
    quarantine: Mapping[str, Any],
) -> dict[str, Any]:
    central = declaration["capabilities"]["required_central"]
    audit: dict[str, dict[str, Any]] = {}
    for group, values in central.items():
        entries = {}
        for value in values:
            status = "passed"
            evidence = []
            if group == "coverage_strata":
                status = "passed" if strata[value]["passed"] else "failed"
                evidence = [strata[value]["capture_id"]]
            elif group == "micro_labels":
                status = "passed" if micro[value]["passed"] else "failed"
                evidence = [identities["physics_capture_bundle"]]
            elif group == "macro_labels":
                evidence = [identities["macro_semantics_bundle"]]
            elif group == "violation_labels":
                evidence = [identities["physical_violation_bundle"]]
            elif group == "observations":
                evidence = [identities["observation_bundle"]]
            elif group in {"exposure_roles", "splits"}:
                evidence = [identities["scenario_and_partition_bundle"]]
            elif group == "replay":
                evidence = [identities["replay_bundle"]]
            elif group == "ingestion":
                evidence = [
                    identities["observation_bundle"],
                    identities["macro_semantics_bundle"],
                    identities["physical_violation_bundle"],
                ]
            elif group == "provenance":
                if value in {
                    "atomic_rollout_validation",
                    "typed_failure_and_quarantine_accounting",
                    "transient_only_retries",
                }:
                    evidence = [quarantine["identity"]]
                elif value == "source_bound_derivations":
                    evidence = [
                        identities["macro_semantics_bundle"],
                        identities["physical_violation_bundle"],
                    ]
                elif value == "role_separated_final_evaluation":
                    evidence = [identities["scenario_and_partition_bundle"]]
                elif value == "explicit_legacy_static":
                    evidence = [BUNDLE_IDENTITY]
                elif value == "immutable_cohort_release":
                    evidence = [BUNDLE_IDENTITY]
                else:
                    evidence = [identities["physics_capture_bundle"]]
            entry = {"status": status, "evidence_identities": evidence}
            if value == "explicit_legacy_static":
                entry["rationale"] = (
                    "the determination-2 supplementary terminal probe preserves exact "
                    "legacy-static XML and importer/source provenance"
                )
            entries[value] = entry
        audit[group] = entries
    passed = all(
        entry["status"] in {"passed", "passed_not_applicable"}
        for entries in audit.values()
        for entry in entries.values()
    ) and all(value["passed"] for value in terminations.values())
    return {"required_central": audit, "supported_terminations": terminations, "passed": passed}


def _report(
    repository_root: Path,
    pilot_plan: Mapping[str, Any],
    sources: Mapping[str, Any],
    supplementary: Mapping[str, Any],
    quarantine: Mapping[str, Any],
    accounting: Mapping[str, Any],
    implementation_revision: str,
    execution_revision: str,
) -> dict[str, Any]:
    declaration = _load_json(repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json")
    identities = _component_identities(repository_root, sources)
    captures = sources["issue44"]["captures"]
    micro = {predicate: _micro_audit(captures, predicate) for predicate in ("contact", "supports")}
    strata = _stratum_audit(captures)
    termination_captures = list(captures.values()) + list(supplementary["captures"].values())
    terminations = {
        termination: {
            "passed": any(
                capture.record["terminal_evidence"]["reason"] == termination
                for capture in termination_captures
            ),
            "capture_ids": [
                capture.capture_id
                for capture in termination_captures
                if capture.record["terminal_evidence"]["reason"] == termination
            ],
        }
        for termination in SUPPORTED_TERMINATIONS
    }
    capability_audit = _capability_audit(
        declaration, identities, micro, strata, terminations, quarantine
    )
    secondary = {
        capability: {
            "status": "not_required",
            "experiment": experiment,
        }
        for capability, experiment in declaration["capabilities"]["required_secondary"].items()
    }
    optional = {
        capability: {"status": "not_required_optional"}
        for capability in declaration["capabilities"]["optional"]
    }
    out_of_scope = {
        capability: {"status": "not_required_out_of_scope"}
        for capability in declaration["capabilities"]["out_of_scope"]
    }
    representative = (
        capability_audit["passed"]
        and accounting["passed"]
        and not accounting["unavailable"]
        and not accounting["unmet"]
        and not accounting["systematic_exporter_defects"]
    )
    if not representative:
        raise Issue51EvidenceError("the representative cohort-v2 pilot audit failed closed")
    partition = _load_json(repository_root / "data/runtime_evidence/issue-47/partition-exposure-manifest.json")
    replay = _load_json(repository_root / "data/runtime_evidence/issue-48/bundle-manifest.json")
    replay_plan = _load_json(repository_root / "data/runtime_evidence/issue-48/replay-plan.json")
    macro = _load_json(repository_root / "data/runtime_evidence/issue-49/bundle-manifest.json")
    violations = _load_json(repository_root / "data/runtime_evidence/issue-50/bundle-manifest.json")
    observations = _load_json(
        repository_root / "data/runtime_evidence/issue-46/observation-evidence-bundle.json"
    )
    physical_probes = _load_json(
        repository_root
        / "data/runtime_evidence/issue-50/source-probes/runtime-bundle-manifest.json"
    )
    return {
        "schema": "representative_cohort_v2_pilot_report_v1",
        "identity": PILOT_REPORT_IDENTITY,
        "issue": 51,
        "pilot_plan_identity": pilot_plan["identity"],
        "capability_declaration_identity": declaration["identity"],
        "disposition": "accepted",
        "evidence_source": "actual_non_fixture_unity_runtime",
        "representative_audit": representative,
        "component_evidence_identities": identities,
        "scenario_identities": {
            "physics_and_supplementary": [list(values) for values in sorted({
                (
                    capture.source_bindings["scenario_lineage_id"],
                    capture.source_bindings["level_instance_id"],
                    capture.source_bindings["scenario_template_id"],
                )
                for capture in termination_captures
            })],
            "observation_probes": [{
                "scenario_lineage_id": probe["source_scenario_lineage_identity"],
                "level_instance_id": probe["level_instance_identity"],
                "scenario_template_id": probe["scenario_template_identity"],
            } for probe in observations["probes"]],
            "partition_roles": [{
                "exposure_role": entry["exposure_role"],
                "scenario_lineage_id": entry["scenario_lineage_identity"],
                "level_instance_id": entry["level_instance_identity"],
                "scenario_template_id": entry["scenario_template_identity"],
            } for entry in partition["entries"]],
            "replay_collections": [{
                "scenario_collection_id": item["scenario_collection_id"],
                "scenario_lineage_id": item["scenario_lineage_identity"],
                "level_instance_id": item["level_instance_identity"],
                "scenario_template_id": item["scenario_template_identity"],
            } for item in replay_plan["scenario_collections"]],
            "physical_violation_probes": [{
                "case": probe["case"],
                "scenario_lineage_id": probe["scenario_lineage_id"],
                "level_instance_id": probe["level_instance_id"],
                "scenario_template_id": probe["scenario_template_id"],
            } for probe in physical_probes["probes"]],
        },
        "attempt_accounting_identity": accounting["identity"],
        "replay_identity": replay["replay_evidence_identity"],
        "partition_identity": partition["identity"],
        "derivation_identities": {
            "macro": macro["accepted_derivation_identities"],
            "physical_violations": violations["accepted_derivation_identities"],
        },
        "environment": {
            "unity_version": sources["issue44"]["report"]["provenance"]["engine_version"],
            "physics_protocol": sources["issue44"]["report"]["provenance"]["protocol_version"],
            "physics_exporter": sources["issue44"]["report"]["provenance"]["exporter_version"],
            "supplementary_runtime_authority": supplementary["runtime"]["identity"],
        },
        "code": {
            "pilot_implementation_revision": implementation_revision,
            "execution_revision": execution_revision,
            "issue_44_source_snapshot_commit": sources["issue44"]["runtime"]["source_snapshot_commit"],
        },
        "coverage_strata": strata,
        "micro_label_evidence": micro,
        "capability_audit": capability_audit,
        "secondary_capabilities": secondary,
        "optional_capabilities": optional,
        "out_of_scope_capabilities": out_of_scope,
        "final_evaluation": {
            "manifest_identity": _load_json(
                repository_root
                / "data/runtime_evidence/issue-47/final-evaluation-workflow-access-manifest.json"
            )["identity"],
            "access": "sealed",
            "consumed": False,
        },
        "quarantine_audit_identity": quarantine["identity"],
        "prior_failed_determination_identity": supplementary["prior"]["identity"],
        "systematic_exporter_defects": [],
        "passed": True,
    }


def _derived_artifacts(
    evidence_root: Path,
    repository_root: Path,
    implementation_revision: str,
    execution_revision: str,
) -> dict[str, dict[str, Any]]:
    sources = _validate_component_sources(repository_root)
    supplementary = _supplementary_sources(evidence_root)
    issue50_runtime = _load_json(
        repository_root
        / "data/runtime_evidence/issue-50/source-probes/runtime-bundle-manifest.json"
    )
    if (
        supplementary["runtime"].get("source_snapshot_commit")
        != issue50_runtime.get("source_snapshot_commit")
    ):
        raise Issue51EvidenceError(
            "supplementary Unity player differs from the accepted physics-v2 source envelope"
        )
    quarantine = source_bound_quarantine_audit(supplementary["capture"].record)
    identities = _component_identities(repository_root, sources)
    plan = build_pilot_plan(
        repository_root,
        identities,
        supplementary["plan"].identity,
    )
    accounting = _attempt_accounting(
        repository_root, sources, supplementary, quarantine
    )
    report = _report(
        repository_root,
        plan,
        sources,
        supplementary,
        quarantine,
        accounting,
        implementation_revision,
        execution_revision,
    )
    return {
        "pilot-plan.json": plan,
        "attempt-accounting.json": accounting,
        "quarantine-audit.json": quarantine,
        "representative-cohort-v2-pilot-report.json": report,
    }


def validate_issue_51_evidence(
    evidence_root: Path,
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(evidence_root)
    bundle = _load_json(root / "bundle-manifest.json")
    implementation_revision = bundle.get("implementation_revision")
    execution_revision = bundle.get("execution_revision")
    if not isinstance(implementation_revision, str) or not implementation_revision:
        raise Issue51EvidenceError("issue-51 implementation revision is missing")
    if not isinstance(execution_revision, str) or not execution_revision:
        raise Issue51EvidenceError("issue-51 execution revision is missing")
    if implementation_revision != _implementation_revision(repository_root):
        raise Issue51EvidenceError("issue-51 implementation revision is stale")
    expected = _derived_artifacts(
        root, repository_root, implementation_revision, execution_revision
    )
    for path, value in expected.items():
        if _load_json(root / path) != value:
            raise Issue51EvidenceError(f"issue-51 artifact is stale: {path}")
    members = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    )
    expected_bundle = {
        "schema": "issue_51_representative_cohort_v2_pilot_bundle_v1",
        "identity": BUNDLE_IDENTITY,
        "implementation_revision": implementation_revision,
        "execution_revision": execution_revision,
        "pilot_report_identity": PILOT_REPORT_IDENTITY,
        "pilot_plan_identity": PILOT_PLAN_IDENTITY,
        "artifacts": members,
        "passed": True,
    }
    if bundle != expected_bundle:
        raise Issue51EvidenceError("issue-51 bundle membership or identity is stale")
    return {
        "schema": "issue_51_representative_cohort_v2_pilot_validation_result_v1",
        "bundle_identity": bundle["identity"],
        "pilot_report_identity": PILOT_REPORT_IDENTITY,
        "representative_audit": True,
        "passed": True,
    }


def build_issue_51_evidence(
    output: Path,
    supplementary_root: Path,
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    output = Path(output)
    supplementary_root = Path(supplementary_root)
    _require_clean_tracked_worktree(repository_root)
    implementation_revision = _implementation_revision(repository_root)
    execution_revision = _execution_revision(repository_root)
    if output.exists():
        raise Issue51EvidenceError(f"immutable issue-51 output already exists: {output}")
    required = {
        "collection-plan.json",
        "prior-determination.json",
        "scenario-manifest.json",
        "scenario.xml",
        "template.xml",
        "runtime-authority.json",
    }
    if (
        {path.name for path in supplementary_root.iterdir() if path.is_file()} != required
        or not (supplementary_root / "captures").is_dir()
        or not (supplementary_root / "prior-captures").is_dir()
    ):
        raise Issue51EvidenceError("supplementary issue-51 source membership is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".issue-51-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        supplement = staging / "supplementary"
        supplement.mkdir(parents=True)
        for name in sorted(required):
            shutil.copyfile(supplementary_root / name, supplement / name)
        shutil.copytree(supplementary_root / "captures", supplement / "captures")
        shutil.copytree(
            supplementary_root / "prior-captures", supplement / "prior-captures"
        )
        _log("revalidating all issue-44 through issue-50 component authorities")
        artifacts = _derived_artifacts(
            staging, repository_root, implementation_revision, execution_revision
        )
        for path, value in artifacts.items():
            write_immutable_cohort_v2_json(value, staging / path)
        members = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file()
        )
        bundle = {
            "schema": "issue_51_representative_cohort_v2_pilot_bundle_v1",
            "identity": BUNDLE_IDENTITY,
            "implementation_revision": implementation_revision,
            "execution_revision": execution_revision,
            "pilot_report_identity": PILOT_REPORT_IDENTITY,
            "pilot_plan_identity": PILOT_PLAN_IDENTITY,
            "artifacts": members,
            "passed": True,
        }
        write_immutable_cohort_v2_json(bundle, staging / "bundle-manifest.json")
        result = validate_issue_51_evidence(staging, repository_root=repository_root)
        os.replace(staging, output)
    _log(f"immutable representative pilot published: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish issue #51 pilot evidence")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--supplementary-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_issue_51_evidence(args.output, args.supplementary_root)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
