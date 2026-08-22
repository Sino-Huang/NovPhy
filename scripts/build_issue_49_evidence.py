"""Build and validate the canonical issue-49 macro-semantics evidence bundle."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from scripts.cohort_v2_macro_semantics import (
    ACCEPTED_PREDICATES,
    CohortV2MacroSemanticsError,
    DERIVATION_SPEC_IDENTITY,
    derive_capture_macro_labels,
    derivation_spec,
    finite_json_tree,
    validate_capture_macro_derivation,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.physics_capture_v2 import parse_physics_capture_v2
from scripts.physics_capture_v2_capability_report import (
    load_physics_capture_v2_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "data/runtime_evidence/issue-44"
DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-49"
CASES = ("collision", "no-contact", "stable-terminal", "support", "support-change")
ADJUDICATION_CASES = ("collision", "support-change")
BUNDLE_IDENTITY = "issue-49-v2-macro-semantics-bundle-v1:accepted-determination-1"
PLAN_IDENTITY = "issue-49-v2-macro-adjudication-plan-v1:determination-1"
ADJUDICATION_IDENTITY = (
    "issue-49-v2-macro-semantics-adjudication-v1:accepted-determination-1"
)


class Issue49EvidenceError(ValueError):
    """Issue #49 evidence is incomplete, stale, or fails its empirical floor."""


def _log(message: str) -> None:
    print(f"[issue-49] {message}", flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Issue49EvidenceError(f"cannot load {path}") from error
    if not isinstance(value, dict):
        raise Issue49EvidenceError(f"{path} must contain a JSON object")
    return value


def _validate_probe_binding(capture: Any, probe: Mapping[str, Any], case: str) -> None:
    expected = {
        "capture_id": capture.capture_id,
        "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
        "level_instance_id": capture.source_bindings["level_instance_id"],
        "scenario_template_id": capture.source_bindings["scenario_template_id"],
    }
    if any(probe.get(field) != value for field, value in expected.items()):
        raise Issue49EvidenceError(f"{case}: capability probe binding is stale")
    if probe.get("source") != "unity_exporter_probe" or probe.get("final_evaluation") is not False:
        raise Issue49EvidenceError(f"{case}: source is not an eligible non-final Unity probe")


def _load_sources(repository_root: Path, source_root: Path) -> dict[str, Any]:
    runtime_bundle = _load_json(source_root / "runtime-bundle-manifest.json")
    capture_bundle = _load_json(source_root / "capture-bundle-manifest.json")
    capability = load_physics_capture_v2_capability_report(
        source_root / "capability-report.json"
    )
    if runtime_bundle.get("schema") != "issue_44_physics_v2_runtime_evidence_bundle_v1":
        raise Issue49EvidenceError("issue-44 runtime evidence has the wrong schema")
    if (
        capture_bundle.get("schema") != "issue_44_physics_v2_capture_bundle_v1"
        or capture_bundle.get("runtime_evidence_bundle_identity")
        != runtime_bundle.get("identity")
    ):
        raise Issue49EvidenceError("issue-44 capture bundle is stale against its runtime bundle")
    if any(
        fact.get("status") != "demonstrated"
        for fact in capability.record["facts"].values()
    ):
        raise Issue49EvidenceError("issue-44 source capability report is not fully demonstrated")

    declared = capture_bundle.get("captures")
    if not isinstance(declared, dict) or tuple(sorted(declared)) != tuple(sorted(CASES)):
        raise Issue49EvidenceError("issue-44 source capture membership is incomplete")
    probes = {probe["case"]: probe for probe in capability.record["probes"]}
    captures = {}
    for case in CASES:
        reference = f"data/runtime_evidence/issue-44/captures/{case}.json"
        entry = declared[case]
        if entry.get("path") != reference:
            raise Issue49EvidenceError(f"{case}: issue-44 capture reference is stale")
        capture = parse_physics_capture_v2(_load_json(repository_root / reference))
        if entry.get("capture_id") != capture.capture_id:
            raise Issue49EvidenceError(f"{case}: issue-44 capture identity is stale")
        _validate_probe_binding(capture, probes[case], case)
        captures[case] = capture
    return {
        "runtime_bundle": runtime_bundle,
        "capture_bundle": capture_bundle,
        "capability_report": capability.record,
        "captures": captures,
    }


def adjudication_plan(source_capture_bundle_identity: str) -> dict[str, Any]:
    """Frozen witness selections; the rules are fixed before derivation executes."""
    return {
        "schema": "issue_49_v2_macro_adjudication_plan_v1",
        "identity": PLAN_IDENTITY,
        "source_capture_bundle_identity": source_capture_bundle_identity,
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "predicates": list(ACCEPTED_PREDICATES),
        "selected_cases": list(ADJUDICATION_CASES),
        "selection_rules": {
            "steady_positive": "same-step stable_entered label",
            "steady_negative": "same-step collision label",
            "steady_boundary": "stable_exited fixed_step minus one, at event, plus one",
            "structure_positive": "same-step collision support-set change",
            "structure_negative": "second fixed step after the selected collision",
            "structure_boundary": "collision fixed_step minus one through plus two",
            "unavailable": "first retained fixed-step label",
        },
        "minimum_floor": {
            "positive_witnesses": 2,
            "negative_witnesses": 2,
            "boundary_windows": 2,
            "non_final_scenario_lineages": 2,
            "level_instances": 2,
            "scenario_templates": 2,
            "unavailable_or_rejection_checks": 1,
        },
    }


def _label(derivation: Mapping[str, Any], step: int, predicate: str) -> Mapping[str, Any]:
    for record in derivation["labels"]:
        if record["fixed_step"] == step:
            return record["predicates"][predicate]
    raise Issue49EvidenceError(
        f"{derivation['source']['capture_id']}: no {predicate} label at fixed step {step}"
    )


def _event(capture: Any, event_type: str) -> Mapping[str, Any]:
    matches = [event for event in capture.record["events"] if event["event_type"] == event_type]
    if len(matches) != 1:
        raise Issue49EvidenceError(
            f"{capture.capture_id}: expected exactly one {event_type} event"
        )
    return matches[0]


def _witness(
    case: str,
    capture: Any,
    derivation: Mapping[str, Any],
    predicate: str,
    step: int,
) -> dict[str, Any]:
    label = _label(derivation, step, predicate)
    return {
        "case": case,
        "capture_id": capture.capture_id,
        "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
        "level_instance_id": capture.source_bindings["level_instance_id"],
        "scenario_template_id": capture.source_bindings["scenario_template_id"],
        "fixed_step": step,
        "value": label["value"],
        "availability": label["availability"],
        "source_interval": label["source_interval"],
        "evidence": label["evidence"],
    }


def _coverage(witnesses: list[Mapping[str, Any]], expected_value: bool) -> dict[str, Any]:
    eligible = [
        witness
        for witness in witnesses
        if witness["availability"] == "available" and witness["value"] is expected_value
    ]
    result = {
        "witness_count": len(eligible),
        "scenario_lineage_count": len({item["scenario_lineage_id"] for item in eligible}),
        "level_instance_count": len({item["level_instance_id"] for item in eligible}),
        "scenario_template_count": len({item["scenario_template_id"] for item in eligible}),
    }
    result["passed"] = min(result.values()) >= 2
    return result


def _window(
    case: str,
    capture: Any,
    derivation: Mapping[str, Any],
    predicate: str,
    steps: list[int],
    expected: list[bool],
) -> dict[str, Any]:
    witnesses = [
        _witness(case, capture, derivation, predicate, step) for step in steps
    ]
    values = [witness["value"] for witness in witnesses]
    if values != expected or any(witness["availability"] != "available" for witness in witnesses):
        raise Issue49EvidenceError(
            f"{case}: {predicate} boundary {steps} produced {values}, expected {expected}"
        )
    return {
        "case": case,
        "capture_id": capture.capture_id,
        "fixed_steps": steps,
        "values": values,
        "source_intervals": [witness["source_interval"] for witness in witnesses],
    }


def _mutation_checks(sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    capture = sources["captures"]["collision"]
    capture_record = capture.record
    checks = []

    missing_event = deepcopy(capture_record)
    missing_event["events"] = [
        event for event in missing_event["events"] if event["event_type"] != "stable_exited"
    ]
    try:
        mutated_capture = parse_physics_capture_v2(missing_event)
        derive_capture_macro_labels(
            mutated_capture,
            source_reference="mutation/missing-stability-event.json",
            source_capture_bundle_identity=sources["capture_bundle"]["identity"],
        )
    except (ValueError, CohortV2MacroSemanticsError) as error:
        checks.append(
            {
                "mutation": "remove_required_stability_transition_event",
                "expected": "rejected",
                "observed": "rejected",
                "reason": str(error),
            }
        )
    else:
        raise Issue49EvidenceError("missing stability-event mutation did not fail closed")

    missing_history = deepcopy(capture_record)
    del missing_history["fixed_step_samples"][1]
    try:
        parse_physics_capture_v2(missing_history)
    except ValueError as error:
        checks.append(
            {
                "mutation": "remove_fixed_step_history",
                "expected": "rejected",
                "observed": "rejected",
                "reason": str(error),
            }
        )
    else:
        raise Issue49EvidenceError("missing fixed-step history mutation did not fail closed")

    stale_binding = deepcopy(capture_record)
    stale_binding["source_bindings"]["scenario_lineage_id"] = "scenario-lineage-v1:stale"
    mutated_capture = parse_physics_capture_v2(stale_binding)
    probe = next(
        item
        for item in sources["capability_report"]["probes"]
        if item["case"] == "collision"
    )
    try:
        _validate_probe_binding(mutated_capture, probe, "collision-mutation")
    except Issue49EvidenceError as error:
        checks.append(
            {
                "mutation": "change_source_lineage_binding",
                "expected": "rejected",
                "observed": "rejected",
                "reason": str(error),
            }
        )
    else:
        raise Issue49EvidenceError("stale source-binding mutation did not fail closed")
    return checks


def _adjudication(
    sources: Mapping[str, Any],
    derivations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    steady_positive = []
    steady_negative = []
    steady_boundaries = []
    unstable_positive = []
    unstable_negative = []
    unstable_boundaries = []
    unavailable = []

    for case in ADJUDICATION_CASES:
        capture = sources["captures"][case]
        derived = derivations[case]
        stable_entered = _event(capture, "stable_entered")["fixed_step"]
        stable_exited = _event(capture, "stable_exited")["fixed_step"]
        collision = _event(capture, "collision")["fixed_step"]
        steady_positive.append(
            _witness(case, capture, derived, "steady-state", stable_entered)
        )
        steady_negative.append(
            _witness(case, capture, derived, "steady-state", collision)
        )
        steady_boundaries.append(
            _window(
                case,
                capture,
                derived,
                "steady-state",
                [stable_exited - 1, stable_exited, stable_exited + 1],
                [True, False, False],
            )
        )
        unstable_positive.append(
            _witness(case, capture, derived, "structure-unstable", collision)
        )
        unstable_negative.append(
            _witness(case, capture, derived, "structure-unstable", collision + 2)
        )
        unstable_boundaries.append(
            _window(
                case,
                capture,
                derived,
                "structure-unstable",
                [collision - 1, collision, collision + 1, collision + 2],
                [False, True, True, False],
            )
        )
        first_step = derived["labels"][0]["fixed_step"]
        unavailable.extend(
            (
                _witness(case, capture, derived, "steady-state", first_step),
                _witness(case, capture, derived, "structure-unstable", first_step),
            )
        )

    steady_coverage = {
        "positive": _coverage(steady_positive, True),
        "negative": _coverage(steady_negative, False),
        "boundary_window_count": len(steady_boundaries),
    }
    unstable_coverage = {
        "positive": _coverage(unstable_positive, True),
        "negative": _coverage(unstable_negative, False),
        "boundary_window_count": len(unstable_boundaries),
    }
    unavailable_passed = all(
        item["value"] is None and item["availability"].startswith("unavailable_")
        for item in unavailable
    )
    mutations = _mutation_checks(sources)
    passed = (
        all(steady_coverage[name]["passed"] for name in ("positive", "negative"))
        and steady_coverage["boundary_window_count"] >= 2
        and all(unstable_coverage[name]["passed"] for name in ("positive", "negative"))
        and unstable_coverage["boundary_window_count"] >= 2
        and unavailable_passed
        and all(item["observed"] == item["expected"] for item in mutations)
    )
    return {
        "schema": "issue_49_v2_macro_semantics_adjudication_v1",
        "identity": ADJUDICATION_IDENTITY,
        "plan_identity": PLAN_IDENTITY,
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "source_capture_bundle_identity": sources["capture_bundle"]["identity"],
        "procedure": {
            "primary": "deterministic automated fixed-step derivation",
            "verification": "independent exact artifact re-derivation during validation",
            "disagreements": [],
        },
        "predicate_dispositions": {
            predicate: "accepted_authoritative" for predicate in ACCEPTED_PREDICATES
        },
        "witnesses": {
            "steady-state": {
                "positive": steady_positive,
                "negative": steady_negative,
                "boundary_windows": steady_boundaries,
            },
            "structure-unstable": {
                "positive": unstable_positive,
                "negative": unstable_negative,
                "boundary_windows": unstable_boundaries,
            },
            "unavailable": unavailable,
        },
        "coverage": {
            "steady-state": steady_coverage,
            "structure-unstable": unstable_coverage,
            "unavailable_passed": unavailable_passed,
            "mutation_check_count": len(mutations),
        },
        "mutation_checks": mutations,
        "exclusions": {
            predicate: "excluded_not_emitted_not_false"
            for predicate in ("cascade-active", "collapsed", "pigs-cleared")
        },
        "failure_cases": [
            "A missing fixed-step predecessor makes structure-unstable unavailable.",
            "An incomplete initial debounce window makes steady-state unavailable.",
            "A missing or inconsistent engine stability transition rejects the derivation.",
            "A fixed-step gap, stale source binding, unavailable source fact, or cross-release "
            "source mismatch rejects the derivation bundle.",
        ],
        "passed": passed,
    }


def _expected_artifacts(
    repository_root: Path,
    source_root: Path,
    implementation_revision: str,
) -> dict[str, dict[str, Any]]:
    sources = _load_sources(repository_root, source_root)
    runtime_bundle = sources["runtime_bundle"]
    capture_bundle = sources["capture_bundle"]
    spec = derivation_spec(
        source_runtime_bundle_identity=runtime_bundle["identity"],
        source_snapshot_commit=runtime_bundle["source_snapshot_commit"],
    )
    plan = adjudication_plan(capture_bundle["identity"])
    derivations = {
        case: derive_capture_macro_labels(
            sources["captures"][case],
            source_reference=f"data/runtime_evidence/issue-44/captures/{case}.json",
            source_capture_bundle_identity=capture_bundle["identity"],
        )
        for case in CASES
    }
    for case in CASES:
        validate_capture_macro_derivation(
            derivations[case],
            sources["captures"][case],
            source_reference=f"data/runtime_evidence/issue-44/captures/{case}.json",
            source_capture_bundle_identity=capture_bundle["identity"],
        )
    adjudication = _adjudication(sources, derivations)
    if not adjudication["passed"]:
        raise Issue49EvidenceError("issue-49 representative evidence floor did not pass")

    artifacts = {
        "derivation-spec.json": spec,
        "adjudication-plan.json": plan,
        "macro-semantics-adjudication.json": adjudication,
        **{f"derivations/{case}.json": value for case, value in derivations.items()},
    }
    bundle_members = [
        {
            "path": path,
            "schema": value["schema"],
            "identity": value["identity"],
        }
        for path, value in sorted(artifacts.items())
    ]
    artifacts["bundle-manifest.json"] = {
        "schema": "issue_49_v2_macro_semantics_bundle_v1",
        "identity": BUNDLE_IDENTITY,
        "implementation_revision": implementation_revision,
        "source": {
            "issue": 44,
            "runtime_bundle_identity": runtime_bundle["identity"],
            "capture_bundle_identity": capture_bundle["identity"],
            "capability_report_id": sources["capability_report"]["report_id"],
            "source_snapshot_commit": runtime_bundle["source_snapshot_commit"],
            "version_envelope": dict(sources["capability_report"]["provenance"]),
        },
        "artifacts": bundle_members,
        "accepted_derivation_identities": [
            derivations[case]["identity"] for case in CASES
        ],
        "adjudication_identity": ADJUDICATION_IDENTITY,
        "passed": True,
    }
    return artifacts


def validate_issue_49_evidence(
    evidence_root: Path,
    *,
    repository_root: Path = ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    bundle = _load_json(evidence_root / "bundle-manifest.json")
    implementation_revision = bundle.get("implementation_revision")
    if not isinstance(implementation_revision, str) or not implementation_revision:
        raise Issue49EvidenceError("bundle implementation revision is missing")
    expected = _expected_artifacts(repository_root, source_root, implementation_revision)
    actual_paths = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*.json")
    }
    if actual_paths != set(expected):
        raise Issue49EvidenceError("issue-49 bundle membership is incomplete or contains extras")
    for relative_path, expected_value in expected.items():
        actual = _load_json(evidence_root / relative_path)
        if actual != expected_value:
            raise Issue49EvidenceError(f"{relative_path} differs from exact re-derivation")
        if not finite_json_tree(actual):
            raise Issue49EvidenceError(f"{relative_path} contains non-finite numeric evidence")
    return {
        "schema": "issue_49_v2_macro_semantics_validation_result_v1",
        "bundle_identity": BUNDLE_IDENTITY,
        "implementation_revision": implementation_revision,
        "capture_count": len(CASES),
        "label_count": sum(
            expected[f"derivations/{case}.json"]["label_count"] for case in CASES
        ),
        "passed": True,
    }


def _head_revision(repository_root: Path) -> str:
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
        raise Issue49EvidenceError(
            "canonical publication requires committed implementation and a clean tracked worktree"
        )


def build_issue_49_evidence(
    output: Path,
    *,
    repository_root: Path = ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    implementation_revision: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    revision = implementation_revision or _head_revision(repository_root)
    _log("validating the immutable issue-44 Unity source bundle")
    artifacts = _expected_artifacts(repository_root, source_root, revision)
    adjudication = artifacts["macro-semantics-adjudication.json"]
    _log(
        "derived 5 captures and "
        f"{sum(artifacts[f'derivations/{case}.json']['label_count'] for case in CASES)} "
        "fixed-step label records"
    )
    _log(
        "witness floor passed: 2 positive, 2 negative, and 2 boundary windows "
        "for each accepted predicate"
    )
    _log(
        f"fail-closed checks passed: {adjudication['coverage']['mutation_check_count']} mutations"
    )
    if dry_run:
        _log("dry-run complete; no files written")
        return {
            "schema": "issue_49_v2_macro_semantics_dry_run_v1",
            "output": str(output),
            "implementation_revision": revision,
            "artifact_count": len(artifacts),
            "passed": True,
        }

    _require_clean_tracked_worktree(repository_root)
    if output.exists():
        raise Issue49EvidenceError(f"immutable issue-49 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _log("publishing the canonical immutable bundle")
    with tempfile.TemporaryDirectory(prefix=".issue-49-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        for relative_path, value in artifacts.items():
            write_immutable_cohort_v2_json(value, staging / relative_path)
        result = validate_issue_49_evidence(
            staging,
            repository_root=repository_root,
            source_root=source_root,
        )
        os.replace(staging, output)
    _log(f"publication complete: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate and publish accepted cohort-v2 macro semantics for issue #49"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--implementation-revision")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_issue_49_evidence(
        args.output,
        source_root=args.source_root,
        implementation_revision=args.implementation_revision,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
