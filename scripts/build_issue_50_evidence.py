"""Build and validate issue #50's physical-violation evidence bundle."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from scripts.cohort_v2_physical_violations import (
    ACCEPTED_PREDICATES,
    AGGREGATE_PREDICATE,
    DERIVATION_SPEC_IDENTITY,
    EXCESS_PENETRATION,
    EXCLUDED_PREDICATES,
    UNSUPPORTED_STATIONARY,
    derive_capture_physical_violations,
    derivation_spec,
    finite_json_tree,
    validate_capture_physical_violation_derivation,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.physics_capture_v2 import parse_physics_capture_v2
from scripts.physics_capture_v2_capability_report import (
    load_physics_capture_v2_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]
ISSUE_44_ROOT = ROOT / "data/runtime_evidence/issue-44"
DEFAULT_PROBE_ROOT = ROOT / ".local-artifacts/issue-50-runtime/source-probes"
DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-50"
ISSUE_44_CASES = ("collision", "no-contact", "stable-terminal", "support", "support-change")
PROBE_CASES = (
    "floating-a-geometry",
    "floating-a-stationary",
    "floating-b-geometry",
    "floating-b-stationary",
)
UNSUPPORTED_CASES = ("floating-a-stationary", "floating-b-stationary")
PENETRATION_CASES = ("collision", "support-change")
MINIMUM_WITNESS_COUNT = 2
MINIMUM_BOUNDARY_WINDOW_COUNT = 2
BUNDLE_IDENTITY = "issue-50-v2-physical-violation-bundle-v1:accepted-determination-1"
PLAN_IDENTITY = "issue-50-v2-physical-violation-adjudication-plan-v1:determination-1"
ADJUDICATION_IDENTITY = (
    "issue-50-v2-physical-violation-adjudication-v1:accepted-determination-1"
)


class Issue50EvidenceError(ValueError):
    """Issue #50 evidence is incomplete, stale, or below its empirical floor."""


def _log(message: str) -> None:
    print(f"[issue-50] {message}", flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Issue50EvidenceError(f"cannot load {path}") from error
    if not isinstance(value, dict):
        raise Issue50EvidenceError(f"{path} must contain a JSON object")
    return value


def _load_issue_44_sources() -> dict[str, Any]:
    runtime = _load_json(ISSUE_44_ROOT / "runtime-bundle-manifest.json")
    bundle = _load_json(ISSUE_44_ROOT / "capture-bundle-manifest.json")
    capability = load_physics_capture_v2_capability_report(
        ISSUE_44_ROOT / "capability-report.json"
    ).record
    if (
        runtime.get("schema") != "issue_44_physics_v2_runtime_evidence_bundle_v1"
        or bundle.get("runtime_evidence_bundle_identity") != runtime.get("identity")
    ):
        raise Issue50EvidenceError("issue-44 source bundles are stale")
    if any(fact.get("status") != "demonstrated" for fact in capability["facts"].values()):
        raise Issue50EvidenceError("issue-44 capability report is not fully demonstrated")
    captures = {}
    for case in ISSUE_44_CASES:
        path = ISSUE_44_ROOT / "captures" / f"{case}.json"
        capture = parse_physics_capture_v2(_load_json(path))
        declared = bundle["captures"].get(case)
        probe = next(item for item in capability["probes"] if item["case"] == case)
        if (
            not isinstance(declared, Mapping)
            or declared.get("capture_id") != capture.capture_id
            or probe.get("capture_id") != capture.capture_id
            or probe.get("final_evaluation") is not False
            or probe.get("source") != "unity_exporter_probe"
        ):
            raise Issue50EvidenceError(f"{case}: issue-44 source binding is stale")
        captures[case] = capture
    return {
        "runtime_bundle": runtime,
        "capture_bundle": bundle,
        "capability_report": capability,
        "captures": captures,
    }


def _load_probe_sources(probe_root: Path) -> dict[str, Any]:
    runtime = _load_json(probe_root / "runtime-bundle-manifest.json")
    bundle = _load_json(probe_root / "capture-bundle-manifest.json")
    if (
        runtime.get("schema") != "issue_50_physical_violation_probe_runtime_bundle_v1"
        or runtime.get("evidence_source") != "unity_runtime_non_fixture"
        or runtime.get("final_evaluation") is not False
        or runtime.get("probe_environment")
        != "NOVPHY_ISSUE_50_CAPABILITY_PROBE=unsupported-stationary-v1"
    ):
        raise Issue50EvidenceError("issue-50 probe runtime authority is ineligible")
    if (
        bundle.get("schema") != "issue_50_physical_violation_probe_capture_bundle_v1"
        or bundle.get("runtime_bundle_identity") != runtime.get("identity")
    ):
        raise Issue50EvidenceError("issue-50 probe capture bundle is stale")
    declared = bundle.get("captures")
    if not isinstance(declared, Mapping) or tuple(sorted(declared)) != PROBE_CASES:
        raise Issue50EvidenceError("issue-50 probe capture membership is incomplete")
    probes = {probe["case"]: probe for probe in runtime.get("probes", [])}
    captures = {}
    records = {}
    for case in PROBE_CASES:
        path = probe_root / "captures" / f"{case}.json"
        record = _load_json(path)
        capture = parse_physics_capture_v2(record)
        entry = declared[case]
        probe = probes.get(case)
        expected_bindings = {
            "capture_id": capture.capture_id,
            "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
            "level_instance_id": capture.source_bindings["level_instance_id"],
            "scenario_template_id": capture.source_bindings["scenario_template_id"],
        }
        if (
            not isinstance(entry, Mapping)
            or entry.get("path") != f"captures/{case}.json"
            or not isinstance(probe, Mapping)
            or any(probe.get(field) != value for field, value in expected_bindings.items())
            or entry.get("capture_id") != capture.capture_id
        ):
            raise Issue50EvidenceError(f"{case}: issue-50 probe binding is stale")
        captures[case] = capture
        records[case] = record
    return {
        "runtime_bundle": runtime,
        "capture_bundle": bundle,
        "probe_plan": _load_json(probe_root / "probe-plan.json"),
        "scenario_manifests": {
            case: _load_json(probe_root / "manifests" / f"{case}.json")
            for case in ("floating-a", "floating-b")
        },
        "capture_records": records,
        "captures": captures,
    }


def adjudication_plan(
    issue_44_capture_bundle_identity: str,
    probe_capture_bundle_identity: str,
) -> dict[str, Any]:
    return {
        "schema": "issue_50_v2_physical_violation_adjudication_plan_v1",
        "identity": PLAN_IDENTITY,
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "source_capture_bundle_identities": [
            issue_44_capture_bundle_identity,
            probe_capture_bundle_identity,
        ],
        "predicates": list(ACCEPTED_PREDICATES),
        "selection_rules": {
            "excess_penetration": {
                "cases": list(PENETRATION_CASES),
                "positive": "same-step collision event",
                "negative": "immediately preceding complete fixed step",
                "boundary": "previous/collision/next fixed-step window",
            },
            "unsupported_stationary_or_floating_body": {
                "cases": list(UNSUPPORTED_CASES),
                "positive": "earliest available true label for authored block:0000",
                "negative": "same-lineage stationary gravity-applicable supported body",
                "motion_boundary": "first two authored-body fixed steps",
                "support_boundary": "same-step authored unsupported and supported bodies",
            },
            "unavailable": "first authored-body fixed-step endpoint",
            "aggregate": "verify true dominance and unavailable preservation",
        },
        "minimum_floor": {
            "positive_witnesses": MINIMUM_WITNESS_COUNT,
            "negative_witnesses": MINIMUM_WITNESS_COUNT,
            "boundary_windows": MINIMUM_BOUNDARY_WINDOW_COUNT,
            "non_final_scenario_lineages": MINIMUM_WITNESS_COUNT,
            "level_instances": MINIMUM_WITNESS_COUNT,
            "scenario_templates": MINIMUM_WITNESS_COUNT,
            "unavailable_or_invalidation_checks": 1,
        },
        "outcome_conditioned_retention": False,
        "spsg_negative_training_examples": False,
        "physical_regime_label_used": False,
    }


def _event_step(capture: Any, event_type: str) -> int:
    matches = [
        event["fixed_step"]
        for event in capture.record["events"]
        if event["event_type"] == event_type
    ]
    if len(matches) != 1:
        raise Issue50EvidenceError(
            f"{capture.capture_id}: expected exactly one {event_type} event"
        )
    return matches[0]


def _record_at(derivation: Mapping[str, Any], fixed_step: int) -> Mapping[str, Any]:
    return next(
        record for record in derivation["labels"] if record["fixed_step"] == fixed_step
    )


def _label_at(
    derivation: Mapping[str, Any],
    fixed_step: int,
    predicate: str,
    entity_id: str | None = None,
) -> Mapping[str, Any]:
    value = _record_at(derivation, fixed_step)["predicates"][predicate]
    if predicate == EXCESS_PENETRATION:
        return value
    return next(label for label in value if label["entity_id"] == entity_id)


def _entity_id(capture: Any, scenario_object_id: str) -> str:
    matches = {
        entity["entity_id"]
        for sample in capture.record["fixed_step_samples"]
        for entity in sample["entities"]
        if entity["scenario_object_id"] == scenario_object_id
    }
    if len(matches) != 1:
        raise Issue50EvidenceError(
            f"{capture.capture_id}: cannot resolve {scenario_object_id}"
        )
    return matches.pop()


def _witness(
    case: str,
    capture: Any,
    derivation: Mapping[str, Any],
    predicate: str,
    fixed_step: int,
    entity_id: str | None = None,
) -> dict[str, Any]:
    label = _label_at(derivation, fixed_step, predicate, entity_id)
    return {
        "case": case,
        "capture_id": capture.capture_id,
        "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
        "level_instance_id": capture.source_bindings["level_instance_id"],
        "scenario_template_id": capture.source_bindings["scenario_template_id"],
        "fixed_step": fixed_step,
        "entity_id": entity_id,
        "value": label["value"],
        "availability": label["availability"],
        "source_records": label["source_records"],
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
    result["passed"] = min(result.values()) >= MINIMUM_WITNESS_COUNT
    return result


def _supported_negative(
    case: str,
    capture: Any,
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    for record in derivation["labels"]:
        for label in record["predicates"][UNSUPPORTED_STATIONARY]:
            evidence = label["evidence"]
            if (
                label["availability"] == "available"
                and label["value"] is False
                and evidence.get("gravity_applicable_all_steps") is True
                and evidence.get("stationary") is True
                and evidence.get("supported") is True
            ):
                return _witness(
                    case,
                    capture,
                    derivation,
                    UNSUPPORTED_STATIONARY,
                    record["fixed_step"],
                    label["entity_id"],
                )
    raise Issue50EvidenceError(f"{case}: no supported stationary negative witness")


def _nonstationary_negative(
    case: str,
    capture: Any,
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    for record in derivation["labels"]:
        for label in record["predicates"][UNSUPPORTED_STATIONARY]:
            evidence = label["evidence"]
            if (
                label["availability"] == "available"
                and label["value"] is False
                and evidence.get("gravity_applicable_all_steps") is True
                and evidence.get("stationary") is False
                and evidence.get("supported") is False
            ):
                return _witness(
                    case,
                    capture,
                    derivation,
                    UNSUPPORTED_STATIONARY,
                    record["fixed_step"],
                    label["entity_id"],
                )
    raise Issue50EvidenceError(f"{case}: no unsupported nonstationary negative witness")


def _mutation_checks(issue_44: Mapping[str, Any], probes: Mapping[str, Any]) -> list[dict[str, str]]:
    checks = []
    mutations = []
    collision = issue_44["captures"]["collision"].record
    missing_geometry = deepcopy(collision)
    del missing_geometry["fixed_step_samples"][0]["colliders"][0]["shape"]
    mutations.append(("remove_collider_geometry", missing_geometry))
    incomplete_contacts = deepcopy(collision)
    incomplete_contacts["fixed_step_samples"][0]["complete_raw_non_trigger_contacts"] = False
    mutations.append(("mark_contact_enumeration_incomplete", incomplete_contacts))
    nonfinite = deepcopy(collision)
    nonfinite["fixed_step_samples"][0]["contacts"][0]["separation"] = float("nan")
    mutations.append(("make_contact_separation_nonfinite", nonfinite))
    floating = probes["captures"][UNSUPPORTED_CASES[0]].record
    missing_gravity = deepcopy(floating)
    del missing_gravity["fixed_step_samples"][0]["entities"][0]["body"][
        "gravity_applicable"
    ]
    mutations.append(("remove_gravity_applicability", missing_gravity))
    missing_lifecycle = deepcopy(floating)
    del missing_lifecycle["fixed_step_samples"][0]["entities"][0]
    mutations.append(("truncate_entity_lifecycle", missing_lifecycle))
    for name, record in mutations:
        try:
            parse_physics_capture_v2(record)
        except ValueError as error:
            checks.append(
                {
                    "mutation": name,
                    "expected": "whole_rollout_rejected",
                    "observed": "whole_rollout_rejected",
                    "reason": str(error),
                }
            )
        else:
            raise Issue50EvidenceError(f"{name} mutation did not fail closed")
    capture = issue_44["captures"]["collision"]
    stale = derive_capture_physical_violations(
        capture,
        source_reference="data/runtime_evidence/issue-44/captures/collision.json",
        source_capture_bundle_identity="stale-cross-release-capture-bundle",
    )
    try:
        validate_capture_physical_violation_derivation(
            stale,
            capture,
            source_reference="data/runtime_evidence/issue-44/captures/collision.json",
            source_capture_bundle_identity=issue_44["capture_bundle"]["identity"],
        )
    except ValueError as error:
        checks.append(
            {
                "mutation": "change_source_capture_bundle_binding",
                "expected": "whole_derivation_rejected",
                "observed": "whole_derivation_rejected",
                "reason": str(error),
            }
        )
    else:
        raise Issue50EvidenceError("cross-release source mutation did not fail closed")
    return checks


def _adjudication(
    issue_44: Mapping[str, Any],
    probes: Mapping[str, Any],
    derivations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    excess_positive = []
    excess_negative = []
    excess_boundaries = []
    for case in PENETRATION_CASES:
        capture = issue_44["captures"][case]
        derived = derivations[f"issue44-{case}"]
        step = _event_step(capture, "collision")
        positive = _witness(case, capture, derived, EXCESS_PENETRATION, step)
        negative = _witness(case, capture, derived, EXCESS_PENETRATION, step - 1)
        window = [
            _witness(case, capture, derived, EXCESS_PENETRATION, current)
            for current in (step - 1, step, step + 1)
        ]
        if [item["value"] for item in window] != [False, True, False]:
            raise Issue50EvidenceError(f"{case}: excess-penetration boundary did not cross")
        excess_positive.append(positive)
        excess_negative.append(negative)
        excess_boundaries.append(
            {
                "case": case,
                "capture_id": capture.capture_id,
                "fixed_steps": [item["fixed_step"] for item in window],
                "values": [item["value"] for item in window],
                "source_records": [item["source_records"] for item in window],
            }
        )

    unsupported_positive = []
    unsupported_negative = []
    unsupported_boundaries = []
    unavailable = []
    aggregate_checks = []
    for case in UNSUPPORTED_CASES:
        capture = probes["captures"][case]
        derived = derivations[f"issue50-{case}"]
        entity_id = _entity_id(capture, "block:0000")
        first_step = derived["labels"][0]["fixed_step"]
        second_step = derived["labels"][1]["fixed_step"]
        first = _witness(
            case, capture, derived, UNSUPPORTED_STATIONARY, first_step, entity_id
        )
        second = _witness(
            case, capture, derived, UNSUPPORTED_STATIONARY, second_step, entity_id
        )
        if (
            first["value"] is not None
            or first["availability"] != "unavailable_incomplete_stability_window"
            or second["value"] is not True
        ):
            raise Issue50EvidenceError(f"{case}: authored stationary witness did not cross")
        supported = _supported_negative(case, capture, derived)
        nonstationary = _nonstationary_negative(case, capture, derived)
        unsupported_positive.append(second)
        unsupported_negative.append(supported)
        unavailable.append(first)
        unsupported_boundaries.append(
            {
                "case": case,
                "kind": "stability_window_and_support_boundary",
                "authored_entity_id": entity_id,
                "fixed_steps": [first_step, second_step],
                "authored_values": [None, True],
                "same_step_supported_negative": supported,
                "same_lineage_motion_negative": nonstationary,
            }
        )
        first_aggregate = _record_at(derived, first_step)["aggregate"]
        second_aggregate = _record_at(derived, second_step)["aggregate"]
        aggregate_checks.append(
            {
                "case": case,
                "first_fixed_step": first_step,
                "first_value": first_aggregate["value"],
                "first_availability": first_aggregate["availability"],
                "second_fixed_step": second_step,
                "second_value": second_aggregate["value"],
                "second_availability": second_aggregate["availability"],
                "passed": (
                    first_aggregate["value"] is None
                    and first_aggregate["availability"] == "unavailable_component"
                    and second_aggregate["value"] is True
                    and second_aggregate["availability"] == "available"
                ),
            }
        )

    coverage = {
        EXCESS_PENETRATION: {
            "positive": _coverage(excess_positive, True),
            "negative": _coverage(excess_negative, False),
            "boundary_window_count": len(excess_boundaries),
        },
        UNSUPPORTED_STATIONARY: {
            "positive": _coverage(unsupported_positive, True),
            "negative": _coverage(unsupported_negative, False),
            "boundary_window_count": len(unsupported_boundaries),
        },
    }
    mutations = _mutation_checks(issue_44, probes)
    passed = (
        all(
            coverage[predicate][kind]["passed"]
            for predicate in ACCEPTED_PREDICATES
            for kind in ("positive", "negative")
        )
        and all(
            coverage[predicate]["boundary_window_count"]
            >= MINIMUM_BOUNDARY_WINDOW_COUNT
            for predicate in ACCEPTED_PREDICATES
        )
        and all(item["value"] is None for item in unavailable)
        and all(item["passed"] for item in aggregate_checks)
        and all(item["observed"] == item["expected"] for item in mutations)
    )
    return {
        "schema": "issue_50_v2_physical_violation_adjudication_v1",
        "identity": ADJUDICATION_IDENTITY,
        "plan_identity": PLAN_IDENTITY,
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "predicate_dispositions": {
            predicate: "accepted_authoritative" for predicate in ACCEPTED_PREDICATES
        },
        "witnesses": {
            EXCESS_PENETRATION: {
                "positive": excess_positive,
                "negative": excess_negative,
                "boundary_windows": excess_boundaries,
            },
            UNSUPPORTED_STATIONARY: {
                "positive": unsupported_positive,
                "negative": unsupported_negative,
                "boundary_windows": unsupported_boundaries,
            },
            "unavailable": unavailable,
        },
        "coverage": coverage,
        "aggregate": {
            "predicate": AGGREGATE_PREDICATE,
            "semantics": "unavailable-preserving Kleene any",
            "checks": aggregate_checks,
        },
        "mutation_checks": mutations,
        "exclusions": {
            predicate: "excluded_not_emitted_not_false"
            for predicate in EXCLUDED_PREDICATES
        },
        "physical_regime_label_used": False,
        "passed": passed,
    }


def _expected_artifacts(probe_root: Path) -> dict[str, dict[str, Any]]:
    issue_44 = _load_issue_44_sources()
    probes = _load_probe_sources(probe_root)
    source_authorities = [
        {
            "runtime_bundle_identity": issue_44["runtime_bundle"]["identity"],
            "source_snapshot_commit": issue_44["runtime_bundle"]["source_snapshot_commit"],
        },
        {
            "runtime_bundle_identity": probes["runtime_bundle"]["identity"],
            "source_snapshot_commit": probes["runtime_bundle"]["source_snapshot_commit"],
        },
    ]
    derivations = {}
    for case, capture in issue_44["captures"].items():
        key = f"issue44-{case}"
        reference = f"data/runtime_evidence/issue-44/captures/{case}.json"
        derivations[key] = derive_capture_physical_violations(
            capture,
            source_reference=reference,
            source_capture_bundle_identity=issue_44["capture_bundle"]["identity"],
        )
    for case, capture in probes["captures"].items():
        key = f"issue50-{case}"
        reference = f"data/runtime_evidence/issue-50/source-probes/captures/{case}.json"
        derivations[key] = derive_capture_physical_violations(
            capture,
            source_reference=reference,
            source_capture_bundle_identity=probes["capture_bundle"]["identity"],
        )
    adjudication = _adjudication(issue_44, probes, derivations)
    if not adjudication["passed"]:
        raise Issue50EvidenceError("issue-50 representative evidence floor did not pass")
    artifacts = {
        "derivation-spec.json": derivation_spec(source_authorities=source_authorities),
        "adjudication-plan.json": adjudication_plan(
            issue_44["capture_bundle"]["identity"],
            probes["capture_bundle"]["identity"],
        ),
        "physical-violation-adjudication.json": adjudication,
        **{f"derivations/{key}.json": value for key, value in derivations.items()},
        "source-probes/runtime-bundle-manifest.json": probes["runtime_bundle"],
        "source-probes/capture-bundle-manifest.json": probes["capture_bundle"],
        "source-probes/probe-plan.json": probes["probe_plan"],
        **{
            f"source-probes/manifests/{case}.json": value
            for case, value in probes["scenario_manifests"].items()
        },
        **{
            f"source-probes/captures/{case}.json": value
            for case, value in probes["capture_records"].items()
        },
    }
    members = [
        {
            "path": path,
            "schema": value.get("schema", value.get("schema_version")),
            "identity": value.get("identity", value.get("capture_id")),
        }
        for path, value in sorted(artifacts.items())
    ]
    artifacts["bundle-manifest.json"] = {
        "schema": "issue_50_v2_physical_violation_bundle_v1",
        "identity": BUNDLE_IDENTITY,
        "derivation_spec_identity": DERIVATION_SPEC_IDENTITY,
        "source_runtime_bundle_identities": [
            authority["runtime_bundle_identity"] for authority in source_authorities
        ],
        "source_capture_bundle_identities": [
            issue_44["capture_bundle"]["identity"],
            probes["capture_bundle"]["identity"],
        ],
        "artifacts": members,
        "accepted_derivation_identities": [
            value["identity"] for value in derivations.values()
        ],
        "adjudication_identity": ADJUDICATION_IDENTITY,
        "passed": True,
    }
    return artifacts


def validate_issue_50_evidence(
    evidence_root: Path,
    *,
    probe_root: Path = DEFAULT_PROBE_ROOT,
) -> dict[str, Any]:
    expected = _expected_artifacts(Path(probe_root))
    actual_paths = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*.json")
    }
    if actual_paths != set(expected):
        raise Issue50EvidenceError("issue-50 bundle membership is incomplete or contains extras")
    for relative_path, expected_value in expected.items():
        actual = _load_json(evidence_root / relative_path)
        if actual != expected_value:
            raise Issue50EvidenceError(f"{relative_path} differs from exact re-derivation")
        if not finite_json_tree(actual):
            raise Issue50EvidenceError(f"{relative_path} contains non-finite numeric evidence")
    return {
        "schema": "issue_50_v2_physical_violation_validation_result_v1",
        "bundle_identity": BUNDLE_IDENTITY,
        "capture_count": len(ISSUE_44_CASES) + len(PROBE_CASES),
        "label_count": sum(
            value["label_count"]
            for path, value in expected.items()
            if path.startswith("derivations/")
        ),
        "passed": True,
    }


def build_issue_50_evidence(
    output: Path,
    *,
    probe_root: Path = DEFAULT_PROBE_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    _log("validating issue-44 and issue-50 non-fixture Unity source bundles")
    artifacts = _expected_artifacts(Path(probe_root))
    adjudication = artifacts["physical-violation-adjudication.json"]
    label_count = sum(
        value["label_count"]
        for path, value in artifacts.items()
        if path.startswith("derivations/")
    )
    _log(f"derived {len(ISSUE_44_CASES) + len(PROBE_CASES)} captures and {label_count} labels")
    _log(
        "witness floor passed: "
        + "; ".join(
            f"{predicate}="
            f"{adjudication['coverage'][predicate]['positive']['witness_count']} positive/"
            f"{adjudication['coverage'][predicate]['negative']['witness_count']} negative/"
            f"{adjudication['coverage'][predicate]['boundary_window_count']} boundary windows"
            for predicate in ACCEPTED_PREDICATES
        )
    )
    _log(
        f"fail-closed checks passed: {len(adjudication['mutation_checks'])} mutations; "
        "aggregate preserved unavailable"
    )
    if dry_run:
        _log("dry-run complete; no files written")
        return {
            "schema": "issue_50_v2_physical_violation_dry_run_v1",
            "output": str(output),
            "artifact_count": len(artifacts),
            "label_count": label_count,
            "passed": True,
        }
    if output.exists():
        raise Issue50EvidenceError(f"immutable issue-50 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _log("publishing the canonical immutable bundle")
    with tempfile.TemporaryDirectory(prefix=".issue-50-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        for relative_path, value in artifacts.items():
            write_immutable_cohort_v2_json(value, staging / relative_path)
        result = validate_issue_50_evidence(staging, probe_root=probe_root)
        os.replace(staging, output)
    _log(f"publication complete: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate and publish cohort-v2 physical violations for issue #50"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_issue_50_evidence(
        args.output,
        probe_root=args.probe_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
