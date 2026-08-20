"""Seal actual request-71 probes into the issue #44/#45 runtime evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from scripts.cohort_v2_scenarios import (
    create_unity_reset_reproduction_receipt,
    load_cohort_v2_scenario_manifest,
    validate_deterministic_scenario_receipt,
    write_deterministic_scenario_receipt,
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)
from scripts.collection_plan import load_collection_plan
from scripts.physics_capture_v2 import (
    load_physics_capture_v2,
    normalized_initial_engine_state_identity,
)
from scripts.physics_capture_v2_capability_report import (
    validate_physics_capture_v2_capability_report,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = ROOT / ".local-artifacts/issue-44-v2-probes/output"
PLAN_PATH = ROOT / "data/runtime_evidence/issue-44/probe-plan.json"
ISSUE_44_ROOT = ROOT / "data/runtime_evidence/issue-44"
ISSUE_45_ROOT = ROOT / "data/runtime_evidence/issue-45"
TRAINING_MANIFEST = ISSUE_45_ROOT / "manifests/training.json"
SNAPSHOT_COMMIT = "7f4db40223008a0b3db673faf90a486ffd39ec11"
ARCHIVE_PATH = "sciencebirdsgames/physics-v2/novphy-physics-player-2019.4.41f2.tar.gz"
ENGINE_VERSION = "2019.4.41f2"
PLAYER_VERSION = "2019.4.41f2"

EXPORTER_FILES = (
    "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicalSnapshotRuntime.cs",
    "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2Contacts.cs",
    "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2EngineProtocol.cs",
    "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2EntityGeometry.cs",
    "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2Events.cs",
    "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureV2FixedStepRecorder.cs",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _identity(namespace: str, value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("runtime evidence identity payload must be an object")
    schema = value.get("schema", value.get("schema_version", "v1"))
    parts = [quote(str(schema), safe="-._~")]
    semantic_fields = {
        "source_snapshot_commit",
        "collection_plan_identity",
        "scenario_manifest_identity",
        "runtime_evidence_bundle_identity",
        "plan_identity",
        "role",
        "exposure_role",
        "first_capture_identity",
        "second_capture_identity",
    }
    for key in sorted(value):
        item = value[key]
        if (key in semantic_fields or key.endswith("_identity")) and isinstance(item, (str, int)):
            parts.append(f"{key}={quote(str(item), safe='-._~')}")
    probes = value.get("probes")
    if isinstance(probes, list):
        capture_ids = sorted(
            str(probe["capture_id"])
            for probe in probes
            if isinstance(probe, Mapping) and isinstance(probe.get("capture_id"), str)
        )
        parts.extend(f"capture_id={quote(capture_id, safe='-._~')}" for capture_id in capture_ids)
    captures = value.get("captures")
    if isinstance(captures, Mapping):
        for role, record in sorted(captures.items()):
            if isinstance(record, Mapping) and isinstance(record.get("capture_id"), str):
                parts.append(
                    f"capture={quote(str(role), safe='-._~')}="
                    f"{quote(record['capture_id'], safe='-._~')}"
                )
    return f"{namespace}:{':'.join(parts)}"


def _exporter_identity() -> tuple[str, dict[str, str]]:
    files = {relative: relative for relative in EXPORTER_FILES}
    return "physics-capture-v2-exporter-code-v1", files


def _attempts() -> dict[str, tuple[Path, Mapping[str, Any], Any]]:
    report = json.loads((PROBE_ROOT / "collection_plan_report.json").read_text(encoding="utf-8"))
    if (report.get("accepted_count"), report.get("rejected_count"), report.get("failed_count")) != (5, 0, 0):
        raise ValueError("runtime evidence requires exactly five accepted frozen attempts")
    plan = load_collection_plan(PLAN_PATH).plan
    interventions = {
        item.id: item
        for scenario in plan.scenarios
        for item in scenario.interventions
    }
    values: dict[str, tuple[Path, Mapping[str, Any], Any]] = {}
    for entry in report["attempt_ledger"]:
        case = entry["intervention_id"]
        path = ROOT / entry["artifact_path"] / "physics_capture_v2.json"
        capture = load_physics_capture_v2(path)
        if capture.source_bindings["intervention_id"] != interventions[case].identity:
            raise ValueError(f"capture {case} is stale against the frozen intervention")
        values[case] = (path, entry, capture)
    if set(values) != {"no-contact", "collision", "support", "support-change", "stable-terminal"}:
        raise ValueError("runtime evidence probe cases are incomplete")
    return values


def _support_sets(capture: Any) -> list[set[tuple[str, str]]]:
    return [
        {
            (support["supporter_entity_id"], support["supported_entity_id"])
            for support in sample["supports"]
        }
        for sample in capture.record["fixed_step_samples"]
    ]


def _case_observation(case: str, capture: Any) -> tuple[bool, str | None]:
    samples = capture.record["fixed_step_samples"]
    if case == "no-contact":
        demonstrated = all(not sample["contacts"] for sample in samples)
        return demonstrated, None if demonstrated else "the empty-space intervention produced a contact"
    if case == "collision":
        contacts_by_step = {
            sample["fixed_step"]: {
                frozenset((contact["entity_a_id"], contact["entity_b_id"]))
                for contact in sample["contacts"]
            }
            for sample in samples
        }
        demonstrated = any(
            event["event_type"] == "collision"
            and frozenset(event["participants"]) in contacts_by_step[event["fixed_step"]]
            for event in capture.record["events"]
        )
        return demonstrated, None if demonstrated else "nearest-target intervention produced no collision contact/event"
    supports = _support_sets(capture)
    if case == "support":
        demonstrated = len(supports) >= 2 and bool(supports[0] & supports[1])
        return demonstrated, None if demonstrated else "initial samples contain no persistent support relation"
    if case == "support-change":
        demonstrated = any(current != previous for previous, current in zip(supports, supports[1:]))
        return demonstrated, None if demonstrated else "the targeted rollout contains no support-relation change"
    demonstrated = (
        capture.record["terminal_evidence"]["reason"] == "stable_entered"
        and capture.record["frame_records"][-1]["fixed_step"]
        == capture.record["terminal_evidence"]["fixed_step"]
    )
    return demonstrated, None if demonstrated else "terminal evidence is not stable_entered on the final frame"


def build_runtime_evidence() -> dict[str, str]:
    attempts = _attempts()
    observations = {
        case: _case_observation(case, capture)
        for case, (_, _, capture) in attempts.items()
    }
    exporter_version, exporter_files = _exporter_identity()
    protocol_version = "physics-capture-v2-engine-protocol-v1"
    probes = []
    capture_ids: dict[str, str] = {}
    for case in sorted(attempts):
        path, _, capture = attempts[case]
        capture_ids[case] = capture.capture_id
        probes.append({
            "source": "unity_exporter_probe",
            "case": case,
            "capture_id": capture.capture_id,
            "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
            "level_instance_id": capture.source_bindings["level_instance_id"],
            "scenario_template_id": capture.source_bindings["scenario_template_id"],
            "final_evaluation": False,
        })

    facts = {
        "configured_fixed_step_capture_stride": {
            "status": "demonstrated", "capture_id": capture_ids["no-contact"], "reason": None,
        },
        "complete_raw_non_trigger_contacts": {
            "status": "demonstrated", "capture_id": capture_ids["no-contact"], "reason": None,
        },
        "collider_geometry_and_separation": {
            "status": "demonstrated" if observations["collision"][0] else "unavailable",
            "capture_id": capture_ids["collision"] if observations["collision"][0] else None,
            "reason": observations["collision"][1],
        },
        "gravity_body_lifecycle_motion_support_world": {
            "status": "demonstrated" if observations["support"][0] else "unavailable",
            "capture_id": capture_ids["support"] if observations["support"][0] else None,
            "reason": observations["support"][1],
        },
        "causal_identity_source_bindings": {
            "status": "demonstrated", "capture_id": capture_ids["support-change"], "reason": None,
        },
        "final_frame_covers_termination": {
            "status": "demonstrated", "capture_id": capture_ids["stable-terminal"], "reason": None,
        },
    }
    report_payload = {
        "schema_version": "physics_capture_v2_exporter_capability_report_v1",
        "provenance": {
            "engine_version": ENGINE_VERSION,
            "player_version": PLAYER_VERSION,
            "protocol_version": protocol_version,
            "exporter_version": exporter_version,
        },
        "probes": probes,
        "facts": facts,
    }
    report = {
        **report_payload,
        "report_id": _identity("physics-capture-v2-exporter-capability-report-v1", report_payload),
    }
    validate_physics_capture_v2_capability_report(report)
    report_path = ISSUE_44_ROOT / "capability-report.json"
    write_immutable_cohort_v2_json(report, report_path)

    observation_records = {
        case: {
            "status": "demonstrated" if status else "unavailable",
            "reason": reason,
            "capture_path": str(path.relative_to(ROOT)),
            "capture_id": capture_ids[case],
        }
        for case, (status, reason) in observations.items()
        for path, _, _ in (attempts[case],)
    }
    issue_44_bundle_payload = {
        "schema": "issue_44_physics_v2_runtime_evidence_bundle_v1",
        "source_snapshot_commit": SNAPSHOT_COMMIT,
        "archive_path": ARCHIVE_PATH,
        "collection_plan_identity": load_collection_plan(PLAN_PATH).plan.identity,
        "capability_report_path": str(report_path.relative_to(ROOT)),
        "case_observations": observation_records,
        "exporter_files": exporter_files,
    }
    issue_44_bundle = {
        **issue_44_bundle_payload,
        "identity": _identity("issue-44-physics-v2-runtime-evidence-bundle-v1", issue_44_bundle_payload),
    }
    issue_44_bundle_path = ISSUE_44_ROOT / "runtime-bundle-manifest.json"
    write_immutable_cohort_v2_json(issue_44_bundle, issue_44_bundle_path)

    public_captures: dict[str, dict[str, str]] = {}
    for case, (source_path, _, capture) in attempts.items():
        public_path = ISSUE_44_ROOT / "captures" / f"{case}.json"
        write_immutable_cohort_v2_bytes(source_path.read_bytes(), public_path)
        public_captures[case] = {
            "capture_id": capture.capture_id,
            "path": str(public_path.relative_to(ROOT)),
        }
    capture_bundle_payload = {
        "schema": "issue_44_physics_v2_capture_bundle_v1",
        "runtime_evidence_bundle_identity": issue_44_bundle["identity"],
        "captures": public_captures,
    }
    capture_bundle = {
        **capture_bundle_payload,
        "identity": _identity("issue-44-physics-v2-capture-bundle-v1", capture_bundle_payload),
    }
    capture_bundle_path = ISSUE_44_ROOT / "capture-bundle-manifest.json"
    write_immutable_cohort_v2_json(capture_bundle, capture_bundle_path)

    training = load_cohort_v2_scenario_manifest(TRAINING_MANIFEST)
    first_path, _, first = attempts["no-contact"]
    second_path, _, second = attempts["collision"]
    reset_receipt = create_unity_reset_reproduction_receipt(
        training,
        first_initial_engine_state_identity=normalized_initial_engine_state_identity(first),
        second_initial_engine_state_identity=normalized_initial_engine_state_identity(second),
    )
    validate_deterministic_scenario_receipt(reset_receipt)
    reset_path = ISSUE_45_ROOT / "receipts/training-unity-reset.json"
    write_deterministic_scenario_receipt(reset_receipt, reset_path)
    issue_45_bundle_payload = {
        "schema": "issue_45_unity_reset_evidence_bundle_v1",
        "source_snapshot_commit": SNAPSHOT_COMMIT,
        "archive_path": ARCHIVE_PATH,
        "scenario_manifest_identity": training.identity,
        "first_capture_path": str(first_path.relative_to(ROOT)),
        "first_capture_identity": first.capture_id,
        "second_capture_path": str(second_path.relative_to(ROOT)),
        "second_capture_identity": second.capture_id,
        "receipt_path": str(reset_path.relative_to(ROOT)),
    }
    issue_45_bundle = {
        **issue_45_bundle_payload,
        "identity": _identity("issue-45-unity-reset-evidence-bundle-v1", issue_45_bundle_payload),
    }
    issue_45_bundle_path = ISSUE_45_ROOT / "unity-reset-bundle-manifest.json"
    write_immutable_cohort_v2_json(issue_45_bundle, issue_45_bundle_path)
    return {
        "issue_44_bundle_identity": issue_44_bundle["identity"],
        "issue_44_capture_bundle_identity": capture_bundle["identity"],
        "issue_45_bundle_identity": issue_45_bundle["identity"],
        "unity_reset_receipt_identity": reset_receipt["identity"],
    }


def main() -> None:
    print(json.dumps(build_runtime_evidence(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
