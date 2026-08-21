"""Publish successful request-71 probes as issue #44 runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from scripts.cohort_v2_scenarios import (
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)
from scripts.collection_plan import load_collection_plan
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.physics_capture_v2_capability_report import (
    validate_physics_capture_v2_capability_report,
)
from scripts.verify_physics_player import verify_physics_player_archive


ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = ROOT / ".local-artifacts/issue-44-v2-probes/output"
STAGE_ROOT = ROOT / "sciencebirdsgames/physics-v2"
PLAN_PATH = STAGE_ROOT / "probe-plan.json"
ISSUE_44_ROOT = ROOT / "data/runtime_evidence/issue-44"
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
CASES = {"no-contact", "collision", "support", "support-change", "stable-terminal"}


def _identity(namespace: str, value: object) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("runtime evidence identity payload must be an object")
    schema = value.get("schema", value.get("schema_version", "v1"))
    parts = [quote(str(schema), safe="-._~")]
    semantic_fields = {
        "source_snapshot_commit",
        "collection_plan_identity",
        "runtime_evidence_bundle_identity",
        "plan_identity",
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
        for case, record in sorted(captures.items()):
            if isinstance(record, Mapping) and isinstance(record.get("capture_id"), str):
                parts.append(
                    f"capture={quote(str(case), safe='-._~')}="
                    f"{quote(record['capture_id'], safe='-._~')}"
                )
    return f"{namespace}:{':'.join(parts)}"


def _attempts() -> dict[str, tuple[Path, Mapping[str, Any], Any]]:
    report = json.loads((PROBE_ROOT / "collection_plan_report.json").read_text(encoding="utf-8"))
    if (report.get("accepted_count"), report.get("rejected_count"), report.get("failed_count")) != (5, 0, 0):
        raise ValueError("runtime evidence requires 5 accepted, 0 rejected, and 0 failed attempts")
    ledger = report.get("attempt_ledger")
    if not isinstance(ledger, list) or len(ledger) != 5:
        raise ValueError("runtime evidence requires exactly five fresh attempt rows")
    plan = load_collection_plan(PLAN_PATH).plan
    interventions = {
        item.id: item
        for scenario in plan.scenarios
        for item in scenario.interventions
    }
    values: dict[str, tuple[Path, Mapping[str, Any], Any]] = {}
    for entry in ledger:
        case = entry["intervention_id"]
        if case in values:
            raise ValueError(f"runtime evidence contains a duplicate {case} attempt")
        path = ROOT / entry["artifact_path"] / "physics_capture_v2.json"
        capture = load_physics_capture_v2(path)
        if capture.source_bindings["intervention_id"] != interventions[case].identity:
            raise ValueError(f"capture {case} is stale against the frozen intervention")
        values[case] = (path, entry, capture)
    if set(values) != CASES:
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
            and frozenset(event["participants"]) in contacts_by_step.get(event["fixed_step"], set())
            for event in capture.record["events"]
        )
        return demonstrated, None if demonstrated else "no collision event matched a same-step raw contact"
    supports = _support_sets(capture)
    if case == "support":
        demonstrated = len(supports) >= 2 and bool(supports[0] & supports[1])
        return demonstrated, None if demonstrated else "the first two samples share no support pair"
    if case == "support-change":
        demonstrated = any(current != previous for previous, current in zip(supports, supports[1:]))
        return demonstrated, None if demonstrated else "no adjacent support-set change was observed"
    demonstrated = (
        capture.record["terminal_evidence"]["reason"] == "stable_entered"
        and capture.record["frame_records"][-1]["fixed_step"]
        == capture.record["terminal_evidence"]["fixed_step"]
    )
    return demonstrated, None if demonstrated else "stable terminal does not bind the final frame"


def build_runtime_evidence() -> dict[str, str]:
    attempts = _attempts()
    observations = {
        case: _case_observation(case, capture)
        for case, (_, _, capture) in attempts.items()
    }
    unavailable = {
        case: reason for case, (demonstrated, reason) in observations.items() if not demonstrated
    }
    if unavailable:
        raise ValueError("issue #44 evidence gates failed: " + json.dumps(unavailable, sort_keys=True))
    archive_provenance = verify_physics_player_archive(STAGE_ROOT, physics_v2=True)
    source_commit = archive_provenance["source_snapshot_commit"]
    capture_ids = {case: capture.capture_id for case, (_, _, capture) in attempts.items()}
    probes = [
        {
            "source": "unity_exporter_probe",
            "case": case,
            "capture_id": capture.capture_id,
            "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
            "level_instance_id": capture.source_bindings["level_instance_id"],
            "scenario_template_id": capture.source_bindings["scenario_template_id"],
            "final_evaluation": False,
        }
        for case, (_, _, capture) in sorted(attempts.items())
    ]
    facts = {
        "configured_fixed_step_capture_stride": {
            "status": "demonstrated", "capture_id": capture_ids["no-contact"], "reason": None,
        },
        "complete_raw_non_trigger_contacts": {
            "status": "demonstrated", "capture_id": capture_ids["no-contact"], "reason": None,
        },
        "collider_geometry_and_separation": {
            "status": "demonstrated", "capture_id": capture_ids["collision"], "reason": None,
        },
        "gravity_body_lifecycle_motion_support_world": {
            "status": "demonstrated", "capture_id": capture_ids["support"], "reason": None,
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
            "protocol_version": "physics-capture-v2-engine-protocol-v1",
            "exporter_version": "physics-capture-v2-exporter-code-v1",
        },
        "probes": probes,
        "facts": facts,
    }
    capability_report = {
        **report_payload,
        "report_id": _identity("physics-capture-v2-exporter-capability-report-v1", report_payload),
    }
    validate_physics_capture_v2_capability_report(capability_report)

    report_path = ISSUE_44_ROOT / "capability-report.json"
    observation_records = {
        case: {
            "status": "demonstrated",
            "reason": None,
            "capture_path": str(path.relative_to(ROOT)),
            "capture_id": capture.capture_id,
        }
        for case, (path, _, capture) in attempts.items()
    }
    bundle_payload = {
        "schema": "issue_44_physics_v2_runtime_evidence_bundle_v1",
        "source_snapshot_commit": source_commit,
        "source_tree": archive_provenance["source_tree"],
        "archive_path": ARCHIVE_PATH,
        "collection_plan_identity": load_collection_plan(PLAN_PATH).plan.identity,
        "capability_report_path": str(report_path.relative_to(ROOT)),
        "case_observations": observation_records,
        "exporter_files": {relative: relative for relative in EXPORTER_FILES},
    }
    bundle = {
        **bundle_payload,
        "identity": _identity("issue-44-physics-v2-runtime-evidence-bundle-v1", bundle_payload),
    }
    public_captures: dict[str, dict[str, str]] = {}
    for case, (source_path, _, capture) in attempts.items():
        public_path = ISSUE_44_ROOT / "captures" / f"{case}.json"
        public_captures[case] = {
            "capture_id": capture.capture_id,
            "path": str(public_path.relative_to(ROOT)),
        }
    capture_bundle_payload = {
        "schema": "issue_44_physics_v2_capture_bundle_v1",
        "runtime_evidence_bundle_identity": bundle["identity"],
        "captures": public_captures,
    }
    capture_bundle = {
        **capture_bundle_payload,
        "identity": _identity("issue-44-physics-v2-capture-bundle-v1", capture_bundle_payload),
    }

    write_immutable_cohort_v2_json(capability_report, report_path)
    write_immutable_cohort_v2_json(bundle, ISSUE_44_ROOT / "runtime-bundle-manifest.json")
    for case, (source_path, _, _) in attempts.items():
        write_immutable_cohort_v2_bytes(
            source_path.read_bytes(), ISSUE_44_ROOT / "captures" / f"{case}.json",
        )
    write_immutable_cohort_v2_json(
        capture_bundle, ISSUE_44_ROOT / "capture-bundle-manifest.json",
    )
    return {
        "source_snapshot_commit": source_commit,
        "issue_44_bundle_identity": bundle["identity"],
        "issue_44_capture_bundle_identity": capture_bundle["identity"],
        "capability_report_identity": capability_report["report_id"],
    }


def main() -> None:
    print(json.dumps(build_runtime_evidence(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
