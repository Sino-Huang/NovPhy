#!/usr/bin/env python3
"""Prepare and publish the immutable issue-48 cohort-v2 replay bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from scripts.build_issue_45_evidence import build_issue_45_evidence
from scripts.build_issue_46_evidence import validate_issue_46_evidence
from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_replay import (
    BUNDLE_NAME,
    CURRENT_DETERMINATION_VERSION,
    PLAN_NAME,
    PLAN_SCHEMA,
    REPORT_NAME,
    build_replay_report,
    build_issue_48_bundle,
    comparison_rules_for_plan_version,
    replay_scenario_collection_identity,
    replay_plan_identity,
    replay_intervention_identity,
    replay_version_envelope_identity,
    validate_issue_48_evidence,
    validate_replay_plan,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.verify_physics_player import verify_physics_player_archive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "sciencebirdsgames/observation-v1"
DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-48"


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"immutable issue-48 source already exists: {path}")
    path.write_bytes(content)


def _copy_file(source: Path, destination: Path) -> None:
    _write_bytes(destination, source.read_bytes())


def _version_envelope(
    stage: Path,
    repository_root: Path,
    *,
    require_clean_revision: bool,
) -> dict[str, Any]:
    verified = verify_physics_player_archive(
        stage,
        physics_v2=False,
        observation_v1=True,
    )
    if require_clean_revision:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status.strip():
            raise ValueError("issue-48 evidence requires a clean tracked worktree")
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    envelope = {
        "schema": "cohort_v2_replay_version_envelope_v1",
        "identity": "",
        "unity_version": verified["unity_version"],
        "player_stage_profile": "observation-v1",
        "player_source_snapshot_commit": verified["source_snapshot_commit"],
        "player_source_tree": verified["source_tree"],
        "player_declared_file_count": verified["declared_file_count"],
        "physics_protocol_version": 1,
        "observation_protocol_version": verified["protocol_version"],
        "physics_engine_contract": "physics_capture_v2_engine_v1",
        "physics_exporter_contract": "physics_capture_v2",
        "observation_exporter_contract": verified["capture_schema"],
        "observation_trace_contract": "observation_trace_manifest_v1",
        "generator_identity": "novphy-task-generator",
        "generator_version": "canonical_materialization_v1",
        "importer_identity": None,
        "importer_version": None,
        "code_revision": code_revision,
    }
    envelope["identity"] = replay_version_envelope_identity(envelope)
    return envelope


def _scenario_collection(
    role: str,
    wrapper: dict[str, Any],
    partition_entry: Any,
    observation_configuration: str,
    observation_configuration_identity: str,
    interface_action: dict[str, Any],
    engine_relative_action: dict[str, Any],
    coverage_strata: list[str],
) -> dict[str, Any]:
    scenario_collection_id = f"{role}-{'collision' if 'collision' in coverage_strata else 'stable'}"
    manifest = wrapper["scenario_manifest"]
    intervention = {
        "identity": "",
        "interface_action": interface_action,
        "engine_relative_action": engine_relative_action,
    }
    intervention["identity"] = replay_intervention_identity(
        scenario_collection_id,
        interface_action,
        engine_relative_action,
    )
    value = {
        "schema": "cohort_v2_replay_scenario_collection_v1",
        "identity": "",
        "scenario_collection_id": scenario_collection_id,
        "exposure_role": role,
        "scenario_manifest_identity": wrapper["identity"],
        "scenario_specification_identity": manifest["scenario_specification"]["identity"],
        "scenario_content_identity": manifest["scenario_specification"]["content_identity"],
        "scenario_template_identity": manifest["scenario_template"]["identity"],
        "level_instance_identity": manifest["level_instance"]["identity"],
        "scenario_lineage_identity": manifest["scenario_lineage"]["identity"],
        "source_manifest_relative_path": f"source/scenarios/{role}.json",
        "source_xml_relative_path": f"source/scenarios/{role}.xml",
        "source_template_relative_path": f"source/templates/{role}.xml",
        "generation": {
            "generator_identity": manifest["generation"]["generator_identity"],
            "generator_version": manifest["generation"]["generator_version"],
            "importer_identity": manifest["generation"]["importer_identity"],
            "importer_version": manifest["generation"]["importer_version"],
        },
        "intervention": intervention,
        "observation_configuration": observation_configuration,
        "observation_configuration_identity": observation_configuration_identity,
        "coverage_strata": coverage_strata,
    }
    partition_projection = {
        "scenario_manifest_identity": partition_entry.scenario_manifest_identity,
        "scenario_specification_identity": partition_entry.scenario_specification_identity,
        "scenario_template_identity": partition_entry.scenario_template_identity,
        "level_instance_identity": partition_entry.level_instance_identity,
        "scenario_lineage_identity": partition_entry.scenario_lineage_identity,
    }
    if any(value[field] != expected for field, expected in partition_projection.items()):
        raise ValueError(f"regenerated {role} source differs from the frozen issue-47 partition")
    value["identity"] = replay_scenario_collection_identity(value)
    return value


def prepare_issue_48_runtime_root(
    repository_root: Path,
    runtime_root: Path,
    *,
    stage: Path = DEFAULT_STAGE,
    determination_version: int = CURRENT_DETERMINATION_VERSION,
    require_clean_revision: bool = True,
) -> dict[str, Any]:
    """Freeze the two-scenario-collection replay plan and all source authorities."""
    repository_root = Path(repository_root).resolve()
    target = Path(runtime_root)
    if target.exists():
        raise ValueError("issue-48 runtime root already exists")
    target.mkdir(parents=True)
    try:
        partition_path = repository_root / "data/runtime_evidence/issue-47/partition-exposure-manifest.json"
        observation_root = repository_root / "data/runtime_evidence/issue-46"
        observation_bundle = validate_issue_46_evidence(observation_root)
        partition_value = json.loads(partition_path.read_text(encoding="utf-8"))
        partition = CohortV2PartitionExposureManifest.from_dict(partition_value)
        _copy_file(partition_path, target / "source/partition-exposure-manifest.json")
        _copy_file(
            observation_root / "observation-evidence-bundle.json",
            target / "source/observation-evidence-bundle.json",
        )
        with tempfile.TemporaryDirectory(prefix="novphy-issue48-sources-") as temporary:
            temporary_root = Path(temporary)
            public = temporary_root / "public"
            sealed = temporary_root / "sealed"
            build_issue_45_evidence(
                repository_root=repository_root,
                public_root=public,
                sealed_root=sealed,
            )
            wrappers = {}
            for role in ("training", "calibration"):
                manifest_path = public / "manifests" / f"{role}.json"
                xml_path = public / "xml" / f"{role}.xml"
                wrapper = json.loads(manifest_path.read_text(encoding="utf-8"))
                wrappers[role] = wrapper
                _copy_file(manifest_path, target / f"source/scenarios/{role}.json")
                _copy_file(xml_path, target / f"source/scenarios/{role}.xml")
                template_reference = wrapper["template_record"]["source_reference"]
                _copy_file(
                    repository_root / template_reference,
                    target / f"source/templates/{role}.xml",
                )
        observation_configuration_ids = {
            probe["exposure_role"]: probe["observation_configuration_identity"]
            for probe in observation_bundle["probes"]
        }
        partition_entries = {entry.exposure_role: entry for entry in partition.entries}
        scenario_collections = [
            _scenario_collection(
                "training",
                wrappers["training"],
                partition_entries["training"],
                "agent_rgb8_native_v1",
                observation_configuration_ids["training"],
                {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_release": [-77, 29],
                    "tapTime": 0,
                    "releaseTime": 1000,
                    "frame_height": 480,
                },
                {
                    "schema": "slingshot_relative_intervention_v1",
                    "drag_delta_canvas_pixels": [-77, 29],
                    "tap_time_milliseconds": 0,
                    "hold_milliseconds": 1000,
                },
                ["collision", "contact"],
            ),
            _scenario_collection(
                "calibration",
                wrappers["calibration"],
                partition_entries["calibration"],
                "agent_rgb8_nearest_320x240_v1",
                observation_configuration_ids["calibration"],
                {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_release": [-74, -31],
                    "tapTime": 0,
                    "releaseTime": 1000,
                    "frame_height": 480,
                },
                {
                    "schema": "slingshot_relative_intervention_v1",
                    "drag_delta_canvas_pixels": [-74, -31],
                    "tap_time_milliseconds": 0,
                    "hold_milliseconds": 1000,
                },
                ["stable"],
            ),
        ]
        envelope = _version_envelope(
            Path(stage),
            repository_root,
            require_clean_revision=require_clean_revision,
        )
        plan = {
            "schema": PLAN_SCHEMA,
            "identity": "",
            "plan_version": determination_version,
            "partition_manifest_identity": partition.identity,
            "observation_capability_bundle_identity": observation_bundle["identity"],
            "observation_access_audit_identity": observation_bundle["access_audit_identity"],
            "version_envelope": envelope,
            "comparison_rules": comparison_rules_for_plan_version(determination_version),
            "max_attempts_per_role": 1,
            "scenario_collections": scenario_collections,
        }
        plan["identity"] = replay_plan_identity(plan)
        write_immutable_cohort_v2_json(plan, target / PLAN_NAME)
        return validate_replay_plan(target)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def build_issue_48_evidence(runtime_root: Path, output_root: Path) -> dict[str, Any]:
    """Publish one immutable bundle from four independently retained attempts."""
    source = Path(runtime_root)
    validate_replay_plan(source)
    target = Path(output_root)
    if target.exists():
        raise ValueError("issue-48 evidence destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        report = build_replay_report(staging)
        write_immutable_cohort_v2_json(report, staging / REPORT_NAME)
        artifacts = sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name != BUNDLE_NAME
        )
        plan = validate_replay_plan(staging)
        bundle = build_issue_48_bundle(report, plan, artifacts)
        write_immutable_cohort_v2_json(bundle, staging / BUNDLE_NAME)
        validate_issue_48_evidence(staging)
        os.replace(staging, target)
        return bundle
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bundle = build_issue_48_evidence(args.runtime_root, args.output)
    print(json.dumps({"identity": bundle["identity"], "passed": bundle["passed"]}, sort_keys=True))
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
