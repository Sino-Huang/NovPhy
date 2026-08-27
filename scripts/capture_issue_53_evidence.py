#!/usr/bin/env python3
"""Run issue #53's frozen production plan and publish its immutable release."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from scripts.build_issue_45_evidence import (
    CONSTRAINTS_WORKBOOK_REFERENCE,
    ROLES,
    _materialize,
    _record_for_family,
)
from scripts.build_issue_52_evidence import validate_issue_52_evidence
from scripts.cohort_v2_partition import CohortV2PartitionExposureManifest
from scripts.cohort_v2_production_plans import (
    ROLE_ORDER,
)
from scripts.cohort_v2_release import (
    ReleaseContract,
    V1_CONTRACT,
    compare_production_replay,
    production_attempt_identity,
    production_intervention_identity,
    production_replay_report,
    planned_termination_for_assignment,
    publish_issue_53_evidence,
    replay_attempt_identity,
    release_contract_for_collection,
    validate_issue_53_execution_report,
    validate_published_issue_53_evidence,
)
from scripts.cohort_v2_scenarios import (
    load_cohort_v2_scenario_manifest,
    write_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)
from scripts.collect_rollouts import collect_fresh_engine_attempt
from scripts.final_evaluation_access import (
    FinalEvaluationWorkflowAccessManifest,
    audit_final_evaluation_workflow_access,
    authorize_final_evaluation_workflow_access,
)
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.smoke_physics_capture import archive_details, free_port, start_display, terminate
from scripts.verify_physics_player import verify_physics_player_archive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = ROOT / ".local-artifacts/issue-53-production-run"
DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-53"
DEFAULT_SEALED_OUTPUT = ROOT / ".local-artifacts/issue-53-final-release"
DEFAULT_PLAN_ROOT = ROOT / "data/runtime_evidence/issue-53-plan-v5"
DEFAULT_RUNTIME_ROOT_V2 = (
    ROOT / ".local-artifacts/issue-53-mixed-termination-production-run-v5"
)
DEFAULT_OUTPUT_V2 = ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5"
DEFAULT_SEALED_OUTPUT_V2 = (
    ROOT / ".local-artifacts/issue-53-mixed-termination-final-release-v5"
)
STAGE_ROOT = ROOT / "sciencebirdsgames/physics-v2"
PENDING_ACCESS_PATH = (
    ROOT
    / "data/runtime_evidence/issue-47/final-evaluation-workflow-access-manifest.json"
)
PARTITION_PATH = ROOT / "data/runtime_evidence/issue-47/partition-exposure-manifest.json"
OBSERVATION_CONFIGURATION = "agent_rgb8_native_v1"
REPLAY_INTERVENTION_ID = "central-collision"


@dataclass(frozen=True, slots=True)
class PlanSelection:
    root: Path
    collection_path: Path
    parameter_path: Path
    partition_path: Path
    pending_access_path: Path
    collection: Mapping[str, Any]
    parameters: Mapping[str, Any]
    final_role: Any
    contract: ReleaseContract


def _load_plan(plan_root: Path) -> PlanSelection:
    selected = Path(plan_root).resolve()
    collection_path = selected / "collection-plan.json"
    parameter_path = selected / "production-parameter-plan.json"
    collection = _load_object(collection_path)
    parameters = _load_object(parameter_path)
    contract = release_contract_for_collection(collection)
    if parameters.get("identity") != contract.parameter_identity:
        raise ValueError("selected collection and parameter plan identities differ")
    if contract == V1_CONTRACT:
        validate_issue_52_evidence(selected, revalidate_pilot=False)
        partition_path = PARTITION_PATH
        access_path = PENDING_ACCESS_PATH
        final_role = ROLES[3]
    elif contract.version == 2:
        from scripts.cohort_v2_production_plans_v2 import (
            FINAL_SEED,
            validate_plan_v2_evidence,
        )

        validate_plan_v2_evidence(selected)
        partition_path = selected / "partition-exposure-manifest.json"
        access_path = selected / "final-evaluation-workflow-access-manifest.json"
        final_role = replace(ROLES[3], seed=FINAL_SEED)
    elif contract.version == 3:
        from scripts.cohort_v2_production_plans_v2 import FINAL_SEED
        from scripts.cohort_v2_production_plans_v3 import validate_plan_v3_evidence

        validate_plan_v3_evidence(selected)
        partition_path = selected / "partition-exposure-manifest.json"
        access_path = selected / "final-evaluation-workflow-access-manifest.json"
        final_role = replace(ROLES[3], seed=FINAL_SEED)
    elif contract.version == 4:
        from scripts.cohort_v2_production_plans_v4 import (
            FINAL_SEED,
            validate_plan_v4_evidence,
        )

        validate_plan_v4_evidence(selected)
        partition_path = selected / "partition-exposure-manifest.json"
        access_path = selected / "final-evaluation-workflow-access-manifest.json"
        final_role = replace(ROLES[3], seed=FINAL_SEED)
    else:
        from scripts.cohort_v2_production_plans_v4 import FINAL_SEED
        from scripts.cohort_v2_production_plans_v5 import validate_plan_v5_evidence

        validate_plan_v5_evidence(selected)
        partition_path = selected / "partition-exposure-manifest.json"
        access_path = selected / "final-evaluation-workflow-access-manifest.json"
        final_role = replace(ROLES[3], seed=FINAL_SEED)
    return PlanSelection(
        selected,
        collection_path,
        parameter_path,
        partition_path,
        access_path,
        collection,
        parameters,
        final_role,
        contract,
    )


def _actual_command(plan: PlanSelection) -> str:
    if plan.contract.version == 1:
        return (
            "python -u -m scripts.capture_issue_53_evidence "
            "--plan-root data/runtime_evidence/issue-52 "
            "--runtime-root .local-artifacts/issue-53-production-run "
            "--output data/runtime_evidence/issue-53 "
            "--sealed-output .local-artifacts/issue-53-final-release "
            "--authorization-identity github-issue-authorization-v1:53:production"
        )
    if plan.contract.version == 2:
        return (
            "python -u -m scripts.capture_issue_53_evidence "
            "--plan-root data/runtime_evidence/issue-53-plan-v2 "
            "--runtime-root .local-artifacts/issue-53-stable-only-production-run-v2 "
            "--output data/runtime_evidence/issue-53-stable-only-v2 "
            "--sealed-output .local-artifacts/issue-53-stable-only-final-release-v2 "
            "--authorization-identity "
            "github-issue-authorization-v2:53:stable-only-production"
        )
    if plan.contract.version == 3:
        return (
            "python -u -m scripts.capture_issue_53_evidence "
            "--plan-root data/runtime_evidence/issue-53-plan-v3 "
            "--runtime-root .local-artifacts/issue-53-stable-only-production-run-v3 "
            "--output data/runtime_evidence/issue-53-stable-only-v3 "
            "--sealed-output .local-artifacts/issue-53-stable-only-final-release-v3 "
            "--authorization-identity "
            "github-issue-authorization-v3:53:stable-only-production-after-anchor-fix"
        )
    if plan.contract.version == 4:
        return (
            "python -u -m scripts.capture_issue_53_evidence "
            "--plan-root data/runtime_evidence/issue-53-plan-v4 "
            "--runtime-root .local-artifacts/issue-53-mixed-termination-production-run-v4 "
            "--output data/runtime_evidence/issue-53-mixed-termination-v4 "
            "--sealed-output .local-artifacts/issue-53-mixed-termination-final-release-v4 "
            "--authorization-identity "
            "github-issue-authorization-v4:53:mixed-termination-production"
        )
    return (
        "python -u -m scripts.capture_issue_53_evidence "
        "--plan-root data/runtime_evidence/issue-53-plan-v5 "
        "--runtime-root .local-artifacts/issue-53-mixed-termination-production-run-v5 "
        "--output data/runtime_evidence/issue-53-mixed-termination-v5 "
        "--sealed-output .local-artifacts/issue-53-mixed-termination-final-release-v5 "
        "--authorization-identity "
        "github-issue-authorization-v5:53:mixed-termination-production"
    )


def _log(message: str) -> None:
    print(f"[issue-53] {message}", flush=True)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _accepted_player(stage_root: Path = STAGE_ROOT) -> dict[str, Any]:
    provenance = verify_physics_player_archive(stage_root, physics_v2=True)
    pilot = _load_object(
        ROOT / "data/runtime_evidence/issue-51/representative-cohort-v2-pilot-report.json"
    )
    expected = pilot["code"]["supplementary_source_snapshot_commit"]
    if (
        provenance.get("source_snapshot_commit") != expected
        or provenance.get("unity_version") != pilot["environment"]["unity_version"]
        or provenance.get("capture_schema") != "physics_capture_v2_engine_v1"
    ):
        raise ValueError("physics-v2 player differs from the accepted issue-51 envelope")
    return provenance


def _assignments(collection: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    assignments = {
        item["exposure_role"]: item for item in collection["assignments"]
    }
    if tuple(assignments) != ROLE_ORDER:
        raise ValueError("selected production roles are not in frozen order")
    return assignments


def _materialize_authority(
    role: Any,
    output_root: Path,
) -> dict[str, Any]:
    workbook_path = ROOT / CONSTRAINTS_WORKBOOK_REFERENCE
    source_path, constraints, record = _record_for_family(
        ROOT,
        workbook_path.read_bytes(),
        role.family,
    )
    result, scenario = _materialize(
        role,
        source_path,
        constraints,
        record,
        workbook_path,
        output_root,
    )
    xml_path = output_root / "xml" / f"{role.name.replace('_', '-')}.xml"
    manifest_path = output_root / "manifests" / f"{role.name.replace('_', '-')}.json"
    write_immutable_cohort_v2_bytes(result.xml_content, xml_path)
    write_cohort_v2_scenario_manifest(scenario, manifest_path)
    validated = load_cohort_v2_scenario_manifest(
        manifest_path,
        xml_path=xml_path,
        template_source_path=source_path,
    )
    return {
        "role": role.name,
        "scenario": validated,
        "manifest_path": manifest_path,
        "xml_path": xml_path,
        "template_path": source_path,
        "workbook_path": workbook_path,
    }


def _validate_authority(
    authority: Mapping[str, Any], assignment: Mapping[str, Any]
) -> None:
    scenario = authority["scenario"]
    manifest = scenario.scenario_manifest
    observed = {
        "scenario_manifest_identity": scenario.identity,
        "scenario_lineage_identity": manifest.scenario_lineage.identity,
        "level_instance_identity": manifest.level_instance.identity,
        "scenario_template_identity": manifest.scenario_template.identity,
        "benchmark_condition_identity": manifest.benchmark_condition.identity,
    }
    if any(assignment[field] != value for field, value in observed.items()):
        raise ValueError(f"{authority['role']} authority differs from the selected plan")


def _prepare_non_final_authorities(
    output_root: Path, assignments: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    authorities = {}
    for role in ROLES[:3]:
        authority = _materialize_authority(role, output_root)
        _validate_authority(authority, assignments[role.name])
        authorities[role.name] = authority
    return authorities


def _pending_access(plan: PlanSelection) -> tuple[
    CohortV2PartitionExposureManifest, FinalEvaluationWorkflowAccessManifest
]:
    partition = CohortV2PartitionExposureManifest.from_dict(
        _load_object(plan.partition_path)
    )
    access = FinalEvaluationWorkflowAccessManifest.from_dict(
        _load_object(plan.pending_access_path)
    )
    if access.authorization_state != "pending":
        raise ValueError("the frozen issue-47 workflow is not in its pending state")
    audit_final_evaluation_workflow_access(partition, access, observed_accesses=[])
    return partition, access


def _authorize_and_materialize_final(
    output_root: Path,
    assignment: Mapping[str, Any],
    authorization_identity: str,
    plan: PlanSelection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    partition, pending = _pending_access(plan)
    authorized_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    authorized = authorize_final_evaluation_workflow_access(
        pending,
        authorization_identity=authorization_identity,
        authorized_at=authorized_at,
    )
    observed = {
        "workflow_identity": authorized.workflow_identity,
        "operator_identity": authorized.operator_identity,
        "artifact_identity": assignment["scenario_manifest_identity"],
        "source_scenario_lineage_identities": [
            assignment["scenario_lineage_identity"]
        ],
        "accessed_at": authorized_at,
        "authorization_identity": authorization_identity,
        "consumer_exposure_role": "final_evaluation",
    }
    access_audit = audit_final_evaluation_workflow_access(
        partition,
        authorized,
        observed_accesses=[observed],
    )
    write_immutable_cohort_v2_json(
        authorized.to_dict(), output_root / "authorized-final-access-manifest.json"
    )
    authority = _materialize_authority(plan.final_role, output_root)
    _validate_authority(authority, assignment)
    return authority, access_audit


def _install_level(game: Path, xml_path: Path, name: str) -> None:
    target_root = game / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"issue-53-{name}.xml"
    shutil.copyfile(xml_path, target)
    evaluation = ET.Element("evaluation")
    ET.SubElement(
        evaluation,
        "novelty_detection_measurement",
        {"step": "1", "measure_in_training": "False", "measure_in_testing": "False"},
    )
    trials = ET.SubElement(evaluation, "trials")
    trial = ET.SubElement(
        trials,
        "trial",
        {
            "id": "0",
            "number_of_executions": "1",
            "checkpoint_time_limit": "9999999",
            "checkpoint_interaction_limit": "9999999",
            "notify_novelty": "False",
        },
    )
    level_set = ET.SubElement(
        trial,
        "game_level_set",
        {
            "mode": "training",
            "time_limit": "9999999",
            "total_interaction_limit": "9999999",
            "attempt_limit_per_level": "1",
            "allow_level_selection": "True",
        },
    )
    ET.SubElement(
        level_set,
        "game_levels",
        {"level_path": target.relative_to(game).as_posix()},
    )
    ET.indent(evaluation, space="  ")
    ET.ElementTree(evaluation).write(
        game / "config.xml", encoding="utf-8", xml_declaration=True
    )


def _attempt_options(
    authority: Mapping[str, Any],
    assignment: Mapping[str, Any],
    intervention: Mapping[str, Any],
    attempt_id: str,
    game: Path,
    contract: ReleaseContract,
    *,
    anchor_actions: bool,
    aligned_observation_capture: bool = False,
) -> dict[str, Any]:
    scenario = authority["scenario"]
    agent_port = free_port()
    game_port = free_port()
    physics_port = free_port()
    os.environ["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
    return {
        "game_dir": game,
        "host": "127.0.0.1",
        "port": agent_port,
        "physics_host": "127.0.0.1",
        "physics_port": physics_port,
        "agent_id": 28853,
        "speed": 50,
        "connect_timeout": 45.0,
        "read_timeout": 45.0,
        "prepare_timeout": 45.0,
        "frame_height": 480,
        "fast": True,
        "headless": False,
        "target_fps": 1.0,
        "duration_seconds": 1.0,
        "ui_level": None,
        "ui_settle_seconds": 0.0,
        "engine_settle_seconds": 1.0,
        "agent_settle_seconds": 1.0,
        "engine_agent_port": agent_port,
        "engine_game_port": game_port,
        "shoot_before_capture": False,
        "anchor_actions": anchor_actions,
        "physics_capture_v2": True,
        "scenario_manifest": scenario.scenario_manifest,
        "scenario_context_override": {
            "cohort_v2_production_collection_plan_identity": contract.collection_identity,
            "cohort_v2_production_parameter_plan_identity": contract.parameter_identity,
            "exposure_role": assignment["exposure_role"],
            "intended_coverage_stratum": intervention[
                "intended_coverage_stratum"
            ],
            "intended_termination_class": planned_termination_for_assignment(
                assignment, intervention
            ),
        },
        "physics_v2_source_bindings": {
            "scenario_template_id": assignment["scenario_template_identity"],
            "level_instance_id": assignment["level_instance_identity"],
            "scenario_lineage_id": assignment["scenario_lineage_identity"],
            "rollout_id": attempt_id,
            "intervention_id": production_intervention_identity(
                intervention["id"], contract.collection_identity
            ),
        },
        "physics_v2_scenario_manifest_identity": assignment[
            "scenario_manifest_identity"
        ],
        "observation_configuration": OBSERVATION_CONFIGURATION,
        "observation_exposure_role": assignment["exposure_role"],
        "aligned_observation_capture": aligned_observation_capture,
    }


def _capture_attempt(
    runtime_root: Path,
    output_root: Path,
    authority: Mapping[str, Any],
    assignment: Mapping[str, Any],
    intervention: Mapping[str, Any],
    attempt_id: str,
    action: Mapping[str, Any],
    contract: ReleaseContract,
    *,
    replay: bool = False,
    aligned_observation_capture: bool = False,
    stage_root: Path = STAGE_ROOT,
) -> dict[str, Any]:
    game = runtime_root / "games" / ("replay" if replay else "production") / attempt_id
    _log(f"{attempt_id}: unpacking the accepted physics-v2 player")
    archive_details(stage_root, game)
    _install_level(game, authority["xml_path"], attempt_id)
    result = collect_fresh_engine_attempt(
        output_root,
        dict(action),
        attempt_id=attempt_id,
        attempt_number=1,
        expected_initial_engine_state_identity=authority[
            "scenario"
        ].scenario_manifest.declared_initial_engine_state.identity,
        **_attempt_options(
            authority,
            assignment,
            intervention,
            attempt_id,
            game,
            contract,
            anchor_actions=not replay,
            aligned_observation_capture=aligned_observation_capture,
        ),
    )
    return result


def _terminal_details(result: Mapping[str, Any]) -> tuple[str | None, int | None]:
    if result["status"] != "accepted":
        return None, None
    capture = load_physics_capture_v2(
        Path(result["artifact_path"]) / "physics_capture_v2.json"
    )
    terminal = capture.record["terminal_evidence"]
    span = terminal["fixed_step"] - capture.record["pre_intervention_fixed_step"]
    if span > 600:
        raise ValueError(f"rollout exceeded the frozen 600-fixed-step ceiling: {span}")
    if terminal["reason"] not in {
        "level_clear",
        "level_fail",
        "stable_entered",
        "rollout_ceiling",
    }:
        raise ValueError(f"unknown production termination: {terminal['reason']}")
    return terminal["reason"], span


def _report_entry(
    assignment: Mapping[str, Any],
    intervention: Mapping[str, Any],
    attempt_id: str,
    result: Mapping[str, Any],
    contract: ReleaseContract,
) -> dict[str, Any]:
    terminal_reason = None
    terminal_span = None
    if result["status"] == "accepted":
        terminal_reason, terminal_span = _terminal_details(result)
    return {
        "attempt_id": attempt_id,
        "exposure_role": assignment["exposure_role"],
        "dataset_partition": assignment["dataset_partition"],
        "scenario_manifest_identity": assignment["scenario_manifest_identity"],
        "scenario_lineage_identity": assignment["scenario_lineage_identity"],
        "level_instance_identity": assignment["level_instance_identity"],
        "scenario_template_identity": assignment["scenario_template_identity"],
        "benchmark_condition_identity": assignment[
            "benchmark_condition_identity"
        ],
        "intervention_id": intervention["id"],
        "intervention_identity": production_intervention_identity(
            intervention["id"], contract.collection_identity
        ),
        "intervention_ordinal": intervention["ordinal"],
        "intervention_source": intervention["intervention_source"],
        "intended_coverage_stratum": intervention["intended_coverage_stratum"],
        "expected_termination": planned_termination_for_assignment(
            assignment, intervention
        ),
        "status": result["status"],
        "reason": result.get("reason"),
        "failure_code": result.get("failure_code"),
        "artifact_path": result.get("artifact_path"),
        "quarantine_path": result.get("quarantine_path"),
        "failure_manifest_path": result.get("failure_manifest_path"),
        "realized_coverage_strata": result.get("realized_coverage_strata", []),
        "terminal_reason": terminal_reason,
        "terminal_span_fixed_steps": terminal_span,
        "attempt_number": 1,
        "retry_decision": "none",
    }


def _execution_report(
    ledger: list[dict[str, Any]], contract: ReleaseContract
) -> dict[str, Any]:
    counts = Counter(item["status"] for item in ledger)
    return {
        "schema": contract.schema("issue_53_production_execution_report"),
        "collection_plan_identity": contract.collection_identity,
        "production_parameter_plan_identity": contract.parameter_identity,
        "attempt_ledger": ledger,
        "counts": {
            "planned": 24,
            "attempted": len(ledger),
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "failed": counts["failed"],
            "quarantined": counts["rejected"] + counts["failed"],
        },
        "retry_count": 0,
        "outcome_independent_accounting": True,
    }


def _write_checkpoint(
    runtime_root: Path,
    ledger: list[dict[str, Any]],
    contract: ReleaseContract,
) -> None:
    path = runtime_root / "production-execution-report.json"
    value = _execution_report(ledger, contract)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dry_run(plan_root: Path = DEFAULT_PLAN_ROOT) -> dict[str, Any]:
    _log("dry-run: revalidating the selected immutable production plans")
    plan = _load_plan(plan_root)
    collection = plan.collection
    assignments = _assignments(collection)
    _log("dry-run: materializing only the three non-final scenario authorities")
    with tempfile.TemporaryDirectory(prefix="novphy-issue53-dry-run-") as temporary:
        authorities = _prepare_non_final_authorities(
            Path(temporary) / "authorities", assignments
        )
        if tuple(authorities) != ROLE_ORDER[:3]:
            raise ValueError("issue-53 dry-run non-final authority set is incomplete")
    _log("dry-run: validating the sealed projection without opening final data")
    partition, pending = _pending_access(plan)
    final_assignment = assignments["final_evaluation"]
    final_entry = next(
        entry
        for entry in partition.entries
        if entry.exposure_role == "final_evaluation"
    )
    if (
        final_entry.scenario_manifest_identity
        != final_assignment["scenario_manifest_identity"]
        or pending.authorized_artifacts[0].artifact_identity
        != final_assignment["scenario_manifest_identity"]
    ):
        raise ValueError("issue-53 final sealed projection is stale")
    provenance = _accepted_player()
    slots = [
        production_attempt_identity(
            assignment["exposure_role"],
            intervention_id,
            plan.contract.collection_identity,
        )
        for assignment in collection["assignments"]
        for intervention_id in assignment["intervention_ids"]
    ]
    if len(slots) != 24 or len(set(slots)) != 24:
        raise ValueError("issue-53 planned attempt identities are not exact and unique")
    _log("dry-run complete: 24 production attempts and four replay proofs are wired")
    return {
        "schema": plan.contract.schema("issue_53_production_dry_run"),
        "collection_plan_identity": plan.contract.collection_identity,
        "production_parameter_plan_identity": plan.contract.parameter_identity,
        "planned_rollouts": 24,
        "planned_replay_proofs": 4,
        "non_final_authority_count": 3,
        "final_access_state": pending.authorization_state,
        "final_data_opened": False,
        "player_source_snapshot_commit": provenance["source_snapshot_commit"],
        "unity_version": provenance["unity_version"],
        "actual_command": _actual_command(plan),
        "plan_root": str(plan.root),
        "files_written": False,
        "passed": True,
    }


def run(
    runtime_root: Path,
    output: Path,
    sealed_output: Path,
    authorization_identity: str,
    plan_root: Path = DEFAULT_PLAN_ROOT,
) -> dict[str, Any]:
    runtime_root = Path(runtime_root).resolve()
    if runtime_root.exists() or Path(output).exists() or Path(sealed_output).exists():
        raise ValueError("issue-53 runtime or immutable output already exists")
    if not authorization_identity:
        raise ValueError("issue-53 actual production requires an authorization identity")
    runtime_root.mkdir(parents=True)
    _log("revalidating the selected immutable production plans")
    plan = _load_plan(plan_root)
    collection = plan.collection
    shutil.copyfile(
        plan.collection_path,
        runtime_root / "frozen-collection-plan.json",
    )
    shutil.copyfile(
        plan.parameter_path,
        runtime_root / "frozen-production-parameter-plan.json",
    )
    assignments = _assignments(collection)
    _log("verifying the accepted production player")
    player_provenance = _accepted_player()
    write_immutable_cohort_v2_json(
        player_provenance, runtime_root / "player-provenance.json"
    )
    authorities_root = runtime_root / "authorities"
    authorities = _prepare_non_final_authorities(authorities_root, assignments)
    _log("authorizing the frozen final-evaluation workflow")
    final_authority, access_audit = _authorize_and_materialize_final(
        authorities_root,
        assignments["final_evaluation"],
        authorization_identity,
        plan,
    )
    authorities["final_evaluation"] = final_authority
    write_immutable_cohort_v2_json(
        access_audit, runtime_root / "final-access-audit.json"
    )

    display_process = None
    old_display = os.environ.get("DISPLAY")
    old_stride = os.environ.get("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE")
    ledger = []
    try:
        display, display_process = start_display(runtime_root / "display.log")
        os.environ["DISPLAY"] = display
        os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        total = 24
        for assignment in collection["assignments"]:
            role = assignment["exposure_role"]
            for intervention_id in assignment["intervention_ids"]:
                intervention = next(
                    item
                    for item in collection["interventions"]
                    if item["id"] == intervention_id
                )
                attempt_id = production_attempt_identity(
                    role, intervention_id, plan.contract.collection_identity
                )
                _log(
                    f"production {len(ledger) + 1}/{total}: {role} / {intervention_id}"
                )
                result = _capture_attempt(
                    runtime_root,
                    runtime_root / "production",
                    authorities[role],
                    assignment,
                    intervention,
                    attempt_id,
                    intervention["interface_action"],
                    plan.contract,
                )
                try:
                    entry = _report_entry(
                        assignment, intervention, attempt_id, result, plan.contract
                    )
                except Exception as error:
                    artifact_value = result.get("artifact_path")
                    if not isinstance(artifact_value, str) or not artifact_value:
                        raise
                    accepted_artifact = Path(artifact_value).resolve()
                    accepted_attempt_root = accepted_artifact.parent
                    expected_attempt_root = (
                        runtime_root / "production/accepted" / attempt_id
                    ).resolve()
                    if accepted_attempt_root != expected_attempt_root:
                        raise ValueError(
                            "post-validation artifact is outside its accepted attempt path"
                        ) from error
                    quarantine_root = (
                        runtime_root / "production/quarantine" / attempt_id
                    )
                    if accepted_attempt_root.is_dir():
                        quarantine_root.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(accepted_attempt_root, quarantine_root)
                    failure_path = quarantine_root / "failure.json"
                    failure_path.write_text(
                        json.dumps(
                            {
                                "schema": "collection_attempt_failure_v1",
                                "attempt_id": attempt_id,
                                "attempt_number": 1,
                                "status": "rejected",
                                "reason": str(error),
                                "failure_code": "post_capture_validation_failed",
                                "failure_class": "permanent",
                                "retryable": False,
                                "retry_decision": "stop",
                                "quarantine_path": str(quarantine_root),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    entry = _report_entry(
                        assignment,
                        intervention,
                        attempt_id,
                        {
                            **result,
                            "status": "rejected",
                            "reason": str(error),
                            "failure_code": "post_capture_validation_failed",
                            "artifact_path": None,
                            "quarantine_path": str(quarantine_root),
                            "failure_manifest_path": str(failure_path),
                            "realized_coverage_strata": [],
                        },
                        plan.contract,
                    )
                ledger.append(entry)
                _write_checkpoint(runtime_root, ledger, plan.contract)
                _log(
                    f"production {len(ledger)}/{total}: {entry['status']} "
                    f"termination={entry['terminal_reason']}"
                )
    finally:
        terminate(display_process)
        if old_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old_display
        if old_stride is None:
            os.environ.pop("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", None)
        else:
            os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = old_stride

    _write_checkpoint(runtime_root, ledger, plan.contract)
    production_complete = all(
        item["status"] == "accepted"
        and item["intended_coverage_stratum"]
        in item["realized_coverage_strata"]
        and item["terminal_reason"] == item["expected_termination"]
        for item in ledger
    )
    verdicts = []
    if production_complete:
        _log("production quotas passed; starting four exact-socket replay proofs")
        display_process = None
        old_display = os.environ.get("DISPLAY")
        old_stride = os.environ.get("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE")
        try:
            display, display_process = start_display(runtime_root / "replay-display.log")
            os.environ["DISPLAY"] = display
            os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
            for index, role in enumerate(ROLE_ORDER, start=1):
                entry = next(
                    item
                    for item in ledger
                    if item["exposure_role"] == role
                    and item["intervention_id"] == REPLAY_INTERVENTION_ID
                )
                assignment = assignments[role]
                intervention = next(
                    item
                    for item in collection["interventions"]
                    if item["id"] == REPLAY_INTERVENTION_ID
                )
                original_artifact = Path(entry["artifact_path"])
                original_manifest = _load_object(original_artifact.parent / "manifest.json")
                action = original_manifest["rollouts"][0]["action"]
                replay_id = replay_attempt_identity(
                    role,
                    REPLAY_INTERVENTION_ID,
                    plan.contract.collection_identity,
                )
                _log(f"replay {index}/4: {role}")
                result = _capture_attempt(
                    runtime_root,
                    runtime_root / "replay",
                    authorities[role],
                    assignment,
                    intervention,
                    replay_id,
                    action,
                    plan.contract,
                    replay=True,
                )
                if result["status"] != "accepted":
                    verdicts.append(
                        {
                            "schema": plan.contract.schema(
                                "cohort_v2_production_replay_verdict"
                            ),
                            "original_attempt_identity": entry["attempt_id"],
                            "replay_attempt_identity": replay_id,
                            "exposure_role": role,
                            "components": [
                                {
                                    "component": "artifact_availability_and_binding",
                                    "status": "unavailable",
                                    "details": {"reason": result.get("reason")},
                                }
                            ],
                            "passed": False,
                        }
                    )
                    continue
                try:
                    _terminal_details(result)
                    replay_artifact = Path(result["artifact_path"])
                    replay_manifest = _load_object(
                        replay_artifact.parent / "manifest.json"
                    )
                    replay_action = replay_manifest["rollouts"][0]["action"]
                    verdict = compare_production_replay(
                        original_artifact,
                        replay_artifact,
                        original_attempt_id=entry["attempt_id"],
                        replay_attempt_id=replay_id,
                        original_action=action,
                        replay_action=replay_action,
                        exposure_role=role,
                        contract=plan.contract,
                    )
                except Exception as error:
                    verdict = {
                        "schema": plan.contract.schema(
                            "cohort_v2_production_replay_verdict"
                        ),
                        "original_attempt_identity": entry["attempt_id"],
                        "replay_attempt_identity": replay_id,
                        "exposure_role": role,
                        "components": [
                            {
                                "component": "post_capture_validation",
                                "status": "unavailable",
                                "details": {"reason": str(error)},
                            }
                        ],
                        "passed": False,
                    }
                verdicts.append(verdict)
                _log(f"replay {index}/4: passed={verdicts[-1]['passed']}")
        finally:
            terminate(display_process)
            if old_display is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = old_display
            if old_stride is None:
                os.environ.pop("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", None)
            else:
                os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = old_stride

    if len(verdicts) == 4 and all(item["passed"] for item in verdicts):
        replay_report = production_replay_report(verdicts, contract=plan.contract)
    else:
        replay_report = {
            "schema": plan.contract.schema("cohort_v2_production_replay_report"),
            "identity": (
                f"cohort-v2-production-replay-report-v{plan.contract.version}:incomplete"
            ),
            "collection_plan_identity": plan.contract.collection_identity,
            "comparison_rules_identity": None,
            "proof_count": sum(item.get("passed") is True for item in verdicts),
            "retry_count": 0,
            "verdicts": verdicts,
            "passed": False,
        }
    write_immutable_cohort_v2_json(
        replay_report, runtime_root / "production-replay-report.json"
    )
    if (
        (runtime_root / "frozen-collection-plan.json").read_bytes()
        != (
            plan.collection_path
        ).read_bytes()
        or (runtime_root / "frozen-production-parameter-plan.json").read_bytes()
        != (
            plan.parameter_path
        ).read_bytes()
    ):
        raise ValueError("selected frozen plan bytes changed during production")
    _log("publishing immutable issue-53 accounting and release artifacts")
    return publish_issue_53_evidence(
        repository_root=ROOT,
        runtime_root=runtime_root,
        output=output,
        sealed_output=sealed_output,
        final_access_audit=access_audit,
        plan_root=plan.root,
    )


def validate_existing(
    *,
    plan_root: Path,
    runtime_root: Path,
    output: Path,
    sealed_output: Path,
) -> dict[str, Any]:
    """Revalidate an existing result without launching or opening Unity."""
    plan = _load_plan(plan_root)
    runtime = Path(runtime_root).resolve()
    if (
        (runtime / "frozen-collection-plan.json").read_bytes()
        != plan.collection_path.read_bytes()
        or (runtime / "frozen-production-parameter-plan.json").read_bytes()
        != plan.parameter_path.read_bytes()
    ):
        raise ValueError("existing runtime is not bound to the selected exact plan bytes")
    report = validate_issue_53_execution_report(
        _load_object(runtime / "production-execution-report.json"),
        plan.collection,
    )
    replay = _load_object(runtime / "production-replay-report.json")
    if (
        replay.get("schema")
        != plan.contract.schema("cohort_v2_production_replay_report")
        or replay.get("collection_plan_identity")
        != plan.contract.collection_identity
    ):
        raise ValueError("existing replay report is not bound to the selected plan")
    published = validate_published_issue_53_evidence(output, sealed_output)
    return {
        "schema": plan.contract.schema("issue_53_existing_result_validation"),
        "collection_plan_identity": plan.contract.collection_identity,
        "attempted_rollouts": report["counts"]["attempted"],
        "accepted_rollouts": report["counts"]["accepted"],
        "replay_proofs": replay.get("proof_count", 0),
        "release_disposition": published["disposition"],
        "passed": published["passed"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and publish issue #53's representative cohort-v2 release"
    )
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_V2)
    parser.add_argument("--sealed-output", type=Path, default=DEFAULT_SEALED_OUTPUT_V2)
    parser.add_argument("--authorization-identity")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        result = dry_run(args.plan_root)
    elif args.validate:
        result = validate_existing(
            plan_root=args.plan_root,
            runtime_root=args.runtime_root,
            output=args.output,
            sealed_output=args.sealed_output,
        )
    else:
        result = run(
            args.runtime_root,
            args.output,
            args.sealed_output,
            args.authorization_identity or "",
            args.plan_root,
        )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
