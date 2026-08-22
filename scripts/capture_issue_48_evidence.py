#!/usr/bin/env python3
"""Collect two original/replay pairs and publish issue-48 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from scripts.build_issue_48_evidence import (
    DEFAULT_OUTPUT,
    DEFAULT_STAGE,
    build_issue_48_evidence,
    prepare_issue_48_runtime_root,
)
from scripts.cohort_v2_replay import (
    ATTEMPT_NAME,
    ATTEMPT_SCHEMA,
    FROZEN_COMMAND_NAME,
    build_frozen_replay_command,
    semantic_identity,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.collect_rollouts import (
    action_to_shot,
    anchor_action_to_slingshot_reference,
    capture_physics_v2_rollout,
    current_slingshot_reference,
)
from scripts.manual_agent import connect_with_retry, prepare_for_play
from scripts.observation_trace import persist_observation_trace
from scripts.smoke_physics_capture import (
    archive_details,
    free_port,
    launch_environment,
    start_display,
    terminate,
)
from src.webui.bridge import ObservationCaptureEngine, PlayingMode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = ROOT / ".local-artifacts/issue-48-replay-run"


def _progress(message: str) -> None:
    print(f"[issue-48] {message}", flush=True)


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _capture_record(capture: ObservationCaptureEngine) -> dict:
    record = _plain(capture.metadata)
    record["canonical_png"] = capture.canonical_png
    return record


def _install_level(runtime: Path, xml_path: Path) -> None:
    target_root = (
        runtime / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels"
    )
    target_root.mkdir(parents=True, exist_ok=True)
    level_target = target_root / "issue-48.xml"
    shutil.copyfile(xml_path, level_target)

    evaluation = ET.Element("evaluation")
    ET.SubElement(
        evaluation,
        "novelty_detection_measurement",
        {"step": "1", "measure_in_training": "False", "measure_in_testing": "False"},
    )
    trials = ET.SubElement(evaluation, "trials")
    trial = ET.SubElement(trials, "trial", {
        "id": "0", "number_of_executions": "1",
        "checkpoint_time_limit": "9999999",
        "checkpoint_interaction_limit": "9999999",
        "notify_novelty": "False",
    })
    level_set = ET.SubElement(trial, "game_level_set", {
        "mode": "training", "time_limit": "9999999",
        "total_interaction_limit": "9999999",
        "attempt_limit_per_level": "5", "allow_level_selection": "True",
    })
    ET.SubElement(
        level_set,
        "game_levels",
        {"level_path": level_target.relative_to(runtime).as_posix()},
    )
    ET.indent(evaluation, space="  ")
    ET.ElementTree(evaluation).write(
        runtime / "config.xml", encoding="utf-8", xml_declaration=True
    )


def _stable_slingshot(agent, *, timeout: float = 60.0) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        current = current_slingshot_reference(agent, 480)
        if current is not None and current == previous:
            return current
        previous = current
        time.sleep(0.25)
    raise TimeoutError("Science Birds slingshot did not settle before the replay attempt")


def _stable_observation(physics, *, timeout: float = 60.0) -> ObservationCaptureEngine:
    deadline = time.monotonic() + timeout
    previous = None
    stable_count = 0
    while time.monotonic() < deadline:
        capture = physics.get_observation_capture()
        metadata = _plain(capture.metadata)
        camera_state = {
            "camera": metadata["camera"],
            "viewport": metadata["viewport"],
            "coordinates": metadata["coordinates"],
            "world_to_observation_transform": metadata["world_to_observation_transform"],
        }
        stable_count = stable_count + 1 if camera_state == previous else 1
        if stable_count >= 4:
            return capture
        previous = camera_state
        time.sleep(0.25)
    raise TimeoutError("Science Birds camera did not settle before the replay observation")


def _physics_metadata(metadata: dict) -> dict:
    fields = (
        "physics_capture_v2_schema", "capture_id", "shot_id",
        "configured_fixed_step_capture_stride", "causal_entity_count", "collider_count",
        "fixed_step_sample_count", "frame_record_count", "event_count",
        "initial_engine_state_identity", "scenario_manifest_identity",
    )
    return {field: metadata[field] for field in fields}


def _capture_attempt(
    stage: Path,
    runtime_root: Path,
    plan: dict,
    scenario_collection: dict,
    role: str,
    *,
    frozen_command: dict | None = None,
) -> dict:
    attempt_identity = semantic_identity(
        "cohort-v2-replay-attempt-v1", plan["identity"], scenario_collection["scenario_collection_id"], role
    )
    attempt_root = runtime_root / "attempts" / scenario_collection["scenario_collection_id"] / role
    attempt_root.mkdir(parents=True, exist_ok=False)
    log_root = runtime_root / "logs" / scenario_collection["scenario_collection_id"] / role
    log_root.mkdir(parents=True, exist_ok=False)
    engine = None
    display_process = None
    agent = None
    physics = None
    with tempfile.TemporaryDirectory(prefix=f"novphy-issue48-{scenario_collection['scenario_collection_id']}-{role}-") as temporary:
        temporary_root = Path(temporary)
        player = temporary_root / "player"
        _progress(f"{scenario_collection['scenario_collection_id']} {role}: unpacking the verified player")
        _, unity_version, protocol_version = archive_details(stage, player)
        envelope = plan["version_envelope"]
        if unity_version != envelope["unity_version"] or int(protocol_version) != envelope["observation_protocol_version"]:
            raise ValueError("unpacked player differs from the frozen version envelope")
        _install_level(runtime=player, xml_path=runtime_root / scenario_collection["source_xml_relative_path"])
        display, display_process = start_display(log_root / "xvnc.log")
        agent_port = free_port()
        game_port = free_port()
        physics_port = free_port()
        environment, _ = launch_environment(display, os.environ)
        environment["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        environment["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
        _progress(f"{scenario_collection['scenario_collection_id']} {role}: starting Unity and the game interface")
        with (log_root / "engine.log").open("wb") as log:
            engine = subprocess.Popen(
                [
                    "java", "-jar", "./game_playing_interface.jar",
                    "--agent-port", str(agent_port),
                    "--game-start-port", str(game_port),
                    "--physics-port", str(physics_port),
                    "--dev",
                ],
                cwd=player,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            _progress(f"{scenario_collection['scenario_collection_id']} {role}: waiting for the player to become ready")
            agent = connect_with_retry(
                "127.0.0.1", agent_port, timeout=300.0, deadline_seconds=90.0
            )
            agent.configure(agent_id=28748, mode=PlayingMode.TRAINING)
            prepare_for_play(agent, timeout=120.0, poll_delay=0.5)
            physics = connect_with_retry(
                "127.0.0.1", physics_port, timeout=120.0, deadline_seconds=90.0
            )
            _progress(f"{scenario_collection['scenario_collection_id']} {role}: waiting for a stable camera and slingshot")
            observation_capture = _stable_observation(physics)
            slingshot = _stable_slingshot(agent)
            if role == "original":
                if frozen_command is not None:
                    raise ValueError("original attempt cannot use a frozen replay command")
                anchored = anchor_action_to_slingshot_reference(
                    scenario_collection["intervention"]["interface_action"], slingshot
                )
                shot = action_to_shot(anchored, frame_height=480)
                actual_action = {
                    **anchored,
                    "socket_command": {
                        "x": shot["x"],
                        "y": shot["y"],
                        "tapTime": shot["tapTime"],
                        "releaseTime": shot["releaseTime"],
                    },
                }
            elif role == "replay":
                if frozen_command is None:
                    raise ValueError("replay attempt requires the original frozen socket command")
                actual_action = frozen_command["interface_action"]
                shot = actual_action["socket_command"]
            else:
                raise ValueError(f"unsupported replay attempt role: {role}")
            _progress(f"{scenario_collection['scenario_collection_id']} {role}: retaining the synchronized pre-intervention observation")
            physics_bindings = {
                "scenario_template_id": scenario_collection["scenario_template_identity"],
                "level_instance_id": scenario_collection["level_instance_identity"],
                "scenario_lineage_id": scenario_collection["scenario_lineage_identity"],
                "rollout_id": attempt_identity,
                "intervention_id": scenario_collection["intervention"]["identity"],
            }

            def shoot():
                mode = "exact frozen socket command" if role == "replay" else "declared intervention"
                _progress(f"{scenario_collection['scenario_collection_id']} {role}: applying the {mode}")
                return agent.shoot(
                    shot["x"],
                    shot["y"],
                    tap_time=shot["tapTime"],
                    fast=True,
                    release_time=shot["releaseTime"],
                )

            _progress(f"{scenario_collection['scenario_collection_id']} {role}: waiting for terminal physics capture")
            physics_metadata = capture_physics_v2_rollout(
                physics,
                attempt_root,
                shoot=shoot,
                source_bindings=physics_bindings,
                scenario_manifest_identity=scenario_collection["scenario_manifest_identity"],
                deadline_seconds=120.0,
            )
            observation_bindings = {
                "scenario_template_identity": scenario_collection["scenario_template_identity"],
                "level_instance_identity": scenario_collection["level_instance_identity"],
                "source_scenario_lineage_identity": scenario_collection["scenario_lineage_identity"],
                "rollout_identity": attempt_identity,
            }
            observation_manifest = persist_observation_trace(
                attempt_root / "observation-trace",
                [_capture_record(observation_capture)],
                observation_configuration=scenario_collection["observation_configuration"],
                source_bindings=observation_bindings,
                exposure_role=scenario_collection["exposure_role"],
            )
            attempt = {
                "schema": ATTEMPT_SCHEMA,
                "identity": attempt_identity,
                "attempt_role": role,
                "scenario_collection_identity": scenario_collection["identity"],
                "rollout_identity": attempt_identity,
                "version_envelope": plan["version_envelope"],
                "partition_manifest_identity": plan["partition_manifest_identity"],
                "collection_plan_identity": plan["identity"],
                "exposure_role": scenario_collection["exposure_role"],
                "scenario_manifest_identity": scenario_collection["scenario_manifest_identity"],
                "scenario_specification_identity": scenario_collection["scenario_specification_identity"],
                "scenario_content_identity": scenario_collection["scenario_content_identity"],
                "scenario_template_identity": scenario_collection["scenario_template_identity"],
                "level_instance_identity": scenario_collection["level_instance_identity"],
                "scenario_lineage_identity": scenario_collection["scenario_lineage_identity"],
                "intervention_identity": scenario_collection["intervention"]["identity"],
                "interface_action": actual_action,
                "engine_relative_action": scenario_collection["intervention"]["engine_relative_action"],
                "physics_capture_relative_path": "physics_capture_v2.json",
                "physics_capture_metadata": _physics_metadata(physics_metadata),
                "observation_trace_relative_path": "observation-trace",
                "observation_trace_manifest_identity": observation_manifest["identity"],
                "observation_configuration_identity": scenario_collection["observation_configuration_identity"],
            }
            write_immutable_cohort_v2_json(attempt, attempt_root / ATTEMPT_NAME)
            _progress(
                f"{scenario_collection['scenario_collection_id']} {role}: accepted capture {physics_metadata['capture_id']}"
            )
            return {
                "scenario_collection_id": scenario_collection["scenario_collection_id"],
                "attempt_role": role,
                "attempt_identity": attempt_identity,
                "status": "accepted",
                "capture_id": physics_metadata["capture_id"],
            }
        finally:
            if physics is not None:
                physics.disconnect()
            if agent is not None:
                agent.disconnect()
            terminate(engine)
            terminate(display_process)


def collect(
    repository_root: Path,
    stage: Path,
    runtime_root: Path,
    output: Path,
    *,
    determination_version: int = 4,
) -> dict:
    _progress("freezing and validating source manifests, partition, plan, and version envelope")
    plan = prepare_issue_48_runtime_root(
        repository_root,
        runtime_root,
        stage=stage,
        determination_version=determination_version,
    )
    ledger = []
    try:
        for scenario_collection in plan["scenario_collections"]:
            frozen_command = None
            frozen_command_path = runtime_root / "attempts" / scenario_collection["scenario_collection_id"] / FROZEN_COMMAND_NAME
            frozen_command_bytes = None
            for role in ("original", "replay"):
                try:
                    if role == "replay" and frozen_command_path.read_bytes() != frozen_command_bytes:
                        raise ValueError("frozen replay command changed before replay")
                    ledger.append(_capture_attempt(
                        stage,
                        runtime_root,
                        plan,
                        scenario_collection,
                        role,
                        frozen_command=frozen_command,
                    ))
                    if role == "original":
                        original_attempt = json.loads(
                            (
                                runtime_root
                                / "attempts"
                                / scenario_collection["scenario_collection_id"]
                                / "original"
                                / ATTEMPT_NAME
                            ).read_text(encoding="utf-8")
                        )
                        frozen_command = build_frozen_replay_command(
                            plan, scenario_collection, original_attempt
                        )
                        write_immutable_cohort_v2_json(
                            frozen_command, frozen_command_path
                        )
                        frozen_command_bytes = frozen_command_path.read_bytes()
                        _progress(
                            f"{scenario_collection['scenario_collection_id']}: froze the original exact socket command for one replay"
                        )
                except Exception as error:
                    failure = {
                        "scenario_collection_id": scenario_collection["scenario_collection_id"],
                        "attempt_role": role,
                        "status": "failed",
                        "reason": str(error),
                    }
                    ledger.append(failure)
                    write_immutable_cohort_v2_json(
                        {
                            "schema": "cohort_v2_replay_attempt_failure_v1",
                            **failure,
                        },
                        runtime_root / "attempts" / scenario_collection["scenario_collection_id"] / role / "attempt-failure.json",
                    )
                    raise
        write_immutable_cohort_v2_json(
            {
                "schema": "cohort_v2_replay_attempt_ledger_v1",
                "plan_identity": plan["identity"],
                "attempts": ledger,
                "retry_count": 0,
            },
            runtime_root / "attempt-ledger.json",
        )
        _progress("comparing every declared replay component and publishing the immutable bundle")
        bundle = build_issue_48_evidence(runtime_root, output)
        _progress(f"completed with passed={str(bundle['passed']).lower()}")
        return bundle
    except Exception:
        ledger_path = runtime_root / "attempt-ledger.json"
        if not ledger_path.exists():
            write_immutable_cohort_v2_json(
                {
                    "schema": "cohort_v2_replay_attempt_ledger_v1",
                    "plan_identity": plan["identity"],
                    "attempts": ledger,
                    "retry_count": 0,
                },
                ledger_path,
            )
        raise


def dry_run(repository_root: Path, stage: Path, *, determination_version: int = 4) -> dict:
    _progress("dry-run: validating the player archive and all source authorities")
    with tempfile.TemporaryDirectory(prefix="novphy-issue48-dry-run-") as temporary:
        plan = prepare_issue_48_runtime_root(
            repository_root,
            Path(temporary) / "runtime",
            stage=stage,
            determination_version=determination_version,
        )
        attempts = [
            {
                "scenario_collection_id": scenario_collection["scenario_collection_id"],
                "attempt_role": role,
                "exposure_role": scenario_collection["exposure_role"],
                "coverage_strata": scenario_collection["coverage_strata"],
            }
            for scenario_collection in plan["scenario_collections"]
            for role in ("original", "replay")
        ]
        result = {
            "schema": "issue_48_replay_dry_run_v1",
            "plan_identity": plan["identity"],
            "version_envelope_identity": plan["version_envelope"]["identity"],
            "attempts": attempts,
            "would_launch_unity_attempts": len(attempts),
            "passed": True,
        }
        _progress("dry-run: command and four-attempt plan are valid; Unity was not launched")
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--determination-version", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        result = dry_run(
            ROOT,
            args.stage,
            determination_version=args.determination_version,
        )
    else:
        result = collect(
            ROOT,
            args.stage,
            args.runtime_root,
            args.output,
            determination_version=args.determination_version,
        )
    rendered = result if args.dry_run else {
        "schema": result["schema"],
        "output": str(args.output),
        "passed": result["passed"],
    }
    print(json.dumps(rendered, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
