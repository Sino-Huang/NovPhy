#!/usr/bin/env python3
"""Collect the two non-final request-72 probes required by issue #46."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.build_issue_46_evidence import build_issue_46_evidence
from scripts.manual_agent import connect_with_retry, prepare_for_play
from scripts.smoke_physics_capture import (
    archive_details,
    free_port,
    launch_environment,
    start_display,
    terminate,
)
from src.webui.bridge import ObservationCaptureEngine, PlayingMode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "sciencebirdsgames" / "observation-v1"
DEFAULT_SOURCE_STAGE = ROOT / "sciencebirdsgames" / "physics-v2"
DEFAULT_OUTPUT = ROOT / "data" / "runtime_evidence" / "issue-46"


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


def _install_level(runtime: Path, source_stage: Path, level_name: str) -> None:
    level_source = source_stage / "review-levels" / f"{level_name}.xml"
    if not level_source.is_file():
        raise ValueError("issue #46 source-bound review level is missing")
    target_root = (
        runtime / "9001_Data" / "StreamingAssets" / "Levels"
        / "novelty_level_0" / "type2" / "Levels"
    )
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(level_source, target_root / level_source.name)

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
    relative_root = target_root.relative_to(runtime).as_posix()
    ET.SubElement(
        level_set, "game_levels",
        {"level_path": f"{relative_root}/{level_source.name}"},
    )
    ET.indent(evaluation, space="  ")
    ET.ElementTree(evaluation).write(
        runtime / "config.xml", encoding="utf-8", xml_declaration=True
    )


def _probe(
    capture: ObservationCaptureEngine,
    manifest_path: Path,
    *,
    probe_identity: str,
    configuration: str,
    exposure_role: str,
    source_snapshot_commit: str,
    player_archive_identity: str,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = manifest["scenario_manifest"]
    return {
        "probe_identity": probe_identity,
        "evidence_source": "unity_runtime_non_fixture",
        "source_snapshot_commit": source_snapshot_commit,
        "player_archive_identity": player_archive_identity,
        "scenario_manifest_identity": manifest["identity"],
        "observation_configuration": configuration,
        "exposure_role": exposure_role,
        "source_bindings": {
            "scenario_template_identity": manifest["template_record"]["identity"],
            "level_instance_identity": scenario["level_instance"]["identity"],
            "source_scenario_lineage_identity": scenario["scenario_lineage"]["identity"],
            "rollout_identity": f"issue-46-runtime-probe-v1:{probe_identity}",
        },
        "captures": [_capture_record(capture)],
    }


def _capture_one(
    stage: Path,
    source_stage: Path,
    level_name: str,
) -> tuple[ObservationCaptureEngine, str, str]:
    engine = None
    display_process = None
    agent = None
    physics = None
    with tempfile.TemporaryDirectory(prefix="novphy-issue46-runtime-") as temporary:
        temporary_root = Path(temporary)
        runtime = temporary_root / "player"
        archive, _, _ = archive_details(stage, runtime)
        _install_level(runtime, source_stage, level_name)
        provenance = json.loads((runtime / "provenance.json").read_text(encoding="utf-8"))
        source_commit = provenance["project"]["git_head"]
        archive_identity = (
            "physics-v2-player-archive-v1:sha256:"
            + hashlib.sha256(archive.read_bytes()).hexdigest()
        )
        display, display_process = start_display(temporary_root / "xvnc.log")
        agent_port = free_port()
        game_port = free_port()
        physics_port = free_port()
        environment, _ = launch_environment(display, os.environ)
        environment["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        environment["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
        with (temporary_root / "engine.log").open("wb") as log:
            engine = subprocess.Popen(
                [
                    "java", "-jar", "./game_playing_interface.jar",
                    "--agent-port", str(agent_port),
                    "--game-start-port", str(game_port),
                    "--physics-port", str(physics_port),
                    "--dev",
                ],
                cwd=runtime,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            agent = connect_with_retry(
                "127.0.0.1", agent_port, timeout=300.0, deadline_seconds=90.0
            )
            agent.configure(agent_id=28746, mode=PlayingMode.TRAINING)
            prepare_for_play(agent, timeout=120.0, poll_delay=0.5)
            physics = connect_with_retry(
                "127.0.0.1", physics_port, timeout=120.0, deadline_seconds=90.0
            )
            return physics.get_observation_capture(), source_commit, archive_identity
        finally:
            if physics is not None:
                physics.disconnect()
            if agent is not None:
                agent.disconnect()
            terminate(engine)
            terminate(display_process)


def collect(stage: Path, source_stage: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("issue #46 output already exists")
    training_capture, source_commit, archive_identity = _capture_one(
        stage, source_stage, "training"
    )
    calibration_capture, calibration_commit, calibration_archive = _capture_one(
        stage, source_stage, "calibration"
    )
    if calibration_commit != source_commit or calibration_archive != archive_identity:
        raise ValueError("issue #46 probes used different player authorities")
    probes = [
        _probe(
            training_capture,
            source_stage / "review-manifests" / "training.json",
            probe_identity="training-native",
            configuration="agent_rgb8_native_v1",
            exposure_role="training",
            source_snapshot_commit=source_commit,
            player_archive_identity=archive_identity,
        ),
        _probe(
            calibration_capture,
            source_stage / "review-manifests" / "calibration.json",
            probe_identity="calibration-resized",
            configuration="agent_rgb8_nearest_320x240_v1",
            exposure_role="calibration",
            source_snapshot_commit=source_commit,
            player_archive_identity=archive_identity,
        ),
    ]
    return build_issue_46_evidence(output, probes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--source-stage", type=Path, default=DEFAULT_SOURCE_STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bundle = collect(args.stage, args.source_stage, args.output)
    print(json.dumps({"identity": bundle["identity"], "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
