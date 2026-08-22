#!/usr/bin/env python3
"""Run issue #50's source-bound Unity probes and publish accepted derivations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from scripts.build_issue_50_evidence import build_issue_50_evidence
from scripts.build_issue_50_probe_plan import (
    ROOT,
    WORKBOOK_REFERENCE,
    build_issue_50_probe_plan,
)
from scripts.cohort_v2_physical_violations import (
    EXCESS_PENETRATION,
    derive_capture_physical_violations,
)
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.physics_capture_v2 import parse_physics_capture_v2
from scripts.smoke_physics_capture import archive_details, free_port, start_display, terminate
from scripts.verify_physics_player import verify_physics_player_archive


DEFAULT_RUNTIME_ROOT = ROOT / ".local-artifacts/issue-50-runtime"
DEFAULT_OUTPUT = ROOT / "data/runtime_evidence/issue-50"
STAGE_ROOT = ROOT / "sciencebirdsgames/physics-v2"
UNITY_EDITOR = Path(
    "/home/sukai/.local/share/novphy-unity/"
    "2019.4.41f2-6b23d448b533/editor/Editor/Unity"
)
PROBE_ENVIRONMENT = "NOVPHY_ISSUE_50_CAPABILITY_PROBE"
PROBE_ENVIRONMENT_VALUE = "unsupported-stationary-v1"


def _log(message: str) -> None:
    print(f"[issue-50] {message}", flush=True)


def _run_with_heartbeat(command: list[str], *, environment: dict[str, str]) -> None:
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    started = time.monotonic()
    while process.poll() is None:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _log(f"still running after {int(time.monotonic() - started)} seconds: {command[0]}")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _install_level(runtime: Path, xml_path: Path) -> None:
    target_root = (
        runtime
        / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels"
    )
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / xml_path.name
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
            "attempt_limit_per_level": "5",
            "allow_level_selection": "True",
        },
    )
    ET.SubElement(
        level_set,
        "game_levels",
        {"level_path": target.relative_to(runtime).as_posix()},
    )
    ET.indent(evaluation, space="  ")
    ET.ElementTree(evaluation).write(
        runtime / "config.xml", encoding="utf-8", xml_declaration=True
    )


def _existing_penetration_check() -> dict[str, object]:
    values = {}
    for case in ("collision", "support-change"):
        path = ROOT / f"data/runtime_evidence/issue-44/captures/{case}.json"
        capture = parse_physics_capture_v2(json.loads(path.read_text(encoding="utf-8")))
        derivation = derive_capture_physical_violations(
            capture,
            source_reference=str(path.relative_to(ROOT)),
            source_capture_bundle_identity="dry-run:issue-44",
        )
        labels = [
            record["predicates"][EXCESS_PENETRATION]["value"]
            for record in derivation["labels"]
        ]
        values[case] = {"positive": labels.count(True), "negative": labels.count(False)}
    if any(value["positive"] < 1 or value["negative"] < 1 for value in values.values()):
        raise ValueError("accepted issue-44 captures no longer cross the frozen penetration tolerance")
    return values


def dry_run() -> dict[str, object]:
    _log("materializing the prospective two-lineage probe plan")
    with tempfile.TemporaryDirectory(prefix="novphy-issue50-dry-run-") as temporary:
        authorities = build_issue_50_probe_plan(Path(temporary))
        for case, scenario in authorities["scenarios"].items():
            xml = Path(scenario["xml_path"]).read_text(encoding="utf-8")
            if 'physicsViolationProbe="unsupported_stationary_v1"' not in xml:
                raise ValueError(f"{case}: generated probe marker is missing")
        _log("validating the already accepted excess-penetration source windows")
        penetration = _existing_penetration_check()
        result = {
            "schema": "issue_50_capture_dry_run_v1",
            "plan_identity": authorities["plan_identity"],
            "scenario_lineage_count": len(
                {
                    value["scenario_lineage_id"]
                    for value in authorities["scenarios"].values()
                }
            ),
            "scenario_template_count": len(
                {
                    value["scenario_template_id"]
                    for value in authorities["scenarios"].values()
                }
            ),
            "existing_penetration_windows": penetration,
            "unity_editor": str(UNITY_EDITOR),
            "actual_command": (
                "python -u -m scripts.capture_issue_50_evidence "
                "--runtime-root .local-artifacts/issue-50-runtime "
                "--output data/runtime_evidence/issue-50"
            ),
            "files_written": False,
            "passed": True,
        }
    _log("dry-run complete; plan, source bindings, and command wiring passed")
    return result


def _prepare_games(authorities: dict[str, object], runtime_root: Path) -> dict[str, Path]:
    games = {}
    for case, scenario in authorities["scenarios"].items():
        game = runtime_root / "games" / case
        archive_details(STAGE_ROOT, game)
        _install_level(game, Path(scenario["xml_path"]))
        games[case] = game
    return games


def _collection_command(
    authorities: dict[str, object],
    games: dict[str, Path],
    runtime_root: Path,
) -> tuple[list[str], int]:
    agent_port = free_port()
    game_port = free_port()
    physics_port = free_port()
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.collect_rollouts",
        "--output-dir",
        str(runtime_root / "collection"),
        "--fresh-engine-per-rollout",
        "--collection-plan",
        authorities["plan_path"],
        "--physics-capture-v2",
        "--port",
        str(agent_port),
        "--physics-port",
        str(physics_port),
        "--engine-agent-port",
        str(agent_port),
        "--engine-game-port",
        str(game_port),
        "--engine-settle-seconds",
        "1",
        "--agent-settle-seconds",
        "1",
        "--fps",
        "1",
        "--duration",
        "1",
    ]
    for case, scenario in authorities["scenarios"].items():
        command.extend(
            [
                "--scenario-v2-input",
                scenario["scenario_id"],
                scenario["manifest_path"],
                scenario["xml_path"],
                scenario["template_path"],
                str(ROOT / WORKBOOK_REFERENCE),
                str(games[case]),
            ]
        )
    return command, physics_port


def _publish_probe_sources(
    authorities: dict[str, object],
    runtime_root: Path,
    source_snapshot_commit: str,
) -> Path:
    source_root = runtime_root / "source-probes"
    report = json.loads(
        (runtime_root / "collection/collection_plan_report.json").read_text(encoding="utf-8")
    )
    if (
        report.get("accepted_count") != 4
        or report.get("rejected_count") != 0
        or report.get("failed_count") != 0
    ):
        raise ValueError("issue-50 requires 4 accepted, 0 rejected, and 0 failed probes")
    captures = {}
    probes = []
    for entry in report["attempt_ledger"]:
        case = entry["intervention_id"]
        artifact = Path(entry["artifact_path"])
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        record = json.loads((artifact / "physics_capture_v2.json").read_text(encoding="utf-8"))
        capture = parse_physics_capture_v2(record)
        write_immutable_cohort_v2_json(record, source_root / "captures" / f"{case}.json")
        captures[case] = {"capture_id": capture.capture_id, "path": f"captures/{case}.json"}
        probes.append(
            {
                "case": case,
                "capture_id": capture.capture_id,
                "scenario_lineage_id": capture.source_bindings["scenario_lineage_id"],
                "level_instance_id": capture.source_bindings["level_instance_id"],
                "scenario_template_id": capture.source_bindings["scenario_template_id"],
            }
        )
    runtime_identity = (
        "issue-50-physical-violation-probe-runtime-bundle-v1:"
        f"{authorities['plan_identity']}:{source_snapshot_commit}"
    )
    runtime = {
        "schema": "issue_50_physical_violation_probe_runtime_bundle_v1",
        "identity": runtime_identity,
        "source_snapshot_commit": source_snapshot_commit,
        "collection_plan_identity": authorities["plan_identity"],
        "evidence_source": "unity_runtime_non_fixture",
        "final_evaluation": False,
        "probe_environment": (
            "NOVPHY_ISSUE_50_CAPABILITY_PROBE=unsupported-stationary-v1"
        ),
        "probes": sorted(probes, key=lambda value: value["case"]),
    }
    capture_bundle = {
        "schema": "issue_50_physical_violation_probe_capture_bundle_v1",
        "identity": (
            "issue-50-physical-violation-probe-capture-bundle-v1:"
            f"{authorities['plan_identity']}"
        ),
        "runtime_bundle_identity": runtime_identity,
        "captures": dict(sorted(captures.items())),
    }
    write_immutable_cohort_v2_json(runtime, source_root / "runtime-bundle-manifest.json")
    write_immutable_cohort_v2_json(
        capture_bundle, source_root / "capture-bundle-manifest.json"
    )
    return source_root


def run(runtime_root: Path, output: Path) -> dict[str, object]:
    if runtime_root.exists():
        raise ValueError(f"runtime root already exists: {runtime_root}")
    if output.exists():
        raise ValueError(f"immutable output already exists: {output}")
    runtime_root.mkdir(parents=True)
    source_root = runtime_root / "source-probes"
    _log("freezing and materializing the issue-50 probe authorities")
    authorities = build_issue_50_probe_plan(source_root)

    environment = dict(os.environ)
    environment["UNITY_2019_4_41F2"] = str(UNITY_EDITOR)
    _log("building the provenance-bound physics-v2 Unity player")
    _run_with_heartbeat(
        [str(ROOT / "scripts/build_physics_player.sh"), "--physics-v2"],
        environment=environment,
    )
    _log("verifying the staged physics-v2 player archive")
    provenance = verify_physics_player_archive(STAGE_ROOT, physics_v2=True)
    games = _prepare_games(authorities, runtime_root)
    display_process = None
    try:
        display, display_process = start_display(runtime_root / "display.log")
        environment["DISPLAY"] = display
        environment["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        environment[PROBE_ENVIRONMENT] = PROBE_ENVIRONMENT_VALUE
        command, physics_port = _collection_command(authorities, games, runtime_root)
        environment["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
        _log("starting four fresh-engine Unity probes; collector logs follow")
        _run_with_heartbeat(command, environment=environment)
    finally:
        receipt = terminate(display_process)
        _log(f"display shutdown: {receipt}")

    _log("binding the exact accepted Unity captures to the frozen probe plan")
    probe_root = _publish_probe_sources(
        authorities,
        runtime_root,
        provenance["source_snapshot_commit"],
    )
    _log("deriving, adjudicating, and publishing the issue-50 bundle")
    return build_issue_50_evidence(output, probe_root=probe_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and adjudicate issue #50's physical-violation evidence"
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = dry_run() if args.dry_run else run(args.runtime_root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
