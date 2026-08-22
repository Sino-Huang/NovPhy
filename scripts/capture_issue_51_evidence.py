#!/usr/bin/env python3
"""Run issue #51's missing terminal probes and publish the pilot report."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

from scripts.build_issue_51_evidence import (
    DEFAULT_OUTPUT,
    FAILED_RUNTIME_DETERMINATION_3_IDENTITY,
    FAILED_RUNTIME_DETERMINATION_IDENTITY,
    PRIOR_DETERMINATION_IDENTITY,
    ROOT,
    _component_identities,
    _implementation_revision,
    _require_clean_tracked_worktree,
    _validate_component_sources,
    build_issue_51_evidence,
    build_pilot_plan,
)
from scripts.collection_plan import load_collection_plan
from scripts.build_issue_51_pilot_plan import (
    build_issue_51_supplementary_plan,
)
from scripts.physics_capture_v2 import parse_physics_capture_v2
from scripts.cohort_v2_scenarios import load_cohort_v2_scenario_manifest
from scripts.collect_rollouts import validate_cohort_v2_constraints_authority
from scripts.smoke_physics_capture import archive_details, free_port, start_display, terminate
from scripts.verify_physics_player import verify_physics_player_archive


DEFAULT_RUNTIME_ROOT = ROOT / ".local-artifacts/issue-51-pilot-run-determination-6"
DEFAULT_PRIOR_RUNTIME_ROOT = ROOT / ".local-artifacts/issue-51-pilot-run"
DEFAULT_FAILED_RUNTIME_ROOTS = (
    ROOT / ".local-artifacts/issue-51-pilot-run-determination-2",
    ROOT / ".local-artifacts/issue-51-pilot-run-determination-3",
)
STAGE_ROOT = ROOT / "sciencebirdsgames/physics-v2"
ACTUAL_COMMAND = (
    "python -u -m scripts.capture_issue_51_evidence "
    "--runtime-root .local-artifacts/issue-51-pilot-run-determination-6 "
    "--prior-runtime-root .local-artifacts/issue-51-pilot-run "
    "--failed-runtime-root .local-artifacts/issue-51-pilot-run-determination-2 "
    "--failed-runtime-root .local-artifacts/issue-51-pilot-run-determination-3 "
    "--output data/runtime_evidence/issue-51"
)


def _log(message: str) -> None:
    print(f"[issue-51] {message}", flush=True)


def _verify_implementation_player() -> dict[str, Any]:
    provenance = verify_physics_player_archive(STAGE_ROOT, physics_v2=True)
    if provenance.get("source_snapshot_commit") != _implementation_revision(ROOT):
        raise ValueError(
            "the physics-v2 player is not built from the issue-51 implementation revision; "
            "run scripts/build_physics_player.sh --physics-v2 after committing"
        )
    return provenance


def _run_with_heartbeat(command: list[str], environment: dict[str, str]) -> None:
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    started = time.monotonic()
    while process.poll() is None:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _log(
                f"still running after {int(time.monotonic() - started)} seconds: "
                f"{Path(command[0]).name}"
            )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _install_level(runtime: Path, xml_path: Path) -> None:
    target_root = runtime / "9001_Data/StreamingAssets/Levels/novelty_level_0/type2/Levels"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / "issue-51-level-clear.xml"
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


def _collection_command(
    authorities: dict[str, object],
    game: Path,
    runtime_root: Path,
) -> tuple[list[str], int]:
    agent_port = free_port()
    game_port = free_port()
    physics_port = free_port()
    return ([
        sys.executable,
        "-u",
        "-m",
        "scripts.collect_rollouts",
        "--output-dir",
        str(runtime_root / "collection"),
        "--fresh-engine-per-rollout",
        "--collection-plan",
        str(authorities["plan_path"]),
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
        "--scenario-v2-input",
        str(authorities["scenario_id"]),
        str(authorities["manifest_path"]),
        str(authorities["xml_path"]),
        str(authorities["template_path"]),
        str(authorities["workbook_path"]),
        str(game),
    ], physics_port)


def _prior_determination_data(
    prior_runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    report = json.loads(
        (prior_runtime_root / "collection/collection_plan_report.json").read_text(
            encoding="utf-8"
        )
    )
    shortfalls = report.get("realized_coverage_shortfalls")
    if (
        report.get("accepted_count") != 2
        or report.get("rejected_count") != 0
        or report.get("failed_count") != 0
        or report.get("quarantined_count") != 0
        or report.get("unmet_slots")
        or not isinstance(shortfalls, list)
        or len(shortfalls) != 1
        or shortfalls[0].get("intervention_id") != "level-clear-targeted"
        or shortfalls[0].get("intended_coverage_stratum") != "level clear"
    ):
        raise ValueError("the prior issue-51 runtime is not the failed determination-1 run")
    records = {}
    attempts = []
    for entry in report["attempt_ledger"]:
        artifact = Path(entry["artifact_path"])
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        record = json.loads(
            (artifact / "physics_capture_v2.json").read_text(encoding="utf-8")
        )
        capture = parse_physics_capture_v2(record)
        if capture.record["terminal_evidence"]["reason"] == "level_clear":
            raise ValueError("the prior failed determination unexpectedly contains level_clear")
        capture_path = f"prior-captures/{entry['intervention_id']}.json"
        records[capture_path] = record
        attempts.append({
            "attempt_identity": entry["attempt_id"],
            "intervention_id": entry["intervention_id"],
            "intervention_identity": entry["intervention_identity"],
            "status": entry["status"],
            "capture_id": capture.capture_id,
            "capture_path": capture_path,
            "terminal_reason": capture.record["terminal_evidence"]["reason"],
            "realized_coverage_strata": entry["realized_coverage_strata"],
        })
    plan_path = prior_runtime_root / "authorities/collection-plan.json"
    summary = {
        "schema": "issue_51_prior_failed_pilot_determination_v1",
        "identity": PRIOR_DETERMINATION_IDENTITY,
        "disposition": "failed",
        "failure_reason": "unmet_level_clear",
        "collection_plan_identity": report["plan_identity"],
        "collection_plan_path": "prior-captures/collection-plan.json",
        "counts": {
            "accepted": 2,
            "rejected": 0,
            "failed": 0,
            "quarantined": 0,
            "retried": 0,
        },
        "realized_coverage_shortfalls": shortfalls,
        "attempts": attempts,
    }
    return summary, records, plan_path


def _failed_runtime_determination_data(
    failed_runtime_root: Path,
    determination: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path, Path]:
    identities = {
        2: FAILED_RUNTIME_DETERMINATION_IDENTITY,
        3: FAILED_RUNTIME_DETERMINATION_3_IDENTITY,
    }
    try:
        identity = identities[determination]
    except KeyError as error:
        raise ValueError(f"unsupported failed issue-51 determination: {determination}") from error
    report_path = failed_runtime_root / "collection/collection_plan_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ledger = report.get("attempt_ledger")
    if (
        report.get("accepted_count") != 0
        or report.get("rejected_count") != 0
        or report.get("failed_count") != 2
        or report.get("quarantined_count") != 2
        or not isinstance(ledger, list)
        or len(ledger) != 2
        or report.get("realized_coverage_shortfalls")
    ):
        raise ValueError(
            f"the failed issue-51 runtime is not the determination-{determination} run"
        )
    plan_path = failed_runtime_root / "authorities/collection-plan.json"
    plan = load_collection_plan(plan_path).plan
    if plan.identity != report.get("plan_identity"):
        raise ValueError(
            f"the failed determination-{determination} collection plan is stale"
        )
    if not plan.identity.endswith(f"determination{determination}"):
        raise ValueError(
            f"the failed runtime does not contain determination {determination}"
        )
    interventions = {
        intervention.id: intervention.identity
        for scenario in plan.scenarios
        for intervention in scenario.interventions
    }
    records = {}
    attempts = []
    for entry in ledger:
        failure_path = Path(entry.get("failure_manifest_path", ""))
        if not failure_path.is_absolute():
            failure_path = ROOT / failure_path
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        reason = entry.get("reason")
        if (
            entry.get("status") != "failed"
            or entry.get("disposition") != "quarantine"
            or entry.get("artifact_path") is not None
            or entry.get("intervention_identity")
            != interventions.get(entry.get("intervention_id"))
            or not isinstance(reason, str)
            or "event is outside recorded fixed-step coverage" not in reason
            or failure.get("schema") != "collection_attempt_failure_v1"
            or failure.get("attempt_id") != entry.get("attempt_id")
            or failure.get("status") != "failed"
            or failure.get("reason") != reason
        ):
            raise ValueError(
                f"the failed determination-{determination} attempt accounting is stale"
            )
        relative = (
            f"prior-failures/determination-{determination}/"
            f"{entry['intervention_id']}-failure.json"
        )
        records[relative] = failure
        attempts.append({
            "attempt_identity": entry["attempt_id"],
            "intervention_id": entry["intervention_id"],
            "intervention_identity": entry["intervention_identity"],
            "status": "failed",
            "failure_code": entry["failure_code"],
            "failure_class": entry["failure_class"],
            "reason": reason,
            "failure_manifest_path": relative,
            "disposition": "quarantined",
            "eligible": False,
        })
    summary = {
        "schema": "issue_51_prior_failed_pilot_determination_v1",
        "identity": identity,
        "disposition": "failed",
        "failure_reason": "fixed_step_capture_gap",
        "collection_plan_identity": report["plan_identity"],
        "collection_plan_path": (
            f"prior-failures/determination-{determination}/collection-plan.json"
        ),
        "collection_report_path": (
            f"prior-failures/determination-{determination}/collection-plan-report.json"
        ),
        "counts": {
            "accepted": 0,
            "rejected": 0,
            "failed": 2,
            "quarantined": 2,
            "retried": 0,
        },
        "unmet_slots": report.get("unmet_slots", []),
        "attempts": attempts,
    }
    return summary, records, plan_path, report_path


def dry_run(
    prior_runtime_root: Path = DEFAULT_PRIOR_RUNTIME_ROOT,
    failed_runtime_roots: tuple[Path, ...] = DEFAULT_FAILED_RUNTIME_ROOTS,
) -> dict[str, object]:
    _log("dry-run: revalidating the accepted issue-44 through issue-50 authorities")
    sources = _validate_component_sources(ROOT)
    identities = _component_identities(ROOT, sources)
    _log("dry-run: verifying the provenance-bound physics-v2 player archive")
    provenance = _verify_implementation_player()
    prior, _, _ = _prior_determination_data(prior_runtime_root)
    failed = [
        _failed_runtime_determination_data(root, determination)[0]
        for determination, root in enumerate(failed_runtime_roots, start=2)
    ]
    with tempfile.TemporaryDirectory(prefix="novphy-issue51-dry-run-") as temporary:
        with redirect_stdout(io.StringIO()):
            authorities = build_issue_51_supplementary_plan(
                Path(temporary) / "authorities"
            )
        scenario = load_cohort_v2_scenario_manifest(
            Path(authorities["manifest_path"]),
            xml_path=Path(authorities["xml_path"]),
            template_source_path=Path(authorities["template_path"]),
        )
        validate_cohort_v2_constraints_authority(
            scenario, Path(authorities["workbook_path"])
        )
        plan = build_pilot_plan(ROOT, identities, authorities["plan_identity"])
    result = {
        "schema": "issue_51_representative_pilot_dry_run_v2",
        "pilot_plan_identity": plan["identity"],
        "supplementary_plan_identity": authorities["plan_identity"],
        "component_evidence_count": len(identities),
        "prior_failed_determination_identities": [
            prior["identity"],
            *(item["identity"] for item in failed),
        ],
        "would_launch_unity_attempts": 2,
        "unity_version": provenance["unity_version"],
        "actual_command": ACTUAL_COMMAND,
        "files_written": False,
        "passed": True,
    }
    _log("dry-run complete: source bindings, two-attempt plan, player, and command passed")
    return result


def _publish_supplementary(
    authorities: dict[str, object],
    runtime_root: Path,
    provenance: dict[str, object],
    prior_runtime_root: Path,
    failed_runtime_roots: tuple[Path, ...],
) -> Path:
    report = json.loads(
        (runtime_root / "collection/collection_plan_report.json").read_text(encoding="utf-8")
    )
    if (
        report.get("accepted_count") != 2
        or report.get("rejected_count") != 0
        or report.get("failed_count") != 0
        or report.get("quarantined_count") != 0
        or report.get("unmet_slots")
        or report.get("realized_coverage_shortfalls")
    ):
        raise ValueError(
            "issue-51 supplementary plan requires two accepted attempts and no "
            "failure, quarantine, unmet slot, or realized coverage shortfall"
        )
    supplement = runtime_root / "supplementary"
    captures = supplement / "captures"
    captures.mkdir(parents=True)
    prior, prior_records, prior_plan_path = _prior_determination_data(prior_runtime_root)
    for relative, record in prior_records.items():
        path = supplement / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    shutil.copyfile(
        prior_plan_path, supplement / "prior-captures/collection-plan.json"
    )
    (supplement / "prior-determination.json").write_text(
        json.dumps(prior, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = []
    for determination, failed_runtime_root in enumerate(
        failed_runtime_roots, start=2
    ):
        summary, failed_records, failed_plan_path, failed_report_path = (
            _failed_runtime_determination_data(failed_runtime_root, determination)
        )
        failed.append(summary)
        for relative, record in failed_records.items():
            path = supplement / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        base = supplement / f"prior-failures/determination-{determination}"
        shutil.copyfile(failed_plan_path, base / "collection-plan.json")
        shutil.copyfile(failed_report_path, base / "collection-plan-report.json")
    (supplement / "failed-determinations.json").write_text(
        json.dumps(
            {
                "schema": "issue_51_prior_failed_pilot_determinations_v1",
                "determinations": failed,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    attempts = []
    level_clear_count = 0
    for entry in report["attempt_ledger"]:
        artifact = Path(entry["artifact_path"])
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        record = json.loads((artifact / "physics_capture_v2.json").read_text(encoding="utf-8"))
        capture = parse_physics_capture_v2(record)
        terminal = capture.record["terminal_evidence"]["reason"]
        level_clear_count += terminal == "level_clear"
        capture_name = f"{entry['intervention_id']}.json"
        (captures / capture_name).write_text(
            json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        attempts.append({
            "attempt_identity": entry["attempt_id"],
            "intervention_id": entry["intervention_id"],
            "intervention_identity": entry["intervention_identity"],
            "status": entry["status"],
            "capture_id": capture.capture_id,
            "capture_path": f"captures/{capture_name}",
            "terminal_reason": terminal,
            "realized_coverage_strata": entry["realized_coverage_strata"],
        })
    targeted = next(
        attempt for attempt in attempts if attempt["intervention_id"] == "level-clear-targeted"
    )
    if level_clear_count < 1 or targeted["terminal_reason"] != "level_clear":
        observed = {attempt["intervention_id"]: attempt["terminal_reason"] for attempt in attempts}
        raise ValueError(f"issue-51 level-clear probe missed its target: {observed}")

    shutil.copyfile(authorities["plan_path"], supplement / "collection-plan.json")
    shutil.copyfile(authorities["manifest_path"], supplement / "scenario-manifest.json")
    shutil.copyfile(authorities["xml_path"], supplement / "scenario.xml")
    shutil.copyfile(authorities["template_path"], supplement / "template.xml")
    runtime_identity = (
        "issue-51-supplementary-runtime-v1:"
        f"{authorities['plan_identity']}:{provenance['source_snapshot_commit']}"
    )
    runtime_authority = {
        "schema": "issue_51_supplementary_runtime_authority_v1",
        "identity": runtime_identity,
        "evidence_source": "unity_runtime_non_fixture",
        "collection_plan_identity": authorities["plan_identity"],
        "scenario_manifest_identity": authorities["scenario_manifest_identity"],
        "source_snapshot_commit": provenance["source_snapshot_commit"],
        "unity_version": provenance["unity_version"],
        "physics_protocol_version": 1,
        "configured_fixed_step_capture_stride": 1,
        "attempts": attempts,
    }
    (supplement / "runtime-authority.json").write_text(
        json.dumps(runtime_authority, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return supplement


def run(
    runtime_root: Path,
    output: Path,
    prior_runtime_root: Path,
    failed_runtime_roots: tuple[Path, ...],
) -> dict[str, Any]:
    _require_clean_tracked_worktree(ROOT)
    _implementation_revision(ROOT)
    if runtime_root.exists():
        raise ValueError(f"runtime root already exists: {runtime_root}")
    if output.exists():
        raise ValueError(f"immutable output already exists: {output}")
    _prior_determination_data(prior_runtime_root)
    for determination, failed_runtime_root in enumerate(failed_runtime_roots, start=2):
        _failed_runtime_determination_data(failed_runtime_root, determination)
    runtime_root.mkdir(parents=True)

    _log("freezing the prospective issue-51 pilot and supplementary terminal plans")
    sources = _validate_component_sources(ROOT)
    identities = _component_identities(ROOT, sources)
    with redirect_stdout(io.StringIO()):
        authorities = build_issue_51_supplementary_plan(runtime_root / "authorities")
    pilot_plan = build_pilot_plan(ROOT, identities, authorities["plan_identity"])
    (runtime_root / "prospective-pilot-plan.json").write_text(
        json.dumps(pilot_plan, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _log("verifying the provenance-bound physics-v2 Unity player")
    provenance = _verify_implementation_player()
    game = runtime_root / "game"
    archive_details(STAGE_ROOT, game)
    _install_level(game, Path(authorities["xml_path"]))
    environment = dict(os.environ)
    display_process = None
    try:
        display, display_process = start_display(runtime_root / "display.log")
        environment["DISPLAY"] = display
        environment["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
        command, physics_port = _collection_command(authorities, game, runtime_root)
        environment["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
        _log("starting two fresh-engine level-clear probes; collector output follows")
        _run_with_heartbeat(command, environment)
    finally:
        receipt = terminate(display_process)
        _log(f"display shutdown: {receipt}")

    _log("binding all supplementary attempts and checking level_clear")
    supplement = _publish_supplementary(
        authorities,
        runtime_root,
        provenance,
        prior_runtime_root,
        failed_runtime_roots,
    )
    _log("auditing central capabilities, atomic quarantine, and exact accounting")
    return build_issue_51_evidence(output, supplement, repository_root=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and accept issue #51's representative cohort-v2 pilot"
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prior-runtime-root", type=Path, default=DEFAULT_PRIOR_RUNTIME_ROOT
    )
    parser.add_argument(
        "--failed-runtime-root", type=Path, action="append", default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    failed_runtime_roots = tuple(
        args.failed_runtime_root or DEFAULT_FAILED_RUNTIME_ROOTS
    )
    if len(failed_runtime_roots) != 2:
        raise ValueError("issue-51 requires failed runtime roots for determinations 2 and 3")
    result = (
        dry_run(args.prior_runtime_root, failed_runtime_roots)
        if args.dry_run
        else run(
            args.runtime_root,
            args.output,
            args.prior_runtime_root,
            failed_runtime_roots,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
