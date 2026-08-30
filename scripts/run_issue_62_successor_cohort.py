"""Freeze, collect, publish, and validate issue #62's successor cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import Any, Final, Mapping

from scripts.build_issue_45_evidence import CONSTRAINTS_WORKBOOK_REFERENCE
from scripts.cohort_v2_release import _write_derivations
from scripts.cohort_v2_scenarios import (
    create_scenario_template_constraints,
    create_scenario_template_record,
    materialize_template_bound_level_instance,
    write_cohort_v2_scenario_manifest,
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)
from scripts.collect_rollouts import capture_physics_v2_rollout
from scripts.capture_issue_53_evidence import _install_level
from scripts.manual_agent import (
    connect_with_retry,
    prepare_for_play,
    start_engine,
    stop_started_engine,
)
from scripts.observation_trace import validate_observation_trace
from scripts.physics_capture_v2 import load_physics_capture_v2
from scripts.run_issue_60_temporal_carrier import main as temporal_carrier_main
from scripts.slingshot_readiness import prepare_screen_shot
from scripts.smoke_physics_capture import (
    archive_details,
    free_port,
    start_display,
    terminate,
)
from scripts.verify_physics_player import verify_physics_player_archive
from scripts.scenario_manifest import BenchmarkCondition
from src.webui.bridge import GameState, PlayingMode, ScienceBirdsBridge
from tasks.task_generator.canonical_materialization import (
    CanonicalMaterializationRequest,
)
from world_model.data.deployment_temporal import (
    DeploymentTrajectoryReader,
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)
from world_model.data.successor_cohort import (
    ACTION_BOUNDS,
    GENERATOR_FAMILIES,
    PILOT_REPORT_SCHEMA,
    PUBLIC_ROLES,
    RELEASE_SCHEMA,
    TRAJECTORY_SCHEMA,
    SuccessorCohortError,
    SuccessorCohortReader,
    build_pilot_plan,
    build_production_plan,
    load_successor_trajectory,
    release_identity_for_plan,
    successor_identity,
    validate_pilot_report,
    validate_successor_plan,
)
from world_model.planning.gameplay import SlingshotAction, SlingshotActionBounds
from world_model.training.manifest import git_revision


ROOT: Final = Path(__file__).resolve().parents[1]
STAGE_ROOT: Final = ROOT / "sciencebirdsgames/aligned-observation-v1"
DEFAULT_PILOT_PLAN: Final = (
    ROOT / "docs/data_contracts/issue_62_pilot_plan_v1.json"
)
DEFAULT_PILOT_RUNTIME: Final = ROOT / ".local-artifacts/issue-62-pilot-run"
DEFAULT_PILOT_REPORT: Final = ROOT / ".local-artifacts/issue-62-pilot-report.json"
DEFAULT_PILOT_AUDIT: Final = ROOT / "data/issue-62-pilot-audit"
DEFAULT_PRODUCTION_PLAN: Final = (
    ROOT / ".local-artifacts/issue-62-production-plan.json"
)
DEFAULT_PRODUCTION_RUNTIME: Final = (
    ROOT / ".local-artifacts/issue-62-production-run"
)
DEFAULT_RELEASE: Final = ROOT / ".local-artifacts/issue-62-successor-cohort"
DEFAULT_SUMMARY: Final = (
    ROOT / "data/runtime_evidence/issue-62/successor-cohort-summary.json"
)
OBSERVATION_CONFIGURATION: Final = "agent_rgb8_native_v1"
PILOT_AUDIT_FPS: Final = 25


def _log(message: str) -> None:
    print(f"[issue-62] {message}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise SuccessorCohortError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise SuccessorCohortError(f"{label} is not an object")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _load_plan(path: Path, phase: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = validate_successor_plan(json.loads(raw))
    if raw != _canonical_bytes(value) or value["phase"] != phase:
        raise SuccessorCohortError(f"{phase} collection plan is not exact")
    return value


def _player() -> dict[str, Any]:
    return verify_physics_player_archive(STAGE_ROOT, physics_v2=True)


def _family(name: str) -> Mapping[str, Any]:
    return next(item for item in GENERATOR_FAMILIES if item["name"] == name)


def _materialize_slot(slot: Mapping[str, Any], root: Path) -> dict[str, Any]:
    family = _family(str(slot["generator_family"]))
    workbook_path = ROOT / CONSTRAINTS_WORKBOOK_REFERENCE
    source_path = ROOT / str(family["template_source"])
    constraints = create_scenario_template_constraints(
        workbook_path.read_bytes(),
        source_reference=CONSTRAINTS_WORKBOOK_REFERENCE,
        sheet_name="Task Variations",
        row_number=int(family["workbook_row"]),
        canonical_generator_template_name=str(family["canonical_template_name"]),
        reference_point=tuple(family["reference_point"]),
        min_coordinate=tuple(family["min_coordinate"]),
        max_coordinate=tuple(family["max_coordinate"]),
    )
    condition = BenchmarkCondition("novelty_level_0", str(family["name"]))
    template = create_scenario_template_record(
        source_path.read_bytes(),
        source_reference=str(family["template_source"]),
        benchmark_conditions=[condition],
        generation_constraints=constraints,
    )
    xml_path = root / "scenario.xml"
    manifest_path = root / "scenario-manifest.json"
    request = CanonicalMaterializationRequest(
        template_path=source_path,
        output_xml_path=xml_path,
        output_manifest_path=root / "generated-scenario.json",
        template_name=str(family["canonical_template_name"]),
        benchmark_condition=condition,
        template_identity=template.identity,
        generation_seed=int(slot["generation_seed"]),
        reference_point=tuple(family["reference_point"]),
        min_coordinate=tuple(family["min_coordinate"]),
        max_coordinate=tuple(family["max_coordinate"]),
        restricted_objects=(),
        template_source_reference=str(family["template_source"]),
    )
    materialized, scenario = materialize_template_bound_level_instance(
        request,
        template,
        constraints_workbook_path=workbook_path,
        publish=False,
    )
    write_immutable_cohort_v2_bytes(materialized.xml_content, xml_path)
    write_cohort_v2_scenario_manifest(scenario, manifest_path)
    return {
        "scenario": scenario,
        "xml_path": xml_path,
        "manifest_path": manifest_path,
    }


def freeze_pilot_plan(path: Path = DEFAULT_PILOT_PLAN) -> dict[str, Any]:
    plan = build_pilot_plan()
    write_immutable_cohort_v2_json(plan, path)
    _log(
        f"pilot plan frozen identity={plan['identity']} "
        f"lineages={len(plan['lineages'])} final_evaluation=absent"
    )
    return plan


def _synthetic_pilot_records(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "slot_identity": slot["slot_identity"],
            "status": "accepted",
            "exposure_role": slot["exposure_role"],
            "generator_family": slot["generator_family"],
            "behavior_policy": slot["behavior_policy"],
            "executed_action_count": 2,
            "decision_state_count": 3,
            "frame_record_count": 16,
            "terminal_reason": "shot_limit",
            "action_strata": [
                action["action_stratum"] for action in slot["planned_actions"][:2]
            ],
            "interaction_coverage": ["collision"],
            "wall_seconds": 1.0,
            "artifact_bytes": 1_000,
            "attempt_count": 1,
        }
        for slot in plan["lineages"]
    ]


def _pilot_report(
    plan: Mapping[str, Any], records: list[Mapping[str, Any]]
) -> dict[str, Any]:
    completed = {item["slot_identity"] for item in records}
    expected = {item["slot_identity"] for item in plan["lineages"]}
    if completed != expected:
        raise SuccessorCohortError("pilot report does not cover every frozen lineage")
    accepted = [item for item in records if item["status"] == "accepted"]
    accepted_roles = Counter(item["exposure_role"] for item in accepted)
    multi_shot_roles = {
        item["exposure_role"]
        for item in accepted
        if item["executed_action_count"] >= 2
    }
    coverage = Counter(
        value for item in accepted for value in item["interaction_coverage"]
    )
    action_strata = Counter(
        value for item in accepted for value in item["action_strata"]
    )
    proxy_values = [
        len(item["interaction_coverage"]) / item["executed_action_count"]
        for item in accepted
        if item["executed_action_count"]
    ]
    runtime_seconds = [float(item["wall_seconds"]) for item in accepted]
    artifact_bytes = [int(item["artifact_bytes"]) for item in accepted]
    passed = (
        len(records) == len(plan["lineages"])
        and len(accepted) >= math.ceil(5 * len(records) / 6)
        and multi_shot_roles == set(PUBLIC_ROLES)
        and {item["generator_family"] for item in accepted}
        == {item["name"] for item in GENERATOR_FAMILIES}
        and {item["behavior_policy"] for item in accepted}
        == {"uniform_random", "stratified_bounds"}
    )
    payload = {
        "schema": PILOT_REPORT_SCHEMA,
        "pilot_plan_identity": plan["identity"],
        "planned_lineage_count": len(plan["lineages"]),
        "completed_lineage_count": len(records),
        "accepted_lineage_count": len(accepted),
        "failed_lineage_count": len(records) - len(accepted),
        "accepted_by_role": dict(sorted(accepted_roles.items())),
        "typical_trajectory": {
            "median_executed_actions": statistics.median(
                item["executed_action_count"] for item in accepted
            ),
            "median_decision_states": statistics.median(
                item["decision_state_count"] for item in accepted
            ),
            "terminal_mix": dict(sorted(Counter(
                item["terminal_reason"] for item in accepted
            ).items())),
        },
        "coverage": {
            "interaction_counts": dict(sorted(coverage.items())),
            "action_stratum_counts": dict(sorted(action_strata.items())),
            "generator_family_counts": dict(sorted(Counter(
                item["generator_family"] for item in accepted
            ).items())),
            "behavior_policy_counts": dict(sorted(Counter(
                item["behavior_policy"] for item in accepted
            ).items())),
        },
        "independent_lineage_learning_curve_variance_proxy": {
            "estimand": "per-lineage realized-interaction-strata per executed action",
            "variance": statistics.pvariance(proxy_values) if len(proxy_values) > 1 else 0.0,
            "model_sufficiency_claim": False,
        },
        "held_out_precision": {
            role: {
                "accepted_lineages": accepted_roles[role],
                "worst_case_binomial_standard_error": (
                    0.5 / math.sqrt(accepted_roles[role])
                    if accepted_roles[role]
                    else None
                ),
            }
            for role in ("calibration", "model_selection")
        },
        "runtime_cost": {
            "median_seconds_per_lineage": statistics.median(runtime_seconds),
            "median_bytes_per_lineage": int(statistics.median(artifact_bytes)),
        },
        "production_freeze_rule": {
            "training_ladder": [6, 200, 1_000, 5_000, 10_000],
            "held_out_lineages_per_role": 200,
            "minimum_pilot_acceptance_fraction": "5/6",
            "outcome_conditioned_replacement": False,
        },
        "final_evaluation_opened": False,
        "passed": passed,
    }
    report = dict(payload)
    report["identity"] = successor_identity("issue-62-pilot-report-v1", payload)
    return report


def _verify_webm_encoder() -> None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "libvpx" not in result.stdout:
        raise SuccessorCohortError("ffmpeg does not provide the VP8 libvpx encoder")


def _encode_agent_frames_webm(frames: list[Path], output: Path) -> None:
    if not frames:
        raise SuccessorCohortError("pilot audit video has no agent frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".issue-62-audit-frames-", dir=output.parent
    ) as temporary:
        sequence = Path(temporary)
        for index, source in enumerate(frames):
            (sequence / f"frame_{index:06d}.png").symlink_to(source.resolve())
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}-", suffix=".webm", dir=output.parent,
            delete=False,
        ) as handle:
            temporary_video = Path(handle.name)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(PILOT_AUDIT_FPS),
            "-i", str(sequence / "frame_%06d.png"),
            "-an", "-c:v", "libvpx", "-deadline", "realtime",
            "-cpu-used", "5", "-crf", "10", "-b:v", "4M",
            "-pix_fmt", "yuv420p", "-f", "webm", str(temporary_video),
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            while True:
                try:
                    return_code = process.wait(timeout=5.0)
                    break
                except subprocess.TimeoutExpired:
                    _log(
                        f"audit encoding video={output.name} frames={len(frames)} "
                        f"elapsed_seconds={time.monotonic() - started:.1f}"
                    )
            if return_code != 0 or temporary_video.stat().st_size == 0:
                raise SuccessorCohortError(
                    f"ffmpeg failed to encode pilot audit video {output.name}"
                )
            os.replace(temporary_video, output)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if temporary_video.exists():
                temporary_video.unlink()


def _pilot_audit_frames(
    trajectory_root: Path,
    trajectory: Mapping[str, Any],
) -> tuple[list[Path], list[dict[str, Any]]]:
    frames = []
    ranges = []
    for shot in trajectory["shots"]:
        observation_root = trajectory_root / shot["path"] / "observation-trace"
        observation = validate_observation_trace(observation_root)
        first = len(frames)
        for frame in observation["frame_records"]:
            path = observation_root / frame["agent_observation"]["relative_path"]
            if not path.is_file():
                raise SuccessorCohortError("pilot audit agent frame is missing")
            frames.append(path)
        source_frames = observation["frame_records"]
        ranges.append({
            "shot_index": shot["shot_index"],
            "action_stratum": shot["action_stratum"],
            "video_frame_start": first,
            "video_frame_end_exclusive": len(frames),
            "source_fixed_step_start": source_frames[0]["fixed_step"],
            "source_fixed_step_end": source_frames[-1]["fixed_step"],
            "source_fixed_time_start_seconds": source_frames[0][
                "fixed_time_seconds"
            ],
            "source_fixed_time_end_seconds": source_frames[-1][
                "fixed_time_seconds"
            ],
            "observation_manifest_identity": observation["identity"],
        })
    return frames, ranges


def _pilot_audit_gallery(manifest: Mapping[str, Any]) -> bytes:
    sections = []
    for item in manifest["videos"]:
        label = (
            f"{item['ordinal'] + 1:03d} · {item['exposure_role']} · "
            f"{item['generator_family']} · {item['behavior_policy']}"
        )
        sections.append(
            "<section>"
            f"<h2>{html.escape(label)}</h2>"
            f"<video controls preload=\"metadata\" src=\"{html.escape(item['path'])}\"></video>"
            f"<p>{item['frame_count']} agent frames · "
            f"{item['executed_action_count']} shots · "
            f"terminal: {html.escape(item['terminal_reason'])}</p>"
            f"<code>{html.escape(item['slot_identity'])}</code>"
            "</section>"
        )
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Issue 62 pilot frame audit</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1000px}"
        "section{border-top:1px solid #bbb;padding:1rem 0}"
        "video{display:block;width:100%;max-width:800px;background:#111}"
        "code{overflow-wrap:anywhere}</style></head><body>"
        "<h1>Issue 62 pilot agent-frame audit</h1>"
        f"<p>{len(manifest['videos'])} accepted lineages; "
        f"VP8/WebM at {manifest['playback_fps']} fps. "
        "These are deployment-valid agent observations; canonical frames are excluded.</p>"
        + "".join(sections)
        + "</body></html>"
    )
    return document.encode("utf-8")


def write_pilot_audit(
    plan: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    runtime_root: Path,
    output: Path,
) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = _load(manifest_path, "pilot audit manifest")
        if (
            manifest.get("pilot_plan_identity") != plan["identity"]
            or manifest.get("pilot_report_identity") != report["identity"]
            or not (output / "index.html").is_file()
            or any(
                not (output / item["path"]).is_file()
                for item in manifest.get("videos", ())
            )
        ):
            raise SuccessorCohortError("existing pilot audit differs")
        _log(
            f"pilot audit already complete videos={len(manifest['videos'])} "
            f"gallery={output / 'index.html'}"
        )
        return manifest
    records_by_slot = {item["slot_identity"]: item for item in records}
    if set(records_by_slot) != {
        item["slot_identity"] for item in plan["lineages"]
    }:
        raise SuccessorCohortError("pilot audit lacks frozen lineage accounting")
    accepted = [
        slot for slot in plan["lineages"]
        if records_by_slot[slot["slot_identity"]]["status"] == "accepted"
    ]
    output.mkdir(parents=True, exist_ok=True)
    videos = []
    for video_index, slot in enumerate(accepted, start=1):
        result = records_by_slot[slot["slot_identity"]]
        trajectory_root = (
            _accepted_root(Path(runtime_root).resolve(), "pilot")
            / slot["exposure_role"] / slot["slot_identity"]
        )
        trajectory = _load(trajectory_root / "trajectory.json", "pilot trajectory")
        if (
            trajectory.get("slot_identity") != slot["slot_identity"]
            or trajectory.get("trajectory_identity") != result["trajectory_identity"]
        ):
            raise SuccessorCohortError("pilot audit trajectory binding differs")
        frames, shot_ranges = _pilot_audit_frames(trajectory_root, trajectory)
        if len(frames) != result["frame_record_count"]:
            raise SuccessorCohortError("pilot audit frame count differs")
        relative = Path("videos") / (
            f"lineage-{slot['ordinal'] + 1:03d}-{slot['exposure_role']}.webm"
        )
        _log(
            f"pilot audit {video_index}/{len(accepted)}: "
            f"role={slot['exposure_role']} frames={len(frames)} "
            f"video={relative.name}"
        )
        _encode_agent_frames_webm(frames, output / relative)
        videos.append({
            "ordinal": slot["ordinal"],
            "slot_identity": slot["slot_identity"],
            "trajectory_identity": result["trajectory_identity"],
            "exposure_role": slot["exposure_role"],
            "generator_family": slot["generator_family"],
            "behavior_policy": slot["behavior_policy"],
            "terminal_reason": result["terminal_reason"],
            "executed_action_count": result["executed_action_count"],
            "frame_count": len(frames),
            "path": relative.as_posix(),
            "shot_ranges": shot_ranges,
        })
    failed = [
        {
            "ordinal": slot["ordinal"],
            "slot_identity": slot["slot_identity"],
            "exposure_role": slot["exposure_role"],
            "failures": records_by_slot[slot["slot_identity"]]["failures"],
        }
        for slot in plan["lineages"]
        if records_by_slot[slot["slot_identity"]]["status"] == "failed"
    ]
    payload = {
        "schema": "issue_62_pilot_frame_audit_v1",
        "pilot_plan_identity": plan["identity"],
        "pilot_report_identity": report["identity"],
        "source_observation_role": "agent",
        "canonical_observations_included": False,
        "video_container": "webm",
        "video_codec": "vp8",
        "playback_fps": PILOT_AUDIT_FPS,
        "video_count": len(videos),
        "videos": videos,
        "failed_lineages": failed,
        "gallery": "index.html",
    }
    manifest = {
        **payload,
        "identity": successor_identity(
            "issue-62-pilot-frame-audit-v1",
            plan["identity"],
            report["identity"],
            tuple(
                (
                    item["trajectory_identity"],
                    tuple(
                        shot["observation_manifest_identity"]
                        for shot in item["shot_ranges"]
                    ),
                )
                for item in videos
            ),
        ),
    }
    write_immutable_cohort_v2_bytes(
        _pilot_audit_gallery(manifest), output / "index.html"
    )
    write_immutable_cohort_v2_json(manifest, manifest_path)
    _log(
        f"pilot audit complete videos={len(videos)} gallery={output / 'index.html'}"
    )
    return manifest


def dry_run(pilot_plan_path: Path = DEFAULT_PILOT_PLAN) -> dict[str, Any]:
    _log("dry-run 1/6: validating the frozen 36-lineage non-final pilot")
    plan = (
        _load_plan(pilot_plan_path, "pilot")
        if Path(pilot_plan_path).is_file()
        else build_pilot_plan()
    )
    _log("dry-run 2/6: verifying the aligned physics-v2 player")
    player = _player()
    _log("dry-run 3/6: materializing one level instance from each generator family")
    with tempfile.TemporaryDirectory(prefix="novphy-issue62-dry-") as temporary:
        root = Path(temporary)
        manifests = []
        for family in GENERATOR_FAMILIES:
            slot = next(
                item for item in plan["lineages"]
                if item["generator_family"] == family["name"]
            )
            authority = _materialize_slot(slot, root / str(family["name"]))
            manifests.append(authority["scenario"].scenario_manifest)
        if len({item.scenario_lineage.identity for item in manifests}) != 2:
            raise SuccessorCohortError("dry-run generator families repeat a lineage")
    _log("dry-run 4/6: exercising the public deployment temporal carrier")
    temporal_carrier_main([
        "--repository-root", str(ROOT),
        "--aligned-root",
        ".local-artifacts/migration-recovery-v1/issue-59-aligned-observation-release",
        "--visual-parser-root",
        ".local-artifacts/migration-recovery-v1/issue-17-visual-parser/parser",
        "--dry-run", "--migration-recovery",
    ])
    _log("dry-run 5/6: verifying VP8/WebM audit encoding support")
    _verify_webm_encoder()
    _log("dry-run 6/6: validating pilot-to-production freeze without writing")
    synthetic = _pilot_report(plan, _synthetic_pilot_records(plan))
    validate_pilot_report(synthetic, pilot_plan=plan)
    production = build_production_plan(
        synthetic,
        pilot_plan=plan,
        maximum_training_lineages=200,
    )
    result = {
        "schema": "issue_62_successor_cohort_dry_run_v1",
        "pilot_plan_identity": plan["identity"],
        "planned_pilot_lineages": len(plan["lineages"]),
        "public_roles": list(PUBLIC_ROLES),
        "generator_families": [item["name"] for item in GENERATOR_FAMILIES],
        "behavior_policies": ["uniform_random", "stratified_bounds"],
        "action_strata": 12,
        "fixed_step_capture_stride": 1,
        "minimum_training_scale": 6,
        "first_larger_training_scale": 200,
        "dry_production_plan_identity": production["identity"],
        "player_source_commit": player["source_snapshot_commit"],
        "pilot_audit_output": str(DEFAULT_PILOT_AUDIT.relative_to(ROOT)),
        "pilot_audit_format": "one VP8/WebM per accepted lineage plus index.html",
        "final_evaluation_opened": False,
        "files_written": False,
        "actual_commands": [
            "python -u -m scripts.run_issue_62_successor_cohort --run-pilot --start-display",
            (
                "python -u -m scripts.run_issue_62_successor_cohort "
                "--freeze-production --maximum-training-lineages 10000"
            ),
            "python -u -m scripts.run_issue_62_successor_cohort --run-production --start-display",
            "python -u -m scripts.run_issue_62_successor_cohort --publish-production",
            "python -u -m scripts.run_issue_62_successor_cohort --validate",
        ],
        "passed": True,
    }
    _log(
        "dry-run complete: multi-observation carrier and pilot/production commands "
        "passed; no files written"
    )
    return result


def _bounds() -> SlingshotActionBounds:
    return SlingshotActionBounds(
        tuple(ACTION_BOUNDS["drag_x"]),
        tuple(ACTION_BOUNDS["drag_y"]),
        tuple(ACTION_BOUNDS["tap_time_ms"]),
        ACTION_BOUNDS["release_time_ms"],
    )


def _terminal_status(state: GameState, exhausted: bool) -> str:
    if state is GameState.WON:
        return "success"
    if state in (GameState.LOST, GameState.EVALUATION_TERMINATED):
        return "failure"
    return "shot_limit" if exhausted else "game_interface_terminal"


def _source_bindings(
    scenario: Any,
    *,
    rollout_identity: str,
    action_identity: str,
) -> dict[str, str]:
    manifest = scenario.scenario_manifest
    return {
        "scenario_template_id": manifest.scenario_template.identity,
        "level_instance_id": manifest.level_instance.identity,
        "scenario_lineage_id": manifest.scenario_lineage.identity,
        "rollout_id": rollout_identity,
        "intervention_id": action_identity,
    }


def _observation_bindings(
    scenario: Any,
    *,
    rollout_identity: str,
) -> dict[str, str]:
    manifest = scenario.scenario_manifest
    return {
        "scenario_template_identity": manifest.scenario_template.identity,
        "level_instance_identity": manifest.level_instance.identity,
        "source_scenario_lineage_identity": manifest.scenario_lineage.identity,
        "rollout_identity": rollout_identity,
    }


def _shot_record(
    shot_root: Path,
    *,
    shot_index: int,
    planned_action: Mapping[str, Any],
    prepared: Any,
    state: GameState,
    derivations: list[Mapping[str, str]],
) -> dict[str, Any]:
    capture = load_physics_capture_v2(shot_root / "physics_capture_v2.json")
    observation = validate_observation_trace(shot_root / "observation-trace")
    engine_action = {
        "schema": "slingshot_relative_intervention_v1",
        "drag_delta_canvas_pixels": [
            planned_action["drag_x"], planned_action["drag_y"]
        ],
        "hold_milliseconds": planned_action["release_time_ms"],
        "tap_time_milliseconds": planned_action["tap_time_ms"],
    }
    action = {
        "identity": planned_action["identity"],
        "legal": True,
        "interface_action": {
            key: value
            for key, value in prepared.action.items()
            if key != "slingshot_reference"
        },
        "engine_relative_action": engine_action,
    }
    return {
        "shot_index": shot_index,
        "path": f"shots/shot-{shot_index:03d}",
        "planned_action_identity": planned_action["identity"],
        "action_stratum": planned_action["action_stratum"],
        "action": action,
        "capture_id": capture.capture_id,
        "shot_id": capture.shot_id,
        "observation_manifest_identity": observation["identity"],
        "frame_count": len(capture.record["frame_records"]),
        "terminal_reason": capture.record["terminal_evidence"]["reason"],
        "game_state_after": state.name,
        "derivations": [
            {**item, "path": f"derivations/{item['path']}"}
            for item in derivations
        ],
    }


def _execute_shot_with_progress(
    shoot: Any,
    aligned_root: Path,
    *,
    lineage_index: int,
    shot_index: int,
) -> Any:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(shoot)
        while True:
            try:
                return future.result(timeout=5.0)
            except FutureTimeout:
                captured = sum(
                    1 for _ in Path(aligned_root).glob("frame_*.json")
                )
                _log(
                    f"lineage={lineage_index} shot={shot_index} "
                    f"captured_frames={captured} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}"
                )


def _trajectory_record(
    attempt_root: Path,
    slot: Mapping[str, Any],
    authority: Mapping[str, Any],
    shot_records: list[dict[str, Any]],
    *,
    release_identity: str,
    final_state: GameState,
) -> dict[str, Any]:
    scenario = authority["scenario"]
    manifest = scenario.scenario_manifest
    transitions = []
    for index, shot in enumerate(shot_records):
        capture = load_physics_capture_v2(
            attempt_root / shot["path"] / "physics_capture_v2.json"
        )
        current_step = capture.record["frame_records"][0]["fixed_step"]
        if index + 1 < len(shot_records):
            target_capture = load_physics_capture_v2(
                attempt_root
                / shot_records[index + 1]["path"]
                / "physics_capture_v2.json"
            )
            next_reference = {
                "shot_index": index + 1,
                "fixed_step": target_capture.record["frame_records"][0]["fixed_step"],
            }
        else:
            next_reference = {
                "shot_index": index,
                "fixed_step": capture.record["frame_records"][-1]["fixed_step"],
            }
        current_reference = {"shot_index": index, "fixed_step": current_step}
        prior_reference = None
        if index:
            previous = load_physics_capture_v2(
                attempt_root / shot_records[index - 1]["path"] / "physics_capture_v2.json"
            )
            prior_reference = {
                "shot_index": index - 1,
                "fixed_step": previous.record["frame_records"][0]["fixed_step"],
            }
        current_observation = validate_observation_trace(
            attempt_root / shot["path"] / "observation-trace"
        )["frame_records"][0]["agent_observation"]["identity"]
        target_manifest = validate_observation_trace(
            attempt_root
            / shot_records[next_reference["shot_index"]]["path"]
            / "observation-trace"
        )
        target_observation = next(
            item["agent_observation"]["identity"]
            for item in target_manifest["frame_records"]
            if item["fixed_step"] == next_reference["fixed_step"]
        )
        transition_identity = successor_identity(
            "issue-62-decision-transition-v1",
            manifest.scenario_lineage.identity,
            index,
            shot["action"]["identity"],
            current_observation,
            target_observation,
        )
        transitions.append({
            "identity": transition_identity,
            "decision_index": index,
            "prior_observation": prior_reference,
            "current_observation": current_reference,
            "next_observation": next_reference,
            "action_identity": shot["action"]["identity"],
            "terminal_status": (
                "ongoing"
                if index + 1 < len(shot_records)
                else _terminal_status(
                    final_state,
                    len(shot_records) == len(slot["planned_actions"]),
                )
            ),
        })
    trajectory_identity = successor_identity(
        "issue-62-decision-trajectory-v1",
        manifest.scenario_lineage.identity,
        tuple(item["identity"] for item in transitions),
    )
    return {
        "schema": TRAJECTORY_SCHEMA,
        "release_identity": release_identity,
        "trajectory_identity": trajectory_identity,
        "slot_identity": slot["slot_identity"],
        "scenario_manifest_identity": scenario.identity,
        "scenario_lineage_identity": manifest.scenario_lineage.identity,
        "level_instance_identity": manifest.level_instance.identity,
        "scenario_template_identity": manifest.scenario_template.identity,
        "exposure_role": slot["exposure_role"],
        "generator_family": slot["generator_family"],
        "generation_seed": slot["generation_seed"],
        "behavior_policy": slot["behavior_policy"],
        "planned_actions": slot["planned_actions"],
        "executed_action_count": len(shot_records),
        "terminal_reason": transitions[-1]["terminal_status"],
        "complete": True,
        "shots": shot_records,
        "transitions": transitions,
    }


def _collect_lineage_attempt(
    slot: Mapping[str, Any],
    attempt_root: Path,
    game: Path,
    *,
    release_identity: str,
    speed: int,
    headless: bool,
) -> dict[str, Any]:
    authority = _materialize_slot(slot, attempt_root)
    scenario = authority["scenario"]
    _install_level(game, authority["xml_path"], slot["slot_identity"])
    agent_port = free_port()
    game_port = free_port()
    physics_port = free_port()
    aligned_root = attempt_root / ".aligned-observation-current"
    os.environ["NOVPHY_PHYSICS_CAPTURE_PORT"] = str(physics_port)
    os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = "1"
    os.environ["NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT"] = str(aligned_root)
    engine = None
    bridge = None
    shot_records = []
    final_state = GameState.UNKNOWN
    try:
        engine = start_engine(
            game,
            headless,
            agent_port=agent_port,
            game_port=game_port,
            physics_port=physics_port,
        )
        log_name = getattr(
            getattr(engine, "novphy_log_file", None), "name", "unknown"
        )
        _log(
            f"engine started pid={engine.pid} slot={slot['slot_identity']} "
            f"engine_log={log_name}"
        )
        bridge = connect_with_retry(
            "127.0.0.1", agent_port, timeout=300, deadline_seconds=60
        )
        bridge.configure(62_000 + int(slot["ordinal"]), PlayingMode.TRAINING)
        bridge.set_speed(speed)
        prepare_for_play(bridge, timeout=60, poll_delay=0.5)
        if bridge.get_current_level() != 1:
            raise SuccessorCohortError(
                "single-level collection runtime did not load index 1"
            )
        for shot_index, planned in enumerate(slot["planned_actions"]):
            state_before = bridge.get_game_state()
            if state_before is not GameState.PLAYING:
                final_state = state_before
                break
            _log(
                f"shot {shot_index + 1}/{len(slot['planned_actions'])} "
                f"policy={slot['behavior_policy']} "
                f"stratum={planned['action_stratum']}"
            )
            action = SlingshotAction(
                int(planned["drag_x"]),
                int(planned["drag_y"]),
                int(planned["tap_time_ms"]),
            )
            prepared = prepare_screen_shot(
                bridge,
                lambda observation, selected=action: selected.to_interface_action(
                    (int(observation["gameX"]), int(observation["gameY"])),
                    _bounds(),
                ),
                frame_height=480,
                execution_speed=speed,
                fast=True,
                record_ground_truth=True,
                ground_truth_frequency=1,
            )
            shot_root = attempt_root / f"shots/shot-{shot_index:03d}"
            rollout_identity = successor_identity(
                "issue-62-shot-rollout-v1", slot["slot_identity"], shot_index
            )
            capture_physics_v2_rollout(
                ScienceBirdsBridge("127.0.0.1", physics_port, timeout=180),
                shot_root,
                shoot=lambda prepared_shot=prepared: _execute_shot_with_progress(
                    prepared_shot.execute,
                    aligned_root,
                    lineage_index=int(slot["ordinal"]) + 1,
                    shot_index=shot_index + 1,
                ),
                source_bindings=_source_bindings(
                    scenario,
                    rollout_identity=rollout_identity,
                    action_identity=str(planned["identity"]),
                ),
                scenario_manifest_identity=scenario.identity,
                deadline_seconds=180,
                aligned_observation_capture_root=aligned_root,
                observation_configuration=OBSERVATION_CONFIGURATION,
                observation_source_bindings=_observation_bindings(
                    scenario, rollout_identity=rollout_identity
                ),
                observation_exposure_role=str(slot["exposure_role"]),
            )
            capture = load_physics_capture_v2(
                shot_root / "physics_capture_v2.json"
            )
            derivations = _write_derivations(
                shot_root / "derivations",
                capture,
                source_reference=f"shots/shot-{shot_index:03d}/physics_capture_v2.json",
                release_identity=release_identity,
            )
            final_state = bridge.get_game_state()
            shot_records.append(_shot_record(
                shot_root,
                shot_index=shot_index,
                planned_action=planned,
                prepared=prepared,
                state=final_state,
                derivations=derivations,
            ))
            _log(
                f"shot {shot_index + 1} complete frames={shot_records[-1]['frame_count']} "
                f"physics_terminal={shot_records[-1]['terminal_reason']} "
                f"game_state={final_state.name}"
            )
            if final_state is not GameState.PLAYING:
                break
    finally:
        if bridge is not None:
            bridge.disconnect()
        stop_started_engine(engine)
    if not shot_records:
        raise SuccessorCohortError("lineage executed no captured action")
    record = _trajectory_record(
        attempt_root,
        slot,
        authority,
        shot_records,
        release_identity=release_identity,
        final_state=final_state,
    )
    write_immutable_cohort_v2_json(record, attempt_root / "trajectory.json")
    load_successor_trajectory(attempt_root, release_identity=release_identity)
    return record


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _interaction_coverage(trajectory_root: Path, record: Mapping[str, Any]) -> list[str]:
    events = set()
    for shot in record["shots"]:
        capture = load_physics_capture_v2(
            trajectory_root / shot["path"] / "physics_capture_v2.json"
        )
        events.update(item["event_type"] for item in capture.record["events"])
    return sorted(events)


def _result_path(runtime_root: Path, slot: Mapping[str, Any]) -> Path:
    return runtime_root / "records" / f"{slot['slot_identity']}.json"


def _accepted_root(runtime_root: Path, phase: str) -> Path:
    if phase == "production":
        return runtime_root / "release-staging/trajectories"
    return runtime_root / "accepted"


def _validate_result(
    result: Mapping[str, Any], slot: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(result)
    if (
        value.get("schema") != "issue_62_lineage_collection_result_v1"
        or value.get("slot_identity") != slot["slot_identity"]
        or value.get("exposure_role") != slot["exposure_role"]
        or value.get("status") not in {"accepted", "failed"}
        or value.get("attempt_count", 0) < 1
    ):
        raise SuccessorCohortError("lineage collection result differs from its slot")
    return value


def run_collection(
    plan: Mapping[str, Any],
    *,
    runtime_root: Path,
    implementation_commit: str,
    start_display_process: bool,
    speed: int,
    headless: bool,
) -> list[dict[str, Any]]:
    plan = validate_successor_plan(plan)
    runtime_root = Path(runtime_root).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    plan_copy = runtime_root / "frozen-collection-plan.json"
    write_immutable_cohort_v2_json(plan, plan_copy)
    if plan_copy.read_bytes() != _canonical_bytes(plan):
        raise SuccessorCohortError("runtime plan copy differs")
    player = _player()
    provenance_path = runtime_root / "provenance.json"
    if provenance_path.exists():
        provenance = _load(provenance_path, "collection provenance")
        if (
            provenance.get("implementation_commit") != implementation_commit
            or provenance.get("player") != player
            or provenance.get("final_evaluation_opened") is not False
        ):
            raise SuccessorCohortError("resumed collection provenance differs")
    else:
        write_immutable_cohort_v2_json({
            "schema": "issue_62_collection_provenance_v1",
            "implementation_commit": implementation_commit,
            "player": player,
            "collected_at": _now(),
            "final_evaluation_opened": False,
        }, provenance_path)
    if plan["phase"] == "production":
        staging = runtime_root / "release-staging"
        staging.mkdir(parents=True, exist_ok=True)
        write_immutable_cohort_v2_json(plan, staging / "production-plan.json")
    game = runtime_root / "game-runtime"
    if not game.exists():
        _log("extracting the accepted aligned-observation player once")
        archive_details(STAGE_ROOT, game)
    release_identity = release_identity_for_plan(plan["identity"])
    display_process = None
    prior_display = os.environ.get("DISPLAY")
    prior_stride = os.environ.get("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE")
    prior_aligned = os.environ.get("NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT")
    prior_physics_port = os.environ.get("NOVPHY_PHYSICS_CAPTURE_PORT")
    records = []
    try:
        if start_display_process:
            display, display_process = start_display(runtime_root / "display.log")
            os.environ["DISPLAY"] = display
            _log(f"display started DISPLAY={display}")
        total = len(plan["lineages"])
        for index, slot in enumerate(plan["lineages"], start=1):
            result_path = _result_path(runtime_root, slot)
            if result_path.exists():
                result = _validate_result(
                    _load(result_path, "lineage collection result"), slot
                )
                records.append(result)
                _log(
                    f"resume {index}/{total}: status={result['status']} "
                    f"role={slot['exposure_role']} slot={slot['slot_identity']}"
                )
                continue
            _log(
                f"lineage {index}/{total} start role={slot['exposure_role']} "
                f"family={slot['generator_family']} policy={slot['behavior_policy']} "
                f"seed={slot['generation_seed']}"
            )
            failures = []
            trajectory_record = None
            accepted_path = (
                _accepted_root(runtime_root, plan["phase"])
                / slot["exposure_role"]
                / slot["slot_identity"]
            )
            started = time.monotonic()
            for attempt_number in range(1, plan["fixed_retry_limit"] + 1):
                attempt_root = (
                    runtime_root / "attempts" / slot["slot_identity"]
                    / f"attempt-{attempt_number:02d}"
                )
                try:
                    trajectory_record = _collect_lineage_attempt(
                        slot,
                        attempt_root,
                        game,
                        release_identity=release_identity,
                        speed=speed,
                        headless=headless,
                    )
                    accepted_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(attempt_root, accepted_path)
                    break
                except Exception as error:
                    failure = {
                        "attempt_number": attempt_number,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                    failures.append(failure)
                    _log(
                        f"lineage {index}/{total} attempt {attempt_number}/"
                        f"{plan['fixed_retry_limit']} failed: "
                        f"{failure['error_type']}: {failure['message']}"
                    )
            wall_seconds = time.monotonic() - started
            if trajectory_record is None:
                result = {
                    "schema": "issue_62_lineage_collection_result_v1",
                    "slot_identity": slot["slot_identity"],
                    "status": "failed",
                    "exposure_role": slot["exposure_role"],
                    "generator_family": slot["generator_family"],
                    "behavior_policy": slot["behavior_policy"],
                    "attempt_count": len(failures),
                    "failures": failures,
                    "wall_seconds": wall_seconds,
                    "outcome_conditioned_replacement": False,
                }
            else:
                result = {
                    "schema": "issue_62_lineage_collection_result_v1",
                    "slot_identity": slot["slot_identity"],
                    "status": "accepted",
                    "exposure_role": slot["exposure_role"],
                    "generator_family": slot["generator_family"],
                    "behavior_policy": slot["behavior_policy"],
                    "generation_seed": slot["generation_seed"],
                    "scenario_lineage_identity": trajectory_record[
                        "scenario_lineage_identity"
                    ],
                    "level_instance_identity": trajectory_record[
                        "level_instance_identity"
                    ],
                    "scenario_template_identity": trajectory_record[
                        "scenario_template_identity"
                    ],
                    "trajectory_identity": trajectory_record["trajectory_identity"],
                    "executed_action_count": trajectory_record[
                        "executed_action_count"
                    ],
                    "decision_state_count": trajectory_record[
                        "executed_action_count"
                    ] + 1,
                    "frame_record_count": sum(
                        item["frame_count"] for item in trajectory_record["shots"]
                    ),
                    "terminal_reason": trajectory_record["terminal_reason"],
                    "action_strata": [
                        item["action_stratum"] for item in trajectory_record["shots"]
                    ],
                    "interaction_coverage": _interaction_coverage(
                        accepted_path, trajectory_record
                    ),
                    "wall_seconds": wall_seconds,
                    "artifact_bytes": _directory_bytes(accepted_path),
                    "attempt_count": len(failures) + 1,
                    "failures": failures,
                    "outcome_conditioned_replacement": False,
                }
            write_immutable_cohort_v2_json(result, result_path)
            records.append(result)
            _log(
                f"lineage {index}/{total} complete status={result['status']} "
                f"shots={result.get('executed_action_count', 0)} "
                f"frames={result.get('frame_record_count', 0)} "
                f"terminal={result.get('terminal_reason', 'unavailable')} "
                f"wall={wall_seconds:.1f}s"
            )
        return records
    finally:
        if display_process is not None:
            _log(f"display stopped result={terminate(display_process)}")
        if prior_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = prior_display
        if prior_stride is None:
            os.environ.pop("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", None)
        else:
            os.environ["NOVPHY_PHYSICS_CAPTURE_V2_STRIDE"] = prior_stride
        if prior_aligned is None:
            os.environ.pop("NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT", None)
        else:
            os.environ["NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT"] = prior_aligned
        if prior_physics_port is None:
            os.environ.pop("NOVPHY_PHYSICS_CAPTURE_PORT", None)
        else:
            os.environ["NOVPHY_PHYSICS_CAPTURE_PORT"] = prior_physics_port


def run_pilot(
    *,
    plan_path: Path,
    runtime_root: Path,
    report_path: Path,
    audit_output: Path,
    implementation_commit: str,
    start_display_process: bool,
    speed: int,
    headless: bool,
) -> dict[str, Any]:
    plan = _load_plan(plan_path, "pilot")
    records = run_collection(
        plan,
        runtime_root=runtime_root,
        implementation_commit=implementation_commit,
        start_display_process=start_display_process,
        speed=speed,
        headless=headless,
    )
    report = _pilot_report(plan, records)
    write_immutable_cohort_v2_json(report, report_path)
    audit = write_pilot_audit(
        plan,
        records,
        report,
        runtime_root=runtime_root,
        output=audit_output,
    )
    _log(
        f"pilot complete accepted={report['accepted_lineage_count']}/"
        f"{report['planned_lineage_count']} passed={report['passed']} "
        f"report={report['identity']}"
    )
    return {
        **report,
        "audit": {
            "manifest_identity": audit["identity"],
            "output": str(Path(audit_output).resolve()),
            "gallery": str((Path(audit_output).resolve() / "index.html")),
            "video_count": audit["video_count"],
        },
    }


def freeze_production(
    *,
    pilot_plan_path: Path,
    pilot_report_path: Path,
    production_plan_path: Path,
    maximum_training_lineages: int,
) -> dict[str, Any]:
    pilot_plan = _load_plan(pilot_plan_path, "pilot")
    pilot_report = validate_pilot_report(
        _load(pilot_report_path, "pilot report"), pilot_plan=pilot_plan
    )
    plan = build_production_plan(
        pilot_report,
        pilot_plan=pilot_plan,
        maximum_training_lineages=maximum_training_lineages,
    )
    write_immutable_cohort_v2_json(plan, production_plan_path)
    _log(
        f"production plan frozen identity={plan['identity']} "
        f"training={plan['role_counts']['training']} "
        f"calibration={plan['role_counts']['calibration']} "
        f"model_selection={plan['role_counts']['model_selection']} "
        f"estimated_hours={plan['resource_decision']['estimated_collection_hours']:.1f}"
    )
    return plan


def _all_results(
    runtime_root: Path, plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        _validate_result(
            _load(_result_path(runtime_root, slot), "lineage result"), slot
        )
        for slot in plan["lineages"]
    ]


def _lineage_manifest_identity(
    release_identity: str, trajectories: list[Any]
) -> str:
    bindings = tuple(
        TrajectoryLineageBinding(
            trajectory_identity=item.identity,
            scenario_lineage_identity=item.scenario_lineage_identity,
            exposure_role=item.exposure_role,
            transition_identities=tuple(
                transition.identity for transition in item.transitions
            ),
            initial_observation_identity=item.transitions[0].current_observation.identity,
            terminal_observation_identity=(
                item.transitions[-1].targets.next_observation.identity
            ),
        )
        for item in trajectories
    )
    return TrajectoryLineageManifest.create(release_identity, bindings).identity


def publish_production(
    *,
    plan_path: Path,
    runtime_root: Path,
    output: Path,
    summary_path: Path,
) -> dict[str, Any]:
    plan = _load_plan(plan_path, "production")
    runtime_root = Path(runtime_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise SuccessorCohortError("immutable successor release already exists")
    results = _all_results(runtime_root, plan)
    failed = [item for item in results if item["status"] != "accepted"]
    if failed:
        raise SuccessorCohortError(
            f"cannot publish: {len(failed)} frozen lineages exhausted retries"
        )
    staging = runtime_root / "release-staging"
    if (staging / "production-plan.json").read_bytes() != Path(plan_path).read_bytes():
        raise SuccessorCohortError("release staging plan differs")
    release_identity = release_identity_for_plan(plan["identity"])
    trajectories_by_role = {role: [] for role in PUBLIC_ROLES}
    release_records = []
    for result in results:
        relative = (
            Path("trajectories") / result["exposure_role"] / result["slot_identity"]
        )
        trajectory = load_successor_trajectory(
            staging / relative, release_identity=release_identity
        )
        trajectories_by_role[result["exposure_role"]].append(trajectory)
        release_records.append({
            "slot_identity": result["slot_identity"],
            "trajectory_identity": result["trajectory_identity"],
            "scenario_lineage_identity": result["scenario_lineage_identity"],
            "level_instance_identity": result["level_instance_identity"],
            "scenario_template_identity": result["scenario_template_identity"],
            "exposure_role": result["exposure_role"],
            "generator_family": result["generator_family"],
            "behavior_policy": result["behavior_policy"],
            "terminal_reason": result["terminal_reason"],
            "executed_action_count": result["executed_action_count"],
            "decision_state_count": result["decision_state_count"],
            "frame_record_count": result["frame_record_count"],
            "path": relative.as_posix(),
        })
    if (
        len({item["scenario_lineage_identity"] for item in release_records})
        != len(release_records)
        or len({item["level_instance_identity"] for item in release_records})
        != len(release_records)
    ):
        raise SuccessorCohortError(
            "production level instances or scenario lineages cross exposure roles"
        )
    lineage_manifests = {
        role: _lineage_manifest_identity(
            release_identity, trajectories_by_role[role]
        )
        for role in PUBLIC_ROLES
    }
    training_by_slot = {
        item["slot_identity"]: item["scenario_lineage_identity"]
        for item in release_records
        if item["exposure_role"] == "training"
    }
    nested_scales = [
        {
            **scale,
            "scenario_lineage_identities": [
                training_by_slot[item] for item in scale["slot_identities"]
            ],
        }
        for scale in plan["nested_training_scales"]
    ]
    manifest = {
        "schema": RELEASE_SCHEMA,
        "identity": release_identity,
        "production_plan_identity": plan["identity"],
        "included_roles": list(PUBLIC_ROLES),
        "final_evaluation_collected": False,
        "role_counts": dict(plan["role_counts"]),
        "trajectory_count": len(release_records),
        "lineage_manifests": lineage_manifests,
        "nested_training_scales": nested_scales,
        "trajectories": release_records,
        "counts": {
            "independent_lineages": len(release_records),
            "level_instances": len(release_records),
            "decision_transitions": sum(
                item["executed_action_count"] for item in release_records
            ),
            "decision_states": sum(
                item["decision_state_count"] for item in release_records
            ),
            "frame_records": sum(
                item["frame_record_count"] for item in release_records
            ),
            "by_generator_family": dict(sorted(Counter(
                item["generator_family"] for item in release_records
            ).items())),
            "by_behavior_policy": dict(sorted(Counter(
                item["behavior_policy"] for item in release_records
            ).items())),
            "by_terminal_reason": dict(sorted(Counter(
                item["terminal_reason"] for item in release_records
            ).items())),
            "by_action_stratum": dict(sorted(Counter(
                value for item in results for value in item["action_strata"]
            ).items())),
        },
        "failure_accounting": {
            "scheduled": len(results),
            "accepted": len(results),
            "failed": 0,
            "outcome_conditioned_replacement": False,
        },
        "passed": True,
    }
    write_immutable_cohort_v2_json(manifest, staging / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)
    summary = {
        "schema": "multi_shot_successor_cohort_summary_v1",
        "artifact_identity": release_identity,
        "production_plan_identity": plan["identity"],
        "role_counts": dict(plan["role_counts"]),
        "counts": manifest["counts"],
        "nested_training_scale_counts": [
            item["lineage_count"] for item in nested_scales
        ],
        "final_evaluation_collected": False,
        "rerun_commands": [
            "python -u -m scripts.run_issue_62_successor_cohort --dry-run",
            "python -u -m scripts.run_issue_62_successor_cohort --run-production --start-display",
            "python -u -m scripts.run_issue_62_successor_cohort --publish-production",
            "python -u -m scripts.run_issue_62_successor_cohort --validate",
        ],
    }
    write_immutable_cohort_v2_json(summary, summary_path)
    _log(
        f"publication complete release={release_identity} "
        f"lineages={manifest['trajectory_count']} "
        f"transitions={manifest['counts']['decision_transitions']}"
    )
    return summary


def validate_release(
    *,
    output: Path = DEFAULT_RELEASE,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    root = Path(output).resolve()
    manifest = _load(root / "manifest.json", "successor release manifest")
    plan = _load_plan(root / "production-plan.json", "production")
    if (
        manifest.get("identity") != release_identity_for_plan(plan["identity"])
        or manifest.get("role_counts") != plan["role_counts"]
        or manifest.get("trajectory_count") != len(plan["lineages"])
        or manifest.get("failure_accounting", {}).get("failed") != 0
    ):
        raise SuccessorCohortError("successor release manifest differs")
    readers = tuple(
        SuccessorCohortReader(root, exposure_role=role) for role in PUBLIC_ROLES
    )
    SuccessorCohortReader.validate_role_isolation(readers)
    training_slots = [
        item["slot_identity"] for item in plan["lineages"]
        if item["exposure_role"] == "training"
    ]
    for scale in manifest["nested_training_scales"]:
        if scale["slot_identities"] != training_slots[:scale["lineage_count"]]:
            raise SuccessorCohortError("release nested training scale differs")
    summary = _load(summary_path, "successor compact summary")
    if (
        summary.get("artifact_identity") != manifest["identity"]
        or summary.get("counts") != manifest["counts"]
        or summary.get("final_evaluation_collected") is not False
    ):
        raise SuccessorCohortError("successor compact summary differs")
    DeploymentTrajectoryReader.validate_role_isolation(tuple(
        reader.trajectory_reader for reader in readers
    ))
    _log(
        f"validate: exact issue-62 release passed "
        f"lineages={manifest['trajectory_count']} "
        f"transitions={manifest['counts']['decision_transitions']} "
        "final_evaluation=unopened"
    )
    return summary


def _implementation_commit(value: str | None, *, required: bool) -> str:
    if value:
        return value
    revision, dirty = git_revision(str(ROOT))
    if dirty and required:
        raise SuccessorCohortError(
            "actual collection from a dirty tree requires --implementation-commit"
        )
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-pilot-plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-pilot", action="store_true")
    mode.add_argument("--freeze-production", action="store_true")
    mode.add_argument("--run-production", action="store_true")
    mode.add_argument("--publish-production", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    parser.add_argument("--pilot-runtime", type=Path, default=DEFAULT_PILOT_RUNTIME)
    parser.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument("--pilot-audit-output", type=Path, default=DEFAULT_PILOT_AUDIT)
    parser.add_argument(
        "--production-plan", type=Path, default=DEFAULT_PRODUCTION_PLAN
    )
    parser.add_argument(
        "--production-runtime", type=Path, default=DEFAULT_PRODUCTION_RUNTIME
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--maximum-training-lineages", type=int, default=10_000)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--start-display", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.freeze_pilot_plan:
        result = freeze_pilot_plan(args.pilot_plan)
    elif args.dry_run:
        result = dry_run(args.pilot_plan)
    elif args.run_pilot:
        result = run_pilot(
            plan_path=args.pilot_plan,
            runtime_root=args.pilot_runtime,
            report_path=args.pilot_report,
            audit_output=args.pilot_audit_output,
            implementation_commit=_implementation_commit(
                args.implementation_commit, required=True
            ),
            start_display_process=args.start_display,
            speed=args.speed,
            headless=args.headless,
        )
    elif args.freeze_production:
        result = freeze_production(
            pilot_plan_path=args.pilot_plan,
            pilot_report_path=args.pilot_report,
            production_plan_path=args.production_plan,
            maximum_training_lineages=args.maximum_training_lineages,
        )
    elif args.run_production:
        plan = _load_plan(args.production_plan, "production")
        records = run_collection(
            plan,
            runtime_root=args.production_runtime,
            implementation_commit=_implementation_commit(
                args.implementation_commit, required=True
            ),
            start_display_process=args.start_display,
            speed=args.speed,
            headless=args.headless,
        )
        result = {
            "schema": "issue_62_production_collection_status_v1",
            "production_plan_identity": plan["identity"],
            "scheduled": len(records),
            "accepted": sum(item["status"] == "accepted" for item in records),
            "failed": sum(item["status"] == "failed" for item in records),
            "final_evaluation_opened": False,
        }
    elif args.publish_production:
        result = publish_production(
            plan_path=args.production_plan,
            runtime_root=args.production_runtime,
            output=args.output,
            summary_path=args.summary,
        )
    else:
        result = validate_release(output=args.output, summary_path=args.summary)
    displayed = result
    if result.get("schema") == "multi_shot_successor_collection_plan_v1":
        displayed = {
            "schema": result["schema"],
            "identity": result["identity"],
            "phase": result["phase"],
            "role_counts": result["role_counts"],
            "lineage_count": len(result["lineages"]),
            "nested_training_scale_counts": [
                item["lineage_count"]
                for item in result["nested_training_scales"]
            ],
            "final_evaluation_opened": False,
        }
    print(json.dumps(displayed, allow_nan=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, SuccessorCohortError, ValueError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
