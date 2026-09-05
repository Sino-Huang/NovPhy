"""Freeze, collect, publish, and validate issue #68's corrective cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Final

import torch

from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_bytes
from scripts.observation_trace import validate_observation_trace
from scripts.run_issue_62_successor_cohort import (
    DEFAULT_RELEASE as DEFAULT_ISSUE_62_RELEASE,
    STAGE_ROOT,
    _collect_lineage_attempt,
    _directory_bytes,
    _encode_agent_frames_webm,
    _interaction_coverage,
    _materialize_slot,
    _pilot_audit_frames,
    _player,
    _verify_webm_encoder,
)
from scripts.run_issue_63_matched_experiment import (
    _build_lineage_pair,
    _load_deployment_adapter,
    _observation,
    _realized_goal_cost,
)
from scripts.smoke_physics_capture import (
    archive_details,
    start_display,
    terminate,
)
from world_model.data.corrective_ranking_cohort import (
    DEFAULT_PRODUCTION_STATES_PER_ROLE,
    FAILURE_COST,
    ISSUE_63_DISCRIMINATING_STATE_FRACTION,
    MINIMUM_DISCRIMINATION_IMPROVEMENT,
    PILOT_REPORT_SCHEMA,
    PLAN_SCHEMA,
    RELEASE_SCHEMA,
    ROLES,
    CorrectiveRankingCohortError,
    action_bounds,
    build_pilot_plan,
    build_production_plan,
    realized_endpoint_cost,
    release_identity,
    validate_pilot_report,
    validate_plan,
)
from world_model.data.deployment_temporal import TemporalObservationContext
from world_model.data.successor_cohort import (
    _load_shot,
    load_successor_trajectory,
)
from world_model.planning.gameplay import SlingshotAction
from world_model.training.action_ranking_probe import summarize_ranking_diversity
from world_model.training.cohort_v2_micro import CohortV2StateCodec
from world_model.training.lineage_scaling import (
    ActionCandidate,
    ActionRankingState,
    CarrierKind,
    CarrierLineage,
    LineageScalingError,
    load_action_ranking_bundle,
    load_carrier_lineage_bundle,
    save_action_ranking_bundle,
    save_carrier_lineage_bundle,
)
from world_model.training.manifest import git_revision


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_ROOT: Final = ROOT / ".local-artifacts/issue-68-corrective-cohort-v2"
DEFAULT_PILOT_PLAN: Final = DEFAULT_ROOT / "pilot-plan.json"
DEFAULT_PILOT_RUNTIME: Final = DEFAULT_ROOT / "pilot-run"
DEFAULT_PILOT_REPORT: Final = DEFAULT_ROOT / "pilot-report.json"
DEFAULT_PRODUCTION_PLAN: Final = DEFAULT_ROOT / "production-plan.json"
DEFAULT_PRODUCTION_RUNTIME: Final = DEFAULT_ROOT / "production-run"
DEFAULT_RELEASE: Final = DEFAULT_ROOT / "release"
DEFAULT_AUDIT: Final = ROOT / "data/issue-68-ranking-audit-v2"
DEFAULT_SUMMARY: Final = (
    ROOT / "data/runtime_evidence/issue-68/corrective-cohort-summary-v2.json"
)
DEFAULT_ISSUE_57_PROTOCOL: Final = (
    ROOT / "data/runtime_evidence/issue-57/"
    "cohort-v2-gameplay-success-protocol-v2.json"
)
DEFAULT_WORKERS: Final = 4
FRAME_HEIGHT: Final = 480


def _log(message: str) -> None:
    print(f"[issue-68] {message}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CorrectiveRankingCohortError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise CorrectiveRankingCohortError(f"{label} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    encoded = _canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != encoded:
            raise CorrectiveRankingCohortError(f"immutable JSON differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_plan(path: Path, phase: str) -> dict[str, Any]:
    plan = validate_plan(_load_json(path, f"{phase} plan"), phase=phase)
    if Path(path).read_bytes() != _canonical_bytes(plan):
        raise CorrectiveRankingCohortError(f"{phase} plan is not canonical")
    return plan


def _state_key(state: Mapping[str, Any]) -> str:
    role = "cal" if state["exposure_role"] == "calibration" else "ms"
    return f"{role}-s{int(state['role_ordinal']) + 1:04d}"


def _candidate_key(
    state: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    return f"{_state_key(state)}-c{int(candidate['ordinal']):02d}"


def _candidate_slot(
    state: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    key = _candidate_key(state, candidate)
    return {
        "phase": state["identity"].split("-")[2],
        "exposure_role": state["exposure_role"],
        "ordinal": int(state["ordinal"]) * 12 + int(candidate["ordinal"]) - 1,
        "role_ordinal": state["role_ordinal"],
        "generator_family": state["generator_family"],
        "generation_seed": state["generation_seed"],
        "behavior_policy": "issue_68_broad_action_design",
        "slot_identity": f"issue-68-{key}-candidate-rollout",
        "planned_actions": [{
            "identity": candidate["identity"],
            "action_stratum": candidate["action_stratum"],
            "selection_mode": "frozen_relative_drag",
            "drag_x": candidate["drag_x"],
            "drag_y": candidate["drag_y"],
            "tap_time_ms": candidate["tap_time_ms"],
        }],
    }


def _result_path(
    runtime: Path, state: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Path:
    return Path(runtime) / "records" / f"{_candidate_key(state, candidate)}.json"


def _trajectory_relative(
    state: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Path:
    return (
        Path("trajectories")
        / str(state["exposure_role"])
        / _state_key(state)
        / f"candidate-{int(candidate['ordinal']):02d}"
    )


def _accepted_root(
    runtime: Path,
    phase: str,
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Path:
    base = (
        Path(runtime) / "release-staging"
        if phase == "production"
        else Path(runtime) / "accepted"
    )
    return base / _trajectory_relative(state, candidate)


def _attempt_root(
    runtime: Path,
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    attempt: int,
) -> Path:
    return (
        Path(runtime)
        / "attempts"
        / _candidate_key(state, candidate)
        / f"attempt-{attempt:02d}"
    )


def _action_tensor(candidate: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(
        (
            float(candidate["drag_x"]) / FRAME_HEIGHT,
            float(candidate["drag_y"]) / FRAME_HEIGHT,
            action_bounds().release_time_ms / 1000.0,
            float(candidate["tap_time_ms"]) / 1000.0,
            1.0,
        ),
        dtype=torch.float32,
    )


def _result_outcome(
    trajectory_root: Path,
    *,
    release: str,
    role: str,
) -> tuple[dict[str, Any], str, list[str]]:
    raw = _load_json(trajectory_root / "trajectory.json", "candidate trajectory")
    shot = _load_shot(
        trajectory_root,
        raw["shots"][-1],
        release_identity=release,
        role=role,
    )
    initial = shot["frames"][min(shot["frames"])]
    endpoint = shot["frames"][max(shot["frames"])]
    coverage = _interaction_coverage(trajectory_root, raw)

    def active_positions(frame: Any, prefix: str) -> dict[str, tuple[float, float]]:
        positions = {}
        for entity in frame.engine_state["entities"]:
            scenario_id = str(entity.get("scenario_object_id", ""))
            body = entity.get("body")
            if (
                entity.get("lifecycle") == "active"
                and scenario_id.startswith(prefix)
                and isinstance(body, Mapping)
                and isinstance(body.get("position"), tuple)
            ):
                x, y = body["position"]
                positions[scenario_id] = (float(x), float(y))
        return positions

    initial_pigs = active_positions(initial, "pig:")
    endpoint_pigs = active_positions(endpoint, "pig:")
    initial_blocks = active_positions(initial, "block:")
    endpoint_blocks = active_positions(endpoint, "block:")

    def displacement(
        before: Mapping[str, tuple[float, float]],
        after: Mapping[str, tuple[float, float]],
    ) -> float:
        return sum(
            math.hypot(after[key][0] - value[0], after[key][1] - value[1])
            for key, value in before.items()
            if key in after
        )

    active_pigs = len(endpoint_pigs)
    active_blocks = len(endpoint_blocks)
    pig_displacement = displacement(initial_pigs, endpoint_pigs)
    block_displacement = displacement(initial_blocks, endpoint_blocks)
    pig_contact = "collision:bird:pig" in coverage
    block_contact = "collision:bird:block" in coverage
    support_change = "non_bird_support_change" in coverage
    cost, progress = realized_endpoint_cost(
        active_pigs=active_pigs,
        active_blocks=active_blocks,
        pig_contact=pig_contact,
        block_contact=block_contact,
        support_change=support_change,
        pig_displacement=pig_displacement,
        block_displacement=block_displacement,
    )
    count_cost = int(_realized_goal_cost(endpoint))
    if count_cost != active_pigs * 1000 + active_blocks:
        raise CorrectiveRankingCohortError("endpoint count cost differs")
    return (
        {
            "realized_cost": cost,
            "goal_count_cost": count_cost,
            "active_pigs": active_pigs,
            "active_blocks": active_blocks,
            "pig_contact": pig_contact,
            "block_contact": block_contact,
            "support_change": support_change,
            "pig_displacement_world": pig_displacement,
            "block_displacement_world": block_displacement,
            "bounded_progress": progress,
        },
        str(raw["terminal_reason"]),
        coverage,
    )


def _validate_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    trajectory_root: Path | None,
) -> dict[str, Any]:
    value = dict(result)
    expected = {
        "schema": "issue_68_candidate_execution_result_v2",
        "plan_identity": plan["identity"],
        "state_identity": state["identity"],
        "candidate_identity": candidate["identity"],
        "candidate_ordinal": candidate["ordinal"],
        "exposure_role": state["exposure_role"],
        "generator_family": state["generator_family"],
        "generation_seed": state["generation_seed"],
        "action_stratum": candidate["action_stratum"],
        "interface_action": {
            "drag_x": candidate["drag_x"],
            "drag_y": candidate["drag_y"],
            "tap_time_ms": candidate["tap_time_ms"],
        },
        "realized_cost_contract_identity": plan["realized_cost_contract"][
            "identity"
        ],
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise CorrectiveRankingCohortError("candidate result differs from its plan")
    if (
        value.get("status") not in {"accepted", "failed"}
        or type(value.get("attempt_count")) is not int
        or not 1 <= value["attempt_count"] <= plan["fixed_retry_limit"]
        or value.get("outcome_conditioned_membership") is not False
        or value.get("final_evaluation_opened") is not False
    ):
        raise CorrectiveRankingCohortError("candidate result accounting differs")
    if value["status"] == "failed":
        if (
            value.get("realized_cost") != FAILURE_COST
            or value.get("goal_count_cost") is not None
            or value.get("endpoint_outcome") is not None
            or value.get("trajectory_relative_path") is not None
            or value["attempt_count"] != plan["fixed_retry_limit"]
        ):
            raise CorrectiveRankingCohortError("failed candidate treatment differs")
        return value
    if trajectory_root is None or not (trajectory_root / "trajectory.json").is_file():
        raise CorrectiveRankingCohortError("accepted candidate trajectory is absent")
    trajectory = load_successor_trajectory(
        trajectory_root, release_identity=release_identity(str(plan["identity"]))
    )
    raw = _load_json(trajectory_root / "trajectory.json", "candidate trajectory")
    planned = raw["planned_actions"]
    raw_shots = raw.get("shots", ())
    executed = (
        raw_shots[0].get("action", {}).get("engine_relative_action", {})
        if len(raw_shots) == 1
        else {}
    )
    if (
        raw.get("exposure_role") != state["exposure_role"]
        or raw.get("generator_family") != state["generator_family"]
        or raw.get("generation_seed") != state["generation_seed"]
        or len(planned) != 1
        or planned[0]["identity"] != candidate["identity"]
        or raw_shots[0].get("action_stratum") != candidate["action_stratum"]
        or executed.get("drag_delta_canvas_pixels")
        != [candidate["drag_x"], candidate["drag_y"]]
        or executed.get("hold_milliseconds") != action_bounds().release_time_ms
        or executed.get("tap_time_milliseconds") != candidate["tap_time_ms"]
        or trajectory.exposure_role != state["exposure_role"]
        or value.get("trajectory_relative_path")
        != _trajectory_relative(state, candidate).as_posix()
        or value.get("scenario_lineage_identity")
        != raw["scenario_lineage_identity"]
        or value.get("level_instance_identity") != raw["level_instance_identity"]
        or value.get("trajectory_identity") != raw["trajectory_identity"]
        or type(value.get("realized_cost")) not in (int, float)
        or not 0 <= float(value["realized_cost"]) < FAILURE_COST
        or type(value.get("goal_count_cost")) is not int
        or math.floor(float(value["realized_cost"]))
        != value.get("goal_count_cost")
        or not isinstance(value.get("endpoint_outcome"), Mapping)
        or value["endpoint_outcome"].get("goal_count_cost")
        != value.get("goal_count_cost")
        or value["endpoint_outcome"].get("realized_cost")
        != value.get("realized_cost")
    ):
        raise CorrectiveRankingCohortError("accepted candidate binding differs")
    outcome, terminal_reason, interaction_coverage = _result_outcome(
        trajectory_root,
        release=release_identity(str(plan["identity"])),
        role=str(state["exposure_role"]),
    )
    if (
        value.get("endpoint_outcome") != outcome
        or value.get("terminal_reason") != terminal_reason
        or value.get("interaction_coverage") != interaction_coverage
    ):
        raise CorrectiveRankingCohortError(
            "candidate result changed its realized endpoint"
        )
    return value


def _attempt_audit(
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    attempt_root: Path,
    *,
    output: Path,
    attempt: int,
    status: str,
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    relative_base = (
        Path(str(state["exposure_role"]))
        / _state_key(state)
        / f"candidate-{int(candidate['ordinal']):02d}-attempt-{attempt:02d}-{status}"
    )
    manifest_relative = relative_base.with_suffix(".json")
    manifest_path = output / manifest_relative
    if manifest_path.is_file():
        existing = _load_json(manifest_path, "candidate frame audit")
        video = existing.get("video_path")
        if (
            existing.get("state_identity") != state["identity"]
            or existing.get("candidate_identity") != candidate["identity"]
            or existing.get("attempt") != attempt
            or existing.get("status") != status
            or (video is not None and not (output / str(video)).is_file())
        ):
            raise CorrectiveRankingCohortError("candidate frame audit differs")
        return {**existing, "manifest_path": manifest_relative.as_posix()}
    frames: list[Path] = []
    shot_ranges: list[dict[str, Any]] = []
    if (attempt_root / "trajectory.json").is_file():
        trajectory = _load_json(attempt_root / "trajectory.json", "audit trajectory")
        frames, shot_ranges = _pilot_audit_frames(attempt_root, trajectory)
    else:
        for observation_root in sorted(
            attempt_root.glob("shots/shot-*/observation-trace")
        ):
            observation = validate_observation_trace(observation_root)
            frames.extend(
                observation_root / item["agent_observation"]["relative_path"]
                for item in observation["frame_records"]
            )
        stalled = sorted(
            (attempt_root / ".aligned-observation-current").glob("frame_*.png")
        )
        if len(stalled) > 250:
            stalled = [
                stalled[round(index * (len(stalled) - 1) / 249)]
                for index in range(250)
            ]
        frames.extend(stalled)
    video_relative = relative_base.with_suffix(".webm") if frames else None
    if video_relative is not None:
        _encode_agent_frames_webm(frames, output / video_relative)
    manifest = {
        "schema": "issue_68_candidate_frame_audit_v2",
        "state_identity": state["identity"],
        "candidate_identity": candidate["identity"],
        "candidate_ordinal": candidate["ordinal"],
        "action_stratum": candidate["action_stratum"],
        "exposure_role": state["exposure_role"],
        "attempt": attempt,
        "status": status,
        "failure": None if failure is None else dict(failure),
        "source_observation_role": "agent",
        "canonical_observations_included": False,
        "video_container": "webm" if video_relative is not None else None,
        "video_codec": "vp8" if video_relative is not None else None,
        "playback_fps": 50,
        "frame_count": len(frames),
        "shot_ranges": shot_ranges,
        "video_path": (
            None if video_relative is None else video_relative.as_posix()
        ),
    }
    _write_json(output / manifest_relative, manifest)
    _log(
        f"audit state={_state_key(state)} candidate={candidate['ordinal']}/12 "
        f"attempt={attempt} status={status} frames={len(frames)} "
        f"video={manifest['video_path'] or 'unavailable'}"
    )
    return {**manifest, "manifest_path": manifest_relative.as_posix()}


def _collect_candidate(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    runtime: Path,
    audit: Path,
    game: Path,
    speed: int,
    headless: bool,
    engine_start_lock: Path,
) -> dict[str, Any]:
    result_path = _result_path(runtime, state, candidate)
    accepted_root = _accepted_root(
        runtime, str(plan["phase"]), state, candidate
    )
    if result_path.is_file():
        return _validate_result(
            _load_json(result_path, "candidate result"),
            plan,
            state,
            candidate,
            trajectory_root=accepted_root,
        )
    failures = []
    record = None
    successful_attempt = None
    final_audit = None
    started = time.monotonic()
    if (accepted_root / "trajectory.json").is_file():
        record = _load_json(accepted_root / "trajectory.json", "recovered trajectory")
        load_successor_trajectory(
            accepted_root, release_identity=release_identity(str(plan["identity"]))
        )
        successful_attempt = 1
        final_audit = _attempt_audit(
            state,
            candidate,
            accepted_root,
            output=audit,
            attempt=successful_attempt,
            status="accepted",
            failure=None,
        )
    else:
        for attempt in range(1, int(plan["fixed_retry_limit"]) + 1):
            attempt_root = _attempt_root(runtime, state, candidate, attempt)
            failure = None
            if attempt_root.exists():
                if (attempt_root / "trajectory.json").is_file():
                    try:
                        record = _load_json(
                            attempt_root / "trajectory.json", "recovered trajectory"
                        )
                        load_successor_trajectory(
                            attempt_root,
                            release_identity=release_identity(str(plan["identity"])),
                        )
                    except Exception as error:
                        record = None
                        failure = {
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                else:
                    failure = {
                        "attempt": attempt,
                        "error_type": "InterruptedAttempt",
                        "message": "attempt directory has no atomic complete trajectory",
                    }
            else:
                try:
                    record = _collect_lineage_attempt(
                        _candidate_slot(state, candidate),
                        attempt_root,
                        game,
                        release_identity=release_identity(str(plan["identity"])),
                        speed=speed,
                        headless=headless,
                        engine_start_lock=engine_start_lock,
                    )
                except Exception as error:
                    record = None
                    failure = {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
            if record is None:
                assert failure is not None
                failures.append(failure)
                final_audit = _attempt_audit(
                    state,
                    candidate,
                    attempt_root,
                    output=audit,
                    attempt=attempt,
                    status="failed",
                    failure=failure,
                )
                _log(
                    f"candidate failed state={_state_key(state)} "
                    f"candidate={candidate['ordinal']}/12 attempt={attempt}/"
                    f"{plan['fixed_retry_limit']} error={failure['error_type']}"
                )
                continue
            final_audit = _attempt_audit(
                state,
                candidate,
                attempt_root,
                output=audit,
                attempt=attempt,
                status="accepted",
                failure=None,
            )
            accepted_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(attempt_root, accepted_root)
            successful_attempt = attempt
            break
    wall_seconds = time.monotonic() - started
    common = {
        "schema": "issue_68_candidate_execution_result_v2",
        "plan_identity": plan["identity"],
        "state_identity": state["identity"],
        "candidate_identity": candidate["identity"],
        "candidate_ordinal": candidate["ordinal"],
        "exposure_role": state["exposure_role"],
        "generator_family": state["generator_family"],
        "generation_seed": state["generation_seed"],
        "action_stratum": candidate["action_stratum"],
        "interface_action": {
            "drag_x": candidate["drag_x"],
            "drag_y": candidate["drag_y"],
            "tap_time_ms": candidate["tap_time_ms"],
        },
        "realized_cost_contract_identity": plan["realized_cost_contract"][
            "identity"
        ],
        "wall_seconds": wall_seconds,
        "attempt_count": len(failures) + (1 if record is not None else 0),
        "failures": failures,
        "audit_manifest": (
            None if final_audit is None else final_audit["manifest_path"]
        ),
        "outcome_conditioned_membership": False,
        "final_evaluation_opened": False,
    }
    if record is None:
        result = {
            **common,
            "status": "failed",
            "trajectory_relative_path": None,
            "scenario_lineage_identity": None,
            "level_instance_identity": None,
            "trajectory_identity": None,
            "realized_cost": FAILURE_COST,
            "goal_count_cost": None,
            "endpoint_outcome": None,
            "terminal_reason": "collection_failure",
            "interaction_coverage": [],
            "frame_record_count": 0,
            "artifact_bytes": 0,
        }
    else:
        outcome, terminal, coverage = _result_outcome(
            accepted_root,
            release=release_identity(str(plan["identity"])),
            role=str(state["exposure_role"]),
        )
        result = {
            **common,
            "status": "accepted",
            "trajectory_relative_path": _trajectory_relative(
                state, candidate
            ).as_posix(),
            "scenario_lineage_identity": record["scenario_lineage_identity"],
            "level_instance_identity": record["level_instance_identity"],
            "trajectory_identity": record["trajectory_identity"],
            "realized_cost": outcome["realized_cost"],
            "goal_count_cost": outcome["goal_count_cost"],
            "endpoint_outcome": outcome,
            "terminal_reason": terminal,
            "interaction_coverage": coverage,
            "frame_record_count": sum(
                int(item["frame_count"]) for item in record["shots"]
            ),
            "artifact_bytes": _directory_bytes(accepted_root),
        }
    _write_json(result_path, result)
    return _validate_result(
        result,
        plan,
        state,
        candidate,
        trajectory_root=accepted_root if record is not None else None,
    )


def _collect_shard(
    plan: Mapping[str, Any],
    tasks: tuple[tuple[int, int], ...],
    *,
    runtime: Path,
    audit: Path,
    speed: int,
    headless: bool,
    worker: int,
) -> list[dict[str, Any]]:
    game = Path(runtime) / "workers" / f"worker-{worker:02d}" / "game-runtime"
    if not game.exists():
        _log(f"worker={worker} extracting accepted player")
        archive_details(STAGE_ROOT, game)
    results = []
    for task_index, (state_index, candidate_index) in enumerate(tasks, start=1):
        state = plan["states"][state_index]
        candidate = state["candidates"][candidate_index]
        _log(
            f"worker={worker} task={task_index}/{len(tasks)} "
            f"state={state_index + 1}/{len(plan['states'])} "
            f"role={state['exposure_role']} candidate={candidate_index + 1}/12 "
            f"stratum={candidate['action_stratum']}"
        )
        result = _collect_candidate(
            plan,
            state,
            candidate,
            runtime=runtime,
            audit=audit,
            game=game,
            speed=speed,
            headless=headless,
            engine_start_lock=Path(runtime) / "engine-start.lock",
        )
        results.append(result)
        _log(
            f"worker={worker} complete state={state_index + 1}/"
            f"{len(plan['states'])} candidate={candidate_index + 1}/12 "
            f"status={result['status']} cost={result['realized_cost']} "
            f"wall={result['wall_seconds']:.1f}s"
        )
    return results


def _initialize_runtime(
    plan: Mapping[str, Any], runtime: Path, *, implementation_revision: str
) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    _write_json(runtime / "frozen-plan.json", plan)
    player = _player()
    provenance_path = runtime / "provenance.json"
    provenance = {
        "schema": "issue_68_collection_provenance_v2",
        "plan_identity": plan["identity"],
        "implementation_revision": implementation_revision,
        "player": player,
        "collected_at": _now(),
        "final_evaluation_opened": False,
    }
    if provenance_path.is_file():
        existing = _load_json(provenance_path, "collection provenance")
        comparable = {**existing, "collected_at": provenance["collected_at"]}
        if comparable != provenance:
            raise CorrectiveRankingCohortError(
                "resumed collection provenance differs"
            )
    else:
        _write_json(provenance_path, provenance)
    if plan["phase"] == "production":
        _write_json(runtime / "release-staging/production-plan.json", plan)


def _validate_state_results(
    plan_header: Mapping[str, Any],
    state: Mapping[str, Any],
    runtime: Path,
) -> tuple[list[dict[str, Any]], float]:
    started = time.monotonic()
    results = []
    for candidate in state["candidates"]:
        path = _result_path(runtime, state, candidate)
        if not path.is_file():
            raise CorrectiveRankingCohortError(
                f"candidate result is missing: {_candidate_key(state, candidate)}"
            )
        result = _load_json(path, "candidate result")
        trajectory_root = (
            None
            if result.get("status") == "failed"
            else _accepted_root(
                runtime, str(plan_header["phase"]), state, candidate
            )
        )
        results.append(_validate_result(
            result,
            plan_header,
            state,
            candidate,
            trajectory_root=trajectory_root,
        ))
    return results, time.monotonic() - started


def _all_results(
    plan: Mapping[str, Any],
    runtime: Path,
    *,
    workers: int = 1,
    progress_label: str = "collection",
) -> list[dict[str, Any]]:
    if type(workers) is not int or workers < 1:
        raise CorrectiveRankingCohortError("validation worker count must be positive")
    states = tuple(plan["states"])
    plan_header = {
        key: plan[key]
        for key in (
            "identity",
            "phase",
            "fixed_retry_limit",
            "realized_cost_contract",
        )
    }
    active_workers = min(workers, len(states))
    candidate_count = sum(len(state["candidates"]) for state in states)
    _log(
        f"{progress_label} exact validation start states={len(states)} "
        f"candidates={candidate_count} workers={active_workers}"
    )
    by_state: dict[int, list[dict[str, Any]]] = {}
    completed = 0
    if active_workers == 1:
        for state_index, state in enumerate(states):
            state_results, wall_seconds = _validate_state_results(
                plan_header, state, runtime
            )
            by_state[state_index] = state_results
            completed += 1
            _log(
                f"{progress_label} exact validation progress={completed}/"
                f"{len(states)} state={state_index + 1}/{len(states)} "
                f"role={state['exposure_role']} candidates=12/12 "
                f"wall={wall_seconds:.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            futures = {
                executor.submit(
                    _validate_state_results,
                    plan_header,
                    state,
                    runtime,
                ): state_index
                for state_index, state in enumerate(states)
            }
            for future in as_completed(futures):
                state_index = futures[future]
                state_results, wall_seconds = future.result()
                by_state[state_index] = state_results
                completed += 1
                state = states[state_index]
                _log(
                    f"{progress_label} exact validation progress={completed}/"
                    f"{len(states)} state={state_index + 1}/{len(states)} "
                    f"role={state['exposure_role']} candidates=12/12 "
                    f"wall={wall_seconds:.1f}s"
                )
    results = [
        result
        for state_index in range(len(states))
        for result in by_state[state_index]
    ]
    records = Path(runtime) / "records"
    actual_count = len(tuple(records.glob("*.json"))) if records.is_dir() else 0
    if actual_count != len(results):
        raise CorrectiveRankingCohortError(
            "collection has candidate results outside the frozen plan"
        )
    return results


def _audit_gallery(
    plan: Mapping[str, Any], results: list[Mapping[str, Any]], output: Path
) -> dict[str, Any]:
    entries = []
    for result in results:
        manifest_reference = result.get("audit_manifest")
        if not isinstance(manifest_reference, str):
            raise CorrectiveRankingCohortError(
                "candidate result has no frame-audit manifest"
            )
        audit = _load_json(output / manifest_reference, "candidate frame audit")
        if (
            audit.get("state_identity") != result["state_identity"]
            or audit.get("candidate_identity") != result["candidate_identity"]
            or audit.get("status") != result["status"]
        ):
            raise CorrectiveRankingCohortError(
                "candidate frame audit is not bound to its result"
            )
        video = audit.get("video_path")
        if video is not None and not (output / str(video)).is_file():
            raise CorrectiveRankingCohortError("candidate audit video is absent")
        entries.append({
            "state_identity": result["state_identity"],
            "candidate_identity": result["candidate_identity"],
            "candidate_ordinal": result["candidate_ordinal"],
            "exposure_role": result["exposure_role"],
            "generator_family": result["generator_family"],
            "action_stratum": result["action_stratum"],
            "status": result["status"],
            "realized_cost": result["realized_cost"],
            "terminal_reason": result["terminal_reason"],
            "frame_count": audit["frame_count"],
            "manifest_path": manifest_reference,
            "video_path": video,
        })
    manifest = {
        "schema": "issue_68_candidate_frame_audit_gallery_v2",
        "plan_identity": plan["identity"],
        "phase": plan["phase"],
        "source_observation_role": "agent",
        "canonical_observations_included": False,
        "playback_fps": 50,
        "candidate_count": len(entries),
        "video_count": sum(item["video_path"] is not None for item in entries),
        "accepted_video_count": sum(
            item["status"] == "accepted" and item["video_path"] is not None
            for item in entries
        ),
        "entries": entries,
        "gallery": "index.html",
        "final_evaluation_opened": False,
    }
    sections = []
    for item in entries:
        media = (
            "<p>No captured frames were available for this failed execution.</p>"
            if item["video_path"] is None
            else (
                "<video controls preload=\"metadata\" src=\""
                + html.escape(str(item["video_path"]))
                + "\"></video>"
            )
        )
        label = (
            f"{item['exposure_role']} · {item['state_identity']} · "
            f"candidate {item['candidate_ordinal']:02d} · {item['action_stratum']}"
        )
        sections.append(
            "<section><h2>"
            + html.escape(label)
            + "</h2>"
            + media
            + f"<p>status={html.escape(str(item['status']))} · "
            + f"cost={item['realized_cost']} · "
            + f"terminal={html.escape(str(item['terminal_reason']))} · "
            + f"frames={item['frame_count']}</p></section>"
        )
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Issue 68 candidate audit</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1100px}"
        "section{border-top:1px solid #bbb;padding:1rem 0}"
        "video{display:block;width:100%;max-width:800px;background:#111}"
        "h2{overflow-wrap:anywhere}</style></head><body>"
        f"<h1>Issue 68 {html.escape(str(plan['phase']))} candidate audit</h1>"
        f"<p>{len(entries)} prospectively scheduled candidates; VP8/WebM at "
        "50 fps from deployment-valid agent observations. Canonical observations "
        "are excluded.</p>"
        + "".join(sections)
        + "</body></html>"
    ).encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    index = output / "index.html"
    if index.exists() and index.read_bytes() != document:
        raise CorrectiveRankingCohortError("immutable audit gallery differs")
    if not index.exists():
        write_immutable_cohort_v2_bytes(document, index)
    _write_json(output / "manifest.json", manifest)
    _log(
        f"audit gallery complete candidates={len(entries)} "
        f"videos={manifest['video_count']} path={index}"
    )
    return manifest


def run_collection(
    plan: Mapping[str, Any],
    *,
    runtime: Path,
    audit: Path,
    implementation_revision: str,
    start_display_process: bool,
    speed: int,
    headless: bool,
    workers: int,
) -> list[dict[str, Any]]:
    plan = validate_plan(plan)
    runtime = Path(runtime).resolve()
    audit = (Path(audit) / str(plan["phase"])).resolve()
    if type(workers) is not int or workers < 1:
        raise CorrectiveRankingCohortError("worker count must be positive")
    _initialize_runtime(
        plan, runtime, implementation_revision=implementation_revision
    )
    tasks = []
    completed = 0
    for state_index, state in enumerate(plan["states"]):
        for candidate_index, candidate in enumerate(state["candidates"]):
            if _result_path(runtime, state, candidate).is_file():
                trajectory_root = _accepted_root(
                    runtime, str(plan["phase"]), state, candidate
                )
                result = _load_json(
                    _result_path(runtime, state, candidate), "candidate result"
                )
                _validate_result(
                    result,
                    plan,
                    state,
                    candidate,
                    trajectory_root=(
                        trajectory_root if result.get("status") == "accepted" else None
                    ),
                )
                completed += 1
            else:
                tasks.append((state_index, candidate_index))
    _log(
        f"collection phase={plan['phase']} scheduled="
        f"{len(plan['states']) * 12} resumed={completed} pending={len(tasks)} "
        f"workers={min(workers, len(tasks)) if tasks else 0}"
    )
    display_process = None
    prior_display = os.environ.get("DISPLAY")
    try:
        if start_display_process and tasks:
            display, display_process = start_display(runtime / "display.log")
            os.environ["DISPLAY"] = display
            _log(f"shared collection display started DISPLAY={display}")
        active_workers = min(workers, len(tasks))
        shards = tuple(
            tuple(tasks[index::active_workers]) for index in range(active_workers)
        ) if active_workers else ()
        if len(shards) == 1:
            _collect_shard(
                plan,
                shards[0],
                runtime=runtime,
                audit=audit,
                speed=speed,
                headless=headless,
                worker=1,
            )
        elif shards:
            with ProcessPoolExecutor(max_workers=len(shards)) as executor:
                futures = {
                    executor.submit(
                        _collect_shard,
                        plan,
                        shard,
                        runtime=runtime,
                        audit=audit,
                        speed=speed,
                        headless=headless,
                        worker=index,
                    ): index
                    for index, shard in enumerate(shards, start=1)
                }
                for future in as_completed(futures):
                    records = future.result()
                    _log(
                        f"worker={futures[future]} shard complete "
                        f"candidates={len(records)} accepted="
                        f"{sum(item['status'] == 'accepted' for item in records)}"
                    )
    finally:
        if display_process is not None:
            _log(f"shared collection display stopped result={terminate(display_process)}")
        if prior_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = prior_display
    results = _all_results(
        plan,
        runtime,
        workers=workers,
        progress_label=f"{plan['phase']} finalization",
    )
    _audit_gallery(plan, results, audit)
    _log(
        f"collection complete phase={plan['phase']} "
        f"accepted={sum(item['status'] == 'accepted' for item in results)}/"
        f"{len(results)} failed={sum(item['status'] == 'failed' for item in results)}"
    )
    return results


def _ranking_states(
    plan: Mapping[str, Any], results: list[Mapping[str, Any]]
) -> tuple[ActionRankingState, ...]:
    by_candidate = {
        (item["state_identity"], item["candidate_identity"]): item
        for item in results
    }
    states = []
    for state in plan["states"]:
        state_results = [
            by_candidate[(state["identity"], candidate["identity"])]
            for candidate in state["candidates"]
        ]
        first_accepted = next(
            (item for item in state_results if item["status"] == "accepted"), None
        )
        candidates = tuple(
            ActionCandidate(
                identity=str(candidate["identity"]),
                action=_action_tensor(candidate),
                realized_cost=float(result["realized_cost"]),
                interface_action=SlingshotAction(
                    int(candidate["drag_x"]),
                    int(candidate["drag_y"]),
                    int(candidate["tap_time_ms"]),
                ),
            )
            for candidate, result in zip(
                state["candidates"], state_results, strict=True
            )
        )
        states.append(ActionRankingState(
            identity=str(state["identity"]),
            scenario_lineage_identity=(
                str(first_accepted["scenario_lineage_identity"])
                if first_accepted is not None
                else f"{state['identity']}:failed-scenario"
            ),
            trajectory_identity=(
                str(first_accepted["trajectory_identity"])
                if first_accepted is not None
                else f"{state['identity']}:failed-trajectory"
            ),
            decision_transition_identity=f"{state['identity']}:initial-decision",
            exposure_role=str(state["exposure_role"]),
            carrier=CarrierKind.SOURCE,
            carrier_identity="issue-68-pilot-diversity-only",
            context=torch.zeros(197),
            action_bounds=action_bounds(),
            frame_height=FRAME_HEIGHT,
            candidates=candidates,
            cost_target=torch.zeros(197),
        ))
    return tuple(states)


def _diversity_payload(
    plan: Mapping[str, Any], results: list[Mapping[str, Any]]
) -> dict[str, Any]:
    report = summarize_ranking_diversity(
        _ranking_states(plan, results), failure_cost=FAILURE_COST
    )
    ties = Counter(report.best_action_tie_sizes)
    by_state = {
        state["identity"]: [
            item for item in results if item["state_identity"] == state["identity"]
        ]
        for state in plan["states"]
    }
    fully_realized = [
        items for items in by_state.values()
        if all(item["status"] == "accepted" for item in items)
    ]
    discriminating = sum(
        len({float(item["realized_cost"]) for item in items}) > 1
        for items in fully_realized
    )
    goal_count_discriminating = sum(
        len({
            int(
                item["goal_count_cost"]
                if item.get("goal_count_cost") is not None
                else math.floor(float(item["realized_cost"]))
            )
            for item in items
        }) > 1
        for items in fully_realized
    )
    return {
        "state_count": report.state_count,
        "candidate_count": report.candidate_count,
        "all_tied_state_count": report.all_tied_state_count,
        "pig_removal_discordant_state_count": (
            report.pig_removal_discordant_state_count
        ),
        "block_only_discordant_state_count": (
            report.block_only_discordant_state_count
        ),
        "progress_only_discordant_state_count": (
            report.progress_only_discordant_state_count
        ),
        "best_action_tie_sizes": list(report.best_action_tie_sizes),
        "best_action_tie_size_counts": {
            str(size): count for size, count in sorted(ties.items())
        },
        "candidate_failure_count": report.candidate_failure_count,
        "state_failure_count": report.state_failure_count,
        "fully_realized_state_count": len(fully_realized),
        "outcome_discriminating_state_count": discriminating,
        "outcome_discriminating_state_fraction": (
            discriminating / len(plan["states"])
        ),
        "goal_count_discriminating_state_count": goal_count_discriminating,
        "goal_count_discriminating_state_fraction": (
            goal_count_discriminating / len(plan["states"])
        ),
    }


def _pilot_report(
    plan: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_plan(plan, phase="pilot")
    scheduled = len(plan["states"]) * 12
    if len(results) != scheduled:
        raise CorrectiveRankingCohortError("pilot lacks frozen candidate accounting")
    diversity = _diversity_payload(plan, results)
    accepted = [item for item in results if item["status"] == "accepted"]
    failure_fraction = (scheduled - len(accepted)) / scheduled
    minimum_fraction = (
        ISSUE_63_DISCRIMINATING_STATE_FRACTION
        + MINIMUM_DISCRIMINATION_IMPROVEMENT
    )
    passed = (
        diversity["outcome_discriminating_state_fraction"] >= minimum_fraction
        and failure_fraction <= 0.05
        and audit.get("candidate_count") == scheduled
        and audit.get("accepted_video_count") == len(accepted)
    )
    runtime_seconds = [float(item["wall_seconds"]) for item in accepted]
    artifact_bytes = [int(item["artifact_bytes"]) for item in accepted]
    payload = {
        "schema": PILOT_REPORT_SCHEMA,
        "identity": f"issue-68-pilot-report-v2:{plan['identity']}",
        "pilot_plan_identity": plan["identity"],
        "planned_state_count": len(plan["states"]),
        "scheduled_candidate_count": scheduled,
        "completed_candidate_count": len(results),
        "accepted_candidate_count": len(accepted),
        "failed_candidate_count": scheduled - len(accepted),
        "role_counts": dict(plan["role_counts"]),
        "generator_family_state_counts": dict(sorted(Counter(
            state["generator_family"] for state in plan["states"]
        ).items())),
        "action_stratum_counts": dict(sorted(Counter(
            item["action_stratum"] for item in results
        ).items())),
        "terminal_reason_counts": dict(sorted(Counter(
            item["terminal_reason"] for item in results
        ).items())),
        "failure_type_counts": dict(sorted(Counter(
            failure["error_type"]
            for item in results
            for failure in item["failures"]
        ).items())),
        "interaction_counts": dict(sorted(Counter(
            interaction
            for item in accepted
            for interaction in item["interaction_coverage"]
        ).items())),
        "realized_cost_contract": dict(plan["realized_cost_contract"]),
        "diversity": diversity,
        "issue_63_local_probe_reference": {
            "state_count": 24,
            "candidate_count_per_state": 5,
            "outcome_discriminating_state_count": 10,
            "outcome_discriminating_state_fraction": (
                ISSUE_63_DISCRIMINATING_STATE_FRACTION
            ),
            "required_absolute_improvement": MINIMUM_DISCRIMINATION_IMPROVEMENT,
            "exploratory_only": True,
        },
        "runtime_cost": {
            "median_seconds_per_candidate": (
                statistics.median(runtime_seconds) if runtime_seconds else 0.0
            ),
            "median_bytes_per_candidate": (
                int(statistics.median(artifact_bytes)) if artifact_bytes else 0
            ),
        },
        "sample_size_justification": {
            "estimand": "independent-state normalized ranking regret in [0,1]",
            "method": "distribution-free Hoeffding planning bound",
            "two_sided_alpha": 0.05,
            "target_half_width": 0.10,
            "minimum_bound_state_count": 185,
            "recommended_states_per_role": DEFAULT_PRODUCTION_STATES_PER_ROLE,
            "reason": (
                "200 exceeds the 185-state bound and matches the established "
                "non-final held-out role size"
            ),
        },
        "audit": {
            "root": str(Path("data/issue-68-ranking-audit-v2/pilot")),
            "gallery": "index.html",
            "candidate_count": audit["candidate_count"],
            "video_count": audit["video_count"],
        },
        "candidate_failure_fraction": failure_fraction,
        "supersedes": dict(plan["supersedes"]),
        "outcome_conditioned_membership": False,
        "final_evaluation_opened": False,
        "passed": passed,
    }
    return payload


def freeze_pilot(path: Path) -> dict[str, Any]:
    plan = build_pilot_plan()
    _write_json(path, plan)
    _log(
        f"pilot plan frozen identity={plan['identity']} "
        f"states={len(plan['states'])} candidates={len(plan['states']) * 12}"
    )
    return plan


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pilot_plan.is_file():
        _log("fresh v2 pilot plan is absent; freezing it before collection")
        freeze_pilot(args.pilot_plan)
    plan = _load_plan(args.pilot_plan, "pilot")
    if args.pilot_report.is_file():
        report = validate_pilot_report(
            _load_json(args.pilot_report, "pilot report")
        )
        _log(f"pilot report validated existing passed={report['passed']}")
        return report
    results = run_collection(
        plan,
        runtime=args.pilot_runtime,
        audit=args.audit_output,
        implementation_revision=_implementation_revision(
            args.implementation_revision, required=True
        ),
        start_display_process=args.start_display,
        speed=args.speed,
        headless=args.headless,
        workers=args.workers,
    )
    audit = _load_json(
        args.audit_output / "pilot/manifest.json", "pilot audit gallery"
    )
    report = _pilot_report(plan, results, audit)
    _write_json(args.pilot_report, report)
    _log(
        f"pilot complete passed={report['passed']} "
        f"discrimination={report['diversity']['outcome_discriminating_state_count']}/"
        f"{report['planned_state_count']} failures={report['failed_candidate_count']}"
    )
    if not report["passed"]:
        raise CorrectiveRankingCohortError(
            "pilot did not improve outcome discrimination enough to freeze production"
        )
    return report


def freeze_production(args: argparse.Namespace) -> dict[str, Any]:
    pilot_plan = _load_plan(args.pilot_plan, "pilot")
    report = validate_pilot_report(
        _load_json(args.pilot_report, "pilot report")
    )
    if report["pilot_plan_identity"] != pilot_plan["identity"]:
        raise CorrectiveRankingCohortError("pilot report ancestry differs")
    plan = build_production_plan(
        report, states_per_role=args.states_per_role
    )
    _write_json(args.production_plan, plan)
    _log(
        f"production plan frozen identity={plan['identity']} "
        f"states={len(plan['states'])} candidates={len(plan['states']) * 12} "
        f"estimated_hours_single_worker="
        f"{plan['resource_decision']['estimated_collection_hours_single_worker']:.1f}"
    )
    return plan


def _candidate_carriers(
    trajectory_root: Path,
    *,
    release: str,
    role: str,
    adapter: Any,
    codec: CohortV2StateCodec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    raw = _load_json(trajectory_root / "trajectory.json", "candidate trajectory")
    shot = _load_shot(
        trajectory_root,
        raw["shots"][-1],
        release_identity=release,
        role=role,
    )
    fixed_steps = tuple(sorted(shot["frames"]))
    if len(fixed_steps) < 2:
        raise CorrectiveRankingCohortError(
            "candidate trajectory has no temporal endpoint"
        )
    selected_steps = tuple(dict.fromkeys(
        (fixed_steps[0], fixed_steps[-2], fixed_steps[-1])
    ))
    observations = tuple(_observation(shot, step) for step in selected_steps)
    parsed = adapter.parse_batch(observations)
    by_step = {
        step: (observation, parse)
        for step, observation, parse in zip(
            selected_steps, observations, parsed, strict=True
        )
    }
    initial_observation, initial_parsed = by_step[fixed_steps[0]]
    prior_observation, prior_parsed = by_step[fixed_steps[-2]]
    endpoint_observation, endpoint_parsed = by_step[fixed_steps[-1]]
    source_initial = codec.encode(shot["frames"][fixed_steps[0]])
    source_endpoint = codec.encode(shot["frames"][fixed_steps[-1]])
    deployment_initial = adapter.build_from_parsed(
        TemporalObservationContext(None, initial_observation),
        initial_parsed,
        None,
    ).tensor
    deployment_endpoint = adapter.build_from_parsed(
        TemporalObservationContext(prior_observation, endpoint_observation),
        endpoint_parsed,
        prior_parsed,
    ).tensor
    return (
        source_initial,
        deployment_initial,
        source_endpoint,
        deployment_endpoint,
        _realized_goal_cost(shot["frames"][fixed_steps[-1]]),
    )


def _matched_bundle_projection(
    source: tuple[CarrierLineage, ...], deployment: tuple[CarrierLineage, ...]
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    source_projection = tuple(
        (
            item.trajectory_identity,
            item.scenario_lineage_identity,
            tuple(transition.identity for transition in item.transitions),
        )
        for item in source
    )
    deployment_projection = tuple(
        (
            item.trajectory_identity,
            item.scenario_lineage_identity,
            tuple(transition.identity for transition in item.transitions),
        )
        for item in deployment
    )
    if source_projection != deployment_projection:
        raise CorrectiveRankingCohortError("source/deployment carrier bundles differ")
    return source_projection


def _matched_ranking_projection(
    source: tuple[ActionRankingState, ...],
    deployment: tuple[ActionRankingState, ...],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    def project(states: tuple[ActionRankingState, ...]) -> tuple[Any, ...]:
        return tuple(
            (
                state.identity,
                tuple(
                    (candidate.identity, float(candidate.realized_cost))
                    for candidate in state.candidates
                ),
            )
            for state in states
        )

    source_projection = project(source)
    if source_projection != project(deployment):
        raise CorrectiveRankingCohortError("source/deployment ranking bundles differ")
    return source_projection


def _atomic_save_carriers(
    path: Path, lineages: tuple[CarrierLineage, ...]
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_carrier_lineage_bundle(temporary, lineages)
    os.replace(temporary, path)


def _atomic_save_rankings(
    path: Path, states: tuple[ActionRankingState, ...]
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_action_ranking_bundle(temporary, states)
    os.replace(temporary, path)


def _build_role_products(
    plan: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    *,
    staging: Path,
    role: str,
    adapter: Any,
    codec: CohortV2StateCodec,
) -> dict[str, Any]:
    source_carrier_path = staging / "carrier-bundles" / f"{role}-source.pt"
    deployment_carrier_path = (
        staging / "carrier-bundles" / f"{role}-deployment.pt"
    )
    source_ranking_path = staging / "ranking-bundles" / f"{role}-source.pt"
    deployment_ranking_path = (
        staging / "ranking-bundles" / f"{role}-deployment.pt"
    )
    paths = (
        source_carrier_path,
        deployment_carrier_path,
        source_ranking_path,
        deployment_ranking_path,
    )
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise CorrectiveRankingCohortError(
                f"interrupted {role} product quartet requires inspection"
            )
        source_carriers = load_carrier_lineage_bundle(source_carrier_path)
        deployment_carriers = load_carrier_lineage_bundle(
            deployment_carrier_path
        )
        source_rankings = load_action_ranking_bundle(source_ranking_path)
        deployment_rankings = load_action_ranking_bundle(
            deployment_ranking_path
        )
        if (
            len(source_carriers) != plan["role_counts"][role]
            or len(source_rankings) != plan["role_counts"][role]
        ):
            raise CorrectiveRankingCohortError(
                f"existing {role} products have the wrong state count"
            )
        _matched_bundle_projection(source_carriers, deployment_carriers)
        _matched_ranking_projection(source_rankings, deployment_rankings)
        _log(f"publish role={role} validated existing carrier/ranking products")
        return {
            "state_count": len(source_rankings),
            "carrier_lineage_count": len(source_carriers),
        }

    state_results = {
        state["identity"]: [
            item
            for item in results
            if item["state_identity"] == state["identity"]
        ]
        for state in plan["states"]
        if state["exposure_role"] == role
    }
    source_carriers: list[CarrierLineage] = []
    deployment_carriers: list[CarrierLineage] = []
    source_rankings: list[ActionRankingState] = []
    deployment_rankings: list[ActionRankingState] = []
    role_states = [
        state for state in plan["states"] if state["exposure_role"] == role
    ]
    for state_index, state in enumerate(role_states, start=1):
        candidates_by_identity = {
            item["candidate_identity"]: item
            for item in state_results[state["identity"]]
        }
        ordered_results = [
            candidates_by_identity[candidate["identity"]]
            for candidate in state["candidates"]
        ]
        anchor_index = int(state["carrier_anchor_candidate_ordinal"]) - 1
        anchor_result = ordered_results[anchor_index]
        if anchor_result["status"] != "accepted":
            raise CorrectiveRankingCohortError(
                f"frozen carrier anchor failed for {state['identity']}"
            )
        successful_carriers = []
        for candidate_index, (candidate, result) in enumerate(
            zip(state["candidates"], ordered_results, strict=True), start=1
        ):
            if result["status"] == "failed":
                continue
            trajectory_root = staging / str(result["trajectory_relative_path"])
            values = _candidate_carriers(
                trajectory_root,
                release=release_identity(str(plan["identity"])),
                role=role,
                adapter=adapter,
                codec=codec,
            )
            if values[-1] != float(result["goal_count_cost"]):
                raise CorrectiveRankingCohortError(
                    "published carrier endpoint changed the realized cost"
                )
            successful_carriers.append((candidate_index, *values[:-1]))
            _log(
                f"publish role={role} state={state_index}/{len(role_states)} "
                f"candidate={candidate_index}/12 endpoint parsed"
            )
        if not successful_carriers:
            raise CorrectiveRankingCohortError(
                f"ranking state has no realized candidate: {state['identity']}"
            )
        reference_source = successful_carriers[0][1]
        reference_deployment = successful_carriers[0][2]
        if any(
            not torch.equal(item[1], reference_source)
            or not torch.allclose(item[2], reference_deployment, atol=0.0, rtol=0.0)
            for item in successful_carriers[1:]
        ):
            raise CorrectiveRankingCohortError(
                f"candidate replays changed the initial state: {state['identity']}"
            )
        successful_results = [
            (index, item)
            for index, item in enumerate(ordered_results)
            if item["status"] == "accepted"
        ]
        best_index, _best_result = min(
            successful_results,
            key=lambda pair: (float(pair[1]["realized_cost"]), pair[0]),
        )
        endpoints = {
            index - 1: (source_endpoint, deployment_endpoint)
            for index, _source_initial, _deployment_initial,
            source_endpoint, deployment_endpoint in successful_carriers
        }
        best_source_target, best_deployment_target = endpoints[best_index]
        anchor_root = staging / str(anchor_result["trajectory_relative_path"])
        record = {
            "path": str(anchor_result["trajectory_relative_path"]),
            "trajectory_identity": anchor_result["trajectory_identity"],
            "scenario_lineage_identity": anchor_result[
                "scenario_lineage_identity"
            ],
            "exposure_role": role,
        }
        source_lineage, deployment_lineage = _build_lineage_pair(
            staging,
            release_identity(str(plan["identity"])),
            record,
            adapter,
            codec,
        )
        source_transition = next(
            item
            for item in source_lineage.transitions
            if item.horizon == 15 and item.decision_index == 0
        )
        deployment_transition = next(
            item
            for item in deployment_lineage.transitions
            if item.horizon == 15 and item.decision_index == 0
        )
        if (
            not torch.equal(source_transition.context, reference_source)
            or not torch.allclose(
                deployment_transition.context,
                reference_deployment,
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise CorrectiveRankingCohortError(
                "carrier anchor context differs from candidate initial state"
            )
        action_candidates = tuple(
            ActionCandidate(
                identity=str(candidate["identity"]),
                action=_action_tensor(candidate),
                realized_cost=float(result["realized_cost"]),
                interface_action=SlingshotAction(
                    int(candidate["drag_x"]),
                    int(candidate["drag_y"]),
                    int(candidate["tap_time_ms"]),
                ),
            )
            for candidate, result in zip(
                state["candidates"], ordered_results, strict=True
            )
        )
        common = {
            "identity": str(state["identity"]),
            "scenario_lineage_identity": str(
                anchor_result["scenario_lineage_identity"]
            ),
            "trajectory_identity": str(anchor_result["trajectory_identity"]),
            "decision_transition_identity": source_transition.identity,
            "exposure_role": role,
            "action_bounds": action_bounds(),
            "frame_height": FRAME_HEIGHT,
            "candidates": action_candidates,
        }
        source_rankings.append(ActionRankingState(
            **common,
            carrier=CarrierKind.SOURCE,
            carrier_identity=source_lineage.carrier_identity,
            context=source_transition.context,
            cost_target=best_source_target,
        ))
        deployment_rankings.append(ActionRankingState(
            **common,
            carrier=CarrierKind.DEPLOYMENT,
            carrier_identity=deployment_lineage.carrier_identity,
            context=deployment_transition.context,
            cost_target=best_deployment_target,
        ))
        source_carriers.append(source_lineage)
        deployment_carriers.append(deployment_lineage)
        _log(
            f"publish role={role} state={state_index}/{len(role_states)} "
            f"anchor={anchor_index + 1} best={best_index + 1} carrier complete"
        )
    source_carrier_tuple = tuple(source_carriers)
    deployment_carrier_tuple = tuple(deployment_carriers)
    source_ranking_tuple = tuple(source_rankings)
    deployment_ranking_tuple = tuple(deployment_rankings)
    _matched_bundle_projection(source_carrier_tuple, deployment_carrier_tuple)
    _matched_ranking_projection(source_ranking_tuple, deployment_ranking_tuple)
    _atomic_save_carriers(source_carrier_path, source_carrier_tuple)
    _atomic_save_carriers(deployment_carrier_path, deployment_carrier_tuple)
    _atomic_save_rankings(source_ranking_path, source_ranking_tuple)
    _atomic_save_rankings(deployment_ranking_path, deployment_ranking_tuple)
    _log(
        f"publish role={role} products written states={len(source_ranking_tuple)}"
    )
    return {
        "state_count": len(source_ranking_tuple),
        "carrier_lineage_count": len(source_carrier_tuple),
    }


def _external_disjointness(
    plan: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    *,
    pilot_plan: Mapping[str, Any],
    issue_62_release: Path,
    issue_57_protocol: Path,
) -> dict[str, Any]:
    issue_62_plan = _load_json(
        Path(issue_62_release) / "production-plan.json", "issue-62 production plan"
    )
    issue_62_manifest = _load_json(
        Path(issue_62_release) / "manifest.json", "issue-62 release manifest"
    )
    issue_57 = _load_json(issue_57_protocol, "issue-57 gameplay protocol")
    current_seeds = {int(state["generation_seed"]) for state in plan["states"]}
    pilot_seeds = {
        int(state["generation_seed"]) for state in pilot_plan["states"]
    }
    issue_62_seeds = {
        int(state["generation_seed"]) for state in issue_62_plan["lineages"]
    }
    scenario_by_state = {}
    level_by_state = {}
    for state in plan["states"]:
        accepted = [
            item
            for item in results
            if item["state_identity"] == state["identity"]
            and item["status"] == "accepted"
        ]
        scenarios = {item["scenario_lineage_identity"] for item in accepted}
        levels = {item["level_instance_identity"] for item in accepted}
        if len(scenarios) != 1 or len(levels) != 1:
            raise CorrectiveRankingCohortError(
                f"candidate executions do not share one scenario: {state['identity']}"
            )
        scenario_by_state[state["identity"]] = next(iter(scenarios))
        level_by_state[state["identity"]] = next(iter(levels))
    current_scenarios = set(scenario_by_state.values())
    current_levels = set(level_by_state.values())
    issue_62_scenarios = {
        item["scenario_lineage_identity"]
        for item in issue_62_manifest["trajectories"]
    }
    issue_62_levels = {
        item["level_instance_identity"]
        for item in issue_62_manifest["trajectories"]
    }
    issue_57_levels = {
        item["level_identity"]
        for values in issue_57["level_inventory"]["roles"].values()
        for item in values
    }
    role_scenarios = {
        role: {
            scenario_by_state[state["identity"]]
            for state in plan["states"]
            if state["exposure_role"] == role
        }
        for role in ROLES
    }
    if (
        current_seeds & pilot_seeds
        or current_seeds & issue_62_seeds
        or current_scenarios & issue_62_scenarios
        or current_levels & issue_62_levels
        or current_levels & issue_57_levels
        or role_scenarios[ROLES[0]] & role_scenarios[ROLES[1]]
        or len(current_scenarios) != len(plan["states"])
        or len(current_levels) != len(plan["states"])
    ):
        raise CorrectiveRankingCohortError(
            "corrective cohort overlaps a pilot, role, issue-57, or issue-62 boundary"
        )
    return {
        "pilot_seed_overlap_count": len(current_seeds & pilot_seeds),
        "issue_62_seed_overlap_count": len(current_seeds & issue_62_seeds),
        "issue_62_scenario_lineage_overlap_count": len(
            current_scenarios & issue_62_scenarios
        ),
        "issue_62_level_instance_overlap_count": len(
            current_levels & issue_62_levels
        ),
        "issue_57_level_instance_overlap_count": len(
            current_levels & issue_57_levels
        ),
        "cross_role_scenario_lineage_overlap_count": len(
            role_scenarios[ROLES[0]] & role_scenarios[ROLES[1]]
        ),
        "future_issue_64_seed_exclusion_declared": True,
        "passed": True,
    }


def _audit_reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _manifest(
    plan: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    *,
    pilot_report: Mapping[str, Any],
    audit_reference: str,
    audit: Mapping[str, Any],
    disjointness: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = [item for item in results if item["status"] == "accepted"]
    diversity_by_role = {}
    for role in ROLES:
        role_plan = {
            **plan,
            "states": [
                state for state in plan["states"]
                if state["exposure_role"] == role
            ],
        }
        role_results = [
            item for item in results if item["exposure_role"] == role
        ]
        diversity_by_role[role] = _diversity_payload(role_plan, role_results)
    coverage_by_role = {
        role: {
            "state_count": plan["role_counts"][role],
            "candidate_count": sum(
                item["exposure_role"] == role for item in results
            ),
            "generator_family_state_counts": dict(sorted(Counter(
                state["generator_family"]
                for state in plan["states"]
                if state["exposure_role"] == role
            ).items())),
            "action_stratum_counts": dict(sorted(Counter(
                item["action_stratum"]
                for item in results
                if item["exposure_role"] == role
            ).items())),
            "outcome_diversity": diversity_by_role[role],
            "terminal_reason_counts": dict(sorted(Counter(
                item["terminal_reason"]
                for item in results
                if item["exposure_role"] == role
            ).items())),
            "failure_type_counts": dict(sorted(Counter(
                failure["error_type"]
                for item in results
                if item["exposure_role"] == role
                for failure in item["failures"]
            ).items())),
        }
        for role in ROLES
    }
    return {
        "schema": RELEASE_SCHEMA,
        "identity": f"issue-68-corrective-cohort-v2:{plan['identity']}",
        "production_plan_identity": plan["identity"],
        "pilot_report_identity": pilot_report["identity"],
        "implementation_revision": provenance["implementation_revision"],
        "candidate_design_identity": plan["action_design"]["identity"],
        "realized_cost_contract": dict(plan["realized_cost_contract"]),
        "failure_treatment": dict(plan["failure_treatment"]),
        "counts": {
            "states": len(plan["states"]),
            "states_by_role": dict(plan["role_counts"]),
            "scheduled_candidates": len(results),
            "accepted_candidates": len(accepted),
            "failed_candidates": len(results) - len(accepted),
            "carrier_lineages_by_role": dict(plan["role_counts"]),
            "ranking_states_by_role": dict(plan["role_counts"]),
        },
        "generator_family_state_counts": dict(sorted(Counter(
            state["generator_family"] for state in plan["states"]
        ).items())),
        "action_stratum_counts": dict(sorted(Counter(
            item["action_stratum"] for item in results
        ).items())),
        "outcome_diversity_by_role": diversity_by_role,
        "coverage_by_role": coverage_by_role,
        "best_action_tie_size_counts": dict(sorted(Counter(
            str(size)
            for report in diversity_by_role.values()
            for size in report["best_action_tie_sizes"]
        ).items())),
        "terminal_reason_counts": dict(sorted(Counter(
            item["terminal_reason"] for item in results
        ).items())),
        "failure_type_counts": dict(sorted(Counter(
            failure["error_type"]
            for item in results
            for failure in item["failures"]
        ).items())),
        "interaction_counts": dict(sorted(Counter(
            interaction
            for item in accepted
            for interaction in item["interaction_coverage"]
        ).items())),
        "carrier_anchor_rule": "candidate ordinal = (role ordinal mod 12) + 1",
        "artifacts": {
            "production_plan": "production-plan.json",
            "pilot_report": "pilot-report.json",
            "candidate_results": "candidate-results",
            "trajectories": "trajectories",
            "carrier_bundles": "carrier-bundles",
            "ranking_bundles": "ranking-bundles",
            "candidate_audit_root": audit_reference,
            "candidate_audit_gallery": f"{audit_reference}/index.html",
            "candidate_audit_count": audit["candidate_count"],
            "candidate_video_count": audit["video_count"],
        },
        "disjointness": dict(disjointness),
        "model_selection_access_rule": plan["model_selection_access_rule"],
        "issue_63_model_selection_reused": False,
        "expert_demonstrations_used": False,
        "target_shot_labels_used": False,
        "outcome_conditioned_membership": False,
        "supersedes": dict(plan["supersedes"]),
        "final_evaluation_opened": False,
        "passed": True,
    }


def publish_production(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        return validate_release(args)
    plan = _load_plan(args.production_plan, "production")
    pilot_plan = _load_plan(args.pilot_plan, "pilot")
    pilot_report = validate_pilot_report(
        _load_json(args.pilot_report, "pilot report")
    )
    if (
        plan["pilot_report_identity"] != pilot_report["identity"]
        or pilot_report["pilot_plan_identity"] != pilot_plan["identity"]
    ):
        raise CorrectiveRankingCohortError("publication pilot ancestry differs")
    runtime = args.production_runtime.resolve()
    staging = runtime / "release-staging"
    if not staging.is_dir():
        raise CorrectiveRankingCohortError("production collection staging is absent")
    results = _all_results(
        plan,
        runtime,
        workers=args.validation_workers,
        progress_label="publish preflight",
    )
    audit_root = (args.audit_output / "production").resolve()
    audit = _load_json(audit_root / "manifest.json", "production audit gallery")
    if audit.get("candidate_count") != len(results):
        raise CorrectiveRankingCohortError("production audit is incomplete")
    provenance = _load_json(runtime / "provenance.json", "collection provenance")
    disjointness = _external_disjointness(
        plan,
        results,
        pilot_plan=pilot_plan,
        issue_62_release=args.issue_62_release,
        issue_57_protocol=args.issue_57_protocol,
    )
    _log("publish loading deployment observation adapter")
    adapter = _load_deployment_adapter(device=args.device)
    codec = CohortV2StateCodec(latent_dim=197, max_entities=15)
    products = {}
    for role in ROLES:
        products[role] = _build_role_products(
            plan,
            results,
            staging=staging,
            role=role,
            adapter=adapter,
            codec=codec,
        )
        gc.collect()
    if any(
        products[role]["state_count"] != plan["role_counts"][role]
        for role in ROLES
    ):
        raise CorrectiveRankingCohortError("published role products are incomplete")
    _write_json(staging / "pilot-report.json", pilot_report)
    _write_json(staging / "provenance.json", provenance)
    for state in plan["states"]:
        for candidate in state["candidates"]:
            result = next(
                item for item in results
                if item["candidate_identity"] == candidate["identity"]
            )
            _write_json(
                staging / "candidate-results" / f"{_candidate_key(state, candidate)}.json",
                result,
            )
    manifest = _manifest(
        plan,
        results,
        pilot_report=pilot_report,
        audit_reference=_audit_reference(audit_root),
        audit=audit,
        disjointness=disjointness,
        provenance=provenance,
    )
    _write_json(staging / "manifest.json", manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, args.output)
    summary = {
        "schema": "issue_68_corrective_cohort_summary_v2",
        "artifact_identity": manifest["identity"],
        "production_plan_identity": plan["identity"],
        "counts": manifest["counts"],
        "outcome_diversity_by_role": manifest["outcome_diversity_by_role"],
        "disjointness": manifest["disjointness"],
        "audit_gallery": manifest["artifacts"]["candidate_audit_gallery"],
        "model_selection_access_rule": manifest["model_selection_access_rule"],
        "final_evaluation_opened": False,
        "passed": True,
    }
    _write_json(args.summary, summary)
    _log(
        f"publication complete states={manifest['counts']['states']} "
        f"candidates={manifest['counts']['scheduled_candidates']} "
        f"release={args.output}"
    )
    return summary


def _validate_role_products(
    plan: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    *,
    root: Path,
    role: str,
) -> None:
    source_carriers = load_carrier_lineage_bundle(
        root / "carrier-bundles" / f"{role}-source.pt"
    )
    deployment_carriers = load_carrier_lineage_bundle(
        root / "carrier-bundles" / f"{role}-deployment.pt"
    )
    source_rankings = load_action_ranking_bundle(
        root / "ranking-bundles" / f"{role}-source.pt"
    )
    deployment_rankings = load_action_ranking_bundle(
        root / "ranking-bundles" / f"{role}-deployment.pt"
    )
    _matched_bundle_projection(source_carriers, deployment_carriers)
    _matched_ranking_projection(source_rankings, deployment_rankings)
    role_states = tuple(
        state for state in plan["states"] if state["exposure_role"] == role
    )
    if (
        len(source_carriers) != len(role_states)
        or len(source_rankings) != len(role_states)
        or any(item.exposure_role != role for item in source_carriers)
        or any(item.exposure_role != role for item in deployment_carriers)
        or any(item.exposure_role != role for item in source_rankings)
        or any(item.exposure_role != role for item in deployment_rankings)
        or any(
            item.source_release_identity != release_identity(str(plan["identity"]))
            for item in (*source_carriers, *deployment_carriers)
        )
    ):
        raise CorrectiveRankingCohortError(f"{role} product boundary differs")
    results_by_candidate = {
        item["candidate_identity"]: item
        for item in results if item["exposure_role"] == role
    }
    for state, source_lineage, source_ranking, deployment_ranking in zip(
        role_states,
        source_carriers,
        source_rankings,
        deployment_rankings,
        strict=True,
    ):
        anchor = state["candidates"][
            int(state["carrier_anchor_candidate_ordinal"]) - 1
        ]
        anchor_result = results_by_candidate[anchor["identity"]]
        expected_candidates = tuple(
            (
                candidate["identity"],
                float(results_by_candidate[candidate["identity"]]["realized_cost"]),
                candidate["drag_x"],
                candidate["drag_y"],
                candidate["tap_time_ms"],
            )
            for candidate in state["candidates"]
        )
        actual_candidates = tuple(
            (
                candidate.identity,
                float(candidate.realized_cost),
                candidate.interface_action.drag_x,
                candidate.interface_action.drag_y,
                candidate.interface_action.tap_time_ms,
            )
            for candidate in source_ranking.candidates
        )
        if (
            source_ranking.identity != state["identity"]
            or deployment_ranking.identity != state["identity"]
            or actual_candidates != expected_candidates
            or source_lineage.trajectory_identity
            != anchor_result["trajectory_identity"]
            or source_lineage.scenario_lineage_identity
            != anchor_result["scenario_lineage_identity"]
            or source_ranking.trajectory_identity
            != anchor_result["trajectory_identity"]
        ):
            raise CorrectiveRankingCohortError(
                f"{role} products differ at {state['identity']}"
            )
    _log(f"validate role={role} carrier/ranking products passed")


def validate_release(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output.resolve()
    plan = validate_plan(
        _load_json(root / "production-plan.json", "published production plan"),
        phase="production",
    )
    pilot_plan = _load_plan(args.pilot_plan, "pilot")
    pilot_report = validate_pilot_report(
        _load_json(root / "pilot-report.json", "published pilot report")
    )
    provenance = _load_json(root / "provenance.json", "published provenance")
    manifest = _load_json(root / "manifest.json", "corrective release manifest")
    results = []
    for state_index, state in enumerate(plan["states"], start=1):
        for candidate in state["candidates"]:
            path = (
                root / "candidate-results" / f"{_candidate_key(state, candidate)}.json"
            )
            result = _load_json(path, "published candidate result")
            trajectory_root = (
                None
                if result.get("status") == "failed"
                else root / _trajectory_relative(state, candidate)
            )
            results.append(_validate_result(
                result,
                plan,
                state,
                candidate,
                trajectory_root=trajectory_root,
            ))
        _log(
            f"validate trajectories state={state_index}/{len(plan['states'])} "
            f"role={state['exposure_role']} candidates=12/12"
        )
    result_files = tuple((root / "candidate-results").glob("*.json"))
    if len(result_files) != len(results):
        raise CorrectiveRankingCohortError(
            "published candidate result inventory differs"
        )
    for role in ROLES:
        _validate_role_products(plan, results, root=root, role=role)
        gc.collect()
    audit_root = (args.audit_output / "production").resolve()
    audit = _load_json(audit_root / "manifest.json", "production audit gallery")
    if (
        audit.get("plan_identity") != plan["identity"]
        or audit.get("candidate_count") != len(results)
        or len(audit.get("entries", ())) != len(results)
        or not (audit_root / "index.html").is_file()
    ):
        raise CorrectiveRankingCohortError("production audit gallery differs")
    audit_by_candidate = {
        item["candidate_identity"]: item for item in audit["entries"]
    }
    for result in results:
        entry = audit_by_candidate.get(result["candidate_identity"])
        if (
            entry is None
            or entry.get("status") != result["status"]
            or entry.get("realized_cost") != result["realized_cost"]
            or (
                entry.get("video_path") is not None
                and not (audit_root / str(entry["video_path"])).is_file()
            )
        ):
            raise CorrectiveRankingCohortError(
                "candidate audit entry differs from its execution"
            )
    disjointness = _external_disjointness(
        plan,
        results,
        pilot_plan=pilot_plan,
        issue_62_release=args.issue_62_release,
        issue_57_protocol=args.issue_57_protocol,
    )
    expected_manifest = _manifest(
        plan,
        results,
        pilot_report=pilot_report,
        audit_reference=_audit_reference(audit_root),
        audit=audit,
        disjointness=disjointness,
        provenance=provenance,
    )
    if manifest != expected_manifest:
        raise CorrectiveRankingCohortError("corrective release manifest differs")
    expected_summary = {
        "schema": "issue_68_corrective_cohort_summary_v2",
        "artifact_identity": manifest["identity"],
        "production_plan_identity": plan["identity"],
        "counts": manifest["counts"],
        "outcome_diversity_by_role": manifest["outcome_diversity_by_role"],
        "disjointness": manifest["disjointness"],
        "audit_gallery": manifest["artifacts"]["candidate_audit_gallery"],
        "model_selection_access_rule": manifest["model_selection_access_rule"],
        "final_evaluation_opened": False,
        "passed": True,
    }
    summary = _load_json(args.summary, "corrective compact summary")
    if summary != expected_summary:
        raise CorrectiveRankingCohortError("corrective compact summary differs")
    _log(
        f"validate exact release passed states={len(plan['states'])} "
        f"candidates={len(results)} final_evaluation=unopened"
    )
    return summary


def _synthetic_results(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    for state in plan["states"]:
        for candidate in state["candidates"]:
            cost = float(
                1000.2 + (candidate["ordinal"] % 3) * 0.1
                if state["ordinal"] < 18
                else 1000.9
            )
            results.append({
                "state_identity": state["identity"],
                "candidate_identity": candidate["identity"],
                "candidate_ordinal": candidate["ordinal"],
                "exposure_role": state["exposure_role"],
                "generator_family": state["generator_family"],
                "generation_seed": state["generation_seed"],
                "action_stratum": candidate["action_stratum"],
                "status": "accepted",
                "realized_cost": cost,
                "trajectory_identity": f"{state['identity']}:dry-trajectory",
                "scenario_lineage_identity": f"{state['identity']}:dry-lineage",
                "terminal_reason": "stable_entered",
                "wall_seconds": 1.0,
                "artifact_bytes": 1000,
                "failures": [],
                "interaction_coverage": ["collision"],
            })
    return results


def dry_run() -> dict[str, Any]:
    _log("dry-run 1/5: building the fresh role-isolated pilot in memory")
    pilot = build_pilot_plan()
    _log("dry-run 2/5: validating all 12 source-bound legal action strata")
    if any(len(state["candidates"]) != 12 for state in pilot["states"]):
        raise CorrectiveRankingCohortError("dry-run candidate design is incomplete")
    _log("dry-run 3/5: materializing one scenario from each generator family")
    with tempfile.TemporaryDirectory(prefix="novphy-issue68-dry-") as temporary:
        for family in {state["generator_family"] for state in pilot["states"]}:
            state = next(
                item for item in pilot["states"]
                if item["generator_family"] == family
            )
            _materialize_slot(
                _candidate_slot(state, state["candidates"][0]),
                Path(temporary) / family,
            )
    _log("dry-run 4/5: checking VP8/WebM audit support")
    _verify_webm_encoder()
    _log("dry-run 5/5: exercising pilot report and production freeze in memory")
    synthetic = _synthetic_results(pilot)
    report = _pilot_report(
        pilot,
        synthetic,
        {
            "candidate_count": len(synthetic),
            "video_count": len(synthetic),
            "accepted_video_count": len(synthetic),
        },
    )
    validate_pilot_report(report)
    production = build_production_plan(report)
    result = {
        "schema": "issue_68_corrective_cohort_dry_run_v2",
        "pilot_states": len(pilot["states"]),
        "pilot_candidates": len(synthetic),
        "action_strata": len(pilot["action_design"]["strata"]),
        "production_states": len(production["states"]),
        "production_candidates": len(production["states"]) * 12,
        "production_states_per_role": DEFAULT_PRODUCTION_STATES_PER_ROLE,
        "files_written": False,
        "unity_opened": False,
        "final_evaluation_opened": False,
        "passed": True,
    }
    _log("dry-run passed files_written=0 unity=unopened final_evaluation=unopened")
    return result


def _implementation_revision(value: str | None, *, required: bool) -> str:
    if value:
        return value
    revision, dirty = git_revision(str(ROOT))
    if dirty and required:
        raise CorrectiveRankingCohortError(
            "actual collection from a dirty tree requires --implementation-revision"
        )
    return revision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-pilot-plan", action="store_true")
    mode.add_argument("--run-pilot", action="store_true")
    mode.add_argument("--freeze-production", action="store_true")
    mode.add_argument("--run-production", action="store_true")
    mode.add_argument("--publish-production", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    parser.add_argument("--pilot-runtime", type=Path, default=DEFAULT_PILOT_RUNTIME)
    parser.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument(
        "--production-plan", type=Path, default=DEFAULT_PRODUCTION_PLAN
    )
    parser.add_argument(
        "--production-runtime", type=Path, default=DEFAULT_PRODUCTION_RUNTIME
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--issue-62-release", type=Path, default=DEFAULT_ISSUE_62_RELEASE
    )
    parser.add_argument(
        "--issue-57-protocol", type=Path, default=DEFAULT_ISSUE_57_PROTOCOL
    )
    parser.add_argument(
        "--states-per-role", type=int,
        default=DEFAULT_PRODUCTION_STATES_PER_ROLE,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="isolated processes for exact trajectory validation (default: 4)",
    )
    parser.add_argument("--implementation-revision")
    parser.add_argument("--start-display", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    args = _parser().parse_args(argv)
    if args.dry_run:
        result = dry_run()
    elif args.freeze_pilot_plan:
        result = freeze_pilot(args.pilot_plan)
    elif args.run_pilot:
        result = run_pilot(args)
    elif args.freeze_production:
        result = freeze_production(args)
    elif args.run_production:
        plan = _load_plan(args.production_plan, "production")
        results = run_collection(
            plan,
            runtime=args.production_runtime,
            audit=args.audit_output,
            implementation_revision=_implementation_revision(
                args.implementation_revision, required=True
            ),
            start_display_process=args.start_display,
            speed=args.speed,
            headless=args.headless,
            workers=args.workers,
        )
        result = {
            "schema": "issue_68_production_collection_status_v2",
            "production_plan_identity": plan["identity"],
            "scheduled": len(results),
            "accepted": sum(item["status"] == "accepted" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "final_evaluation_opened": False,
        }
    elif args.publish_production:
        result = publish_production(args)
    else:
        result = validate_release(args)
    displayed = result
    if result.get("schema") == PLAN_SCHEMA:
        displayed = {
            "schema": result["schema"],
            "identity": result["identity"],
            "phase": result["phase"],
            "role_counts": result["role_counts"],
            "state_count": len(result["states"]),
            "candidate_count": len(result["states"]) * 12,
            "resource_decision": result["resource_decision"],
            "final_evaluation_opened": False,
        }
    print(json.dumps(displayed, allow_nan=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        CorrectiveRankingCohortError,
        LineageScalingError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
