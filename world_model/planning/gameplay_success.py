"""Prospective protocol and evidence accounting for issue #57 gameplay trials."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
from io import StringIO
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from scripts.cohort_v2_scenarios import (
    write_immutable_cohort_v2_bytes,
    write_immutable_cohort_v2_json,
)
from world_model.planning.gameplay import ControlResult
from world_model.training.grid_artifacts import canonical_json_bytes


PROTOCOL_SCHEMA = "cohort_v2_gameplay_success_protocol_v1"
TRIAL_SCHEMA = "cohort_v2_gameplay_success_trial_v1"
AGGREGATE_SCHEMA = "cohort_v2_gameplay_success_aggregate_v1"
RUN_SCHEMA = "cohort_v2_gameplay_success_run_v1"
EVIDENCE_SCHEMA = "cohort_v2_gameplay_success_evidence_v1"
AUTHORIZATION_IDENTITY = "github-issue-authorization-v1:57:final-gameplay-evaluation"
PROTOCOL_FILENAME = "cohort-v2-gameplay-success-protocol-v1.json"

SYSTEM_IDS = (
    "random_legal",
    "heuristic_no_model",
    "open_loop_cem_adaptive",
    "repeated_h1_cem_mpc",
    "adaptive_cem_mpc",
)
FINAL_LEVEL_NUMBERS = (10, 20, 30, 40, 50)
ROLE_LEVEL_NUMBERS = {
    "smoke": (1, 11, 21, 31, 41),
    "training_tuning": (2, 12, 22, 32, 42),
    "calibration": (3, 13, 23, 33, 43),
    "model_selection": (4, 14, 24, 34, 44),
    "final_evaluation": FINAL_LEVEL_NUMBERS,
}
TRIAL_SEEDS = (20260831, 20260832, 20260833)
FAILURE_CLASSES = (
    "action_planning_failure",
    "model_rollout_failure",
    "game_execution_failure",
    "timeout",
    "invalid_action",
    "infrastructure_failure",
    "compute_limit",
    "level_terminal_failure",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def identity_digest(identity: str) -> str:
    return _sha256(identity.encode("utf-8"))


def _identity(namespace: str, payload: Mapping[str, Any]) -> str:
    return f"{namespace}:sha256:{_sha256(canonical_json_bytes(payload))}"


def _with_identity(
    namespace: str,
    field: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {**payload, field: _identity(namespace, payload)}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _player_levels(game_dir: Path) -> tuple[dict[str, Any], ...]:
    game_dir = Path(game_dir)
    config_path = game_dir / "9001_Data/StreamingAssets/config.xml"
    try:
        text = config_path.read_bytes().decode("utf-8")
        text = text.replace('encoding="utf-16"', 'encoding="utf-8"', 1)
        root = ET.fromstring(text)
    except (OSError, UnicodeDecodeError, ET.ParseError) as error:
        raise ValueError(f"cannot read packaged gameplay inventory: {error}") from error
    trials = []
    for trial in root.findall("./trials/trial"):
        trials.append(tuple(
            item.attrib.get("level_path", "")
            for item in trial.findall("./game_level_set/game_levels")
        ))
    if len(trials) != 2 or len(trials[0]) != 50 or trials[0] != trials[1]:
        raise ValueError("packaged gameplay inventory is not the frozen duplicated 50-level set")

    marker = "9001_Data/StreamingAssets/Levels/"
    records = []
    for number, declared_path in enumerate(trials[0], start=1):
        normalized = declared_path.replace("\\", "/")
        if marker not in normalized:
            raise ValueError("packaged gameplay level path is outside StreamingAssets/Levels")
        relative = marker + normalized.split(marker, 1)[1]
        path = game_dir / relative
        if not path.is_file():
            raise ValueError(f"packaged gameplay level is missing: {relative}")
        parts = Path(relative).parts
        try:
            novelty_level = next(value for value in parts if value.startswith("novelty_level_"))
            novelty_type = next(value for value in parts if value.startswith("type"))
        except StopIteration as error:
            raise ValueError("packaged gameplay level lacks benchmark condition") from error
        content_digest = _sha256(path.read_bytes())
        records.append({
            "level_number": number,
            "source_path": relative,
            "source_content_sha256": content_digest,
            "level_identity": f"science-birds-level-v1:sha256:{content_digest}",
            "condition_identity": (
                f"benchmark-condition-v1:{novelty_level}:{novelty_type}"
            ),
            "novelty_level": novelty_level,
            "novelty_type": novelty_type,
        })
    return tuple(records)


def _systems() -> list[dict[str, Any]]:
    return [
        {
            "system_id": "random_legal",
            "planner": "random",
            "control_mode": "mpc",
            "sequence_length": 2,
            "uses_world_model": False,
            "fixed_prediction_pair": None,
        },
        {
            "system_id": "heuristic_no_model",
            "planner": "heuristic",
            "control_mode": "mpc",
            "sequence_length": 2,
            "uses_world_model": False,
            "fixed_prediction_pair": None,
        },
        {
            "system_id": "open_loop_cem_adaptive",
            "planner": "cem",
            "control_mode": "open_loop",
            "sequence_length": 6,
            "uses_world_model": True,
            "fixed_prediction_pair": None,
        },
        {
            "system_id": "repeated_h1_cem_mpc",
            "planner": "cem",
            "control_mode": "mpc",
            "sequence_length": 2,
            "uses_world_model": True,
            "fixed_prediction_pair": {"horizon": 1, "abstraction": "continuous"},
        },
        {
            "system_id": "adaptive_cem_mpc",
            "planner": "cem",
            "control_mode": "mpc",
            "sequence_length": 2,
            "uses_world_model": True,
            "fixed_prediction_pair": None,
        },
    ]


def build_trial_schedule(
    systems: Sequence[Mapping[str, Any]],
    levels: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    exposure_role: str = "final_evaluation",
) -> list[dict[str, Any]]:
    system_ids = [str(value["system_id"]) for value in systems]
    schedule = []
    for seed_index, seed in enumerate(seeds):
        for level_index, level in enumerate(levels):
            offset = (seed_index + level_index) % len(system_ids)
            ordered = system_ids[offset:] + system_ids[:offset]
            for system_id in ordered:
                index = len(schedule) + 1
                schedule.append({
                    "trial_index": index,
                    "trial_id": f"issue-57-trial-{index:03d}",
                    "system_id": system_id,
                    "seed": int(seed),
                    "exposure_role": exposure_role,
                    "level_number": int(level["level_number"]),
                    "level_identity": str(level["level_identity"]),
                    "condition_identity": str(level["condition_identity"]),
                })
    return schedule


def build_protocol(
    game_dir: Path,
    planning_evidence: Mapping[str, Any],
    migration_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    levels = _player_levels(game_dir)
    by_number = {int(value["level_number"]): value for value in levels}
    source = planning_evidence.get("source_bindings")
    if not isinstance(source, Mapping):
        raise ValueError("issue-56 planning evidence has no source bindings")
    systems = _systems()
    roles = {
        role: [by_number[number] for number in numbers]
        for role, numbers in ROLE_LEVEL_NUMBERS.items()
    }
    schedule = build_trial_schedule(
        systems, roles["final_evaluation"], TRIAL_SEEDS
    )
    executable = Path(game_dir) / "9001.x86_64"
    interface = Path(game_dir) / "game_playing_interface.jar"
    if not executable.is_file() or not interface.is_file():
        raise ValueError("gameplay player executable or interface is missing")

    payload = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_version": 1,
        "frozen_before_final_access": True,
        "authorization_identity": AUTHORIZATION_IDENTITY,
        "source_bindings": {
            "issue_56_planning_artifact_identity": planning_evidence.get(
                "artifact_identity"
            ),
            "issue_56_implementation_revision": source.get(
                "implementation_revision"
            ),
            "world_model_checkpoint_identity_sha256": identity_digest(str(
                source.get("world_model_checkpoint_identity")
            )),
            "controller_checkpoint_identity_sha256": identity_digest(str(
                source.get("controller_checkpoint_identity")
            )),
            "visual_parser_checkpoint_identity_sha256": identity_digest(str(
                source.get("visual_parser_checkpoint_identity")
            )),
            "observation_adapter_identity": source.get(
                "observation_adapter_identity"
            ),
            "goal_cost_version": source.get("goal_cost_version"),
            "environment_version": source.get("environment_version"),
            "migration_recovery_manifest_identity": migration_manifest.get(
                "identity"
            ),
            "player_executable_sha256": _sha256(executable.read_bytes()),
            "game_interface_sha256": _sha256(interface.read_bytes()),
            "player_config_sha256": _sha256((
                Path(game_dir) / "9001_Data/StreamingAssets/config.xml"
            ).read_bytes()),
        },
        "claim_boundary": {
            "primary_question": (
                "whether frozen adaptive CEM/MPC improves held-out level success "
                "over the strongest declared gameplay comparator"
            ),
            "held_out_scope": "packaged instance-held-out gameplay levels",
            "excluded_claims": [
                "human-level gameplay",
                "unseen-physics generalization",
                "perception novelty",
                "changes to issue-1, issue-14, issue-15, or issue-17 conclusions",
            ],
        },
        "level_inventory": {
            "source": "packaged Science Birds config.xml trial 0; trial 1 is identical",
            "roles": roles,
            "unused_level_numbers": [
                number
                for number in range(1, 51)
                if all(number not in values for values in ROLE_LEVEL_NUMBERS.values())
            ],
            "isolation_rule": (
                "level identities are disjoint across smoke, training/tuning, "
                "calibration, model selection, and final evaluation; final levels "
                "cannot change any checkpoint, parser, cost, planner, metric, or limit"
            ),
        },
        "systems": systems,
        "trial_seeds": list(TRIAL_SEEDS),
        "trial_schedule": schedule,
        "action_bounds": {
            "drag_x": [-160, -40],
            "drag_y": [-80, 80],
            "tap_time_ms": [0, 1000],
            "release_time_ms": 600,
        },
        "cem": {
            "population_size": 64,
            "elite_count": 8,
            "iterations": 5,
            "minimum_std": 1.0,
        },
        "execution_limits": {
            "max_shots": 6,
            "max_planner_compute": 100000000000000.0,
            "fixed_steps_per_shot": 15,
            "max_trial_wall_clock_seconds": 300.0,
            "retry_count": 0,
        },
        "metrics": {
            "success": "WON before any frozen execution limit",
            "shots_to_success": (
                "executed shots for successful trials; unsuccessful trials remain "
                "right-censored and receive max_shots+1 only in the declared "
                "penalized-shots estimand"
            ),
            "uncertainty": "two-sided 95% Wilson success intervals and paired bootstrap intervals",
            "compute": [
                "model rollout count and declared multiply-accumulate compute",
                "CEM candidate, goal-evaluation, replan, and inclusive planner wall time",
                "game-interface inclusive and parser-exclusive wall time",
                "observation/parser call count and wall time",
            ],
            "prediction": [
                "requested/effective horizon counts",
                "requested abstraction counts",
                "repeated h1 transition count",
                "accumulated executed-transition recursive rollout error",
            ],
        },
        "failure_policy": {
            "taxonomy": list(FAILURE_CLASSES),
            "denominator": "every scheduled trial, including every failure",
            "exclusions": "none",
            "missing_trial": "artifact remains incomplete and cannot claim a conclusion",
            "stopping": (
                "run the fixed matrix once; no early success stop, replacement trial, "
                "seed change, level change, or outcome-conditioned rerun"
            ),
        },
        "analysis": {
            "confidence_level": 0.95,
            "bootstrap_seed": 20261201,
            "bootstrap_replicates": 10000,
            "primary_system": "adaptive_cem_mpc",
            "strongest_comparator_candidates": [
                "random_legal",
                "heuristic_no_model",
                "open_loop_cem_adaptive",
                "repeated_h1_cem_mpc",
            ],
            "practical_success_rate_margin": 0.10,
            "supported_rule": (
                "adaptive_cem_mpc is supported only when its paired success-rate "
                "gain over the strongest declared comparator is at least 0.10, "
                "the paired 95% bootstrap lower bound is above 0, and its mean "
                "penalized shots is no larger; otherwise the disposition is "
                "not_supported_by_this_experiment"
            ),
        },
        "visual_parser_handoff": {
            "structure_unstable_training_prevalence": "28/2150",
            "structure_unstable_final_confusion": {
                "true_positive": 14,
                "false_positive": 1590,
                "true_negative": 0,
                "false_negative": 0,
            },
            "structure_unstable_direct_cost_weight": 0.0,
            "logging_rule": (
                "record raw probability, thresholded value, selected abstraction, "
                "and whether visual carrier or macro symbols enter action ranking/execution"
            ),
            "retraining": "forbidden on every gameplay level and outcome",
        },
        "required_outputs": {
            "trial_records": "one immutable JSON record per scheduled trial",
            "tables": [
                "success_rates.csv",
                "shots_to_success.csv",
                "compute_failures.csv",
            ],
            "plots": ["success-rate.svg", "shots-to-success.svg"],
            "machine_readable": [
                "trials.jsonl",
                "aggregate.json",
                "exposure-audit.json",
                "evidence.json",
            ],
        },
        "rerun_commands": [
            "python -u -m scripts.run_issue_57_gameplay_success --dry-run",
            (
                "python -u -m scripts.run_issue_57_gameplay_success --run-final "
                "--start-display --start-engine "
                f"--authorization-identity {AUTHORIZATION_IDENTITY}"
            ),
            "python -u -m scripts.run_issue_57_gameplay_success --aggregate",
            "python -u -m scripts.run_issue_57_gameplay_success --validate",
        ],
    }
    return _with_identity(
        "cohort-v2-gameplay-success-protocol-v1",
        "protocol_identity",
        payload,
    )


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    game_dir: Path | None = None,
) -> dict[str, Any]:
    value = dict(protocol)
    identity = value.pop("protocol_identity", None)
    if (
        value.get("schema") != PROTOCOL_SCHEMA
        or value.get("protocol_version") != 1
        or value.get("frozen_before_final_access") is not True
        or value.get("authorization_identity") != AUTHORIZATION_IDENTITY
        or identity != _identity("cohort-v2-gameplay-success-protocol-v1", value)
    ):
        raise ValueError("gameplay-success protocol identity or freeze differs")
    roles = value.get("level_inventory", {}).get("roles", {})
    if set(roles) != set(ROLE_LEVEL_NUMBERS):
        raise ValueError("gameplay-success exposure roles differ")
    if any(
        tuple(item.get("level_number") for item in roles[role]) != numbers
        for role, numbers in ROLE_LEVEL_NUMBERS.items()
    ):
        raise ValueError("gameplay-success exposure-role level inventory differs")
    role_identities = {
        role: {item.get("level_identity") for item in items}
        for role, items in roles.items()
    }
    all_identities = [item for values in role_identities.values() for item in values]
    if len(all_identities) != len(set(all_identities)):
        raise ValueError("gameplay-success level identities leak across roles")
    systems = value.get("systems")
    if systems != _systems():
        raise ValueError("gameplay-success system matrix differs")
    seeds = value.get("trial_seeds")
    if seeds != list(TRIAL_SEEDS):
        raise ValueError("gameplay-success seed policy differs")
    expected_schedule = build_trial_schedule(
        systems, roles["final_evaluation"], seeds
    )
    if value.get("trial_schedule") != expected_schedule:
        raise ValueError("gameplay-success trial schedule differs")
    if value.get("failure_policy", {}).get("exclusions") != "none":
        raise ValueError("gameplay-success outcome exclusions are forbidden")
    if value.get("action_bounds") != {
        "drag_x": [-160, -40],
        "drag_y": [-80, 80],
        "tap_time_ms": [0, 1000],
        "release_time_ms": 600,
    } or value.get("cem") != {
        "population_size": 64,
        "elite_count": 8,
        "iterations": 5,
        "minimum_std": 1.0,
    } or value.get("execution_limits") != {
        "max_shots": 6,
        "max_planner_compute": 100000000000000.0,
        "fixed_steps_per_shot": 15,
        "max_trial_wall_clock_seconds": 300.0,
        "retry_count": 0,
    }:
        raise ValueError("gameplay-success action, search, or execution limits differ")
    if value.get("visual_parser_handoff", {}).get("retraining") != (
        "forbidden on every gameplay level and outcome"
    ):
        raise ValueError("gameplay-success parser retraining boundary differs")
    if game_dir is not None:
        packaged = {item["level_number"]: item for item in _player_levels(game_dir)}
        for items in roles.values():
            for item in items:
                if packaged.get(item["level_number"]) != item:
                    raise ValueError("gameplay-success packaged level identity differs")
        source = value["source_bindings"]
        executable = Path(game_dir) / "9001.x86_64"
        interface = Path(game_dir) / "game_playing_interface.jar"
        config = Path(game_dir) / "9001_Data/StreamingAssets/config.xml"
        if (
            _sha256(executable.read_bytes()) != source["player_executable_sha256"]
            or _sha256(interface.read_bytes()) != source["game_interface_sha256"]
            or _sha256(config.read_bytes()) != source["player_config_sha256"]
        ):
            raise ValueError("gameplay-success player binding differs")
    return {**value, "protocol_identity": identity}


def load_protocol(path: Path, *, game_dir: Path | None = None) -> dict[str, Any]:
    value = _load_object(path, "gameplay-success protocol")
    expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if Path(path).read_bytes() != expected:
        raise ValueError("gameplay-success protocol is not canonical")
    return validate_protocol(value, game_dir=game_dir)


def write_protocol(protocol: Mapping[str, Any], path: Path) -> Path:
    validate_protocol(protocol)
    return write_immutable_cohort_v2_json(dict(protocol), path)


def stack_identity_bindings(
    world_model_checkpoint_identity: str,
    controller_checkpoint_identity: str,
    visual_parser_checkpoint_identity: str,
    observation_adapter_identity: str,
) -> dict[str, str]:
    return {
        "world_model_checkpoint_identity_sha256": identity_digest(
            world_model_checkpoint_identity
        ),
        "controller_checkpoint_identity_sha256": identity_digest(
            controller_checkpoint_identity
        ),
        "visual_parser_checkpoint_identity_sha256": identity_digest(
            visual_parser_checkpoint_identity
        ),
        "observation_adapter_identity": observation_adapter_identity,
    }


def validate_stack_bindings(
    protocol: Mapping[str, Any], bindings: Mapping[str, str]
) -> None:
    source = protocol["source_bindings"]
    if any(source.get(key) != value for key, value in bindings.items()):
        raise ValueError("gameplay-success frozen model/controller/parser stack differs")


def _unique_plans(result: ControlResult) -> list[Any]:
    plans = []
    seen = set()
    for step in result.steps:
        marker = id(step.plan)
        if marker not in seen:
            plans.append(step.plan)
            seen.add(marker)
    return plans


def _failure_counts(
    result: ControlResult | None,
    *,
    timed_out: bool,
    infrastructure_failure: str | None,
    pre_control_failure_class: str,
) -> dict[str, int]:
    counts = {key: 0 for key in FAILURE_CLASSES}
    if infrastructure_failure is not None:
        counts[pre_control_failure_class] = 1
        return counts
    assert result is not None
    if timed_out:
        counts["timeout"] = 1
    elif result.termination_reason == "planner_failure":
        counts["action_planning_failure"] = 1
    elif result.termination_reason == "game_interface_failure":
        counts["game_execution_failure"] = 1
    elif result.termination_reason == "timeout":
        counts["timeout"] = 1
    elif result.termination_reason == "compute_limit":
        counts["compute_limit"] = 1
    elif result.termination_reason == "terminal_failure":
        counts["level_terminal_failure"] = 1
    counts["invalid_action"] = int(result.invalid_candidate_count)
    counts["model_rollout_failure"] = sum(
        failure is not None
        for plan in _unique_plans(result)
        for iteration in plan.iterations
        for failure in iteration.candidate_failures
    )
    return counts


def build_trial_record(
    protocol: Mapping[str, Any],
    schedule_entry: Mapping[str, Any],
    *,
    implementation_revision: str,
    stack_bindings: Mapping[str, str],
    result: ControlResult | None,
    observation_parser_call_count: int = 0,
    observation_parser_wall_clock_seconds: float = 0.0,
    infrastructure_failure: str | None = None,
    pre_control_failure_class: str = "infrastructure_failure",
) -> dict[str, Any]:
    validate_stack_bindings(protocol, stack_bindings)
    systems = {item["system_id"]: item for item in protocol["systems"]}
    system = systems[str(schedule_entry["system_id"])]
    max_shots = int(protocol["execution_limits"]["max_shots"])
    max_wall = float(protocol["execution_limits"]["max_trial_wall_clock_seconds"])
    elapsed = 0.0 if result is None else float(result.wall_clock_seconds)
    timed_out = result is not None and elapsed > max_wall
    success = bool(result is not None and result.success and not timed_out)
    executed_shots = 0 if result is None else len(result.steps)
    plans = [] if result is None else _unique_plans(result)
    requested_horizons = Counter(
        horizon
        for plan in plans
        for horizon in plan.selected_evaluation.requested_horizons
    )
    effective_horizons = Counter(
        horizon
        for plan in plans
        for horizon in plan.selected_evaluation.effective_horizons
    )
    abstractions = Counter(
        abstraction
        for plan in plans
        for abstraction in plan.selected_evaluation.requested_abstractions
    )
    recursive_errors = [] if result is None else [
        step.recursive_rollout_error
        for step in result.steps
        if step.recursive_rollout_error is not None
    ]
    observations: dict[str, Mapping[str, Any]] = {}
    if result is not None:
        for step in result.steps:
            if step.observation_before_diagnostics is not None:
                observations.setdefault(
                    step.observation_before, step.observation_before_diagnostics
                )
            if step.observation_after_diagnostics is not None:
                observations.setdefault(
                    step.observation_after, step.observation_after_diagnostics
                )
    probabilities = [
        float(value["structure_unstable_probability"])
        for value in observations.values()
        if isinstance(value.get("structure_unstable_probability"), (int, float))
    ]
    thresholded = [
        bool(value["structure_unstable_thresholded"])
        for value in observations.values()
        if isinstance(value.get("structure_unstable_thresholded"), bool)
    ]
    failure_counts = _failure_counts(
        result,
        timed_out=timed_out,
        infrastructure_failure=infrastructure_failure,
        pre_control_failure_class=pre_control_failure_class,
    )
    termination = (
        pre_control_failure_class
        if infrastructure_failure is not None
        else "timeout"
        if timed_out
        else str(result.termination_reason)
    )
    source_failures = [infrastructure_failure] if infrastructure_failure else (
        [] if result is None else list(result.failures)
    )
    action_trace = [] if result is None else [
        {
            "shot_index": step.shot_index,
            "observation_before": step.observation_before,
            "observation_after": step.observation_after,
            "executed_action": {
                "drag_x": step.executed_action.drag_x,
                "drag_y": step.executed_action.drag_y,
                "tap_time_ms": step.executed_action.tap_time_ms,
            },
            "planner_seed": step.plan.seed,
            "selected_cost": step.plan.selected_evaluation.total_cost,
            "requested_horizons": list(
                step.plan.selected_evaluation.requested_horizons
            ),
            "effective_horizons": list(
                step.plan.selected_evaluation.effective_horizons
            ),
            "requested_abstractions": list(
                step.plan.selected_evaluation.requested_abstractions
            ),
            "recursive_rollout_error": step.recursive_rollout_error,
        }
        for step in result.steps
    ]
    payload = {
        "schema": TRIAL_SCHEMA,
        "protocol_identity": protocol["protocol_identity"],
        "implementation_revision": implementation_revision,
        "stack_bindings": dict(stack_bindings),
        "schedule": dict(schedule_entry),
        "system_configuration": system,
        "outcome": {
            "success": success,
            "termination_reason": termination,
            "included_in_denominator": True,
            "excluded": False,
            "censored_unsuccessful": not success,
            "executed_shot_count": executed_shots,
            "shots_to_success": executed_shots if success else None,
            "penalized_shots": executed_shots if success else max_shots + 1,
        },
        "failures": {
            "taxonomy_counts": failure_counts,
            "messages": source_failures,
        },
        "compute": {
            "candidate_count": 0 if result is None else result.candidate_count,
            "invalid_candidate_count": (
                0 if result is None else result.invalid_candidate_count
            ),
            "model_rollout_count": (
                0 if result is None else result.model_rollout_count
            ),
            "world_model_declared_compute": (
                0.0 if result is None else result.planner_compute
            ),
            "goal_evaluation_count": (
                0 if result is None else result.goal_evaluation_count
            ),
            "replan_count": 0 if result is None else result.replan_count,
            "planner_wall_clock_seconds": (
                0.0 if result is None else result.planner_wall_clock_seconds
            ),
            "game_interface_inclusive_wall_clock_seconds": (
                0.0 if result is None else result.game_interface_wall_clock_seconds
            ),
            "game_interface_exclusive_estimate_seconds": max(
                0.0,
                (0.0 if result is None else result.game_interface_wall_clock_seconds)
                - observation_parser_wall_clock_seconds,
            ),
            "observation_parser_call_count": observation_parser_call_count,
            "observation_parser_wall_clock_seconds": (
                observation_parser_wall_clock_seconds
            ),
            "end_to_end_wall_clock_seconds": elapsed,
        },
        "prediction_diagnostics": {
            "requested_horizon_counts": {
                str(key): requested_horizons[key] for key in sorted(requested_horizons)
            },
            "effective_horizon_counts": {
                str(key): effective_horizons[key] for key in sorted(effective_horizons)
            },
            "requested_abstraction_counts": {
                str(key): abstractions[key] for key in sorted(abstractions)
            },
            "repeated_h1_transition_count": requested_horizons[1],
            "executed_transition_error_count": len(recursive_errors),
            "accumulated_recursive_rollout_error": sum(recursive_errors),
        },
        "parser_usage": {
            "observation_count": len(observations),
            "structure_unstable_probabilities": probabilities,
            "structure_unstable_thresholded": thresholded,
            "structure_unstable_direct_cost_weight": 0.0,
            "visual_carrier_affects_action_ranking": bool(system["uses_world_model"]),
            "symbolic_parser_may_affect_action_ranking": bool(
                abstractions["micro"] or abstractions["macro"]
            ),
            "structure_unstable_enters_selected_macro_rollout": bool(
                abstractions["macro"]
            ),
            "parser_outputs_affect_execution_path": bool(
                system["uses_world_model"] and executed_shots
            ),
            "outcome_conditioned_retraining": False,
        },
        "action_trace": action_trace,
    }
    return _with_identity(
        "cohort-v2-gameplay-success-trial-v1",
        "evidence_identity",
        payload,
    )


def validate_trial_record(
    protocol: Mapping[str, Any],
    record: Mapping[str, Any],
    schedule_entry: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(record)
    identity = value.pop("evidence_identity", None)
    if (
        value.get("schema") != TRIAL_SCHEMA
        or value.get("protocol_identity") != protocol["protocol_identity"]
        or value.get("schedule") != dict(schedule_entry)
        or identity != _identity("cohort-v2-gameplay-success-trial-v1", value)
    ):
        raise ValueError("gameplay-success trial identity or schedule differs")
    validate_stack_bindings(protocol, value.get("stack_bindings", {}))
    outcome = value.get("outcome", {})
    if (
        outcome.get("included_in_denominator") is not True
        or outcome.get("excluded") is not False
    ):
        raise ValueError("gameplay-success trial denominator or exclusion differs")
    max_shots = protocol["execution_limits"]["max_shots"]
    if outcome.get("success"):
        if outcome.get("shots_to_success") != outcome.get("executed_shot_count"):
            raise ValueError("successful gameplay trial shots-to-success differs")
    elif (
        outcome.get("shots_to_success") is not None
        or outcome.get("penalized_shots") != max_shots + 1
        or outcome.get("censored_unsuccessful") is not True
    ):
        raise ValueError("unsuccessful gameplay trial censoring differs")
    parser = value.get("parser_usage", {})
    if (
        parser.get("structure_unstable_direct_cost_weight") != 0.0
        or parser.get("outcome_conditioned_retraining") is not False
    ):
        raise ValueError("gameplay-success parser handoff differs")
    return {**value, "evidence_identity": identity}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * probability)])


def wilson_interval(successes: int, denominator: int) -> tuple[float, float]:
    if denominator <= 0 or not 0 <= successes <= denominator:
        raise ValueError("Wilson interval counts are invalid")
    z = 1.959963984540054
    rate = successes / denominator
    scale = 1.0 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(
        rate * (1.0 - rate) / denominator + z * z / (4 * denominator * denominator)
    ) / scale
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, replicates: int
) -> tuple[float, float]:
    if not values or replicates <= 0:
        raise ValueError("bootstrap inputs are invalid")
    rng = random.Random(seed)
    estimates = [
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(replicates)
    ]
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _system_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    unsuccessful_penalty: int,
) -> dict[str, Any]:
    successes = sum(bool(value["outcome"]["success"]) for value in records)
    denominator = len(records)
    lower, upper = wilson_interval(successes, denominator)
    penalized = [float(value["outcome"]["penalized_shots"]) for value in records]
    shots_lower, shots_upper = bootstrap_mean_interval(
        penalized, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    successful_shots = [
        int(value["outcome"]["shots_to_success"])
        for value in records
        if value["outcome"]["success"]
    ]
    failure_counts = {
        key: sum(value["failures"]["taxonomy_counts"][key] for value in records)
        for key in FAILURE_CLASSES
    }
    compute_fields = tuple(records[0]["compute"])
    return {
        "denominator": denominator,
        "success_count": successes,
        "success_rate": successes / denominator,
        "success_rate_wilson_95": [lower, upper],
        "unsuccessful_censored_count": denominator - successes,
        "successful_shots_to_success": {
            "count": len(successful_shots),
            "mean": (
                statistics.mean(successful_shots) if successful_shots else None
            ),
            "median": (
                statistics.median(successful_shots) if successful_shots else None
            ),
        },
        "penalized_shots": {
            "unsuccessful_value": unsuccessful_penalty,
            "mean": statistics.mean(penalized),
            "bootstrap_95": [shots_lower, shots_upper],
        },
        "failure_counts": failure_counts,
        "compute": {
            field: {
                "total": sum(float(value["compute"][field]) for value in records),
                "mean_per_trial": statistics.mean(
                    float(value["compute"][field]) for value in records
                ),
            }
            for field in compute_fields
        },
        "prediction": {
            "repeated_h1_transition_count": sum(
                value["prediction_diagnostics"]["repeated_h1_transition_count"]
                for value in records
            ),
            "executed_transition_error_count": sum(
                value["prediction_diagnostics"]["executed_transition_error_count"]
                for value in records
            ),
            "accumulated_recursive_rollout_error": sum(
                value["prediction_diagnostics"]["accumulated_recursive_rollout_error"]
                for value in records
            ),
        },
        "parser_usage": {
            "structure_unstable_thresholded_true_count": sum(
                sum(value["parser_usage"]["structure_unstable_thresholded"])
                for value in records
            ),
            "macro_rollout_trial_count": sum(
                value["parser_usage"]["structure_unstable_enters_selected_macro_rollout"]
                for value in records
            ),
            "ranking_dependency_trial_count": sum(
                value["parser_usage"]["visual_carrier_affects_action_ranking"]
                for value in records
            ),
            "execution_dependency_trial_count": sum(
                value["parser_usage"]["parser_outputs_affect_execution_path"]
                for value in records
            ),
        },
    }


def aggregate_trials(
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    schedule = protocol["trial_schedule"]
    by_trial = {value["schedule"]["trial_id"]: value for value in records}
    if len(by_trial) != len(records) or set(by_trial) != {
        value["trial_id"] for value in schedule
    }:
        raise ValueError("gameplay-success trial matrix is incomplete or duplicated")
    validated = [
        validate_trial_record(protocol, by_trial[item["trial_id"]], item)
        for item in schedule
    ]
    if len({value["implementation_revision"] for value in validated}) != 1:
        raise ValueError("gameplay-success trial implementation revisions differ")
    analysis = protocol["analysis"]
    replicates = int(analysis["bootstrap_replicates"])
    master_seed = int(analysis["bootstrap_seed"])
    unsuccessful_penalty = int(protocol["execution_limits"]["max_shots"]) + 1
    system_summaries = {}
    for system_index, system_id in enumerate(SYSTEM_IDS):
        selected = [
            value for value in validated
            if value["schedule"]["system_id"] == system_id
        ]
        system_summaries[system_id] = _system_summary(
            selected,
            bootstrap_seed=master_seed + system_index,
            bootstrap_replicates=replicates,
            unsuccessful_penalty=unsuccessful_penalty,
        )
    condition_summaries = {}
    conditions = sorted({
        value["schedule"]["condition_identity"] for value in validated
    })
    for condition in conditions:
        condition_summaries[condition] = {}
        for system_index, system_id in enumerate(SYSTEM_IDS):
            selected = [
                value for value in validated
                if value["schedule"]["condition_identity"] == condition
                and value["schedule"]["system_id"] == system_id
            ]
            condition_summaries[condition][system_id] = _system_summary(
                selected,
                bootstrap_seed=master_seed + 100 + system_index,
                bootstrap_replicates=replicates,
                unsuccessful_penalty=unsuccessful_penalty,
            )

    primary = str(analysis["primary_system"])
    comparator_candidates = list(analysis["strongest_comparator_candidates"])
    strongest = max(
        comparator_candidates,
        key=lambda value: (
            system_summaries[value]["success_rate"],
            -comparator_candidates.index(value),
        ),
    )
    paired: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for value in validated:
        key = (
            value["schedule"]["level_identity"],
            value["schedule"]["seed"],
        )
        paired.setdefault(key, {})[value["schedule"]["system_id"]] = value
    success_differences = [
        float(pair[primary]["outcome"]["success"])
        - float(pair[strongest]["outcome"]["success"])
        for pair in paired.values()
    ]
    shots_differences = [
        float(pair[primary]["outcome"]["penalized_shots"])
        - float(pair[strongest]["outcome"]["penalized_shots"])
        for pair in paired.values()
    ]
    success_interval = bootstrap_mean_interval(
        success_differences,
        seed=master_seed + 1000,
        replicates=replicates,
    )
    shots_interval = bootstrap_mean_interval(
        shots_differences,
        seed=master_seed + 1001,
        replicates=replicates,
    )
    success_difference = statistics.mean(success_differences)
    shots_difference = statistics.mean(shots_differences)
    supported = (
        success_difference >= float(analysis["practical_success_rate_margin"])
        and success_interval[0] > 0.0
        and shots_difference <= 0.0
    )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "protocol_identity": protocol["protocol_identity"],
        "trial_count": len(validated),
        "scheduled_trial_count": len(schedule),
        "all_trials_included": True,
        "systems": system_summaries,
        "conditions": condition_summaries,
        "primary_comparison": {
            "primary_system": primary,
            "strongest_comparator": strongest,
            "paired_unit_count": len(paired),
            "success_rate_difference": success_difference,
            "success_rate_difference_bootstrap_95": list(success_interval),
            "penalized_shots_mean_difference": shots_difference,
            "penalized_shots_difference_bootstrap_95": list(shots_interval),
            "practical_success_rate_margin": analysis[
                "practical_success_rate_margin"
            ],
        },
        "gameplay_conclusion": (
            "supported" if supported else "not_supported_by_this_experiment"
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    return _with_identity(
        "cohort-v2-gameplay-success-aggregate-v1",
        "artifact_identity",
        payload,
    )


def exposure_audit(
    protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    roles = protocol["level_inventory"]["roles"]
    final_ids = {item["level_identity"] for item in roles["final_evaluation"]}
    nonfinal_ids = {
        item["level_identity"]
        for role, items in roles.items()
        if role != "final_evaluation"
        for item in items
    }
    observed = {value["schedule"]["level_identity"] for value in records}
    passed = (
        not final_ids & nonfinal_ids
        and observed == final_ids
        and all(value["schedule"]["exposure_role"] == "final_evaluation" for value in records)
        and all(not value["parser_usage"]["outcome_conditioned_retraining"] for value in records)
    )
    return {
        "schema": "cohort_v2_gameplay_success_exposure_audit_v1",
        "protocol_identity": protocol["protocol_identity"],
        "authorization_identity": protocol["authorization_identity"],
        "workflow_authorization_access_count": 1,
        "final_level_identities": sorted(final_ids),
        "observed_final_level_identities": sorted(observed),
        "nonfinal_level_identities": sorted(nonfinal_ids),
        "final_outcomes_influenced_world_model": False,
        "final_outcomes_influenced_parser": False,
        "final_outcomes_influenced_goal_cost": False,
        "final_outcomes_influenced_planner": False,
        "outcome_conditioned_retraining": False,
        "passed": passed,
    }


def _csv_bytes(rows: Sequence[Sequence[Any]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def aggregate_tables(aggregate: Mapping[str, Any]) -> dict[str, bytes]:
    success_rows: list[list[Any]] = [[
        "scope", "condition", "system", "successes", "denominator",
        "success_rate", "wilson_95_lower", "wilson_95_upper",
    ]]
    for system_id, value in aggregate["systems"].items():
        success_rows.append([
            "overall", "all", system_id, value["success_count"], value["denominator"],
            value["success_rate"], *value["success_rate_wilson_95"],
        ])
    for condition, systems in aggregate["conditions"].items():
        for system_id, value in systems.items():
            success_rows.append([
                "condition", condition, system_id, value["success_count"],
                value["denominator"], value["success_rate"],
                *value["success_rate_wilson_95"],
            ])
    shots_rows: list[list[Any]] = [[
        "system", "successful_count", "successful_mean", "successful_median",
        "censored_count", "penalized_mean", "bootstrap_95_lower",
        "bootstrap_95_upper",
    ]]
    compute_rows: list[list[Any]] = [[
        "system", "candidate_count", "model_rollout_count", "declared_compute",
        "replan_count", "planner_wall_seconds", "game_interface_wall_seconds",
        "observation_parser_wall_seconds", *FAILURE_CLASSES,
    ]]
    for system_id, value in aggregate["systems"].items():
        successful = value["successful_shots_to_success"]
        penalized = value["penalized_shots"]
        shots_rows.append([
            system_id, successful["count"], successful["mean"], successful["median"],
            value["unsuccessful_censored_count"], penalized["mean"],
            *penalized["bootstrap_95"],
        ])
        compute = value["compute"]
        compute_rows.append([
            system_id,
            compute["candidate_count"]["total"],
            compute["model_rollout_count"]["total"],
            compute["world_model_declared_compute"]["total"],
            compute["replan_count"]["total"],
            compute["planner_wall_clock_seconds"]["total"],
            compute["game_interface_inclusive_wall_clock_seconds"]["total"],
            compute["observation_parser_wall_clock_seconds"]["total"],
            *(value["failure_counts"][key] for key in FAILURE_CLASSES),
        ])
    return {
        "success_rates.csv": _csv_bytes(success_rows),
        "shots_to_success.csv": _csv_bytes(shots_rows),
        "compute_failures.csv": _csv_bytes(compute_rows),
    }


def _bar_plot(
    aggregate: Mapping[str, Any], *, metric: str, title: str
) -> bytes:
    width, height = 900, 420
    left, top, plot_width, plot_height = 90, 55, 760, 285
    values = []
    for system_id in SYSTEM_IDS:
        summary = aggregate["systems"][system_id]
        if metric == "success_rate":
            value = float(summary["success_rate"])
            lower, upper = summary["success_rate_wilson_95"]
            maximum = 1.0
        else:
            value = float(summary["penalized_shots"]["mean"])
            lower, upper = summary["penalized_shots"]["bootstrap_95"]
            maximum = max(
                float(item["penalized_shots"]["unsuccessful_value"])
                for item in aggregate["systems"].values()
            )
        values.append((system_id, value, float(lower), float(upper)))
    bar_width = plot_width / len(values) * 0.55
    spacing = plot_width / len(values)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
    ]
    for index, (system_id, value, lower, upper) in enumerate(values):
        center = left + spacing * (index + 0.5)
        y = top + plot_height * (1.0 - value / maximum)
        bar_height = top + plot_height - y
        lower_y = top + plot_height * (1.0 - lower / maximum)
        upper_y = top + plot_height * (1.0 - upper / maximum)
        elements.extend([
            f'<rect x="{center - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#3b82f6"/>',
            f'<line x1="{center:.2f}" y1="{upper_y:.2f}" x2="{center:.2f}" y2="{lower_y:.2f}" stroke="black"/>',
            f'<line x1="{center - 6:.2f}" y1="{upper_y:.2f}" x2="{center + 6:.2f}" y2="{upper_y:.2f}" stroke="black"/>',
            f'<line x1="{center - 6:.2f}" y1="{lower_y:.2f}" x2="{center + 6:.2f}" y2="{lower_y:.2f}" stroke="black"/>',
            f'<text x="{center:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{html.escape(system_id)}</text>',
            f'<text x="{center:.2f}" y="{max(top + 12, y - 6):.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3f}</text>',
        ])
    elements.extend([
        f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(metric.replace("_", " "))}</text>',
        '</svg>\n',
    ])
    return "".join(elements).encode("utf-8")


def _trials_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in records)


def rendered_aggregate_outputs(aggregate: Mapping[str, Any]) -> dict[str, bytes]:
    return {
        **aggregate_tables(aggregate),
        "success-rate.svg": _bar_plot(
            aggregate,
            metric="success_rate",
            title="Held-out level success rate (95% CI)",
        ),
        "shots-to-success.svg": _bar_plot(
            aggregate,
            metric="penalized_shots",
            title="Shots to success (unsuccessful trials = max shots + 1)",
        ),
    }


def final_artifact_documents(
    protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    aggregate = aggregate_trials(protocol, records)
    audit = exposure_audit(protocol, records)
    if not audit["passed"]:
        raise ValueError("gameplay-success exposure audit failed")
    files = {
        "trials.jsonl": _trials_jsonl(records),
        "aggregate.json": (
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "exposure-audit.json": (
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        **rendered_aggregate_outputs(aggregate),
    }
    evidence_payload = {
        "schema": EVIDENCE_SCHEMA,
        "protocol_identity": protocol["protocol_identity"],
        "aggregate_artifact_identity": aggregate["artifact_identity"],
        "trial_evidence_identities": [value["evidence_identity"] for value in records],
        "exposure_audit_passed": True,
        "gameplay_conclusion": aggregate["gameplay_conclusion"],
        "artifact_files": sorted((*files, "evidence.json")),
        "rerun_commands": protocol["rerun_commands"],
    }
    evidence = _with_identity(
        "cohort-v2-gameplay-success-evidence-v1",
        "evidence_identity",
        evidence_payload,
    )
    files["evidence.json"] = (
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return evidence, files


def write_final_artifacts(
    root: Path,
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence, files = final_artifact_documents(protocol, records)
    for name, content in files.items():
        write_immutable_cohort_v2_bytes(content, Path(root) / name)
    return evidence


def load_trial_records(root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for schedule_entry in protocol["trial_schedule"]:
        path = Path(root) / "trial-records" / f"{schedule_entry['trial_id']}.json"
        record = _load_object(path, f"trial {schedule_entry['trial_id']}")
        expected = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        if path.read_bytes() != expected:
            raise ValueError(
                f"gameplay-success trial is not canonical: {schedule_entry['trial_id']}"
            )
        records.append(validate_trial_record(protocol, record, schedule_entry))
    return records


def validate_final_artifacts(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    records = load_trial_records(root, protocol)
    expected_evidence, expected_files = final_artifact_documents(protocol, records)
    for name, expected in expected_files.items():
        path = Path(root) / name
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ValueError(f"gameplay-success output is missing {name}: {error}") from error
        if actual != expected:
            raise ValueError(f"gameplay-success output differs: {name}")
    return expected_evidence


def write_run_manifest(
    root: Path,
    protocol: Mapping[str, Any],
    *,
    implementation_revision: str,
    stack_bindings: Mapping[str, str],
    authorization_identity: str,
) -> dict[str, Any]:
    validate_stack_bindings(protocol, stack_bindings)
    if authorization_identity != protocol["authorization_identity"]:
        raise PermissionError("gameplay-success final authorization identity differs")
    manifest = {
        "schema": RUN_SCHEMA,
        "protocol_identity": protocol["protocol_identity"],
        "implementation_revision": implementation_revision,
        "stack_bindings": dict(stack_bindings),
        "authorization_identity": authorization_identity,
        "scheduled_trial_count": len(protocol["trial_schedule"]),
        "retry_count": 0,
        "outcome_conditioned_changes": False,
    }
    write_immutable_cohort_v2_json(manifest, Path(root) / "run-manifest.json")
    return manifest
