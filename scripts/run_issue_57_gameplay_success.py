"""Freeze, run, aggregate, and validate issue #57 gameplay-success evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any, Final

from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from scripts.manual_agent import (
    connect_with_retry,
    prepare_for_play,
    start_engine,
    stop_started_engine,
)
from scripts.run_issue_56_gameplay_planner import (
    _DryRunEnvironment,
    _LiveScienceBirdsEnvironment,
    _bounds,
    _load_stack,
    _planner,
)
from scripts.smoke_physics_capture import start_display, terminate
from src.webui.bridge import PlayingMode
from world_model.data.deployment_temporal import (
    AgentObservation,
    TemporalObservationContext,
)
from world_model.model import Abstraction, PredictionPair
from world_model.planning.gameplay import (
    ControlConfig,
    ControlMode,
    PlanningObservation,
    TerminalStatus,
    run_gameplay_control,
)
from world_model.planning.gameplay_success import (
    AUTHORIZATION_IDENTITY,
    PROTOCOL_FILENAME,
    RUN_SCHEMA,
    aggregate_trials,
    build_protocol,
    build_trial_record,
    build_trial_schedule,
    load_aborted_v1_run,
    load_protocol,
    load_trial_records,
    materialize_protocol_runtimes,
    rendered_aggregate_outputs,
    stack_identity_bindings,
    validate_final_artifacts,
    validate_trial_record,
    write_final_artifacts,
    write_protocol,
    write_run_manifest,
)
from world_model.training.manifest import git_revision


RECOVERY_ROOT: Final = Path(".local-artifacts/migration-recovery-v1")
DEFAULT_PROTOCOL: Final = Path("data/runtime_evidence/issue-57") / PROTOCOL_FILENAME
DEFAULT_OUTPUT: Final = RECOVERY_ROOT / "issue-57-gameplay-success-v2"
DEFAULT_GAME_DIR: Final = RECOVERY_ROOT / "game-engine-runtime"
DEFAULT_GAME_RUNTIME_ROOT: Final = RECOVERY_ROOT / "issue-57-game-runtimes-v2"
DEFAULT_SUPERSEDED_RUN: Final = RECOVERY_ROOT / "issue-57-gameplay-success"
DEFAULT_PREFLIGHT_OUTPUT: Final = RECOVERY_ROOT / "issue-57-live-level-preflight-v2"
DEFAULT_PLANNING_EVIDENCE: Final = (
    RECOVERY_ROOT / "issue-56-gameplay-planner/evidence.json"
)
DEFAULT_MIGRATION_MANIFEST: Final = (
    RECOVERY_ROOT
    / "issue-53-authority/cohort-v2-migration-recovery-manifest-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR)
    parser.add_argument(
        "--game-runtime-root", type=Path, default=DEFAULT_GAME_RUNTIME_ROOT
    )
    parser.add_argument(
        "--superseded-run-root", type=Path, default=DEFAULT_SUPERSEDED_RUN
    )
    parser.add_argument(
        "--preflight-output", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT
    )
    parser.add_argument(
        "--planning-evidence", type=Path, default=DEFAULT_PLANNING_EVIDENCE
    )
    parser.add_argument(
        "--migration-recovery", type=Path, default=DEFAULT_MIGRATION_MANIFEST
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/runtime_evidence/issue-53-mixed-termination-v5"),
    )
    parser.add_argument(
        "--protocol-root", type=Path, default=RECOVERY_ROOT / "issue-15-amendment-v2"
    )
    parser.add_argument(
        "--integrated-root",
        type=Path,
        default=RECOVERY_ROOT / "issue-15-capacity-integrated",
    )
    parser.add_argument(
        "--reliability-root",
        type=Path,
        default=RECOVERY_ROOT / "issue-12-reliability",
    )
    parser.add_argument(
        "--integrated-compact",
        type=Path,
        default=RECOVERY_ROOT / "summaries/issue-15-capacity-integrated-summary.json",
    )
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=RECOVERY_ROOT / "issue-59-aligned-observation-release",
    )
    parser.add_argument(
        "--visual-parser-root",
        type=Path,
        default=RECOVERY_ROOT / "issue-17-visual-parser/parser",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2004)
    parser.add_argument("--agent-id", type=int, default=28889)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--start-display", action="store_true")
    parser.add_argument("--start-engine", action="store_true")
    parser.add_argument("--game-headless", action="store_true")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--authorization-identity")
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-level-preflight", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    return {
        "release": _resolve(root, args.release_root),
        "protocol": _resolve(root, args.protocol_root),
        "integrated": _resolve(root, args.integrated_root),
        "reliability": _resolve(root, args.reliability_root),
        "integrated_compact": _resolve(root, args.integrated_compact),
        "migration_recovery": _resolve(root, args.migration_recovery),
        "aligned": _resolve(root, args.aligned_root),
        "visual": _resolve(root, args.visual_parser_root),
        "output": _resolve(root, args.output),
        "game": _resolve(root, args.game_dir),
        "game_runtimes": _resolve(root, args.game_runtime_root),
        "superseded_run": _resolve(root, args.superseded_run_root),
        "preflight_output": _resolve(root, args.preflight_output),
        "gameplay_protocol": _resolve(root, args.protocol),
        "planning_evidence": _resolve(root, args.planning_evidence),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _stack_bindings(frozen, parser_checkpoint, observation_adapter) -> dict[str, str]:
    return stack_identity_bindings(
        frozen["checkpoint"].identity,
        frozen["aggregation_manifest"]["aggregation_artifact_identity"],
        parser_checkpoint.identity,
        observation_adapter.carrier_adapter_identity,
    )


def _system_planner(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    system: dict[str, Any],
    seed: int,
    frozen,
    observation_adapter,
    *,
    dry_run: bool,
):
    local = copy.copy(args)
    local.planner = system["planner"]
    local.mode = system["control_mode"]
    local.sequence_length = system["sequence_length"]
    local.population_size = protocol["cem"]["population_size"]
    local.elite_count = protocol["cem"]["elite_count"]
    local.cem_iterations = protocol["cem"]["iterations"]
    local.fixed_steps_per_shot = protocol["execution_limits"][
        "fixed_steps_per_shot"
    ]
    local.seed = seed
    fixed = system["fixed_prediction_pair"]
    fixed_pair = None if fixed is None else PredictionPair(
        int(fixed["horizon"]), Abstraction(str(fixed["abstraction"]))
    )
    return _planner(
        local,
        frozen,
        observation_adapter,
        dry_run=dry_run,
        fixed_pair=fixed_pair,
    )


def _control_config(protocol: dict[str, Any], system: dict[str, Any], *, dry_run: bool):
    limits = protocol["execution_limits"]
    return ControlConfig(
        ControlMode(system["control_mode"]),
        2 if dry_run else int(limits["max_shots"]),
        float(limits["max_planner_compute"]),
        None if dry_run else float(limits["max_trial_wall_clock_seconds"]),
    )


class _TimedObservationAdapter:
    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.call_count = 0
        self.wall_clock_seconds = 0.0

    def from_agent_rgb(self, **kwargs):
        started = time.monotonic()
        try:
            return self.adapter.from_agent_rgb(**kwargs)
        finally:
            self.call_count += 1
            self.wall_clock_seconds += time.monotonic() - started

    def from_temporal_context(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return self.adapter.from_temporal_context(*args, **kwargs)
        finally:
            self.call_count += 1
            self.wall_clock_seconds += time.monotonic() - started


def _dry_observations(aligned, observation_adapter) -> tuple[PlanningObservation, ...]:
    rollout = aligned[0].rollouts[0]
    observations = []
    prior = None
    for frame in rollout.frame_records[:3]:
        metadata = aligned[0].frame_observation_metadata(rollout, frame)
        current = AgentObservation(
            identity=aligned[0].frame_agent_observation_identity(rollout, frame),
            fixed_step=frame.fixed_step,
            fixed_time_seconds=float(metadata["fixed_time_seconds"]),
            png=aligned[0].load_frame_observation(
                rollout, frame, observation_role="agent"
            ),
        )
        observations.append(observation_adapter.from_temporal_context(
            TemporalObservationContext(prior, current),
            slingshot_anchor=(312, 227),
            terminal_status=TerminalStatus.ONGOING,
        ))
        prior = current
    return tuple(observations)


def _dry_run(args, protocol, frozen, aligned, parser_checkpoint, adapter) -> int:
    observations = _dry_observations(aligned, adapter)
    stack = _stack_bindings(frozen, parser_checkpoint, adapter)
    smoke = copy.deepcopy(protocol)
    smoke["protocol_identity"] = f"dry-run:{protocol['protocol_identity']}"
    smoke["trial_seeds"] = protocol["trial_seeds"][:2]
    smoke_level = protocol["level_inventory"]["roles"]["smoke"][0]
    smoke["trial_schedule"] = build_trial_schedule(
        smoke["systems"], [smoke_level], smoke["trial_seeds"], exposure_role="smoke"
    )
    records = []
    systems = {value["system_id"]: value for value in smoke["systems"]}
    total = len(smoke["trial_schedule"])
    for entry in smoke["trial_schedule"]:
        print(
            f"[dry trial {entry['trial_index']}/{total}] "
            f"system={entry['system_id']} seed={entry['seed']}",
            flush=True,
        )
        system = systems[entry["system_id"]]
        planner = _system_planner(
            args, smoke, system, entry["seed"], frozen, adapter, dry_run=True
        )
        environment = _DryRunEnvironment(observations)
        result = run_gameplay_control(
            planner,
            environment,
            _control_config(smoke, system, dry_run=True),
            progress=lambda message: print(message, flush=True),
        )
        record = build_trial_record(
            smoke,
            entry,
            implementation_revision="dry-run-no-final-access",
            stack_bindings=stack,
            result=result,
        )
        validate_trial_record(smoke, record, entry)
        records.append(record)
        print(
            f"[dry trial complete] success={result.success} "
            f"shots={len(result.steps)} replans={result.replan_count}",
            flush=True,
        )
    aggregate = aggregate_trials(smoke, records)
    rendered = rendered_aggregate_outputs(aggregate)
    if set(rendered) != {
        "success_rates.csv",
        "shots_to_success.csv",
        "compute_failures.csv",
        "success-rate.svg",
        "shots-to-success.svg",
    }:
        raise RuntimeError("dry-run aggregate table or plot generation differs")
    print(
        f"[dry-run complete] trials={len(records)} systems={len(smoke['systems'])} "
        f"aggregate={aggregate['artifact_identity']} tables_plots={len(rendered)} "
        "no final levels opened; "
        "no files written",
        flush=True,
    )
    return 0


def _record_path(output: Path, entry: dict[str, Any]) -> Path:
    return output / "trial-records" / f"{entry['trial_id']}.json"


def _open_single_level_runtime(
    args,
    runtime: Path,
    *,
    level_label: str,
):
    engine = None
    bridge = None
    try:
        engine = start_engine(
            runtime, args.game_headless, agent_port=args.port
        )
        log_name = getattr(
            getattr(engine, "novphy_log_file", None), "name", "unknown"
        )
        print(
            f"[game start] level={level_label} pid={engine.pid} "
            f"engine_log={log_name}",
            flush=True,
        )
        bridge = connect_with_retry(
            args.host, args.port, timeout=300, deadline_seconds=60
        )
        bridge.configure(args.agent_id, PlayingMode.TRAINING)
        bridge.set_speed(args.speed)
        prepare_for_play(bridge, timeout=60, poll_delay=0.5)
        runtime_index = bridge.get_current_level()
        if runtime_index != 1:
            raise RuntimeError(
                "single-level runtime did not establish runtime level index 1"
            )
        print(
            f"[level ready] level={level_label} runtime_index={runtime_index}",
            flush=True,
        )
        return bridge, engine
    except Exception:
        if bridge is not None:
            bridge.disconnect()
        stop_started_engine(engine)
        raise


def _close_single_level_runtime(bridge, engine) -> None:
    if bridge is not None:
        bridge.disconnect()
    stop_started_engine(engine)


def _preflight_single_level_runtimes(
    args,
    protocol: dict[str, Any],
    runtimes: dict[str, Path],
) -> list[dict[str, Any]]:
    roles = protocol["level_inventory"]["roles"]
    runtime_configs = {
        value["level_identity"]: value
        for value in protocol["execution_runtime"]["configs"]
    }
    records = []
    for role in protocol["execution_runtime"]["pre_final_live_preflight_roles"]:
        level = roles[role][0]
        identity = str(level["level_identity"])
        print(
            f"[preflight] role={role} level={level['level_number']} "
            f"identity={identity}",
            flush=True,
        )
        bridge, engine = _open_single_level_runtime(
            args,
            runtimes[identity],
            level_label=f"preflight-{level['level_number']}",
        )
        _close_single_level_runtime(bridge, engine)
        records.append({
            "exposure_role": role,
            "level_number": level["level_number"],
            "level_identity": identity,
            "runtime_config_sha256": runtime_configs[identity]["config_sha256"],
            "runtime_level_index": 1,
            "state": "PLAYING",
        })
    print(
        "[preflight complete] exact smoke and training/tuning one-level runtimes "
        "reached PLAYING; final levels opened=0",
        flush=True,
    )
    return records


def _run_final(
    args,
    protocol,
    paths,
    frozen,
    parser_checkpoint,
    adapter,
    implementation: str,
) -> int:
    stack = _stack_bindings(frozen, parser_checkpoint, adapter)
    runtimes = materialize_protocol_runtimes(
        paths["game"], paths["game_runtimes"], protocol
    )
    systems = {value["system_id"]: value for value in protocol["systems"]}
    schedule = protocol["trial_schedule"]
    display_process = None
    previous_display = os.environ.get("DISPLAY")
    try:
        paths["output"].mkdir(parents=True, exist_ok=True)
        if args.start_display:
            display, display_process = start_display(paths["output"] / "display.log")
            os.environ["DISPLAY"] = display
            print(f"[display start] DISPLAY={display}", flush=True)
        _preflight_single_level_runtimes(args, protocol, runtimes)
        write_run_manifest(
            paths["output"],
            protocol,
            implementation_revision=implementation,
            stack_bindings=stack,
            authorization_identity=str(args.authorization_identity),
        )
        for entry in schedule:
            path = _record_path(paths["output"], entry)
            if path.exists():
                record = _load_json(path, f"trial {entry['trial_id']}")
                validate_trial_record(protocol, record, entry)
                print(
                    f"[resume {entry['trial_index']}/{len(schedule)}] "
                    f"validated existing {entry['trial_id']}",
                    flush=True,
                )
                continue
            system = systems[entry["system_id"]]
            print(
                f"[trial {entry['trial_index']}/{len(schedule)} start] "
                f"system={entry['system_id']} level={entry['level_number']} "
                f"condition={entry['condition_identity']} seed={entry['seed']}",
                flush=True,
            )
            timed_adapter = _TimedObservationAdapter(adapter)
            result = None
            failure = None
            failure_class = "infrastructure_failure"
            identity = str(entry["level_identity"])
            bridge, engine = _open_single_level_runtime(
                args,
                runtimes[identity],
                level_label=f"final-{entry['level_number']}",
            )
            try:
                planner = _system_planner(
                    args,
                    protocol,
                    system,
                    entry["seed"],
                    frozen,
                    timed_adapter,
                    dry_run=False,
                )
                environment = _LiveScienceBirdsEnvironment(
                    bridge,
                    timed_adapter,
                    _bounds(),
                    speed=args.speed,
                    level_label=f"science-birds-level-{entry['level_number']}",
                )
                result = run_gameplay_control(
                    planner,
                    environment,
                    _control_config(protocol, system, dry_run=False),
                    progress=lambda message: print(message, flush=True),
                )
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                if isinstance(error, TimeoutError):
                    failure_class = "timeout"
                print(f"[trial failure] {failure}", flush=True)
            finally:
                _close_single_level_runtime(bridge, engine)
            record = build_trial_record(
                protocol,
                entry,
                implementation_revision=implementation,
                stack_bindings=stack,
                result=result,
                observation_parser_call_count=timed_adapter.call_count,
                observation_parser_wall_clock_seconds=(
                    timed_adapter.wall_clock_seconds
                ),
                infrastructure_failure=failure,
                pre_control_failure_class=failure_class,
            )
            write_immutable_cohort_v2_json(record, path)
            print(
                f"[trial {entry['trial_index']}/{len(schedule)} complete] "
                f"termination={record['outcome']['termination_reason']} "
                f"success={record['outcome']['success']} "
                f"shots={record['outcome']['executed_shot_count']} "
                f"wall={record['compute']['end_to_end_wall_clock_seconds']:.3f}s "
                f"evidence={record['evidence_identity']}",
                flush=True,
            )
        records = load_trial_records(paths["output"], protocol)
        print("[aggregate] regenerating exact tables and plots", flush=True)
        evidence = write_final_artifacts(paths["output"], protocol, records)
        print(
            f"[complete] conclusion={evidence['gameplay_conclusion']} "
            f"evidence={evidence['evidence_identity']}",
            flush=True,
        )
        return 0
    finally:
        if display_process is not None:
            print(f"[display stop] {terminate(display_process)}", flush=True)
        if previous_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = previous_display


def _run_live_level_preflight(args, protocol, paths) -> int:
    runtimes = materialize_protocol_runtimes(
        paths["game"], paths["game_runtimes"], protocol
    )
    display_process = None
    previous_display = os.environ.get("DISPLAY")
    try:
        paths["preflight_output"].mkdir(parents=True, exist_ok=True)
        if args.start_display:
            display, display_process = start_display(
                paths["preflight_output"] / "display.log"
            )
            os.environ["DISPLAY"] = display
            print(f"[display start] DISPLAY={display}", flush=True)
        records = _preflight_single_level_runtimes(args, protocol, runtimes)
        write_immutable_cohort_v2_json({
            "schema": "cohort_v2_gameplay_success_live_level_preflight_v2",
            "protocol_identity": protocol["protocol_identity"],
            "records": records,
            "intended_final_level_access_count": 0,
            "passed": True,
        }, paths["preflight_output"] / "preflight.json")
        print(
            f"[preflight report] {paths['preflight_output'] / 'preflight.json'}",
            flush=True,
        )
        return 0
    finally:
        if display_process is not None:
            print(f"[display stop] {terminate(display_process)}", flush=True)
        if previous_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = previous_display


def _validate_run_manifest(
    output: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    manifest = _load_json(output / "run-manifest.json", "gameplay-success run manifest")
    if (
        manifest.get("schema") != RUN_SCHEMA
        or manifest.get("protocol_identity") != protocol["protocol_identity"]
        or manifest.get("authorization_identity") != protocol["authorization_identity"]
        or manifest.get("scheduled_trial_count") != len(protocol["trial_schedule"])
        or manifest.get("retry_count") != 0
        or manifest.get("outcome_conditioned_changes") is not False
    ):
        raise ValueError("gameplay-success run manifest differs")
    return manifest


def _validate_records_against_run(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    if any(
        record["implementation_revision"] != manifest["implementation_revision"]
        or record["stack_bindings"] != manifest["stack_bindings"]
        for record in records
    ):
        raise ValueError("gameplay-success trial provenance differs from run manifest")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    selected = sum((
        args.freeze_protocol,
        args.dry_run,
        args.live_level_preflight,
        args.run_final,
        args.aggregate,
        args.validate,
    ))
    if selected != 1:
        parser.error(
            "select exactly one of --freeze-protocol, --dry-run, "
            "--live-level-preflight, --run-final, --aggregate, or --validate"
        )
    root = args.repository_root.resolve()
    paths = _paths(args, root)

    if args.freeze_protocol:
        planning = _load_json(paths["planning_evidence"], "issue-56 planning evidence")
        migration = _load_json(
            paths["migration_recovery"], "migration-recovery manifest"
        )
        superseded = load_aborted_v1_run(paths["superseded_run"])
        protocol = build_protocol(
            paths["game"], planning, migration, superseded
        )
        write_protocol(protocol, paths["gameplay_protocol"])
        print(
            f"[freeze] protocol={protocol['protocol_identity']} "
            f"final_levels={len(protocol['level_inventory']['roles']['final_evaluation'])} "
            f"trials={len(protocol['trial_schedule'])} final_outcomes_opened=0",
            flush=True,
        )
        return 0

    protocol = load_protocol(
        paths["gameplay_protocol"],
        game_dir=(
            paths["game"]
            if args.dry_run or args.live_level_preflight or args.run_final
            else None
        ),
    )
    if args.live_level_preflight:
        if not args.start_engine:
            parser.error("--live-level-preflight requires --start-engine")
        return _run_live_level_preflight(args, protocol, paths)
    if args.aggregate:
        manifest = _validate_run_manifest(paths["output"], protocol)
        records = load_trial_records(paths["output"], protocol)
        _validate_records_against_run(manifest, records)
        evidence = write_final_artifacts(paths["output"], protocol, records)
        print(
            f"[aggregate] exact tables and plots passed "
            f"evidence={evidence['evidence_identity']}",
            flush=True,
        )
        return 0
    if args.validate:
        manifest = _validate_run_manifest(paths["output"], protocol)
        records = load_trial_records(paths["output"], protocol)
        _validate_records_against_run(manifest, records)
        evidence = validate_final_artifacts(paths["output"], protocol)
        print(
            f"[validate] exact issue-57 validation passed "
            f"conclusion={evidence['gameplay_conclusion']} "
            f"evidence={evidence['evidence_identity']}",
            flush=True,
        )
        return 0

    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and args.run_final:
            parser.error(
                "a dirty worktree requires --implementation-commit for final gameplay"
            )
    if args.run_final and args.authorization_identity != AUTHORIZATION_IDENTITY:
        parser.error(
            f"--run-final requires --authorization-identity {AUTHORIZATION_IDENTITY}"
        )
    if args.run_final and not args.start_engine:
        parser.error("--run-final requires --start-engine for per-trial runtimes")
    print(
        f"[protocol] identity={protocol['protocol_identity']} "
        f"systems={len(protocol['systems'])} trials={len(protocol['trial_schedule'])}",
        flush=True,
    )
    print(
        "[boundary] final gameplay outcomes cannot change levels, seeds, systems, "
        "limits, metrics, checkpoints, parser, goal/cost, or planner",
        flush=True,
    )
    print(
        "[parser] structure-unstable remains training-inadequate; raw/thresholded "
        "values and macro-ranking pathways are logged; direct cost weight=0",
        flush=True,
    )
    device = "cpu" if args.dry_run else args.device
    frozen, aligned, parser_checkpoint, adapter = _load_stack(root, paths, device)
    if args.dry_run:
        return _dry_run(
            args, protocol, frozen, aligned, parser_checkpoint, adapter
        )
    return _run_final(
        args,
        protocol,
        paths,
        frozen,
        parser_checkpoint,
        adapter,
        str(implementation),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PermissionError, RuntimeError, ValueError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
