"""Run issue #56 CEM/open-loop or receding-horizon gameplay planning."""

from __future__ import annotations

import argparse
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Final

from PIL import Image

from scripts.manual_agent import (
    connect_with_retry,
    prepare_for_play,
    start_engine,
    stop_started_engine,
)
from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.run_cohort_v2_visual_parser import DEFAULT_ALIGNED, DEFAULT_OUTPUT as VISUAL_OUTPUT
from scripts.run_issue_15_confirmatory_v2 import (
    CAPACITY_COMPACT,
    DEFAULT_INTEGRATED,
    DEFAULT_PROTOCOL_ROOT,
    _load_frozen,
)
from scripts.run_cohort_v2_feature_parser import DEFAULT_RELIABILITY
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE
from scripts.slingshot_readiness import (
    prepare_screen_shot,
    slingshot_observation_from_symbolic_state,
)
from src.webui.bridge import GameState, PlayingMode, ScienceBirdsBridge
from world_model.data import CohortV2AlignedObservationReader
from world_model.model import PredictionPair
from world_model.planning.gameplay import (
    CEMConfig,
    CEMPlanner,
    ControlConfig,
    ControlMode,
    FrozenCohortV2WorldModel,
    GameplayCost,
    GameplayCostConfig,
    GameplayEvidenceBindings,
    HeuristicNoModelPlanner,
    PlanningObservation,
    RandomLegalPlanner,
    SlingshotAction,
    SlingshotActionBounds,
    TerminalStatus,
    VisualPlanningObservationAdapter,
    WorldModelCandidateEvaluator,
    run_gameplay_control,
    validate_gameplay_evidence,
    write_gameplay_evidence,
)
from world_model.training.cohort_v2_controller import CohortV2ControllerFeatureCodec
from world_model.training.cohort_v2_integrated import (
    IntegratedVariant,
    integrated_compute_calibration,
)
from world_model.training.cohort_v2_visual_parser import load_visual_parser_checkpoint
from world_model.training.manifest import git_revision


DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-56-gameplay-planner")
GOAL_COST_VERSION: Final = "cohort-v2-gameplay-cost-v1"
ENVIRONMENT_VERSION: Final = "ScienceBirds-Linux-Unity-2019.4.41f2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--integrated-root", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--reliability-root", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--integrated-compact", type=Path, default=CAPACITY_COMPACT)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--visual-parser-root", type=Path, default=VISUAL_OUTPUT / "parser")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planner", choices=("cem", "random", "heuristic"), default="cem")
    parser.add_argument("--mode", choices=("mpc", "open_loop"), default="mpc")
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=2)
    parser.add_argument("--fixed-steps-per-shot", type=int, default=15)
    parser.add_argument("--max-shots", type=int, default=2)
    parser.add_argument("--max-planner-compute", type=float, default=1e12)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2004)
    parser.add_argument("--agent-id", type=int, default=28888)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--game-dir", type=Path, default=Path("sciencebirdsgames/Linux"))
    parser.add_argument("--game-headless", action="store_true")
    parser.add_argument("--start-engine", action="store_true")
    parser.add_argument("--level", type=int)
    parser.add_argument("--implementation-commit")
    parser.add_argument(
        "--migration-recovery",
        type=Path,
        nargs="?",
        const=DEFAULT_MANIFEST,
        metavar="MANIFEST",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-smoke", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    return {
        "release": (root / args.release_root).resolve(),
        "protocol": (root / args.protocol_root).resolve(),
        "integrated": (root / args.integrated_root).resolve(),
        "reliability": (root / args.reliability_root).resolve(),
        "integrated_compact": (root / args.integrated_compact).resolve(),
        "migration_recovery": (
            None
            if args.migration_recovery is None
            else (root / args.migration_recovery).resolve()
        ),
        "aligned": (root / args.aligned_root).resolve(),
        "visual": (root / args.visual_parser_root).resolve(),
        "output": (root / args.output).resolve(),
    }


def _bounds() -> SlingshotActionBounds:
    return SlingshotActionBounds(
        drag_x=(-160, -40),
        drag_y=(-80, 80),
        tap_time_ms=(0, 1000),
        release_time_ms=600,
    )


def _cost_config() -> GameplayCostConfig:
    return GameplayCostConfig(
        goal_progress_weight=10.0,
        terminal_success_cost=-20.0,
        terminal_failure_cost=20.0,
        illegal_action_cost=50.0,
        physical_penalty_weight=1.0,
        rollout_penalty_weight=0.1,
        compute_weight=1e-8,
        structure_unstable_weight=0.0,
    )


def _load_stack(root: Path, paths: dict[str, Path], device: str):
    print("[load 1/3] frozen issue-15 world model and joint controller", flush=True)
    frozen = _load_frozen(root, paths, device)
    print("[load 2/3] issue-59 public aligned observation roles", flush=True)
    aligned = tuple(
        CohortV2AlignedObservationReader(paths["aligned"], source_reader=reader)
        for reader in frozen["readers"]
    )
    print("[load 3/3] frozen issue-17 visual parser", flush=True)
    parser_model, parser_checkpoint, _manifest = load_visual_parser_checkpoint(
        paths["visual"], readers=aligned, device=device
    )
    observation_adapter = VisualPlanningObservationAdapter(
        parser_model,
        parser_checkpoint_identity=parser_checkpoint.identity,
        temperatures=parser_checkpoint.temperatures,
        thresholds=parser_checkpoint.thresholds,
        object_kind_temperature=parser_checkpoint.object_kind_temperature,
        latent_dim=frozen["config"].latent_dim,
        max_entities=frozen["config"].max_entities,
    )
    return frozen, aligned, parser_checkpoint, observation_adapter


def _planner(
    args: argparse.Namespace,
    frozen,
    observation_adapter,
    *,
    dry_run: bool,
    fixed_pair: PredictionPair | None = None,
):
    bounds = _bounds()
    sequence_length = 2 if dry_run else args.sequence_length
    if args.planner == "random":
        return RandomLegalPlanner(
            bounds, sequence_length=sequence_length, seed=args.seed
        )
    if args.planner == "heuristic":
        return HeuristicNoModelPlanner(bounds, sequence_length=sequence_length)
    compute = integrated_compute_calibration(
        frozen["config"],
        IntegratedVariant.CANDIDATE,
        controller_config=frozen["controller_config"],
    )
    world_model = FrozenCohortV2WorldModel(
        predictor=frozen["predictor"],
        pair_controller=frozen["candidate_model"],
        controller_codec=CohortV2ControllerFeatureCodec(
            frozen["controller_config"]
        ),
        compute=compute,
        fixed_steps_per_shot=1 if dry_run else args.fixed_steps_per_shot,
        release_time_ms=bounds.release_time_ms,
        fixed_pair=fixed_pair,
    )
    evaluator = WorldModelCandidateEvaluator(
        world_model, bounds, GameplayCost(_cost_config())
    )
    config = CEMConfig(
        population_size=4 if dry_run else args.population_size,
        elite_count=2 if dry_run else args.elite_count,
        iterations=1 if dry_run else args.cem_iterations,
        sequence_length=sequence_length,
        seed=args.seed,
    )
    return CEMPlanner(config, bounds, evaluator, progress=lambda value: print(value, flush=True))


class _DryRunEnvironment:
    def __init__(self, observations: tuple[PlanningObservation, ...]) -> None:
        self.observations = observations
        self.index = 0

    def observe(self) -> PlanningObservation:
        return self.observations[0]

    def execute(self, action: SlingshotAction) -> PlanningObservation:
        self.index += 1
        value = self.observations[self.index]
        if self.index == len(self.observations) - 1:
            value = replace(value, terminal_status=TerminalStatus.SUCCESS)
        print(
            f"[dry game] accepted legal action {action}; next={value.identity}",
            flush=True,
        )
        return value


def _dry_run(args, frozen, aligned, observation_adapter) -> int:
    rollout = aligned[0].rollouts[0]
    frames = rollout.frame_records[:3]
    observations = tuple(
        observation_adapter.from_agent_rgb(
            identity=f"dry-run:{frame.identity}",
            png=aligned[0].load_frame_observation(
                rollout, frame, observation_role="agent"
            ),
            slingshot_anchor=(312, 227),
            terminal_status=TerminalStatus.ONGOING,
        )
        for frame in frames
    )
    planner = _planner(args, frozen, observation_adapter, dry_run=True)
    mode = ControlMode(args.mode)
    result = run_gameplay_control(
        planner,
        _DryRunEnvironment(observations),
        ControlConfig(mode, max_shots=2, max_planner_compute=1e12),
        progress=lambda value: print(value, flush=True),
    )
    expected_replans = 2 if mode is ControlMode.MPC else 1
    if (
        result.replan_count != expected_replans
        or len(result.steps) != 2
        or not result.success
    ):
        raise RuntimeError(
            f"dry-run did not complete observe-plan-act-observe-replan: {result}"
        )
    print(
        f"[dry-run complete] replans={result.replan_count} shots={len(result.steps)} "
        f"model_rollouts={result.model_rollout_count}; no files written",
        flush=True,
    )
    return 0


class _LiveScienceBirdsEnvironment:
    def __init__(
        self,
        bridge: ScienceBirdsBridge,
        observation_adapter: VisualPlanningObservationAdapter,
        bounds: SlingshotActionBounds,
        *,
        speed: int,
        level_label: str | None = None,
    ) -> None:
        self.bridge = bridge
        self.observation_adapter = observation_adapter
        self.bounds = bounds
        self.speed = speed
        self.level_label = level_label
        self.index = 0
        self.current: PlanningObservation | None = None
        self.last_anchor: tuple[int, int] | None = None

    @staticmethod
    def _terminal(state: GameState) -> TerminalStatus:
        if state is GameState.WON:
            return TerminalStatus.SUCCESS
        if state in (GameState.LOST, GameState.EVALUATION_TERMINATED):
            return TerminalStatus.FAILURE
        return TerminalStatus.ONGOING

    def observe(self) -> PlanningObservation:
        state = self.bridge.get_game_state()
        screenshot = self.bridge.screenshot()
        image = Image.frombytes(
            "RGB", (screenshot.width, screenshot.height), screenshot.rgb
        )
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        if state is GameState.PLAYING:
            reference = slingshot_observation_from_symbolic_state(
                self.bridge.get_symbolic_state_without_screenshot(), screenshot.height
            )
            if reference is None:
                raise RuntimeError("live observation has no slingshot anchor")
            self.last_anchor = (int(reference["gameX"]), int(reference["gameY"]))
        if self.last_anchor is None:
            raise RuntimeError("live observation has no retained slingshot anchor")
        self.index += 1
        level_label = self.level_label or f"level-{self.bridge.get_current_level()}"
        self.current = self.observation_adapter.from_agent_rgb(
            identity=(
                f"live:{level_label}:observation-{self.index}"
            ),
            png=encoded.getvalue(),
            slingshot_anchor=self.last_anchor,
            terminal_status=self._terminal(state),
        )
        diagnostics = self.current.parser_diagnostics or {}
        print(
            f"[observe {self.index}] state={state.name} "
            f"structure_unstable_probability="
            f"{diagnostics.get('structure_unstable_probability')} "
            f"thresholded={diagnostics.get('structure_unstable_thresholded')} "
            "cost_weight=0",
            flush=True,
        )
        return self.current

    def execute(self, action: SlingshotAction) -> PlanningObservation:
        if self.current is None:
            self.observe()
        assert self.current is not None
        interface_action = action.to_interface_action(
            self.current.slingshot_anchor, self.bounds
        )
        print(f"[game execute] {interface_action}", flush=True)
        prepare_screen_shot(
            self.bridge,
            interface_action,
            frame_height=self.current.frame_height,
            execution_speed=self.speed,
            fast=False,
        ).execute()
        return self.observe()


def _live(args, root, paths, frozen, parser_checkpoint, observation_adapter, implementation) -> int:
    engine = None
    bridge = None
    if args.start_engine:
        engine = start_engine((root / args.game_dir).resolve(), args.game_headless)
        log_name = getattr(getattr(engine, "novphy_log_file", None), "name", "unknown")
        print(f"[game start] pid={engine.pid} engine_log={log_name}", flush=True)
    try:
        bridge = connect_with_retry(args.host, args.port, timeout=300, deadline_seconds=60)
        print(f"[game connect] {args.host}:{args.port}", flush=True)
        bridge.configure(args.agent_id, PlayingMode.TRAINING)
        bridge.set_speed(args.speed)
        if args.level is not None:
            bridge.load_level(args.level)
        prepare_for_play(bridge, timeout=60, poll_delay=0.5)
        level = bridge.get_current_level()
        planner = _planner(args, frozen, observation_adapter, dry_run=False)
        mode = ControlMode(args.mode)
        control = ControlConfig(mode, args.max_shots, args.max_planner_compute)
        environment = _LiveScienceBirdsEnvironment(
            bridge, observation_adapter, _bounds(), speed=args.speed
        )
        result = run_gameplay_control(
            planner,
            environment,
            control,
            progress=lambda value: print(value, flush=True),
        )
        cem_config = CEMConfig(
            args.population_size,
            args.elite_count,
            args.cem_iterations,
            args.sequence_length,
            args.seed,
        )
        command = (
            "python -u -m scripts.run_issue_56_gameplay_planner --live-smoke "
            f"--start-engine --implementation-commit {implementation}"
        )
        bindings = GameplayEvidenceBindings(
            implementation_revision=implementation,
            world_model_checkpoint_identity=frozen["checkpoint"].identity,
            controller_checkpoint_identity=frozen["aggregation_manifest"][
                "aggregation_artifact_identity"
            ],
            visual_parser_checkpoint_identity=parser_checkpoint.identity,
            observation_adapter_identity=observation_adapter.carrier_adapter_identity,
            goal_cost_version=GOAL_COST_VERSION,
            goal_cost_config=_cost_config(),
            action_bounds=_bounds(),
            cem_config=cem_config,
            control_config=control,
            seed=args.seed,
            level_identity=f"science-birds-level-{level}",
            environment_version=ENVIRONMENT_VERSION,
            rerun_commands=(
                "python -u -m scripts.run_issue_56_gameplay_planner --dry-run",
                command,
                "python -u -m scripts.run_issue_56_gameplay_planner --validate",
            ),
        )
        document = write_gameplay_evidence(paths["output"], result, bindings)
        print(
            f"[complete] termination={result.termination_reason} "
            f"artifact={document['artifact_identity']}",
            flush=True,
        )
        print(f"[report] {paths['output'] / 'evidence.json'}", flush=True)
        return 0
    finally:
        if bridge is not None:
            bridge.disconnect()
        stop_started_engine(engine)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    selected = sum((args.dry_run, args.live_smoke, args.validate))
    if selected != 1:
        parser.error("select exactly one of --dry-run, --live-smoke, or --validate")
    root = args.repository_root.resolve()
    paths = _paths(args, root)
    if args.validate:
        document = validate_gameplay_evidence(paths["output"])
        print(
            f"[validate] exact issue-56 validation passed "
            f"artifact={document['artifact_identity']}",
            flush=True,
        )
        return 0
    implementation = args.implementation_commit
    if implementation is None:
        implementation, dirty = git_revision(str(root))
        if dirty and args.live_smoke:
            parser.error("a dirty worktree requires --implementation-commit")
    device = "cpu" if args.dry_run else args.device
    print(
        "[design] issue=56 planner distinct from prediction-pair controller; "
        f"planner={args.planner} mode={args.mode} seed={args.seed}",
        flush=True,
    )
    print(
        "[observation] deployment RGB visual carrier; structure-unstable logged "
        "but zero-weighted; oracle carrier disabled",
        flush=True,
    )
    print(
        "[rollout] candidate sequences reuse the current real observation for the "
        "frozen pair controller/symbol input; MPC refreshes it after each real shot",
        flush=True,
    )
    frozen, aligned, parser_checkpoint, observation_adapter = _load_stack(
        root, paths, device
    )
    if args.dry_run:
        return _dry_run(args, frozen, aligned, observation_adapter)
    return _live(
        args,
        root,
        paths,
        frozen,
        parser_checkpoint,
        observation_adapter,
        str(implementation),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
