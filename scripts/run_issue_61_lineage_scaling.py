"""Train and validate issue 61's lineage-scaled continuous predictor cells."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

import torch

from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE, _readers
from world_model.data import CohortV2AlignedObservationReader
from world_model.data.deployment_temporal import (
    TemporalObservationContext,
    TrajectoryLineageBinding,
    TrajectoryLineageManifest,
)
from world_model.model import DualOutputPredictor, PredictorConfig
from world_model.planning import (
    GameplayCostConfig,
    PlanningObservation,
    SlingshotActionBounds,
    TerminalStatus,
)
from world_model.planning.gameplay import VisualPlanningObservationAdapter
from world_model.training.cohort_v2_micro import CohortV2StateCodec
from world_model.training.cohort_v2_visual_parser import load_visual_parser_checkpoint
from world_model.training.lineage_scaling import (
    CarrierLineage,
    CarrierKind,
    ContinuousTransitionExample,
    FrozenLineageScale,
    GameplayCheckpointBindings,
    LineageScalingError,
    MatchedGameplayProtocol,
    TrainingCell,
    evaluate_action_ranking,
    evaluate_continuous_prediction,
    build_matched_gameplay_planners,
    load_action_ranking_bundle,
    load_carrier_lineage_bundle,
    load_lineage_scaling_protocol,
    load_lineage_scaled_checkpoint,
    save_lineage_scaled_checkpoint,
    matched_gameplay_systems,
    train_continuous_predictor,
    validate_carrier_alignment,
    validate_lineage_scaled_checkpoint_matrix,
    validate_matched_carrier_lineages,
)


DEFAULT_MIGRATION_ROOT = Path(".local-artifacts/migration-recovery-v1")
DEFAULT_ALIGNED = DEFAULT_MIGRATION_ROOT / "issue-59-aligned-observation-release"
DEFAULT_VISUAL_PARSER = DEFAULT_MIGRATION_ROOT / "issue-17-visual-parser/parser"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--rank", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--deployment-bundle", type=Path)
    parser.add_argument("--scale")
    parser.add_argument("--carrier", choices=tuple(item.value for item in CarrierKind))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluation-bundle", type=Path)
    parser.add_argument("--ranking-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cell-checkpoint",
        action="append",
        default=[],
        metavar="SCALE,CARRIER,SEED,PATH",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--visual-parser-root", type=Path, default=DEFAULT_VISUAL_PARSER)
    parser.add_argument(
        "--migration-recovery",
        type=Path,
        nargs="?",
        const=DEFAULT_MANIFEST,
        default=DEFAULT_MANIFEST,
        metavar="MANIFEST",
    )
    return parser


def _required(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    fields = (
        "protocol",
        "source_bundle",
        "deployment_bundle",
        "scale",
        "carrier",
        "seed",
        "checkpoint",
    )
    missing = tuple(name for name in fields if getattr(args, name) is None)
    if missing:
        parser.error(
            "--train requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def _train(args: argparse.Namespace) -> int:
    print("[load 1/3] frozen lineage-scaling protocol", flush=True)
    protocol = load_lineage_scaling_protocol(args.protocol)
    print("[load 2/3] source-derived carrier lineage bundle", flush=True)
    source = load_carrier_lineage_bundle(args.source_bundle)
    print("[load 3/3] deployment carrier lineage bundle", flush=True)
    deployment = load_carrier_lineage_bundle(args.deployment_bundle)
    alignment = validate_matched_carrier_lineages(protocol, source, deployment)
    print(
        f"[validate] matched carriers lineages={alignment['lineage_count']} "
        f"transitions={alignment['transition_count']}",
        flush=True,
    )
    cell = TrainingCell(args.scale, CarrierKind(args.carrier), args.seed)
    scale = protocol.scale(cell.scale_name)
    bundle = source if cell.carrier is CarrierKind.SOURCE else deployment
    selected = tuple(
        item
        for lineage_identity in scale.lineage_identities
        for item in bundle
        if item.scenario_lineage_identity == lineage_identity
    )
    model, report = train_continuous_predictor(
        protocol,
        cell,
        selected,
        device=args.device,
        progress=lambda value: print(value, flush=True),
    )
    save_lineage_scaled_checkpoint(args.checkpoint, model, report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "scale": cell.scale_name,
                "carrier": cell.carrier.value,
                "seed": cell.seed,
                "lineage_count": report.lineage_count,
                "transition_count": report.transition_count,
                "optimizer_examples": report.optimizer_examples,
                "optimizer_steps": report.optimizer_steps,
                "epochs": report.epochs,
                "final_loss": report.final_loss,
                "wall_seconds": report.wall_seconds,
                "parameter_count": report.parameter_count,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _score(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    fields = (
        "protocol",
        "scale",
        "carrier",
        "seed",
        "checkpoint",
        "evaluation_bundle",
        "output",
    )
    missing = tuple(name for name in fields if getattr(args, name) is None)
    if missing:
        parser.error(
            "--score requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if args.output.exists():
        raise LineageScalingError(f"score output already exists: {args.output}")
    print("[load 1/3] frozen lineage-scaling protocol", flush=True)
    protocol = load_lineage_scaling_protocol(args.protocol)
    cell = TrainingCell(args.scale, CarrierKind(args.carrier), args.seed)
    print("[load 2/3] exact cell checkpoint", flush=True)
    model, checkpoint = load_lineage_scaled_checkpoint(
        args.checkpoint,
        protocol,
        expected_cell=cell,
        device=args.device,
    )
    print("[load 3/3] isolated evaluation carrier lineages", flush=True)
    lineages = load_carrier_lineage_bundle(args.evaluation_bundle)
    training_membership = set(protocol.training_scales[-1].lineage_identities)
    if any(
        item.exposure_role not in ("calibration", "model_selection")
        or item.scenario_lineage_identity in training_membership
        or item.source_release_identity
        != protocol.training_scales[-1].source_release_identity
        or item.carrier is not cell.carrier
        or item.carrier_identity != protocol.carrier_identity(cell.carrier)
        for item in lineages
    ):
        raise LineageScalingError(
            "score inputs contain role leakage, carrier mismatch, or another release"
        )
    evaluation = evaluate_continuous_prediction(
        model,
        lineages,
        horizons=(1, 15),
        progress=lambda value: print(value, flush=True),
    )
    payload = {
        "schema": "lineage_scaled_prediction_evaluation_v1",
        "checkpoint_identity": checkpoint.identity,
        "protocol_identity": protocol.identity,
        "cell_identity": cell.identity,
        "evaluation_role": lineages[0].exposure_role,
        "lineage_count": len(lineages),
        "local_mse": evaluation.local_mse,
        "recursive": [
            {
                "horizon": item.horizon,
                "mean_mse": item.mean_mse,
                "error_auc": item.error_auc,
                "evaluated_transitions": item.evaluated_transitions,
            }
            for item in evaluation.recursive
        ],
        "nonfinite_failures": evaluation.nonfinite_failures,
        "execution_failures": list(evaluation.execution_failures),
        "physical_diagnostics": dict(evaluation.physical_diagnostics),
        "compute": {
            "model_evaluations": evaluation.model_evaluations,
            "wall_seconds": evaluation.wall_seconds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[score] local={evaluation.local_mse} "
        f"recursive_horizons={tuple(item.horizon for item in evaluation.recursive)} "
        f"nonfinite={evaluation.nonfinite_failures} "
        f"execution_failures={len(evaluation.execution_failures)} "
        f"model_evaluations={evaluation.model_evaluations}",
        flush=True,
    )
    print(f"[validate] wrote {args.output}", flush=True)
    return 0


def _rank(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    fields = (
        "protocol",
        "scale",
        "carrier",
        "seed",
        "checkpoint",
        "ranking_bundle",
        "output",
    )
    missing = tuple(name for name in fields if getattr(args, name) is None)
    if missing:
        parser.error(
            "--rank requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if args.output.exists():
        raise LineageScalingError(f"ranking output already exists: {args.output}")
    print("[load 1/3] frozen lineage-scaling protocol", flush=True)
    protocol = load_lineage_scaling_protocol(args.protocol)
    cell = TrainingCell(args.scale, CarrierKind(args.carrier), args.seed)
    print("[load 2/3] exact cell checkpoint", flush=True)
    model, checkpoint = load_lineage_scaled_checkpoint(
        args.checkpoint,
        protocol,
        expected_cell=cell,
        device=args.device,
    )
    print("[load 3/3] frozen legal candidate sets and realized costs", flush=True)
    states = load_action_ranking_bundle(args.ranking_bundle)
    training_membership = set(protocol.training_scales[-1].lineage_identities)
    if any(
        state.scenario_lineage_identity in training_membership
        or state.carrier is not cell.carrier
        or state.carrier_identity != protocol.carrier_identity(cell.carrier)
        for state in states
    ):
        raise LineageScalingError(
            "ranking inputs contain role leakage or a carrier mismatch"
        )

    def carrier_goal_cost(state, _candidate, predicted) -> float:
        if state.cost_target is None:
            raise LineageScalingError("ranking state has no frozen carrier goal")
        return float(torch.mean((predicted - state.cost_target) ** 2))

    evaluation = evaluate_action_ranking(
        model,
        states,
        horizon=1,
        predicted_cost=carrier_goal_cost,
        progress=lambda value: print(value, flush=True),
    )
    payload = {
        "schema": "lineage_scaled_action_ranking_evaluation_v1",
        "checkpoint_identity": checkpoint.identity,
        "protocol_identity": protocol.identity,
        "cell_identity": cell.identity,
        "evaluation_role": states[0].exposure_role,
        "state_count": evaluation.state_count,
        "mean_top_action_regret": evaluation.mean_top_action_regret,
        "states": [
            {
                "state_identity": item.state_identity,
                "candidate_set_identity": item.candidate_set_identity,
                "selected_candidate_identity": item.selected_candidate_identity,
                "best_realized_candidate_identity": (
                    item.best_realized_candidate_identity
                ),
                "top_action_regret": item.top_action_regret,
            }
            for item in evaluation.states
        ],
        "execution_failures": list(evaluation.execution_failures),
        "compute": {
            "model_evaluations": evaluation.model_evaluations,
            "wall_seconds": evaluation.wall_seconds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[rank] states={evaluation.state_count} "
        f"mean_top_action_regret={evaluation.mean_top_action_regret} "
        f"execution_failures={len(evaluation.execution_failures)} "
        f"model_evaluations={evaluation.model_evaluations}",
        flush=True,
    )
    print(f"[validate] wrote {args.output}", flush=True)
    return 0


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    missing = tuple(
        name
        for name in ("protocol", "source_bundle", "deployment_bundle")
        if getattr(args, name) is None
    )
    if missing:
        parser.error(
            "--validate requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    protocol = load_lineage_scaling_protocol(args.protocol)
    source = load_carrier_lineage_bundle(args.source_bundle)
    deployment = load_carrier_lineage_bundle(args.deployment_bundle)
    alignment = validate_matched_carrier_lineages(protocol, source, deployment)
    checkpoints = {}
    for value in args.cell_checkpoint:
        try:
            scale, carrier, seed, raw_path = value.split(",", 3)
            cell = TrainingCell(scale, CarrierKind(carrier), int(seed))
        except (ValueError, TypeError) as error:
            parser.error(
                f"invalid --cell-checkpoint {value!r}; expected SCALE,CARRIER,SEED,PATH"
            )
        if cell in checkpoints:
            raise LineageScalingError(f"duplicate checkpoint for {cell.identity}")
        checkpoints[cell] = Path(raw_path)
    validated = validate_lineage_scaled_checkpoint_matrix(
        protocol,
        checkpoints,
        device=args.device,
    )
    print(
        f"[validate] exact issue-61 workflow passed "
        f"scales={len(protocol.training_scales)} "
        f"lineages={alignment['lineage_count']} "
        f"transitions={alignment['transition_count']} "
        f"checkpoints={len(validated)} final_evaluation=unopened",
        flush=True,
    )
    return 0


def _action_tensor(intervention: Mapping[str, object]) -> torch.Tensor:
    action = intervention["engine_relative_action"]
    if not isinstance(action, Mapping):
        raise RuntimeError("public dry-run intervention is malformed")
    drag = action["drag_delta_canvas_pixels"]
    if not isinstance(drag, tuple) or len(drag) != 2:
        raise RuntimeError("public dry-run drag binding is malformed")
    return torch.tensor((
        float(drag[0]) / 480.0,
        float(drag[1]) / 480.0,
        float(action["hold_milliseconds"]) / 1000.0,
        float(action["tap_time_milliseconds"]) / 1000.0,
        1.0,
    ), dtype=torch.float32)


def _physical_diagnostics(labels: Mapping[str, object]) -> dict[str, float | bool | None]:
    result = {}
    for name, value in labels.items():
        if isinstance(value, Mapping) and type(value.get("value")) in (
            bool,
            int,
            float,
            type(None),
        ):
            result[name] = value.get("value")
    return result


def _dry_run(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    release_root = (root / args.release_root).resolve()
    aligned_root = (root / args.aligned_root).resolve()
    visual_root = (root / args.visual_parser_root).resolve()
    migration = (
        None
        if args.migration_recovery is None
        else (root / args.migration_recovery).resolve()
    )
    print("[load 1/3] public non-final source roles", flush=True)
    source_readers = _readers(
        root,
        release_root,
        migration_recovery_authority=migration,
    )
    print("[load 2/3] public issue-59 aligned agent observations", flush=True)
    aligned_readers = tuple(
        CohortV2AlignedObservationReader(aligned_root, source_reader=reader)
        for reader in source_readers
    )
    roles = tuple(reader.rollouts[0].exposure_role for reader in aligned_readers)
    if roles != ("training", "calibration", "model_selection"):
        raise RuntimeError("issue-61 dry run opened a non-public exposure role")
    print("[load 3/3] frozen issue-17 visual parser", flush=True)
    parser_model, parser_checkpoint, _manifest = load_visual_parser_checkpoint(
        visual_root,
        readers=aligned_readers,
        device="cpu",
    )
    deployment_adapter = VisualPlanningObservationAdapter(
        parser_model,
        parser_checkpoint_identity=parser_checkpoint.identity,
        temperatures=parser_checkpoint.temperatures,
        thresholds=parser_checkpoint.thresholds,
        object_kind_temperature=parser_checkpoint.object_kind_temperature,
        latent_dim=197,
        max_entities=15,
    )
    source_codec = CohortV2StateCodec(latent_dim=197, max_entities=15)
    training_reader = aligned_readers[0]
    source_lineages = []
    deployment_lineages = []
    bindings = []
    planning_observation = None
    for rollout in training_reader.rollouts[:1]:
        current_frame = rollout.frame_records[0]
        next_frame = rollout.frame_records[-1]
        current_observation = training_reader.load_agent_observation(
            rollout, current_frame
        )
        next_observation = training_reader.load_agent_observation(rollout, next_frame)
        action = _action_tensor(rollout.intervention)
        transition_identity = f"{rollout.attempt_id}:issue-61-dry-transition"
        trajectory_identity = f"{rollout.attempt_id}:issue-61-dry-trajectory"
        diagnostics = _physical_diagnostics(next_frame.labels)
        source_transition = ContinuousTransitionExample(
            identity=transition_identity,
            context=source_codec.encode(current_frame),
            action=action,
            target=source_codec.encode(next_frame),
            physical_diagnostics=diagnostics,
        )
        deployment_context = deployment_adapter.build(
            TemporalObservationContext(None, current_observation)
        )
        deployment_transition = ContinuousTransitionExample(
            identity=transition_identity,
            context=deployment_context.tensor,
            action=action,
            target=deployment_adapter.build(TemporalObservationContext(
                current_observation, next_observation
            )).tensor,
            physical_diagnostics=diagnostics,
        )
        planning_observation = PlanningObservation(
            identity=current_observation.identity,
            carrier=deployment_context.tensor,
            pig_slots=tuple(
                slot
                for slot, value in enumerate(parser_model.object_vocabulary)
                if value.startswith("pig:")
            ),
            slingshot_anchor=(312, 227),
            agent_rgb=current_observation.png,
            terminal_status=TerminalStatus.ONGOING,
            parser_diagnostics=deployment_context.diagnostics,
            symbols=deployment_context.symbols,
        )
        common = {
            "trajectory_identity": trajectory_identity,
            "scenario_lineage_identity": rollout.scenario_lineage_identity,
            "exposure_role": "training",
            "source_release_identity": training_reader.release_identity,
            "complete": True,
        }
        source_lineages.append(CarrierLineage(
            **common,
            carrier=CarrierKind.SOURCE,
            carrier_identity=source_codec.identity,
            transitions=(source_transition,),
        ))
        deployment_lineages.append(CarrierLineage(
            **common,
            carrier=CarrierKind.DEPLOYMENT,
            carrier_identity=deployment_adapter.identity,
            transitions=(deployment_transition,),
        ))
        bindings.append(TrajectoryLineageBinding(
            trajectory_identity=trajectory_identity,
            scenario_lineage_identity=rollout.scenario_lineage_identity,
            exposure_role="training",
            transition_identities=(transition_identity,),
            initial_observation_identity=current_observation.identity,
            terminal_observation_identity=next_observation.identity,
        ))
        print("[carrier] one complete public scenario-lineage probe", flush=True)
    scale = FrozenLineageScale.from_manifest(
        "public-probe",
        TrajectoryLineageManifest.create(
            training_reader.release_identity, tuple(bindings)
        ),
    )
    alignment = validate_carrier_alignment(
        scale,
        tuple(source_lineages),
        tuple(deployment_lineages),
        source_carrier_identity=source_codec.identity,
        deployment_carrier_identity=deployment_adapter.identity,
    )
    torch.manual_seed(20260830)
    predictor = DualOutputPredictor(PredictorConfig(
        latent_dim=197,
        action_dim=5,
        hidden_dim=384,
        depth=3,
    ))
    scoring = evaluate_continuous_prediction(
        predictor,
        (deployment_lineages[0],),
        horizons=(1, 15),
        progress=lambda value: print(value, flush=True),
    )
    if planning_observation is None:
        raise RuntimeError("public dry run found no planning observation")
    gameplay_protocol = MatchedGameplayProtocol(
        action_candidate_set_identity="issue-61-public-dry-legal-actions-v1",
        cost_terms_identity="cohort-v2-gameplay-cost-v1",
        action_bounds=SlingshotActionBounds(
            drag_x=(-160, -40),
            drag_y=(-80, 80),
            tap_time_ms=(0, 1000),
        ),
        cost_config=GameplayCostConfig(
            goal_progress_weight=10.0,
            terminal_success_cost=-20.0,
            terminal_failure_cost=20.0,
            illegal_action_cost=50.0,
            physical_penalty_weight=1.0,
            rollout_penalty_weight=0.1,
            compute_weight=1e-8,
            structure_unstable_weight=0.0,
        ),
        population_size=4,
        elite_count=2,
        cem_iterations=1,
        sequence_length=2,
        max_shots=2,
        max_planner_compute=1e12,
        fixed_steps_per_shot=1,
        transition_compute=1.0,
        controller_compute=1.0,
        seed=20260831,
    )
    gameplay_system_specs = matched_gameplay_systems(
        gameplay_protocol,
        GameplayCheckpointBindings(
            legacy_predictor=(root / ".local-artifacts/issue-15-legacy.pt").resolve(),
            retrained_predictor=(root / ".local-artifacts/issue-61-retrained.pt").resolve(),
            adaptive_controller=(root / ".local-artifacts/issue-15-controller.pt").resolve(),
        ),
    )
    gameplay_planners = build_matched_gameplay_planners(
        gameplay_protocol,
        gameplay_system_specs,
        predictor_loader=lambda _path: predictor,
        adaptive_selector_loader=lambda _path: (
            lambda _observation, _action: 15
        ),
        progress=lambda value: print(value, flush=True),
    )
    plans = tuple(item.planner.plan(planning_observation) for item in gameplay_planners)
    print(
        f"[validate] public carrier probe lineages={alignment['lineage_count']} "
        f"transitions={alignment['transition_count']} carriers=source,deployment",
        flush=True,
    )
    print(
        f"[score] local={scoring.local_mse} "
        f"recursive_horizons={tuple(item.horizon for item in scoring.recursive)} "
        f"model_evaluations={scoring.model_evaluations} "
        f"nonfinite={scoring.nonfinite_failures} "
        f"execution_failures={len(scoring.execution_failures)}",
        flush=True,
    )
    print(
        f"[cem/mpc] systems={len(plans)} "
        f"candidates={sum(item.candidate_count for item in plans)} "
        f"model_rollouts={sum(item.model_rollout_count for item in plans)}",
        flush=True,
    )
    print(
        "[dry-run] public training/calibration/model_selection roles only; "
        "final_evaluation unopened; no files written",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    if args.score:
        return _score(args, parser)
    if args.rank:
        return _rank(args, parser)
    if args.validate:
        return _validate(args, parser)
    _required(args, parser)
    return _train(args)


if __name__ == "__main__":
    raise SystemExit(main())
