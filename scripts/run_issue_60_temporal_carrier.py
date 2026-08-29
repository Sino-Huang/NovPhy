"""No-write validation of issue 60's deployment temporal carrier contract."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import torch

from scripts.cohort_v2_migration_recovery import DEFAULT_MANIFEST
from scripts.run_cohort_v2_macro_experiment import DEFAULT_RELEASE, _readers
from scripts.run_cohort_v2_visual_parser import (
    DEFAULT_ALIGNED,
    DEFAULT_OUTPUT as VISUAL_OUTPUT,
)
from world_model.data import CohortV2AlignedObservationReader
from world_model.data.deployment_temporal import (
    DecisionTargets,
    DecisionTransition,
    DeploymentCarrierDataset,
    DeploymentTrajectory,
    DeploymentTrajectoryReader,
    ExecutedAction,
    TrajectoryLineageBinding,
)
from world_model.planning.gameplay import (
    TerminalStatus,
    VisualPlanningObservationAdapter,
)
from world_model.training.cohort_v2_visual_parser import load_visual_parser_checkpoint


DEFAULT_VISUAL_PARSER: Final = VISUAL_OUTPUT / "parser"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--visual-parser-root", type=Path, default=DEFAULT_VISUAL_PARSER)
    parser.add_argument(
        "--migration-recovery",
        type=Path,
        nargs="?",
        const=DEFAULT_MANIFEST,
        metavar="MANIFEST",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    release_root = (root / args.release_root).resolve()
    aligned_root = (root / args.aligned_root).resolve()
    parser_root = (root / args.visual_parser_root).resolve()
    migration = (
        None
        if args.migration_recovery is None
        else (root / args.migration_recovery).resolve()
    )
    source_readers = _readers(
        root,
        release_root,
        migration_recovery_authority=migration,
    )
    aligned = tuple(
        CohortV2AlignedObservationReader(aligned_root, source_reader=reader)
        for reader in source_readers
    )
    if tuple(reader.rollouts[0].exposure_role for reader in aligned) != (
        "training",
        "calibration",
        "model_selection",
    ):
        raise RuntimeError("issue-60 dry run opened a non-public exposure role")
    model, checkpoint, _manifest = load_visual_parser_checkpoint(
        parser_root,
        readers=aligned,
        device="cpu",
    )
    adapter = VisualPlanningObservationAdapter(
        model,
        parser_checkpoint_identity=checkpoint.identity,
        temperatures=checkpoint.temperatures,
        thresholds=checkpoint.thresholds,
        object_kind_temperature=checkpoint.object_kind_temperature,
        latent_dim=197,
        max_entities=15,
    )
    reader = aligned[0]
    rollout = next(item for item in reader.rollouts if len(item.frame_records) >= 2)
    current_frame_record = rollout.frame_records[0]
    next_frame_record = rollout.frame_records[-1]
    current = reader.load_agent_observation(rollout, current_frame_record)
    next_observation = reader.load_agent_observation(rollout, next_frame_record)
    intervention = rollout.intervention
    action = ExecutedAction(
        identity=f"{rollout.attempt_id}:executed-intervention",
        interface_action=intervention["interface_action"],
        engine_relative_action=intervention["engine_relative_action"],
        legal=True,
    )
    terminal_status = (
        str(next_frame_record.terminal["reason"])
        if next_frame_record.terminal is not None
        else "ongoing"
    )
    transition = DecisionTransition(
        identity=f"{rollout.attempt_id}:decision-transition-probe",
        scenario_lineage_identity=rollout.scenario_lineage_identity,
        exposure_role=rollout.exposure_role,
        decision_index=0,
        prior_observation=None,
        current_observation=current,
        action=action,
        targets=DecisionTargets(
            next_observation=next_observation,
            source_frame_record_identity=next_frame_record.identity,
            source_state_identity=next_frame_record.state_id,
            source_targets=next_frame_record.labels,
        ),
        terminal_status=terminal_status,
        source_bindings={
            "release_identity": reader.release_identity,
            "partition_identity": reader.partition_identity,
            "attempt_id": rollout.attempt_id,
        },
    )
    trajectory = DeploymentTrajectory(
        identity=f"{rollout.attempt_id}:complete-probe-trajectory",
        scenario_lineage_identity=rollout.scenario_lineage_identity,
        exposure_role=rollout.exposure_role,
        transitions=(transition,),
        complete=True,
    )
    trajectory_reader = DeploymentTrajectoryReader(
        (trajectory,),
        exposure_role="training",
        lineage_bindings=(TrajectoryLineageBinding(
            trajectory_identity=trajectory.identity,
            scenario_lineage_identity=trajectory.scenario_lineage_identity,
            exposure_role=trajectory.exposure_role,
            transition_identities=(transition.identity,),
            initial_observation_identity=current.identity,
            terminal_observation_identity=next_observation.identity,
        ),),
    )
    DeploymentTrajectoryReader.validate_role_isolation((trajectory_reader,))
    carriers = DeploymentCarrierDataset(trajectory_reader, adapter)[0]
    gameplay = adapter.from_temporal_context(
        transition.inference.observations,
        slingshot_anchor=(312, 227),
        terminal_status=TerminalStatus.ONGOING,
    )
    torch.testing.assert_close(carriers.context.tensor, gameplay.carrier)
    available_motion = sum(
        item.motion_available for item in carriers.context.object_slots
    )
    print(
        "[dry-run] issue-59 public roles validated; final_evaluation unopened",
        flush=True,
    )
    print(
        f"[dry-run] transition={transition.schema} prior=unavailable "
        f"current={current.fixed_step} next={next_observation.fixed_step} "
        f"motion_available_slots={available_motion} action={action.identity}",
        flush=True,
    )
    print(
        f"[dry-run] shared carrier adapter={adapter.identity} "
        f"shape={tuple(carriers.context.tensor.shape)}; no files written",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("issue 60 provides only the no-write --dry-run")
    return _dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
