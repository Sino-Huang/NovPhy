"""Adapt the issue #10 controller to closed-loop predicted carrier states."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Final

from scripts.run_cohort_v2_controller import DEFAULT_OUTPUT as DEFAULT_CONTROLLERS
from scripts.run_cohort_v2_pair_measurements import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION,
    DEFAULT_RELEASE,
)
from scripts.run_cohort_v2_policy_baselines import DEFAULT_OUTPUT as DEFAULT_BASELINES
from scripts.run_cohort_v2_trajectory_labels import (
    DEFAULT_MEASUREMENTS,
    DEFAULT_OUTPUT as DEFAULT_TRAJECTORY_LABELS,
    _source_inputs,
    issue_8_cost_spec,
)
from world_model.training import (
    CohortV2AggregationConfig,
    CohortV2MacroConfig,
    generate_cohort_v2_trajectory_labels,
    git_revision,
    load_cohort_v2_aggregated_controllers,
    load_cohort_v2_controller_checkpoint,
    load_cohort_v2_macro_checkpoint,
    run_cohort_v2_controller_aggregation,
    validate_cohort_v2_controller_aggregation,
    validate_cohort_v2_controllers,
    validate_cohort_v2_policy_baselines,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_controller_aggregation,
)
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_PREDICTOR: Final = Path(".local-artifacts/issue-6-macro-experiment/checkpoint.pt")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-11-controller-aggregation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--measurements-root", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--trajectory-labels-root", type=Path, default=DEFAULT_TRAJECTORY_LABELS)
    parser.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--controllers-root", type=Path, default=DEFAULT_CONTROLLERS)
    parser.add_argument("--predictor-checkpoint", type=Path, default=DEFAULT_PREDICTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def _score_line(score) -> str:
    return (
        f"[score:{score.name}] rollouts={score.rollout_count} "
        f"decisions={score.decision_count} "
        f"endpoint_mse={score.mean_terminal_carrier_mse:.6f} "
        f"endpoint_violation={score.mean_endpoint_violation_rate:.6f} "
        f"pair_cost={score.mean_selected_segment_cost:.6f} "
        f"pair_regret={score.mean_pair_regret:.6f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    evaluation_root = (repository_root / args.evaluation_root).resolve()
    checkpoint_path = (repository_root / args.checkpoint).resolve()
    measurements_root = (repository_root / args.measurements_root).resolve()
    trajectory_root = (repository_root / args.trajectory_labels_root).resolve()
    baselines_root = (repository_root / args.baselines_root).resolve()
    controllers_root = (repository_root / args.controllers_root).resolve()
    predictor_path = (repository_root / args.predictor_checkpoint).resolve()
    output = (repository_root / args.output).resolve()
    implementation_revision = args.implementation_commit
    if implementation_revision is None:
        implementation_revision, dirty = git_revision(str(repository_root))
        if dirty and not args.dry_run and not args.validate:
            raise ValueError("a dirty worktree requires an explicit --implementation-commit")

    print("[load] validating issue #7 pair evidence", flush=True)
    evaluation, measurement, calibration, readers = _source_inputs(
        repository_root,
        release_root,
        evaluation_root,
        checkpoint_path,
        measurements_root,
    )
    spec = issue_8_cost_spec(calibration)
    trajectory_manifest = json.loads((trajectory_root / "manifest.json").read_bytes())
    print("[load] validating issue #8 trajectory labels", flush=True)
    trajectory_receipt = validate_cohort_v2_trajectory_labels(
        trajectory_root,
        evaluation,
        measurement,
        spec,
        implementation_revision=trajectory_manifest["implementation_revision"],
    )
    labels = generate_cohort_v2_trajectory_labels(evaluation, measurement, spec)
    baseline_manifest = json.loads((baselines_root / "manifest.json").read_bytes())
    print("[load] validating issue #9 policy baselines", flush=True)
    baseline_receipt = validate_cohort_v2_policy_baselines(
        baselines_root,
        evaluation,
        measurement,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=baseline_manifest["derivation_index_identity"],
        implementation_revision=baseline_manifest["implementation_revision"],
    )

    controller_manifest = json.loads((controllers_root / "manifest.json").read_bytes())
    print("[load] validating issue #10 oracle-state controller baseline", flush=True)
    validate_cohort_v2_controllers(
        controllers_root,
        readers,
        evaluation,
        measurement,
        labels,
        spec,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        baseline_artifact_identity=baseline_receipt.baseline_artifact_identity,
        derivation_index_identity=controller_manifest["derivation_index_identity"],
        implementation_revision=controller_manifest["implementation_revision"],
    )
    baseline_models, controller_config, controller_checkpoint_identity = (
        load_cohort_v2_controller_checkpoint(controllers_root / "checkpoint.pt")
    )

    print(f"[load] loading issue #6 shared predictor on {args.device}", flush=True)
    predictor, codec, predictor_checkpoint = load_cohort_v2_macro_checkpoint(
        predictor_path,
        reader=readers[0],
        config=CohortV2MacroConfig(),
        device=args.device,
    )
    if predictor_checkpoint.identity != evaluation.checkpoint_identity:
        raise ValueError("aggregation predictor differs from the accepted pair objective")

    aggregation_config = CohortV2AggregationConfig(args.rounds)
    if args.validate:
        manifest = validate_cohort_v2_controller_aggregation(output)
        load_cohort_v2_aggregated_controllers(output)
        print(
            f"[validate] passed rounds={manifest['aggregation_config']['rounds']} "
            f"artifact={manifest['aggregation_artifact_identity']}",
            flush=True,
        )
        return 0

    training_config = replace(controller_config, epochs=1) if args.dry_run else controller_config
    if args.dry_run:
        print("[dry-run] using one rollout per role and one controller epoch", flush=True)
    print(
        f"[aggregate] rounds={aggregation_config.rounds} epochs={training_config.epochs} "
        f"device={args.device}",
        flush=True,
    )
    run = run_cohort_v2_controller_aggregation(
        readers,
        evaluation,
        measurement,
        labels,
        spec,
        predictor,
        codec,
        baseline_models,
        training_config,
        aggregation_config,
        progress=lambda message: print(message, flush=True),
        rollout_limit=1 if args.dry_run else None,
    )
    for score in run.result.scores:
        print(_score_line(score), flush=True)
    if args.dry_run:
        print(
            f"[dry-run] passed aggregated_states={len(run.result.states)}; no files written",
            flush=True,
        )
        return 0

    manifest = write_cohort_v2_controller_aggregation(
        output,
        run,
        controller_config,
        aggregation_config,
        evaluation,
        measurement,
        spec,
        source_controller_artifact_identity=controller_manifest["controller_artifact_identity"],
        source_controller_checkpoint_identity=controller_checkpoint_identity,
        source_predictor_checkpoint_identity=predictor_checkpoint.identity,
        trajectory_label_artifact_identity=trajectory_receipt.label_artifact_identity,
        derivation_index_identity=controller_manifest["derivation_index_identity"],
        implementation_revision=implementation_revision,
    )
    validate_cohort_v2_controller_aggregation(output)
    load_cohort_v2_aggregated_controllers(output)
    print(
        f"[complete] aggregated_states={len(run.result.states)} output={output}",
        flush=True,
    )
    if args.compact_report is not None:
        scores = json.loads((output / "scores.json").read_bytes())
        report = {
            "artifact_type": "cohort_v2_controller_aggregation_summary",
            "aggregation_artifact_identity": manifest["aggregation_artifact_identity"],
            "aggregation_rounds": args.rounds,
            "final_evaluation_consumed": False,
            "implementation_commit": implementation_revision,
            "release_identity": evaluation.release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_controller_aggregation --dry-run",
                "python -u -m scripts.run_cohort_v2_controller_aggregation "
                f"--implementation-commit {implementation_revision}",
                "python -u -m scripts.run_cohort_v2_controller_aggregation --validate",
            ],
            "schema": "cohort_v2_controller_aggregation_summary_v1",
            "scores": scores["scores"],
            "source_cohort_mutated": False,
            "source_controller_artifact_identity": controller_manifest[
                "controller_artifact_identity"
            ],
            "source_predictor_checkpoint_identity": predictor_checkpoint.identity,
        }
        report_path = (repository_root / args.compact_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes(report))
        print(f"[report] {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
