"""Train and evaluate issue #10 joint and matched-capacity controllers."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Final

from scripts.run_cohort_v2_pair_measurements import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION,
    DEFAULT_RELEASE,
)
from scripts.run_cohort_v2_policy_baselines import (
    DEFAULT_OUTPUT as DEFAULT_BASELINES,
)
from scripts.run_cohort_v2_trajectory_labels import (
    DEFAULT_MEASUREMENTS,
    DEFAULT_OUTPUT as DEFAULT_TRAJECTORY_LABELS,
    _source_inputs,
    issue_8_cost_spec,
)
from world_model.training import (
    CohortV2ControllerConfig,
    build_cohort_v2_controller_examples,
    evaluate_cohort_v2_controllers,
    generate_cohort_v2_trajectory_labels,
    git_revision,
    train_cohort_v2_controllers,
    validate_cohort_v2_controllers,
    validate_cohort_v2_policy_baselines,
    validate_cohort_v2_trajectory_labels,
    write_cohort_v2_controllers,
)
from world_model.training.grid_artifacts import canonical_json_bytes


DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-10-controller")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--measurements-root", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--trajectory-labels-root", type=Path, default=DEFAULT_TRAJECTORY_LABELS)
    parser.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def _print_scores(result) -> None:
    for score in result.scores:
        regret = "unavailable" if score.mean_pair_regret is None else f"{score.mean_pair_regret:.6f}"
        print(
            f"[score:{score.exposure_role}:{score.controller_id}] "
            f"pair_accuracy={score.pair_accuracy:.4f} "
            f"horizon_accuracy={score.horizon_accuracy:.4f} "
            f"mode_accuracy={score.description_mode_accuracy:.4f} "
            f"utility={score.utility_available_count}/{score.state_count} "
            f"mean_regret={regret}",
            flush=True,
        )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    evaluation_root = (repository_root / args.evaluation_root).resolve()
    checkpoint_path = (repository_root / args.checkpoint).resolve()
    measurements_root = (repository_root / args.measurements_root).resolve()
    trajectory_root = (repository_root / args.trajectory_labels_root).resolve()
    baselines_root = (repository_root / args.baselines_root).resolve()
    output = (repository_root / args.output).resolve()
    implementation_revision = args.implementation_commit
    if implementation_revision is None:
        implementation_revision, dirty = git_revision(str(repository_root))
        if dirty and not args.dry_run:
            parser.error("a dirty worktree requires an explicit --implementation-commit")

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
    derivations = {reader.derivation_identity for reader in readers}
    if derivations != {baseline_manifest["derivation_index_identity"]}:
        raise ValueError("controller readers differ from the accepted derivation index")
    derivation_identity = next(iter(derivations))

    config = CohortV2ControllerConfig()
    if args.dry_run:
        config = replace(config, epochs=1)
        print("[dry-run] using one training epoch per controller", flush=True)
        examples = build_cohort_v2_controller_examples(readers, labels, config)
        print(
            f"[data] training={sum(item.exposure_role == 'training' for item in examples)} "
            f"held_out={sum(item.exposure_role != 'training' for item in examples)}",
            flush=True,
        )
        models = train_cohort_v2_controllers(
            examples,
            evaluation.grid.pairs,
            config,
            progress=lambda message: print(message, flush=True),
        )
        result = evaluate_cohort_v2_controllers(
            models, examples, evaluation, measurement, spec
        )
        _print_scores(result)
        print("[dry-run] no files written", flush=True)
        return 0

    kwargs = {
        "trajectory_label_artifact_identity": trajectory_receipt.label_artifact_identity,
        "baseline_artifact_identity": baseline_receipt.baseline_artifact_identity,
        "derivation_index_identity": derivation_identity,
        "implementation_revision": implementation_revision,
    }
    if args.validate:
        print("[validate] loading checkpoint and recomputing held-out metrics", flush=True)
        receipt = validate_cohort_v2_controllers(
            output, readers, evaluation, measurement, labels, spec, **kwargs
        )
    else:
        print(
            f"[train] epochs={config.epochs} batch_size={config.batch_size} seed={config.seed}",
            flush=True,
        )
        receipt = write_cohort_v2_controllers(
            output,
            readers,
            evaluation,
            measurement,
            labels,
            spec,
            config,
            progress=lambda message: print(message, flush=True),
            **kwargs,
        )
    manifest = json.loads((output / "manifest.json").read_bytes())
    scores = json.loads((output / "scores.json").read_bytes())
    for score in scores["scores"]:
        regret = score["mean_pair_regret"]
        print(
            f"[score:{score['exposure_role']}:{score['controller_id']}] "
            f"pair_accuracy={score['pair_accuracy']:.4f} "
            f"horizon_accuracy={score['horizon_accuracy']:.4f} "
            f"mode_accuracy={score['description_mode_accuracy']:.4f} "
            f"utility={score['utility_available_count']}/{score['state_count']} "
            f"mean_regret={'unavailable' if regret is None else f'{regret:.6f}'}",
            flush=True,
        )
    print(
        f"[complete] training_states={receipt.training_state_count} "
        f"held_out_states={receipt.evaluation_state_count} "
        f"parameters_each={receipt.parameter_count} output={output}",
        flush=True,
    )
    if args.compact_report is not None:
        report = {
            "artifact_type": "cohort_v2_controller_summary",
            "baseline_artifact_identity": baseline_receipt.baseline_artifact_identity,
            "checkpoint_identity": receipt.checkpoint_identity,
            "controller_artifact_identity": receipt.controller_artifact_identity,
            "controllers": manifest["controllers"],
            "deployment_inputs": manifest["deployment_inputs"],
            "evaluation_state_count": receipt.evaluation_state_count,
            "final_evaluation_consumed": False,
            "implementation_commit": implementation_revision,
            "matched_parameter_count": receipt.parameter_count,
            "oracle_engine_state_is_controller_input": False,
            "release_identity": evaluation.release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_controller --dry-run",
                "python -u -m scripts.run_cohort_v2_controller "
                f"--implementation-commit {implementation_revision}",
                "python -u -m scripts.run_cohort_v2_controller --validate "
                f"--implementation-commit {implementation_revision}",
            ],
            "schema": "cohort_v2_controller_summary_v1",
            "scores": scores["scores"],
            "source_bound_validation": "passed",
            "training_state_count": receipt.training_state_count,
            "trajectory_label_artifact_identity": trajectory_receipt.label_artifact_identity,
        }
        path = (repository_root / args.compact_report).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(report))
        print(f"[report] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
