"""Derive and ablate model-relative micro reliability for issue #12."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Final

from scripts.run_cohort_v2_macro_experiment import (
    DEFAULT_RELEASE,
    _readers,
)
from world_model.training.cohort_v2_micro import (
    CohortV2MicroTrainer,
    CohortV2MicroTrainingData,
)
from world_model.training.cohort_v2_reliability import (
    CohortV2ReliabilityConfig,
    derive_cohort_v2_reliability_labels,
    evaluate_cohort_v2_reliability_models,
    preliminary_checkpoint_identity,
    reliability_symbolic_gate,
    score_micro_carrier_objective,
    split_reliability_training_attempts,
    train_cohort_v2_reliability_models,
    validate_cohort_v2_reliability_artifact,
    write_cohort_v2_reliability_artifact,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.loop import seed_all
from world_model.training.manifest import git_revision


DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-12-reliability")


def _short_identity(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--preliminary-steps", type=int, default=1200)
    parser.add_argument("--final-steps", type=int, default=1200)
    parser.add_argument("--estimator-epochs", type=int, default=80)
    parser.add_argument("--controller-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact-report", type=Path)
    parser.add_argument("--implementation-commit")
    return parser


def _config(args: argparse.Namespace) -> CohortV2ReliabilityConfig:
    return CohortV2ReliabilityConfig(
        preliminary_steps=args.preliminary_steps,
        final_steps=args.final_steps,
        estimator_epochs=args.estimator_epochs,
        controller_epochs=args.controller_epochs,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        seed=args.seed,
        device=args.device,
    )


def _train_predictor(
    trainer: CohortV2MicroTrainer,
    *,
    label: str,
    log_every: int,
) -> None:
    first = None
    latest = None
    for step in range(trainer.config.steps):
        latest = trainer.train_step()
        if first is None:
            first = latest.total_loss
        if step == 0 or step + 1 == trainer.config.steps or (step + 1) % log_every == 0:
            print(
                f"[train:{label} {step + 1}/{trainer.config.steps}] "
                f"pair=h{latest.pair.delta}/{latest.pair.abstraction} "
                f"total={latest.total_loss:.6f} carrier={latest.carrier_loss:.6f} "
                f"micro={latest.micro_loss:.6f}",
                flush=True,
            )
    assert first is not None and latest is not None
    print(
        f"[train:{label}] first_loss={first:.6f} final_loss={latest.total_loss:.6f}",
        flush=True,
    )


def _scores(
    model_scores: dict[str, object],
    ungated_objective: float,
    gated_objective: float,
) -> dict[str, object]:
    gate_value = ungated_objective - gated_objective
    controller = model_scores["controller_feature_ablation"]
    assert isinstance(controller, dict)
    controller_value = float(controller["incremental_held_out_value"])
    keep = gate_value > 0.0 or controller_value > 0.0
    return {
        **model_scores,
        "keep_remove_decision": {
            "decision": "keep" if keep else "remove",
            "rule": "keep iff either independent path has positive incremental model-selection value",
            "controller_feature_incremental_held_out_value": controller_value,
            "loss_gate_incremental_held_out_value": gate_value,
        },
        "loss_gate_ablation": {
            "controller_feature_fixed": "absent",
            "gated_mean_micro_carrier_objective": gated_objective,
            "incremental_held_out_value": gate_value,
            "ungated_mean_micro_carrier_objective": ungated_objective,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.log_every <= 0:
        raise ValueError("progress interval must be positive")
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    output = (repository_root / args.output).resolve()
    config = _config(args)
    implementation_revision = args.implementation_commit
    if not args.dry_run and not args.validate and implementation_revision is None:
        implementation_revision, dirty = git_revision(str(repository_root))
        if dirty:
            parser.error(
                "a dirty worktree requires --implementation-commit for a provenance-bound run"
            )
    if args.dry_run:
        config = replace(
            config,
            preliminary_steps=6,
            final_steps=6,
            estimator_epochs=1,
            controller_epochs=1,
            batch_size=min(2, config.batch_size),
            evaluation_batch_size=min(16, config.evaluation_batch_size),
        )
        print("[dry-run] reduced training and bounded scoring; no files will be written", flush=True)

    print("[load] validating public training/calibration/model-selection readers", flush=True)
    readers = _readers(repository_root, release_root)
    if args.validate:
        manifest = validate_cohort_v2_reliability_artifact(
            output, readers=readers, config=config
        )
        print(
            f"[validate] passed artifact={_short_identity(str(manifest['artifact_identity']))} "
            f"training_labels={manifest['training_label_count']} "
            f"model_selection_labels={manifest['model_selection_label_count']}",
            flush=True,
        )
        return 0

    preliminary_attempts, label_attempts = split_reliability_training_attempts(
        readers[0]
    )
    print(
        f"[split] preliminary_train={len(preliminary_attempts)} "
        f"out_of_sample_label={len(label_attempts)} complete_rollouts",
        flush=True,
    )
    seed_all(config.seed)
    preliminary_data = CohortV2MicroTrainingData(
        readers[0],
        config.micro_config,
        included_attempt_ids=preliminary_attempts,
    )
    preliminary = CohortV2MicroTrainer(preliminary_data, config.micro_config)
    _train_predictor(preliminary, label="preliminary", log_every=args.log_every)
    checkpoint_identity = preliminary_checkpoint_identity(
        preliminary, preliminary_attempts
    )
    print(f"[freeze] checkpoint={_short_identity(checkpoint_identity)}", flush=True)

    derivation = derive_cohort_v2_reliability_labels(
        preliminary.predictor,
        preliminary.codec,
        readers,
        label_attempts,
        config,
        checkpoint_identity,
        max_examples_per_role=16 if args.dry_run else None,
        progress=lambda message: print(message, flush=True),
    )
    print(
        f"[labels] available={len(derivation.labels)} "
        f"excluded_unavailable={derivation.excluded_unavailable_count} "
        f"derivation={_short_identity(derivation.target_identity)}",
        flush=True,
    )
    estimator, raw_controller, feature_controller = (
        train_cohort_v2_reliability_models(
            derivation, config, progress=lambda message: print(message, flush=True)
        )
    )
    model_scores = evaluate_cohort_v2_reliability_models(
        derivation, estimator, raw_controller, feature_controller
    )

    final_data = CohortV2MicroTrainingData(readers[0], config.final_micro_config)
    seed_all(config.final_micro_config.seed)
    ungated = CohortV2MicroTrainer(final_data, config.final_micro_config)
    _train_predictor(ungated, label="loss_ungated", log_every=args.log_every)
    seed_all(config.final_micro_config.seed)
    gated = CohortV2MicroTrainer(
        final_data,
        config.final_micro_config,
        symbolic_gate=reliability_symbolic_gate(estimator, config, readers),
    )
    _train_predictor(gated, label="loss_gated", log_every=args.log_every)
    score_limit = 16 if args.dry_run else None
    ungated_objective = score_micro_carrier_objective(
        ungated,
        readers[2],
        config,
        max_examples=score_limit,
        progress=lambda message: print(message, flush=True),
    )
    gated_objective = score_micro_carrier_objective(
        gated,
        readers[2],
        config,
        max_examples=score_limit,
        progress=lambda message: print(message, flush=True),
    )
    scores = _scores(model_scores, ungated_objective, gated_objective)
    decision = scores["keep_remove_decision"]
    assert isinstance(decision, dict)
    print(
        f"[decision] {decision['decision']} "
        f"loss_gate_value={decision['loss_gate_incremental_held_out_value']:.8f} "
        f"controller_feature_value={decision['controller_feature_incremental_held_out_value']:.8f}",
        flush=True,
    )
    if args.dry_run:
        print("[dry-run] pipeline passed; no files written", flush=True)
        return 0

    assert implementation_revision is not None
    manifest = write_cohort_v2_reliability_artifact(
        output,
        readers=readers,
        config=config,
        preliminary_attempt_ids=preliminary_attempts,
        label_attempt_ids=label_attempts,
        derivation=derivation,
        estimator=estimator,
        raw_controller=raw_controller,
        feature_controller=feature_controller,
        preliminary_trainer=preliminary,
        ungated_trainer=ungated,
        gated_trainer=gated,
        scores=scores,
        implementation_revision=implementation_revision,
    )
    validate_cohort_v2_reliability_artifact(
        output, readers=readers, config=config
    )
    if args.compact_report is not None:
        report_path = (repository_root / args.compact_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(canonical_json_bytes({
            "artifact_identity": manifest["artifact_identity"],
            "derivation_identity": derivation.target_identity,
            "final_evaluation_consumed": False,
            "implementation_commit": implementation_revision,
            "release_identity": readers[0].release_identity,
            "rerun_commands": [
                "python -u -m scripts.run_cohort_v2_reliability --dry-run",
                "python -u -m scripts.run_cohort_v2_reliability "
                f"--implementation-commit {implementation_revision}",
                "python -u -m scripts.run_cohort_v2_reliability --validate",
            ],
            "schema": "cohort_v2_micro_reliability_summary_v1",
            "scores": scores,
        }))
        print(f"[report] {report_path}", flush=True)
    print(
        f"[complete] artifact={_short_identity(str(manifest['artifact_identity']))} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
