"""Diagnose shared-predictor interference with pair experts (issue #14)."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Final

import torch

from world_model.data import CohortV2ReleaseReader
from world_model.model import PredictionPair
from world_model.training import (
    CohortV2ParallelExhaustiveEvaluator,
    load_cohort_v2_evaluation,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_macro import (
    MACRO_CAPABILITIES,
    MACRO_EVENT_ENDPOINT_AUTHORITY,
    MACRO_PAIRS,
    MACRO_STATE_AUTHORITY,
    CohortV2MacroCheckpoint,
    CohortV2MacroConfig,
    CohortV2MacroError,
    CohortV2MacroPairScorer,
    CohortV2MacroTrainer,
    CohortV2MacroTrainingData,
    load_cohort_v2_macro_checkpoint,
    save_cohort_v2_macro_checkpoint,
)
from world_model.training.cohort_v2_micro import MICRO_RELATION_AUTHORITY
from world_model.training.cohort_v2_pair_experts import (
    CohortV2PairExpertScorer,
    CohortV2PairExpertTrainer,
    CohortV2PairExpertTrainingData,
    compare_preferred_pair_maps,
    pair_label,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.loop import seed_all


DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-14-pair-experts")
ROLE_INFLUENCE: Final = (
    ("training", "learned_parameters"),
    ("calibration", "threshold_values"),
    ("model_selection", "configuration_selection"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation-devices",
        default="auto",
        help="comma-separated devices; auto uses every visible GPU",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--updates-per-pair", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--symbolic-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--score-log-every", type=int, default=250)
    parser.add_argument("--implementation-commit", default="working-tree")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def _readers(repository_root: Path, release_root: Path):
    declaration = repository_root / "docs/data_contracts/cohort_v2_capabilities_v1.json"
    production_plan = repository_root / "data/runtime_evidence/issue-53-plan-v5"
    readers = []
    for index, (role, influence) in enumerate(ROLE_INFLUENCE, start=1):
        print(f"[load {index}/3] validating {role} release records", flush=True)
        reader = CohortV2ReleaseReader(
            release_root,
            capability_declaration_path=declaration,
            production_plan_root=production_plan,
            workflow_kind=role,
            influence=influence,
        )
        frame_count = sum(len(rollout.frame_records) for rollout in reader.rollouts)
        print(
            f"[load {index}/3] {role}: rollouts={len(reader.rollouts)} "
            f"frame_records={frame_count}",
            flush=True,
        )
        readers.append(reader)
    return tuple(readers)


def _config(args: argparse.Namespace, steps: int) -> CohortV2MacroConfig:
    return CohortV2MacroConfig(
        seed=args.seed,
        steps=steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        symbolic_weight=args.symbolic_weight,
        device=args.device,
    )


def _evaluation_devices(args: argparse.Namespace) -> tuple[str, ...]:
    if args.evaluation_batch_size <= 0:
        raise CohortV2MacroError("evaluation batch size must be positive")
    if args.evaluation_devices == "auto":
        if torch.device(args.device).type == "cuda":
            count = torch.cuda.device_count()
            if count == 0:
                raise CohortV2MacroError("CUDA requested but no GPU is visible")
            return tuple(f"cuda:{index}" for index in range(count))
        return (args.device,)
    devices = tuple(value.strip() for value in args.evaluation_devices.split(","))
    if not devices or any(not value for value in devices) or len(set(devices)) != len(devices):
        raise CohortV2MacroError("evaluation devices must be unique and nonempty")
    for value in devices:
        device = torch.device(value)
        if device.type == "cuda" and (
            device.index is None or device.index >= torch.cuda.device_count()
        ):
            raise CohortV2MacroError(f"evaluation device is unavailable: {value}")
    return devices


def _expert_name(pair: PredictionPair) -> str:
    return f"h{pair.delta}-{pair.abstraction}"


def _empty_losses() -> dict[PredictionPair, list[object]]:
    return {pair: [] for pair in MACRO_PAIRS}


def _loss_report(results: dict[PredictionPair, list[object]]) -> list[dict[str, object]]:
    rows = []
    for pair in MACRO_PAIRS:
        values = results[pair]
        if not values:
            raise CohortV2MacroError(f"no training results recorded for {pair_label(pair)}")
        rows.append({
            "mean_carrier_loss": sum(item.carrier_loss for item in values) / len(values),
            "mean_macro_loss": sum(item.macro_loss for item in values) / len(values),
            "mean_micro_loss": sum(item.micro_loss for item in values) / len(values),
            "mean_total_loss": sum(item.total_loss for item in values) / len(values),
            "optimizer_update_count": len(values),
            "pair": pair_label(pair),
        })
    return rows


def _train_shared(
    reader,
    config: CohortV2MacroConfig,
    checkpoint_path: Path,
    log_every: int,
):
    seed_all(config.seed)
    trainer = CohortV2MacroTrainer(CohortV2MacroTrainingData(reader, config), config)
    results = _empty_losses()
    print(
        f"[shared train] steps={config.steps} batch={config.batch_size} "
        f"balanced_pairs={len(MACRO_PAIRS)} device={config.device}",
        flush=True,
    )
    for step in range(config.steps):
        result = trainer.train_step()
        results[result.pair].append(result)
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == config.steps:
            print(
                f"[shared train {step + 1}/{config.steps}] "
                f"pair={pair_label(result.pair)} total={result.total_loss:.6f} "
                f"carrier={result.carrier_loss:.6f} micro={result.micro_loss:.6f} "
                f"macro={result.macro_loss:.6f} lr={result.learning_rate:.2e}",
                flush=True,
            )
    checkpoint = save_cohort_v2_macro_checkpoint(checkpoint_path, trainer)
    print(f"[shared checkpoint] {checkpoint.path}", flush=True)
    return trainer, checkpoint, results


def _train_experts(
    reader,
    config: CohortV2MacroConfig,
    shared_steps: int,
    output: Path,
    log_every: int,
):
    checkpoints = []
    reports = _empty_losses()
    for index, pair in enumerate(MACRO_PAIRS, start=1):
        seed_all(config.seed)
        data = CohortV2PairExpertTrainingData(
            reader, config, pair, shared_step_count=shared_steps
        )
        trainer = CohortV2PairExpertTrainer(data, config)
        print(
            f"[expert {index}/{len(MACRO_PAIRS)}] {pair_label(pair)} "
            f"updates={config.steps} eligible_windows={len(data.pools[pair])}",
            flush=True,
        )
        for step in range(config.steps):
            result = trainer.train_step()
            reports[pair].append(result)
            if step == 0 or (step + 1) % log_every == 0 or step + 1 == config.steps:
                print(
                    f"[expert {index}/{len(MACRO_PAIRS)} train "
                    f"{step + 1}/{config.steps}] {pair_label(pair)} "
                    f"total={result.total_loss:.6f} carrier={result.carrier_loss:.6f} "
                    f"micro={result.micro_loss:.6f} macro={result.macro_loss:.6f} "
                    f"lr={result.learning_rate:.2e}",
                    flush=True,
                )
        checkpoint = save_cohort_v2_macro_checkpoint(
            output / "experts" / _expert_name(pair) / "checkpoint.pt", trainer
        )
        checkpoints.append((pair, checkpoint))
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return tuple(checkpoints), reports


def _load_shared_scorers(
    checkpoint_path: Path,
    reader,
    config: CohortV2MacroConfig,
    readers,
    devices: tuple[str, ...],
    progress_every: int,
):
    result = []
    for device in devices:
        predictor, codec, checkpoint = load_cohort_v2_macro_checkpoint(
            checkpoint_path, reader=reader, config=config, device=device
        )
        result.append(CohortV2MacroPairScorer(
            predictor,
            codec,
            checkpoint,
            config,
            readers,
            progress_every=progress_every,
            worker_name=f"shared:{device}",
        ))
    return tuple(result)


def _load_expert_scorers(
    output: Path,
    reader,
    config: CohortV2MacroConfig,
    readers,
    devices: tuple[str, ...],
    progress_every: int,
):
    composites = []
    for device in devices:
        routed = []
        for pair in MACRO_PAIRS:
            predictor, codec, checkpoint = load_cohort_v2_macro_checkpoint(
                output / "experts" / _expert_name(pair) / "checkpoint.pt",
                reader=reader,
                config=config,
                device=device,
            )
            routed.append((pair, CohortV2MacroPairScorer(
                predictor,
                codec,
                checkpoint,
                config,
                readers,
                progress_every=progress_every,
                worker_name=f"expert:{pair_label(pair)}:{device}",
            )))
        composites.append(CohortV2PairExpertScorer(tuple(routed)))
    return tuple(composites)


def _dry_run(readers, shared_config, expert_config) -> int:
    dry_shared = replace(shared_config, batch_size=min(2, shared_config.batch_size))
    dry_expert = replace(expert_config, batch_size=min(2, expert_config.batch_size))
    seed_all(dry_shared.seed)
    shared = CohortV2MacroTrainer(
        CohortV2MacroTrainingData(readers[0], dry_shared), dry_shared
    )
    for step in range(dry_shared.steps):
        result = shared.train_step()
        print(
            f"[dry-run shared {step + 1}/{dry_shared.steps}] "
            f"{pair_label(result.pair)} loss={result.total_loss:.6f}",
            flush=True,
        )

    routed = []
    for index, pair in enumerate(MACRO_PAIRS, start=1):
        seed_all(dry_expert.seed)
        data = CohortV2PairExpertTrainingData(
            readers[0], dry_expert, pair, shared_step_count=dry_shared.steps
        )
        trainer = CohortV2PairExpertTrainer(data, dry_expert)
        result = trainer.train_step()
        checkpoint = CohortV2MacroCheckpoint(Path("<dry-run>"), f"expert:{pair_label(pair)}", 1, ())
        scorer = CohortV2MacroPairScorer(
            trainer.predictor, trainer.codec, checkpoint, dry_expert, readers
        )
        probe = scorer.objective(data.pools[pair][0], pair)
        routed.append((pair, scorer))
        print(
            f"[dry-run expert {index}/{len(MACRO_PAIRS)}] {pair_label(pair)} "
            f"loss={result.total_loss:.6f} probe_objective={probe:.6f}",
            flush=True,
        )
    composite = CohortV2PairExpertScorer(tuple(routed))
    print(
        f"[dry-run routing] pairs={len(composite.scorers)} "
        f"checkpoint={composite.checkpoint_identity}",
        flush=True,
    )
    print(
        "[dry-run] exact shared minibatches, all target modes, and composite scoring passed; "
        "no files written and final evaluation remained sealed",
        flush=True,
    )
    return 0


def _validate_existing(
    output: Path,
    readers,
    shared_config: CohortV2MacroConfig,
    expert_config: CohortV2MacroConfig,
    devices: tuple[str, ...],
) -> int:
    shared_scorers = _load_shared_scorers(
        output / "shared" / "checkpoint.pt",
        readers[0], shared_config, readers, devices[:1], 0,
    )
    expert_scorers = _load_expert_scorers(
        output, readers[0], expert_config, readers, devices[:1], 0
    )
    for label, root, scorer in (
        ("shared", output / "shared" / "pair_evaluation", shared_scorers[0]),
        ("pair_experts", output / "pair_expert_evaluation", expert_scorers[0]),
    ):
        receipt = validate_cohort_v2_evaluation(
            root,
            readers=readers,
            checkpoint_identity=scorer.checkpoint_identity,
            checkpoint_capabilities=MACRO_CAPABILITIES,
            objective_identity=scorer.objective_identity,
        )
        print(
            f"[validate {label}] states={receipt.state_count} "
            f"available={receipt.available_count}",
            flush=True,
        )
    shared = load_cohort_v2_evaluation(
        output / "shared" / "pair_evaluation",
        readers=readers,
        checkpoint_identity=shared_scorers[0].checkpoint_identity,
        checkpoint_capabilities=MACRO_CAPABILITIES,
        objective_identity=shared_scorers[0].objective_identity,
    )
    experts = load_cohort_v2_evaluation(
        output / "pair_expert_evaluation",
        readers=readers,
        checkpoint_identity=expert_scorers[0].checkpoint_identity,
        checkpoint_capabilities=MACRO_CAPABILITIES,
        objective_identity=expert_scorers[0].objective_identity,
    )
    comparison = compare_preferred_pair_maps(shared, experts)
    summary = json.loads((output / "summary.json").read_bytes())
    if summary.get("preferred_pair_comparison") != comparison:
        raise CohortV2MacroError("stored preferred-pair comparison does not recompute")
    print(
        f"[validate] passed map_changes={comparison['changed_preferred_pair_count']}/"
        f"{comparison['state_count']}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.updates_per_pair <= 0
        or args.log_every <= 0
        or args.score_log_every <= 0
    ):
        raise CohortV2MacroError("updates and progress intervals must be positive")
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    output = (repository_root / args.output).resolve()
    readers = _readers(repository_root, release_root)
    shared_steps = args.updates_per_pair * len(MACRO_PAIRS)
    shared_config = _config(args, shared_steps)
    expert_config = _config(args, args.updates_per_pair)
    devices = _evaluation_devices(args)
    if args.dry_run:
        dry_shared = replace(shared_config, steps=len(MACRO_PAIRS))
        dry_expert = replace(expert_config, steps=1)
        return _dry_run(readers, dry_shared, dry_expert)
    if args.validate:
        return _validate_existing(output, readers, shared_config, expert_config, devices)

    shared_trainer, shared_checkpoint, shared_results = _train_shared(
        readers[0], shared_config, output / "shared" / "checkpoint.pt", args.log_every
    )
    del shared_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    expert_checkpoints, expert_results = _train_experts(
        readers[0], expert_config, shared_steps, output, args.log_every
    )

    shared_scorers = _load_shared_scorers(
        shared_checkpoint.path,
        readers[0], shared_config, readers, devices, args.score_log_every,
    )
    print(
        f"[shared score] devices={devices} batch_size={args.evaluation_batch_size}",
        flush=True,
    )
    shared_evaluation = CohortV2ParallelExhaustiveEvaluator(
        shared_scorers, batch_size=args.evaluation_batch_size
    ).evaluate(readers)
    shared_receipt = write_cohort_v2_evaluation(
        output / "shared" / "pair_evaluation", shared_evaluation, readers=readers
    )
    del shared_scorers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    expert_scorers = _load_expert_scorers(
        output, readers[0], expert_config, readers, devices, args.score_log_every
    )
    print(
        f"[pair-expert score] experts={len(MACRO_PAIRS)} devices={devices} "
        f"batch_size={args.evaluation_batch_size}",
        flush=True,
    )
    expert_evaluation = CohortV2ParallelExhaustiveEvaluator(
        expert_scorers, batch_size=args.evaluation_batch_size
    ).evaluate(readers)
    expert_receipt = write_cohort_v2_evaluation(
        output / "pair_expert_evaluation", expert_evaluation, readers=readers
    )
    comparison = compare_preferred_pair_maps(shared_evaluation, expert_evaluation)

    summary = {
        "artifact_type": "cohort_v2_shared_predictor_pair_expert_diagnostic",
        "availability_policy": "unavailable targets are masked, never converted to negatives",
        "capabilities": sorted(MACRO_CAPABILITIES),
        "capability_declaration_identity": readers[0].capability_declaration_identity,
        "expert_checkpoint_identities": [
            {"checkpoint_identity": checkpoint.identity, "pair": pair_label(pair)}
            for pair, checkpoint in expert_checkpoints
        ],
        "expert_training": _loss_report(expert_results),
        "final_evaluation_accessed": False,
        "implementation_commit": args.implementation_commit,
        "macro_event_endpoint_authority": MACRO_EVENT_ENDPOINT_AUTHORITY,
        "macro_state_authority": MACRO_STATE_AUTHORITY,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "objective": (
            "duration-weighted carrier MSE plus the selected mode's normalized "
            "contact/support or steady-state/structure-unstable readout loss"
        ),
        "pair_expert_evaluation_identity": expert_receipt.evaluation_identity,
        "pair_grid": [pair_label(pair) for pair in MACRO_PAIRS],
        "partition_identity": readers[0].partition_identity,
        "preferred_pair_comparison": comparison,
        "release_identity": readers[0].release_identity,
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_pair_experts --dry-run",
            "python -u -m scripts.run_cohort_v2_pair_experts",
            "python -u -m scripts.run_cohort_v2_pair_experts --validate",
        ],
        "role_influences": [
            {"exposure_role": role, "influence": influence}
            for role, influence in ROLE_INFLUENCE
        ],
        "schema": "cohort_v2_shared_predictor_pair_expert_diagnostic_v1",
        "shared_checkpoint_identity": shared_checkpoint.identity,
        "shared_evaluation_identity": shared_receipt.evaluation_identity,
        "shared_training": _loss_report(shared_results),
        "task_loss_normalization": {
            "carrier": "mean latent MSE per example",
            "macro": "mean of macro-state, effective-horizon, and event-class losses",
            "micro": "mean of availability-masked relation-set and predicate losses",
            "optimizer_updates_per_pair": args.updates_per_pair,
        },
        "target_vocabulary": {
            "macro": ["steady-state", "structure-unstable"],
            "micro": ["contact", "supports"],
        },
        "training_batch_parity": (
            "each expert receives the exact minibatches used for its pair in the "
            "shared model's balanced schedule and the learning rate at those global steps"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_bytes(canonical_json_bytes(summary))
    print(
        f"[complete] map_changes={comparison['changed_preferred_pair_count']}/"
        f"{comparison['state_count']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CohortV2MacroError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
