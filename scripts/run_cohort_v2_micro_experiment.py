"""Train and exhaustively score cohort-v2 oracle micro transitions (issue #5)."""
from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Final

import torch

from world_model.data import CohortV2OracleWindowDataset, CohortV2ReleaseReader
from world_model.model import Abstraction, DualOutputPredictor, PredictionPair
from world_model.training import (
    CohortV2ExhaustiveEvaluator,
    CohortV2ParallelExhaustiveEvaluator,
    validate_cohort_v2_evaluation,
    write_cohort_v2_evaluation,
)
from world_model.training.cohort_v2_micro import (
    MICRO_CAPABILITIES,
    MICRO_PAIRS,
    MICRO_RELATION_AUTHORITY,
    CohortV2MicroConfig,
    CohortV2MicroError,
    CohortV2MicroPairScorer,
    CohortV2MicroTrainer,
    CohortV2MicroTrainingData,
    load_cohort_v2_micro_checkpoint,
    save_cohort_v2_micro_checkpoint,
    validate_cohort_v2_micro_frontier_artifacts,
    validate_cohort_v2_micro_frontier_input,
    write_cohort_v2_micro_frontier_input,
)
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.loop import seed_all


DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-5-micro-experiment")
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
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--symbolic-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--score-log-every", type=int, default=250)
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
        frame_records = sum(len(rollout.frame_records) for rollout in reader.rollouts)
        print(
            f"[load {index}/3] {role}: rollouts={len(reader.rollouts)} "
            f"frame_records={frame_records}",
            flush=True,
        )
        readers.append(reader)
    return tuple(readers)


def _config(args: argparse.Namespace) -> CohortV2MicroConfig:
    return CohortV2MicroConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        symbolic_weight=args.symbolic_weight,
        device=args.device,
    )


def _evaluation_devices(args: argparse.Namespace) -> tuple[str, ...]:
    if args.evaluation_batch_size <= 0:
        raise CohortV2MicroError("evaluation batch size must be positive")
    if args.evaluation_devices == "auto":
        if torch.device(args.device).type == "cuda":
            count = torch.cuda.device_count()
            if count == 0:
                raise CohortV2MicroError("CUDA training requested but no GPU is visible")
            return tuple(f"cuda:{index}" for index in range(count))
        return (args.device,)
    devices = tuple(item.strip() for item in args.evaluation_devices.split(","))
    if not devices or any(not item for item in devices) or len(set(devices)) != len(devices):
        raise CohortV2MicroError("evaluation devices must be unique and nonempty")
    for device in devices:
        parsed = torch.device(device)
        if parsed.type == "cuda" and (
            parsed.index is None or parsed.index >= torch.cuda.device_count()
        ):
            raise CohortV2MicroError(f"evaluation device is unavailable: {device}")
    return devices


def _eligible_score_count(readers: tuple[CohortV2ReleaseReader, ...]) -> int:
    total = 0
    for reader in readers:
        dataset = CohortV2OracleWindowDataset(
            reader, requested_horizons=(1, 5, 15)
        )
        for window in dataset:
            total += 1  # continuous
            if all(
                window.context.labels[predicate].get("availability") == "available"
                and window.target.labels[predicate].get("availability") == "available"
                for predicate in ("contact", "supports")
            ):
                total += 1  # micro
    return total


def _dry_run(
    readers: tuple[CohortV2ReleaseReader, ...],
    config: CohortV2MicroConfig,
    evaluation_devices: tuple[str, ...],
    evaluation_batch_size: int,
) -> int:
    dry_config = replace(config, batch_size=min(2, config.batch_size))
    seed_all(dry_config.seed)
    data = CohortV2MicroTrainingData(readers[0], dry_config)
    print("[dry-run] strict training pools", flush=True)
    for pair in MICRO_PAIRS:
        print(
            f"[dry-run] h{pair.delta}/{pair.abstraction}: "
            f"eligible={len(data.pools[pair])}",
            flush=True,
        )
    trainer = CohortV2MicroTrainer(data, dry_config)
    seen = set()
    while seen != {Abstraction.CONTINUOUS, Abstraction.MICRO}:
        result = trainer.train_step()
        seen.add(result.pair.abstraction)
        print(
            f"[dry-run] in-memory step passed pair=h{result.pair.delta}/"
            f"{result.pair.abstraction} total={result.total_loss:.6f} "
            f"carrier={result.carrier_loss:.6f} micro={result.micro_loss:.6f} "
            f"queries={result.relation_query_count}",
            flush=True,
        )
    calibration = CohortV2OracleWindowDataset(
        readers[1], requested_horizons=(1,)
    )
    windows = tuple(
        item for item in calibration
        if all(
            item.context.labels[predicate].get("availability") == "available"
            and item.target.labels[predicate].get("availability") == "available"
            for predicate in ("contact", "supports")
        )
    )[:max(len(evaluation_devices) * 2, evaluation_batch_size)]
    # The dry run scores the in-memory model directly; no checkpoint or artifact
    # is written, and calibration never influences a learned parameter.
    from world_model.training.cohort_v2_micro import CohortV2MicroCheckpoint

    checkpoint = CohortV2MicroCheckpoint(
        Path("<dry-run>"), "checkpoint:dry-run", 1, ()
    )
    state = {
        name: value.detach().cpu()
        for name, value in trainer.predictor.state_dict().items()
    }
    scorers = []
    for device in evaluation_devices:
        predictor = DualOutputPredictor(dry_config.predictor_config)
        predictor.load_state_dict(state, strict=True)
        predictor.to(device).eval()
        scorers.append(CohortV2MicroPairScorer(
            predictor,
            trainer.codec,
            checkpoint,
            dry_config,
            readers,
            worker_name=device,
        ))
    shards = tuple(
        windows[index::len(scorers)] for index in range(len(scorers))
    )

    def score_probe(index: int) -> tuple[str, int, float, float]:
        shard = shards[index][:evaluation_batch_size]
        continuous = scorers[index].objective_batch(
            shard, PredictionPair(1, Abstraction.CONTINUOUS)
        )
        micro = scorers[index].objective_batch(
            shard, PredictionPair(1, Abstraction.MICRO)
        )
        return (
            evaluation_devices[index],
            len(shard),
            sum(continuous) / len(continuous),
            sum(micro) / len(micro),
        )

    with ThreadPoolExecutor(max_workers=len(scorers)) as executor:
        probes = tuple(executor.map(score_probe, range(len(scorers))))
    for device, count, continuous, micro in probes:
        print(
            f"[dry-run] {device} batched={count} continuous={continuous:.6f} "
            f"micro={micro:.6f}",
            flush=True,
        )
    print(
        f"[dry-run] deterministic round-robin shards={len(shards)} "
        f"batch_size={evaluation_batch_size}",
        flush=True,
    )
    print("[dry-run] no files written", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.log_every <= 0 or args.score_log_every <= 0:
        raise CohortV2MicroError("progress intervals must be positive")
    repository_root = args.repository_root.resolve()
    release_root = (repository_root / args.release_root).resolve()
    output = (repository_root / args.output).resolve()
    readers = _readers(repository_root, release_root)
    config = _config(args)
    evaluation_devices = _evaluation_devices(args)
    if args.dry_run:
        return _dry_run(
            readers, config, evaluation_devices, args.evaluation_batch_size
        )

    checkpoint_path = output / "checkpoint.pt"
    evaluation_root = output / "pair_evaluation"
    frontier_path = output / "frontier_input.json"
    if args.validate:
        predictor, codec, checkpoint = load_cohort_v2_micro_checkpoint(
            checkpoint_path, reader=readers[0], config=config, device=args.device
        )
        scorer = CohortV2MicroPairScorer(
            predictor, codec, checkpoint, config, readers
        )
        print(f"[validate] reading {evaluation_root}", flush=True)
        receipt = validate_cohort_v2_evaluation(
            evaluation_root,
            readers=readers,
            checkpoint_identity=checkpoint.identity,
            checkpoint_capabilities=MICRO_CAPABILITIES,
            objective_identity=scorer.objective_identity,
        )
        validate_cohort_v2_micro_frontier_artifacts(
            frontier_path, evaluation_root
        )
        print(
            f"[validate] passed states={receipt.state_count} "
            f"outcomes={receipt.outcome_count} available={receipt.available_count} "
            f"unavailable={receipt.unavailable_count}",
            flush=True,
        )
        return 0

    seed_all(config.seed)
    data = CohortV2MicroTrainingData(readers[0], config)
    print(
        f"[train] steps={config.steps} batch={config.batch_size} "
        f"device={config.device} balanced_pairs={len(MICRO_PAIRS)}",
        flush=True,
    )
    trainer = CohortV2MicroTrainer(data, config)
    first_loss = None
    latest = None
    for step in range(config.steps):
        latest = trainer.train_step()
        if first_loss is None:
            first_loss = latest.total_loss
        if (
            step == 0
            or (step + 1) % args.log_every == 0
            or step + 1 == config.steps
        ):
            print(
                f"[train {step + 1}/{config.steps}] "
                f"pair=h{latest.pair.delta}/{latest.pair.abstraction} "
                f"total={latest.total_loss:.6f} carrier={latest.carrier_loss:.6f} "
                f"micro={latest.micro_loss:.6f} queries={latest.relation_query_count} "
                f"lr={latest.learning_rate:.2e}",
                flush=True,
            )
    assert first_loss is not None and latest is not None
    checkpoint = save_cohort_v2_micro_checkpoint(checkpoint_path, trainer)
    print(
        f"[checkpoint] step={checkpoint.step} path={checkpoint.path}", flush=True
    )

    total_scores = _eligible_score_count(readers)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scorers = []
    for device in evaluation_devices:
        predictor, codec, loaded_checkpoint = load_cohort_v2_micro_checkpoint(
            checkpoint_path,
            reader=readers[0],
            config=config,
            device=device,
        )
        scorers.append(CohortV2MicroPairScorer(
            predictor,
            codec,
            loaded_checkpoint,
            config,
            readers,
            progress_every=args.score_log_every,
            worker_name=device,
        ))
    print(
        f"[score] exhaustive states across training/calibration/model_selection; "
        f"eligible_objectives={total_scores} devices={evaluation_devices} "
        f"batch_size={args.evaluation_batch_size}",
        flush=True,
    )
    evaluation = CohortV2ParallelExhaustiveEvaluator(
        tuple(scorers), batch_size=args.evaluation_batch_size
    ).evaluate(readers)
    receipt = write_cohort_v2_evaluation(
        evaluation_root, evaluation, readers=readers
    )
    write_cohort_v2_micro_frontier_input(frontier_path, evaluation)
    validate_cohort_v2_micro_frontier_input(frontier_path, evaluation)
    summary = {
        "available_count": receipt.available_count,
        "checkpoint_identity": checkpoint.identity,
        "final_loss": latest.total_loss,
        "first_loss": first_loss,
        "frontier_input": "frontier_input.json",
        "micro_capabilities": list(sorted(MICRO_CAPABILITIES)),
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "evaluation_batch_size": args.evaluation_batch_size,
        "evaluation_devices": list(evaluation_devices),
        "objective_identity": scorers[0].objective_identity,
        "pair_evaluation": "pair_evaluation",
        "release_identity": receipt.release_identity,
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_micro_experiment --dry-run",
            "python -u -m scripts.run_cohort_v2_micro_experiment",
            "python -u -m scripts.run_cohort_v2_micro_experiment --validate",
        ],
        "schema": "cohort_v2_micro_experiment_summary_v1",
        "state_count": receipt.state_count,
        "step": checkpoint.step,
        "unavailable_count": receipt.unavailable_count,
    }
    (output / "summary.json").write_bytes(canonical_json_bytes(summary))
    print(
        f"[complete] states={receipt.state_count} outcomes={receipt.outcome_count} "
        f"available={receipt.available_count} unavailable={receipt.unavailable_count} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CohortV2MicroError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
