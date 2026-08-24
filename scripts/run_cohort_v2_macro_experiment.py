"""Train and exhaustively score cohort-v2 oracle macro-event transitions (issue #6)."""
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
    CohortV2ParallelExhaustiveEvaluator,
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
    macro_event_endpoint_available,
    save_cohort_v2_macro_checkpoint,
    validate_cohort_v2_macro_frontier_artifacts,
    validate_cohort_v2_macro_frontier_input,
    write_cohort_v2_macro_frontier_input,
)
from world_model.training.cohort_v2_micro import MICRO_RELATION_AUTHORITY
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.loop import seed_all


DEFAULT_RELEASE: Final = Path("data/runtime_evidence/issue-53-mixed-termination-v5")
DEFAULT_OUTPUT: Final = Path(".local-artifacts/issue-6-macro-experiment")
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
    parser.add_argument("--steps", type=int, default=1800)
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
        terminal_counts: dict[str, int] = {}
        for rollout in reader.rollouts:
            reason = rollout.frame_records[-1].terminal["reason"]
            terminal_counts[reason] = terminal_counts.get(reason, 0) + 1
        print(
            f"[load {index}/3] {role}: rollouts={len(reader.rollouts)} "
            f"frame_records={frame_records} terminals={terminal_counts}",
            flush=True,
        )
        readers.append(reader)
    return tuple(readers)


def _config(args: argparse.Namespace) -> CohortV2MacroConfig:
    return CohortV2MacroConfig(
        seed=args.seed,
        steps=args.steps,
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
                raise CohortV2MacroError("CUDA training requested but no GPU is visible")
            return tuple(f"cuda:{index}" for index in range(count))
        return (args.device,)
    devices = tuple(item.strip() for item in args.evaluation_devices.split(","))
    if not devices or any(not item for item in devices) or len(set(devices)) != len(devices):
        raise CohortV2MacroError("evaluation devices must be unique and nonempty")
    for device in devices:
        parsed = torch.device(device)
        if parsed.type == "cuda" and (
            parsed.index is None or parsed.index >= torch.cuda.device_count()
        ):
            raise CohortV2MacroError(f"evaluation device is unavailable: {device}")
    return devices


def _micro_available(window) -> bool:
    return all(
        window.context.labels[predicate].get("availability") == "available"
        and window.target.labels[predicate].get("availability") == "available"
        for predicate in ("contact", "supports")
    )


def _eligible_score_count(readers: tuple[CohortV2ReleaseReader, ...]) -> int:
    total = 0
    for reader in readers:
        for window in CohortV2OracleWindowDataset(
            reader, requested_horizons=(1, 5, 15)
        ):
            total += 1
            if _micro_available(window):
                total += 1
            if macro_event_endpoint_available(window):
                total += 1
    return total


def _dry_run(
    readers: tuple[CohortV2ReleaseReader, ...],
    config: CohortV2MacroConfig,
    evaluation_devices: tuple[str, ...],
    evaluation_batch_size: int,
) -> int:
    dry_config = replace(config, batch_size=min(2, config.batch_size))
    seed_all(dry_config.seed)
    data = CohortV2MacroTrainingData(readers[0], dry_config)
    print("[dry-run] full-grid training pools", flush=True)
    for pair in MACRO_PAIRS:
        print(
            f"[dry-run] h{pair.delta}/{pair.abstraction}: "
            f"eligible={len(data.pools[pair])}",
            flush=True,
        )
    trainer = CohortV2MacroTrainer(data, dry_config)
    seen = set()
    while seen != set(Abstraction):
        result = trainer.train_step()
        seen.add(result.pair.abstraction)
        print(
            f"[dry-run] in-memory step passed pair=h{result.pair.delta}/"
            f"{result.pair.abstraction} total={result.total_loss:.6f} "
            f"carrier={result.carrier_loss:.6f} micro={result.micro_loss:.6f} "
            f"macro={result.macro_loss:.6f} endpoints={result.endpoint_count}",
            flush=True,
        )

    calibration = tuple(
        window
        for window in CohortV2OracleWindowDataset(
            readers[1], requested_horizons=(15,)
        )
        if macro_event_endpoint_available(window) and _micro_available(window)
    )
    needed = max(len(evaluation_devices) * 2, evaluation_batch_size)
    windows = calibration[:needed]
    if len(windows) < len(evaluation_devices):
        raise CohortV2MacroError("dry-run has too few eligible endpoint windows")
    checkpoint = CohortV2MacroCheckpoint(
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
        scorers.append(CohortV2MacroPairScorer(
            predictor,
            trainer.codec,
            checkpoint,
            dry_config,
            readers,
            worker_name=device,
        ))
    shards = tuple(windows[index::len(scorers)] for index in range(len(scorers)))

    def score_probe(index: int) -> tuple[str, int, tuple[float, ...]]:
        shard = shards[index][:evaluation_batch_size]
        means = []
        for abstraction in Abstraction:
            values = scorers[index].objective_batch(
                shard, PredictionPair(15, abstraction)
            )
            means.append(sum(values) / len(values))
        return evaluation_devices[index], len(shard), tuple(means)

    with ThreadPoolExecutor(max_workers=len(scorers)) as executor:
        probes = tuple(executor.map(score_probe, range(len(scorers))))
    for device, count, means in probes:
        print(
            f"[dry-run] {device} batched={count} "
            f"continuous={means[0]:.6f} micro={means[1]:.6f} macro={means[2]:.6f}",
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
        raise CohortV2MacroError("progress intervals must be positive")
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
        predictor, codec, checkpoint = load_cohort_v2_macro_checkpoint(
            checkpoint_path, reader=readers[0], config=config, device=args.device
        )
        scorer = CohortV2MacroPairScorer(
            predictor, codec, checkpoint, config, readers
        )
        print(f"[validate] reading {evaluation_root}", flush=True)
        receipt = validate_cohort_v2_evaluation(
            evaluation_root,
            readers=readers,
            checkpoint_identity=checkpoint.identity,
            checkpoint_capabilities=MACRO_CAPABILITIES,
            objective_identity=scorer.objective_identity,
        )
        validate_cohort_v2_macro_frontier_artifacts(frontier_path, evaluation_root)
        print(
            f"[validate] passed states={receipt.state_count} "
            f"outcomes={receipt.outcome_count} available={receipt.available_count} "
            f"unavailable={receipt.unavailable_count}",
            flush=True,
        )
        return 0

    seed_all(config.seed)
    data = CohortV2MacroTrainingData(readers[0], config)
    print(
        f"[train] steps={config.steps} batch={config.batch_size} "
        f"device={config.device} balanced_pairs={len(MACRO_PAIRS)}",
        flush=True,
    )
    trainer = CohortV2MacroTrainer(data, config)
    first_loss = None
    latest = None
    for step in range(config.steps):
        latest = trainer.train_step()
        if first_loss is None:
            first_loss = latest.total_loss
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == config.steps:
            print(
                f"[train {step + 1}/{config.steps}] "
                f"pair=h{latest.pair.delta}/{latest.pair.abstraction} "
                f"total={latest.total_loss:.6f} carrier={latest.carrier_loss:.6f} "
                f"micro={latest.micro_loss:.6f} macro={latest.macro_loss:.6f} "
                f"endpoints={latest.endpoint_count} lr={latest.learning_rate:.2e}",
                flush=True,
            )
    assert first_loss is not None and latest is not None
    checkpoint = save_cohort_v2_macro_checkpoint(checkpoint_path, trainer)
    print(f"[checkpoint] step={checkpoint.step} path={checkpoint.path}", flush=True)

    total_scores = _eligible_score_count(readers)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scorers = []
    for device in evaluation_devices:
        predictor, codec, loaded = load_cohort_v2_macro_checkpoint(
            checkpoint_path, reader=readers[0], config=config, device=device
        )
        scorers.append(CohortV2MacroPairScorer(
            predictor,
            codec,
            loaded,
            config,
            readers,
            progress_every=args.score_log_every,
            worker_name=device,
        ))
    print(
        "[score] exhaustive states across training/calibration/model_selection; "
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
    write_cohort_v2_macro_frontier_input(frontier_path, evaluation)
    validate_cohort_v2_macro_frontier_input(frontier_path, evaluation)
    summary = {
        "available_count": receipt.available_count,
        "checkpoint_identity": checkpoint.identity,
        "evaluation_batch_size": args.evaluation_batch_size,
        "evaluation_devices": list(evaluation_devices),
        "final_loss": latest.total_loss,
        "first_loss": first_loss,
        "frontier_input": "frontier_input.json",
        "macro_capabilities": list(sorted(MACRO_CAPABILITIES)),
        "macro_event_endpoint_authority": MACRO_EVENT_ENDPOINT_AUTHORITY,
        "macro_state_authority": MACRO_STATE_AUTHORITY,
        "micro_relation_authority": MICRO_RELATION_AUTHORITY,
        "objective_identity": scorers[0].objective_identity,
        "pair_evaluation": "pair_evaluation",
        "release_identity": receipt.release_identity,
        "rerun_commands": [
            "python -u -m scripts.run_cohort_v2_macro_experiment --dry-run",
            "python -u -m scripts.run_cohort_v2_macro_experiment",
            "python -u -m scripts.run_cohort_v2_macro_experiment --validate",
        ],
        "schema": "cohort_v2_macro_experiment_summary_v1",
        "state_count": receipt.state_count,
        "step": checkpoint.step,
        "unavailable_count": receipt.unavailable_count,
    }
    output.mkdir(parents=True, exist_ok=True)
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
    except CohortV2MacroError as error:
        print(f"error: {error}", flush=True)
        raise SystemExit(2) from error
