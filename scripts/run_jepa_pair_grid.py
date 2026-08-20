#!/usr/bin/env python3
"""Run the Phase-A temporal pair-grid train, score, frontier, or all flow."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.model import JepaBackbone, JepaConfig
from world_model.training import (
    CheckpointInfo,
    FIXTURE_CATALOG_IDENTITY,
    GridRunError,
    PhaseAConfig,
    RealPhaseData,
    fixture_batch,
    fixture_jepa_config,
    fixture_partition_identity,
    fixture_state_set_identity,
    load_checkpoint,
    save_checkpoint,
    score_fixture_checkpoint,
    score_real_checkpoint,
    seed_all,
    TeacherForcedTrainer,
    validate_score_artifacts,
    write_score_artifacts,
    write_frontier_input,
    write_real_sweep_manifest,
)

DEFAULT_DATASET_ROOT = Path(
    "/mnt/array/sukaih/Project/NovPhy/data/novphy_rollouts_dataset_20260708_171531"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "validate", "frontier", "all"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/m1ef-phase-a"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--delta", type=int, default=None)
    parser.add_argument("--abstraction", default=None)
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> PhaseAConfig:
    if args.delta not in (None, 1, 5, 15) or args.abstraction not in (None, "continuous"):
        raise GridRunError("Phase-A evaluates only approved continuous pairs")
    return PhaseAConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        split=args.split,
        device=args.device,
    )


def _model_config(args: argparse.Namespace) -> JepaConfig:
    if args.fixture:
        return fixture_jepa_config()
    from world_model.model import EncoderConfig, JepaConfig, PredictorConfig

    return JepaConfig(encoder=EncoderConfig(), predictor=PredictorConfig())


def _validate_output_root(args: argparse.Namespace) -> None:
    if args.fixture:
        return
    dataset = args.dataset_root.resolve()
    outputs = [args.output_dir]
    if args.command in ("train", "all") and args.checkpoint is not None:
        outputs.append(args.checkpoint)
    if any(output.resolve().is_relative_to(dataset) for output in outputs):
        raise GridRunError("run output cannot be written inside the protected dataset")


def _train(
    args: argparse.Namespace,
    config: PhaseAConfig,
    model_config: JepaConfig,
    real_data: RealPhaseData | None = None,
) -> tuple[Path, CheckpointInfo]:
    output = args.output_dir
    checkpoint = args.checkpoint or output / "checkpoint.pt"
    seed_all(config.seed, reproducibility=config.reproducibility)
    trainer = TeacherForcedTrainer(
        JepaBackbone(model_config), config.training_config(device="cpu" if args.device == "cpu" else args.device)
    )
    if args.resume:
        loaded = load_checkpoint(
            checkpoint,
            trainer,
            config_identity=config.identity,
            grid_identity=config.grid_identity,
            expected_catalog_identity=None if real_data is None else real_data.catalog_identity,
            expected_run_identity=None if real_data is None else real_data.run_identity,
        )
        start = loaded.step
        counts = dict(loaded.key_counts)
    else:
        start = 0
        counts = {}
    for step in range(start, config.steps):
        if real_data is None:
            batch = fixture_batch(
                model_config, seed=config.seed, batch_size=config.batch_size, step=step
            )
        else:
            pair, regime = trainer.schedule_at(step)
            batch = real_data.training_batch(pair, regime, config.batch_size, step)
            key = f"delta={pair.delta},regime={regime.value}"
            counts[key] = counts.get(key, 0) + 1
        trainer.train_step(batch)
    if real_data is not None and sum(counts.values()) != config.steps:
        raise GridRunError("checkpoint key counts do not match the completed schedule")
    info = save_checkpoint(
        checkpoint,
        trainer,
        config_identity=config.identity,
        grid_identity=config.grid_identity,
        catalog_identity=None if real_data is None else real_data.catalog_identity,
        run_identity=None if real_data is None else real_data.run_identity,
        key_counts=tuple(sorted(counts.items())),
    )
    return checkpoint, info


def _run_real_frontier(args: argparse.Namespace) -> None:
    score_root = args.output_dir / "score_artifacts"
    validate_score_artifacts(score_root)
    source = args.output_dir / "frontier_input.json"
    write_frontier_input(score_root, source)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_jepa_pair_frontier.py"),
            "--input",
            str(source),
            "--output-dir",
            str(args.output_dir / "frontier"),
            "--seed",
            str(args.seed),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise GridRunError(f"frontier generation failed: {completed.stderr.strip()}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _validate_output_root(args)
    if args.command == "validate":
        receipt = validate_score_artifacts(args.output_dir / "score_artifacts")
        print(
            f"validated states={receipt.state_count} scores={receipt.score_count}",
            flush=True,
        )
        return 0
    config = _config(args)
    model_config = _model_config(args)
    checkpoint = args.checkpoint or args.output_dir / "checkpoint.pt"
    real_data = None
    if not args.fixture and args.command in ("train", "score", "all"):
        real_data = RealPhaseData.build(args.dataset_root, config, model_config)
        print(
            f"catalog dev: {len(real_data.catalog.episodes)} accepted, "
            f"{real_data.catalog.rejection_count} rejected identity={real_data.catalog_identity}",
            flush=True,
        )
    if args.command in ("train", "all"):
        checkpoint, checkpoint_info = _train(args, config, model_config, real_data)
    else:
        checkpoint_info = None
    if args.command in ("score", "frontier", "all"):
        if args.fixture:
            for partition in ("controller-train", "calibration", "evaluation"):
                print(f"scoring partition={partition} states={config.steps}", flush=True)
            exhaustive, scored_checkpoint = score_fixture_checkpoint(
                checkpoint, config, model_config
            )
            receipt = write_score_artifacts(
                args.output_dir / "score_artifacts",
                exhaustive,
                checkpoint_path=scored_checkpoint.path,
                checkpoint_identity=scored_checkpoint.identity,
                config_identity=scored_checkpoint.config_identity,
                catalog_identity=FIXTURE_CATALOG_IDENTITY,
                partition_identity=fixture_partition_identity(config),
                state_set_identity=fixture_state_set_identity(config),
                resume=args.resume,
            )
            validate_score_artifacts(
                args.output_dir / "score_artifacts",
                expected_checkpoint_identity=scored_checkpoint.identity,
                expected_catalog_identity=FIXTURE_CATALOG_IDENTITY,
                expected_partition_identity=fixture_partition_identity(config),
                expected_state_set_identity=fixture_state_set_identity(config),
            )
            (args.output_dir / "score.json").write_text(
                json.dumps(
                    {
                        "error_scale": exhaustive.score_spec.error_scale,
                        "score_count": receipt.score_count,
                        "state_count": receipt.state_count,
                    },
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            print(
                f"score states={receipt.state_count} scores={receipt.score_count} "
                f"error_scale={exhaustive.score_spec.error_scale:.8f}",
                flush=True,
            )
            return 0
        if args.command == "frontier":
            _run_real_frontier(args)
            print(f"frontier source={args.output_dir / 'frontier_input.json'}")
            return 0
        if real_data is None:
            raise GridRunError("real scoring requires the dev catalog")
        exhaustive, scored_checkpoint = score_real_checkpoint(
            checkpoint, config, model_config, real_data
        )
        receipt = write_score_artifacts(
            args.output_dir / "score_artifacts",
            exhaustive,
            checkpoint_path=scored_checkpoint.path,
            checkpoint_identity=scored_checkpoint.identity,
            config_identity=scored_checkpoint.config_identity,
            catalog_identity=real_data.catalog_identity,
            partition_identity=real_data.partition_identity,
            state_set_identity=real_data.state_set_identity,
            resume=args.resume,
        )
        validated = validate_score_artifacts(
            args.output_dir / "score_artifacts",
            expected_checkpoint_identity=scored_checkpoint.identity,
            expected_catalog_identity=real_data.catalog_identity,
            expected_partition_identity=real_data.partition_identity,
            expected_state_set_identity=real_data.state_set_identity,
        )
        write_real_sweep_manifest(
            args.output_dir / "sweep_manifest.json",
            data=real_data,
            phase_config=config,
            checkpoint=scored_checkpoint,
            score=validated,
        )
        (args.output_dir / "score.json").write_text(
            json.dumps(
                {
                    "error_scale": exhaustive.score_spec.error_scale,
                    "score_count": receipt.score_count,
                    "state_count": receipt.state_count,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        print(
            f"score states={receipt.state_count} scores={receipt.score_count} "
            f"error_scale={exhaustive.score_spec.error_scale:.8f}",
            flush=True,
        )
        if args.command == "all":
            _run_real_frontier(args)
        return 0
    elif checkpoint_info is not None:
        print(f"train step={checkpoint_info.step} checkpoint={checkpoint_info.path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GridRunError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
