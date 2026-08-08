#!/usr/bin/env python3
"""Run the Phase-A temporal pair-grid train, score, frontier, or all flow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.model import JepaBackbone, JepaConfig
from world_model.training import (
    GridRunError,
    PhaseAConfig,
    fixture_batch,
    fixture_jepa_config,
    load_checkpoint,
    save_checkpoint,
    score_checkpoint,
    seed_all,
    TeacherForcedTrainer,
    write_sweep_manifest,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "frontier", "all"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixture", action="store_true")
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


def _train(args: argparse.Namespace, config: PhaseAConfig, model_config: JepaConfig) -> tuple[Path, object]:
    output = args.output_dir
    checkpoint = args.checkpoint or output / "checkpoint.pt"
    seed_all(config.seed)
    trainer = TeacherForcedTrainer(
        JepaBackbone(model_config), config.training_config(device="cpu" if args.device == "cpu" else args.device)
    )
    if args.resume:
        loaded = load_checkpoint(checkpoint, trainer, config_digest=config.identity, grid_digest=config.grid_digest)
        start = loaded.step
    else:
        start = 0
    for step in range(start, config.steps):
        batch = fixture_batch(model_config, seed=config.seed, batch_size=config.batch_size, step=step)
        trainer.train_step(batch)
    info = save_checkpoint(checkpoint, trainer, config_digest=config.identity, grid_digest=config.grid_digest)
    return checkpoint, info


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = _config(args)
    model_config = _model_config(args)
    checkpoint = args.checkpoint or args.output_dir / "checkpoint.pt"
    expected_digest = None
    manifest_path = args.output_dir / "sweep_manifest.json"
    if args.command in ("score", "frontier") and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_digest = manifest.get("checkpoint_digest")
    if args.command in ("train", "all"):
        checkpoint, checkpoint_info = _train(args, config, model_config)
    else:
        checkpoint_info = None
    if args.command in ("score", "frontier", "all"):
        if expected_digest is not None:
            from world_model.training import checkpoint_digest

            if checkpoint_digest(checkpoint) != expected_digest:
                raise GridRunError("checkpoint digest mismatch")
        batches = tuple(
            fixture_batch(model_config, seed=config.seed, batch_size=min(config.batch_size, 8), step=step)
            for step in range(9)
        )
        result = score_checkpoint(checkpoint, phase_config=config, model_config=model_config, batches=batches)
        manifest_checkpoint = checkpoint_info
        if manifest_checkpoint is None:
            manifest_checkpoint = load_checkpoint(
                checkpoint,
                TeacherForcedTrainer(JepaBackbone(model_config), config.training_config(device="cpu")),
                config_digest=config.identity,
                grid_digest=config.grid_digest,
            )
        write_sweep_manifest(args.output_dir / "sweep_manifest.json", checkpoint=manifest_checkpoint, phase_config=config, score=result)
        (args.output_dir / "score.json").write_text(json.dumps({"step": result.step, "count": result.count, "mean_loss": result.mean_loss}, sort_keys=True) + "\n", encoding="utf-8")
        if args.command == "frontier":
            (args.output_dir / "frontier.json").write_text(json.dumps({"verdict": "inconclusive", "reason": "fixture score only"}, sort_keys=True) + "\n", encoding="utf-8")
        print(f"score step={result.step} count={result.count} mean_loss={result.mean_loss:.8f}")
    elif checkpoint_info is not None:
        print(f"train step={checkpoint_info.step} checkpoint={checkpoint_info.path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GridRunError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
