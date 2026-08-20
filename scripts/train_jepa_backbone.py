#!/usr/bin/env python3
"""Train the Milestone 1a/1b JEPA backbone against the legacy RGB cohort.

Two modes:

- ``--mode overfit``: overfit a fixed subset of ``--window-count`` windows and
  record the anti-collapse evidence (loss ~0 AND representation spread, rank,
  and retrieval) in ``--output-dir/<run-id>/manifest.json``.
- ``--mode train``: a short teacher-forced smoke over a seeded sample of dev
  windows.  This is a stability smoke, not a training claim.

Ground rules: the dataset root is read-only; checkpoints and run artifacts are
written under ``--output-dir`` (default ``runs/``), which stays out of git.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.data import LEGACY_RGB_V1, EpisodeCatalog  # noqa: E402
from world_model.model import (  # noqa: E402
    Abstraction,
    EncoderConfig,
    JepaConfig,
    PredictorConfig,
)
from world_model.training import (  # noqa: E402
    TeacherForcedTrainer,
    TrainingConfig,
    build_window_loader,
    run_overfit,
)

DEFAULT_DATASET_ROOT = (
    Path("/mnt/array/sukaih/Project/NovPhy/data/novphy_rollouts_dataset_20260708_171531")
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overfit", "train"), default="overfit")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="read-only rollout dataset root (legacy RGB cohort)",
    )
    parser.add_argument("--split", default="dev", choices=("dev", "train", "test"))
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--delta", type=int, default=1)
    parser.add_argument("--abstraction", default="continuous", choices=tuple(Abstraction))
    parser.add_argument("--window-count", type=int, default=8)
    parser.add_argument(
        "--window-selection",
        default="diverse",
        choices=("diverse", "motion", "uniform"),
        help=(
            "how the overfit subset is drawn. 'diverse' takes one window per "
            "episode: uniformly drawn dev windows are frequently near-duplicates "
            "whose target embeddings land 7.4e-05 apart, which makes retrieval a "
            "coin flip regardless of predictor quality"
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help=(
            "LR warmup steps. Defaults to 0 for --mode overfit and 100 for "
            "--mode train: on an 8-window overfit a warmup measurably costs "
            "retrieval (1.000 -> 0.250 at 1500 steps), because the cosine decay "
            "is measured against the full run, so the LR is already decaying by "
            "the time warmup ends"
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-base-momentum", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args(argv)


def _build_jepa_config(args: argparse.Namespace) -> JepaConfig:
    encoder = EncoderConfig()
    predictor = PredictorConfig()
    return JepaConfig(encoder=encoder, predictor=predictor)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    jepa_config = _build_jepa_config(args)
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = 0 if args.mode == "overfit" else 100
    training_config = TrainingConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        grad_clip=args.grad_clip,
        ema_base_momentum=args.ema_base_momentum,
        delta=args.delta,
        abstraction=args.abstraction,
        device=args.device,
    )
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {args.dataset_root}")
    catalog = EpisodeCatalog.build(
        root=args.dataset_root,
        split=args.split,
        capture_contract=LEGACY_RGB_V1,
    )
    print(
        f"catalog {args.split}: {len(catalog.episodes)} accepted, "
        f"{catalog.rejection_count} rejected "
        f"({catalog.rejection_code_counts})"
    )

    if args.mode == "overfit":
        report = run_overfit(
            catalog,
            jepa_config=jepa_config,
            training_config=training_config,
            window_count=args.window_count,
            output_dir=args.output_dir,
            window_selection=args.window_selection,
            candidate_count=args.candidate_count,
        )
        print(f"overfit run: {report.run_dir}")
        print(f"  initial_loss      {report.initial_loss:.6f}")
        print(f"  final_loss        {report.final_loss:.6f}")
        print(f"  mean_feature_std  {report.diagnostics.mean_feature_std:.6f}")
        print(f"  effective_rank    {report.diagnostics.effective_rank:.3f}")
        print(f"  retrieval_acc     {report.diagnostics.retrieval_accuracy:.3f}")
        print(f"  acceptance        {report.acceptance}")
        manifest = report.run_dir / "manifest.json"
        print(f"  manifest          {manifest}")
        return 0 if report.acceptance == "pass" else 2

    # Train smoke: a short deterministic pass over sampled dev windows.
    loader, window_count, index_identity = build_window_loader(
        catalog,
        encoder_config=jepa_config.encoder,
        delta=training_config.delta,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        draw_count=training_config.batch_size * training_config.steps,
    )
    print(f"windows {window_count} sampled {len(loader.sampler)} identity {index_identity}")
    from world_model.model import JepaBackbone
    from world_model.training import seed_all

    seed_all(training_config.seed)
    backbone = JepaBackbone(jepa_config)
    trainer = TeacherForcedTrainer(backbone, training_config)
    iterator = iter(loader)
    first_loss: float | None = None
    last_loss: float | None = None
    for step in range(training_config.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        result = trainer.train_step(batch)
        if first_loss is None:
            first_loss = result.loss
        last_loss = result.loss
        if step % max(1, training_config.steps // 10) == 0:
            print(f"  step {step:5d} loss {result.loss:.6f} lr {result.learning_rate:.2e}")
    assert first_loss is not None and last_loss is not None
    print(f"train smoke: {first_loss:.6f} -> {last_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
