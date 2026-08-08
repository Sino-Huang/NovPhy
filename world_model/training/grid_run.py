from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import torch

from world_model.model import (
    Abstraction,
    EncoderConfig,
    JepaBackbone,
    JepaConfig,
    PredictionPair,
    PredictorConfig,
    digest,
)
from world_model.training.grid_artifacts import ALPHA_EXCLUSIONS, APPROVED_DELTAS
from world_model.training.grid_data import MotionRegime
from world_model.training.loop import TeacherForcedTrainer, TrainingConfig, seed_all

CHECKPOINT_VERSION: Final[str] = "jepa-pair-grid-checkpoint-v1"
GRID_VERSION: Final[str] = "pair-grid-v1"


class GridRunError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class PhaseAConfig:
    seed: int = 20260807
    steps: int = 3600
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 0
    grad_clip: float = 1.0
    ema_base_momentum: float = 0.996
    split: str = "dev"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise GridRunError("seed must be a nonnegative integer")
        for name in ("steps", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise GridRunError(f"{name} must be positive")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise GridRunError("warmup_steps must be within the run")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise GridRunError("invalid optimizer configuration")
        if not 0 <= self.ema_base_momentum <= 1 or self.split != "dev":
            raise GridRunError("invalid EMA or split configuration")
        if not self.device.strip():
            raise GridRunError("device must be nonempty")

    @property
    def grid_digest(self) -> str:
        return digest((GRID_VERSION, APPROVED_DELTAS, ("continuous",), ALPHA_EXCLUSIONS))

    @property
    def identity(self) -> str:
        return digest(
            (
                "phase-a-config-v1",
                self.seed,
                self.steps,
                self.batch_size,
                self.learning_rate,
                self.weight_decay,
                self.warmup_steps,
                self.grad_clip,
                self.ema_base_momentum,
                self.split,
                self.device,
                self.grid_digest,
            )
        )

    def training_config(self, *, device: str | None = None) -> TrainingConfig:
        return TrainingConfig(
            seed=self.seed,
            steps=self.steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            grad_clip=self.grad_clip,
            ema_base_momentum=self.ema_base_momentum,
            delta=1,
            abstraction=Abstraction.CONTINUOUS,
            device=device or self.device,
            grid_schedule=True,
        )


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    digest: str
    step: int
    config_digest: str
    catalog_digest: str | None = None
    run_identity: str | None = None
    key_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ScoreResult:
    step: int
    count: int
    mean_loss: float
    config_digest: str


def fixture_jepa_config() -> JepaConfig:
    encoder = EncoderConfig(
        input_height=16,
        input_width=16,
        stem_channels=8,
        stage_channels=(8,),
        blocks_per_stage=1,
        group_norm_groups=4,
        latent_dim=8,
        pool_heads=2,
    )
    predictor = PredictorConfig(
        latent_dim=8, action_dim=5, hidden_dim=16, depth=1, pair_code_dim=4
    )
    return JepaConfig(encoder=encoder, predictor=predictor)


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checkpoint_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(
    path: Path,
    trainer: TeacherForcedTrainer,
    *,
    config_digest: str,
    grid_digest: str,
    catalog_digest: str | None = None,
    run_identity: str | None = None,
    key_counts: tuple[tuple[str, int], ...] = (),
) -> CheckpointInfo:
    if trainer.config.grid_schedule is not True:
        raise GridRunError("Phase-A checkpoints require grid_schedule=True")
    if (catalog_digest is None) != (run_identity is None):
        raise GridRunError("catalog digest and run identity must be recorded together")
    for name, value in (("catalog_digest", catalog_digest), ("run_identity", run_identity)):
        if value is not None and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
            raise GridRunError(f"{name} must be a lowercase SHA-256 digest")
    if any(type(key) is not str or not key or type(count) is not int or count < 0 for key, count in key_counts):
        raise GridRunError("key counts must contain named nonnegative counts")
    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "config_digest": config_digest,
        "grid_digest": grid_digest,
        "catalog_digest": catalog_digest,
        "run_identity": run_identity,
        "key_counts": key_counts,
        "step": trainer._step_count,
        "model_config_digest": trainer.backbone.config.identity,
        "online_encoder": trainer.backbone.encoder.state_dict(),
        "ema_target": trainer.backbone.target.state_dict(),
        "predictor": trainer.backbone.predictor.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    _atomic_torch_save(payload, path)
    actual_digest = checkpoint_digest(path)
    digest_path = path.with_name(path.name + ".sha256")
    temporary = digest_path.with_name(digest_path.name + ".tmp")
    temporary.write_text(actual_digest + "\n", encoding="ascii")
    os.replace(temporary, digest_path)
    return CheckpointInfo(
        path,
        actual_digest,
        trainer._step_count,
        config_digest,
        catalog_digest,
        run_identity,
        key_counts,
    )


def load_checkpoint(
    path: Path,
    trainer: TeacherForcedTrainer,
    *,
    config_digest: str,
    grid_digest: str,
    expected_digest: str | None = None,
    expected_catalog_digest: str | None = None,
    expected_run_identity: str | None = None,
) -> CheckpointInfo:
    if not path.is_file() or path.name.endswith(".tmp"):
        raise GridRunError("checkpoint is missing or partial")
    actual_digest = checkpoint_digest(path)
    digest_path = path.with_name(path.name + ".sha256")
    if digest_path.is_file() and digest_path.read_text(encoding="ascii").strip() != actual_digest:
        raise GridRunError("checkpoint digest mismatch")
    if expected_digest is not None and actual_digest != expected_digest:
        raise GridRunError("checkpoint digest mismatch")
    try:
        payload = torch.load(path, map_location=trainer.device, weights_only=False)
        if payload.get("version") != CHECKPOINT_VERSION:
            raise GridRunError("unsupported checkpoint version")
        if payload.get("config_digest") != config_digest or payload.get("grid_digest") != grid_digest:
            raise GridRunError("checkpoint config or grid digest mismatch")
        catalog_digest = payload.get("catalog_digest")
        run_identity = payload.get("run_identity")
        if expected_catalog_digest is not None and catalog_digest != expected_catalog_digest:
            raise GridRunError("checkpoint catalog digest mismatch")
        if expected_run_identity is not None and run_identity != expected_run_identity:
            raise GridRunError("checkpoint run identity mismatch")
        key_counts = payload.get("key_counts", ())
        if type(key_counts) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 0
            for item in key_counts
        ):
            raise GridRunError("checkpoint key counts are invalid")
        if payload.get("model_config_digest") != trainer.backbone.config.identity:
            raise GridRunError("checkpoint model config mismatch")
        trainer.backbone.encoder.load_state_dict(payload["online_encoder"])
        trainer.backbone.target.load_state_dict(payload["ema_target"])
        trainer.backbone.predictor.load_state_dict(payload["predictor"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        step = payload["step"]
        if type(step) is not int or step < 0 or step > trainer.config.steps:
            raise GridRunError("checkpoint step is invalid")
        trainer._step_count = step
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
    except GridRunError:
        raise
    except (KeyError, TypeError, RuntimeError, ValueError, OSError) as error:
        raise GridRunError("checkpoint payload is invalid") from error
    return CheckpointInfo(
        path,
        actual_digest,
        trainer._step_count,
        config_digest,
        catalog_digest,
        run_identity,
        key_counts,
    )


def fixture_batch(config: JepaConfig, *, seed: int, batch_size: int, step: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed + step)
    shape = (batch_size, 3, config.encoder.input_height, config.encoder.input_width)
    keys = [(PredictionPair(delta, Abstraction.CONTINUOUS), value) for delta in APPROVED_DELTAS for value in MotionRegime]
    generator_order = random.Random(seed)
    cycle = step // len(keys)
    for _ in range(cycle + 1):
        generator_order.shuffle(keys)
    pair_value, regime_value = keys[step % len(keys)]
    return {
        "context_image": torch.rand(shape, generator=generator),
        "target_images": torch.rand((batch_size, 1, *shape[1:]), generator=generator),
        "action": torch.rand((batch_size, 5), generator=generator),
        "prediction_pair": pair_value,
        "motion_regime": regime_value,
        "prediction_steps": torch.full((batch_size,), pair_value.delta, dtype=torch.long),
        "shot_frame_count": torch.full((batch_size,), 16, dtype=torch.long),
        "frame_indices": [[0, pair_value.delta] for _ in range(batch_size)],
    }


def score_checkpoint(
    checkpoint: Path,
    *,
    phase_config: PhaseAConfig,
    model_config: JepaConfig,
    batches: tuple[dict[str, object], ...],
) -> ScoreResult:
    score_config = replace(phase_config.training_config(device="cpu"), grid_schedule=False)
    trainer = TeacherForcedTrainer(JepaBackbone(model_config), score_config)
    loaded = load_checkpoint(checkpoint, trainer, config_digest=phase_config.identity, grid_digest=phase_config.grid_digest)
    trainer.backbone.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            target, pair, _, weights = trainer._prepare_batch(batch, loaded.step)
            context = batch["context_image"]
            action = batch["action"]
            assert isinstance(context, torch.Tensor) and isinstance(action, torch.Tensor)
            prediction = trainer.backbone.predict(trainer.backbone.encode(context).latent, action, pair).carrier
            target_latent = trainer.backbone.encode_target(target).latent
            losses = (prediction - target_latent).pow(2).mean(dim=1) * weights
            total += float(losses.sum().item())
            count += int(losses.numel())
    if count == 0:
        raise GridRunError("score received no states")
    return ScoreResult(loaded.step, count, total / count, phase_config.identity)


def write_sweep_manifest(path: Path, *, checkpoint: CheckpointInfo, phase_config: PhaseAConfig, score: ScoreResult) -> None:
    payload = {
        "schema_version": "pair_sweep_manifest_v1",
        "checkpoint_digest": checkpoint.digest,
        "checkpoint_step": checkpoint.step,
        "config_digest": phase_config.identity,
        "grid_digest": phase_config.grid_digest,
        "evaluated_alpha": "continuous",
        "excluded_abstractions": list(ALPHA_EXCLUSIONS),
        "split": phase_config.split,
        "score": {"count": score.count, "mean_loss": score.mean_loss},
    }
    data = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
