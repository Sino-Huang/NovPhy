"""Torch adapters for exhaustive scoring from a frozen checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from world_model.model import Abstraction, JepaBackbone, JepaConfig, PredictionPair
from world_model.training.grid_data import MotionRegime, ScoringState, ScoringTarget
from world_model.training.grid_run import (
    CheckpointInfo,
    GridRunError,
    PhaseAConfig,
    checkpoint_digest,
    load_checkpoint,
)
from world_model.training.loop import TeacherForcedTrainer
from world_model.training.real_data import RealPhaseData
from world_model.training.scoring import ExhaustiveScoreResult, ExhaustiveScorer, Partition, ScoringExample


class TorchFixturePredictor:
    def __init__(self, backbone: JepaBackbone, model_config: JepaConfig) -> None:
        self._backbone = backbone
        self._config = model_config

    def latent_mse(
        self,
        examples: tuple[ScoringExample, ...],
        requested_delta: int,
        effective_delta: int,
    ) -> tuple[float, ...]:
        contexts: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        shape = (3, self._config.encoder.input_height, self._config.encoder.input_width)
        for example in examples:
            seed_bytes = hashlib.sha256(f"{example.state_id}:{effective_delta}".encode("ascii")).digest()[:8]
            generator = torch.Generator().manual_seed(int.from_bytes(seed_bytes, "big"))
            contexts.append(torch.rand(shape, generator=generator))
            targets.append(torch.rand(shape, generator=generator))
            actions.append(torch.rand((self._config.predictor.action_dim,), generator=generator))
        context = torch.stack(contexts)
        target = torch.stack(targets)
        action = torch.stack(actions)
        pair = PredictionPair(requested_delta, Abstraction.CONTINUOUS)
        with torch.no_grad():
            prediction = self._backbone.predict(self._backbone.encode(context).latent, action, pair).carrier
            target_latent = self._backbone.encode_target(target).latent
            losses = (prediction - target_latent).pow(2).mean(dim=1)
        return tuple(float(value) for value in losses.tolist())


class TorchCatalogPredictor:
    """Decode catalog frames in bounded batches and score frozen model weights."""

    def __init__(self, backbone: JepaBackbone, data: RealPhaseData, batch_size: int) -> None:
        self._backbone = backbone
        self._data = data
        self._batch_size = batch_size
        self._device = next(backbone.parameters()).device

    def latent_mse(
        self,
        examples: tuple[ScoringExample, ...],
        requested_delta: int,
        effective_delta: int,
    ) -> tuple[float, ...]:
        losses: list[float] = []
        pair = PredictionPair(requested_delta, Abstraction.CONTINUOUS)
        for offset in range(0, len(examples), self._batch_size):
            batch = examples[offset : offset + self._batch_size]
            triples = tuple(
                self._data.tensor_triplet(
                    example.state_id, example.context_position + effective_delta
                )
                for example in batch
            )
            context = torch.stack([item[0] for item in triples]).to(self._device)
            target = torch.stack([item[1] for item in triples]).to(self._device)
            action = torch.stack([item[2] for item in triples]).to(self._device)
            with torch.no_grad():
                prediction = self._backbone.predict(
                    self._backbone.encode(context).latent, action, pair
                ).carrier
                target_latent = self._backbone.encode_target(target).latent
                values = (prediction - target_latent).pow(2).mean(dim=1)
            losses.extend(float(value) for value in values.cpu().tolist())
        return tuple(losses)


def fixture_scoring_examples(steps: int) -> tuple[ScoringExample, ...]:
    if type(steps) is not int or steps < 3:
        raise GridRunError("fixture exhaustive scoring requires at least three steps")
    examples: list[ScoringExample] = []
    for partition in Partition:
        for position in range(steps):
            terminal = steps
            state = ScoringState(
                catalog_digest="f" * 64,
                split="dev",
                episode_relative_path=f"fixture/{partition}",
                shot_relative_path=f"fixture/{partition}/shot",
                context_position=position,
                shot_frame_count=steps + 1,
                targets=tuple(
                    ScoringTarget(delta, min(delta, terminal - position), min(position + delta, terminal))
                    for delta in (1, 5, 15)
                ),
            )
            examples.append(
                ScoringExample.from_grid_state(
                    state,
                    partition,
                    motion_regime=tuple(MotionRegime)[position % len(MotionRegime)],
                )
            )
    return tuple(examples)


def score_fixture_checkpoint(
    checkpoint: Path,
    phase_config: PhaseAConfig,
    model_config: JepaConfig,
) -> tuple[ExhaustiveScoreResult, str]:
    trainer = TeacherForcedTrainer(JepaBackbone(model_config), phase_config.training_config(device="cpu"))
    loaded = load_checkpoint(
        checkpoint,
        trainer,
        config_digest=phase_config.identity,
        grid_digest=phase_config.grid_digest,
        expected_digest=checkpoint_digest(checkpoint),
    )
    if loaded.step != phase_config.steps:
        raise GridRunError("exhaustive scoring requires a completed train-mode checkpoint")
    trainer.backbone.eval()
    result = ExhaustiveScorer(TorchFixturePredictor(trainer.backbone, model_config)).score(
        fixture_scoring_examples(phase_config.steps)
    )
    return result, loaded.digest


def score_real_checkpoint(
    checkpoint: Path,
    phase_config: PhaseAConfig,
    model_config: JepaConfig,
    data: RealPhaseData,
) -> tuple[ExhaustiveScoreResult, CheckpointInfo]:
    trainer = TeacherForcedTrainer(
        JepaBackbone(model_config), phase_config.training_config(device=phase_config.device)
    )
    loaded = load_checkpoint(
        checkpoint,
        trainer,
        config_digest=phase_config.identity,
        grid_digest=phase_config.grid_digest,
        expected_digest=checkpoint_digest(checkpoint),
        expected_catalog_digest=data.catalog_digest,
        expected_run_identity=data.run_identity,
    )
    if loaded.step != phase_config.steps:
        raise GridRunError("exhaustive scoring requires a completed train-mode checkpoint")
    trainer.backbone.eval()
    result = ExhaustiveScorer(
        TorchCatalogPredictor(trainer.backbone, data, phase_config.batch_size)
    ).score(data.examples)
    return result, loaded


__all__ = [
    "TorchCatalogPredictor",
    "TorchFixturePredictor",
    "fixture_scoring_examples",
    "score_fixture_checkpoint",
    "score_real_checkpoint",
]
