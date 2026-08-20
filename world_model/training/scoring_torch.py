"""Torch adapters for exhaustive scoring from a frozen checkpoint."""
from __future__ import annotations

import zlib
from pathlib import Path

import torch

from world_model.model import Abstraction, JepaBackbone, JepaConfig, PredictionPair, identity
from world_model.training.grid_data import MotionRegime, ScoringState, ScoringTarget
from world_model.training.grid_run import (
    CheckpointInfo,
    GridRunError,
    PhaseAConfig,
    load_checkpoint,
)
from world_model.training.loop import TeacherForcedTrainer
from world_model.training.real_data import RealPhaseData
from world_model.training.scoring import (
    ExhaustiveScoreResult,
    ExhaustiveScorer,
    Partition,
    ScoringExample,
    score_state_set_identity,
)


FIXTURE_CATALOG_IDENTITY = (
    "episode-catalog-v1:fixture-cohort:fixture-dataset:fixture-source:dev:legacy_rgb_v1"
)


def fixture_partition_identity(phase_config: PhaseAConfig) -> str:
    return identity(
        (
            "pair-grid-partition-v1",
            FIXTURE_CATALOG_IDENTITY,
            phase_config.seed,
            "balanced-fixture-partitions",
        )
    )


def fixture_state_set_identity(phase_config: PhaseAConfig) -> str:
    examples = fixture_scoring_examples(phase_config.steps)
    return score_state_set_identity(
        FIXTURE_CATALOG_IDENTITY,
        fixture_partition_identity(phase_config),
        tuple(example.state_id for example in examples),
    )


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
            # deterministic non-cryptographic derivation, not an integrity check
            seed = zlib.crc32(f"{example.state_id}:{effective_delta}".encode("ascii"))
            generator = torch.Generator().manual_seed(seed)
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
    """Score catalog shots with bounded decode/encode/predictor batches."""

    def __init__(
        self,
        backbone: JepaBackbone,
        data: RealPhaseData,
        batch_size: int,
        examples: tuple[ScoringExample, ...],
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise GridRunError("real scoring batch_size must be a positive integer")
        self._backbone = backbone
        self._data = data
        self._batch_size = batch_size
        self._device = next(backbone.parameters()).device
        self._examples = examples
        self._shot_latents: dict[
            tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _cache_shot(
        self, key: tuple[str, str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._shot_latents.get(key)
        if cached is not None:
            return cached
        frame_count = self._data.shot_frame_count(*key)
        context_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        for offset in range(0, frame_count, self._batch_size):
            positions = tuple(range(offset, min(offset + self._batch_size, frame_count)))
            frames = self._data.shot_frame_batch(*key, positions)
            if frames.shape[0] != len(positions) or frames.shape[0] > self._batch_size:
                raise GridRunError("catalog decoder returned an invalid batch")
            device_frames = frames.to(self._device)
            with torch.no_grad():
                context = self._backbone.encode(device_frames).latent.detach().cpu()
                target = self._backbone.encode_target(device_frames).latent.detach().cpu()
            if context.shape[0] > self._batch_size or target.shape[0] > self._batch_size:
                raise GridRunError("catalog encoder returned an oversized batch")
            context_chunks.append(context)
            target_chunks.append(target)
        cached = (
            torch.cat(context_chunks, dim=0),
            torch.cat(target_chunks, dim=0),
            self._data.shot_action(*key).detach().cpu(),
        )
        self._shot_latents[key] = cached
        return cached

    def latent_mse(
        self,
        examples: tuple[ScoringExample, ...],
        requested_delta: int,
        effective_delta: int,
    ) -> tuple[float, ...]:
        del effective_delta
        losses: list[float] = []
        pair = PredictionPair(requested_delta, Abstraction.CONTINUOUS)
        for offset in range(0, len(examples), self._batch_size):
            batch = examples[offset : offset + self._batch_size]
            rows = tuple(self._data.state_record(item.state_id) for item in batch)
            shots = {
                (row.episode_relative_path, row.shot_relative_path): self._cache_shot(
                    (row.episode_relative_path, row.shot_relative_path)
                )
                for row in rows
            }
            for item, row in zip(batch, rows, strict=True):
                if item.frame_count != row.shot_frame_count or item.context_position != row.context_position:
                    raise GridRunError("scoring example does not match its canonical state")
                key = (row.episode_relative_path, row.shot_relative_path)
                if self._data.shot_frame_count(*key) != row.shot_frame_count:
                    raise GridRunError("canonical state does not match the catalog shot")
            context_rows: list[torch.Tensor] = []
            target_rows: list[torch.Tensor] = []
            action_rows: list[torch.Tensor] = []
            for row in rows:
                key = (row.episode_relative_path, row.shot_relative_path)
                context_latents, target_latents, action = shots[key]
                context_position = row.context_position
                target_position = min(context_position + requested_delta, row.shot_frame_count - 1)
                context_rows.append(context_latents[context_position])
                target_rows.append(target_latents[target_position])
                action_rows.append(action)
            context = torch.stack(context_rows).to(self._device)
            target = torch.stack(target_rows).to(self._device)
            action = torch.stack(action_rows).to(self._device)
            with torch.no_grad():
                prediction = self._backbone.predict(
                    context,
                    action,
                    pair,
                ).carrier
                values = (prediction - target).pow(2).mean(dim=1)
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
                catalog_identity=FIXTURE_CATALOG_IDENTITY,
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
) -> tuple[ExhaustiveScoreResult, CheckpointInfo]:
    trainer = TeacherForcedTrainer(JepaBackbone(model_config), phase_config.training_config(device="cpu"))
    loaded = load_checkpoint(
        checkpoint,
        trainer,
        config_identity=phase_config.identity,
        grid_identity=phase_config.grid_identity,
    )
    if loaded.step != phase_config.steps:
        raise GridRunError("exhaustive scoring requires a completed train-mode checkpoint")
    trainer.backbone.eval()
    result = ExhaustiveScorer(TorchFixturePredictor(trainer.backbone, model_config)).score(
        fixture_scoring_examples(phase_config.steps)
    )
    return result, loaded


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
        config_identity=phase_config.identity,
        grid_identity=phase_config.grid_identity,
        expected_catalog_identity=data.catalog_identity,
        expected_run_identity=data.run_identity,
    )
    if loaded.step != phase_config.steps:
        raise GridRunError("exhaustive scoring requires a completed train-mode checkpoint")
    trainer.backbone.eval()
    result = ExhaustiveScorer(
        TorchCatalogPredictor(trainer.backbone, data, phase_config.batch_size, data.examples)
    ).score(data.examples)
    return result, loaded


__all__ = [
    "FIXTURE_CATALOG_IDENTITY",
    "TorchCatalogPredictor",
    "TorchFixturePredictor",
    "fixture_partition_identity",
    "fixture_scoring_examples",
    "fixture_state_set_identity",
    "score_fixture_checkpoint",
    "score_real_checkpoint",
]
