"""Read-only legacy-dev adapter for Phase-A training and exhaustive scoring."""
from __future__ import annotations

import os
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.nn import functional

from world_model.data import LEGACY_RGB_V1, EpisodeCatalog, catalog_identity
from world_model.data.types import ShotRecord
from world_model.model import JepaConfig, PredictionPair, identity
from world_model.training.frontier import FRONTIER_INPUT_SCHEMA
from world_model.training.grid_artifacts import ALPHA_EXCLUSIONS, canonical_json_bytes
from world_model.training.grid_data import (
    DIAGNOSTIC_IMAGE_SIZE,
    EpisodePartitions,
    MotionCalibration,
    MotionRegime,
    ScoringState,
    calibrate_motion_regimes,
    enumerate_scoring_states,
    partition_episodes,
)
from world_model.training.grid_run import CheckpointInfo, GridRunError, PhaseAConfig
from world_model.training.scoring import Partition, ScoringExample
from world_model.training.scoring_artifacts import (
    ScoreArtifactReceipt,
    score_state_set_identity,
    validate_score_artifacts,
)

_PARTITIONS: Final = (
    Partition.CONTROLLER_TRAIN,
    Partition.CALIBRATION,
    Partition.EVALUATION,
)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _decode(path: Path, size: tuple[int, int], *, antialias: bool = True) -> torch.Tensor:
    try:
        with Image.open(path) as opened:
            values = np.asarray(opened.convert("RGB"), dtype=np.float32).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise GridRunError(f"cannot decode catalog frame: {path}") from error
    tensor = torch.from_numpy(values).permute(2, 0, 1).div(255.0)
    return functional.interpolate(
        tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False, antialias=antialias
    ).squeeze(0)


def _shot_motion_scores(root: Path, shot: ShotRecord) -> tuple[float, ...]:
    frames = tuple(
        _decode(root / frame.relative_path, DIAGNOSTIC_IMAGE_SIZE, antialias=False)
        for frame in shot.frames
    )
    terminal = len(frames) - 1
    return tuple(
        float(torch.mean(torch.abs(frames[min(context + 15, terminal)] - frames[context])).item())
        for context in range(terminal)
    )


def _key(pair: PredictionPair, regime: MotionRegime) -> str:
    return f"delta={pair.delta},regime={regime.value}"


@dataclass(frozen=True, slots=True)
class RealPhaseData:
    """Frozen catalog/index snapshot used by both training and scoring."""

    catalog: EpisodeCatalog
    catalog_identity: str
    partitions: EpisodePartitions
    calibration: MotionCalibration
    states: tuple[ScoringState, ...]
    examples: tuple[ScoringExample, ...]
    run_identity: str
    _root: Path
    _shots: dict[tuple[str, str], ShotRecord]
    _state_by_id: dict[str, ScoringState]
    _regime_by_id: dict[str, MotionRegime]
    _train_pools: dict[str, tuple[str, ...]]
    _image_size: tuple[int, int]
    _seed: int

    @property
    def partition_identity(self) -> str:
        return self.partitions.identity

    @property
    def state_set_identity(self) -> str:
        return score_state_set_identity(
            self.catalog_identity,
            self.partition_identity,
            tuple(example.state_id for example in self.examples),
        )

    @classmethod
    def build(
        cls,
        dataset_root: Path,
        phase_config: PhaseAConfig,
        model_config: JepaConfig,
    ) -> RealPhaseData:
        if not dataset_root.is_dir():
            raise GridRunError(f"dataset root not found: {dataset_root}")
        catalog = EpisodeCatalog.build(dataset_root, "dev", LEGACY_RGB_V1)
        if not catalog.episodes:
            raise GridRunError("legacy dev catalog contains no accepted episodes")
        catalog_id = catalog_identity(catalog)
        partitions = partition_episodes(catalog, seed=phase_config.seed)
        membership = {
            Partition.CONTROLLER_TRAIN: frozenset(item.relative_path for item in partitions.controller_train),
            Partition.CALIBRATION: frozenset(item.relative_path for item in partitions.calibration),
            Partition.EVALUATION: frozenset(item.relative_path for item in partitions.evaluation),
        }
        if any(not membership[partition] for partition in _PARTITIONS):
            raise GridRunError("real exhaustive scoring requires three nonempty episode partitions")
        states = enumerate_scoring_states(catalog)
        shots = {
            (episode.relative_path, shot.relative_path): shot
            for episode in catalog.episodes
            for shot in episode.shots
        }
        root = object.__getattribute__(catalog, "_root")
        scores = tuple(
            score
            for episode in catalog.episodes
            for shot in episode.shots
            for score in _shot_motion_scores(root, shot)
        )
        calibration_scores = tuple(
            score
            for state, score in zip(states, scores, strict=True)
            if state.episode_relative_path in membership[Partition.CALIBRATION]
        )
        calibration = calibrate_motion_regimes(calibration_scores)
        examples: list[ScoringExample] = []
        state_by_id: dict[str, ScoringState] = {}
        regime_by_id: dict[str, MotionRegime] = {}
        pools: dict[str, list[str]] = {
            _key(PredictionPair(delta, "continuous"), regime): []
            for delta in (1, 5, 15)
            for regime in MotionRegime
        }
        for state, score in zip(states, scores, strict=True):
            partition = next(item for item in _PARTITIONS if state.episode_relative_path in membership[item])
            regime = calibration.classify(score)
            example = ScoringExample.from_grid_state(state, partition, regime)
            examples.append(example)
            state_by_id[example.state_id] = state
            regime_by_id[example.state_id] = regime
            if partition is Partition.CONTROLLER_TRAIN:
                remaining = state.shot_frame_count - 1 - state.context_position
                for delta in (1, 5, 15):
                    if remaining >= delta:
                        pools[_key(PredictionPair(delta, "continuous"), regime)].append(example.state_id)
        empty = tuple(name for name, values in pools.items() if not values)
        if empty:
            raise GridRunError(f"real training has no eligible windows for keys: {empty}")
        run_identity = identity(
            ("phase-a-real-run-v1", catalog_id, phase_config.identity, phase_config.grid_identity, model_config.identity)
        )
        return cls(
            catalog, catalog_id, partitions, calibration, states, tuple(examples), run_identity,
            root, shots, state_by_id, regime_by_id,
            {name: tuple(values) for name, values in pools.items()},
            (model_config.encoder.input_height, model_config.encoder.input_width), phase_config.seed,
        )

    def tensor_triplet(self, state_id: str, target_position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._state_by_id[state_id]
        shot = self._shots[(state.episode_relative_path, state.shot_relative_path)]
        context = _decode(self._root / shot.frames[state.context_position].relative_path, self._image_size)
        target = _decode(self._root / shot.frames[target_position].relative_path, self._image_size)
        action = torch.tensor(shot.action.values, dtype=torch.float32)
        return context, target, action

    def state_record(self, state_id: str) -> ScoringState:
        """Return the immutable scoring index record for a state identity."""
        return self._state_by_id[state_id]

    def shot_frame_count(self, episode_relative_path: str, shot_relative_path: str) -> int:
        """Return the immutable frame count for one catalog shot."""
        shot = self._shots[(episode_relative_path, shot_relative_path)]
        return len(shot.frames)

    def shot_action(self, episode_relative_path: str, shot_relative_path: str) -> torch.Tensor:
        """Return one shot's action as a CPU tensor."""
        shot = self._shots[(episode_relative_path, shot_relative_path)]
        return torch.tensor(shot.action.values, dtype=torch.float32)

    def shot_frame_batch(
        self,
        episode_relative_path: str,
        shot_relative_path: str,
        frame_positions: tuple[int, ...],
    ) -> torch.Tensor:
        """Decode only the requested frame positions as one bounded batch."""
        if type(frame_positions) is not tuple or not frame_positions:
            raise GridRunError("frame batch must be a nonempty immutable tuple")
        shot = self._shots[(episode_relative_path, shot_relative_path)]
        if any(type(position) is not int or not 0 <= position < len(shot.frames) for position in frame_positions):
            raise GridRunError("frame batch position is outside the catalog shot")
        return torch.stack(
            tuple(_decode(self._root / shot.frames[position].relative_path, self._image_size) for position in frame_positions)
        )

    def training_batch(
        self, pair: PredictionPair, regime: MotionRegime, batch_size: int, step: int
    ) -> dict[str, object]:
        pool = self._train_pools[_key(pair, regime)]
        selector_declaration = identity(
            ("real-grid-selector-v1", self._seed, step, pair.identity, regime.value)
        )
        # Deterministic non-integrity derivation for the local PRNG seed.
        selector_seed = zlib.crc32(selector_declaration.encode("utf-8"))
        selector = random.Random(selector_seed)
        selected = tuple(pool[selector.randrange(len(pool))] for _ in range(batch_size))
        triples = tuple(
            self.tensor_triplet(state_id, self._state_by_id[state_id].context_position + pair.delta)
            for state_id in selected
        )
        return {
            "context_image": torch.stack([item[0] for item in triples]),
            "target_images": torch.stack([item[1] for item in triples]).unsqueeze(1),
            "action": torch.stack([item[2] for item in triples]),
            "state_ids": selected,
            "prediction_pair": pair,
            "motion_regime": regime,
            "prediction_steps": torch.full((batch_size,), pair.delta, dtype=torch.long),
            "shot_frame_count": torch.tensor(
                [self._state_by_id[state_id].shot_frame_count for state_id in selected], dtype=torch.long
            ),
            "frame_indices": [
                [self._state_by_id[state_id].context_position, self._state_by_id[state_id].context_position + pair.delta]
                for state_id in selected
            ],
        }


def write_real_sweep_manifest(
    path: Path,
    *,
    data: RealPhaseData,
    phase_config: PhaseAConfig,
    checkpoint: CheckpointInfo,
    score: ScoreArtifactReceipt,
) -> None:
    payload = {
        "schema_version": "phase_a_real_sweep_v1",
        "catalog_identity": data.catalog_identity,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_step": checkpoint.step,
        "config_identity": phase_config.identity,
        "excluded_abstractions": list(ALPHA_EXCLUSIONS),
        "grid_identity": phase_config.grid_identity,
        "key_counts": dict(checkpoint.key_counts),
        "motion_calibration": data.calibration.metadata,
        "partitions": data.partitions.membership,
        "run_identity": data.run_identity,
        "reproducibility": phase_config.reproducibility.canonical,
        "score_count": score.score_count,
        "state_count": score.state_count,
    }
    _atomic_write(path, canonical_json_bytes(payload))


def write_frontier_input(score_root: Path, path: Path) -> None:
    validate_score_artifacts(score_root)
    payload = {
        "partition": "evaluation",
        "schema_version": FRONTIER_INPUT_SCHEMA,
        "score_artifact_root": Path(
            os.path.relpath(score_root.resolve(), start=path.parent.resolve())
        ).as_posix(),
    }
    _atomic_write(path, canonical_json_bytes(payload))


__all__ = ["RealPhaseData", "write_frontier_input", "write_real_sweep_manifest"]
