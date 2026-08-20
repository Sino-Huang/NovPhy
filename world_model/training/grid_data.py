"""Deterministic episode partitions, scoring states, and motion diagnostics."""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

import torch
import torch.nn.functional as functional
import numpy as np
from PIL import Image, UnidentifiedImageError

from world_model.data.catalog import EpisodeCatalog
from world_model.data.curriculum import catalog_identity as declared_catalog_identity
from world_model.data.types import ContractValueError, EpisodeRecord
from world_model.model import identity


DIAGNOSTIC_IMAGE_SIZE: Final[tuple[int, int]] = (240, 320)
REFERENCE_DELTA: Final[int] = 15
PARTITION_VERSION: Final[str] = "pair-grid-partition-v1"


class GridDataContractError(ContractValueError):
    """Raised when grid data cannot be represented without ambiguity."""


@unique
class MotionRegime(StrEnum):
    """Target-aware diagnostic motion buckets."""

    QUIESCENT = "quiescent"
    TRANSITIONAL = "transitional"
    HIGH_MOTION = "high_motion"


@dataclass(frozen=True, slots=True)
class ScoringTarget:
    """One requested delta and its scoring-only target position."""

    requested_delta: int
    effective_delta: int
    frame_position: int

    def __post_init__(self) -> None:
        if type(self.requested_delta) is not int or self.requested_delta <= 0:
            raise GridDataContractError("requested_delta", "must be a positive integer")
        if type(self.effective_delta) is not int or not 1 <= self.effective_delta <= self.requested_delta:
            raise GridDataContractError("effective_delta", "must be in [1, requested_delta]")
        if type(self.frame_position) is not int or self.frame_position < 1:
            raise GridDataContractError("frame_position", "must be a positive integer")

    @property
    def target_position(self) -> int:
        """Backward-compatible name for the target frame position."""
        return self.frame_position


@dataclass(frozen=True, slots=True)
class ScoringState:
    """Canonical nonterminal state and its three continuous scoring targets."""

    catalog_identity: str
    split: str
    episode_relative_path: str
    shot_relative_path: str
    context_position: int
    shot_frame_count: int
    targets: tuple[ScoringTarget, ...]

    def __post_init__(self) -> None:
        _require_identity(self.catalog_identity, "catalog_identity")
        if not self.split or not self.episode_relative_path or not self.shot_relative_path:
            raise GridDataContractError("state identity", "paths and split must be nonempty")
        if type(self.shot_frame_count) is not int or self.shot_frame_count < 2:
            raise GridDataContractError("shot_frame_count", "must be at least two")
        terminal = self.shot_frame_count - 1
        if type(self.context_position) is not int or not 0 <= self.context_position < terminal:
            raise GridDataContractError("context_position", "must be in [0, T-1]")
        if type(self.targets) is not tuple or len(self.targets) != 3:
            raise GridDataContractError("targets", "must contain exactly the approved three deltas")
        requested = tuple(target.requested_delta for target in self.targets)
        if requested != (1, 5, 15):
            raise GridDataContractError("targets", "must be ordered as requested deltas (1, 5, 15)")
        for target in self.targets:
            if target.frame_position != min(self.context_position + target.requested_delta, terminal):
                raise GridDataContractError("target frame", "must use terminal clamp")

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        """Canonical state key used by score artifacts."""
        return (
            self.catalog_identity,
            self.split,
            self.episode_relative_path,
            self.shot_relative_path,
            self.context_position,
        )

    @property
    def state_key(self) -> tuple[str, str, str, str, int]:
        return self.key

    @property
    def horizon(self) -> int:
        return self.shot_frame_count - 1

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return (self.context_position, *(target.frame_position for target in self.targets))

    @property
    def target_frame_indices(self) -> tuple[int, ...]:
        return tuple(target.frame_position for target in self.targets)


@dataclass(frozen=True, slots=True)
class EpisodePartitions:
    """Exact, deterministic episode membership for the three controller splits."""

    catalog_identity: str
    seed: int | str
    controller_train: tuple[EpisodeRecord, ...]
    calibration: tuple[EpisodeRecord, ...]
    evaluation: tuple[EpisodeRecord, ...]

    @property
    def identity(self) -> str:
        return identity(
            (
                PARTITION_VERSION,
                self.catalog_identity,
                self.seed,
                ("controller-train", 0.0, 0.8),
                ("calibration", 0.8, 0.9),
                ("evaluation", 0.9, 1.0),
            )
        )

    @property
    def membership(self) -> dict[str, tuple[str, ...]]:
        return {
            "controller-train": tuple(ep.relative_path for ep in self.controller_train),
            "calibration": tuple(ep.relative_path for ep in self.calibration),
            "evaluation": tuple(ep.relative_path for ep in self.evaluation),
        }

    @property
    def controller_train_episodes(self) -> tuple[EpisodeRecord, ...]:
        return self.controller_train


@dataclass(frozen=True, slots=True)
class MotionCalibration:
    """Calibrated P50/P90 target-aware motion thresholds."""

    p50: float
    p90: float
    reference_delta: int = REFERENCE_DELTA
    image_size: tuple[int, int] = DIAGNOSTIC_IMAGE_SIZE

    def __post_init__(self) -> None:
        if not all(type(value) in (int, float) and math.isfinite(float(value)) for value in (self.p50, self.p90)):
            raise GridDataContractError("motion thresholds", "must be finite")
        if self.p50 < 0.0 or self.p90 < self.p50:
            raise GridDataContractError("motion thresholds", "must satisfy 0 <= p50 <= p90")
        if self.reference_delta != REFERENCE_DELTA or self.image_size != DIAGNOSTIC_IMAGE_SIZE:
            raise GridDataContractError("motion calibration", "must use reference delta 15 and 240x320")

    def classify(self, score: float) -> MotionRegime:
        if type(score) not in (int, float) or not math.isfinite(float(score)) or score < 0.0:
            raise GridDataContractError("motion score", "must be finite and nonnegative")
        if score <= self.p50:
            return MotionRegime.QUIESCENT
        if score <= self.p90:
            return MotionRegime.TRANSITIONAL
        return MotionRegime.HIGH_MOTION

    @property
    def thresholds(self) -> tuple[float, float]:
        return (self.p50, self.p90)

    @property
    def metadata(self) -> dict[str, int | float | str | tuple[int, int]]:
        return {
            "target_aware": True,
            "reference_delta": self.reference_delta,
            "image_size": self.image_size,
            "p50": self.p50,
            "p90": self.p90,
        }


def _require_identity(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise GridDataContractError(field, "must be a nonempty declared identity")


def canonical_partition_payload(seed: int | str, catalog_identity: str, episode_relative_path: str) -> bytes:
    """Encode the versioned partition tuple with canonical JSON bytes."""
    _require_identity(catalog_identity, "catalog_identity")
    if type(seed) not in (int, str):
        raise GridDataContractError("seed", "must be an integer or string")
    if type(episode_relative_path) is not str or not episode_relative_path:
        raise GridDataContractError("episode_relative_path", "must be nonempty")
    return json.dumps(
        (PARTITION_VERSION, seed, catalog_identity, episode_relative_path),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def partition_fraction(seed: int | str, catalog_identity: str, episode_relative_path: str) -> float:
    payload = canonical_partition_payload(seed, catalog_identity, episode_relative_path)
    # deterministic non-cryptographic derivation, not an integrity check
    return zlib.crc32(payload) / 2**32


def _resolve_episodes(
    catalog_or_episodes: EpisodeCatalog | tuple[EpisodeRecord, ...],
    supplied_identity: str | None,
    supplied_split: str | None,
) -> tuple[tuple[EpisodeRecord, ...], str, str]:
    if isinstance(catalog_or_episodes, EpisodeCatalog):
        declared_identity = declared_catalog_identity(catalog_or_episodes)
        split = catalog_or_episodes.split
        if supplied_identity is not None and supplied_identity != declared_identity:
            raise GridDataContractError(
                "catalog_identity", "does not match the catalog's declared identity"
            )
        if supplied_split is not None and supplied_split != split:
            raise GridDataContractError("split", "does not match the catalog snapshot")
        return catalog_or_episodes.episodes, declared_identity, split
    if type(catalog_or_episodes) is not tuple or not catalog_or_episodes:
        raise GridDataContractError("episodes", "must be a nonempty immutable tuple")
    if supplied_identity is None:
        raise GridDataContractError("catalog_identity", "is required for episode tuples")
    _require_identity(supplied_identity, "catalog_identity")
    split = supplied_split or str(catalog_or_episodes[0].split)
    if any(str(episode.split) != split for episode in catalog_or_episodes):
        raise GridDataContractError("split", "episodes must belong to one split")
    return catalog_or_episodes, supplied_identity, split


def partition_episodes(
    catalog_or_episodes: EpisodeCatalog | tuple[EpisodeRecord, ...],
    *,
    catalog_identity: str | None = None,
    seed: int | str,
) -> EpisodePartitions:
    """Partition dev episodes using deterministic 80/10/10 ranges and stable ordering."""
    episodes, identity_value, split = _resolve_episodes(catalog_or_episodes, catalog_identity, "dev")
    if split != "dev":
        raise GridDataContractError("split", "only dev episodes can be partitioned")
    buckets: dict[str, list[EpisodeRecord]] = {
        "controller-train": [],
        "calibration": [],
        "evaluation": [],
    }
    for episode in episodes:
        fraction = partition_fraction(seed, identity_value, episode.relative_path)
        bucket = "controller-train" if fraction < 0.8 else "calibration" if fraction < 0.9 else "evaluation"
        buckets[bucket].append(episode)
    for values in buckets.values():
        values.sort(key=lambda episode: episode.relative_path)
    return EpisodePartitions(
        catalog_identity=identity_value,
        seed=seed,
        controller_train=tuple(buckets["controller-train"]),
        calibration=tuple(buckets["calibration"]),
        evaluation=tuple(buckets["evaluation"]),
    )


def enumerate_scoring_states(
    catalog_or_episodes: EpisodeCatalog | tuple[EpisodeRecord, ...],
    *,
    catalog_identity: str | None = None,
    split: str | None = None,
) -> tuple[ScoringState, ...]:
    """Enumerate every nonterminal state with all three scoring-only targets."""
    episodes, identity_value, split_value = _resolve_episodes(catalog_or_episodes, catalog_identity, split)
    states: list[ScoringState] = []
    for episode in episodes:
        for shot in episode.shots:
            frame_count = len(shot.frames)
            if frame_count < 2:
                raise GridDataContractError("shot_frame_count", "one-frame shots cannot produce states")
            terminal = frame_count - 1
            for context_position in range(terminal):
                targets = tuple(
                    ScoringTarget(
                        requested_delta=delta,
                        effective_delta=min(delta, terminal - context_position),
                        frame_position=min(context_position + delta, terminal),
                    )
                    for delta in (1, 5, 15)
                )
                states.append(
                    ScoringState(
                        catalog_identity=identity_value,
                        split=split_value,
                        episode_relative_path=episode.relative_path,
                        shot_relative_path=shot.relative_path,
                        context_position=context_position,
                        shot_frame_count=frame_count,
                        targets=targets,
                    )
                )
    return tuple(states)


def diagnostic_motion_for_state(catalog: EpisodeCatalog, state: ScoringState) -> float:
    """Read one state and its reference-delta target from a catalog snapshot."""
    if (
        state.catalog_identity != declared_catalog_identity(catalog)
        or state.split != catalog.split
    ):
        raise GridDataContractError("state", "does not match the catalog snapshot")
    if state.targets[-1].requested_delta != REFERENCE_DELTA:
        raise GridDataContractError("state targets", "reference delta 15 is required")
    episode = next((item for item in catalog.episodes if item.relative_path == state.episode_relative_path), None)
    if episode is None:
        raise GridDataContractError("state episode", "is absent from the catalog snapshot")
    shot = next((item for item in episode.shots if item.relative_path == state.shot_relative_path), None)
    if shot is None:
        raise GridDataContractError("state shot", "is absent from the catalog snapshot")
    frame_count = len(shot.frames)
    if state.shot_frame_count != frame_count:
        raise GridDataContractError("state shot_frame_count", "does not match the catalog snapshot")
    terminal = frame_count - 1
    if not 0 <= state.context_position < terminal:
        raise GridDataContractError("state context_position", "is not nonterminal in the catalog snapshot")
    for target in state.targets:
        expected_position = min(state.context_position + target.requested_delta, terminal)
        if target.frame_position != expected_position:
            raise GridDataContractError("state target", "does not match the catalog snapshot")
    root = object.__getattribute__(catalog, "_root")
    context_path = root / shot.frames[state.context_position].relative_path
    target_path = root / shot.frames[state.targets[-1].frame_position].relative_path
    return diagnostic_motion_score(context_path, target_path)


def _image_tensor(image: torch.Tensor | Image.Image | Path | str) -> torch.Tensor:
    if isinstance(image, (Path, str)):
        try:
            with Image.open(image) as opened:
                return _image_tensor(opened)
        except (OSError, UnidentifiedImageError) as error:
            raise GridDataContractError("image", f"cannot decode {image!s}") from error
    if isinstance(image, Image.Image):
        values = torch.from_numpy(np.array(image.convert("RGB"), dtype="float32"))
        return values.permute(2, 0, 1).div(255.0)
    if not isinstance(image, torch.Tensor):
        raise GridDataContractError("image", "must be a torch tensor, PIL image, or path")
    values = image.detach().to(dtype=torch.float32)
    if values.ndim != 3:
        raise GridDataContractError("image", "must have three dimensions")
    if values.shape[0] not in (1, 3) and values.shape[-1] in (1, 3):
        values = values.permute(2, 0, 1)
    if values.shape[0] not in (1, 3):
        raise GridDataContractError("image", "must be CHW or HWC with one or three channels")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values > 1).any()):
        raise GridDataContractError("image", "float values must lie in [0, 1]")
    return values if values.shape[0] == 3 else values.repeat(3, 1, 1)


def diagnostic_motion_score(
    context: torch.Tensor | Image.Image | Path | str,
    target: torch.Tensor | Image.Image | Path | str,
) -> float:
    """Measure target-aware mean absolute RGB motion after fixed resizing."""
    context_tensor = functional.interpolate(
        _image_tensor(context).unsqueeze(0), size=DIAGNOSTIC_IMAGE_SIZE, mode="bilinear", align_corners=False
    )
    target_tensor = functional.interpolate(
        _image_tensor(target).unsqueeze(0), size=DIAGNOSTIC_IMAGE_SIZE, mode="bilinear", align_corners=False
    )
    return float(torch.mean(torch.abs(target_tensor - context_tensor)).item())


def calibrate_motion_regimes(scores: tuple[float, ...] | list[float]) -> MotionCalibration:
    """Calibrate P50/P90 thresholds with linear percentile interpolation."""
    if not scores:
        raise GridDataContractError("motion scores", "must be nonempty")
    values = tuple(float(score) for score in scores)
    if any(not math.isfinite(score) or score < 0.0 for score in values):
        raise GridDataContractError("motion scores", "must be finite and nonnegative")

    ordered = sorted(values)

    def percentile(probability: float) -> float:
        rank = probability * (len(ordered) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)

    return MotionCalibration(p50=percentile(0.5), p90=percentile(0.9))


def classify_motion(score: float, calibration: MotionCalibration) -> MotionRegime:
    """Classify one score with deterministic threshold tie handling."""
    return calibration.classify(score)


partition_dev_episodes = partition_episodes
build_state_index = enumerate_scoring_states
state_index = enumerate_scoring_states
motion_score = diagnostic_motion_score


__all__ = [
    "DIAGNOSTIC_IMAGE_SIZE",
    "MotionCalibration",
    "MotionRegime",
    "EpisodePartitions",
    "GridDataContractError",
    "REFERENCE_DELTA",
    "ScoringState",
    "ScoringTarget",
    "calibrate_motion_regimes",
    "canonical_partition_payload",
    "classify_motion",
    "diagnostic_motion_score",
    "diagnostic_motion_for_state",
    "enumerate_scoring_states",
    "build_state_index",
    "state_index",
    "partition_dev_episodes",
    "partition_fraction",
    "partition_episodes",
    "motion_score",
]
