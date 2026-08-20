"""Reproducibility manifest for a JEPA training run.

The declared identity covers every input that can change a seeded run's outcome
and deliberately excludes wall-clock timing, so equivalent experiments share
the same plain semantic identity.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Final, Literal, TypedDict

import torch

from world_model.data.types import ContractValueError
from world_model.model.config import identity

MANIFEST_VERSION: Final = "jepa-run-manifest-v1"

Acceptance: Final = Literal["pass", "fail"]


class RunManifestPayload(TypedDict, total=False):
    manifest_version: str
    run_id: str
    mode: str
    seed: int
    git_commit: str
    git_dirty: bool
    torch_version: str
    cuda_version: str
    device_name: str
    dataset_root: str
    split: str
    catalog_identity: str
    accepted_episode_count: int
    rejected_episode_count: int
    window_count: int
    prediction_steps: int
    stride_frames: int
    abstraction: str
    batch_size: int
    steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    ema_base_momentum: float
    model_config_identity: str
    sampled_index_identity: str
    window_selection: str
    candidate_count: int
    symbolic_loss_active: bool
    final_loss: float
    mean_feature_std: float
    relative_spread: float
    effective_rank: float
    retrieval_accuracy: float
    acceptance: str
    started_at_unix: float
    wall_clock_seconds: float
    identity: str


def _require_nonempty(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ContractValueError(field, "must be a nonempty string")


def _require_positive_integer(value: int, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise ContractValueError(field, "must be a positive integer")


def _require_nonnegative_integer(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ContractValueError(field, "must be a nonnegative integer")


def _require_finite(value: float, field: str) -> None:
    if type(value) not in (int, float) or value != value or abs(value) == float("inf"):
        raise ContractValueError(field, "must be a finite number")


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The machine-readable record of one reproducible run."""

    manifest_version: str
    run_id: str
    mode: str
    seed: int
    git_commit: str
    git_dirty: bool
    torch_version: str
    cuda_version: str
    device_name: str
    dataset_root: str
    split: str
    catalog_identity: str
    accepted_episode_count: int
    rejected_episode_count: int
    window_count: int
    prediction_steps: int
    stride_frames: int
    abstraction: str
    batch_size: int
    steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    ema_base_momentum: float
    model_config_identity: str
    sampled_index_identity: str
    window_selection: str
    candidate_count: int
    symbolic_loss_active: bool
    final_loss: float
    mean_feature_std: float
    relative_spread: float
    effective_rank: float
    retrieval_accuracy: float
    acceptance: str
    started_at_unix: float
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        _require_nonempty(self.manifest_version, "manifest_version")
        _require_nonempty(self.run_id, "run_id")
        _require_nonempty(self.mode, "mode")
        _require_nonempty(self.git_commit, "git_commit")
        _require_nonempty(self.torch_version, "torch_version")
        _require_nonempty(self.dataset_root, "dataset_root")
        _require_nonempty(self.split, "split")
        _require_nonempty(self.catalog_identity, "catalog_identity")
        _require_nonempty(self.abstraction, "abstraction")
        _require_nonempty(self.model_config_identity, "model_config_identity")
        _require_nonempty(self.sampled_index_identity, "sampled_index_identity")
        _require_nonempty(self.window_selection, "window_selection")
        if self.window_selection not in ("motion", "uniform", "diverse"):
            raise ContractValueError(
                "window_selection", "must be 'motion', 'uniform', or 'diverse'"
            )
        _require_positive_integer(self.candidate_count, "candidate_count")
        _require_positive_integer(self.window_count, "window_count")
        _require_positive_integer(self.prediction_steps, "prediction_steps")
        _require_positive_integer(self.stride_frames, "stride_frames")
        _require_positive_integer(self.batch_size, "batch_size")
        _require_positive_integer(self.steps, "steps")
        # Zero warmup is a legitimate schedule (cosine decay from step one).
        _require_nonnegative_integer(self.warmup_steps, "warmup_steps")
        _require_nonnegative_integer(self.seed, "seed")
        _require_nonnegative_integer(self.accepted_episode_count, "accepted_episode_count")
        _require_nonnegative_integer(self.rejected_episode_count, "rejected_episode_count")
        if self.learning_rate <= 0.0:
            raise ContractValueError("learning_rate", "must be positive")
        if self.weight_decay < 0.0:
            raise ContractValueError("weight_decay", "must be nonnegative")
        if not 0.0 <= self.grad_clip <= 1e3:
            raise ContractValueError("grad_clip", "must lie in [0, 1000]")
        if not 0.0 <= self.ema_base_momentum <= 1.0:
            raise ContractValueError("ema_base_momentum", "must lie in the unit interval")
        _require_finite(self.final_loss, "final_loss")
        _require_finite(self.mean_feature_std, "mean_feature_std")
        _require_finite(self.relative_spread, "relative_spread")
        _require_finite(self.effective_rank, "effective_rank")
        _require_finite(self.retrieval_accuracy, "retrieval_accuracy")
        if self.acceptance not in ("pass", "fail"):
            raise ContractValueError("acceptance", "must be 'pass' or 'fail'")
        _require_finite(self.started_at_unix, "started_at_unix")
        _require_finite(self.wall_clock_seconds, "wall_clock_seconds")

    @property
    def identity(self) -> str:
        """Identify the *experiment*, not its outcome.

        Covers every input that determines what the run does: seed, code
        revision, environment, data selection, model configuration, and
        optimizer settings.

        Deliberately excluded:

        - wall-clock timing and the timestamped ``run_id`` — artifacts of when
          the run happened, not of what it was;
        - the measured metrics — CUDA float reductions are not bitwise
          reproducible across processes, so two runs of the *same* experiment
          differ around the 5th significant digit (measured on one identical
          pair: final loss 3.9408e-08 vs 3.9342e-08).  Including them would
          make the identity unable to answer the question it exists to answer.

        Compare ``identity`` for exact experiment identity, and compare the
        metrics numerically with a tolerance.
        """
        return identity(
            (
                self.manifest_version,
                self.mode,
                self.seed,
                self.git_commit,
                self.git_dirty,
                self.torch_version,
                self.cuda_version,
                self.device_name,
                self.dataset_root,
                self.split,
                self.catalog_identity,
                self.accepted_episode_count,
                self.rejected_episode_count,
                self.window_count,
                self.prediction_steps,
                self.stride_frames,
                self.abstraction,
                self.batch_size,
                self.steps,
                float(self.learning_rate),
                float(self.weight_decay),
                self.warmup_steps,
                float(self.grad_clip),
                float(self.ema_base_momentum),
                self.model_config_identity,
                self.sampled_index_identity,
                self.window_selection,
                self.candidate_count,
                self.symbolic_loss_active,
            )
        )

    def to_dict(self) -> RunManifestPayload:
        payload = RunManifestPayload(
            manifest_version=self.manifest_version,
            run_id=self.run_id,
            mode=self.mode,
            seed=self.seed,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            torch_version=self.torch_version,
            cuda_version=self.cuda_version,
            device_name=self.device_name,
            dataset_root=self.dataset_root,
            split=self.split,
            catalog_identity=self.catalog_identity,
            accepted_episode_count=self.accepted_episode_count,
            rejected_episode_count=self.rejected_episode_count,
            window_count=self.window_count,
            prediction_steps=self.prediction_steps,
            stride_frames=self.stride_frames,
            abstraction=self.abstraction,
            batch_size=self.batch_size,
            steps=self.steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            grad_clip=self.grad_clip,
            ema_base_momentum=self.ema_base_momentum,
            model_config_identity=self.model_config_identity,
            sampled_index_identity=self.sampled_index_identity,
            window_selection=self.window_selection,
            candidate_count=self.candidate_count,
            symbolic_loss_active=self.symbolic_loss_active,
            final_loss=self.final_loss,
            mean_feature_std=self.mean_feature_std,
            relative_spread=self.relative_spread,
            effective_rank=self.effective_rank,
            retrieval_accuracy=self.retrieval_accuracy,
            acceptance=self.acceptance,
            started_at_unix=self.started_at_unix,
            wall_clock_seconds=self.wall_clock_seconds,
            identity=self.identity,
        )
        return payload

    def write(self, path: str) -> None:
        """Persist the manifest as strict JSON (no NaN, no loose floats)."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def git_revision(root: str | None = None) -> tuple[str, bool]:
    """Return ``(commit, dirty)`` for the repository at ``root`` (or the cwd)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return ("unknown", True)
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        return (commit, True)
    return (commit, bool(status.strip()))


def capture_environment() -> dict:
    """Return the torch/CUDA/device identity used for a run.

    Every value is coerced to a plain ``str``: ``torch.__version__`` is a
    ``TorchVersion`` (a str subclass), and the manifest's validators use the
    strict ``type(x) is str`` idiom this package uses everywhere.
    """
    cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        return {
            "torch_version": str(torch.__version__),
            "cuda_version": cuda_version,
            "device_name": str(torch.cuda.get_device_name(device)),
        }
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_version,
        "device_name": "cpu",
    }
