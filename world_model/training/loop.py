"""Teacher-forced single-step JEPA training loop.

One optimizer step: encode the context online, encode the target through the
EMA/stop-grad branch, predict the carrier from ``(z_t, a_t, (Delta, alpha))``,
and regress ``z_hat`` to the detached ``z*_{t+Delta}``.  Nothing symbolic is
optimized here — the symbolic heads exist and are tested, but the legacy RGB
cohort carries no labels, so only the carrier MSE trains.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence as SequenceABC
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from world_model.data.catalog import EpisodeCatalog
from world_model.data.curriculum import catalog_digest
from world_model.data.dataset import TemporalWindowDataset
from world_model.data.sampling import EpochSampler, TemporalWindowCollator
from world_model.data.types import ContractValueError, TemporalWindowRequest
from world_model.model import (
    Abstraction,
    EncoderConfig,
    JepaBackbone,
    JepaConfig,
    PredictionPair,
    coerce_abstraction,
    digest,
)
from world_model.training.grid_data import MotionRegime
from world_model.training.pair_grid import APPROVED_PAIRS
from world_model.training.diagnostics import (
    CollapseReport,
    collapse_diagnostics,
)
from world_model.training.manifest import (
    MANIFEST_VERSION,
    RunManifest,
    capture_environment,
    git_revision,
)

OVERFIT_LOSS_THRESHOLD = 1e-3
# Spread is judged relative to the representation's own scale: an absolute
# threshold is meaningless because the encoder can shrink z and let the
# predictor compensate.
OVERFIT_SPREAD_THRESHOLD = 1e-3
# Centring an N-row batch caps the effective rank at N-1, and a healthy JEPA
# latent legitimately compresses: on 8 distinct dev scenes the raw images have
# centred rank 6.87 while the learned representation reaches 2.80 with perfect
# 8/8 retrieval.  Compression is not collapse.  The threshold therefore only
# has to exclude the degenerate regimes (a true collapse measures ~1.0), and
# retrieval carries the real discriminative weight.
OVERFIT_RANK_THRESHOLD = 2.0
OVERFIT_RETRIEVAL_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization configuration for one teacher-forced run."""

    seed: int = 20260807
    steps: int = 2000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    grad_clip: float = 1.0
    ema_base_momentum: float = 0.996
    delta: int = 1
    abstraction: Abstraction | str = Abstraction.CONTINUOUS
    device: str = "cuda"
    grid_schedule: bool = False

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ContractValueError("seed", "must be a nonnegative integer")
        if type(self.steps) is not int or self.steps <= 0:
            raise ContractValueError("steps", "must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ContractValueError("batch_size", "must be a positive integer")
        if self.learning_rate <= 0.0:
            raise ContractValueError("learning_rate", "must be positive")
        if self.weight_decay < 0.0:
            raise ContractValueError("weight_decay", "must be nonnegative")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ContractValueError("warmup_steps", "must be a nonnegative integer")
        if self.warmup_steps > self.steps:
            raise ContractValueError("warmup_steps", "must not exceed steps")
        if not 0.0 <= self.grad_clip <= 1e3:
            raise ContractValueError("grad_clip", "must lie in [0, 1000]")
        if not 0.0 <= self.ema_base_momentum <= 1.0:
            raise ContractValueError("ema_base_momentum", "must lie in the unit interval")
        if type(self.delta) is not int or self.delta <= 0:
            raise ContractValueError("delta", "must be a positive integer")
        object.__setattr__(
            self, "abstraction", coerce_abstraction(self.abstraction)
        )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ContractValueError("device", "must be a nonempty string")
        if type(self.grid_schedule) is not bool:
            raise ContractValueError("grid_schedule", "must be a boolean")


@dataclass(frozen=True, slots=True)
class StepResult:
    """One training step's observables."""

    step: int
    loss: float
    learning_rate: float
    momentum: float
    unweighted_loss: float | None = None
    weighted_loss: float | None = None
    pair: PredictionPair | None = None
    motion_regime: MotionRegime | None = None


@dataclass(frozen=True, slots=True)
class OverfitReport:
    """The evidence a Milestone 1 overfit run produces."""

    run_dir: Path
    manifest: RunManifest
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    diagnostics: CollapseReport
    acceptance: str


def seed_all(seed: int) -> None:
    """Seed every RNG the loop touches, for bitwise reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_transform(input_height: int, input_width: int) -> nn.Module:
    """Return a transform that resizes one [3, H, W] frame to the encoder input."""

    class _Resize(nn.Module):
        def __init__(self, height: int, width: int) -> None:
            super().__init__()
            self.height = int(height)
            self.width = int(width)

        def forward(self, frame: torch.Tensor) -> torch.Tensor:
            if not isinstance(frame, torch.Tensor):
                raise ContractValueError("frame", "must be a torch tensor")
            if frame.ndim != 3:
                raise ContractValueError("frame", "must have shape [C, H, W]")
            if frame.shape[0] != 3:
                raise ContractValueError("frame", "must carry three channels")
            return F.interpolate(
                frame.unsqueeze(0),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)

    return _Resize(input_height, input_width)


def select_motion_windows(
    dataset: TemporalWindowDataset,
    *,
    seed: int,
    window_count: int,
    candidate_count: int,
) -> tuple[int, ...]:
    """Return the ``window_count`` most dynamic windows from a seeded candidate pool.

    Ranks a seeded candidate pool by context-to-target motion.  Measured on dev:
    the median window's motion is ~2.5e-5 and no Delta=1 window in 400 draws
    exceeds 1e-3, so this lifts the subset out of the quiescent bulk — but it
    does *not* guarantee the windows are distinct from each other, which is what
    an overfit demo actually needs.  Prefer ``select_diverse_windows``.
    """
    if type(window_count) is not int or window_count <= 0:
        raise ContractValueError("window_count", "must be a positive integer")
    if type(candidate_count) is not int or candidate_count < window_count:
        raise ContractValueError(
            "candidate_count", "must be an integer of at least window_count"
        )
    sampler = EpochSampler(
        dataset, seed=seed, draw_count=min(candidate_count, len(dataset))
    )
    scored: list[tuple[float, int]] = []
    for index in sampler:
        sample = dataset[index]
        motion = float(
            (sample["context_image"] - sample["target_images"][0]).pow(2).mean()
        )
        scored.append((motion, index))
    # Sort by motion descending, then by index for a total, stable order.
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return tuple(index for _, index in scored[:window_count])


def select_diverse_windows(
    dataset: TemporalWindowDataset,
    *,
    seed: int,
    window_count: int,
    candidate_count: int,
) -> tuple[int, ...]:
    """Return ``window_count`` windows drawn from distinct episodes.

    This is the selection an overfit demo needs, and the reason is measured, not
    assumed.  Windows drawn uniformly (or ranked by motion) from the dev cohort
    are frequently near-duplicates: their target embeddings land 7.4e-05 apart,
    so "is each prediction nearest to its own target" is a coin flip no matter
    how good the predictor is.  One window per episode gives visually distinct
    scenes (pairwise image L2 >= 20) and makes retrieval a real test — the same
    backbone scores 8/8 there and 1/8 on near-duplicates.

    Selection walks the seeded candidate order and keeps the first window of
    each unseen episode, so it stays deterministic and recorded.  Episode
    identity is read from the dataset's materialized window index, so no image
    is decoded during selection.
    """
    if type(window_count) is not int or window_count <= 0:
        raise ContractValueError("window_count", "must be a positive integer")
    if type(candidate_count) is not int or candidate_count < window_count:
        raise ContractValueError(
            "candidate_count", "must be an integer of at least window_count"
        )
    window_index = object.__getattribute__(dataset, "_index")
    sampler = EpochSampler(
        dataset, seed=seed, draw_count=min(candidate_count, len(dataset))
    )
    chosen: list[int] = []
    seen_episodes: set[str] = set()
    for index in sampler:
        episode = window_index[index].episode.name
        if episode in seen_episodes:
            continue
        seen_episodes.add(episode)
        chosen.append(index)
        if len(chosen) == window_count:
            return tuple(chosen)
    raise ContractValueError(
        "candidate_count",
        f"only {len(chosen)} distinct episodes in the pool, need {window_count}",
    )


def build_window_loader(
    catalog: EpisodeCatalog,
    *,
    encoder_config: EncoderConfig,
    delta: int,
    batch_size: int,
    seed: int,
    draw_count: int,
    window_selection: str = "uniform",
    candidate_count: int = 256,
) -> tuple[torch.utils.data.DataLoader, int, str]:
    """Build a deterministic single-step loader over a catalog snapshot.

    Returns ``(loader, window_count, sampled_index_digest)`` where
    ``window_count`` is the number of eligible ``(steps=1, stride=delta)``
    windows in the catalog and ``sampled_index_digest`` fixes exactly which
    indices the run draws (the reproducibility identity of the data).

    ``window_selection`` is ``"uniform"`` (seeded draw as-is) or ``"motion"``
    (the most dynamic windows from a seeded candidate pool, see
    ``select_motion_windows``).  The digest always covers the final order, so
    the selection is reproducible regardless of mode.
    """
    if window_selection not in ("uniform", "motion", "diverse"):
        raise ContractValueError(
            "window_selection", "must be 'uniform', 'motion', or 'diverse'"
        )
    request = TemporalWindowRequest(prediction_steps=1, stride_frames=delta)
    dataset = TemporalWindowDataset(
        catalog,
        request,
        transform=resize_transform(
            encoder_config.input_height, encoder_config.input_width
        ),
    )
    if window_selection == "motion":
        order = select_motion_windows(
            dataset, seed=seed, window_count=draw_count, candidate_count=candidate_count
        )
    elif window_selection == "diverse":
        order = select_diverse_windows(
            dataset, seed=seed, window_count=draw_count, candidate_count=candidate_count
        )
    else:
        sampler = EpochSampler(dataset, seed=seed, draw_count=draw_count)
        order = tuple(sampler)
    # The chosen order IS the sampler: a sequence of dataset indices is a valid
    # PyTorch sampler, so the loader yields exactly these windows in exactly
    # this order.  (Wrapping the dataset in a Subset and sampling *that* would
    # silently yield indices 0..N-1 of the original dataset instead.)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=list(order),
        collate_fn=TemporalWindowCollator(),
        num_workers=0,
    )
    return loader, len(dataset), digest(("sampled-window-indices-v1", seed, order))


class TeacherForcedTrainer:
    """One atomic train step over a batch from the lazy window reader."""

    def __init__(self, backbone: JepaBackbone, config: TrainingConfig) -> None:
        if type(backbone) is not JepaBackbone:
            raise ContractValueError("backbone", "must be a JepaBackbone")
        if type(config) is not TrainingConfig:
            raise ContractValueError("training config", "must be a TrainingConfig")
        self._backbone = backbone
        self._config = config
        self.pair = PredictionPair(delta=config.delta, abstraction=config.abstraction)
        self.device = torch.device(config.device)

        parameters = list(backbone.trainable_parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self._step_count = 0
        backbone.to(self.device)

    @property
    def backbone(self) -> JepaBackbone:
        return self._backbone

    @property
    def config(self) -> TrainingConfig:
        return self._config

    def _learning_rate(self, step: int) -> float:
        config = self._config
        if config.warmup_steps > 0 and step < config.warmup_steps:
            factor = (step + 1) / config.warmup_steps
        else:
            progress = min(step, config.steps - 1) / max(1, config.steps - 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.learning_rate * factor

    def _momentum_at(self, step: int) -> float:
        """Cosine-ramped EMA momentum, driven by the run's training config.

        The run config is the source of truth here — it is what the manifest
        records — so the overfit driver can use a faster target (higher
        ``ema_base_momentum``) than the default backbone schedule.
        """
        base = self._config.ema_base_momentum
        progress = min(step, self._config.steps) / self._config.steps
        return 1.0 - (1.0 - base) * (math.cos(math.pi * progress) + 1.0) / 2.0

    def schedule_at(self, step: int) -> tuple[PredictionPair, MotionRegime]:
        if type(step) is not int or step < 0:
            raise ContractValueError("step", "must be a nonnegative integer")
        keys = [
            (PredictionPair(delta, Abstraction.CONTINUOUS), regime)
            for delta in (1, 5, 15)
            for regime in MotionRegime
        ]
        cycle = step // len(keys)
        offset = step % len(keys)
        generator = random.Random(self._config.seed)
        for _ in range(cycle + 1):
            generator.shuffle(keys)
        return keys[offset]

    def encode_target(self, batch: dict) -> torch.Tensor:
        """Return the detached target latent for a single-step batch."""
        return self._backbone.encode_target(self._validate_batch(batch)).latent

    def train_step(self, batch: dict) -> StepResult:
        """Optimize one teacher-forced step and return its observables."""
        config = self._config
        step = self._step_count
        target_images, pair, regime, weights = self._prepare_batch(batch, step)
        context = batch["context_image"].to(self.device)
        action = batch["action"].to(self.device)

        latent = self._backbone.encode(context).latent
        target_latent = self._backbone.encode_target(target_images).latent
        prediction = self._backbone.predict(latent, action, pair).carrier
        per_example = (prediction - target_latent).pow(2).mean(dim=1)
        unweighted_loss = per_example.mean()
        weighted_loss = (per_example * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(
            self._backbone.trainable_parameters(), config.grad_clip
        )
        learning_rate = self._learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.step()
        # The run config drives the schedule, so the reported momentum is the
        # momentum actually applied — the backbone's own default schedule would
        # silently disagree whenever the run overrides the base momentum.
        momentum = self._momentum_at(step)
        self._backbone.target.update(self._backbone.encoder, momentum=momentum)
        self._step_count = step + 1

        return StepResult(
            step=step,
            loss=float(unweighted_loss.detach().item()),
            learning_rate=learning_rate,
            momentum=float(momentum),
            unweighted_loss=float(unweighted_loss.detach().item()),
            weighted_loss=float(weighted_loss.detach().item()),
            pair=pair,
            motion_regime=regime,
        )

    def _prepare_batch(
        self, batch: dict, step: int
    ) -> tuple[torch.Tensor, PredictionPair, MotionRegime | None, torch.Tensor]:
        if not isinstance(batch, dict):
            raise ContractValueError("batch", "must be a mapping")
        target_images = batch.get("target_images")
        if not isinstance(target_images, torch.Tensor):
            raise ContractValueError("target_images", "must be a torch tensor")
        if target_images.ndim != 5 or target_images.shape[1] != 1:
            raise ContractValueError(
                "target_images",
                "the single-step loop requires exactly one target per window",
            )
        for key in ("context_image", "action"):
            if not isinstance(batch.get(key), torch.Tensor):
                raise ContractValueError(key, "must be a torch tensor")
        batch_size = target_images.shape[0]
        pair_value = batch.get("prediction_pair", batch.get("pair"))
        regime_value = batch.get("motion_regime")
        if self._config.grid_schedule:
            scheduled_pair, scheduled_regime = self.schedule_at(step)
            if pair_value is None:
                pair_value = scheduled_pair
            elif pair_value != scheduled_pair:
                raise ContractValueError("prediction_pair", "does not match the seeded grid schedule")
            if regime_value is None:
                regime_value = scheduled_regime
            elif self._coerce_regime(regime_value) is not scheduled_regime:
                raise ContractValueError("motion_regime", "does not match the seeded grid schedule")
        elif pair_value is None:
            pair_value = self.pair

        if type(pair_value) is not PredictionPair:
            raise ContractValueError("prediction_pair", "must be a PredictionPair")
        if pair_value not in APPROVED_PAIRS and pair_value != self.pair:
            raise ContractValueError("prediction_pair", "must use an approved continuous pair")
        regime = self._coerce_regime(regime_value) if regime_value is not None else None
        grid_metadata = self._config.grid_schedule or any(
            key in batch
            for key in (
                "prediction_pair",
                "pair",
                "motion_regime",
                "shot_frame_count",
                "frame_count",
                "frame_indices",
            )
        )
        if not grid_metadata:
            weights = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            return target_images[:, 0].to(self.device), pair_value, regime, weights

        frame_counts = batch.get("shot_frame_count", batch.get("frame_count"))
        frame_indices = batch.get("frame_indices")
        if not isinstance(frame_counts, torch.Tensor) or frame_counts.ndim != 1:
            raise ContractValueError("shot_frame_count", "must be a tensor of one frame count per example")
        if frame_counts.shape[0] != batch_size:
            raise ContractValueError("shot_frame_count", "must match the batch size")
        if frame_indices is None or not isinstance(frame_indices, SequenceABC) or len(frame_indices) != batch_size:
            raise ContractValueError("frame_indices", "must contain context and target positions per example")
        prediction_steps = batch.get("prediction_steps")
        if prediction_steps is not None:
            if not isinstance(prediction_steps, torch.Tensor) or prediction_steps.ndim != 1 or prediction_steps.shape[0] != batch_size:
                raise ContractValueError("prediction_steps", "must be one-dimensional and match the batch size")
            if not torch.all(prediction_steps == pair_value.delta):
                raise ContractValueError("prediction_steps", "must equal prediction_pair.delta")
        counts = frame_counts.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(counts).all() or torch.any(counts <= 1):
            raise ContractValueError("shot_frame_count", "must be greater than one")
        for index, frames in enumerate(frame_indices):
            if not isinstance(frames, SequenceABC) or len(frames) < 2:
                raise ContractValueError("frame_indices", "must contain context and target positions")
            context_position, target_position = frames[0], frames[1]
            if type(context_position) is not int or type(target_position) is not int:
                raise ContractValueError("frame_indices", "positions must be integers")
            if context_position < 0 or target_position <= context_position:
                raise ContractValueError("frame_indices", "target must follow context")
            if target_position - context_position != pair_value.delta:
                raise ContractValueError("frame_indices", "target delta does not match prediction_pair.delta")
            if target_position >= int(frame_counts[index].item()):
                raise ContractValueError("frame_indices", "target lies outside the shot")
        return target_images[:, 0].to(self.device), pair_value, regime, pair_value.delta / (counts - 1.0)

    @staticmethod
    def _coerce_regime(value: object) -> MotionRegime:
        if isinstance(value, MotionRegime):
            return value
        try:
            return MotionRegime(value)
        except (TypeError, ValueError) as error:
            raise ContractValueError("motion_regime", "must be a valid MotionRegime") from error

    def _validate_batch(self, batch: dict) -> torch.Tensor:
        return self._prepare_batch(batch, self._step_count)[0]


def run_overfit(
    catalog: EpisodeCatalog,
    *,
    jepa_config: JepaConfig,
    training_config: TrainingConfig,
    window_count: int,
    output_dir: Path,
    window_selection: str = "diverse",
    candidate_count: int = 4096,
) -> OverfitReport:
    """Overfit a fixed subset of windows and record the anti-collapse evidence.

    Every step trains on the same ``window_count`` windows.  Acceptance requires
    a near-zero final loss AND a non-collapsed representation (per-dim spread,
    effective rank, and retrieval of each prediction to its own target).

    ``window_selection`` chooses how the subset is drawn:

    - ``"motion"`` (default) ranks a seeded candidate pool by inter-frame motion
      and keeps the most dynamic windows.  The legacy cohort is action-sparse,
      so a uniform draw is almost always near-identical frame pairs on which a
      zero loss is achievable by learning the identity — which would make the
      overfit evidence vacuous.
    - ``"uniform"`` takes the seeded sampler's draw as-is.  Kept so the
      degenerate baseline stays reproducible and comparable.
    """
    if type(window_count) is not int or window_count <= 0:
        raise ContractValueError("window_count", "must be a positive integer")
    if window_selection not in ("motion", "uniform", "diverse"):
        raise ContractValueError(
            "window_selection", "must be 'motion', 'uniform', or 'diverse'"
        )
    seed_all(training_config.seed)
    started_at_unix = time.time()

    loader, window_count_actual, index_digest = build_window_loader(
        catalog,
        encoder_config=jepa_config.encoder,
        delta=training_config.delta,
        batch_size=window_count,
        seed=training_config.seed,
        draw_count=window_count,
        window_selection=window_selection,
        candidate_count=candidate_count,
    )
    backbone = JepaBackbone(jepa_config)
    trainer = TeacherForcedTrainer(backbone, training_config)

    loss_history: list[float] = []
    first_loss: float | None = None
    final_batch = None
    for _ in range(training_config.steps):
        batch = next(iter(loader))
        result = trainer.train_step(batch)
        loss_history.append(result.loss)
        if first_loss is None:
            first_loss = result.loss
        final_batch = batch
    assert final_batch is not None
    assert first_loss is not None

    with torch.no_grad():
        predictions = backbone.predict(
            backbone.encode(final_batch["context_image"].to(trainer.device)).latent,
            final_batch["action"].to(trainer.device),
            trainer.pair,
        ).carrier
        targets = backbone.encode_target(
            final_batch["target_images"][:, 0].to(trainer.device)
        ).latent
        final_loss = float(F.mse_loss(predictions, targets).item())
        diagnostics = collapse_diagnostics(predictions, targets)
    acceptance = "pass" if (
        final_loss < OVERFIT_LOSS_THRESHOLD
        and diagnostics.relative_spread > OVERFIT_SPREAD_THRESHOLD
        and diagnostics.effective_rank >= OVERFIT_RANK_THRESHOLD
        and diagnostics.retrieval_accuracy >= OVERFIT_RETRIEVAL_THRESHOLD
    ) else "fail"

    run_id = _run_id(training_config)
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment()
    commit, dirty = git_revision(str(Path(__file__).resolve().parents[2]))
    manifest = RunManifest(
        manifest_version=MANIFEST_VERSION,
        run_id=run_id,
        mode="overfit",
        seed=training_config.seed,
        git_commit=commit,
        git_dirty=dirty,
        torch_version=environment["torch_version"],
        cuda_version=environment["cuda_version"],
        device_name=environment["device_name"],
        dataset_root=str(_catalog_root(catalog)),
        split=catalog.split,
        catalog_digest=catalog_digest(catalog),
        accepted_episode_count=len(catalog.episodes),
        rejected_episode_count=catalog.rejection_count,
        window_count=window_count_actual,
        prediction_steps=1,
        stride_frames=training_config.delta,
        abstraction=str(training_config.abstraction),
        batch_size=window_count,
        steps=training_config.steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        grad_clip=training_config.grad_clip,
        ema_base_momentum=training_config.ema_base_momentum,
        model_config_digest=jepa_config.identity,
        sampled_index_digest=index_digest,
        window_selection=window_selection,
        candidate_count=candidate_count,
        symbolic_loss_active=False,
        final_loss=final_loss,
        mean_feature_std=diagnostics.mean_feature_std,
        relative_spread=diagnostics.relative_spread,
        effective_rank=diagnostics.effective_rank,
        retrieval_accuracy=diagnostics.retrieval_accuracy,
        acceptance=acceptance,
        started_at_unix=started_at_unix,
        wall_clock_seconds=time.time() - started_at_unix,
    )
    manifest.write(str(run_dir / "manifest.json"))

    return OverfitReport(
        run_dir=run_dir,
        manifest=manifest,
        initial_loss=first_loss,
        final_loss=final_loss,
        loss_history=tuple(loss_history),
        diagnostics=diagnostics,
        acceptance=acceptance,
    )


def _run_id(config: TrainingConfig) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-jepa-overfit-seed{config.seed}"


def _catalog_root(catalog: EpisodeCatalog) -> Path:
    """Return the catalog's dataset root (EpisodeCatalog stores it privately)."""
    return Path(object.__getattribute__(catalog, "_root"))
