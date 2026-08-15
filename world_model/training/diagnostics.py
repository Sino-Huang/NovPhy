"""Anti-collapse diagnostics for a teacher-forced JEPA overfit run.

A teacher-forced MSE over a stop-grad target can reach zero by representation
collapse — a constant encoder produces a constant prediction and zero loss.  The
metrics here are what make a near-zero loss meaningful evidence.

Two measurement decisions matter, and both were forced by real data:

- **Centred.**  NovPhy frames share a large constant background, so the raw
  images have an *uncentred* effective rank of ~1.15 and any encoder of them
  does too.  Uncentred rank therefore measures "how much of the signal is the
  shared mean", which is ~everything, and would report collapse for a perfectly
  healthy representation (measured: uncentred 1.03 vs centred 5.35 on the same
  fresh encoder).  Collapse is about *variation across samples*, so the spectrum
  is taken after removing the batch mean.
- **Scale-relative.**  An absolute spread threshold is meaningless because the
  latent's scale is free — the encoder can shrink ``z`` and grow the predictor to
  compensate.  Spread is reported relative to the representation's own norm.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model.data.types import ContractValueError


@dataclass(frozen=True, slots=True)
class CollapseReport:
    """The diagnostics that qualify a near-zero loss as real evidence."""

    mean_feature_std: float
    relative_spread: float
    effective_rank: float
    retrieval_accuracy: float

    def to_dict(self) -> dict:
        return {
            "mean_feature_std": self.mean_feature_std,
            "relative_spread": self.relative_spread,
            "effective_rank": self.effective_rank,
            "retrieval_accuracy": self.retrieval_accuracy,
        }


def mean_feature_std(latents: torch.Tensor) -> float:
    """Mean over dimensions of each dimension's spread across the batch."""
    return float(latents.std(dim=0).mean())


def relative_spread(latents: torch.Tensor) -> float:
    """Batch spread as a fraction of the representation's own scale.

    ``||z - mean(z)||_F / ||z||_F``.  Scale-invariant, so it cannot be gamed by
    shrinking the latent and growing the predictor to compensate.
    """
    scale = float(latents.norm())
    if scale == 0.0:
        return 0.0
    return float((latents - latents.mean(dim=0, keepdim=True)).norm()) / scale


def effective_rank(latents: torch.Tensor, *, centred: bool = True) -> float:
    """Spectral-entropy rank ``exp(H(s))`` over the singular values.

    Centred by default: collapse means the samples do not vary, not that they
    lack a common component.  Pass ``centred=False`` only to inspect the raw
    spectrum (a constant-but-nonzero matrix then scores 1.0 rather than 0.0).
    """
    if latents.numel() == 0:
        return 0.0
    matrix = latents.float()
    if centred:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    if total == 0.0:
        return 0.0
    probabilities = singular_values / total
    # Zero singular values contribute nothing to the entropy; the explicit
    # guard keeps 0 * log(0) from becoming NaN.
    log_probabilities = torch.where(
        probabilities > 0.0,
        torch.log(probabilities.clamp_min(1e-30)),
        torch.zeros_like(probabilities),
    )
    entropy = -(probabilities * log_probabilities).sum()
    return float(torch.exp(entropy))


def retrieval_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of predictions whose nearest target is their own target."""
    distances = torch.cdist(predictions, targets)
    nearest = distances.argmin(dim=1)
    self_match = (nearest == torch.arange(predictions.shape[0], device=predictions.device)).sum()
    return float(self_match) / predictions.shape[0]


def collapse_diagnostics(predictions: torch.Tensor, targets: torch.Tensor) -> CollapseReport:
    """Compute all diagnostics for an overfit run's final batch."""
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ContractValueError("latents", "must be torch tensors")
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ContractValueError("latents", "must be two-dimensional")
    if predictions.shape != targets.shape:
        raise ContractValueError("latents", "predictions and targets must match in shape")
    if predictions.shape[0] < 2:
        raise ContractValueError("latents", "at least two rows are required")
    return CollapseReport(
        mean_feature_std=mean_feature_std(predictions),
        relative_spread=relative_spread(predictions),
        effective_rank=effective_rank(predictions),
        retrieval_accuracy=retrieval_accuracy(predictions, targets),
    )
