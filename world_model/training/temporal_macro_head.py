"""Issue #113 item 3: a macro head with temporal context for the observable macro predicates.

Interface (declared before training; it is the same for engine-parsed and predicted carriers):

    head(history, delta) -> logits [B, 2]  (observable_macro.PREDICATES at that Delta)

``history`` is [B, K, 236]: ``history[:, -1]`` is the carrier being read at the landing frame f
(an engine-parsed carrier, or a predicted one), and ``history[:, -1 - j]`` is the carrier at frame
f - j * Delta, clamped at frame 0 (the decision frame absorbs earlier positions). The history
stride is the request's own Delta, so a Delta-step rollout supplies it natively: at the n-th
predicted landing a + n * Delta of a rollout anchored at frame a, the history holds the rollout's
earlier landings and, before a, the observed context carriers (``rollout_history``). K = 1 is the
single-frame head (the landing carrier only).

Features: the landing carrier and the K - 1 stride-Delta motion differences
(z_f - z_{f-Delta}, z_{f-Delta} - z_{f-2 Delta}, ...), standardised with training statistics and
clipped to +-CLIP (terminal frames of early-ending shots carry very large motion values). One MLP
per Delta: Linear(K * 236, HIDDEN) -> SiLU -> Linear(HIDDEN, 2); HIDDEN = 128 is the width of the
current single-frame ``CNNHybridPredictor.macro_head``.

Loss: class-balanced binary cross-entropy (pos_weight = negatives / positives per predicate and
Delta over the training rows), masked by label availability.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

DIM = 236
DELTAS = (1, 5, 15)
PREDICATES = 2
HIDDEN = 128
CLIP = 10.0


class TemporalMacroHead(nn.Module):
    def __init__(self, history, hidden=HIDDEN, deltas=DELTAS):
        super().__init__()
        if history < 1:
            raise ValueError("history must hold at least the landing carrier")
        self.history = history
        self.deltas = tuple(deltas)
        width = history * DIM
        self.register_buffer("mean", torch.zeros(len(self.deltas), width))
        self.register_buffer("std", torch.ones(len(self.deltas), width))
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, PREDICATES))
                                   for _ in self.deltas)

    @staticmethod
    def raw_features(history):
        """[B, K, D] -> [B, K * D]: the landing carrier and the K - 1 stride-Delta differences."""
        if history.ndim != 3 or history.shape[-1] != DIM:
            raise ValueError("history must be [B, K, 236]")
        parts = [history[:, -1]]
        if history.shape[1] > 1:
            parts.append((history[:, 1:] - history[:, :-1]).flip(1).flatten(1))
        return torch.cat(parts, -1)

    def features(self, history, delta):
        if history.shape[1] != self.history:
            raise ValueError(f"head reads K = {self.history} carriers, got {history.shape[1]}")
        d = self.deltas.index(delta)
        return ((self.raw_features(history) - self.mean[d]) / self.std[d]).clamp(-CLIP, CLIP)

    def forward(self, history, delta):
        return self.heads[self.deltas.index(delta)](self.features(history, delta))


def history_positions(positions, delta, history):
    """[N] in-shot landing positions -> [N, K] in-shot positions f - (K - 1 - j) * Delta, clamped at 0."""
    offsets = torch.arange(history - 1, -1, -1, device=positions.device) * delta
    return (positions[:, None] - offsets[None, :]).clamp_min(0)


def gather_history(carriers, starts, positions, delta, history):
    """Engine-parsed history from concatenated shot carriers: carriers [N_total, D], starts [N] (row of
    each sample's shot position 0), positions [N] -> [N, K, D]."""
    return carriers[starts[:, None] + history_positions(positions, delta, history)]


def rollout_history(context, landings, delta, history):
    """History at the last landing of a stride-Delta rollout.

    context: [B, C, D] observed carriers at frames a - (C - 1) * Delta .. a (stride Delta, ending at
    the anchor a; repeat the earliest frame for positions before frame 0); landings: [B, n, D] the
    predicted carriers at a + Delta .. a + n * Delta. Returns [B, K, D] for the landing a + n * Delta."""
    chain = torch.cat((context, landings), 1)
    if chain.shape[1] < history:
        chain = torch.cat((chain[:, :1].expand(-1, history - chain.shape[1], -1), chain), 1)
    return chain[:, -history:]


def fit_normalisation(head, samples):
    """Set the per-Delta standardisation from training feature samples {delta: [N, K * D]}."""
    with torch.no_grad():
        for d, delta in enumerate(head.deltas):
            x = samples[delta].double()
            head.mean[d] = x.mean(0).float()
            head.std[d] = x.std(0).clamp_min(1e-6).float()


def balanced_loss(logits, values, available, pos_weight):
    """Masked class-balanced BCE: logits/values/available [B, 2], pos_weight [2]."""
    raw = F.binary_cross_entropy_with_logits(logits, values.float(), reduction="none", pos_weight=pos_weight)
    mask = available.float()
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def pos_weights(values, available):
    """negatives / positives per predicate over available rows ([N, 2] each) -> [2]."""
    positives = (values & available).sum(0).double()
    negatives = (~values & available).sum(0).double()
    return (negatives / positives.clamp_min(1)).float()
