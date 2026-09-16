"""Outcome readout and losses for the source-bound #76 event-ranking stage."""

import torch
from torch import nn
from torch.nn import functional as F


STOP_KINDS = ("native_clear", "native_fail", "stable_without_clear", "right_censored")
CLEAR_INDEX = STOP_KINDS.index("native_clear")


class EventReadout(nn.Module):
    """Predict an observed stop kind from one permitted dynamics transition."""

    def __init__(self, state_dim: int, action_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(STOP_KINDS)),
        )

    def forward(self, initial, successor, action):
        if initial.shape != successor.shape or initial.ndim != 2:
            raise ValueError("event readout requires aligned rank-two state tensors")
        if action.shape != (len(initial), 5):
            raise ValueError("event readout requires one five-component action per state")
        return self.layers(torch.cat((initial, successor, action), dim=-1))


def event_loss(logits, labels, class_weights, *, paired_ranking: bool):
    """Weighted stop-kind loss plus a within-lineage clear-ranking term."""

    if logits.ndim != 2 or logits.shape[1] != len(STOP_KINDS):
        raise ValueError("event logits have the wrong class inventory")
    if labels.shape != (len(logits),) or class_weights.shape != (len(STOP_KINDS),):
        raise ValueError("event labels or class weights are misaligned")
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    clear = labels == CLEAR_INDEX
    if paired_ranking and bool(clear.any()):
        scores = logits[:, CLEAR_INDEX]
        loss = loss + torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[clear], dim=0)
    return loss


@torch.no_grad()
def pair_teacher(pair_logits, labels):
    """Choose the pair with the smallest observed-class loss for each record."""

    if pair_logits.ndim != 3 or pair_logits.shape[2] != len(STOP_KINDS):
        raise ValueError("pair teacher requires record-by-pair stop-kind logits")
    if labels.shape != (len(pair_logits),):
        raise ValueError("pair teacher labels are misaligned")
    losses = -pair_logits.log_softmax(-1).gather(
        2, labels[:, None, None].expand(-1, pair_logits.shape[1], 1)
    ).squeeze(-1)
    return losses.argmin(1)


def second_fraction(counts):
    """Fraction assigned to the second-most-used choice."""

    values = sorted((int(value) for value in counts.values()), reverse=True)
    return 0.0 if len(values) < 2 or sum(values) == 0 else values[1] / sum(values)
