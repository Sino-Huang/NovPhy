"""Request-policy input variants of #119: the remaining-frames feature and an elapsed-frames channel.

``request_policy.py`` is hash-pinned by later plans, so the #119 additions live here. The policy input
is ``[z (236) | action (5) | time features]``; the variants change only the time features:

- ``a``: ``min(kappa, 60) / 60`` (ICLR, ``request_policy`` ``clamp60``; the replication control);
- ``b``: ``min(kappa, 225) / 225`` (``request_policy`` ``over225`` = kappa / 225, equal on every path
  here because kappa <= 225: windows end at most 225 frames after their start and rollouts at 225);
- ``c``: ``log1p(kappa) / log1p(225)`` (``request_policy`` ``log1p``);
- ``d``: ``a`` plus the elapsed-frames channel ``min(k, 225) / 225`` (``ElapsedRequestPolicy``).

kappa is the frames to the end of the window or rollout and k the frames since its start. Every
window here starts at shot frame 0 and every scoring / continuation rollout starts at the anchor
(position 0) and ends at ``ENDPOINT``; so in a rollout k = ENDPOINT - kappa (the default of
``ElapsedRequestPolicy.forward``) and in a window k is passed explicitly (window rows carry an
``elapsed`` column).

Variants a-c are ``request_policy.RequestPolicy`` and train through ``request_policy.window_rows`` /
``distil_round`` unchanged (bit for bit the #101 path); the helpers below dispatch on the policy class
and add the elapsed column / argument only where variant d needs it.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import DIM

ENDPOINT = rp.ENDPOINT
ELAPSED_HORIZON = rp.ENDPOINT
VARIANTS = {
    "a": {"remaining_feature": "clamp60", "elapsed": False, "text": "min(kappa, 60) / 60 (ICLR control)"},
    "b": {"remaining_feature": "over225", "elapsed": False, "text": "min(kappa, 225) / 225"},
    "c": {"remaining_feature": "log1p", "elapsed": False, "text": "log1p(kappa) / log1p(225)"},
    "d": {"remaining_feature": "clamp60", "elapsed": True,
          "text": "min(kappa, 60) / 60 and elapsed frames min(k, 225) / 225"},
}


class ElapsedRequestPolicy(nn.Module):
    """``[z | action | min(kappa, 60) / 60 | min(k, 225) / 225] -> Linear-SiLU-Linear -> |R|``; logits
    with Delta > kappa are -inf (``RequestPolicy`` semantics). ``elapsed=None`` means a rollout from
    position 0 to ``horizon``: k = horizon - kappa."""

    def __init__(self, pairs, width: int = 128, horizon: int = ELAPSED_HORIZON):
        super().__init__()
        self.pairs = tuple(pairs)
        self.remaining_feature = "clamp60"
        self.horizon = horizon
        self.layers = nn.Sequential(nn.Linear(DIM+5+2, width), nn.SiLU(), nn.Linear(width, len(self.pairs)))

    def forward(self, z, action, remaining, elapsed=None):
        if z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("the request policy accepts only carrier, action, remaining and elapsed time")
        elapsed = self.horizon-remaining if elapsed is None else elapsed
        x = torch.cat((z, action, rp.REMAINING_FEATURES["clamp60"][1](remaining, z),
                       elapsed.reshape(-1, 1).to(z).clamp(max=self.horizon)/self.horizon), -1)
        horizons = torch.tensor([p.delta for p in self.pairs], device=z.device)
        return self.layers(x).masked_fill(horizons[None] > remaining[:, None], -torch.inf)


def uses_elapsed(policy):
    return isinstance(policy, ElapsedRequestPolicy)


def build_policy(variant, pairs, width=128):
    spec = VARIANTS[variant]
    if spec["elapsed"]:
        return ElapsedRequestPolicy(pairs, width)
    return rp.RequestPolicy(pairs, width, spec["remaining_feature"])


def new_policy(variant, pairs, seed, *, width=128, device="cuda"):
    """The #77 initialization (``torch.manual_seed(seed + 7700)`` then construct); variants a-c are
    exactly ``request_policy.new_policy``."""
    spec = VARIANTS[variant]
    if not spec["elapsed"]:
        return rp.new_policy(pairs, seed, width=width, remaining_feature=spec["remaining_feature"], device=device)
    torch.manual_seed(seed+rp.INIT_OFFSET)
    return ElapsedRequestPolicy(pairs, width).to(device)


def variant_of(policy):
    if uses_elapsed(policy):
        return "d"
    return next(v for v, s in VARIANTS.items() if not s["elapsed"] and s["remaining_feature"] == policy.remaining_feature)


def call(policy, z, action, remaining, elapsed=None):
    """Logits; the elapsed argument reaches only an elapsed-channel policy."""
    if uses_elapsed(policy):
        return policy(z, action, remaining, elapsed)
    return policy(z, action, remaining)


# ---------------------------------------------------------------------------
# prediction-error DP rows (request_policy.window_rows plus the elapsed column)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dagger_contexts(model, policy, z, action, length):
    """``request_policy.dagger_contexts`` with the elapsed frames t passed to the policy."""
    contexts = z.clone()
    current, t = z[:, 0], 0
    while t < int(length[0]):
        pair = policy.pairs[int(call(policy, current, action, length-t, torch.full_like(length, t)).argmax(-1))]
        current = model.carrier(current, action, pair)
        t += pair.delta
        contexts[:, t] = current
    return contexts


def window_rows(model, policy, windows, *, aggregated, compute_weight=rp.ICLR_COMPUTE_WEIGHT, device="cuda"):
    """#77 ``label_lineage`` rows {z, action, remaining, labels, elapsed}. Variants a-c: exactly
    ``request_policy.window_rows`` plus ``elapsed`` = frames since the window start; variant d: the
    same DP with the DAgger rollout fed the elapsed frames."""
    if not uses_elapsed(policy):
        rows = rp.window_rows(model, policy, windows, aggregated=aggregated, compute_weight=compute_weight,
                              device=device)
        rows["elapsed"] = torch.cat([torch.arange(int(w["length"])) for w in windows])
        return rows
    records = []
    for window in windows:
        z = window["z"][None].to(device)
        a = window["action"].reshape(1, 5).to(device)
        length = torch.as_tensor(window["length"]).reshape(1).to(device)
        contexts = dagger_contexts(model, policy, z, a, length) if aggregated else z.clone()
        labels = rp.prediction_error_labels(model, policy, z, a, length, contexts, compute_weight=compute_weight)
        n = int(length[0])
        records.append({"z": contexts[0, :n].cpu(), "action": a.expand(n, -1).cpu(),
                        "remaining": torch.arange(n, 0, -1), "labels": labels[0, :n].cpu(),
                        "elapsed": torch.arange(n)})
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}


ROW_KEYS = ("z", "action", "remaining", "labels", "elapsed")


def aggregate_rounds(history):
    """``request_policy.aggregate_rounds`` keeping the elapsed column."""
    return [{k: torch.cat([rounds[u][k] for rounds in history]) for k in ROW_KEYS} for u in range(len(history[0]))]


def decision_elapsed(rows):
    """Decision rows start at the anchor: elapsed = position."""
    return {**rows, "elapsed": rows["position"].clone()}


def distil_round(policy, rows, round_index, *, seed, steps=rp.DISTIL_RECIPE["steps_per_round"],
                 batch_size=rp.DISTIL_RECIPE["batch_size"], learning_rate=rp.DISTIL_RECIPE["learning_rate"],
                 device="cuda"):
    """Variants a-c: ``request_policy.distil_round`` (the extra column is moved, never read; the sampler
    draws the same indices). Variant d: the same loop with the elapsed column as the policy input."""
    if not uses_elapsed(policy):
        return rp.distil_round(policy, rows, round_index, seed=seed, steps=steps, batch_size=batch_size,
                               learning_rate=learning_rate, device=device)
    optim = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed+rp.INIT_OFFSET+round_index)
    losses = []
    for step in range(steps):
        batch = rp.sample(rows[step % len(rows)], batch_size, generator, device)
        loss = F.cross_entropy(policy(batch["z"], batch["action"], batch["remaining"], batch["elapsed"]),
                               batch["labels"])
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite policy distillation loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    return losses


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

KAPPA_BINS = ((166, 225), (106, 165), (61, 105), (31, 60), (1, 30))


def kappa_bin(kappa):
    """Index into ``KAPPA_BINS`` (inclusive bounds) of a remaining-frames value 1..225."""
    for index, (low, high) in enumerate(KAPPA_BINS):
        if low <= kappa <= high:
            return index
    raise ValueError(f"remaining frames {kappa} outside 1..{ENDPOINT}")


@torch.no_grad()
def label_accuracy(policy, rows, *, chunk=65536, device="cuda"):
    """Argmax agreement of ``policy`` with the DP labels of ``rows`` (rows {z, action, remaining,
    labels[, elapsed]}; the elapsed column is needed only by an elapsed-channel policy): overall and
    per ``KAPPA_BINS`` bin of the row's remaining frames."""
    hits = torch.zeros(len(KAPPA_BINS), dtype=torch.long)
    totals = torch.zeros(len(KAPPA_BINS), dtype=torch.long)
    edges = torch.tensor([low for low, _ in KAPPA_BINS])
    for begin in range(0, len(rows["labels"]), chunk):
        part = {k: rows[k][begin:begin+chunk] for k in ROW_KEYS if k in rows}
        elapsed = part["elapsed"].to(device) if "elapsed" in part else None
        logits = call(policy, part["z"].to(device), part["action"].to(device), part["remaining"].to(device), elapsed)
        hit = (logits.argmax(-1).cpu() == part["labels"]).double()
        index = (part["remaining"][:, None] < edges[None]).sum(1)
        hits += torch.bincount(index, weights=hit, minlength=len(KAPPA_BINS)).long()
        totals += torch.bincount(index, minlength=len(KAPPA_BINS))
    return {"rows": int(totals.sum()), "accuracy": float(hits.sum()/totals.sum()),
            "by_kappa": [{"kappa": list(b), "rows": int(n), "accuracy": float(h/n) if n else None}
                         for b, h, n in zip(KAPPA_BINS, hits.tolist(), totals.tolist())]}


def kappa_mix(sequences, pairs, endpoint=ENDPOINT):
    """Request counts per ``KAPPA_BINS`` bin of the true remaining frames at each step: a rollout from
    position 0 takes request ``s[j]`` with kappa = endpoint - (sum of the earlier Deltas)."""
    counts = [[0]*len(pairs) for _ in KAPPA_BINS]
    for sequence in sequences:
        position = 0
        for index in sequence:
            counts[kappa_bin(endpoint-position)][index] += 1
            position += pairs[index].delta
    return counts


__all__ = ["ENDPOINT", "ELAPSED_HORIZON", "VARIANTS", "ElapsedRequestPolicy", "uses_elapsed", "build_policy",
           "new_policy", "variant_of", "call", "dagger_contexts", "window_rows", "ROW_KEYS", "aggregate_rounds",
           "decision_elapsed", "distil_round", "KAPPA_BINS", "kappa_bin", "label_accuracy", "kappa_mix"]
