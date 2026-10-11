"""Issue #117: interference between the requests that share one predictor.

The #77 / #112 schedule trains one request per update, round-robin over the nine hybrid
pairs (``PAIRS[step % 9]``). Requests therefore interact through the shared weights and the
shared AdamW moments between updates, not through one summed multi-task gradient. This
module holds the data-free pieces of the measurement:

- ``schedule``: the update sequence of an arm that trains a subset of the nine pairs.
  Under the per-request budget the arm makes exactly the joint run's updates of its pairs,
  each at the joint step that sets the horizon curriculum and the ranking ramp. Under the
  full budget it makes the joint run's update count, cycling its pairs, with the joint
  schedule indexed by its own update count.
- ``trunk_names``: the parameters that every request reads (the conditioner's fuse MLP,
  the input projection, the FiLM blocks and the output projection). It excludes the
  per-description embedding row and the micro / macro heads and adapters.
- ``request_gram`` / ``cosines``: per-member Gram matrix of per-request gradients on the
  trunk, from which per-request gradient norms and pairwise cosines follow.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

STEPS = 9000
CYCLE = 9
BUDGETS = ("request", "full")
TRUNK_PREFIXES = ("conditioner.fuse.", "input_projection.", "blocks.", "output_norm.", "output_projection.")


def schedule(pair_indices: Sequence[int], budget: str, *, steps: int = STEPS,
             cycle: int = CYCLE) -> list[tuple[int, int]]:
    """[(pair index, joint step)] per update. The joint step drives the curriculum and ramps."""
    chosen = sorted(set(pair_indices))
    if not chosen or any(not 0 <= index < cycle for index in chosen):
        raise ValueError("an arm trains a non-empty subset of the cycle's pairs")
    if budget == "request":
        return [(step % cycle, step) for step in range(steps) if step % cycle in chosen]
    if budget == "full":
        return [(chosen[update % len(chosen)], update) for update in range(steps)]
    raise ValueError(f"unknown budget {budget!r}")


def trunk_names(names: Sequence[str]) -> list[str]:
    return [name for name in names if name.startswith(TRUNK_PREFIXES)]


def request_gram(gradients: Sequence[Mapping[str, torch.Tensor | None]], names: Sequence[str]) -> torch.Tensor:
    """gradients: one mapping per request, parameter name -> stacked gradient [M, ...] (None:
    the request does not reach that parameter, a zero gradient). Returns the float64 Gram
    matrix [M, R, R] of the per-request gradients restricted to ``names``."""
    present = [g[name] for g in gradients for name in names if g.get(name) is not None]
    if not present:
        raise ValueError("no request reaches the named parameters")
    members, device = present[0].shape[0], present[0].device
    gram = torch.zeros(members, len(gradients), len(gradients), dtype=torch.float64, device=device)
    for name in names:
        parts = [g.get(name) for g in gradients]
        reference = next((p for p in parts if p is not None), None)
        if reference is None:
            continue
        stacked = torch.stack([torch.zeros_like(reference) if p is None else p for p in parts])
        flat = stacked.flatten(2).double()
        gram += torch.einsum("rmn,smn->mrs", flat, flat)
    return gram


def cosines(gram: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarities from a Gram matrix [..., R, R]; NaN where a norm is 0."""
    norms = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(0).sqrt()
    denominator = norms[..., :, None] * norms[..., None, :]
    return torch.where(denominator > 0, gram / denominator.clamp_min(1e-300), torch.full_like(gram, float("nan")))


__all__ = ["STEPS", "CYCLE", "BUDGETS", "TRUNK_PREFIXES", "schedule", "trunk_names", "request_gram", "cosines"]
