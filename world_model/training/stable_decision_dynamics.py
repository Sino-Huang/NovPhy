"""Issue #112 v3: numerically stable long-horizon / decision-contrastive training.

v1 (``decision_dynamics``) retired 28 of 312 members, all on Delta = 1 updates late in the
horizon curriculum: back-propagation through up to 225 recursive transitions gave gradient
norms of 1e3 - 1e4 (the #77 local term: about 0.5) and occasionally overflowed. v1 is
hash-bound by its frozen plan, so the two stability changes live here:

- ``truncated_long_and_rank_loss``: ``decision_dynamics.long_and_rank_loss`` with the
  recursion detached after every ``truncation`` transitions, so no gradient path is longer
  than ``truncation`` transitions. Forward values (and therefore every loss value) are
  unchanged; with ``truncation`` at least the number of transitions it equals v1 exactly.
- ``SkippingEnsemble``: a member whose loss or gradient norm is nonfinite skips that update
  (its parameters and AdamW moments are left exactly as they were) instead of being retired;
  it is retired only after ``patience`` consecutive skipped updates. Skips are counted.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from world_model.model import PredictionPair
from world_model.training import decision_dynamics as dd


def truncated_long_and_rank_loss(ensemble: dd.Ensemble, pair: PredictionPair, *, long: Mapping | None,
                                 rank: Mapping | None, horizon: int, endpoint: int, presence: int,
                                 center: Sequence[int], temperature: float,
                                 truncation: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``dd.long_and_rank_loss`` with truncated back-propagation (see the module docstring)."""
    if truncation < 1:
        raise ValueError("truncation must be at least one transition")
    members = ensemble.size
    zero = torch.zeros(members, device=ensemble.device)
    long_steps = horizon // pair.delta if long is not None else 0
    rank_steps = endpoint // pair.delta if rank is not None else 0
    if long is not None and (horizon % pair.delta or long["targets"].shape[2] != long_steps):
        raise ValueError("long-horizon targets must hold every delta multiple up to the horizon")
    if rank is not None and endpoint % pair.delta:
        raise ValueError("the decision endpoint must be a multiple of the pair horizon")
    states, actions = [], []
    long_rows = 0
    if long is not None:
        states.append(long["start"])
        actions.append(long["action"])
        long_rows = long["start"].shape[1]
    if rank is not None:
        cells, candidates = rank["verdict"].shape[1:]
        start = rank["z0"][:, :, None].expand(-1, -1, candidates, -1).reshape(members, cells * candidates, -1)
        states.append(start)
        actions.append(rank["action"].reshape(members, cells * candidates, -1))
    z, action = torch.cat(states, 1), torch.cat(actions, 1)
    long_sum = zero
    for step in range(max(long_steps, rank_steps)):
        if step == long_steps and long_rows and rank is not None:
            z, action, long_rows = z[:, long_rows:], action[:, long_rows:], 0
        if step and step % truncation == 0:
            z = z.detach()
        z = ensemble.transition(pair, z, action)
        if step < long_steps:
            head = z[:, :long_rows]
            error = (head - long["targets"][:, :, step]).square().mean(-1)
            bound = F.relu(head.abs() - dd.BOUND_LIMIT).square().mean(-1)
            long_sum = long_sum + (error + dd.BOUND_WEIGHT * bound).mean(1)
    long_loss = long_sum / long_steps if long_steps else zero
    if rank is None:
        return long_loss, zero
    cells, candidates = rank["verdict"].shape[1:]
    z_end = z[:, -cells * candidates:].reshape(members, cells, candidates, -1)
    cost = dd.tie_free_cost(z_end, rank["z0"][:, :, None], presence, center)
    return long_loss, dd.pairwise_ranking(cost, rank["verdict"], temperature)


class SkippingEnsemble(dd.Ensemble):
    """``dd.Ensemble`` whose nonfinite updates are skipped per member (see the module docstring)."""

    patience = 10

    def __init__(self, models, **kwargs) -> None:
        super().__init__(models, **kwargs)
        self.skipped = torch.zeros(self.size, dtype=torch.long, device=self.device)
        self.consecutive = torch.zeros(self.size, dtype=torch.long, device=self.device)

    def update(self, loss: torch.Tensor) -> dict:
        if loss.shape != (self.size,):
            raise ValueError("update expects one loss per member")
        active = torch.isfinite(loss.detach()) & ~self.retired
        self.optimizer.zero_grad(set_to_none=True)
        torch.where(active, loss, torch.zeros_like(loss)).sum().backward()
        grads = [p.grad for p in self.params.values() if p.grad is not None]
        squares = torch.zeros(self.size, device=self.device)
        for grad in grads:
            squares = squares + torch.nan_to_num(grad, nan=float("inf")).flatten(1).square().sum(1)
        norms = squares.sqrt()
        active = active & torch.isfinite(norms)
        coefficient = torch.where(active, (self.grad_clip / (norms + 1e-6)).clamp(max=1.0), torch.zeros_like(norms))
        for grad in grads:
            shape = (-1,) + (1,) * (grad.ndim - 1)
            grad.copy_(torch.where(active.view(shape), grad * coefficient.view(shape), torch.zeros_like(grad)))
        frozen = (~active).nonzero(as_tuple=True)[0]
        saved = None
        if len(frozen):
            # per-member slices of the parameters and AdamW moments; the shared scalar step count
            # still advances (bias correction differs by < 1e-3 after the first thousand updates)
            saved = [(p, p.detach()[frozen].clone(),
                      {k: v[frozen].clone() for k, v in self.optimizer.state.get(p, {}).items()
                       if torch.is_tensor(v) and v.ndim and v.shape[0] == self.size})
                     for p in self.params.values()]
        self.optimizer.step()
        if saved is not None:
            with torch.no_grad():
                for p, value, moments in saved:
                    p[frozen] = value
                    for key, moment in moments.items():
                        self.optimizer.state[p][key][frozen] = moment
        skipped = ~active & ~self.retired
        self.skipped += skipped.long()
        self.consecutive = torch.where(skipped, self.consecutive + 1, torch.zeros_like(self.consecutive))
        newly = skipped & (self.consecutive >= self.patience)
        self.retired |= newly
        return {"active": active, "skipped": skipped, "newly_retired": newly, "grad_norm": norms.detach()}

    def state(self) -> dict:
        return {**super().state(), "skipped": self.skipped.cpu(), "consecutive": self.consecutive.cpu()}

    def load_state(self, state: Mapping) -> None:
        super().load_state(state)
        self.skipped = state["skipped"].to(self.device).clone()
        self.consecutive = state["consecutive"].to(self.device).clone()


__all__ = ["truncated_long_and_rank_loss", "SkippingEnsemble"]
