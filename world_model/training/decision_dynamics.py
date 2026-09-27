"""Issue #112: long-horizon and decision-contrastive training of request-conditioned predictors.

An ``Ensemble`` holds independent ``CNNHybridPredictor`` / ``ContinuousDynamics`` members
(own initialization, data pool, generators, AdamW state and gradient clip). Members are
stacked only so one vmapped transition call serves all of them; every loss below returns
one value per member and no member's value depends on another member's parameters.

Losses (leading member axis M everywhere):

- ``local_loss``: the #74/#77 recipe (``cnn_hybrid.training_loss`` with ``corrected=True``
  for the hybrid family, ``matched_dynamics.continuous_loss`` for the continuous family)
  in masked form. Boolean row selection becomes row weights, so it runs under vmap; the
  value equals the reference implementation.
- ``long_and_rank_loss``: one recursion of the step's pair that serves two terms.
  Long-horizon: rows start at a window start carrier and are unrolled ``horizon // delta``
  transitions; every visited position is supervised by the parsed carrier there
  (terminal absorption past the shot's last frame) plus the #77 carrier-bound penalty.
  Ranking: rows start at a member anchor carrier, each candidate action is unrolled to
  the decision endpoint, and a pairwise logistic loss on the tie-free cost (the #111
  ``tie_free`` definition on tensors) orders engine-successful candidates below (lower
  cost than) failed ones inside one anchor.
"""
from __future__ import annotations

import copy
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.planning.scoring_harness import tie_free
from world_model.training.matched_dynamics import pairs_for

BOUND_WEIGHT = 0.01
BOUND_LIMIT = 2.0
SYMBOLIC_WEIGHT = 0.1
SYMBOLIC_POS_WEIGHT = 5.0
LOCAL_STEPS = 4
DISPLACEMENT_EPS = 1e-12

_BASES: dict = {}
_FUNCTIONS: dict = {}
# every (pair, row count, member count) is its own graph; one shared code object serves all
# of them, so the per-code-object recompile limit must exceed that product
RECOMPILE_LIMIT = 1024


class _Transition(nn.Module):
    def __init__(self, model: nn.Module, pair: PredictionPair) -> None:
        super().__init__()
        self.m = model
        self.pair = pair

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.m.carrier(z, action, self.pair)


class _Symbols(nn.Module):
    def __init__(self, model: nn.Module, mode: Abstraction) -> None:
        super().__init__()
        self.m = model
        self.mode = mode

    def forward(self, z: torch.Tensor):
        return self.m.symbols(z, self.mode)


class Ensemble:
    """Independent members trained in lock-step; one AdamW over stacked parameters is the
    per-member AdamW (every AdamW operation is elementwise), and gradients are clipped per
    member to ``grad_clip`` (the ``clip_grad_norm_`` rule on that member's parameters).

    A member whose loss or gradient norm is nonfinite is retired at that update (a typed
    terminal failure): its gradient is zeroed from then on and it is never reported as
    trained."""

    def __init__(self, models: Sequence[nn.Module], *, learning_rate: float, weight_decay: float,
                 grad_clip: float, compile: bool = True) -> None:
        if not models:
            raise ValueError("an ensemble needs at least one member")
        kinds = {type(m) for m in models}
        if len(kinds) != 1:
            raise ValueError("ensemble members must share one architecture")
        params, buffers = stack_module_state(list(models))
        if buffers:
            raise ValueError("stacked members must not carry buffers")
        self.size = len(models)
        self.device = next(models[0].parameters()).device
        self.params = {name: value.detach().clone().requires_grad_() for name, value in params.items()}
        self._prefixed = {f"m.{name}": value for name, value in self.params.items()}
        self.signature = (type(models[0]).__name__, tuple((k, tuple(v.shape[1:])) for k, v in params.items()))
        if self.signature not in _BASES:
            _BASES[self.signature] = copy.deepcopy(models[0]).to("meta")
        self.base = _BASES[self.signature]
        self.pairs = pairs_for(models[0])
        self.hybrid = hasattr(models[0], "symbols")
        self.grad_clip = grad_clip
        self.compile = compile
        self.optimizer = torch.optim.AdamW(list(self.params.values()), lr=learning_rate, weight_decay=weight_decay)
        self.retired = torch.zeros(self.size, dtype=torch.bool, device=self.device)
        self.retired_at = [None] * self.size

    def _function(self, kind: str, key, module_factory):
        """Vmapped (and compiled) call; one cache per architecture so later ensembles of the
        same architecture reuse the compiled graphs (parameters are always arguments)."""
        cache_key = (self.signature, kind, key, self.compile)
        if cache_key not in _FUNCTIONS:
            module = module_factory()

            def call(params, *args):
                return functional_call(module, params, args)
            fn = vmap(call, in_dims=(0,) + (0,) * (1 if kind == "symbols" else 2))
            if self.compile:
                import torch._dynamo
                torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
                torch._dynamo.config.accumulated_recompile_limit = max(
                    torch._dynamo.config.accumulated_recompile_limit, 4 * RECOMPILE_LIMIT)
            _FUNCTIONS[cache_key] = torch.compile(fn) if self.compile else fn
        return _FUNCTIONS[cache_key]

    def transition(self, pair: PredictionPair, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z [M, rows, D], action [M, rows, 5] -> [M, rows, D]."""
        if pair not in self.pairs:
            raise ValueError(f"pair {pair} is not a trained pair of this family")
        return self._function("transition", pair, lambda: _Transition(self.base, pair))(self._prefixed, z, action)

    def symbols(self, mode: Abstraction, z: torch.Tensor):
        return self._function("symbols", mode, lambda: _Symbols(self.base, mode))(self._prefixed, z)

    def update(self, loss: torch.Tensor) -> dict:
        """One AdamW update from per-member losses [M]; returns per-member diagnostics."""
        if loss.shape != (self.size,):
            raise ValueError("update expects one loss per member")
        finite = torch.isfinite(loss.detach())
        active = finite & ~self.retired
        self.optimizer.zero_grad(set_to_none=True)
        torch.where(active, loss, torch.zeros_like(loss)).sum().backward()
        # parameters a pair never reads keep grad None, exactly as in single-model training
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
        self.optimizer.step()
        newly = ~active & ~self.retired
        self.retired |= newly
        return {"active": active, "newly_retired": newly, "grad_norm": norms.detach()}

    def mark_retired_step(self, step: int, newly: torch.Tensor) -> None:
        for index in newly.nonzero(as_tuple=True)[0].tolist():
            self.retired_at[index] = step

    def member_state(self, index: int) -> dict:
        return {name: value[index].detach().cpu().clone() for name, value in self.params.items()}

    def state(self) -> dict:
        return {"params": {k: v.detach().cpu() for k, v in self.params.items()},
                "optimizer": self.optimizer.state_dict(), "retired": self.retired.cpu(),
                "retired_at": list(self.retired_at)}

    def load_state(self, state: Mapping) -> None:
        with torch.no_grad():
            for name, value in self.params.items():
                value.copy_(state["params"][name].to(value.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self.retired = state["retired"].to(self.device).clone()
        self.retired_at = list(state["retired_at"])


def _row_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """values, weight [M, rows] -> weighted row mean [M] (0 where no row carries weight)."""
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1)


def symbolic_term(ensemble: Ensemble, z: torch.Tensor, batch: Mapping, position: int, keep: torch.Tensor,
                  mode: Abstraction) -> torch.Tensor:
    """``cnn_hybrid.symbolic_loss`` on the kept rows, per member."""
    if mode == Abstraction.CONTINUOUS:
        return torch.zeros(len(z), device=z.device)
    logits, availability = ensemble.symbols(mode, z)
    name = "relations" if mode == Abstraction.MICRO else "macros"
    labels = batch[name][:, :, position].float()
    mask = batch[name + "_mask"][:, :, position].bool()
    available = (mask.any(2).any(2) if mode == Abstraction.MICRO else mask).float()
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none",
                                             pos_weight=torch.tensor(SYMBOLIC_POS_WEIGHT, device=z.device))
    weight = mask.float() * keep.view(keep.shape + (1,) * (mask.ndim - 2))
    labelled = (raw * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1)
    availability_loss = F.binary_cross_entropy_with_logits(availability, available, reduction="none").mean(-1)
    return labelled + _row_mean(availability_loss, keep)


def local_loss(ensemble: Ensemble, batch: Mapping, pair: PredictionPair) -> torch.Tensor:
    """The #74/#77 loss (four local transitions, their recursion, carrier bound; the hybrid
    family adds the symbolic losses) on 60-transition windows. batch: z [M, B, 61, D],
    action [M, B, 5], length [M, B]; hybrid adds relations/macros and their masks."""
    z, action, length = batch["z"], batch["action"], batch["length"]
    symbolic = ensemble.hybrid and pair.abstraction != Abstraction.CONTINUOUS
    local_terms, recursive_terms, valid_steps = [], [], []
    current = None
    for step in range(LOCAL_STEPS):
        start, end = step * pair.delta, (step + 1) * pair.delta
        if end >= z.shape[2]:
            break
        keep = (length >= end).to(z.dtype)
        valid_steps.append((keep.sum(1) > 0).to(z.dtype))
        local = ensemble.transition(pair, z[:, :, start], action)
        target = z[:, :, end]
        term = _row_mean((local - target).square().mean(-1), keep)
        if symbolic:
            term = term + SYMBOLIC_WEIGHT * symbolic_term(ensemble, local, batch, end, keep, pair.abstraction)
            term = term + SYMBOLIC_WEIGHT * symbolic_term(ensemble, z[:, :, start], batch, start, keep,
                                                          pair.abstraction)
        local_terms.append(term)
        current = local if step == 0 else ensemble.transition(pair, current, action)
        recursive_terms.append(_row_mean((current - target).square().mean(-1), keep)
                               + BOUND_WEIGHT * _row_mean(F.relu(current.abs() - BOUND_LIMIT).square().mean(-1),
                                                          keep))
    valid = torch.stack(valid_steps)
    steps = valid.sum(0).clamp_min(1)
    return ((torch.stack(local_terms) * valid).sum(0) / steps
            + (torch.stack(recursive_terms) * valid).sum(0) / steps)


def tie_free_cost(z_end: torch.Tensor, z_start: torch.Tensor, presence: int, center: Sequence[int]) -> torch.Tensor:
    """Differentiable tie-free cost of endpoint carriers against their start carriers."""
    dx = z_end[..., center[0]] - z_start[..., center[0]]
    dy = z_end[..., center[1]] - z_start[..., center[1]]
    return tie_free(z_end[..., presence], (dx.square() + dy.square() + DISPLACEMENT_EPS).sqrt())


def pairwise_ranking(cost: torch.Tensor, verdict: torch.Tensor, temperature: float) -> torch.Tensor:
    """cost [M, C, P]; verdict [M, C, P] in {1 success, 0 failure, -1 padding}.
    Mean over the cells of the mean over (success, failure) pairs of
    softplus((cost_success - cost_failure) / temperature); a cell without both verdicts
    carries no weight."""
    success = (verdict == 1).to(cost.dtype)
    failure = (verdict == 0).to(cost.dtype)
    weight = success[..., :, None] * failure[..., None, :]
    pairs = F.softplus((cost[..., :, None] - cost[..., None, :]) / temperature)
    per_cell = (pairs * weight).sum((-1, -2)) / weight.sum((-1, -2)).clamp_min(1)
    cells = (weight.sum((-1, -2)) > 0).to(cost.dtype)
    return (per_cell * cells).sum(-1) / cells.sum(-1).clamp_min(1)


def long_and_rank_loss(ensemble: Ensemble, pair: PredictionPair, *, long: Mapping | None, rank: Mapping | None,
                       horizon: int, endpoint: int, presence: int, center: Sequence[int],
                       temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One fused recursion of ``pair``.

    long: start [M, BL, D], action [M, BL, 5], targets [M, BL, K, D] with K = horizon // delta
    (the parsed carrier at start + j delta, j = 1..K). rank: z0 [M, C, D], action
    [M, C, P, 5], verdict [M, C, P]; every candidate is unrolled endpoint // delta
    transitions. Returns (long loss [M], ranking loss [M]); an absent term is 0."""
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
        z = ensemble.transition(pair, z, action)
        if step < long_steps:
            head = z[:, :long_rows]
            error = (head - long["targets"][:, :, step]).square().mean(-1)
            bound = F.relu(head.abs() - BOUND_LIMIT).square().mean(-1)
            long_sum = long_sum + (error + BOUND_WEIGHT * bound).mean(1)
    long_loss = long_sum / long_steps if long_steps else zero
    if rank is None:
        return long_loss, zero
    cells, candidates = rank["verdict"].shape[1:]
    z_end = z[:, -cells * candidates:].reshape(members, cells, candidates, -1)
    cost = tie_free_cost(z_end, rank["z0"][:, :, None], presence, center)
    return long_loss, pairwise_ranking(cost, rank["verdict"], temperature)


def horizon_at(step: int, *, first: int, last: int, increment: int, every: int) -> int:
    """Curriculum: ``first`` frames, growing by ``increment`` every ``every`` updates, capped at ``last``."""
    return min(last, first + increment * (step // every))


def ramp_at(step: int, *, start: int, length: int) -> float:
    """0 before ``start``, then linear to 1 over ``length`` updates."""
    if step < start:
        return 0.0
    return min(1.0, (step - start + 1) / length) if length else 1.0


__all__ = ["Ensemble", "local_loss", "long_and_rank_loss", "pairwise_ranking", "tie_free_cost", "symbolic_term",
           "horizon_at", "ramp_at"]
