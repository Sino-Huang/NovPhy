"""Issue #105: a three-level hierarchical latent predictor on the 236-value E99 carrier.

Hierarchical world models (H-JEPA, CW-VAE, MSPred, Director) tie each time scale to one level of
abstraction. This module is that baseline under the repo's request interface: level k predicts
only at horizon Delta_k with description alpha_k,

    level 0: (1, continuous)    level 1: (5, micro)    level 2: (15, macro)

i.e. the diagonal of the 3 x 3 request grid; every off-diagonal request is rejected. Each level
has its own trunk (input projection, FiLM blocks, output projection, residual carrier step).
Level 1 adds the micro head and adapter of ``CNNHybridPredictor``; level 2 adds the macro head
and adapter. The coupling is top-down (H-JEPA / CW-VAE style): the FiLM code of level k is a
linear read-out of the displacement that level k + 1 predicts from the same state at its own
(coarser) horizon and description; the top level uses a learned constant code. A level-0 step
therefore runs level 2, then level 1, then level 0. The model is stateless, so it keeps the
``carrier(z, action, pair)`` contract of ``CNNHybridPredictor`` and trains under the unchanged
#112 v3 ensemble, losses and scoring rollouts.

``WIDTH`` is the level width whose total parameter count is closest to ``CNNHybridPredictor``
(1,837,690): the matched-capacity rule of ``matched_dynamics.capacity_contract``.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from world_model.model import Abstraction, PredictionPair
from world_model.model.predictor import FiLMBlock
from world_model.training.cnn_hybrid import CONFIG, DIM, SLOTS, CNNHybridPredictor

LEVEL_PAIRS = (PredictionPair(1, Abstraction.CONTINUOUS), PredictionPair(5, Abstraction.MICRO),
               PredictionPair(15, Abstraction.MACRO))
LEVEL_OF = {pair: index for index, pair in enumerate(LEVEL_PAIRS)}
DEPTH = CONFIG.depth
CODE_DIM = CONFIG.pair_code_dim
MICRO_FEATURES = SLOTS * SLOTS * 2 + 2
MACRO_FEATURES = 4
WIDTH_RANGE = range(128, 385)


class Level(nn.Module):
    """One level's residual carrier step: [z | action] -> width -> FiLM blocks -> 236."""

    def __init__(self, width):
        super().__init__()
        self.input_projection = nn.Linear(DIM + 5, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, CODE_DIM) for _ in range(DEPTH))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, DIM)

    def step(self, z, action, code, extra=None):
        hidden = self.input_projection(torch.cat((z, action), -1))
        if extra is not None:
            hidden = hidden + extra
        for block in self.blocks:
            hidden = block(hidden, code)
        return z + self.output_projection(self.output_norm(hidden))


class HierarchicalPredictor(nn.Module):
    """Three levels tied to the diagonal requests, coupled top-down (see the module docstring)."""

    pairs = LEVEL_PAIRS

    def __init__(self, width=None):
        super().__init__()
        self.width = WIDTH if width is None else width
        self.levels = nn.ModuleList(Level(self.width) for _ in LEVEL_PAIRS)
        self.top_code = nn.Parameter(torch.zeros(CODE_DIM))
        self.top_down = nn.ModuleList(nn.Linear(DIM, CODE_DIM) for _ in LEVEL_PAIRS[:-1])
        self.micro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, MICRO_FEATURES))
        self.macro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, MACRO_FEATURES))
        self.micro_adapter = nn.Linear(MICRO_FEATURES, self.width, bias=False)
        self.macro_adapter = nn.Linear(MACRO_FEATURES, self.width, bias=False)

    # the hybrid's symbolic read-outs, bound to the level that owns the description
    symbols = CNNHybridPredictor.symbols
    symbolic_features = CNNHybridPredictor.symbolic_features

    def level_step(self, z, action, level):
        """Carrier after one step of ``level`` (its own Delta and alpha), with top-down context."""
        if level == len(LEVEL_PAIRS) - 1:
            code = self.top_code[None].expand(len(z), -1)
        else:
            code = self.top_down[level](self.level_step(z, action, level + 1) - z)
        mode = LEVEL_PAIRS[level].abstraction
        extra = None
        if mode == Abstraction.MICRO:
            extra = self.micro_adapter(self.symbolic_features(z, mode))
        elif mode == Abstraction.MACRO:
            extra = self.macro_adapter(self.symbolic_features(z, mode))
        return self.levels[level].step(z, action, code, extra)

    def carrier(self, z, action, pair):
        if pair not in LEVEL_OF or z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("the hierarchical predictor answers only its level requests (diagonal pairs)")
        return self.level_step(z, action, LEVEL_OF[pair])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def matched_width():
    """The level width whose parameter count is closest to CNNHybridPredictor's (ties: smaller)."""
    with torch.device("meta"):
        target = parameter_count(CNNHybridPredictor())
        counts = {w: parameter_count(HierarchicalPredictor(w)) for w in WIDTH_RANGE}
    return min(counts, key=lambda w: (abs(counts[w] - target), w)), target, counts


WIDTH = 202  # matched_width()[0]; asserted by the #105 smoke and tests


def _linear(module):
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


def level_macs(model, level):
    """Linear MACs of one ``level_step`` call, including the coarser levels it runs for its code."""
    count = _linear(model.levels[level])
    mode = LEVEL_PAIRS[level].abstraction
    if mode == Abstraction.MICRO:
        count += _linear(model.micro_head) + _linear(model.micro_adapter)
    elif mode == Abstraction.MACRO:
        count += _linear(model.macro_head) + _linear(model.macro_adapter)
    if level < len(LEVEL_PAIRS) - 1:
        count += _linear(model.top_down[level]) + level_macs(model, level + 1)
    return count


def linear_macs(model, pair):
    return level_macs(model, LEVEL_OF[pair])


@torch.no_grad()
def fixed_rollout(model, z0, actions, pair, macs_per_call, endpoint=225):
    """``m99.rollout_endpoints`` for a fixed request, op for op, for any ``carrier(z, action, pair)``
    model: the anchor carrier expanded over the candidate actions, residual steps of ``pair`` for every
    alive row; a row retires when nonfinite or at ``endpoint``. Returns the endpoint carriers and the
    transition / MAC accounting (``macs_per_call`` Linear MACs per transition)."""
    count = len(actions)
    z = z0[None].to(actions.device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    calls = 0
    if z.is_cuda:
        torch.cuda.synchronize()
    began = time.monotonic()
    while bool(alive.any()):
        rows = alive.nonzero(as_tuple=True)[0]
        z_next = z.clone()
        z_next[rows] = model.carrier(z[rows], actions[rows], pair)
        calls += len(rows)
        positions[rows] += pair.delta
        z = z_next
        alive = alive & torch.isfinite(z).all(dim=1) & (positions < endpoint)
    if z.is_cuda:
        torch.cuda.synchronize()
    return z, {"transition_calls": calls, "controller_calls": 0, "linear_macs": calls * macs_per_call,
               "gpu_seconds": time.monotonic() - began}


__all__ = ["LEVEL_PAIRS", "LEVEL_OF", "WIDTH", "HierarchicalPredictor", "matched_width", "parameter_count",
           "level_macs", "linear_macs", "fixed_rollout"]
