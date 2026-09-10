"""CNN-carrier hybrid readiness, not the historical oracle-symbol experiment.

Only the 236-value carrier persists. Selected-mode symbols are decoded afresh
from that carrier; engine labels are accepted only by training functions.
"""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair, PredictorConfig
from world_model.model.predictor import PairConditioner, FiLMBlock


PAIRS = tuple(PredictionPair(h, mode) for h in (1, 5, 15) for mode in Abstraction)
ARCHITECTURE = "issue-71-cnn-carrier-soft-ordered-predicates-v1"
DIM = 236
SLOTS = 18
CONFIG = PredictorConfig(latent_dim=DIM, hidden_dim=384, depth=3)


class CNNHybridPredictor(nn.Module):
    """Reuse the shared FiLM trunk, with fixed-slot soft predicate adapters.

The dense ordering replaces legacy variable-ID graph construction. Contact is
symmetrized; support is directed. No legacy event/duration head is instantiated:
the deployed macro interface is the two declared state predicates only.
"""

    def __init__(self, config=CONFIG):
        super().__init__()
        if config.latent_dim != DIM or config.action_dim != 5:
            raise ValueError("issue-71 requires the unchanged 236-value CNN carrier")
        self.config = config
        self.conditioner = PairConditioner(config)
        self.input_projection = nn.Linear(DIM + 5, config.hidden_dim)
        self.blocks = nn.ModuleList(FiLMBlock(config.hidden_dim, config.pair_code_dim)
                                    for _ in range(config.depth))
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, DIM)
        self.micro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(),
                                        nn.Linear(128, SLOTS * SLOTS * 2 + 2))
        self.macro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, 4))
        self.micro_adapter = nn.Linear(SLOTS * SLOTS * 2 + 2, config.hidden_dim, bias=False)
        self.macro_adapter = nn.Linear(4, config.hidden_dim, bias=False)

    def symbols(self, z, mode):
        if mode == Abstraction.MICRO:
            raw = self.micro_head(z)
            relations = raw[:, :-2].reshape(-1, SLOTS, SLOTS, 2)
            contact = (relations[..., 0] + relations[..., 0].transpose(1, 2)) / 2
            relations = torch.stack((contact, relations[..., 1]), -1)
            return relations, raw[:, -2:]
        if mode == Abstraction.MACRO:
            raw = self.macro_head(z)
            return raw[:, :2], raw[:, 2:]
        raise ValueError("continuous mode has no symbolic readout")

    def symbolic_features(self, z, mode):
        logits, availability = self.symbols(z, mode)
        available = availability.sigmoid()
        values = logits.sigmoid()
        if mode == Abstraction.MICRO:
            presence = z[:, 2::13].clamp(0, 1)
            active = presence[:, :, None] * presence[:, None, :]
            active = active * (1 - torch.eye(SLOTS, device=z.device))
            values = values * active[..., None] * available[:, None, None, :]
        else:
            values = values * available
        return torch.cat((values.flatten(1), available), -1)

    def carrier(self, z, action, pair):
        if pair not in PAIRS or z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("invalid trained pair/carrier/action contract")
        hidden = self.input_projection(torch.cat((z, action), -1))
        if pair.abstraction == Abstraction.MICRO:
            hidden = hidden + self.micro_adapter(self.symbolic_features(z, pair.abstraction))
        elif pair.abstraction == Abstraction.MACRO:
            hidden = hidden + self.macro_adapter(self.symbolic_features(z, pair.abstraction))
        code = self.conditioner.code(pair, len(z), z.device)
        for block in self.blocks:
            hidden = block(hidden, code)
        return z + self.output_projection(self.output_norm(hidden))


def symbolic_loss(model, z, batch, positions, mode):
    if mode == Abstraction.CONTINUOUS:
        return z.sum() * 0
    logits, availability = model.symbols(z, mode)
    rows = torch.arange(len(z), device=z.device)
    name = "relations" if mode == Abstraction.MICRO else "macros"
    labels = batch[name][rows, positions].float()
    mask = batch[name + "_mask"][rows, positions].bool()
    availability_labels = (mask.any(1).any(1) if mode == Abstraction.MICRO else mask).float()
    # Fixed, modest imbalance weight; never fitted using held-out outcomes.
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none",
                                             pos_weight=torch.tensor(5., device=z.device))
    return ((raw * mask).sum() / mask.sum().clamp_min(1)
            + F.binary_cross_entropy_with_logits(availability, availability_labels))


def training_loss(model, batch, pair, corrected):
    """Both arms see the identical four local transitions and symbolic labels.

    Only the corrected arm adds recursive endpoint and carrier-bound losses.
    Short terminal windows are masked, not relabeled as a full requested step.
    """
    z, a = batch["z"], batch["action"]
    valid = batch["length"]
    current = z[:, 0]
    losses, recursive = [], []
    for step in range(4):
        start, end = step * pair.delta, (step + 1) * pair.delta
        keep = valid >= end
        if not bool(keep.any()):
            continue
        local = model.carrier(z[:, start], a, pair)
        target = z[:, end]
        positions = torch.full((len(z),), end, device=z.device, dtype=torch.long)
        subset = {k: v[keep] for k, v in batch.items()}
        loss = F.mse_loss(local[keep], target[keep])
        loss = loss + .1 * symbolic_loss(model, local[keep], subset, positions[keep], pair.abstraction)
        # Supervise the decoder at observed contexts too, not only predictions.
        loss = loss + .1 * symbolic_loss(model, z[keep, start], subset,
                                        positions[keep] - pair.delta, pair.abstraction)
        losses.append(loss)
        if corrected:
            current = local if step == 0 else model.carrier(current, a, pair)
            recursive.append(F.mse_loss(current[keep], target[keep])
                             + .01 * F.relu(current[keep].abs() - 2).square().mean())
    if not losses:
        raise ValueError("batch has no complete requested-horizon transition")
    result = torch.stack(losses).mean()
    if recursive:
        result = result + torch.stack(recursive).mean()
    return result


class CarrierPairController(nn.Module):
    """Joint pair classifier: current carrier, interface action, remaining steps."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(DIM + 6, 128), nn.SiLU(), nn.Linear(128, 9))

    def forward(self, z, action, remaining):
        if z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("controller accepts only carrier/action/remaining time")
        features = torch.cat((z, action, remaining.reshape(-1, 1).to(z).clamp(max=60) / 60), -1)
        logits = self.layers(features)
        horizons = torch.tensor([p.delta for p in PAIRS], device=z.device)
        return logits.masked_fill(horizons[None, :] > remaining[:, None], -torch.inf)


def linear_macs(model, pair, controller=False):
    """Explicit multiply-accumulate count for executed linear operators.

    Nonlinear/mask work is separately declared, not called a full FLOP count.
    """
    count = model.input_projection.in_features * model.input_projection.out_features
    count += sum(m.in_features * m.out_features for m in model.conditioner.modules()
                 if isinstance(m, nn.Linear))
    count += sum(m.in_features * m.out_features for block in model.blocks for m in block.modules()
                 if isinstance(m, nn.Linear))
    count += model.output_projection.in_features * model.output_projection.out_features
    if pair.abstraction != Abstraction.CONTINUOUS:
        head = model.micro_head if pair.abstraction == Abstraction.MICRO else model.macro_head
        adapter = model.micro_adapter if pair.abstraction == Abstraction.MICRO else model.macro_adapter
        count += sum(m.in_features * m.out_features for m in head.modules() if isinstance(m, nn.Linear))
        count += adapter.in_features * adapter.out_features
    if controller:
        count += (DIM + 6) * 128 + 128 * 9
    return count


@torch.no_grad()
def trajectory_labels(model, batch, compute_weight=.0001, *, contexts=None):
    """Training-only duration-weighted dynamic programming on 60-step windows.

    Endpoint MSE * duration + charged linear MACs / continuous-h15 MACs,
    then continuation value. No physical-violation claim is made.
    """
    z, a, lengths = batch["z"], batch["action"], batch["length"]
    contexts = z if contexts is None else contexts
    size, frames, _ = z.shape
    costs = torch.full((size, frames, 9), torch.inf, device=z.device)
    reference = linear_macs(model, PAIRS[6])
    for i, pair in enumerate(PAIRS):
        positions = frames - pair.delta
        context = contexts[:, :positions].reshape(-1, DIM)
        actions = a[:, None, :].expand(-1, positions, -1).reshape(-1, 5)
        predicted = model.carrier(context, actions, pair).reshape(size, positions, DIM)
        error = (predicted - z[:, pair.delta:]).square().mean(-1)
        costs[:, :positions, i] = pair.delta * error + compute_weight * linear_macs(model, pair, True) / reference
    value = torch.zeros((size, frames), device=z.device)
    labels = torch.zeros((size, frames - 1), device=z.device, dtype=torch.long)
    for t in range(frames - 2, -1, -1):
        choices = costs[:, t].clone()
        for i, pair in enumerate(PAIRS):
            end = t + pair.delta
            if end < frames:
                choices[:, i] += value[:, end]
                choices[:, i].masked_fill_(lengths < end, torch.inf)
        minimum, selected = choices.min(-1)  # frozen PAIRS ordering breaks exact ties
        value[:, t] = torch.where(lengths > t, minimum, 0.)
        labels[:, t] = selected
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite controller teacher costs")
    return labels


@torch.no_grad()
def adaptive_rollout(model, controller, z, action, fixed_steps=225, fixed_pair=None):
    """One pair per decision; no future state or initial-symbol side channel."""
    if z.shape != (1, DIM) or fixed_steps < 1:
        raise ValueError("readiness rollout requires one carrier and positive endpoint")
    if fixed_pair is not None and (fixed_pair not in PAIRS or fixed_steps % fixed_pair.delta):
        raise ValueError("fixed rollout endpoint must be divisible by its trained horizon")
    elapsed, trace = 0, []
    while elapsed < fixed_steps:
        remaining = fixed_steps - elapsed
        pair = fixed_pair
        if pair is None:
            logits = controller(z, action, torch.tensor([remaining], device=z.device))
            pair = PAIRS[int(logits.argmax(-1))]
        z = model.carrier(z, action, pair)
        if not bool(torch.isfinite(z).all()):
            raise ValueError("nonfinite recursive carrier")
        trace.append({"start_fixed_step": elapsed, "requested_horizon": pair.delta,
                      "effective_horizon": pair.delta, "mode": str(pair.abstraction),
                      "controller_calls": int(fixed_pair is None), "transition_calls": 1,
                      "symbol_decoder_calls": int(pair.abstraction != Abstraction.CONTINUOUS),
                      "linear_macs": linear_macs(model, pair, fixed_pair is None)})
        elapsed += pair.delta
    return z, trace


@torch.no_grad()
def intervention_audit(model, z, action):
    outputs = [model.carrier(z, action, pair) for pair in PAIRS]
    mode_deltas = [float((outputs[h * 3 + a] - outputs[h * 3 + b]).abs().max())
                   for h in range(3) for a, b in ((0, 1), (0, 2), (1, 2))]
    horizon_deltas = [float((outputs[a * 3 + m] - outputs[b * 3 + m]).abs().max())
                      for m in range(3) for a, b in ((0, 1), (0, 2), (1, 2))]
    return {"mode_max_abs_change": max(mode_deltas), "horizon_max_abs_change": max(horizon_deltas),
            "minimum_mode_intervention": min(mode_deltas), "minimum_horizon_intervention": min(horizon_deltas),
            "finite": all(bool(torch.isfinite(x).all()) for x in outputs),
            "passed": min(mode_deltas) > 1e-7 and min(horizon_deltas) > 1e-7
                      and all(bool(torch.isfinite(x).all()) for x in outputs)}


def checkpoint_contract(vocabulary, parser_identity, carrier_identity):
    if len(vocabulary) != SLOTS or len(set(vocabulary)) != SLOTS:
        raise ValueError("checkpoint requires the frozen 18-slot vocabulary")
    return {"architecture": ARCHITECTURE, "config": asdict(CONFIG), "vocabulary": list(vocabulary),
            "parser_identity": parser_identity, "carrier_identity": carrier_identity,
            "pairs": [list(p.identity) for p in PAIRS], "runtime_inputs": ["carrier", "action", "remaining_steps"]}
