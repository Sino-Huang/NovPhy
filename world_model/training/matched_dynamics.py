"""Independent continuous dynamics and matched-policy execution for issue #74."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.model.predictor import FiLMBlock
from world_model.training.cnn_hybrid import CONFIG, DIM, PAIRS, CNNHybridPredictor, linear_macs

CONTINUOUS_PAIRS = tuple(p for p in PAIRS if p.abstraction == Abstraction.CONTINUOUS)


class HorizonConditioner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.count = config.delta_frequency_count
        self.fuse = nn.Sequential(nn.Linear(2*self.count, 2*config.pair_code_dim), nn.SiLU(),
                                  nn.Linear(2*config.pair_code_dim, config.pair_code_dim))

    def code(self, pair, size, device):
        frequencies = 10000. ** (-torch.arange(self.count, device=device)/self.count)
        angles = pair.delta * frequencies
        return self.fuse(torch.cat((angles.sin(), angles.cos())))[None].expand(size, -1)


class ContinuousDynamics(nn.Module):
    """No symbolic modules, mode embedding, symbolic input, or hybrid initialization."""
    def __init__(self, width):
        super().__init__()
        self.config = replace(CONFIG, hidden_dim=width)
        self.conditioner = HorizonConditioner(self.config)
        self.input_projection = nn.Linear(DIM+5, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, CONFIG.pair_code_dim) for _ in range(CONFIG.depth))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, DIM)

    def carrier(self, z, action, pair):
        if pair not in CONTINUOUS_PAIRS or z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("pure dynamics require a 236-carrier, action and continuous horizon")
        hidden = self.input_projection(torch.cat((z, action), -1))
        code = self.conditioner.code(pair, len(z), z.device)
        for block in self.blocks:
            hidden = block(hidden, code)
        return z + self.output_projection(self.output_norm(hidden))


def pairs_for(model):
    return CONTINUOUS_PAIRS if isinstance(model, ContinuousDynamics) else PAIRS


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def capacity_contract():
    # Architecture counts only; no data, optimization or width-performance sweep.
    with torch.device("meta"):
        target = parameter_count(CNNHybridPredictor())
        counts = {w: parameter_count(ContinuousDynamics(w)) for w in range(384, 513, 8)}
    width = min(counts, key=lambda w: (abs(counts[w]-target), w))
    error = abs(counts[width]-target)/target
    if error > .02:
        raise ValueError("deterministic continuous width cannot meet 2% capacity tolerance")
    return {"rule": "nearest total parameter count; widths 384..512 step8, smaller breaks ties",
            "tolerance": .02, "continuous_width": width, "continuous_parameters": counts[width],
            "hybrid_parameters": target, "relative_difference": error,
            "padding_parameters": 0, "active_capacity_matched": False}


def continuous_loss(model, z, action, length, pair):
    """Targets are continuous carriers only; symbolic dictionaries cannot enter."""
    losses, recursive = [], []
    current = z[:, 0]
    for step in range(4):
        start, end = step*pair.delta, (step+1)*pair.delta
        keep = length >= end
        if not bool(keep.any()):
            continue
        local = model.carrier(z[:, start], action, pair)
        losses.append(F.mse_loss(local[keep], z[keep, end]))
        current = local if step == 0 else model.carrier(current, action, pair)
        recursive.append(F.mse_loss(current[keep], z[keep, end])
                         + .01*F.relu(current[keep].abs()-2).square().mean())
    if not losses:
        raise ValueError("no complete continuous transition")
    return torch.stack(losses).mean() + torch.stack(recursive).mean()


class MatchedController(nn.Module):
    def __init__(self, pure):
        super().__init__()
        self.pairs = CONTINUOUS_PAIRS if pure else PAIRS
        # 131x3 versus 128x9: <0.2% total capacity difference, no dead logits.
        width = 131 if pure else 128
        self.layers = nn.Sequential(nn.Linear(DIM+6, width), nn.SiLU(), nn.Linear(width, len(self.pairs)))

    def forward(self, z, action, remaining):
        if z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("controller accepts only carrier, action, remaining time")
        x = torch.cat((z, action, remaining.reshape(-1, 1).to(z).clamp(max=60)/60), -1)
        horizons = torch.tensor([p.delta for p in self.pairs], device=z.device)
        return self.layers(x).masked_fill(horizons[None] > remaining[:, None], -torch.inf)


def work(model, pair, controller=None):
    count = linear_macs(model, pair)
    if controller is not None:
        count += sum(m.in_features*m.out_features for m in controller.modules() if isinstance(m, nn.Linear))
    return count


@torch.no_grad()
def active_capacity(model, z, action, pair):
    """Parameters in executed leaf operators; shared tables counted in full."""
    used, hooks = {}, []
    def capture(module, inputs, output):
        for p in module.parameters(recurse=False):
            used[id(p)] = p.numel()
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            hooks.append(module.register_forward_hook(capture))
    try:
        model.carrier(z, action, pair)
    finally:
        for hook in hooks:
            hook.remove()
    return sum(used.values())


@torch.no_grad()
def rollout(model, controller, z, action, steps=225, fixed_pair=None):
    if z.shape != (1, DIM) or steps < 1:
        raise ValueError("deployment requires B=1 and positive endpoint")
    pairs = pairs_for(model)
    if fixed_pair is not None and (fixed_pair not in pairs or steps % fixed_pair.delta):
        raise ValueError("unsupported pair or incomplete fixed horizon")
    if fixed_pair is None and controller.pairs != pairs:
        raise ValueError("controller/predictor capabilities differ")
    t, trace = 0, []
    while t < steps:
        pair = fixed_pair
        if pair is None:
            pair = pairs[int(controller(z, action, torch.tensor([steps-t], device=z.device)).argmax(-1))]
        z = model.carrier(z, action, pair)
        if not bool(torch.isfinite(z).all()):
            raise ValueError("nonfinite recursive carrier")
        trace.append({"start_fixed_step": t, "horizon": pair.delta, "mode": str(pair.abstraction),
                      "linear_macs": work(model, pair, controller if fixed_pair is None else None),
                      "controller_calls": int(fixed_pair is None), "transition_calls": 1,
                      "symbol_decoder_calls": int(pair.abstraction != Abstraction.CONTINUOUS)})
        t += pair.delta
    return z, trace


@torch.no_grad()
def controller_labels(model, controller, z, action, length, contexts=None):
    """Training-only duration MSE DP; no symbolic or task outcome labels."""
    contexts = z if contexts is None else contexts
    size, frames, _ = z.shape
    pairs = pairs_for(model)
    costs = torch.full((size, frames, len(pairs)), torch.inf, device=z.device)
    reference = work(model, CONTINUOUS_PAIRS[-1])
    for i, pair in enumerate(pairs):
        positions = frames-pair.delta
        pred = model.carrier(contexts[:, :positions].reshape(-1, DIM),
                             action[:, None].expand(-1, positions, -1).reshape(-1, 5), pair)
        error = (pred.reshape(size, positions, DIM)-z[:, pair.delta:]).square().mean(-1)
        costs[:, :positions, i] = pair.delta*error + .0001*work(model, pair, controller)/reference
    values = torch.zeros((size, frames), device=z.device)
    labels = torch.zeros((size, frames-1), dtype=torch.long, device=z.device)
    for t in range(frames-2, -1, -1):
        choices = costs[:, t].clone()
        for i, pair in enumerate(pairs):
            end = t+pair.delta
            if end < frames:
                choices[:, i] += values[:, end]
                choices[:, i].masked_fill_(length < end, torch.inf)
        minimum, labels[:, t] = choices.min(-1)
        values[:, t] = torch.where(length > t, minimum, 0.)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite controller DP costs")
    return labels
