"""New, independently fitted #76 dynamics over the shared observed-history carrier."""
from dataclasses import replace

import torch
from torch import nn

from world_model.model import Abstraction, PredictionPair
from world_model.model.predictor import FiLMBlock, PairConditioner
from world_model.training.cnn_hybrid import CONFIG, DIM, SLOTS, linear_macs
from world_model.training.matched_dynamics import HorizonConditioner
from world_model.training.observed_history import HISTORY_DIM, history_contract

STATE_DIM = DIM + HISTORY_DIM
NATIVE_FIXED_DELTA_SECONDS = .0004
NATIVE_HORIZONS = (50, 250, 750)
PAIRS = tuple(PredictionPair(h, mode) for h in NATIVE_HORIZONS for mode in Abstraction)
CONTINUOUS_PAIRS = tuple(p for p in PAIRS if p.abstraction == Abstraction.CONTINUOUS)


class HistoryDynamics(nn.Module):
    """The pure arm has neither symbolic modules nor a mode embedding.

    Both variants predict the full carrier, including memory during imagined
    rollouts. Only new real observations/executed actions update the real
    episode's shared encoder state. No pretrained hybrid parameters initialize
    the pure arm, and the two model instances share no trainable parameters.
    """
    def __init__(self, *, pure, width):
        super().__init__()
        self.pure = pure
        self.pairs = CONTINUOUS_PAIRS if pure else PAIRS
        self.config = replace(CONFIG, latent_dim=STATE_DIM, hidden_dim=width)
        self.conditioner = HorizonConditioner(self.config) if pure else PairConditioner(self.config)
        self.input_projection = nn.Linear(STATE_DIM + 5, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, self.config.pair_code_dim)
                                    for _ in range(self.config.depth))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, STATE_DIM)
        if not pure:
            self.micro_head = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.SiLU(),
                                            nn.Linear(128, SLOTS * SLOTS * 2 + 2))
            self.macro_head = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.SiLU(), nn.Linear(128, 4))
            self.micro_adapter = nn.Linear(SLOTS * SLOTS * 2 + 2, width, bias=False)
            self.macro_adapter = nn.Linear(4, width, bias=False)

    def symbols(self, z, mode):
        if self.pure:
            raise ValueError("pure history dynamics do not have symbolic readouts")
        if mode == Abstraction.MICRO:
            raw = self.micro_head(z)
            relations = raw[:, :-2].reshape(-1, SLOTS, SLOTS, 2)
            contact = (relations[..., 0] + relations[..., 0].transpose(1, 2)) / 2
            return torch.stack((contact, relations[..., 1]), -1), raw[:, -2:]
        if mode == Abstraction.MACRO:
            raw = self.macro_head(z)
            return raw[:, :2], raw[:, 2:]
        raise ValueError("continuous mode has no symbolic readout")

    def symbolic_features(self, z, mode):
        logits, availability = self.symbols(z, mode)
        available, values = availability.sigmoid(), logits.sigmoid()
        if mode == Abstraction.MICRO:
            presence = z[:, :DIM][:, 2::13].clamp(0, 1)
            active = presence[:, :, None] * presence[:, None, :]
            active = active * (1 - torch.eye(SLOTS, device=z.device))
            values = values * active[..., None] * available[:, None, None, :]
        else:
            values = values * available
        return torch.cat((values.flatten(1), available), -1)

    def carrier(self, z, action, pair):
        if pair not in self.pairs or z.ndim != 2 or z.shape[1] != STATE_DIM or action.shape != (len(z), 5):
            raise ValueError("history dynamics require a permitted pair, 300-carrier and five-component action")
        hidden = self.input_projection(torch.cat((z, action), -1))
        if pair.abstraction == Abstraction.MICRO:
            hidden = hidden + self.micro_adapter(self.symbolic_features(z, pair.abstraction))
        elif pair.abstraction == Abstraction.MACRO:
            hidden = hidden + self.macro_adapter(self.symbolic_features(z, pair.abstraction))
        code = self.conditioner.code(pair, len(z), z.device)
        for block in self.blocks:
            hidden = block(hidden, code)
        return z + self.output_projection(self.output_norm(hidden))


def capacity_contract():
    """Choose width using parameter counts only; never inspect outcomes."""
    def count(model):
        return sum(p.numel() for p in model.parameters())
    with torch.device("meta"):
        hybrid = HistoryDynamics(pure=False, width=384)
        target = count(hybrid)
        counts = {w: count(HistoryDynamics(pure=True, width=w)) for w in range(384, 513, 8)}
        width = min(counts, key=lambda w: (abs(counts[w] - target), w))
        pure = HistoryDynamics(pure=True, width=width)
    difference = abs(counts[width] - target) / target
    if difference > .02:
        raise ValueError("history dynamics exceed the declared 2% parameter-matching tolerance")
    return {"schema": "issue_76_native_observed_history_dynamics_v2", "history": history_contract(),
            "fixed_delta_seconds": NATIVE_FIXED_DELTA_SECONDS,
            "horizons_native_steps": list(NATIVE_HORIZONS),
            "horizons_seconds": [h * NATIVE_FIXED_DELTA_SECONDS for h in NATIVE_HORIZONS],
            "equal_time_endpoint_native_steps": 11250,
            "state_dim": STATE_DIM, "hybrid_width": 384, "continuous_width": width,
            "hybrid_parameters": target, "continuous_parameters": counts[width],
            "relative_parameter_difference": difference,
            "width_rule": "nearest total count over 384..512 step8; smaller width breaks ties",
            "shared_encoder_included_in_arm_counts": False,
            "shared_encoder_cost": "identical standalone fitting and deployment accounting for both arms",
            "dead_padding_parameters": 0,
            "linear_macs_per_transition": {
                "hybrid": {str(p.identity): linear_macs(hybrid, p) for p in PAIRS},
                "continuous": {str(p.identity): linear_macs(pure, p) for p in CONTINUOUS_PAIRS}},
            "compute_note": "MACs are not total FLOPs or equal deployment-time evidence",
            "fit_executed": False}
