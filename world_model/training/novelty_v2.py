"""Issue #110 contract-v2 models: the E99 spatial slot parser, carrier and #112 predictors at 28 slots / 370 values.

Every class extends its frozen v1 counterpart (``SpatialSlotParser``, ``CNNHybridPredictor``,
``ContinuousDynamics``) without editing it (their modules are hash-bound by earlier plans):

* ``SpatialSlotParserV2``: the E99 parser with one query per v2 slot and heads sized for the 9 v2
  kinds, 3 relations (contact, supports, force-on) and 3 macros (steady-state,
  structure-unstable, external-force-active).
* ``carrier_values``: the ``TemporalVisualCarrierAdapter.build_from_parsed`` features, vectorized
  over a shot, in the v2 layout (2 header + 4 scene + 13 x 28). ``engine_carriers`` is the same
  builder on engine labels (the engine-projected carrier of the rollout-error metric).
* ``SceneHead``: the #113 temporal-head interface (last k = 15 carriers) emitting the scene state.
* ``HybridPredictorV2`` / ``ContinuousDynamicsV2``: the #71/#74 predictor architectures at 370 values.
* ``BatchedProbe``: F independent identifiability-probe MLPs trained in lock-step.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from world_model.data import novelty_contract as contract
from world_model.data import novelty_labels_v2 as labels
from world_model.model import Abstraction, PredictorConfig
from world_model.model.predictor import FiLMBlock, PairConditioner
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor
from world_model.training.matched_dynamics import CONTINUOUS_PAIRS, HorizonConditioner, ContinuousDynamics
from world_model.training.spatial_slot_parser import SpatialSlotParser

DIM = contract.CARRIER_DIM_V2
SLOTS = len(contract.VOCABULARY_V2)
WIDTH = contract.SLOT_WIDTH
HEADER = contract.CARRIER_HEADER
SCENE = HEADER                       # scene fields occupy columns 2..5
SLOT0 = HEADER + len(contract.SCENE_FIELDS)
KINDS = contract.KIND_VOCABULARY_V2
RELATIONS = labels.RELATIONS
MACROS = labels.MACROS
CONFIG = PredictorConfig(latent_dim=DIM, hidden_dim=384, depth=3)
PRESENCE_COLUMNS = tuple(SLOT0 + WIDTH * i for i in range(SLOTS))
PIG_PRESENCE = SLOT0 + WIDTH * labels.PIG_SLOT
PIG_CENTER = (PIG_PRESENCE + 5, PIG_PRESENCE + 6)
SCENE_HISTORY = 15
PARSER_ARCHITECTURE = "issue-110-spatial-slot-parser-v2"
PREDICTOR_ARCHITECTURE = "issue-110-cnn-carrier-hybrid-v2"


# ------------------------------------------------------------------ parser

class SpatialSlotParserV2(SpatialSlotParser):
    architecture_identity = PARSER_ARCHITECTURE

    def __init__(self, config, layers=2, heads=4):
        super().__init__(config, contract.VOCABULARY_V2, layers=layers, heads=heads)
        width = config.hidden_dim
        self.kind_head = nn.Linear(width, len(KINDS))
        self.relation_head = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, len(RELATIONS)))
        self.macro_head = nn.Linear(width, len(MACROS))


def class_balance_weights(total, positive):
    """The #70 rule: N / (2 N_class) per slot; a slot with only one observed class keeps unit weight."""
    counts = torch.stack((positive, total - positive)).float()
    classes = (counts > 0).sum(0)
    return torch.where(counts > 0, total / (classes * counts.clamp_min(1)), 0)


def parser_loss(model, batch, weights):
    """The #70/E99 ``parser_loss`` terms on the v2 targets: presence x4, balanced pig/block
    presence x4, pig/block count MSE, smooth-L1 centers and kind CE on present slots (kind
    targets per sample: the novelty slot's kind depends on its source), BCE relations and
    macros where masked."""
    output = model(batch["images"])
    vocabulary = contract.VOCABULARY_V2
    presence = batch["presence"].float()
    present = batch["presence"].bool()
    logits = output["presence_logits"]
    loss = 4 * F.binary_cross_entropy_with_logits(logits, presence)
    bce = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
    weighted = bce * (presence * weights[0] + (1 - presence) * weights[1])
    for kind in ("pig", "block"):
        index = [i for i, v in enumerate(vocabulary) if v.startswith(kind + ":")]
        loss = loss + 4 * weighted[:, index].mean()
    probability = logits.sigmoid()
    for kind in ("pig", "block"):
        index = [i for i, v in enumerate(vocabulary) if v.startswith(kind + ":")]
        loss = loss + F.mse_loss(probability[:, index].sum(1), presence[:, index].sum(1))
    if present.any():
        loss = loss + F.smooth_l1_loss(output["centers"][present], batch["centers"][present])
        loss = loss + F.cross_entropy(output["kind_logits"][present], batch["kinds"].long()[present])
    for key, mask, target in (("relation_logits", "relation_mask", "relations"), ("macro_logits", "macro_mask", "macros")):
        if batch[mask].any():
            loss = loss + F.binary_cross_entropy_with_logits(output[key][batch[mask]], batch[target][batch[mask]].float())
    return loss


def parsed(model, images, temperatures=None):
    """Parser probabilities (the adapter's ``parse_batch`` transform; #70 temperatures are 1)."""
    with torch.no_grad():
        output = model(images)
    return {"presence": output["presence_logits"].sigmoid(), "centers": output["centers"],
            "kinds": output["kind_logits"].softmax(-1), "relations": output["relation_logits"].sigmoid(),
            "macros": output["macro_logits"].sigmoid()}


# ----------------------------------------------------------------- carrier

EXPECTED_KINDS = labels.expected_kinds()


def carrier_values(presence, centers, kinds, times, threshold=0.5):
    """``build_from_parsed`` over one shot's positions, v2 layout, scene columns left zero.

    presence [T, 28], centers [T, 28, 2], kinds [T, 28, 9] probabilities; times [T] fixed-time
    seconds. Position t's prior is t - 1 (position 0 has none: header [0, 0], no motion)."""
    T = len(presence)
    device = presence.device
    presence, centers, kinds = presence.double(), centers.double(), kinds.double()
    times = times.to(device=device, dtype=torch.float64)
    out = torch.zeros(T, DIM, dtype=torch.float64, device=device)
    elapsed = torch.zeros(T, dtype=torch.float64, device=device)
    elapsed[1:] = times[1:] - times[:-1]
    out[1:, 0] = 1.0
    out[:, 1] = elapsed
    confidence, kind_index = kinds.max(-1)
    expected = kinds.gather(-1, EXPECTED_KINDS.to(device)[None, :, None].expand(T, -1, 1)).squeeze(-1)
    available = presence >= threshold
    motion_available = torch.zeros_like(available)
    motion_available[1:] = available[1:] & available[:-1]
    motion = torch.zeros_like(centers)
    motion[1:] = (centers[1:] - centers[:-1]) / elapsed[1:, None, None].clamp_min(1e-12)
    motion = motion * motion_available[..., None]
    slot = torch.stack((presence, torch.ones_like(presence), kind_index.double() / (len(KINDS) - 1), confidence,
                        torch.ones_like(presence), centers[..., 0] * available, centers[..., 1] * available,
                        available.double(), motion[..., 0], motion[..., 1], motion_available.double(), expected,
                        torch.ones_like(presence)), -1)
    out[:, SLOT0:] = slot.reshape(T, -1)
    return out.float()


def engine_carriers(shot):
    """The engine-projected v2 carrier of a label record: presence 0/1, engine centers, one-hot
    engine kinds, and the trace scene state (available)."""
    presence = shot["presence"].float()
    kinds = shot["kinds"].long().clamp_min(0)
    one_hot = F.one_hot(kinds, len(KINDS)).float()
    absent = ~shot["presence"]
    one_hot[absent] = F.one_hot(EXPECTED_KINDS, len(KINDS)).float()[None].expand(len(presence), -1, -1)[absent]
    out = carrier_values(presence, shot["centers"], one_hot, shot["times"])
    out[:, SCENE + 0] = shot["scene"][:, 0]
    out[:, SCENE + 1] = 1.0
    out[:, SCENE + 2] = shot["scene"][:, 1]
    out[:, SCENE + 3] = 1.0
    return out


def masked_errors(prediction, target, presence):
    """Rollout-error metrics against the engine-projected carrier.

    masked_carrier_mse: mean squared error over the 13 features of every engine-present slot and
    the two scene values (gravity_y_normalized, storm_active); position_mse: center x/y only."""
    p = prediction[..., SLOT0:].reshape(*prediction.shape[:-1], SLOTS, WIDTH)
    t = target[..., SLOT0:].reshape(*target.shape[:-1], SLOTS, WIDTH)
    mask = presence.float()
    slot_sq = ((p - t) ** 2).sum(-1) * mask
    scene_sq = (prediction[..., SCENE] - target[..., SCENE]) ** 2 + (prediction[..., SCENE + 2] - target[..., SCENE + 2]) ** 2
    count = mask.sum(-1) * WIDTH + 2
    masked = (slot_sq.sum(-1) + scene_sq) / count
    position = (((p[..., 5:7] - t[..., 5:7]) ** 2).sum(-1) * mask).sum(-1) / (2 * mask.sum(-1)).clamp_min(1)
    return masked, position


# ------------------------------------------------------------------ scene

def scene_inputs(carriers, positions, starts=None, k=SCENE_HISTORY):
    """History of the k carriers ending at each position (left-padded with the shot's first
    position, ``starts``; default 0), header and slot columns only, plus their 1-frame
    differences: [N, k * 366 + (k - 1) * 366]."""
    keep = torch.cat((carriers[:, :HEADER], carriers[:, SLOT0:]), -1)
    offsets = torch.arange(-k + 1, 1, device=carriers.device)
    floor = torch.zeros_like(positions) if starts is None else starts
    index = torch.maximum(positions[:, None] + offsets[None], floor[:, None])
    history = keep[index]
    return torch.cat((history.flatten(1), (history[:, 1:] - history[:, :-1]).flatten(1)), -1)


class SceneHead(nn.Module):
    """gravity_y_normalized (regression) and storm-active (logit) from the last 15 carriers."""

    def __init__(self, k=SCENE_HISTORY, hidden=128):
        super().__init__()
        self.k = k
        width = (DIM - len(contract.SCENE_FIELDS)) * (2 * k - 1)
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, inputs):
        return self.net(inputs)

    @torch.no_grad()
    def fill(self, carriers):
        """Write the scene columns of one shot's carriers [T, 370] in place and return them."""
        positions = torch.arange(len(carriers), device=carriers.device)
        raw = self(scene_inputs(carriers, positions, k=self.k))
        available = (positions >= self.k - 1).float()
        carriers[:, SCENE + 0] = raw[:, 0]
        carriers[:, SCENE + 1] = available
        carriers[:, SCENE + 2] = raw[:, 1].sigmoid()
        carriers[:, SCENE + 3] = available
        return carriers


def scene_loss(head, inputs, targets):
    raw = head(inputs)
    return F.mse_loss(raw[:, 0], targets[:, 0]) + F.binary_cross_entropy_with_logits(raw[:, 1], targets[:, 1])


# -------------------------------------------------------------- predictors

class HybridPredictorV2(CNNHybridPredictor):
    """``CNNHybridPredictor`` (FiLM trunk, fixed-slot soft predicate adapters; contact symmetrized,
    supports and force-on directed) at the 370-value v2 carrier: 28 x 28 x 3 relations + 3
    availabilities, 3 macros + 3 availabilities; presence read at the v2 slot columns."""
    architecture = PREDICTOR_ARCHITECTURE

    def __init__(self, config=CONFIG):
        nn.Module.__init__(self)
        if config.latent_dim != DIM or config.action_dim != 5:
            raise ValueError("the v2 predictor requires the 370-value carrier")
        self.config = config
        self.conditioner = PairConditioner(config)
        self.input_projection = nn.Linear(DIM + 5, config.hidden_dim)
        self.blocks = nn.ModuleList(FiLMBlock(config.hidden_dim, config.pair_code_dim) for _ in range(config.depth))
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, DIM)
        micro = SLOTS * SLOTS * len(RELATIONS) + len(RELATIONS)
        macro = 2 * len(MACROS)
        self.micro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, micro))
        self.macro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, macro))
        self.micro_adapter = nn.Linear(micro, config.hidden_dim, bias=False)
        self.macro_adapter = nn.Linear(macro, config.hidden_dim, bias=False)

    def symbols(self, z, mode):
        if mode == Abstraction.MICRO:
            raw = self.micro_head(z)
            count = len(RELATIONS)
            relations = raw[:, :-count].reshape(-1, SLOTS, SLOTS, count)
            contact = (relations[..., 0] + relations[..., 0].transpose(1, 2)) / 2
            relations = torch.cat((contact[..., None], relations[..., 1:]), -1)
            return relations, raw[:, -count:]
        if mode == Abstraction.MACRO:
            raw = self.macro_head(z)
            return raw[:, :len(MACROS)], raw[:, len(MACROS):]
        raise ValueError("continuous mode has no symbolic readout")

    def symbolic_features(self, z, mode):
        logits, availability = self.symbols(z, mode)
        available = availability.sigmoid()
        values = logits.sigmoid()
        if mode == Abstraction.MICRO:
            presence = z[:, SLOT0::WIDTH].clamp(0, 1)
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


class ContinuousDynamicsV2(ContinuousDynamics):
    """``ContinuousDynamics`` (no symbolic modules) at the 370-value v2 carrier."""

    def __init__(self, width):
        nn.Module.__init__(self)
        self.config = replace(CONFIG, hidden_dim=width)
        self.conditioner = HorizonConditioner(self.config)
        self.input_projection = nn.Linear(DIM + 5, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, CONFIG.pair_code_dim) for _ in range(CONFIG.depth))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, DIM)

    def carrier(self, z, action, pair):
        if pair not in CONTINUOUS_PAIRS or z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("pure dynamics require a 370-carrier, action and continuous horizon")
        hidden = self.input_projection(torch.cat((z, action), -1))
        code = self.conditioner.code(pair, len(z), z.device)
        for block in self.blocks:
            hidden = block(hidden, code)
        return z + self.output_projection(self.output_norm(hidden))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def continuous_width():
    """The #74 capacity rule at v2: the continuous width whose parameter count is closest to the
    v2 hybrid's (widths 384..1024, step 8; ties to the smaller width)."""
    with torch.device("meta"):
        target = parameter_count(HybridPredictorV2())
        counts = {w: parameter_count(ContinuousDynamicsV2(w)) for w in range(384, 1025, 8)}
    width = min(counts, key=lambda w: (abs(counts[w] - target), w))
    return {"width": width, "hybrid_parameters": target, "continuous_parameters": counts[width],
            "relative_error": abs(counts[width] - target) / target}


# ------------------------------------------------------------------ probe

class BatchedProbe(nn.Module):
    """F independent MLP probes (hidden 64) over one input width, trained in lock-step."""

    def __init__(self, count, width, hidden=64):
        super().__init__()
        scale1, scale2 = width ** -0.5, hidden ** -0.5
        self.w1 = nn.Parameter(torch.empty(count, width, hidden).uniform_(-scale1, scale1))
        self.b1 = nn.Parameter(torch.empty(count, 1, hidden).uniform_(-scale1, scale1))
        self.w2 = nn.Parameter(torch.empty(count, hidden, 1).uniform_(-scale2, scale2))
        self.b2 = nn.Parameter(torch.empty(count, 1, 1).uniform_(-scale2, scale2))

    def forward(self, x):
        """x [F, B, width] -> logits [F, B]."""
        return (torch.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2).squeeze(-1)
