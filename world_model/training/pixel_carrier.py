"""Issue #106: pixel carriers (the #99 spatial slot parser), request-conditioned predictors on them,
recorded request rollouts and the two downstream decision readouts (CLEVRER QA, Physion OCP).

- Parser: ``SpatialSlotParser`` unchanged (E99 architecture and optimiser recipe); the E99 / #70
  ``parser_loss`` terms that exist for these benchmarks (presence x4, class-balanced presence x4,
  presence-count MSE, smooth-L1 centers on valid slots, kind cross-entropy on present slots; the
  benchmark kinds use the first ``KINDS[bench]`` kind logits). Relation and macro parser heads are
  untrained: the descriptions are the predictor's heads.
- Carrier: ``carrier_values`` = the deployment adapter's ``build_from_parsed`` over a clip (presence
  threshold 0.5, motion = center difference / elapsed when both frames are available), 236 values.
- Predictors: ``predicate_feedback.DeltaMacroPredictor`` (#113 Delta-indexed macro head) and
  ``LabelFreeMicroPredictor`` (the #115 label-free binary micro bottleneck, same Delta-indexed macro).
- ``rollout``: ``request_policy.rollout_endpoints`` with the same ops in the same order, but one
  initial carrier per row, per-row endpoints and recorded landings.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from world_model.model import Abstraction
from world_model.training import predicate_feedback as pf
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import DIM, PAIRS, SLOTS
from world_model.training.cohort_v2_visual_parser import CohortV2VisualParserConfig
from world_model.training.matched_dynamics import linear_macs
from world_model.training.novelty_v2 import class_balance_weights
from world_model.training.spatial_slot_parser import SpatialSlotParser

ENCODER = {"hidden_dim": 128, "epochs": 12, "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
           "seed": 9900001, "attention_layers": 2, "attention_heads": 4}
KINDS = {"clevrer": 3, "physion": 6}
PRESENCE_THRESHOLD = 0.5
FEATURES = 13
MICRO_PAIR = PAIRS[1]  # (1, micro): any micro pair reads the same micro head


def vocabulary(bench):
    from world_model.data import pixel_benchmarks as pb
    if bench == "clevrer":
        keyed = tuple(f"{c}-{m}" for c in pb.COLORS for m in pb.MATERIALS)
        return keyed + tuple(f"empty-{i}" for i in range(len(keyed), SLOTS))
    return tuple(f"object-{i:02d}" for i in range(SLOTS))


def make_parser(bench, image_size, seed=ENCODER["seed"]):
    h, w = image_size
    config = CohortV2VisualParserConfig(seed=seed, image_height=h, image_width=w, hidden_dim=ENCODER["hidden_dim"],
                                        epochs=ENCODER["epochs"], batch_size=ENCODER["batch_size"],
                                        learning_rate=ENCODER["learning_rate"], weight_decay=ENCODER["weight_decay"])
    return SpatialSlotParser(config, vocabulary(bench), layers=ENCODER["attention_layers"],
                             heads=ENCODER["attention_heads"])


def presence_weights(presence):
    """#70 class-balance weights [2, 18] from training presence [N, 18] (bool)."""
    p = torch.as_tensor(presence)
    return class_balance_weights(torch.tensor(float(len(p))), p.float().sum(0))


def parser_loss(model, batch, weights, kinds):
    """images uint8 [B, 3, H, W]; presence [B, 18] bool; centers [B, 18, 2]; center_valid [B, 18];
    kinds [B, 18] (-1 empty)."""
    output = model(batch["images"])
    presence = batch["presence"].float()
    logits = output["presence_logits"]
    loss = 4 * F.binary_cross_entropy_with_logits(logits, presence)
    raw = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
    loss = loss + 4 * (raw * (presence * weights[0] + (1 - presence) * weights[1])).mean()
    loss = loss + F.mse_loss(logits.sigmoid().sum(1), presence.sum(1))
    valid = batch["center_valid"].bool()
    if valid.any():
        loss = loss + F.smooth_l1_loss(output["centers"][valid], batch["centers"][valid])
    present = batch["presence"].bool()
    if present.any():
        loss = loss + F.cross_entropy(output["kind_logits"][present][:, :kinds], batch["kinds"][present])
    return loss


@torch.no_grad()
def parse(model, images, kinds):
    output = model(images)
    return {"presence": output["presence_logits"].sigmoid(), "centers": output["centers"],
            "kinds": output["kind_logits"][..., :kinds].softmax(-1)}


def carrier_values(presence, centers, kinds, elapsed, threshold=PRESENCE_THRESHOLD):
    """One clip: presence [T, 18], centers [T, 18, 2], kinds [T, 18, K] probabilities -> [T, 236].
    Header [has prior, elapsed]; slot columns: presence, 1, kind index/(K-1), kind confidence, 1,
    center x, center y, center available, motion x, motion y, motion available, 1, 1."""
    T = len(presence)
    presence, centers, kinds = presence.double(), centers.double(), kinds.double()
    out = torch.zeros(T, DIM, dtype=torch.float64, device=presence.device)
    out[1:, 0] = 1.0
    out[1:, 1] = elapsed
    confidence, index = kinds.max(-1)
    available = presence >= threshold
    motion_available = torch.zeros_like(available)
    motion_available[1:] = available[1:] & available[:-1]
    motion = torch.zeros_like(centers)
    motion[1:] = (centers[1:] - centers[:-1]) / elapsed
    slots = out[:, 2:2 + SLOTS * FEATURES].view(T, SLOTS, FEATURES)
    slots[..., 0] = presence
    slots[..., 1] = 1.0
    slots[..., 2] = index.double() / (kinds.shape[-1] - 1)
    slots[..., 3] = confidence
    slots[..., 4] = 1.0
    slots[..., 5:7] = torch.where(available[..., None], centers, 0.)
    slots[..., 7] = available.double()
    slots[..., 8:10] = torch.where(motion_available[..., None], motion, 0.)
    slots[..., 10] = motion_available.double()
    slots[..., 11] = 1.0
    slots[..., 12] = 1.0
    return out.float()


def annotation_carrier(targets, kinds, elapsed):
    """The same builder on annotation targets (presence 0/1, one-hot kinds): the secondary truth."""
    presence = torch.as_tensor(targets["presence"]).double()
    centers = torch.as_tensor(targets["centers"]).double()
    labels = torch.as_tensor(targets["kinds"]).clamp_min(0)
    one_hot = F.one_hot(labels, kinds).double()
    return carrier_values(presence, centers, one_hot, elapsed)


# --------------------------------------------------------------------------- predictors


class LabelFreeMicroPredictor(pf.VariantPredictor):
    """#115 label-free binary micro bottleneck (micro head trained only through the prediction loss),
    with the #113 Delta-indexed macro head of the reference arm."""
    DELTA_MACRO = True
    BINARY = frozenset({"micro"})


FAMILIES = {"hybrid": (pf.DeltaMacroPredictor, pf.SymbolicSpec(macro_delta=True)),
            "label-free": (LabelFreeMicroPredictor, pf.SymbolicSpec(micro="none", macro_delta=True))}


def contact_values(model, z):
    """Gated micro contact values [N, 18, 18] (symmetric) of the model's own micro head."""
    features = pf.own_features(model, z, MICRO_PAIR)
    return features[:, :SLOTS * SLOTS * 2].view(-1, SLOTS, SLOTS, 2)[..., 0]


def macro_values(model, z, delta):
    """Gated Delta-indexed macro values [N, 2]."""
    return model.symbolic_features(z, Abstraction.MACRO, delta)[:, :2]


# --------------------------------------------------------------------------- rollouts


@torch.no_grad()
def rollout(model, policy, z0, endpoint, spec, *, record=False):
    """``rp.rollout_endpoints`` ops (fixed / policy argmax), one initial carrier per row, endpoint
    scalar or [N]. Returns z_end, positions [N], timing and, if ``record``, landings: list of
    (rows, positions, z, pair_index) per iteration."""
    pairs = policy.pairs if policy is not None else PAIRS
    count = len(z0)
    z = z0.clone()
    device = z.device
    endpoint = torch.as_tensor(endpoint, device=device).expand(count).clone()
    actions = torch.zeros(count, 5, device=device)
    alive = endpoint > 0
    positions = torch.zeros(count, dtype=torch.long, device=device)
    pair_macs = [linear_macs(model, pair) for pair in pairs]
    transition_calls = controller_calls = macs = 0
    landings = []
    if z.is_cuda:
        torch.cuda.synchronize()
    began = time.monotonic()
    while bool(alive.any()):
        index = alive.nonzero(as_tuple=True)[0]
        if spec["kind"] == "fixed":
            chosen = torch.full((len(index),), spec["pair_index"], dtype=torch.long, device=device)
        else:
            chosen = policy(z[index], actions[index], endpoint[index] - positions[index]).argmax(-1)
            controller_calls += len(index)
        z_next = z
        for pair_index, pair in enumerate(pairs):
            rows = index[chosen == pair_index]
            if not len(rows):
                continue
            z_next = z_next.clone()
            z_next[rows] = model.carrier(z[rows], actions[rows], pair)
            transition_calls += len(rows)
            macs += len(rows) * pair_macs[pair_index]
            positions[rows] += pair.delta
        z = z_next
        if record:
            landings.append((index.clone(), positions[index].clone(), z[index].clone(), chosen.clone()))
        alive = alive & torch.isfinite(z).all(dim=1) & (positions < endpoint)
    if z.is_cuda:
        torch.cuda.synchronize()
    macs += controller_calls * (rp.policy_macs(policy) if spec["kind"] != "fixed" else 0)
    timing = {"transition_calls": transition_calls, "controller_calls": controller_calls, "linear_macs": macs,
              "gpu_seconds": time.monotonic() - began}
    return z, positions, timing, landings


def readout_macs(model, kind):
    """Linear MACs of one decision readout call on a landed carrier (micro or macro head)."""
    head = model.micro_head if kind == "micro" else model.macro_head
    return sum(m.in_features * m.out_features for m in head.modules() if isinstance(m, torch.nn.Linear))


# --------------------------------------------------------------------------- CLEVRER QA


def slot_attributes(presence, kinds, min_frames=1):
    """Parsed attributes per slot over one video: exists (presence >= 0.5 on at least ``min_frames``
    frames; the frozen readout uses 1, i.e. max presence >= 0.5), shape (argmax of the presence-weighted
    mean kind probabilities); colour and material come from the slot key."""
    from world_model.data import pixel_benchmarks as pb
    exists = (presence >= PRESENCE_THRESHOLD).sum(0) >= min_frames
    weights = presence[..., None]
    shape = (kinds * weights).sum(0).argmax(-1)
    out = []
    for slot in range(SLOTS):
        if slot >= len(pb.COLORS) * len(pb.MATERIALS) or not bool(exists[slot]):
            continue
        out.append({"slot": slot, "color": pb.COLORS[slot // 2], "material": pb.MATERIALS[slot % 2],
                    "shape": pb.SHAPES[int(shape[slot])]})
    return out


FILTERS = {"filter_color": "color", "filter_material": "material", "filter_shape": "shape"}


def resolve_objects(tokens, attributes):
    """Object references of a CLEVRER program in order: each ``objects ... unique`` run resolves to a
    slot (exactly one parsed object matches its filters) or None."""
    resolved, candidates, pending = [], None, None
    for token in tokens:
        if token == "objects":
            candidates, pending = list(attributes), None
        elif token in FILTERS and candidates is not None:
            candidates = [a for a in candidates if a[FILTERS[token]] == pending]
            pending = None
        elif token == "unique" and candidates is not None:
            resolved.append(candidates[0]["slot"] if len(candidates) == 1 else None)
            candidates = None
        elif candidates is not None:
            pending = token
    return resolved


def choice_pair(choice, attributes):
    objects = resolve_objects(choice["program"], attributes)
    if len(objects) != 2 or None in objects or objects[0] == objects[1]:
        return None
    return tuple(objects)


# --------------------------------------------------------------------------- statistics


def clustered_bootstrap(values, *, groups=None, draws=10000, seed=7201, quantiles=(0.025, 0.975)):
    """values [S seeds, I items] (nan = absent): resample seeds and item clusters independently with
    replacement (PCG64 seed). Without ``groups`` every item is its own cluster; with ``groups`` [I] (cluster
    id per item, e.g. the video of a question option) items of one cluster are drawn together. The
    statistic is the mean over drawn seeds of (sum over drawn clusters / their item count), so the point
    estimate is the pooled per-item mean of each seed, averaged over seeds."""
    values = np.asarray(values, np.float64)
    seeds = values.shape[0]
    if groups is None:
        sums, counts = np.nan_to_num(values), np.isfinite(values).astype(np.float64)
    else:
        keys, inverse = np.unique(np.asarray(groups), return_inverse=True)
        sums = np.zeros((seeds, len(keys)))
        counts = np.zeros((seeds, len(keys)))
        for s in range(seeds):
            np.add.at(sums[s], inverse, np.nan_to_num(values[s]))
            np.add.at(counts[s], inverse, np.isfinite(values[s]).astype(np.float64))
    clusters = sums.shape[1]
    generator = np.random.default_rng(np.random.PCG64(seed))
    per_seed = sums.sum(1) / counts.sum(1)
    stats = np.empty(draws)
    for k in range(draws):
        s = generator.integers(0, seeds, seeds)
        i = generator.integers(0, clusters, clusters)
        stats[k] = np.mean(sums[s][:, i].sum(1) / counts[s][:, i].sum(1))
    low, high = np.quantile(stats, quantiles)
    return {"mean": float(per_seed.mean()), "interval": [float(low), float(high)], "seeds": seeds,
            "clusters": clusters, "items": int(values.shape[1]), "per_seed": [float(x) for x in per_seed],
            "label": "DESCRIPTIVE"}
