"""Issue #115: predicate-feedback robustness of the CNN hybrid predictor.

Runtime feedback of a description request is ``symbolic_features`` of the current carrier fed through
the request's adapter (``cnn_hybrid.CNNHybridPredictor.carrier``). This module separates that feedback
from the carrier step so it can be intervened on, and adds the training-side variants:

- ``carry(model, z, action, pair, features)``: the carrier step with given adapter input, op for op the
  body of ``CNNHybridPredictor.carrier`` (``features`` = the model's own feedback reproduces it bitwise);
- ``gate``: ``symbolic_features`` from (logits, availability logits), optionally through a binary
  straight-through bottleneck;
- predictor variants with unchanged adapter dimensions: ``DeltaMacroPredictor`` (the macro head reads the
  #113 observable predicates per request Delta: 4 outputs per Delta), ``BinaryMicroPredictor`` and
  ``BinaryDeltaMacroPredictor`` (label-free discrete bottleneck: binary codes over slot pairs / macro
  predicates, trained only through the prediction loss);
- ``SymbolicSpec`` and ``local_loss``: the ``decision_dynamics.local_loss`` recipe with per-description
  label use (``labels`` / ``none``), Delta-indexed macro labels and per-position label weights (low-label);
  the default spec reproduces ``decision_dynamics.local_loss`` op for op;
- label corruption (symmetric flips), the #100 readout counts, the ``Feedback`` interventions (default,
  oracle, flip, Gaussian logit noise, removed, shuffled, temporal head) and ``rollout_endpoints``
  (``request_selectors.rollout_endpoints`` op for op, with the feedback hook).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.training import decision_dynamics as dd
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import DIM, PAIRS, SLOTS, CNNHybridPredictor
from world_model.training.matched_dynamics import linear_macs, pairs_for
from world_model.training.request_selectors import NO_POLICY_KINDS, SELECTOR_KINDS, select_requests

DELTAS = (1, 5, 15)
READ_THRESHOLD = 0.5  # the #100 readout threshold on gated values
MICRO, MACRO, CONTINUOUS = Abstraction.MICRO, Abstraction.MACRO, Abstraction.CONTINUOUS
MICRO_WIDTH = SLOTS * SLOTS * 2 + 2
MACRO_WIDTH = 4
FEEDBACK_KINDS = ("default", "oracle", "flip", "gauss", "removed", "shuffled", "temporal")


def upper_triangle(device="cpu"):
    return torch.triu(torch.ones(SLOTS, SLOTS, dtype=torch.bool, device=device), 1)


# ---------------------------------------------------------------------------
# feedback and the carrier step
# ---------------------------------------------------------------------------

def binary_ste(probability):
    """Hard {0, 1} code (>= 0.5) forward, identity gradient backward; forward value exactly the code."""
    hard = (probability >= READ_THRESHOLD).to(probability.dtype)
    return hard + (probability - probability.detach())


def gate(z, mode, logits, availability, binary=False):
    """``CNNHybridPredictor.symbolic_features`` from given logits (same ops in the same order)."""
    available = availability.sigmoid()
    values = logits.sigmoid()
    if binary:
        available = binary_ste(available)
        values = binary_ste(values)
    if mode == MICRO:
        presence = z[:, 2::13].clamp(0, 1)
        active = presence[:, :, None] * presence[:, None, :]
        active = active * (1 - torch.eye(SLOTS, device=z.device))
        values = values * active[..., None] * available[:, None, None, :]
    else:
        values = values * available
    return torch.cat((values.flatten(1), available), -1)


def carry(model, z, action, pair, features):
    """The body of ``CNNHybridPredictor.carrier`` with the adapter input given (None for continuous)."""
    if pair not in PAIRS or z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
        raise ValueError("invalid trained pair/carrier/action contract")
    hidden = model.input_projection(torch.cat((z, action), -1))
    if pair.abstraction == MICRO:
        hidden = hidden + model.micro_adapter(features)
    elif pair.abstraction == MACRO:
        hidden = hidden + model.macro_adapter(features)
    code = model.conditioner.code(pair, len(z), z.device)
    for block in model.blocks:
        hidden = block(hidden, code)
    return z + model.output_projection(model.output_norm(hidden))


class VariantPredictor(CNNHybridPredictor):
    """``CNNHybridPredictor`` with a Delta-indexed macro head and/or binary bottleneck codes. Adapter
    dimensions, trunk and micro head are unchanged; ``carrier`` = ``carry`` with ``symbolic_features``."""

    DELTA_MACRO = False
    BINARY: frozenset = frozenset()

    def __init__(self):
        super().__init__()
        if self.DELTA_MACRO:
            self.macro_head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, 4 * len(DELTAS)))

    def symbols(self, z, mode, delta=None):
        if Abstraction(mode) == MACRO and self.DELTA_MACRO:
            if delta is None:
                raise ValueError("the Delta-indexed macro head needs the request Delta")
            raw = self.macro_head(z).unflatten(-1, (len(DELTAS), 4))[:, DELTAS.index(int(delta))]
            return raw[:, :2], raw[:, 2:]
        return super().symbols(z, mode)

    def symbolic_features(self, z, mode, delta=None):
        logits, availability = self.symbols(z, mode, delta)
        return gate(z, Abstraction(mode), logits, availability, binary=Abstraction(mode).value in self.BINARY)

    def carrier(self, z, action, pair):
        features = None if pair.abstraction == CONTINUOUS else self.symbolic_features(z, pair.abstraction, pair.delta)
        return carry(self, z, action, pair, features)


class BinaryMicroPredictor(VariantPredictor):
    BINARY = frozenset({"micro"})


class DeltaMacroPredictor(VariantPredictor):
    DELTA_MACRO = True


class BinaryDeltaMacroPredictor(VariantPredictor):
    DELTA_MACRO = True
    BINARY = frozenset({"macro"})


PREDICTOR_CLASSES = {"hybrid": CNNHybridPredictor, "binary_micro": BinaryMicroPredictor,
                     "delta_macro": DeltaMacroPredictor, "binary_delta_macro": BinaryDeltaMacroPredictor}


def own_symbols(model, z, pair):
    if isinstance(model, VariantPredictor):
        return model.symbols(z, pair.abstraction, pair.delta)
    return model.symbols(z, pair.abstraction)


def own_features(model, z, pair):
    if isinstance(model, VariantPredictor):
        return model.symbolic_features(z, pair.abstraction, pair.delta)
    return model.symbolic_features(z, pair.abstraction)


def is_binary(model, mode):
    return isinstance(model, VariantPredictor) and Abstraction(mode).value in model.BINARY


# ---------------------------------------------------------------------------
# training loss: decision_dynamics.local_loss with per-description label use
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolicSpec:
    """micro / macro: ``labels`` (symbolic loss on the batch tensors) or ``none`` (no symbolic term: the
    head is trained only through the prediction loss); macro_delta: batch macros are [..., 2, len(DELTAS)]
    and the request's Delta column is supervised; weighted: descriptions whose symbolic terms are weighted
    by ``batch['<mode>_labelled']`` (positions without labels contribute nothing, availability included)."""

    micro: str = "labels"
    macro: str = "labels"
    macro_delta: bool = False
    weighted: frozenset = field(default_factory=frozenset)

    def uses(self, mode):
        return getattr(self, Abstraction(mode).value) == "labels"


DEFAULT_SPEC = SymbolicSpec()


class _DeltaSymbols(nn.Module):
    def __init__(self, model, mode, delta):
        super().__init__()
        self.m = model
        self.mode = mode
        self.delta = delta

    def forward(self, z):
        return self.m.symbols(z, self.mode, self.delta)


def ensemble_symbols(ensemble, pair, z, spec):
    if pair.abstraction == MACRO and spec.macro_delta:
        key = (pair.abstraction, int(pair.delta))
        return ensemble._function("symbols", key, lambda: _DeltaSymbols(ensemble.base, pair.abstraction,
                                                                        int(pair.delta)))(ensemble._prefixed, z)
    return ensemble.symbols(pair.abstraction, z)


def symbolic_term(ensemble, z, batch, position, keep, pair, spec=DEFAULT_SPEC):
    """``decision_dynamics.symbolic_term`` per member, with the spec's label selection and weights."""
    mode = pair.abstraction
    if mode == CONTINUOUS:
        return torch.zeros(len(z), device=z.device)
    logits, availability = ensemble_symbols(ensemble, pair, z, spec)
    name = "relations" if mode == MICRO else "macros"
    labels = batch[name][:, :, position].float()
    mask = batch[name + "_mask"][:, :, position].bool()
    if mode == MACRO and spec.macro_delta:
        column = DELTAS.index(int(pair.delta))
        labels, mask = labels[..., column], mask[..., column]
    available = (mask.any(2).any(2) if mode == MICRO else mask).float()
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none",
                                             pos_weight=torch.tensor(dd.SYMBOLIC_POS_WEIGHT, device=z.device))
    rows = keep
    if mode.value in spec.weighted:
        rows = keep * batch[f"{mode.value}_labelled"][:, :, position].to(keep.dtype)
    weight = mask.float() * rows.view(rows.shape + (1,) * (mask.ndim - 2))
    labelled = (raw * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1)
    availability_loss = F.binary_cross_entropy_with_logits(availability, available, reduction="none").mean(-1)
    return labelled + dd._row_mean(availability_loss, rows)


def local_loss(ensemble, batch, pair, spec=DEFAULT_SPEC):
    """``decision_dynamics.local_loss`` (four local transitions, their recursion, carrier bound, symbolic
    terms at weight 0.1) with the spec's label use; DEFAULT_SPEC reproduces it op for op."""
    z, action, length = batch["z"], batch["action"], batch["length"]
    symbolic = ensemble.hybrid and pair.abstraction != CONTINUOUS and spec.uses(pair.abstraction)
    local_terms, recursive_terms, valid_steps = [], [], []
    current = None
    for step in range(dd.LOCAL_STEPS):
        start, end = step * pair.delta, (step + 1) * pair.delta
        if end >= z.shape[2]:
            break
        keep = (length >= end).to(z.dtype)
        valid_steps.append((keep.sum(1) > 0).to(z.dtype))
        local = ensemble.transition(pair, z[:, :, start], action)
        target = z[:, :, end]
        term = dd._row_mean((local - target).square().mean(-1), keep)
        if symbolic:
            term = term + dd.SYMBOLIC_WEIGHT * symbolic_term(ensemble, local, batch, end, keep, pair, spec)
            term = term + dd.SYMBOLIC_WEIGHT * symbolic_term(ensemble, z[:, :, start], batch, start, keep, pair, spec)
        local_terms.append(term)
        current = local if step == 0 else ensemble.transition(pair, current, action)
        recursive_terms.append(dd._row_mean((current - target).square().mean(-1), keep)
                               + dd.BOUND_WEIGHT * dd._row_mean(F.relu(current.abs() - dd.BOUND_LIMIT).square().mean(-1),
                                                                keep))
    valid = torch.stack(valid_steps)
    steps = valid.sum(0).clamp_min(1)
    return ((torch.stack(local_terms) * valid).sum(0) / steps
            + (torch.stack(recursive_terms) * valid).sum(0) / steps)


# ---------------------------------------------------------------------------
# label corruption (symmetric flips on labelled entries)
# ---------------------------------------------------------------------------

def flip_relations(relations, mask, p, generator, chunk=1 << 16):
    """Flip each labelled edge with probability p (0 -> 1 and 1 -> 0 alike). Contact is symmetric: the
    unordered pair is drawn once (i < j) and mirrored; supports are drawn per directed pair. Only entries
    with ``mask`` are flipped. Returns (flipped relations, {flipped, eligible}) with contact counted once
    per unordered pair."""
    out = relations.clone()
    upper = upper_triangle(relations.device)
    flipped = eligible = 0
    for begin in range(0, len(relations), chunk):
        part = slice(begin, begin + chunk)
        draw = torch.rand(out[part].shape, generator=generator, device=relations.device) < p
        contact = draw[..., 0] & upper
        contact = contact | contact.transpose(1, 2)
        flips = torch.stack((contact, draw[..., 1]), -1) & mask[part]
        out[part] ^= flips
        flipped += int((flips[..., 0] & upper).sum()) + int(flips[..., 1].sum())
        eligible += int((mask[part][..., 0] & upper).sum()) + int(mask[part][..., 1].sum())
    return out, {"flipped": flipped, "eligible": eligible}


def flip_values(values, available, p, generator, chunk=1 << 20):
    """Flip each available predicate value with probability p (macro labels)."""
    out = values.clone()
    flipped = eligible = 0
    for begin in range(0, len(values), chunk):
        part = slice(begin, begin + chunk)
        flips = (torch.rand(out[part].shape, generator=generator, device=values.device) < p) & available[part]
        out[part] ^= flips
        flipped += int(flips.sum())
        eligible += int(available[part].sum())
    return out, {"flipped": flipped, "eligible": eligible}


# ---------------------------------------------------------------------------
# readout counts (#100 rule: gated value >= 0.5; contact on the upper triangle)
# ---------------------------------------------------------------------------

def micro_counts(values, relations, mask):
    """[2, 3] (tp, fp, fn) of contact (upper triangle) and supports: values [B, S, S, 2] gated values,
    relations / mask [B, S, S, 2] bool engine labels."""
    predicted = values >= READ_THRESHOLD
    upper = upper_triangle(values.device)
    out = []
    for channel, region in ((0, mask[..., 0] & upper), (1, mask[..., 1])):
        p, t = predicted[..., channel][region], relations[..., channel][region]
        out.append(torch.stack(((p & t).sum(), (p & ~t).sum(), (~p & t).sum())))
    return torch.stack(out)


def macro_counts(values, truth, available):
    """[2, 3] (tp, fp, fn) per macro predicate: values [B, 2] gated, truth / available [B, 2] bool."""
    predicted = values >= READ_THRESHOLD
    out = []
    for k in range(values.shape[1]):
        region = available[:, k]
        p, t = predicted[:, k][region], truth[:, k][region]
        out.append(torch.stack(((p & t).sum(), (p & ~t).sum(), (~p & t).sum())))
    return torch.stack(out)


def f1_from(counts):
    """F1 of (tp, fp, fn) along the last axis (nan when nothing is positive in either)."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    return torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), torch.full_like(tp, math.nan))


def macro_available(positions, delta):
    """The #113 positional availability at landing frames ``positions`` for request Delta:
    settle iff f >= max(1, Delta) (debounce 1), structure-change iff f >= Delta. -> [B, 2] bool."""
    return torch.stack((positions >= max(1, int(delta)), positions >= int(delta)), -1)


# ---------------------------------------------------------------------------
# feedback interventions
# ---------------------------------------------------------------------------

class BlockTruth:
    """Engine labels of a block's candidate branches, positions 0..T-1 (clamped at each branch's last)."""

    def __init__(self, last, relations=None, relation_mask=None, macro_values=None, macro_available_=None):
        self.last = last
        self.relations = relations
        self.relation_mask = relation_mask
        self.macro_values = macro_values
        self.macro_available = macro_available_

    def index(self, rows, positions):
        return torch.minimum(positions, self.last[rows])

    def micro(self, rows, positions):
        t = self.index(rows, positions)
        return self.relations[rows, t], self.relation_mask[rows, t], int((positions > self.last[rows]).sum())

    def macro(self, rows, positions, delta):
        t = self.index(rows, positions)
        column = DELTAS.index(int(delta))
        return (self.macro_values[rows, t][..., column], self.macro_available[rows, t][..., column],
                int((positions > self.last[rows]).sum()))


def truth_features(mode, labels, mask):
    """Engine labels through the adapter's input layout: micro values = label & mask with per-channel
    availability = any labelled pair; macro values = label & available with availability."""
    if mode == MICRO:
        values = (labels & mask).to(torch.float32)
        available = mask.flatten(1, 2).any(1).to(torch.float32)
        return torch.cat((values.flatten(1), available), -1)
    return torch.cat(((labels & mask).to(torch.float32), mask.to(torch.float32)), -1)


class Feedback:
    """One intervention on the feedback of ``mode`` requests (requests of the other description keep the
    model's own feedback). ``features`` returns the adapter input for rows ``rows`` of the full batch
    ``z_all`` at their rollout positions and accumulates the #100 readout counts of the delivered feedback
    against the branch's engine labels at the matching frame (when ``truth`` is given).

    kinds: default (the model's heads), oracle (engine labels), flip (engine labels with symmetric flips at
    rate p, drawn per call from ``generator``), gauss (the model's logits + sigma N(0, 1)), removed (zero
    adapter input), shuffled (the model's feedback of ``partner[row]``, another candidate of the same
    anchor), temporal (macro: the #113 temporal head on the row's stride-Delta rollout history; with
    ``thresholds`` [2, len(DELTAS)] the delivered value is the head's binary reading sigmoid >= threshold)."""

    def __init__(self, kind, mode, *, truth=None, p=None, sigma=None, generator=None, partner=None,
                 temporal=None, thresholds=None, rows=None, endpoint=rp.ENDPOINT, device="cuda"):
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"unknown feedback kind {kind}")
        if kind in ("oracle", "flip") and truth is None:
            raise ValueError(f"{kind} feedback needs engine labels")
        if kind == "temporal" and (mode != MACRO or temporal is None):
            raise ValueError("temporal feedback is the macro temporal head")
        self.kind, self.mode = kind, Abstraction(mode)
        self.truth, self.p, self.sigma, self.generator = truth, p, sigma, generator
        self.partner, self.temporal, self.thresholds = partner, temporal, thresholds
        self.counts = torch.zeros(len(DELTAS), 2, 3, dtype=torch.long, device=device)
        self.calls = torch.zeros(len(DELTAS), dtype=torch.long, device=device)
        self.flipped = self.eligible = self.clamped = self.nonzero = 0
        if kind == "temporal":
            self.trajectory = torch.zeros(rows, endpoint + 1, DIM, device=device)
            self.latest = torch.full((rows, endpoint + 1), -1, dtype=torch.long, device=device)

    def observe(self, z, index, positions):
        """Record the current carriers of the alive rows (temporal history)."""
        if self.kind != "temporal":
            return
        t = positions[index]
        self.trajectory[index, t] = z[index]
        span = torch.arange(self.latest.shape[1], device=z.device)
        update = (span[None, :] >= t[:, None])
        self.latest[index] = torch.where(update, t[:, None], self.latest[index])

    def _engine(self, rows, positions, pair):
        if self.mode == MICRO:
            labels, mask, clamped = self.truth.micro(rows, positions)
        else:
            labels, mask, clamped = self.truth.macro(rows, positions, pair.delta)
        return labels, mask, clamped

    def features(self, model, z_all, rows, positions, pair):
        if pair.abstraction != self.mode:
            return own_features(model, z_all[rows], pair)
        z = z_all[rows]
        if self.kind == "default":
            out = own_features(model, z, pair)
        elif self.kind == "removed":
            out = torch.zeros(len(rows), MICRO_WIDTH if self.mode == MICRO else MACRO_WIDTH, device=z.device)
        elif self.kind == "shuffled":
            out = own_features(model, z_all[self.partner[rows]], pair)
        elif self.kind == "gauss":
            logits, availability = own_symbols(model, z, pair)
            noise = torch.randn(logits.shape, generator=self.generator, device=z.device)
            if self.mode == MICRO:  # contact logits are symmetric: one draw per unordered pair, mirrored
                contact = torch.triu(noise[..., 0], 1)
                noise = torch.stack((contact + contact.transpose(1, 2), noise[..., 1]), -1)
            out = gate(z, self.mode, logits + self.sigma * noise, availability, binary=is_binary(model, self.mode))
        elif self.kind in ("oracle", "flip"):
            labels, mask, _ = self._engine(rows, positions, pair)
            if self.kind == "flip":
                if self.mode == MICRO:
                    labels, tally = flip_relations(labels, mask, self.p, self.generator)
                else:
                    labels, tally = flip_values(labels, mask, self.p, self.generator)
                self.flipped += tally["flipped"]
                self.eligible += tally["eligible"]
            out = truth_features(self.mode, labels, mask)
        else:  # temporal
            k, delta = self.temporal.history, int(pair.delta)
            offsets = torch.arange(k - 1, -1, -1, device=z.device) * delta
            wanted = (positions[:, None] - offsets[None, :]).clamp_min(0)
            history = self.trajectory[rows[:, None], self.latest[rows[:, None], wanted]]
            probability = torch.sigmoid(self.temporal(history, delta))
            if self.thresholds is not None:
                probability = (probability >= self.thresholds[:, DELTAS.index(delta)]).to(probability.dtype)
            available = macro_available(positions, delta).to(probability.dtype)
            out = torch.cat((probability * available, available), -1)
        if self.kind == "removed":
            self.nonzero += int((out != 0).sum())
        if self.truth is not None:
            labels, mask, clamped = self._engine(rows, positions, pair)
            self.clamped += clamped
            d = DELTAS.index(int(pair.delta))
            self.calls[d] += len(rows)
            if self.mode == MICRO:
                self.counts[d] += micro_counts(out[:, :-2].reshape(len(rows), SLOTS, SLOTS, 2), labels, mask)
            else:
                self.counts[d] += macro_counts(out[:, :2], labels, mask)
        return out

    def summary(self):
        return {"kind": self.kind, "mode": self.mode.value, "counts": self.counts.cpu().tolist(),
                "calls": self.calls.cpu().tolist(), "flipped": self.flipped, "eligible": self.eligible,
                "clamped_calls": self.clamped, "removed_nonzero": self.nonzero}


# ---------------------------------------------------------------------------
# rollout (request_selectors.rollout_endpoints with the feedback hook)
# ---------------------------------------------------------------------------

def rollout_endpoints(model, policy, z0, actions, spec, *, select: Callable | None = None, feedback=None,
                      endpoint=rp.ENDPOINT, record_schedules=False):
    """``request_selectors.rollout_endpoints`` op for op (kinds fixed, schedule, joint/temporal/description
    via ``select``, restricted/monotone), with every description step's adapter input taken from
    ``feedback.features`` when ``feedback`` is given."""
    if spec["kind"] == "replay":
        raise ValueError("replay is not a feedback-rollout kind")
    pairs = policy.pairs if policy is not None else pairs_for(model)
    count = len(actions)
    z = z0[None].to(actions.device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    previous = torch.full((count,), -1, dtype=torch.long, device=z.device)
    pair_macs = [linear_macs(model, pair) for pair in pairs]
    transition_calls = controller_calls = macs = 0
    trace = []
    iteration = 0
    if z.is_cuda:
        torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            if spec["kind"] == "fixed":
                chosen = torch.full((len(index),), spec["pair_index"], dtype=torch.long, device=z.device)
            elif spec["kind"] == "schedule":
                chosen = torch.full((len(index),), spec["schedule"][iteration], dtype=torch.long, device=z.device)
            else:
                logits = policy(z[index], actions[index], endpoint-positions[index])
                if spec["kind"] in SELECTOR_KINDS:
                    chosen = select_requests(logits, spec, pairs, previous[index])
                else:
                    chosen = select(spec["kind"], logits, spec)
                controller_calls += len(index)
            if record_schedules:
                trace.append((index.cpu(), chosen.cpu()))
            previous[index] = chosen
            if feedback is not None:
                feedback.observe(z, index, positions)
            z_next = z
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next = z_next.clone()
                if feedback is None:
                    z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                else:
                    features = (None if pair.abstraction == CONTINUOUS
                                else feedback.features(model, z, rows, positions[rows], pair))
                    z_next[rows] = carry(model, z[rows], actions[rows], pair, features)
                transition_calls += len(rows)
                macs += len(rows)*pair_macs[pair_index]
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < endpoint)
            iteration += 1
    if z.is_cuda:
        torch.cuda.synchronize()
    macs += controller_calls*rp.policy_macs(policy if spec["kind"] not in NO_POLICY_KINDS else None)
    timing = {"transition_calls": transition_calls, "controller_calls": controller_calls, "linear_macs": macs,
              "gpu_seconds": time.monotonic()-began}
    if record_schedules:
        sequences = [[] for _ in range(count)]
        for index, chosen in trace:
            for r, c in zip(index.tolist(), chosen.tolist()):
                sequences[r].append(c)
        timing["schedules"] = sequences
    return z, timing


# ---------------------------------------------------------------------------
# calibration and the micro perception head
# ---------------------------------------------------------------------------

def bisect_log(f1_at, target, *, low=-2.0, high=2.0, iterations=14):
    """sigma = 10 ** x on [10 ** low, 10 ** high] with f1_at(sigma) closest to ``target`` by bisection,
    assuming f1_at decreases in sigma. Returns {sigma, f1, reached, trace}; ``reached`` is False when the
    target lies outside [f1_at(10 ** high), f1_at(10 ** low)] (the nearer end is returned)."""
    trace = []

    def at(x):
        value = float(f1_at(10.0 ** x))
        trace.append({"sigma": 10.0 ** x, "f1": value})
        return value

    top, bottom = at(low), at(high)
    if target >= top:
        return {"sigma": 10.0 ** low, "f1": top, "reached": target == top, "trace": trace}
    if target <= bottom:
        return {"sigma": 10.0 ** high, "f1": bottom, "reached": target == bottom, "trace": trace}
    lo, hi = low, high
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if at(mid) > target:
            lo = mid
        else:
            hi = mid
    best = min(trace, key=lambda r: (abs(r["f1"] - target), r["sigma"]))
    return {"sigma": best["sigma"], "f1": best["f1"], "reached": True, "trace": trace}


class MicroPerceptionHead(nn.Module):
    """A micro predicate head on the observed carrier: the predictor's micro head architecture and
    readout (contact symmetrised, supports directed, two availability logits)."""

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, MICRO_WIDTH))

    def forward(self, z):
        raw = self.head(z)
        relations = raw[:, :-2].reshape(-1, SLOTS, SLOTS, 2)
        contact = (relations[..., 0] + relations[..., 0].transpose(1, 2)) / 2
        return torch.stack((contact, relations[..., 1]), -1), raw[:, -2:]


def perception_loss(head, z, relations, mask):
    """The predictor's symbolic loss (masked BCE, pos_weight 5, plus availability BCE) on one batch."""
    logits, availability = head(z)
    raw = F.binary_cross_entropy_with_logits(logits, relations.float(), reduction="none",
                                             pos_weight=torch.tensor(dd.SYMBOLIC_POS_WEIGHT, device=z.device))
    return ((raw * mask).sum() / mask.sum().clamp_min(1)
            + F.binary_cross_entropy_with_logits(availability, mask.flatten(1, 2).any(1).float()))


def perception_labels(head, z, thresholds):
    """Pseudo-labels of the frozen micro perception head: relations = sigmoid(logit) >= threshold per
    channel; mask = both slots present in the carrier (presence >= 0.5, i != j) and the head's channel
    availability >= 0.5. -> (relations, mask) bool [B, S, S, 2]."""
    logits, availability = head(z)
    probability = logits.sigmoid()
    relations = torch.stack([probability[..., c] >= thresholds[c] for c in range(2)], -1)
    presence = z[:, 2::13] >= READ_THRESHOLD
    active = presence[:, :, None] & presence[:, None, :] & ~torch.eye(SLOTS, dtype=torch.bool, device=z.device)
    mask = active[..., None] & (availability.sigmoid() >= READ_THRESHOLD)[:, None, None, :]
    return relations & mask, mask


__all__ = ["DELTAS", "READ_THRESHOLD", "FEEDBACK_KINDS", "binary_ste", "gate", "carry", "VariantPredictor",
           "BinaryMicroPredictor", "DeltaMacroPredictor", "BinaryDeltaMacroPredictor", "PREDICTOR_CLASSES",
           "own_symbols", "own_features", "is_binary", "SymbolicSpec", "DEFAULT_SPEC", "symbolic_term",
           "local_loss", "flip_relations", "flip_values", "micro_counts", "macro_counts", "f1_from",
           "macro_available", "BlockTruth", "truth_features", "Feedback", "rollout_endpoints", "bisect_log",
           "MicroPerceptionHead", "perception_loss", "perception_labels", "PredictionPair"]
