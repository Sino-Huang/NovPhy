"""Issue #116: declared request sets R beyond R_9 = {1, 5, 15} x alpha.

The request set was hard-coded in three places: ``cnn_hybrid.PAIRS`` (and ``carrier()``'s membership test),
the controller's 9-way output layer, and the 9-column DP cost tensor. ``cnn_hybrid`` and
``request_policy`` are hash-pinned by earlier plans, so this sibling module generalizes them. With
``REQUEST_SETS["R_9"]`` every function here reproduces the #112 v3 LC / #101 path bit for bit (#116 smoke):

- ``RequestSet``: a named Delta grid x the three descriptions, Delta-major (R_9 order = ``PAIRS``).
- ``predictor_class(request_set)``: ``CNNHybridPredictor`` whose ``carrier`` accepts exactly R (same
  parameters, state dict and ops; ``PairConditioner`` already encodes any integer Delta sinusoidally).
- Training (#112 v3 LC recipe): ``RequestSetEnsemble`` (v3 skipping ensemble over R), ``schedule`` (round
  robin over R; two budgets), ``curriculum_step`` (the R_9 curriculum clock), ``window_positions`` (local
  windows of ``max(60, LOCAL_STEPS * Delta)`` transitions so every Delta keeps its four local transitions),
  ``long_and_rank_loss`` (v3 truncated recursion with the endpoint rule) and ``train_member``.
- Endpoint rule (``endpoint_schedule``): a fixed request (Delta, alpha) makes floor(E / Delta) transitions,
  then, if Delta does not divide E, the #101 same-alpha fallback (the largest Delta' in R with Delta' <=
  remaining) until E; every rollout ends exactly at E = 225. For Delta | E it is the fixed request.
- Policy heads: ``request_policy.RequestPolicy`` (flat |R|-way, the current design), ``FactorisedPolicy``
  (Delta head x alpha head) and ``OrdinalPolicy`` (cumulative-logit Delta head x alpha head). Every head
  returns per-request scores with Delta > remaining at -inf, so ``argmax`` is the greedy selection and
  ``request_policy.distil_round`` (cross-entropy) trains all three (for the factorised and ordinal heads
  the joint log-probabilities are normalized, so the loss is the sum of the factor NLLs).
- DP labelling: ``prediction_error_costs`` (``request_policy.prediction_error_labels`` cost op for op),
  ``backward_dp`` (vectorized over R, equal to the per-request loop), ``dagger_contexts`` (batched #77
  DAgger), ``window_labels`` (chunked over windows), ``pruned_deltas`` (log-spaced Delta grid) and the label
  statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math
import os
import time
from typing import Mapping, Sequence

import torch
import torch._dynamo
from torch import nn
from torch.nn import functional as F

from world_model.model import Abstraction, PredictionPair
from world_model.training import decision_dynamics as dd
from world_model.training import request_policy as rp
from world_model.training import stable_decision_dynamics as sd
from world_model.training.cnn_hybrid import DIM, PAIRS, CNNHybridPredictor
from world_model.training.matched_dynamics import work

ENDPOINT = rp.ENDPOINT                 # 225 observed frames
BASE_WINDOW = 60                       # the #77 / #99 local window (s99.WINDOW_LENGTH)
LOCAL_STEPS = dd.LOCAL_STEPS           # four local transitions per update
REFERENCE_UPDATES = 9000               # the #112 v3 / Gate-B update count (curriculum clock)
UPDATES_PER_REQUEST = REFERENCE_UPDATES // len(PAIRS)
BUDGETS = ("total", "request")
ALPHAS = tuple(Abstraction)
PATIENCE = 10                          # v3 skipping ensemble: retire after 10 consecutive skips
TRUNCATION = 45                        # v3 truncated back-propagation
DELTA_GRIDS = {
    "R_9": (1, 5, 15),
    "R_18": (1, 2, 3, 5, 10, 15),
    "R_30": (1, 2, 3, 5, 8, 10, 15, 25, 45, 75),
    "R_cont": tuple(range(1, 76)),
}


# ---------------------------------------------------------------------------
# request sets and indexing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestSet:
    """A Delta grid x the three descriptions, Delta-major: index = delta_index * 3 + alpha_index."""

    name: str
    deltas: tuple

    def __post_init__(self):
        if not self.deltas or list(self.deltas) != sorted(set(self.deltas)) or self.deltas[0] != 1:
            raise ValueError("a request set's Delta grid is strictly increasing and starts at 1 (always feasible)")
        if any(type(d) is not int or d < 1 or d > ENDPOINT for d in self.deltas):
            raise ValueError("every Delta is a positive integer number of carrier frames <= the endpoint")

    @functools.cached_property
    def pairs(self) -> tuple:
        return tuple(PredictionPair(delta, alpha) for delta in self.deltas for alpha in ALPHAS)

    @property
    def size(self) -> int:
        return len(self.pairs)

    def index(self, pair: PredictionPair) -> int:
        return self.pairs.index(pair)

    def delta_of(self, index: int) -> int:
        return self.deltas[index // len(ALPHAS)]

    def alpha_of(self, index: int) -> Abstraction:
        return ALPHAS[index % len(ALPHAS)]

    def request_name(self, index: int) -> str:
        pair = self.pairs[index]
        return f"F-{pair.delta}-{pair.abstraction}"


REQUEST_SETS = {name: RequestSet(name, deltas) for name, deltas in DELTA_GRIDS.items()}


def request_set(name: str) -> RequestSet:
    if name not in REQUEST_SETS:
        raise ValueError(f"undeclared request set {name}")
    return REQUEST_SETS[name]


def pruned_deltas(max_delta: int = DELTA_GRIDS["R_cont"][-1], count: int = 12) -> tuple:
    """Log-spaced Delta grid: round-half-up of ``count`` geometric points from 1 to ``max_delta``, unique."""
    ratio = max_delta ** (1 / (count - 1))
    return tuple(sorted({int(math.floor(ratio ** k + 0.5)) for k in range(count - 1)} | {max_delta}))


def sub_requests(request_set_: RequestSet, deltas: Sequence[int]) -> list:
    """Indices into ``request_set_.pairs`` of the requests whose Delta is in ``deltas`` (order kept)."""
    keep = set(deltas)
    return [i for i, pair in enumerate(request_set_.pairs) if pair.delta in keep]


# ---------------------------------------------------------------------------
# predictor over R
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def predictor_class(request_set_: RequestSet):
    """``CNNHybridPredictor`` whose ``carrier`` accepts exactly the pairs of R (same parameters and ops).
    One class per set: ``decision_dynamics.Ensemble`` caches its meta base module per class name."""

    def carrier(self, z, action, pair):
        if pair not in self.pairs or z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("invalid declared pair/carrier/action contract")
        hidden = self.input_projection(torch.cat((z, action), -1))
        if pair.abstraction == Abstraction.MICRO:
            hidden = hidden + self.micro_adapter(self.symbolic_features(z, pair.abstraction))
        elif pair.abstraction == Abstraction.MACRO:
            hidden = hidden + self.macro_adapter(self.symbolic_features(z, pair.abstraction))
        code = self.conditioner.code(pair, len(z), z.device)
        for block in self.blocks:
            hidden = block(hidden, code)
        return z + self.output_projection(self.output_norm(hidden))

    name = f"RequestSetPredictor_{request_set_.name}"
    return type(name, (CNNHybridPredictor,), {"pairs": request_set_.pairs, "request_set": request_set_,
                                              "carrier": carrier, "__module__": __name__,
                                              "__doc__": f"CNNHybridPredictor over the declared set {request_set_.name}"})


def load_predictor(request_set_: RequestSet, state: Mapping, device="cuda"):
    model = predictor_class(request_set_)()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# training: schedules, curriculum clock, windows, the endpoint rule
# ---------------------------------------------------------------------------

def budget_updates(request_set_: RequestSet, budget: str) -> int:
    """equal total updates (9000, the Gate-B count) or equal updates per request (1000 x |R|)."""
    if budget == "total":
        return REFERENCE_UPDATES
    if budget == "request":
        return UPDATES_PER_REQUEST * request_set_.size
    raise ValueError(f"unknown budget {budget!r}")


def schedule(request_set_: RequestSet, budget: str) -> list:
    """Pair index of every update: round robin over R (R_9: ``PAIRS[step % 9]``, the #77 / #112 schedule)."""
    return [update % request_set_.size for update in range(budget_updates(request_set_, budget))]


def curriculum_step(update: int, total: int) -> int:
    """The R_9 curriculum clock: update u of a ``total``-update run reads the horizon curriculum and the ranking
    ramp at u * 9000 // total, so each request's k-th update meets the curriculum of R_9's k-th update of it."""
    return update * REFERENCE_UPDATES // total


def skip_cap(total: int, reference_cap: int) -> int:
    """The retrain's skip cap (45 of 9000 updates) at the same rate."""
    return reference_cap * total // REFERENCE_UPDATES


def local_window(delta: int) -> int:
    """Transitions of the local window an update of ``delta`` reads: the #77 window, extended to LOCAL_STEPS
    transitions of Delta when 60 frames hold fewer."""
    return max(BASE_WINDOW, LOCAL_STEPS * delta)


def window_positions(starts: torch.Tensor, offsets: torch.Tensor, lasts: torch.Tensor, delta: int):
    """Positions (into the concatenated carriers) and lengths of windows at ``starts`` of shots with ``offsets``
    / ``lasts``: ``offset + start + min(i, length)``, i = 0..L, length = min(L, last - start), L =
    ``local_window(delta)`` (the s99 window rule with terminal absorption)."""
    span = local_window(delta)
    length = torch.clamp(lasts - starts, max=span)
    steps = torch.arange(span + 1, device=starts.device)
    return offsets[..., None] + starts[..., None] + torch.minimum(steps, length[..., None]), length


@functools.lru_cache(maxsize=None)
def _endpoint_schedules(pairs: tuple, endpoint: int) -> tuple:
    table = rp.fallback_table(pairs, endpoint).tolist()
    out = []
    for index in range(len(pairs)):
        schedule, position = [], 0
        while position < endpoint:
            chosen = table[index][endpoint - position]
            schedule.append(chosen)
            position += pairs[chosen].delta
        if position != endpoint:
            raise ValueError("the endpoint rule overshot the endpoint")
        out.append(tuple(schedule))
    return tuple(out)


def endpoint_schedule(pairs: Sequence[PredictionPair], index: int, endpoint: int = ENDPOINT) -> list:
    """Endpoint rule of a fixed request: floor(E / Delta) transitions of it, then the #101 same-alpha fallback
    (``request_policy.fallback_table``: the largest feasible Delta of that alpha) until exactly E."""
    return list(_endpoint_schedules(tuple(pairs), endpoint)[index])


class RequestSetEnsemble(sd.SkippingEnsemble):
    """The #112 v3 skipping ensemble whose transitions accept the members' declared set R."""

    patience = PATIENCE

    def __init__(self, models, **kwargs):
        super().__init__(models, **kwargs)
        self.pairs = tuple(models[0].pairs)


def long_and_rank_loss(ensemble, pair: PredictionPair, *, long: Mapping | None, rank: Mapping | None,
                       rank_pairs: Sequence[PredictionPair], horizon: int, presence: int, center: Sequence[int],
                       temperature: float, truncation: int = TRUNCATION):
    """``stable_decision_dynamics.truncated_long_and_rank_loss`` with the endpoint rule: the long rows make
    horizon // Delta transitions of ``pair`` (``horizon`` a multiple of Delta); the ranking rows follow
    ``rank_pairs`` (``endpoint_schedule``: ``pair`` repeated, then the fallback) to exactly the endpoint.
    For Delta | E the ops equal v3's one for one."""
    if truncation < 1:
        raise ValueError("truncation must be at least one transition")
    members = ensemble.size
    zero = torch.zeros(members, device=ensemble.device)
    long_steps = horizon // pair.delta if long is not None else 0
    if long is not None and (horizon % pair.delta or long["targets"].shape[2] != long_steps):
        raise ValueError("long-horizon targets must hold every delta multiple up to the horizon")
    schedule_ = list(rank_pairs) if rank is not None else []
    if rank is not None and (not schedule_ or any(p != pair for p in schedule_[:long_steps])):
        raise ValueError("the ranking schedule starts with the trained pair for every long-horizon step")
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
    if not states:
        return zero, zero
    z, action = torch.cat(states, 1), torch.cat(actions, 1)
    long_sum = zero
    for step in range(max(long_steps, len(schedule_))):
        if step == long_steps and long_rows and rank is not None:
            z, action, long_rows = z[:, long_rows:], action[:, long_rows:], 0
        if step and step % truncation == 0:
            z = z.detach()
        z = ensemble.transition(schedule_[step] if step < len(schedule_) else pair, z, action)
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


def train_member(request_set_: RequestSet, updates: Sequence[int], data, seed: int, *, recipe: Mapping,
                 long: Mapping, rank: Mapping, presence: int, center: Sequence[int], device="cuda",
                 progress=None, smoke_full=False, stop=None, step_times=False, every=1000, log=None):
    """One fold-'all' member under the #112 v3 LC recipe over R (v1 ``train_group`` op for op for R_9).

    updates: pair index per update (``schedule``); data: ``window(pools, generators, count, delta)``,
    ``long_batch``, ``rank_batch``, ``pools`` (``rt.IndexedTrainData`` API); recipe: s99 training recipe
    (batch_size, learning_rate, weight_decay, grad_clip); long / rank: the v3 LONG / RANK blocks;
    progress: resumable checkpoint path (every ``every`` updates). smoke_full: horizon 225 and the full
    ranking weight from update 0. stop: run updates [0, stop) only (the curriculum clock still reads
    ``len(updates)``); step_times: synchronized wall seconds of every update. Returns (report, ensemble,
    logs) with per-update local loss, pre-clip gradient norm and skip flag."""
    pairs = request_set_.pairs
    total = len(updates)
    # one compiled transition per (pair, rows): |R| = 225 needs more graphs than the decision_dynamics default
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 8 * dd.RECOMPILE_LIMIT)
    torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit,
                                                           32 * dd.RECOMPILE_LIMIT)
    torch.manual_seed(seed)
    model = predictor_class(request_set_)().to(device)
    ensemble = RequestSetEnsemble([model], learning_rate=recipe["learning_rate"], weight_decay=recipe["weight_decay"],
                                  grad_clip=recipe["grad_clip"], compile=True)
    del model
    pools = [data.pools("all")]
    window_generators = [torch.Generator().manual_seed(seed)]
    long_generators = [torch.Generator().manual_seed(seed + long["seed_offset"])]
    rank_generators = [torch.Generator().manual_seed(seed + rank["seed_offset"])]
    rank_schedules = {k: [pairs[j] for j in endpoint_schedule(pairs, k)] for k in sorted(set(updates))}
    logs = {"local": torch.full((total,), float("nan"), device=device),
            "grad_norm": torch.full((total,), float("nan"), device=device),
            "skipped": torch.zeros(total, dtype=torch.bool, device=device)}
    first, trace, gpu, first_seen = 0, [], 0.0, {}
    if progress is not None and progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        ensemble.load_state(saved["ensemble"])
        for generators, states in ((window_generators, saved["window"]), (long_generators, saved["long"]),
                                   (rank_generators, saved["rank"])):
            for generator, state in zip(generators, states):
                generator.set_state(state)
        first, trace, gpu, first_seen = saved["step"], saved["trace"], saved["gpu_seconds"], saved["first_seen"]
        logs = {key: value.to(device) for key, value in saved["logs"].items()}
        if log:
            log(f"resumed at update {first}")
    torch.cuda.synchronize()
    began = segment_began = time.monotonic()
    seconds = []
    for step in range(first, total if stop is None else stop):
        k = updates[step]
        pair = pairs[k]
        fresh = k not in first_seen
        if fresh or step_times:
            torch.cuda.synchronize()
            pair_began = time.monotonic()
        clock = curriculum_step(step, total)
        batch = data.windows(pools, window_generators, recipe["batch_size"], pair.delta)
        local = dd.local_loss(ensemble, batch, pair)
        weight = rank["weight"] * (1.0 if smoke_full else dd.ramp_at(clock, start=rank["start"], length=rank["ramp"]))
        horizon = ENDPOINT if smoke_full else dd.horizon_at(clock, first=long["first"], last=long["last"],
                                                            increment=long["increment"], every=long["every"])
        horizon = horizon // pair.delta * pair.delta
        long_batch = data.long_batch(pools, long_generators, long["batch_size"], horizon, pair.delta)
        rank_batch = data.rank_batch(pools, rank_generators, rank["cells_per_step"])
        long_term, rank_term = long_and_rank_loss(
            ensemble, pair, long=long_batch if horizon else None, rank=rank_batch if weight > 0 else None,
            rank_pairs=rank_schedules[k], horizon=horizon, presence=presence, center=center,
            temperature=rank["temperature"])
        loss = local + long["weight"] * long_term
        if weight > 0:
            loss = loss + weight * rank_term
        report = ensemble.update(loss)
        ensemble.mark_retired_step(step, report["newly_retired"])
        logs["local"][step] = local.detach()[0]
        logs["grad_norm"][step] = report["grad_norm"][0].float()
        logs["skipped"][step] = report["skipped"][0]
        if fresh or step_times:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - pair_began
            if fresh:
                first_seen[k] = elapsed
            if step_times:
                seconds.append({"update": step, "request": request_set_.request_name(k), "first": fresh,
                                "seconds": elapsed})
        if (step + 1) % 900 == 0 or step + 1 == total:
            trace.append({"step": step + 1, "clock": clock, "horizon": horizon, "rank_weight": weight,
                          "loss": loss.detach().cpu().tolist(), "local": local.detach().cpu().tolist(),
                          "long": long_term.detach().cpu().tolist(), "rank": rank_term.detach().cpu().tolist(),
                          "grad_norm": report["grad_norm"].cpu().tolist(), "retired": ensemble.retired.cpu().tolist()})
            if log:
                now = time.monotonic()
                log(f"{request_set_.name} update {step + 1}/{total}: horizon {horizon}, loss "
                    f"{float(loss.detach()[0]):.4f}, skipped {int(ensemble.skipped[0])}; {now - began:.0f}s; eta "
                    f"{(now - began) / (step + 1 - first) * (total - step - 1):.0f}s")
        if progress is not None and (step + 1) % every == 0 and step + 1 < total:
            torch.cuda.synchronize()
            now = time.monotonic()
            gpu += now - segment_began
            segment_began = now
            torch.save({"ensemble": ensemble.state(), "step": step + 1, "trace": trace, "gpu_seconds": gpu,
                        "logs": {key: value.cpu() for key, value in logs.items()}, "first_seen": first_seen,
                        "window": [g.get_state() for g in window_generators],
                        "long": [g.get_state() for g in long_generators],
                        "rank": [g.get_state() for g in rank_generators]}, str(progress) + ".tmp")
            os.replace(str(progress) + ".tmp", progress)
    torch.cuda.synchronize()
    gpu += time.monotonic() - segment_began
    report = {"updates": total, "gpu_seconds": gpu, "complete": ensemble.retired_at[0] is None,
              "retired_at": ensemble.retired_at[0], "skipped": int(ensemble.skipped[0]),
              "parameters": sum(v[0].numel() for v in ensemble.params.values()),
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9, "trace": trace,
              "first_update_seconds": {request_set_.request_name(k): v for k, v in sorted(first_seen.items())},
              "update_seconds": seconds if step_times else None}
    return report, ensemble, {key: value.cpu() for key, value in logs.items()}


# ---------------------------------------------------------------------------
# policy heads
# ---------------------------------------------------------------------------

HEADS = ("flat", "factorised", "ordinal")


def _features(policy, z, action, remaining):
    if z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
        raise ValueError("the request policy accepts only carrier, action, remaining time")
    return torch.cat((z, action, rp.REMAINING_FEATURES[policy.remaining_feature][1](remaining, z)), -1)


class _Head(nn.Module):
    def __init__(self, request_set_: RequestSet, width: int, remaining_feature: str):
        super().__init__()
        if remaining_feature not in rp.REMAINING_FEATURES:
            raise ValueError(f"unknown remaining feature {remaining_feature}")
        self.request_set = request_set_
        self.pairs = request_set_.pairs
        self.remaining_feature = remaining_feature
        columns, _ = rp.REMAINING_FEATURES[remaining_feature]
        self.trunk = nn.Sequential(nn.Linear(DIM + 5 + columns, width), nn.SiLU())
        self.alpha_head = nn.Linear(width, len(ALPHAS))
        self.register_buffer("deltas", torch.tensor(request_set_.deltas), persistent=False)
        self.register_buffer("delta_of", torch.arange(request_set_.size) // len(ALPHAS), persistent=False)
        self.register_buffer("alpha_of", torch.arange(request_set_.size) % len(ALPHAS), persistent=False)

    def delta_log_probs(self, hidden, remaining):
        raise NotImplementedError

    def forward(self, z, action, remaining):
        hidden = self.trunk(_features(self, z, action, remaining))
        log_delta = self.delta_log_probs(hidden, remaining)
        log_alpha = torch.log_softmax(self.alpha_head(hidden), -1)
        return log_delta[:, self.delta_of] + log_alpha[:, self.alpha_of]


class FactorisedPolicy(_Head):
    """Shared trunk; a |Delta|-way head (Delta > remaining masked) x a 3-way alpha head. Scores are the joint
    log-probabilities log p(Delta) + log p(alpha); argmax picks the argmax of each factor."""

    def __init__(self, request_set_: RequestSet, width: int = 128, remaining_feature: str = "clamp60"):
        super().__init__(request_set_, width, remaining_feature)
        self.delta_head = nn.Linear(width, len(request_set_.deltas))

    def delta_log_probs(self, hidden, remaining):
        logits = self.delta_head(hidden).masked_fill(self.deltas[None] > remaining[:, None], -torch.inf)
        return torch.log_softmax(logits, -1)


class OrdinalPolicy(_Head):
    """Discretised-ordinal Delta head (cumulative logit / proportional odds): one score s and ordered thresholds
    theta_1 < ... < theta_{K-1}; P(Delta > d_k) = sigmoid(s - theta_k). The distribution over the Delta grid is
    renormalized over the feasible Delta (<= remaining); x a 3-way alpha head."""

    def __init__(self, request_set_: RequestSet, width: int = 128, remaining_feature: str = "clamp60"):
        super().__init__(request_set_, width, remaining_feature)
        count = len(request_set_.deltas)
        if count < 3:
            raise ValueError("the ordinal head needs at least three Delta values")
        self.score = nn.Linear(width, 1)
        # initial thresholds: logit(k / K), a uniform Delta distribution at s = 0
        initial = torch.tensor([math.log(k / (count - k)) for k in range(1, count)])
        gaps = initial[1:] - initial[:-1]
        self.first_threshold = nn.Parameter(initial[:1].clone())
        self.gap_parameters = nn.Parameter(gaps + torch.log(-torch.expm1(-gaps)))  # softplus^-1(gap)

    def thresholds(self):
        return torch.cat((self.first_threshold, self.first_threshold + torch.cumsum(F.softplus(self.gap_parameters), 0)))

    def delta_log_probs(self, hidden, remaining):
        s = self.score(hidden)                                   # [B, 1]
        theta = self.thresholds()                                # [K-1]
        gaps = F.softplus(self.gap_parameters)                   # theta_k - theta_{k-1}
        first = F.logsigmoid(theta[:1] - s)                      # p(d_1) = P(Delta <= d_1)
        middle = F.logsigmoid(s - theta[:-1]) + F.logsigmoid(theta[1:] - s) + torch.log(-torch.expm1(-gaps))
        last = F.logsigmoid(s - theta[-1:])
        log_p = torch.cat((first, middle, last), -1)
        log_p = log_p.masked_fill(self.deltas[None] > remaining[:, None], -torch.inf)
        return log_p - torch.logsumexp(log_p, -1, keepdim=True)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_head(kind: str, request_set_: RequestSet, width: int, remaining_feature: str = "clamp60"):
    if kind == "flat":
        return rp.RequestPolicy(request_set_.pairs, width, remaining_feature)
    if kind == "factorised":
        return FactorisedPolicy(request_set_, width, remaining_feature)
    if kind == "ordinal":
        return OrdinalPolicy(request_set_, width, remaining_feature)
    raise ValueError(f"unknown head {kind}")


def head_parameters(kind: str, request_set_: RequestSet, width: int, remaining_feature: str = "clamp60") -> int:
    return parameter_count(build_head(kind, request_set_, width, remaining_feature))


def matched_width(kind: str, request_set_: RequestSet, target: int, remaining_feature: str = "clamp60") -> int:
    """The hidden width whose parameter count is closest to ``target`` (ties to the smaller width)."""
    return min(range(8, 1025), key=lambda w: (abs(head_parameters(kind, request_set_, w, remaining_feature) - target), w))


def new_head(kind: str, request_set_: RequestSet, seed: int, *, width: int, remaining_feature: str = "clamp60",
             device="cuda"):
    """The #77 initialization (``torch.manual_seed(seed + 7700)``), then construct."""
    torch.manual_seed(seed + rp.INIT_OFFSET)
    return build_head(kind, request_set_, width, remaining_feature).to(device)


# ---------------------------------------------------------------------------
# prediction-error DP labels (ICLR eq. dp) over R
# ---------------------------------------------------------------------------

@torch.no_grad()
def prediction_error_costs(model, pairs: Sequence[PredictionPair], z, action, contexts=None, *,
                           compute_weight=rp.ICLR_COMPUTE_WEIGHT, mac_policy=None):
    """costs [N, F, |pairs|]: Delta * mean (carrier(context_t, a, pair) - z_{t+Delta})^2 + lambda * work(pair,
    mac_policy) / work(REFERENCE_PAIR); inf where t + Delta >= F (``prediction_error_labels`` op for op)."""
    contexts = z if contexts is None else contexts
    size, frames, _ = z.shape
    costs = torch.full((size, frames, len(pairs)), torch.inf, device=z.device)
    reference = work(model, rp.REFERENCE_PAIR)
    for i, pair in enumerate(pairs):
        positions = frames - pair.delta
        if positions <= 0:
            continue
        pred = model.carrier(contexts[:, :positions].reshape(-1, DIM),
                             action[:, None].expand(-1, positions, -1).reshape(-1, 5), pair)
        error = (pred.reshape(size, positions, DIM) - z[:, pair.delta:]).square().mean(-1)
        costs[:, :positions, i] = pair.delta * error + compute_weight * work(model, pair, mac_policy) / reference
    return costs


@torch.no_grad()
def backward_dp(costs, deltas: Sequence[int], length):
    """V(t) = min_i c(t, i) + V(t + Delta_i), infeasible past ``length``, ties to the lower index; vectorized
    over requests, equal to ``prediction_error_labels``' per-request loop. Returns (labels [N, F-1], values [N, F])."""
    size, frames, count = costs.shape
    delta = torch.tensor(list(deltas), device=costs.device)
    values = torch.zeros((size, frames), device=costs.device)
    labels = torch.zeros((size, frames - 1), dtype=torch.long, device=costs.device)
    for t in range(frames - 2, -1, -1):
        end = t + delta
        choices = costs[:, t] + values[:, end.clamp(max=frames - 1)]
        choices = torch.where(end[None] < frames, choices, costs[:, t])
        choices = choices.masked_fill(length[:, None] < end[None], torch.inf)
        minimum, labels[:, t] = choices.min(-1)
        values[:, t] = torch.where(length > t, minimum, 0.)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite prediction-error DP costs")
    return labels, values


@torch.no_grad()
def dagger_contexts(model, policy, z, action, length):
    """Batched #77 DAgger round: greedy rollout of ``policy`` from z[:, 0] per window; every visited position's
    context is overwritten by the predicted carrier (``request_policy.dagger_contexts`` per row)."""
    contexts = z.clone()
    current = z[:, 0].clone()
    t = torch.zeros(len(z), dtype=torch.long, device=z.device)
    pairs = policy.pairs
    while True:
        index = (t < length).nonzero(as_tuple=True)[0]
        if not len(index):
            return contexts
        chosen = policy(current[index], action[index], length[index] - t[index]).argmax(-1)
        for k in torch.unique(chosen).tolist():  # ascending: the per-request loop without empty requests
            pair = pairs[k]
            rows = index[chosen == k]
            current[rows] = model.carrier(current[rows], action[rows], pair)
            t[rows] += pair.delta
            contexts[rows, t[rows]] = current[rows]


def rollout_endpoints(model, policy, z0, actions, spec, *, select, pairs=None, endpoint=ENDPOINT,
                      record_schedules=False):
    """``request_policy.rollout_endpoints`` (= ``m99.rollout_endpoints``) op for op over a declared R: the per-request
    loop visits only the requests chosen at that iteration, in ascending order (the pinned loop skips the others),
    and the per-request MACs are computed once per chosen request. ``pairs``: R for the fixed / schedule kinds
    without a policy."""
    pairs = policy.pairs if policy is not None else tuple(pairs)
    count = len(actions)
    z = z0[None].to(actions.device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    pair_macs = {}
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
                logits = policy(z[index], actions[index], endpoint - positions[index])
                chosen = select(spec["kind"], logits, spec)
                controller_calls += len(index)
            if record_schedules:
                trace.append((index.cpu(), chosen.cpu()))
            z_next = z
            for pair_index in torch.unique(chosen).tolist():
                pair = pairs[pair_index]
                rows = index[chosen == pair_index]
                if pair_index not in pair_macs:
                    pair_macs[pair_index] = rp.linear_macs(model, pair)
                z_next = z_next.clone()
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                transition_calls += len(rows)
                macs += len(rows) * pair_macs[pair_index]
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < endpoint)
            iteration += 1
    if z.is_cuda:
        torch.cuda.synchronize()
    macs += controller_calls * rp.policy_macs(policy if spec["kind"] not in ("fixed", "schedule") else None)
    timing = {"transition_calls": transition_calls, "controller_calls": controller_calls, "linear_macs": macs,
              "gpu_seconds": time.monotonic() - began}
    if record_schedules:
        sequences = [[] for _ in range(count)]
        for index, chosen in trace:
            for r, c in zip(index.tolist(), chosen.tolist()):
                sequences[r].append(c)
        timing["schedules"] = sequences
    return z, timing


@torch.no_grad()
def window_labels(model, pairs: Sequence[PredictionPair], z, action, length, *, contexts=None,
                  compute_weight=rp.ICLR_COMPUTE_WEIGHT, mac_policy=None, chunk=256, subset=None):
    """Exact DP labels of windows z [N, F, D] (CPU or GPU), chunked over windows; ``subset``: request indices
    the DP may choose (the pruned DP), labels still index ``pairs``. Returns labels [N, F-1] and V(0) [N]."""
    device = next(model.parameters()).device
    chosen = list(range(len(pairs))) if subset is None else list(subset)
    candidate_pairs = [pairs[i] for i in chosen]
    mapping = torch.tensor(chosen, device=device)
    labels, values = [], []
    for begin in range(0, len(z), chunk):
        zc = z[begin:begin + chunk].to(device)
        cc = None if contexts is None else contexts[begin:begin + chunk].to(device)
        costs = prediction_error_costs(model, candidate_pairs, zc, action[begin:begin + chunk].to(device), cc,
                                       compute_weight=compute_weight, mac_policy=mac_policy)
        label, value = backward_dp(costs, [p.delta for p in candidate_pairs], length[begin:begin + chunk].to(device))
        labels.append(mapping[label].cpu())
        values.append(value[:, 0].cpu())
    return torch.cat(labels), torch.cat(values)


@torch.no_grad()
def batched_dagger(model, policy, z, action, length, *, chunk=256):
    device = next(model.parameters()).device
    out = [dagger_contexts(model, policy, z[b:b + chunk].to(device), action[b:b + chunk].to(device),
                           length[b:b + chunk].to(device)).cpu() for b in range(0, len(z), chunk)]
    return torch.cat(out)


def window_rows(z, action, length, labels):
    """#77 rows of each window: contexts z[:n], action, remaining = n..1 (frames to the window end), labels[:n]."""
    out = []
    for i in range(len(z)):
        n = int(length[i])
        out.append({"z": z[i, :n], "action": action[i].reshape(1, 5).expand(n, -1), "remaining": torch.arange(n, 0, -1),
                    "labels": labels[i, :n]})
    return out


def label_histogram(labels, length, size) -> list:
    rows = torch.cat([labels[i, :int(length[i])] for i in range(len(labels))])
    return torch.bincount(rows, minlength=size).tolist()


def entropy_bits(counts: Sequence[int]) -> float:
    total = sum(counts)
    return float(-sum(c / total * math.log2(c / total) for c in counts if c)) if total else 0.0


def disagreement(first, second, length) -> float:
    """Share of labelled positions (t < length) where two label maps differ."""
    mask = torch.arange(first.shape[1])[None] < length[:, None]
    return float((first != second)[mask].double().mean())


__all__ = [
    "ENDPOINT", "BASE_WINDOW", "LOCAL_STEPS", "REFERENCE_UPDATES", "UPDATES_PER_REQUEST", "BUDGETS", "ALPHAS",
    "PATIENCE", "TRUNCATION", "DELTA_GRIDS", "RequestSet", "REQUEST_SETS", "request_set", "pruned_deltas",
    "sub_requests", "predictor_class", "load_predictor", "budget_updates", "schedule", "curriculum_step",
    "skip_cap", "local_window", "window_positions", "endpoint_schedule", "RequestSetEnsemble", "long_and_rank_loss",
    "train_member", "HEADS", "FactorisedPolicy", "OrdinalPolicy", "parameter_count", "build_head", "head_parameters",
    "matched_width", "new_head", "prediction_error_costs", "backward_dp", "dagger_contexts", "window_labels",
    "batched_dagger", "rollout_endpoints", "window_rows", "label_histogram", "entropy_bits", "disagreement",
]
