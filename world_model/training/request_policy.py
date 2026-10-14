"""Request-policy training path: the controller that picks one (Delta, alpha) request per rollout step.

Shared by #101 (decision-aware policy) and the later policy tickets (#114, #116, #119, #121). The
module is checkpoint- and data-agnostic: callers pass a predictor with ``carrier(z, action, pair)``
(``CNNHybridPredictor`` / ``ContinuousDynamics``), windows or ranking cells, and the knobs below.

Parameters (defaults reproduce the ICLR / #77 path bit for bit; #101 smoke):
- request set R: ``RequestPolicy(pairs=...)``; logits index into ``policy.pairs``;
- compute weight lambda: ``compute_weight`` of both label kinds and of the policy gradient; the
  MAC term is ``lambda * work(model, pair, policy) / work(model, REFERENCE_PAIR)`` per request
  (``matched_dynamics.work``: predictor Linear MACs plus the policy's Linear MACs);
- remaining-frames input feature: ``RequestPolicy(remaining_feature=...)``, one of
  ``REMAINING_FEATURES`` (``clamp60`` = ICLR ``min(kappa, 60) / 60``);
- label kind: ``prediction_error_labels`` (ICLR eq. dp: Delta x one-step carrier MSE + MAC term,
  backward DP over a window) or ``decision_rows`` (decision-valued: per visited rollout state the
  change in the cell's pairwise success-vs-failure ranking loss of each request, + MAC term);
- label window length: ``first_window(..., window_length=60)`` for prediction-error labels; the
  decision labels always run from the anchor to ``endpoint``.

Training: ``new_policy`` (init ``torch.manual_seed(seed + 7700)``), ``distil_round`` (the #77
``train_control`` loop: AdamW, CPU sampler ``Generator(seed + 7700 + round)``, unit ``step % units``,
cross-entropy), ``aggregate_rounds`` (DAgger: round r trains on rounds 0..r per unit),
``window_rows`` (#77 ``label_lineage``, one window at a time), ``policy_gradient`` (REINFORCE on
per-cell AUC minus the MAC term over imagined rollouts, no engine). Scoring:
``rollout_endpoints`` (``m99.rollout_endpoints`` op for op, plus the open-loop ``schedule`` kind
and per-row schedules).
"""
from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from world_model.training.cnn_hybrid import DIM
from world_model.training.matched_dynamics import CONTINUOUS_PAIRS, linear_macs, pairs_for, work

ENDPOINT = 225
REFERENCE_PAIR = CONTINUOUS_PAIRS[-1]  # (15, continuous): the #77 MAC normaliser
ICLR_COMPUTE_WEIGHT = .0001
INIT_OFFSET = 7700
DISTIL_RECIPE = {"rounds": 2, "steps_per_round": 1800, "batch_size": 128, "learning_rate": 1e-3}
RANK_TEMPERATURE = 0.1  # #112 v1 RANK temperature


def _clamp60(remaining, like):
    return remaining.reshape(-1, 1).to(like).clamp(max=60)/60


def _over225(remaining, like):
    return remaining.reshape(-1, 1).to(like)/ENDPOINT


def _log1p(remaining, like):
    return torch.log1p(remaining.reshape(-1, 1).to(like))/math.log1p(ENDPOINT)


# name -> (input columns, feature(remaining [B], like) -> [B, columns])
REMAINING_FEATURES = {"clamp60": (1, _clamp60), "over225": (1, _over225), "log1p": (1, _log1p)}


class RequestPolicy(nn.Module):
    """``[z (236) | action (5) | remaining feature] -> Linear-SiLU-Linear -> |R|`` logits; logits with
    Delta > remaining are -inf. ``RequestPolicy(PAIRS)`` equals ``MatchedController(False)`` (same
    parameter names, initialization order and forward ops)."""

    def __init__(self, pairs: Sequence, width: int = 128, remaining_feature: str = "clamp60"):
        super().__init__()
        if remaining_feature not in REMAINING_FEATURES:
            raise ValueError(f"unknown remaining feature {remaining_feature}")
        self.pairs = tuple(pairs)
        self.remaining_feature = remaining_feature
        columns, _ = REMAINING_FEATURES[remaining_feature]
        self.layers = nn.Sequential(nn.Linear(DIM+5+columns, width), nn.SiLU(), nn.Linear(width, len(self.pairs)))

    def forward(self, z, action, remaining):
        if z.ndim != 2 or z.shape[1] != DIM or action.shape != (len(z), 5):
            raise ValueError("the request policy accepts only carrier, action, remaining time")
        x = torch.cat((z, action, REMAINING_FEATURES[self.remaining_feature][1](remaining, z)), -1)
        horizons = torch.tensor([p.delta for p in self.pairs], device=z.device)
        return self.layers(x).masked_fill(horizons[None] > remaining[:, None], -torch.inf)


def new_policy(pairs, seed, *, width=128, remaining_feature="clamp60", device="cuda"):
    """The #77 initialization: ``torch.manual_seed(seed + 7700)`` then construct."""
    torch.manual_seed(seed + INIT_OFFSET)
    return RequestPolicy(pairs, width, remaining_feature).to(device)


def policy_macs(policy):
    """Linear MACs of one policy call (0 without a policy; ``m99.macs_of``)."""
    return (sum(m.in_features*m.out_features for m in policy.modules() if isinstance(m, nn.Linear))
            if policy is not None else 0)


# ---------------------------------------------------------------------------
# label kind 1: prediction error (ICLR eq. dp; matched_dynamics.controller_labels)
# ---------------------------------------------------------------------------

def first_window(carriers, action, end, window_length=60):
    """The start-0 window of a shot (s99.windows rule): positions min(i, length), i = 0..window_length,
    length = min(window_length, end); terminal absorption by repeating the last position."""
    length = min(window_length, int(end))
    index = [min(i, length) for i in range(window_length+1)]
    return {"z": carriers[index], "action": action, "length": torch.tensor(length)}


@torch.no_grad()
def prediction_error_labels(model, policy, z, action, length, contexts=None, *,
                            compute_weight=ICLR_COMPUTE_WEIGHT):
    """Backward DP over a window: c(t, i) = Delta_i * mean (carrier(context_t, a, pair_i) - z_{t+Delta_i})^2
    + lambda * work(pair_i, policy) / work(REFERENCE_PAIR); V(t) = min_i c(t, i) + V(t + Delta_i);
    infeasible past ``length``; ties to the lower request index. Returns labels [N, F-1]."""
    contexts = z if contexts is None else contexts
    size, frames, _ = z.shape
    pairs = policy.pairs
    costs = torch.full((size, frames, len(pairs)), torch.inf, device=z.device)
    reference = work(model, REFERENCE_PAIR)
    for i, pair in enumerate(pairs):
        positions = frames-pair.delta
        pred = model.carrier(contexts[:, :positions].reshape(-1, DIM),
                             action[:, None].expand(-1, positions, -1).reshape(-1, 5), pair)
        error = (pred.reshape(size, positions, DIM)-z[:, pair.delta:]).square().mean(-1)
        costs[:, :positions, i] = pair.delta*error + compute_weight*work(model, pair, policy)/reference
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
        raise ValueError("nonfinite prediction-error DP costs")
    return labels


@torch.no_grad()
def dagger_contexts(model, policy, z, action, length):
    """#77 DAgger round: greedy rollout of ``policy`` from z[:, 0] (B = 1); every visited position's
    context is overwritten by the predicted carrier, unvisited positions keep the observed one."""
    contexts = z.clone()
    current, t = z[:, 0], 0
    while t < int(length[0]):
        pair = policy.pairs[int(policy(current, action, length-t).argmax(-1))]
        current = model.carrier(current, action, pair)
        t += pair.delta
        contexts[:, t] = current
    return contexts


def window_rows(model, policy, windows, *, aggregated, compute_weight=ICLR_COMPUTE_WEIGHT, device="cuda"):
    """#77 ``label_lineage`` on windows ({z [F, D], action [5], length}), one window at a time.
    Rows: z (contexts), action, remaining (= frames to the window end), labels."""
    records = []
    for window in windows:
        z = window["z"][None].to(device)
        a = window["action"].reshape(1, 5).to(device)
        length = torch.as_tensor(window["length"]).reshape(1).to(device)
        contexts = dagger_contexts(model, policy, z, a, length) if aggregated else z.clone()
        labels = prediction_error_labels(model, policy, z, a, length, contexts, compute_weight=compute_weight)
        n = int(length[0])
        records.append({"z": contexts[0, :n].cpu(), "action": a.expand(n, -1).cpu(),
                        "remaining": torch.arange(n, 0, -1), "labels": labels[0, :n].cpu()})
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}


# ---------------------------------------------------------------------------
# label kind 2: decision value (ranking loss of the cell) over imagined rollouts
# ---------------------------------------------------------------------------

def fallback_table(pairs, endpoint=ENDPOINT):
    """[|R|, endpoint + 1] pair index actually used when a fixed request is asked with r frames
    remaining: itself if Delta <= r, else the same-alpha request with the largest feasible Delta
    (the largest feasible Delta of any alpha if that alpha has none)."""
    table = torch.zeros(len(pairs), endpoint+1, dtype=torch.long)
    for k, pair in enumerate(pairs):
        for r in range(1, endpoint+1):
            if pair.delta <= r:
                table[k, r] = k
                continue
            same = [j for j, p in enumerate(pairs) if p.abstraction == pair.abstraction and p.delta <= r]
            pool = same or [j for j, p in enumerate(pairs) if p.delta <= r]
            if not pool:
                raise ValueError(f"no request is feasible with {r} frames remaining")
            table[k, r] = max(pool, key=lambda j: (pairs[j].delta, -j))
    return table


@torch.no_grad()
def continue_rollout(model, policy, z, actions, positions, continuation, *, pairs, step_work,
                     endpoint=ENDPOINT, first=None, record=False):
    """Batched rollout from per-row positions to ``endpoint``.

    continuation: {"kind": "fixed", "pair_index": k} (``fallback_table`` near the end) or
    {"kind": "policy"} (greedy joint argmax of ``policy``). ``first``: per-row request index of the
    first transition (rows must have it feasible). step_work [|R|]: work of one transition per request.
    Returns z_end, work [rows] (sum over the transitions taken here) and, if ``record``, the visited
    states before each transition as (row index, position, carrier) tensors."""
    z = z.clone()
    positions = positions.clone()
    rows_total = len(z)
    spent = torch.zeros(rows_total, dtype=step_work.dtype, device=z.device)
    alive = positions < endpoint
    table = fallback_table(pairs, endpoint).to(z.device) if continuation["kind"] == "fixed" else None
    trace = ([], [], [])
    pending_first = first is not None
    while bool(alive.any()):
        index = alive.nonzero(as_tuple=True)[0]
        remaining = endpoint-positions[index]
        if pending_first:
            chosen = first[index]
            pending_first = False
        elif continuation["kind"] == "fixed":
            chosen = table[continuation["pair_index"], remaining]
        else:
            chosen = policy(z[index], actions[index], remaining).argmax(-1)
        if record:
            trace[0].append(index)
            trace[1].append(positions[index].clone())
            trace[2].append(z[index].clone())
        z_next = z.clone()
        for pair_index, pair in enumerate(pairs):
            rows = index[chosen == pair_index]
            if not len(rows):
                continue
            z_next[rows] = model.carrier(z[rows], actions[rows], pair)
            positions[rows] += pair.delta
            spent[rows] += step_work[pair_index]
        z = z_next
        alive = torch.isfinite(z).all(dim=1) & (positions < endpoint)
    visited = tuple(torch.cat(part) for part in trace) if record and trace[0] else None
    return z, spent, visited


def partner_terms(verdicts, costs):
    """Per candidate of one cell: its opposite-verdict partners with a finite cost. Returns
    (sign [n], partner costs [n, n] with nan where not a partner)."""
    verdicts = verdicts.bool()
    finite = torch.isfinite(costs)
    sign = torch.where(verdicts, 1.0, -1.0).to(costs)
    partner = (verdicts[:, None] != verdicts[None, :]) & finite[None, :]
    return sign, torch.where(partner, costs[None, :].expand(len(costs), -1), torch.nan)


def candidate_rank_loss(cost, sign, partners, temperature=RANK_TEMPERATURE):
    """Mean over the candidate's opposite-verdict partners of softplus(sign * (cost - partner) / T):
    a success wants a lower cost than every failure, a failure a higher one (the candidate's share of
    ``decision_dynamics.pairwise_ranking``). cost [rows], sign [rows], partners [rows, P] (nan = none)."""
    mask = ~torch.isnan(partners)
    terms = F.softplus(sign[:, None]*(cost[:, None]-torch.nan_to_num(partners))/temperature)
    count = mask.sum(1)
    loss = (terms*mask).sum(1)/count.clamp_min(1)
    return torch.where((count > 0) & torch.isfinite(cost), loss, torch.inf)


@torch.no_grad()
def decision_rows(model, policy, cells, *, continuation, cost_fn: Callable, compute_weight,
                  endpoint=ENDPOINT, temperature=RANK_TEMPERATURE, chunk=65536, device="cuda"):
    """Decision-valued labels (one-step lookahead over a continuation policy; approximate DP).

    For every ranking cell ({z0 [D], actions [n, 5], verdicts [n] bool, unit}) every candidate is
    rolled out from the anchor under ``continuation`` (``continue_rollout``); its endpoint tie-free
    cost s_c = cost_fn(z_end, z0). At every visited state (position t, imagined carrier z_t) and every
    request i feasible at t, the rollout deviates to request i and then follows the continuation to
    the endpoint (cost s'); Q(t, i) = loss_c(s') + lambda * W(t, i) / work(REFERENCE_PAIR), where
    loss_c is the candidate's pairwise ranking loss against the other candidates' continuation costs
    (``candidate_rank_loss``) and W the work of every transition from t (``work(model, pair, policy)``
    per transition). The label is argmin_i Q (ties to the lower index); a state whose candidate has
    no finite opposite-verdict partner, or whose every Q is infinite, yields no row.
    Rows: z, action, remaining = endpoint - t, labels, unit; plus per-cell continuation costs."""
    pairs = policy.pairs
    reference = work(model, REFERENCE_PAIR)
    step_work = torch.tensor([float(work(model, pair, policy)) for pair in pairs], dtype=torch.float64,
                             device=device)
    deltas = torch.tensor([p.delta for p in pairs], device=device)
    z0 = torch.stack([c["z0"] for c in cells]).to(device)
    sizes = [len(c["actions"]) for c in cells]
    cell_of = torch.repeat_interleave(torch.arange(len(cells), device=device), torch.tensor(sizes, device=device))
    actions = torch.cat([c["actions"] for c in cells]).to(device)
    verdicts = torch.cat([torch.as_tensor(c["verdicts"]).bool() for c in cells]).to(device)
    start = z0[cell_of]
    z_end, _, visited = continue_rollout(model, policy, start, actions,
                                         torch.zeros(len(start), dtype=torch.long, device=device),
                                         continuation, pairs=pairs, step_work=step_work, endpoint=endpoint,
                                         record=True)
    base = cost_fn(z_end, start)
    signs, partners = torch.empty(len(start), device=device), torch.full((len(start), max(sizes)), torch.nan,
                                                                           device=device)
    offset = 0
    for size in sizes:
        sign, part = partner_terms(verdicts[offset:offset+size], base[offset:offset+size])
        signs[offset:offset+size] = sign
        partners[offset:offset+size, :size] = part
        offset += size
    row, position, state = visited
    q = torch.full((len(row), len(pairs)), torch.inf, dtype=torch.float64, device=device)
    for i in range(len(pairs)):
        feasible = (deltas[i] <= endpoint-position).nonzero(as_tuple=True)[0]
        for begin in range(0, len(feasible), chunk):
            take = feasible[begin:begin+chunk]
            rows = row[take]
            first = torch.full((len(take),), i, dtype=torch.long, device=device)
            end, spent, _ = continue_rollout(model, policy, state[take], actions[rows], position[take],
                                             continuation, pairs=pairs, step_work=step_work, endpoint=endpoint,
                                             first=first)
            loss = candidate_rank_loss(cost_fn(end, start[rows]), signs[rows], partners[rows], temperature)
            q[take, i] = loss.double()+compute_weight*spent/reference
    minimum, labels = q.min(-1)
    keep = torch.isfinite(minimum)
    return {"z": state[keep].cpu(), "action": actions[row[keep]].cpu(),
            "remaining": (endpoint-position[keep]).cpu(), "labels": labels[keep].cpu(),
            "cell": cell_of[row[keep]].cpu(), "candidate": row[keep].cpu(), "position": position[keep].cpu(),
            "q": q[keep].float().cpu(), "continuation_costs": base.cpu(), "cell_of_candidate": cell_of.cpu()}


# ---------------------------------------------------------------------------
# distillation (the #77 train_control loop) and DAgger aggregation
# ---------------------------------------------------------------------------

def sample(data, batch_size, generator, device):
    """``run_issue_71_hybrid_readiness.sample``."""
    indices = torch.randint(len(data["z"]), (batch_size,), generator=generator)
    return {k: v[indices].to(device) for k, v in data.items()}


def aggregate_rounds(history):
    """DAgger aggregation: per unit, the rows of rounds 0..r concatenated in round order."""
    return [{k: torch.cat([rounds[u][k] for rounds in history]) for k in ("z", "action", "remaining", "labels")}
            for u in range(len(history[0]))]


def distil_round(policy, rows, round_index, *, seed, steps=DISTIL_RECIPE["steps_per_round"],
                 batch_size=DISTIL_RECIPE["batch_size"], learning_rate=DISTIL_RECIPE["learning_rate"],
                 device="cuda"):
    """One #77 round: new AdamW, sampler ``Generator(seed + 7700 + round)``; step s draws
    ``batch_size`` rows of unit ``rows[s % len(rows)]``; cross-entropy on the labels. Returns the
    per-step losses."""
    optim = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed+INIT_OFFSET+round_index)
    losses = []
    for step in range(steps):
        batch = sample(rows[step % len(rows)], batch_size, generator, device)
        loss = F.cross_entropy(policy(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite policy distillation loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    return losses


# ---------------------------------------------------------------------------
# policy gradient on per-cell AUC over imagined rollouts (no engine)
# ---------------------------------------------------------------------------

def cell_auc_tensor(costs, verdicts):
    """P(success cost < failure cost), ties 0.5, over finite candidates; 0.5 if a class is empty."""
    finite = torch.isfinite(costs)
    pos, neg = costs[verdicts & finite], costs[~verdicts & finite]
    if not len(pos) or not len(neg):
        return 0.5
    diff = pos[:, None]-neg[None, :]
    return float(((diff < 0).double()+0.5*(diff == 0).double()).mean())


def sampled_rollout(model, policy, z, actions, *, step_work, endpoint, generator):
    """Stochastic rollout (softmax over the feasible logits) without gradient; returns z_end, work
    [rows] and the decisions (row, carrier, remaining, choice) for the log-probability pass."""
    pairs = policy.pairs
    positions = torch.zeros(len(z), dtype=torch.long, device=z.device)
    spent = torch.zeros(len(z), dtype=step_work.dtype, device=z.device)
    alive = torch.ones(len(z), dtype=torch.bool, device=z.device)
    rows_seen, states, remainders, choices = [], [], [], []
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            remaining = endpoint-positions[index]
            probs = torch.softmax(policy(z[index], actions[index], remaining), -1)
            chosen = torch.multinomial(probs, 1, generator=generator)[:, 0]
            rows_seen.append(index)
            states.append(z[index].clone())
            remainders.append(remaining)
            choices.append(chosen)
            z_next = z.clone()
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                positions[rows] += pair.delta
                spent[rows] += step_work[pair_index]
            z = z_next
            alive = torch.isfinite(z).all(dim=1) & (positions < endpoint)
    return z, spent, (torch.cat(rows_seen), torch.cat(states), torch.cat(remainders), torch.cat(choices))


def policy_gradient(model, policy, cells, *, seed, cost_fn, compute_weight, steps, cells_per_step=8, samples=8,
                    learning_rate=3e-4, grad_clip=1.0, endpoint=ENDPOINT, device="cuda", on_step=None):
    """REINFORCE on R = AUC_cell(endpoint costs) - lambda * mean_candidates W / work(REFERENCE_PAIR).

    Each update draws ``cells_per_step`` cells (CPU ``Generator(seed + 7700)``) and ``samples``
    stochastic schedule draws per cell (every candidate's rollout sampled from the policy; CUDA
    ``Generator(seed + 7700)``); advantage = R minus the cell's mean R over its draws; loss = -mean
    over draws of advantage x (sum of the draw's log-probabilities / candidates). AdamW, grad-norm clip.
    Returns per-step records (mean reward, mean AUC, mean work units, loss)."""
    pairs = policy.pairs
    reference = work(model, REFERENCE_PAIR)
    step_work = torch.tensor([float(work(model, pair, policy)) for pair in pairs], dtype=torch.float64,
                             device=device)
    optim = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    picker = torch.Generator().manual_seed(seed+INIT_OFFSET)
    sampler = torch.Generator(device=device).manual_seed(seed+INIT_OFFSET)
    history = []
    for step in range(steps):
        chosen_cells = torch.randint(len(cells), (cells_per_step,), generator=picker).tolist()
        starts, actions, groups, sizes, verdicts = [], [], [], [], []
        for g, k in enumerate((k for k in chosen_cells for _ in range(samples))):
            cell = cells[k]
            n = len(cell["actions"])
            starts.append(cell["z0"][None].expand(n, -1))
            actions.append(cell["actions"])
            groups.append(torch.full((n,), g, dtype=torch.long))
            sizes.append(n)
            verdicts.append(torch.as_tensor(cell["verdicts"]).bool())
        z0 = torch.cat(starts).to(device)
        a = torch.cat(actions).to(device)
        group = torch.cat(groups).to(device)
        z_end, spent, (rows, states, remainders, choices) = sampled_rollout(
            model, policy, z0, a, step_work=step_work, endpoint=endpoint, generator=sampler)
        costs = cost_fn(z_end, z0)
        rewards, aucs, units = [], [], []
        offset = 0
        for g, n in enumerate(sizes):
            auc = cell_auc_tensor(costs[offset:offset+n], verdicts[g].to(device))
            unit = float(spent[offset:offset+n].mean())/reference
            rewards.append(auc-compute_weight*unit)
            aucs.append(auc)
            units.append(unit)
            offset += n
        reward = torch.tensor(rewards, device=device).reshape(cells_per_step, samples)
        advantage = (reward-reward.mean(1, keepdim=True)).reshape(-1)
        logp = torch.log_softmax(policy(states, a[rows], remainders), -1).gather(1, choices[:, None])[:, 0]
        per_draw = torch.zeros(len(sizes), device=device).index_add(0, group[rows], logp)
        loss = -(advantage*per_draw/torch.tensor(sizes, device=device)).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite policy-gradient loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optim.step()
        record = {"step": step+1, "reward": float(reward.mean()), "auc": sum(aucs)/len(aucs),
                  "work_units": sum(units)/len(units), "loss": float(loss.detach())}
        history.append(record)
        if on_step is not None:
            on_step(record)
    return history


# ---------------------------------------------------------------------------
# scoring rollout (m99.rollout_endpoints op for op, plus open-loop schedules)
# ---------------------------------------------------------------------------

def rollout_endpoints(model, policy, z0, actions, spec, *, select: Callable, endpoint=ENDPOINT,
                      record_schedules=False):
    """``run_decision_chain_attribution.rollout_endpoints`` (= t96.rollout) with the same ops in the
    same order, plus the kind ``schedule`` (spec["schedule"][iteration] for every alive row; #96 OL)
    and, if ``record_schedules``, each row's request sequence. ``select(kind, logits, spec)`` is the
    harness selector (t96.select_pairs) for controller kinds."""
    pairs = policy.pairs if policy is not None else pairs_for(model)
    count = len(actions)
    z = z0[None].to(actions.device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
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
                chosen = select(spec["kind"], logits, spec)
                controller_calls += len(index)
            if record_schedules:
                trace.append((index.cpu(), chosen.cpu()))
            z_next = z
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next = z_next.clone()
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                transition_calls += len(rows)
                macs += len(rows)*pair_macs[pair_index]
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < endpoint)
            iteration += 1
    if z.is_cuda:
        torch.cuda.synchronize()
    macs += controller_calls*policy_macs(policy if spec["kind"] not in ("fixed", "schedule") else None)
    timing = {"transition_calls": transition_calls, "controller_calls": controller_calls, "linear_macs": macs,
              "gpu_seconds": time.monotonic()-began}
    if record_schedules:
        sequences = [[] for _ in range(count)]
        for index, chosen in trace:
            for r, c in zip(index.tolist(), chosen.tolist()):
                sequences[r].append(c)
        timing["schedules"] = sequences
    return z, timing


def schedule_counts(sequences, size):
    """Per-request step counts pooled over sequences."""
    counts = [0]*size
    for sequence in sequences:
        for index in sequence:
            counts[index] += 1
    return counts


def open_loop_schedule(weights, pairs, generator, endpoint=ENDPOINT):
    """#96 OL draw: i.i.d. requests from ``weights`` renormalized over the requests feasible at the
    remaining horizon (uniform if their mass is 0); numpy ``generator``."""
    import numpy as np
    position, sequence = 0, []
    while position < endpoint:
        feasible = [k for k, pair in enumerate(pairs) if pair.delta <= endpoint-position]
        mass = np.array([weights[k] for k in feasible], dtype=float)
        if mass.sum() == 0:
            mass = np.ones(len(feasible))
        pick = feasible[int(generator.choice(len(feasible), p=mass/mass.sum()))]
        sequence.append(pick)
        position += pairs[pick].delta
    return sequence


__all__ = [
    "ENDPOINT", "REFERENCE_PAIR", "ICLR_COMPUTE_WEIGHT", "DISTIL_RECIPE", "RANK_TEMPERATURE",
    "REMAINING_FEATURES", "RequestPolicy", "new_policy", "policy_macs", "first_window",
    "prediction_error_labels", "dagger_contexts", "window_rows", "fallback_table", "continue_rollout",
    "partner_terms", "candidate_rank_loss", "decision_rows", "sample", "aggregate_rounds", "distil_round",
    "cell_auc_tensor", "sampled_rollout", "policy_gradient", "rollout_endpoints", "schedule_counts",
    "open_loop_schedule",
]
