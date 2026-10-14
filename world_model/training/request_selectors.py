"""Restricted-argmax request selection: the policy's argmax taken over a subset of the request set R.

Shared by #114 (T-alpha / D-Delta arms, the horizon-only policy J-cont) and #105 (J_diag, the
anti-diagonal control, J_monotone). It extends ``request_policy`` without editing it: the #101 frozen
plan binds ``request_policy.py`` by sha256, so the restricted kinds live here.

Restriction means the same logits and the same policy with the argmax over an allowed subset
S of ``policy.pairs``: ``logits.masked_fill(~allowed, -inf).argmax(-1)``, ties to the lower request
index (``torch.argmax`` returns the first maximum). The policy's own Delta > remaining mask still
applies, so the choice is the best feasible allowed request. No retraining.

Selector specs (pass to ``rollout_endpoints``):
- ``restricted_spec(pairs, requests)`` -> ``{"kind": "restricted", "allowed": (indices)}``; generic subset
  given as (Delta, alpha) tuples. ``restricted_spec(pairs, pairs)`` (all of R) equals the unrestricted
  joint argmax J.
- ``temporal_spec(pairs, alpha)``: T-alpha, argmax over Delta at fixed alpha (= t96 ``temporal``).
- ``description_spec(pairs, delta)``: D-Delta, argmax over alpha at fixed Delta (= t96 ``description``).
- ``diagonal_spec(pairs)``: {(1, continuous), (5, micro), (15, macro)} (#105 J_diag).
- ``anti_diagonal_spec(pairs)``: {(1, macro), (5, micro), (15, continuous)} (#105 control).
- ``monotone_spec()``: path-dependent. Given the row's previous request (Delta_p, alpha_p), a request
  (Delta, alpha) is allowed iff sign(Delta - Delta_p) * sign(alpha - alpha_p) >= 0, i.e. alpha may only
  coarsen (continuous < micro < macro) when Delta grows and only refine when Delta shrinks; equal
  Delta or equal alpha is always allowed; the first request is unrestricted. (1, continuous) is always
  allowed after any request with Delta > 1, and every Delta = 1 request after a Delta = 1 request, so
  a feasible allowed request always exists on the 3 x 3 grid.
- ``replay_spec(schedules)``: no policy call; row r takes ``schedules[r][iteration]`` (per-row
  replay of recorded schedules, e.g. a policy's schedule with every alpha replaced by continuous).

``rollout_endpoints`` is ``request_policy.rollout_endpoints`` op for op (same kinds ``fixed``,
``schedule`` and any ``select`` kind such as ``t96.select_pairs`` ``joint`` / ``factorized``), plus
the kinds above and the per-row previous request.

Training with a restricted request set on the full policy (#114 J-cont: R_cont on the hybrid trunk,
the other six logits masked): ``MaskedRequestPolicy`` / ``new_masked_policy`` (a ``RequestPolicy``
whose disallowed logits are -inf; same parameters and initialization as the unrestricted policy) and
``restricted_window_rows`` (the #77 DP labels and DAgger contexts over the allowed requests only,
labels returned as indices into ``policy.pairs``).
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import torch

from world_model.model import Abstraction
from world_model.training import request_policy as rp
from world_model.training.matched_dynamics import linear_macs, pairs_for

SELECTOR_KINDS = ("restricted", "monotone")
NO_POLICY_KINDS = ("fixed", "schedule", "replay")
ALPHA_ORDER = tuple(Abstraction)  # continuous < micro < macro (coarsening order)


def pair_key(pair):
    return int(pair.delta), Abstraction(pair.abstraction).value


def request_indices(pairs, requests):
    """Indices into ``pairs`` of (Delta, alpha) requests (alpha as a name or Abstraction), sorted."""
    keys = [pair_key(p) for p in pairs]
    wanted = {(int(d), Abstraction(a).value) for d, a in
              ((r.delta, r.abstraction) if hasattr(r, "delta") else r for r in requests)}
    missing = wanted - set(keys)
    if missing:
        raise ValueError(f"requests not in the request set: {sorted(missing)}")
    return tuple(i for i, key in enumerate(keys) if key in wanted)


def restricted_spec(pairs, requests):
    allowed = request_indices(pairs, requests)
    if not allowed:
        raise ValueError("empty restricted request set")
    return {"kind": "restricted", "allowed": allowed}


def temporal_spec(pairs, alpha):
    """T-alpha: the argmax over Delta at fixed alpha."""
    return restricted_spec(pairs, [p for p in pairs if Abstraction(p.abstraction) == Abstraction(alpha)])


def description_spec(pairs, delta):
    """D-Delta: the argmax over alpha at fixed Delta."""
    return restricted_spec(pairs, [p for p in pairs if int(p.delta) == int(delta)])


def diagonal_spec(pairs):
    return restricted_spec(pairs, [(1, "continuous"), (5, "micro"), (15, "macro")])


def anti_diagonal_spec(pairs):
    return restricted_spec(pairs, [(1, "macro"), (5, "micro"), (15, "continuous")])


def monotone_spec():
    return {"kind": "monotone"}


def replay_spec(schedules):
    return {"kind": "replay", "schedules": [list(map(int, s)) for s in schedules]}


def monotone_table(pairs):
    """[|R|, |R|] bool: entry [p, k] = request k allowed after request p (monotone rule)."""
    deltas = torch.tensor([int(p.delta) for p in pairs])
    alphas = torch.tensor([ALPHA_ORDER.index(Abstraction(p.abstraction)) for p in pairs])
    return (torch.sign(deltas[None]-deltas[:, None])*torch.sign(alphas[None]-alphas[:, None])) >= 0


def allowed_mask(spec, pairs, previous, device):
    """[B, |R|] bool mask of allowed requests for rows with previous requests ``previous`` [B]
    (-1 = no previous request)."""
    size = len(previous)
    if spec["kind"] == "restricted":
        mask = torch.zeros(len(pairs), dtype=torch.bool, device=device)
        mask[list(spec["allowed"])] = True
        return mask[None].expand(size, -1)
    if spec["kind"] == "monotone":
        table = torch.cat((monotone_table(pairs), torch.ones(1, len(pairs), dtype=torch.bool))).to(device)
        return table[previous]  # index -1 -> the all-allowed last row
    raise ValueError(f"unknown selector kind {spec['kind']}")


def select_requests(logits, spec, pairs, previous):
    """Restricted argmax over the allowed requests (ties to the lower index). Raises if a row has no
    feasible allowed request."""
    masked = logits.masked_fill(~allowed_mask(spec, pairs, previous, logits.device), -torch.inf)
    chosen = masked.argmax(-1)
    if not bool(torch.isfinite(masked.gather(-1, chosen[:, None])).all()):
        raise ValueError(f"no feasible allowed request under {spec['kind']}")
    return chosen


def rollout_endpoints(model, policy, z0, actions, spec, *, select: Callable | None = None,
                      endpoint=rp.ENDPOINT, record_schedules=False):
    """``request_policy.rollout_endpoints`` with the same ops in the same order for its kinds, plus the
    restricted / monotone selector kinds (previous request per row) and per-row ``replay``."""
    pairs = policy.pairs if policy is not None else pairs_for(model)
    count = len(actions)
    z = z0[None].to(actions.device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    previous = torch.full((count,), -1, dtype=torch.long, device=z.device)
    replay = None
    if spec["kind"] == "replay":
        if len(spec["schedules"]) != count:
            raise ValueError("replay needs one schedule per row")
        width = max(len(s) for s in spec["schedules"])
        replay = torch.full((count, width), -1, dtype=torch.long)
        for r, sequence in enumerate(spec["schedules"]):
            replay[r, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        replay = replay.to(z.device)
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
            elif spec["kind"] == "replay":
                if iteration >= replay.shape[1]:
                    raise ValueError("replay schedule ends before the endpoint")
                chosen = replay[index, iteration]
                if bool((chosen < 0).any()):
                    raise ValueError("replay schedule ends before the endpoint")
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
# training on a restricted request set (logits outside the subset masked)
# ---------------------------------------------------------------------------

class MaskedRequestPolicy(rp.RequestPolicy):
    """``RequestPolicy`` whose logits outside ``allowed`` are -inf (after the Delta > remaining mask).
    Same parameters, state-dict keys and initialization as ``RequestPolicy(pairs)``; the mask is a
    non-persistent buffer, so ``allowed`` = every request gives the unrestricted forward."""

    def __init__(self, pairs: Sequence, allowed: Sequence[int], width: int = 128, remaining_feature: str = "clamp60"):
        super().__init__(pairs, width, remaining_feature)
        mask = torch.zeros(len(self.pairs), dtype=torch.bool)
        mask[list(allowed)] = True
        if not bool(mask.any()):
            raise ValueError("empty allowed request set")
        self.allowed = tuple(int(i) for i in allowed)
        self.register_buffer("allowed_mask", mask, persistent=False)

    def forward(self, z, action, remaining):
        return super().forward(z, action, remaining).masked_fill(~self.allowed_mask[None], -torch.inf)


def new_masked_policy(pairs, allowed, seed, *, width=128, remaining_feature="clamp60", device="cuda"):
    """The #77 initialization (``torch.manual_seed(seed + 7700)``), then the masked policy."""
    torch.manual_seed(seed+rp.INIT_OFFSET)
    return MaskedRequestPolicy(pairs, allowed, width, remaining_feature).to(device)


class SubsetView:
    """A policy seen through a request subset: ``pairs`` = the allowed pairs, logits = the policy's
    logits at those indices, ``modules()`` = the policy's (so the work term counts the full policy).
    Used to run the #77 DP over the allowed requests only."""

    def __init__(self, policy, allowed):
        self.policy = policy
        self.allowed = tuple(int(i) for i in allowed)
        self.pairs = tuple(policy.pairs[i] for i in self.allowed)
        self._index = None

    def __call__(self, z, action, remaining):
        if self._index is None or self._index.device != z.device:
            self._index = torch.tensor(self.allowed, device=z.device)
        return self.policy(z, action, remaining).index_select(-1, self._index)

    def modules(self):
        return self.policy.modules()


def restricted_window_rows(model, policy, windows, allowed, *, aggregated, compute_weight=rp.ICLR_COMPUTE_WEIGHT,
                           device="cuda"):
    """``request_policy.window_rows`` with the DP (and the DAgger argmax) over the allowed requests
    only; labels are indices into ``policy.pairs``. ``allowed`` = every request gives
    ``window_rows`` bit for bit."""
    rows = rp.window_rows(model, SubsetView(policy, allowed), windows, aggregated=aggregated,
                          compute_weight=compute_weight, device=device)
    rows["labels"] = torch.tensor(tuple(int(i) for i in allowed), dtype=torch.long)[rows["labels"]]
    return rows


__all__ = [
    "SELECTOR_KINDS", "NO_POLICY_KINDS", "ALPHA_ORDER", "pair_key", "request_indices", "restricted_spec",
    "temporal_spec", "description_spec", "diagonal_spec", "anti_diagonal_spec", "monotone_spec", "replay_spec",
    "monotone_table", "allowed_mask", "select_requests", "rollout_endpoints", "MaskedRequestPolicy",
    "new_masked_policy", "SubsetView", "restricted_window_rows",
]
