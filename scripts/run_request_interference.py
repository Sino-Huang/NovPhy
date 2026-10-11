"""Issue-117: interference between the requests that share one predictor (items 1-3).

Binding runner module: scripts/run_request_interference.py
Exact validation command: python -u -m scripts.run_request_interference --validate

The #112 hybrid predictor trains nine requests (Delta in {1, 5, 15} x description in
{continuous, micro, macro}) on every weight, one request per update, round-robin. This
runner measures what that sharing costs each request, on E99 carriers and the 12 fit
lineages, before the #112 recipe is retrained on the #104 fit split:

1. Specialists. For three recipes, the #77 recipe (B) and the #112 long-horizon +
   contrastive recipe as published in v1 (LC1, full back-propagation) and v3 (LC3,
   truncated back-propagation; the Gate-B recipe), it trains the joint nine-request model
   (J), three per-Delta models shared over the descriptions (D1, D5, D15) and nine
   single-request specialists (S-*). Every arm has the joint model's architecture and
   parameter count. Every non-joint arm runs at two budgets: the joint model's per-request
   update count (exactly the joint run's updates of its requests, each at the joint step
   that sets the curriculum and the ranking ramp), and the joint model's full update count.
   Each (recipe, arm, budget) has 13 folds (each fit lineage left out, plus 'all') x 3 seeds.
   Per-request rollout error (#100 engine-referenced metric, t = 15 ... 225) and per-
   inventory ranking AUC (tie-free cost) are scored by leave-one-lineage-out (LOLO), and
   the fold-'all' members once on the held-out lineages after LOLO.
2. Asymmetry. Joint / specialist rollout-error ratios and joint - specialist AUC by Delta
   and by description, separately per recipe; Delta = 1 against Delta = 15.
3. Training dynamics of the joint model. Every 250 schedule steps: per-request validation
   loss on the left-out lineage (every arm), and per-request gradients on the shared trunk
   at a fixed training probe batch (pairwise cosines and pre-clip norms). Every update's
   loss and pre-clip gradient norm are logged, so every nonfinite update and every retired
   member is attributed to the request it trained.

The frozen plan pre-declares the 1.5x disposition and the rule that decides whether the
#112 v3 recipe has an interference problem that warrants a v4.

Modes: --dry-run --smoke --prepare --train --score-lolo --score-heldout --publish
--validate. Training and scoring take --workers N --worker K (one process per GPU, set
CUDA_VISIBLE_DEVICES). Training runs in its own invocations (torch.compile); every
scoring mode and --validate runs under the #111 deterministic_scoring() policy. Zero
engine seconds; intervals are member-clustered percentile bootstraps (10000 draws, PCG64
7201); LOLO results are EXPLORATORY, held-out results DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import functools
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_versions as versions
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_issue_77_n1_diagnostic as diag
from scripts import run_lookahead_control as wlc
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import decision_dynamics as dd
from world_model.training import request_interference as ri
from world_model.training import stable_decision_dynamics as sd
from world_model.training.cnn_hybrid import PAIRS

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-117-request-interference-v1"
IDENTITY = "issue-117-request-interference-v1"
SCHEMA_PLAN = "issue_117_plan_v1"
SCHEMA_SMOKE = "issue_117_smoke_v1"
SCHEMA_PREDICTOR = "issue_117_predictor_v1"
SCHEMA_UNIT = "issue_117_training_unit_v1"
SCHEMA_LOLO = "issue_117_lolo_record_v1"
SCHEMA_HELDOUT = "issue_117_heldout_record_v1"
SCHEMA_LEDGER = "issue_117_ledger_v1"
SCHEMA_COMPUTE = "issue_117_compute_v1"
SCHEMA_REPORT = "issue_117_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_request_interference --validate"
RUNNER = "scripts/run_request_interference.py"
MODULE = "world_model/training/request_interference.py"

DEVICE = "cuda"
SEEDS = v1.SEEDS
ENDPOINT = v1.ENDPOINT
INVENTORIES = v1.INVENTORIES
PRIMARY = v1.PRIMARY
HELD_OUT = v1.HELD_OUT
COSTS = v1.COSTS
PRIMARY_COST = v1.PRIMARY_COST
TIMES = v1.TIMES
METRIC = "engine_position_mse"
REQUESTS = t96.FIXED_ARMS          # F-{1,5,15}-{continuous,micro,macro}, the order of cnn_hybrid.PAIRS
DELTAS = t96.DELTAS
ALPHAS = t96.ALPHAS
STEPS = ri.STEPS

RECIPES = {
    "B": {"definition": ("#77 recipe: #112 v3 arm B (identical to v1 arm B; a nonfinite update would be skipped, "
                         "which never happened in #112)"),
          "long": False, "rank": False, "truncation": None, "nonfinite": "skip"},
    "LC1": {"definition": ("#112 v1 arm LC: #77 loss + long-horizon recursive loss (curriculum 60 -> 225) + "
                           "decision-contrastive ranking loss; full back-propagation; a member is retired at its "
                           "first nonfinite loss or gradient norm"),
            "long": True, "rank": True, "truncation": None, "nonfinite": "retire"},
    "LC3": {"definition": ("#112 v3 arm LC, the Gate-B recipe: LC1 with truncated back-propagation every 45 "
                           "transitions; a nonfinite update is skipped, retirement after 10 consecutive skips"),
            "long": True, "rank": True, "truncation": 45, "nonfinite": "skip"},
}
PATIENCE = 10
GATE_B_RECIPE = "LC3"
GATE_B_UNIT_REQUEST = "F-15-macro"
ARMS = {"J": tuple(range(9)),
        **{f"D{d}": tuple(i for i, r in enumerate(REQUESTS) if int(r.split("-")[1]) == d) for d in DELTAS},
        **{"S-" + r[2:]: (i,) for i, r in enumerate(REQUESTS)}}
ARM_DEFINITIONS = {"J": "joint: all nine requests, round-robin (the #112 hybrid predictor)",
                   **{f"D{d}": f"per-Delta: the three Delta = {d} requests, shared over the descriptions" for d in DELTAS},
                   **{"S-" + r[2:]: f"specialist: request {r} only" for r in REQUESTS}}
BUDGET_DEFINITIONS = {
    "request": ("the joint model's per-request update count: exactly the joint run's updates of the arm's requests "
                "(1000 per request), each at the joint step that sets the horizon curriculum and the ranking ramp"),
    "full": ("the joint model's full update count (9000), cycling the arm's requests; the curriculum and ramp follow "
             "the arm's own update count"),
}

PROBE_EVERY = 250
PROBE_ROWS = 64
PROBE_SEED_OFFSET = 117001
VALIDATION_SEED_OFFSET = 117002
CLAIM = 1.5
DELTA_AUC = v1.DELTA
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
STOP_TOKEN = v1.STOP_TOKEN
MIN_CLUSTERS = 3
SKIP_CAP = 45
MEMORY_BUDGET_GB = 26.0
MAX_GROUP = 39
SMOKE_WARMUP = 9
SMOKE_TIMED = 36
SMOKE_EQUIVALENCE_STEPS = 12
EQUIVALENCE_TOLERANCE = 1e-5
PROGRESS_EVERY = 1000
LOG_EVERY = 1000
PHASES = ((0, 3000), (3000, 6000), (6000, STEPS))

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
bootstrap = v1.bootstrap
fmt_interval = v1.fmt_interval
atomic_torch = v1.atomic_torch
num = v1.num


def log(message):
    print(f"[issue-117] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@functools.lru_cache(maxsize=None)
def v3():
    """The private #112 v3 implementation (v1 code bound to the v3 output, data and LC spec)."""
    return versions.implementation(3)


# ---------------------------------------------------------------------------
# units (no statistic)
# ---------------------------------------------------------------------------

def arm_budgets(arm):
    """J's per-request budget is its full budget (1000 updates per request of nine)."""
    return ("full",) if arm == "J" else ri.BUDGETS


UNITS = tuple((recipe, arm, budget) for recipe in RECIPES for arm in ARMS for budget in arm_budgets(arm))


def unit_key(recipe, arm, budget):
    return f"{recipe}/{arm}/{budget}"


def unit_name(recipe, arm, budget):
    return f"{recipe}--{arm}--{budget}"


def arm_requests(arm):
    return [REQUESTS[i] for i in ARMS[arm]]


def request_delta(request):
    return int(request.split("-")[1])


def request_alpha(request):
    return request.split("-")[2]


def specialist_arm(request):
    return "S-" + request[2:]


def delta_arm(request):
    return f"D{request_delta(request)}"


def schedule_of(arm, budget):
    return ri.schedule(ARMS[arm], budget, steps=STEPS)


def fit_lineages():
    return v3().fit_lineages()


def ensemble_members():
    return v3().ensemble_members()


def member_groups(size):
    return v3().member_groups(size)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def unit_path(output, recipe, arm, budget, index):
    return Path(output) / "training" / f"{unit_name(recipe, arm, budget)}--group{index}.json"


def updates_path(output, recipe, arm, budget, index):
    return Path(output) / "updates" / f"{unit_name(recipe, arm, budget)}--group{index}.pt"


def progress_path(output, recipe, arm, budget, index):
    return Path(output) / "progress" / f"{unit_name(recipe, arm, budget)}--group{index}.pt"


def claim_path(output, recipe, arm, budget, index):
    return Path(output) / "claims" / f"{unit_name(recipe, arm, budget)}--group{index}.claim"


def member_path(output, recipe, arm, budget, fold, seed):
    return Path(output) / "models" / recipe / arm / budget / f"fold-{fold}" / f"seed-{seed}" / "predictor.pt"


def lolo_path(output, recipe, arm, budget, fold, seed):
    return Path(output) / "records" / f"lolo--{unit_name(recipe, arm, budget)}--{fold}--seed{seed}.json"


def heldout_path(output, recipe, arm, budget, seed):
    return Path(output) / "records" / f"heldout--{unit_name(recipe, arm, budget)}--seed{seed}.json"


@functools.lru_cache(maxsize=None)
def _record_json(path, modified):
    return read_json(path)


def read_record(path, schema):
    record = _record_json(str(path), Path(path).stat().st_mtime_ns)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


def finite_list(tensor):
    return [float(v) if math.isfinite(v) else None for v in tensor.detach().double().cpu().tolist()]


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


# ---------------------------------------------------------------------------
# training data
# ---------------------------------------------------------------------------

class Data:
    """The #112 v3 training tensors (v3 TrainData, hybrid family) plus indexed batches for the probes."""

    def __init__(self, payload):
        self.base = v3().TrainData(payload, "hybrid")
        self.window_members = self.base.window_members
        self.long_members = self.base.long_members
        self.cell_members = self.base.cell_members

    # training draws: exactly v3 TrainData
    def pools(self, fold):
        return self.base.pools(fold)

    def windows(self, pools, generators, count):
        return self.base.windows(pools, generators, count)

    def long_batch(self, pools, generators, count, horizon, delta):
        return self.base.long_batch(pools, generators, count, horizon, delta)

    def rank_batch(self, pools, generators, count):
        return self.base.rank_batch(pools, generators, count)

    # probe batches from explicit indices [M, rows]
    def windows_at(self, index):
        return {k: v[index] for k, v in self.base.window.items()}

    def long_at(self, index, horizon, delta):
        b = self.base
        shot, start = b.long[index, 0], b.long[index, 1]
        offset, last = b.offsets[shot], b.lasts[shot]
        steps = torch.arange(1, horizon // delta + 1, device=DEVICE) * delta
        positions = torch.minimum(start[..., None] + steps, last[..., None])
        return {"start": b.carriers[offset + start], "action": b.shot_actions[shot],
                "targets": b.carriers[offset[..., None] + positions]}

    def rank_at(self, index, valid):
        b = self.base
        verdict = b.cell_verdict[index]
        verdict = torch.where(valid[..., None], verdict, torch.full_like(verdict, -1))
        return {"z0": b.cell_z0[index], "action": b.cell_action[index], "verdict": verdict}

    def lineage(self, fold):
        """Indices of the fold's own (left-out) lineage; fold 'all' has none (in-sample pools)."""
        if fold == "all":
            return None
        return {key: torch.tensor([i for i, m in enumerate(members) if m == fold], dtype=torch.long)
                for key, members in (("window", self.window_members), ("long", self.long_members),
                                     ("cell", self.cell_members))}

    def cells_pad(self):
        return max(sum(1 for m in self.cell_members if m == lineage) for lineage in fit_lineages())


def probe_batches(data, members, pools):
    """Fixed per-member probe indices: a training-pool batch (gradient probe) and a validation
    batch on the member's left-out lineage (fold 'all': its training pool, flagged in-sample)."""
    pad = data.cells_pad()
    out = {k: [] for k in ("train_windows", "train_long", "train_cells", "val_windows", "val_long", "val_cells",
                           "val_valid")}
    info = []
    for (fold, seed), pool in zip(members, pools):
        generator = torch.Generator().manual_seed(seed + PROBE_SEED_OFFSET)
        for key, name, rows in (("window", "train_windows", PROBE_ROWS), ("long", "train_long", PROBE_ROWS),
                                ("cell", "train_cells", v1.RANK["cells_per_step"])):
            out[name].append(pool[key][torch.randint(len(pool[key]), (rows,), generator=generator)])
        held = data.lineage(fold)
        source = held if held is not None else pool
        generator = torch.Generator().manual_seed(seed + VALIDATION_SEED_OFFSET)
        for key, name in (("window", "val_windows"), ("long", "val_long")):
            out[name].append(source[key][torch.randint(len(source[key]), (PROBE_ROWS,), generator=generator)])
        cells = source["cell"] if held is not None else source["cell"][
            torch.randperm(len(source["cell"]), generator=generator)[:pad]]
        index = torch.zeros(pad, dtype=torch.long)
        valid = torch.zeros(pad, dtype=torch.bool)
        index[:len(cells)], valid[:len(cells)] = cells, True
        out["val_cells"].append(index)
        out["val_valid"].append(valid)
        info.append({"fold": fold, "seed": seed, "in_sample": held is None, "validation_cells": int(len(cells))})
    return {k: torch.stack(v).to(DEVICE) for k, v in out.items()}, info


# ---------------------------------------------------------------------------
# training (its own invocation; torch.compile)
# ---------------------------------------------------------------------------

def make_ensemble(recipe, models, frozen):
    kwargs = dict(learning_rate=frozen["learning_rate"], weight_decay=frozen["weight_decay"],
                  grad_clip=frozen["grad_clip"], compile=True)
    if RECIPES[recipe]["nonfinite"] == "retire":
        return dd.Ensemble(models, **kwargs)

    class Skipping(sd.SkippingEnsemble):
        patience = PATIENCE
    return Skipping(models, **kwargs)


def long_function(recipe):
    truncation = RECIPES[recipe]["truncation"]
    if truncation is None:
        return dd.long_and_rank_loss
    return functools.partial(sd.truncated_long_and_rank_loss, truncation=truncation)


def schedule_values(g, smoke_full=False):
    """(horizon, ranking weight) at joint step g (the #112 curriculum and ramp)."""
    long, rank = v1.LONG, v1.RANK
    weight = rank["weight"] * (1.0 if smoke_full else dd.ramp_at(g, start=rank["start"], length=rank["ramp"]))
    horizon = ENDPOINT if smoke_full else dd.horizon_at(g, first=long["first"], last=long["last"],
                                                        increment=long["increment"], every=long["every"])
    return horizon, weight


def compose(ensemble, recipe, pair, local, long_batch, rank_batch, horizon, weight, columns):
    """The #112 train_group loss composition, verbatim."""
    spec = RECIPES[recipe]
    loss, long_term, rank_term = local, None, None
    if spec["long"] or (spec["rank"] and weight > 0):
        long_term, rank_term = long_function(recipe)(
            ensemble, pair, long=long_batch, rank=rank_batch if weight > 0 else None, horizon=horizon,
            endpoint=ENDPOINT, presence=columns[0], center=columns[1], temperature=v1.RANK["temperature"])
        if spec["long"]:
            loss = loss + v1.LONG["weight"] * long_term
        if spec["rank"] and weight > 0:
            loss = loss + weight * rank_term
    return loss, long_term, rank_term


@torch.no_grad()
def validation_probe(ensemble, data, probe, requests, columns):
    """Per request: the #77 local loss, the long-horizon loss at 225 and the ranking loss on the
    member's left-out lineage (every recipe; values only)."""
    out = {}
    for request in requests:
        pair = PAIRS[REQUESTS.index(request)]
        local = dd.local_loss(ensemble, data.windows_at(probe["val_windows"]), pair)
        long_value, rank_value = dd.long_and_rank_loss(
            ensemble, pair, long=data.long_at(probe["val_long"], ENDPOINT, pair.delta),
            rank=data.rank_at(probe["val_cells"], probe["val_valid"]), horizon=ENDPOINT, endpoint=ENDPOINT,
            presence=columns[0], center=columns[1], temperature=v1.RANK["temperature"])
        out[request] = {"local": finite_list(local), "long225": finite_list(long_value), "rank": finite_list(rank_value)}
    return out


def gradient_probe(ensemble, data, probe, recipe, g, columns):
    """Per request: the recipe's training loss at joint step g on the fixed training probe batch,
    its pre-clip gradient norm (all parameters) and the trunk Gram matrix across requests."""
    horizon, weight = schedule_values(g)
    spec = RECIPES[recipe]
    names = list(ensemble.params)
    trunk = ri.trunk_names(names)
    trunk_grads, losses, norms = [], [], []
    for pair in PAIRS:
        local = dd.local_loss(ensemble, data.windows_at(probe["train_windows"]), pair)
        long_batch = data.long_at(probe["train_long"], horizon, pair.delta) if spec["long"] else None
        rank_batch = (data.rank_at(probe["train_cells"], torch.ones_like(probe["train_cells"], dtype=torch.bool))
                      if spec["rank"] else None)
        loss, _, _ = compose(ensemble, recipe, pair, local, long_batch, rank_batch, horizon, weight, columns)
        active = torch.isfinite(loss.detach()) & ~ensemble.retired
        values = torch.autograd.grad(torch.where(active, loss, torch.zeros_like(loss)).sum(),
                                     [ensemble.params[n] for n in names], allow_unused=True)
        squares = torch.zeros(ensemble.size, device=ensemble.device, dtype=torch.float64)
        for value in values:
            if value is not None:
                squares = squares + value.double().flatten(1).square().sum(1)
        norm = torch.where(active, squares.sqrt(), torch.full_like(squares, float("nan")))
        trunk_grads.append({n: v for n, v in zip(names, values) if n in trunk})
        losses.append(torch.where(active, loss.detach().double(), torch.full_like(squares, float("nan"))))
        norms.append(norm)
        del values
    gram = ri.request_gram(trunk_grads, trunk)
    gram = torch.where(torch.stack(losses, 1).isfinite()[:, :, None] & torch.stack(losses, 1).isfinite()[:, None, :],
                       gram, torch.full_like(gram, float("nan")))
    del trunk_grads
    return {"horizon": horizon, "rank_weight": weight,
            "gram": [[finite_list(row) for row in member] for member in gram],
            "norm": [finite_list(row) for row in torch.stack(norms, 1)],
            "loss": [finite_list(row) for row in torch.stack(losses, 1)]}


def run_probe(ensemble, data, probe, recipe, arm, done, g, counts, columns, gradients):
    horizon, weight = schedule_values(g)
    entry = {"update": done, "step": g, "request_updates": dict(counts), "horizon": horizon, "rank_weight": weight,
             "validation": validation_probe(ensemble, data, probe, arm_requests(arm), columns)}
    entry["gradient"] = gradient_probe(ensemble, data, probe, recipe, g, columns) if gradients else None
    return entry


LOG_KEYS = ("loss", "local", "long", "rank", "grad_norm")
FLAG_KEYS = ("loss_finite", "norm_finite", "active")


def train_unit(output, recipe, arm, budget, index, members, data, *, steps=None, save=True, probe_every=None,
               smoke_full=False, timing_after=None, step_times=False):
    """Train one group of members of one (recipe, arm, budget) unit (resumable); probes every
    ``probe_every`` joint-schedule steps (default PROBE_EVERY; 0 disables them)."""
    probe_every = PROBE_EVERY if probe_every is None else probe_every
    frozen = v3().recipe()
    plan = schedule_of(arm, budget)
    total = len(plan) if steps is None else steps
    models = []
    for fold, seed in members:
        torch.manual_seed(seed)
        models.append(s99.new_predictor("hybrid").to(DEVICE))
    ensemble = make_ensemble(recipe, models, frozen)
    del models
    size = ensemble.size
    pools = [data.pools(fold) for fold, _ in members]
    window_generators = [torch.Generator().manual_seed(seed) for _, seed in members]
    long_generators = [torch.Generator().manual_seed(seed + v1.LONG["seed_offset"]) for _, seed in members]
    rank_generators = [torch.Generator().manual_seed(seed + v1.RANK["seed_offset"]) for _, seed in members]
    columns = v3().pig_columns()
    spec = RECIPES[recipe]
    probe, validation_members = probe_batches(data, members, pools)
    gradients = arm == "J"
    logs = {k: torch.full((total, size), float("nan"), device=DEVICE) for k in LOG_KEYS}
    flags = {k: torch.zeros((total, size), dtype=torch.bool, device=DEVICE) for k in FLAG_KEYS}
    counts = {request: 0 for request in arm_requests(arm)}
    first, probes, gpu, last_bucket, retired_step = 0, [], 0.0, 0, [None] * size
    progress = progress_path(output, recipe, arm, budget, index)
    if save and progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        ensemble.load_state(saved["ensemble"])
        for generators, states in ((window_generators, saved["window"]), (long_generators, saved["long"]),
                                   (rank_generators, saved["rank"])):
            for generator, state in zip(generators, states):
                generator.set_state(state)
        first, probes, gpu, last_bucket = saved["step"], saved["probes"], saved["gpu_seconds"], saved["last_bucket"]
        counts, retired_step = saved["counts"], saved["retired_step"]
        for k in LOG_KEYS:
            logs[k][:first] = saved["logs"][k][:first].to(DEVICE)
        for k in FLAG_KEYS:
            flags[k][:first] = saved["flags"][k][:first].to(DEVICE)
        log(f"resumed {unit_name(recipe, arm, budget)}/group{index} at update {first}")
    torch.cuda.synchronize()
    began = time.monotonic()
    segment_began, timed_began, times = began, None, []
    if first == 0 and probe_every:
        probes.append(run_probe(ensemble, data, probe, recipe, arm, 0, 0, counts, columns, gradients))
    for k in range(first, total):
        if k == timing_after:
            torch.cuda.synchronize()
            timed_began = time.monotonic()
        if step_times:
            torch.cuda.synchronize()
            update_began = time.monotonic()
        pair_index, g = plan[k]
        pair = PAIRS[pair_index]
        batch = data.windows(pools, window_generators, frozen["batch_size"])
        local = dd.local_loss(ensemble, batch, pair)
        horizon, weight = schedule_values(g, smoke_full)
        long_batch = data.long_batch(pools, long_generators, v1.LONG["batch_size"], horizon, pair.delta) \
            if spec["long"] else None
        rank_batch = data.rank_batch(pools, rank_generators, v1.RANK["cells_per_step"]) if spec["rank"] else None
        loss, long_term, rank_term = compose(ensemble, recipe, pair, local, long_batch, rank_batch, horizon, weight,
                                             columns)
        report = ensemble.update(loss)
        ensemble.mark_retired_step(k, report["newly_retired"])
        for i in report["newly_retired"].nonzero(as_tuple=True)[0].tolist():
            retired_step[i] = g
        logs["loss"][k] = loss.detach()
        logs["local"][k] = local.detach()
        if long_term is not None:
            logs["long"][k] = long_term.detach()
        if rank_term is not None:
            logs["rank"][k] = rank_term.detach()
        logs["grad_norm"][k] = report["grad_norm"]
        flags["loss_finite"][k] = torch.isfinite(loss.detach())
        flags["norm_finite"][k] = torch.isfinite(report["grad_norm"])
        flags["active"][k] = report["active"]
        counts[REQUESTS[pair_index]] += 1
        if step_times:
            torch.cuda.synchronize()
            times.append({"pair": pair_index, "seconds": time.monotonic() - update_began})
        bucket = (g + 1) // probe_every if probe_every else 0
        if probe_every and (bucket > last_bucket or k + 1 == total):
            last_bucket = max(bucket, last_bucket)
            probes.append(run_probe(ensemble, data, probe, recipe, arm, k + 1, g, counts, columns, gradients))
        if (k + 1) % LOG_EVERY == 0 or k + 1 == total:
            torch.cuda.synchronize()
            now = time.monotonic()
            alive = ~ensemble.retired
            log(f"{unit_name(recipe, arm, budget)}/group{index} update {k + 1}/{total} (step {g}): horizon {horizon}, "
                f"mean loss {float(loss.detach()[alive].mean()) if alive.any() else 'n/a'}, "
                f"retired {int(ensemble.retired.sum())}; {now - began:.0f}s; "
                f"eta {(now - began) / (k + 1 - first) * (total - k - 1):.0f}s")
        if save and (k + 1) % PROGRESS_EVERY == 0 and k + 1 < total:
            torch.cuda.synchronize()
            now = time.monotonic()
            gpu += now - segment_began
            segment_began = now
            atomic_torch(progress, {"ensemble": ensemble.state(), "step": k + 1, "probes": probes, "gpu_seconds": gpu,
                                    "last_bucket": last_bucket, "counts": counts, "retired_step": retired_step,
                                    "logs": {key: v[:k + 1].cpu() for key, v in logs.items()},
                                    "flags": {key: v[:k + 1].cpu() for key, v in flags.items()},
                                    "window": [g_.get_state() for g_ in window_generators],
                                    "long": [g_.get_state() for g_ in long_generators],
                                    "rank": [g_.get_state() for g_ in rank_generators]})
    torch.cuda.synchronize()
    gpu += time.monotonic() - segment_began
    timed = None if timed_began is None else (time.monotonic() - timed_began) / (total - timing_after)
    parameters = sum(v[0].numel() for v in ensemble.params.values())
    cpu_logs = {k: v.cpu() for k, v in logs.items()}
    cpu_flags = {k: v.cpu() for k, v in flags.items()}
    skipped = ensemble.skipped.cpu().tolist() if hasattr(ensemble, "skipped") else [0] * size
    report = {"schema": SCHEMA_UNIT, "plan_identity": IDENTITY, "recipe": recipe, "arm": arm, "budget": budget,
              "group": index, "requests": arm_requests(arm), "updates": total, "trained_at": utc_now(),
              "members": [{"fold": fold, "seed": seed, "complete": ensemble.retired_at[i] is None,
                           "retired_at_update": ensemble.retired_at[i], "retired_at_step": retired_step[i],
                           "retired_request": (REQUESTS[plan[ensemble.retired_at[i]][0]]
                                               if ensemble.retired_at[i] is not None else None),
                           "skipped": int(skipped[i])} for i, (fold, seed) in enumerate(members)],
              "validation_members": validation_members,
              "nonfinite_events": nonfinite_events(cpu_logs, cpu_flags, plan, members, ensemble.retired_at),
              "grad_norm_summary": grad_norm_summary(cpu_logs, cpu_flags, plan, ensemble.retired_at),
              "final": {k: finite_list(cpu_logs[k][total - 1]) for k in ("loss", "local", "long", "rank")},
              "probes": probes, "gpu_seconds": gpu, "parameters_per_member": parameters,
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
              "pool_sizes": [{k: len(v) for k, v in pool.items()} for pool in pools],
              "timed_seconds_per_update": timed, "update_times": times or None}
    if save:
        for i, (fold, seed) in enumerate(members):
            atomic_torch(member_path(output, recipe, arm, budget, fold, seed),
                         {"schema": SCHEMA_PREDICTOR, "plan_identity": IDENTITY, "recipe": recipe, "arm": arm,
                          "budget": budget, "fold": fold, "seed": seed, "updates": total,
                          "complete": ensemble.retired_at[i] is None, "retired_at": ensemble.retired_at[i],
                          "parameters": parameters, "model": ensemble.member_state(i)})
        atomic_torch(updates_path(output, recipe, arm, budget, index),
                     {"plan_identity": IDENTITY, "members": members, "schedule": plan[:total],
                      "logs": cpu_logs, "flags": cpu_flags})
        write_json(unit_path(output, recipe, arm, budget, index), report)
        progress.unlink(missing_ok=True)
    return report, ensemble


def nonfinite_events(logs, flags, plan, members, retired_at):
    """Per member: every update whose loss or pre-clip gradient norm was nonfinite (before
    retirement), with the request it trained, and for the first one the largest finite
    pre-clip norm of every request over the preceding 250 updates."""
    events = []
    total = logs["loss"].shape[0]
    pairs = torch.tensor([p for p, _ in plan[:total]])
    for i, (fold, seed) in enumerate(members):
        last = total if retired_at[i] is None else retired_at[i] + 1
        bad = (~(flags["loss_finite"][:last, i] & flags["norm_finite"][:last, i])).nonzero(as_tuple=True)[0].tolist()
        if not bad:
            continue
        first = bad[0]
        window = slice(max(0, first - PROBE_EVERY), first)
        norms = logs["grad_norm"][window, i]
        preceding = {}
        for p in sorted(set(pairs[window].tolist())):
            values = norms[pairs[window] == p]
            values = values[torch.isfinite(values)]
            preceding[REQUESTS[p]] = float(values.max()) if len(values) else None
        events.append({"fold": fold, "seed": seed, "count": len(bad), "retired": retired_at[i] is not None,
                       "events": [{"update": k, "step": plan[k][1], "request": REQUESTS[plan[k][0]],
                                   "loss_finite": bool(flags["loss_finite"][k, i]),
                                   "grad_norm_finite": bool(flags["norm_finite"][k, i]),
                                   "horizon": schedule_values(plan[k][1])[0]} for k in bad[:20]],
                       "first_preceding_max_grad_norm": preceding})
    return events


def grad_norm_summary(logs, flags, plan, retired_at):
    """Pre-clip gradient norm of every update, per request and member (finite updates before retirement)."""
    total = logs["loss"].shape[0]
    pairs = torch.tensor([p for p, _ in plan[:total]])
    out = {}
    for p in sorted(set(pairs.tolist())):
        rows = (pairs == p).nonzero(as_tuple=True)[0]
        per = {"median": [], "p99": [], "max": [], "updates": int(len(rows))}
        for i in range(logs["grad_norm"].shape[1]):
            last = total if retired_at[i] is None else retired_at[i]
            keep = rows[rows < last]
            values = logs["grad_norm"][keep, i]
            values = values[torch.isfinite(values)].double()
            for name, fn in (("median", lambda v: v.median()), ("p99", lambda v: torch.quantile(v, 0.99)),
                             ("max", lambda v: v.max())):
                per[name].append(float(fn(values)) if len(values) else None)
        out[REQUESTS[p]] = per
    return out


# ---------------------------------------------------------------------------
# work distribution, ledger
# ---------------------------------------------------------------------------

def claim(path):
    """Exclusive claim of one unit by this process; a claim of a dead process is stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(path.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid:
            try:
                os.kill(pid, 0)
                return False
            except ProcessLookupError:
                pass
        path.unlink(missing_ok=True)
        return claim(path)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    return True


def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "phases": {}, "gpu_seconds_elapsed": 0.0,
            "wall_seconds_elapsed": 0.0}


def mark(output, phase, wall, gpu, **extra):
    """Ledger entry under a file lock (several worker processes write it)."""
    Path(output).mkdir(parents=True, exist_ok=True)
    with open(Path(output) / "ledger.lock", "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ledger = load_ledger(output)
        block = ledger["phases"].get(phase, {"wall_seconds": 0.0, "gpu_seconds": 0.0})
        ledger["phases"][phase] = {**block, "wall_seconds": block["wall_seconds"] + wall,
                                   "gpu_seconds": block["gpu_seconds"] + gpu, "last_at": utc_now(), **extra}
        ledger["gpu_seconds_elapsed"] += gpu
        ledger["wall_seconds_elapsed"] += wall
        write_json(Path(output) / "ledger.json", ledger)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    plan = read_json(Path(output) / "plan.json") if (Path(output) / "plan.json").is_file() else None
    if plan and ledger["gpu_seconds_elapsed"] > plan["caps"]["gpu_seconds"]:
        raise ValueError(f"{STOP_TOKEN}: GPU-seconds cap exceeded")


# ---------------------------------------------------------------------------
# scoring (always inside deterministic_scoring, in processes that did not train)
# ---------------------------------------------------------------------------

def load_member(output, recipe, arm, budget, fold, seed):
    """The trained predictor, or None for a retired (typed-failure) member."""
    payload = torch.load(member_path(output, recipe, arm, budget, fold, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_PREDICTOR or payload.get("plan_identity") != IDENTITY
            or (payload["recipe"], payload["arm"], payload["budget"], payload["fold"], payload["seed"])
            != (recipe, arm, budget, fold, seed) or payload["updates"] != len(schedule_of(arm, budget))):
        raise ValueError(f"predictor {unit_name(recipe, arm, budget)}/{fold}/{seed} binding differs")
    if not payload["complete"]:
        return None
    model = s99.new_predictor("hybrid")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval()


def endpoint_block(model, requests, z0, items, costs):
    """v1 endpoint_block restricted to the arm's requests."""
    actions = m99.action_batch(items)
    out = {}
    for request in requests:
        z_end, timing = m99.rollout_endpoints(model, None, z0.to(DEVICE), actions, t96.arm_spec(request))
        out[request] = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu())
                                 for slot, item in enumerate(items)], "candidates": len(items), **timing}
    return out


@torch.no_grad()
def rollout_curves(model, requests, context, sources):
    """v1 rollout_curves restricted to the arm's requests."""
    actions = torch.cat([diag.n1_action_tensor(s["action"], DEVICE) for s in sources])
    out = {}
    for request in requests:
        pair = PAIRS[REQUESTS.index(request)]
        z = context.to(DEVICE)[None].expand(len(sources), -1).clone()
        elapsed, curve = 0, {}
        for t in TIMES:
            while elapsed < t:
                z = model.carrier(z, actions, pair)
                elapsed += pair.delta
            curve[t] = z.cpu()
        out[request] = curve
    return out


def lineage_rollout(model, requests, context, sources, objective):
    return v1.curve_errors(rollout_curves(model, requests, context, sources), sources, objective)


class Scoring:
    """Inputs shared by every scoring record."""

    def __init__(self, cohort):
        _, _, self.universe = v3().load_universe()
        self.samples = v3().rollout_samples_all()
        self.costs = harness.EndpointCosts(wlc.load_objective())
        self.objective = diag.base.TaskObjective.from_vocabulary(s99.vocabulary())
        source = v3().OUTPUT if cohort == "fit" else v1.OUTPUT
        self.anchors = torch.load(v1.anchors_path(source, cohort), weights_only=True)["anchors"]
        self.contexts = torch.load(v1.contexts_path(source, cohort), weights_only=True)["contexts"]


def lolo_record(output, scoring, recipe, arm, budget, fold, seed, policy):
    model = load_member(output, recipe, arm, budget, fold, seed)
    base = {"schema": SCHEMA_LOLO, "plan_identity": IDENTITY, "scoring_policy": policy, "scored_at": utc_now(),
            "cell": {"recipe": recipe, "arm": arm, "budget": budget, "fold": fold, "seed": seed}, "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": "member_retired_nonfinite", "inventories": {}, "rollout": None}
    requests = arm_requests(arm)
    inventories = {}
    for inventory in INVENTORIES:
        entry = scoring.universe[inventory].get(fold)
        if entry is not None:
            inventories[inventory] = endpoint_block(model, requests, scoring.anchors[fold], entry["inventory"],
                                                    scoring.costs)
    rollout = (lineage_rollout(model, requests, scoring.contexts[fold], scoring.samples[fold], scoring.objective)
               if fold in scoring.contexts else None)
    return {**base, "typed_failure": None, "inventories": inventories, "rollout": rollout}


def heldout_record(output, scoring, recipe, arm, budget, seed, policy):
    model = load_member(output, recipe, arm, budget, "all", seed)
    base = {"schema": SCHEMA_HELDOUT, "plan_identity": IDENTITY, "scoring_policy": policy, "scored_at": utc_now(),
            "cell": {"recipe": recipe, "arm": arm, "budget": budget, "seed": seed}, "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": "member_retired_nonfinite", "inventories": {}, "rollout": {}}
    requests = arm_requests(arm)
    universe = scoring.universe
    inventories = {inventory: {member: endpoint_block(model, requests, scoring.anchors[member],
                                                      universe[inventory][member]["inventory"], scoring.costs)
                               for member in sorted(m for m in universe[inventory] if m in HELD_OUT)}
                   for inventory in INVENTORIES}
    rollout = {member: lineage_rollout(model, requests, scoring.contexts[member], scoring.samples[member],
                                       scoring.objective) for member in sorted(scoring.contexts)}
    return {**base, "typed_failure": None, "inventories": inventories, "rollout": rollout}


# ---------------------------------------------------------------------------
# smoke (pre-freeze: nothing kept, no statistic, no held-out access)
# ---------------------------------------------------------------------------

def max_delta(first, second):
    return max(float((first.params[k] - second.params[k]).abs().max()) for k in first.params)


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    checks, timings, memory = [], {}, {}
    # schedules
    joint = schedule_of("J", "full")
    checks.append({"control": "joint_schedule_is_the_issue_112_schedule",
                   "ok": joint == [(k % 9, k) for k in range(STEPS)]
                   and all(PAIRS[p] == v1.step_pair("hybrid", k) for p, k in joint)})
    restricted = all(schedule_of(arm, "request") == [(p, g) for p, g in joint if p in ARMS[arm]]
                     for arm in ARMS if arm != "J")
    counts = {arm: {b: len(schedule_of(arm, b)) for b in arm_budgets(arm)} for arm in ARMS}
    checks.append({"control": "per_request_budget_is_the_joint_runs_updates_of_the_arm", "update_counts": counts,
                   "ok": restricted and all(counts[a].get("request", 1000 * len(ARMS[a])) == 1000 * len(ARMS[a])
                                            and counts[a]["full"] == STEPS for a in ARMS)})
    checks.append({"control": "requests_follow_cnn_hybrid_pairs",
                   "ok": all(v3().request_pair(r) == PAIRS[i] for i, r in enumerate(REQUESTS))})
    payload = v3().load_training_data(v3().OUTPUT)
    data = Data(payload)
    fitted = fit_lineages()
    members2 = [(fitted[0], SEEDS[0]), ("all", SEEDS[1])]
    # equivalence with the #112 training loops (probes every 4 updates must not change training)
    references = {"B": (v3(), "B"), "LC1": (v1, "LC"), "LC3": (v3(), "LC")}
    for recipe, (module, arm) in references.items():
        ours, ensemble = train_unit(scratch, recipe, "J", "full", 0, members2, data, steps=SMOKE_EQUIVALENCE_STEPS,
                                    save=False, probe_every=4, smoke_full=True)
        theirs, reference = module.train_group(scratch, arm, "hybrid", 0, members2, data.base,
                                               steps=SMOKE_EQUIVALENCE_STEPS, save=False, smoke_full=True)
        delta = max_delta(ensemble, reference)
        checks.append({"control": f"{recipe}_joint_equals_issue_112_{'v1' if module is v1 else 'v3'}_{arm}_with_probes",
                       "max_abs_parameter_delta": delta, "probes": len(ours["probes"]),
                       "gradient_probe_requests": len(ours["probes"][-1]["gradient"]["norm"][0]),
                       "ok": delta <= EQUIVALENCE_TOLERANCE and len(ours["probes"]) == 4})
        del ensemble, reference
        torch.cuda.empty_cache()
    # cost and memory: the joint arm at the most expensive setting (horizon 225, ranking on), one
    # gradient probe at the end; per-pair update seconds after the compile warm-up
    for recipe in RECIPES:
        for size in (2, 4):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            members = [(fitted[0], SEEDS[i % 3]) for i in range(size)]
            report, _ = train_unit(scratch, recipe, "J", "full", 0, members, data, steps=SMOKE_WARMUP + SMOKE_TIMED,
                                   save=False, probe_every=SMOKE_WARMUP + SMOKE_TIMED, smoke_full=True,
                                   timing_after=SMOKE_WARMUP, step_times=True)
            memory[(recipe, size)] = report["peak_memory_gb"]
            per_pair = {}
            for entry in report["update_times"][SMOKE_WARMUP:]:
                per_pair.setdefault(entry["pair"], []).append(entry["seconds"])
            # median: a recompile inside the timed window must not dominate the projection
            timings[f"{recipe}_m{size}"] = {str(p): float(np.median(v)) for p, v in sorted(per_pair.items())}
            checks.append({"control": f"{recipe}_m{size}_trains_finite",
                           "retired": sum(not m["complete"] for m in report["members"]),
                           "ok": all(m["complete"] for m in report["members"])})
    # scoring mechanics: the restricted scorer reproduces a #112 v3 LOLO record of the joint model
    scoring_check = None
    with harness.deterministic_scoring() as policy:
        scoring = Scoring("fit")
        member = next(m for m in fitted if m in scoring.universe[PRIMARY])
        record = v3().read_record(v3().lolo_path(v3().OUTPUT, "LC", "hybrid", member, SEEDS[0]), v3().SCHEMA_LOLO)
        model = v3().load_member(v3().OUTPUT, "LC", "hybrid", member, SEEDS[0])
        began = time.monotonic()
        fresh = endpoint_block(model, [GATE_B_UNIT_REQUEST], scoring.anchors[member],
                               scoring.universe[PRIMARY][member]["inventory"], scoring.costs)
        curves = lineage_rollout(model, ["F-1-continuous"], scoring.contexts[member], scoring.samples[member],
                                 scoring.objective)
        timings["score_seconds_one_request"] = time.monotonic() - began
        replication = harness.endpoint_replication(record["inventories"][PRIMARY][GATE_B_UNIT_REQUEST]["rows"],
                                                   fresh[GATE_B_UNIT_REQUEST]["rows"])
        stored = [b["errors"]["225"][METRIC] for b in record["rollout"]["F-1-continuous"]]
        ours = [b["errors"]["225"][METRIC] for b in curves["F-1-continuous"]]
        rollout_delta = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(stored, ours))
        scoring_check = {"policy": policy, "endpoint": replication, "rollout_max_relative_delta": rollout_delta}
    checks.append({"control": "restricted_scorer_reproduces_issue_112_v3_lolo_record", **scoring_check,
                   "ok": replication["pass"] and rollout_delta <= 1e-3})
    training_members = v3().load_training_data_members(v3().OUTPUT)
    checks.append({"control": "held_out_isolation", "training_members": training_members,
                   "ok": not (set(training_members) & set(HELD_OUT))})
    projection = project(timings, memory, data_gb(payload))
    ok = all(c["ok"] for c in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timings": timings, "memory_gb": {f"{r}_m{s}": v for (r, s), v in memory.items()},
                "projection": projection, "device": torch.cuda.get_device_name(0),
                "reading": "infrastructure assertions only; nothing kept; no statistic; no held-out access", "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected GPU hours {projection['gpu_seconds'] / 3600:.1f}; "
        f"group sizes {projection['group_sizes']}")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


def data_gb(payload):
    return v1.payload_gb(payload, "hybrid")


def project(timings, memory, extra_gb):
    """Group size per recipe from the memory slope; GPU seconds per unit from per-pair update
    seconds at that size (full horizon throughout: an upper bound)."""
    sizes, out = {}, {}
    for recipe in RECIPES:
        slope = max((memory[(recipe, 4)] - memory[(recipe, 2)]) / 2, 1e-3)
        base = memory[(recipe, 2)] - 2 * slope
        size = max(1, min(MAX_GROUP, int((MEMORY_BUDGET_GB - base) // slope)))
        groups = math.ceil(len(ensemble_members()) / size)
        sizes[recipe] = math.ceil(len(ensemble_members()) / groups)
        t2, t4 = timings[f"{recipe}_m2"], timings[f"{recipe}_m4"]
        per_pair = {int(p): t2[p] + (t4[p] - t2[p]) / 2 * (sizes[recipe] - 2) for p in t2}
        out[recipe] = {"group_size": sizes[recipe], "groups": groups, "gb_per_member": slope, "base_gb": base,
                       "data_gb": extra_gb, "projected_peak_gb": base + slope * sizes[recipe],
                       "seconds_per_update": per_pair}
    units = {}
    for recipe, arm, budget in UNITS:
        p = out[recipe]
        units[unit_key(recipe, arm, budget)] = p["groups"] * sum(p["seconds_per_update"][q]
                                                                 for q, _ in schedule_of(arm, budget))
    return {"group_sizes": sizes, "per_recipe": out, "unit_seconds": units, "gpu_seconds": sum(units.values())}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def published_retirement_paths():
    return sorted((v1.OUTPUT / "training").glob("*--group*.json"))


def input_bindings(output):
    names = {
        "issue_112_v3_plan": v3().OUTPUT / "plan.json",
        "issue_112_v3_training_data": v1.data_path(v3().OUTPUT),
        "issue_112_v3_anchors_fit": v1.anchors_path(v3().OUTPUT, "fit"),
        "issue_112_v3_contexts_fit": v1.contexts_path(v3().OUTPUT, "fit"),
        "issue_112_v1_anchors_heldout": v1.anchors_path(v1.OUTPUT, "heldout"),
        "issue_112_v1_contexts_heldout": v1.contexts_path(v1.OUTPUT, "heldout"),
        "issue_112_v3_compute": v3().OUTPUT / "compute.json",
        "issue_112_v1_runner": ROOT / v1.RUNNER,
        "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_112_v1_module": ROOT / v1.MODEL_MODULE,
        "issue_112_v3_module": ROOT / versions.STABLE_MODULE,
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "module": ROOT / MODULE,
        "smoke": Path(output) / "smoke.json",
        **{f"issue_112_v1_training_{p.stem}": p for p in published_retirement_paths()},
    }
    return [{"name": name, "artifact": str(Path(path).resolve().relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence):
    projection = smoke_evidence["projection"]
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 117, "parent_issue": 108,
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "module": MODULE, "engine_seconds": 0,
        "question": ("do the nine requests that share every weight of the #112 hybrid predictor interfere: what does "
                     "sharing cost each request against a matched single-request specialist and a per-Delta model, "
                     "is the cost asymmetric across Delta and description, does it show in the training dynamics, "
                     "and does the #112 v3 (Gate-B) recipe have an interference problem that warrants a v4 before "
                     "it is retrained on the #104 fit split?"),
        "disclosure": ("Known before freeze: every #112 v1/v2/v3 table, including the v1 retirements (28 of 312, all "
                       "Delta = 1 updates; their per-request attribution in the #112 closing comment), the v3 LOLO "
                       "AUCs and rollout errors of the joint arms, and the ICLR shared-nine vs shared-three ratios "
                       "quoted in #117. No specialist or per-Delta model of these recipes has been trained or scored. "
                       "Held-out lineages 007/008/014/016 are exposed (v1 held-out look); here they are scored once, "
                       "descriptively, after every LOLO record."),
        "hardware": ("4x RTX 5090 (server migration, SERVER_SETUP.md), one training or scoring process per GPU; the "
                     "#85 rule names a single RTX 3090. The joint arm is retrained here on the same hardware as "
                     "every other arm instead of reusing the #112 checkpoints; the #112 v3 joint numbers are "
                     "reported next to it as a cross-hardware replication reading."),
        "encoder": "E99 (#99 spatial slot parser); its carriers, windows, anchors and contexts as cached by #112 v3",
        "recipes": RECIPES, "arms": {a: {"requests": arm_requests(a), "definition": ARM_DEFINITIONS[a],
                                         "budgets": list(arm_budgets(a))} for a in ARMS},
        "budgets": BUDGET_DEFINITIONS, "units": [unit_key(*u) for u in UNITS],
        "requests": list(REQUESTS), "seeds": list(SEEDS),
        "folds": {"fit_lineages": fit_lineages(), "folds": list(v3().folds()),
                  "members_per_unit": len(ensemble_members()),
                  "rule": "the #112 rule: fold L trains on every fit lineage but L; fold 'all' on all 12"},
        "training": {
            "recipe": v3().recipe(), "long": v1.LONG, "rank": v1.RANK, "truncation_LC3": 45, "patience": PATIENCE,
            "initialization": "torch.manual_seed(seed); s99.new_predictor('hybrid') (the #112 initialization)",
            "data_streams": ("the #112 per-member generators (windows: seed; long: seed + 112001; ranking: seed + "
                             "112002) and pools; every arm makes the #112 draws for each of its updates"),
            "matched": ("every arm: the CNNHybridPredictor architecture and parameter count (1 837 690); a "
                        "request's updates touch only the parameters it reads"),
            "group_sizes": projection["group_sizes"],
        },
        "probes": {
            "cadence": (f"after the update at which the joint step g crosses a multiple of {PROBE_EVERY} (every "
                        f"{PROBE_EVERY} updates for the joint and full-budget arms), after the last update, and "
                        "before the first"),
            "validation": (f"every arm, each of its requests: on the member's left-out lineage ({PROBE_ROWS} #77 "
                           f"windows, {PROBE_ROWS} long windows drawn once by seed + {VALIDATION_SEED_OFFSET}, every "
                           "ranking cell of the lineage): the #77 local loss, the long-horizon loss at 225 and the "
                           "ranking loss (values, no gradient); fold 'all' members are in-sample and excluded"),
            "gradients": (f"joint arm, each of the nine requests: the recipe's training loss at step g (curriculum "
                          f"horizon and ranking ramp) on a fixed training-pool batch ({PROBE_ROWS} windows, "
                          f"{PROBE_ROWS} long windows, 2 ranking cells; drawn once by seed + {PROBE_SEED_OFFSET}); "
                          "its gradient before clipping: norm over all parameters, and the Gram matrix of the nine "
                          f"gradients on the shared trunk ({', '.join(ri.TRUNK_PREFIXES)}) -> pairwise cosines"),
            "updates": ("every update: loss, its terms and the pre-clip gradient norm per member; every nonfinite "
                        "loss or gradient norm is attributed to the request that update trained"),
            "non_interference": "probes use their own generators and no optimizer state (smoke-verified)",
        },
        "scoring": {
            "lolo": ("fold L member of every unit on lineage L: every inventory cell (all items) x the unit's "
                     "requests (m99.rollout_endpoints from L's E99 anchor to 225; EndpointCosts rows) and L's N1 "
                     f"rollout error at t = {list(TIMES)} (#100 engine-referenced metric); AUC per mixed cell"),
            "heldout": ("after every LOLO record exists, once: fold-'all' members of every unit on the held-out "
                        "lineages, every inventory and request, plus their N1 rollout error; DESCRIPTIVE"),
        },
        "statistics": {
            "rollout_ratio": ("per (recipe, budget, request, t): per lineage, the mean over available seeds of each "
                              "unit's branch-mean engine-referenced error; log(joint / specialist) per lineage; "
                              "member-clustered bootstrap of the mean log ratio; reported as a geometric ratio "
                              "exp(mean) [exp(low), exp(high)]; the same for joint / per-Delta and per-Delta / "
                              "specialist"),
            "auc": ("tie-free AUC per inventory, unit and request (m99.view; member-clustered bootstrap); paired "
                    "joint - specialist, joint - per-Delta, per-Delta - specialist per request and request-mean"),
            "asymmetry": ("per recipe and budget: the mean log ratio (and mean AUC difference) over the three "
                          "descriptions of each Delta and over the three Deltas of each description; Delta = 1 "
                          "minus Delta = 15 per lineage, bootstrapped"),
            "dynamics": ("validation curves: mean over LOLO members per probe; gradient cosines: mean over members "
                         f"per probe and per phase {list(PHASES)}; share of negative cosines; pre-clip norms"),
        },
        "decision_rule": {
            "claim_1_5x": (f"per (recipe, budget) and request: joint / specialist rollout-error ratio at t = 225 "
                           f"(LOLO): 'supported' (sharing costs no more than {CLAIM}x) iff its upper bound < {CLAIM}; "
                           f"'not_supported_by_this_experiment' iff its lower bound > {CLAIM}; otherwise, or with "
                           f"fewer than {MIN_CLUSTERS} lineages, {STOP_TOKEN}. Per (recipe, budget): supported iff "
                           f"every request is supported; not_supported iff any request is not_supported; otherwise "
                           f"{STOP_TOKEN}; {STOP_TOKEN} if a guard of that (recipe, budget) fails"),
            "v4_trigger": (f"the #112 v3 recipe (LC3) has an interference problem iff, at the per-request budget "
                           f"(the negative-transfer contrast: same updates of the request, with or without the other "
                           f"requests' updates), any of: T1 some request's joint / specialist t = 225 ratio has lower "
                           f"bound > {CLAIM}; T2 the request-mean paired LOLO grid AUC joint - specialist, or that of "
                           f"the Gate-B unit {GATE_B_UNIT_REQUEST}, has upper bound < 0; T3 a joint member retires or "
                           f"skips more than {SKIP_CAP} updates while no LC3 specialist or per-Delta member retires on "
                           "the same request. If any fires, a v4 with a per-request mitigation named by the item-3 "
                           "diagnosis is built under the #112 version policy and re-selected on LOLO; otherwise v3 "
                           "stays the Gate-B recipe. The full budget is reported, not used by the trigger"),
            "multiplicity": ("9 requests x 3 recipes x 2 budgets of per-request intervals; nothing is selected; the "
                             "trigger scans 9 requests (T1) and 2 AUC contrasts (T2) of one recipe and budget"),
            "labels": "LOLO: EXPLORATORY; held-out: DESCRIPTIVE",
        },
        "guards": {
            "G1": "no training window, long window or ranking cell of a held-out lineage",
            "G2": "every unit trained its full schedule; every member complete or retired as a typed failure",
            "G3": ("every scoring record carries the #111 SCORING_POLICY; --validate re-scores one LOLO and one "
                   "held-out record and endpoint_replication passes"),
            "G4": (f"per (recipe, budget): nonfinite share <= {NONFINITE_SHARE_MAX} on LOLO grid cells of its units "
                   "(else its tokens read readiness)"),
            "chronology": "plan frozen before the first update; every LOLO record scored before the first held-out record",
            "caps": "GPU and wall seconds within the caps",
        },
        "caps": {"gpu_seconds": max(2.0 * projection["gpu_seconds"], 24 * 3600.0),
                 "wall_seconds": 48 * 3600.0, "stop": STOP_TOKEN},
        "projection": projection,
        "prohibited": ["changing any recipe, arm, budget, schedule, probe or rule after training starts",
                       "retrying typed failures", "reading a held-out record before every LOLO record exists",
                       "re-opening any prior disposition (#15-#100, #109, #111, #112)"],
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("NovPhy only (items 1-3 of #117; CLEVRER and Physion-Dominoes are item 5); E99 carriers; "
                           "decision-only ranking of retained engine-verdicted executions; fit lineages by LOLO and "
                           "the 4 exposed held-out lineages once, descriptively; no engine access, rendering or "
                           "capture; no closed-loop or competence claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-117 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if entry["name"] != "smoke" and sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("models", "training", "records", "updates"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {len(UNITS)} units x {len(ensemble_members())} members; projected "
        f"{plan['projection']['gpu_seconds'] / 3600:.1f} GPU hours")
    return 0


# ---------------------------------------------------------------------------
# modes: train, score-lolo, score-heldout
# ---------------------------------------------------------------------------

def unit_groups(plan):
    out = []
    for recipe, arm, budget in UNITS:
        size = plan["training"]["group_sizes"][recipe]
        for index, members in enumerate(member_groups(size)):
            out.append((recipe, arm, budget, index, members))
    cost = plan["projection"]["unit_seconds"]
    return sorted(out, key=lambda u: -cost[unit_key(*u[:3])])


def train(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    torch.backends.cudnn.benchmark = True
    data = None
    for recipe, arm, budget, index, members in unit_groups(plan):
        if unit_path(output, recipe, arm, budget, index).is_file():
            continue
        if not claim(claim_path(output, recipe, arm, budget, index)):
            continue
        if unit_path(output, recipe, arm, budget, index).is_file():
            continue
        data = data or Data(v3().load_training_data(v3().OUTPUT))
        began = time.monotonic()
        report, ensemble = train_unit(output, recipe, arm, budget, index, members, data)
        del ensemble
        torch.cuda.empty_cache()
        claim_path(output, recipe, arm, budget, index).unlink(missing_ok=True)
        mark(output, f"train_{unit_name(recipe, arm, budget)}_group{index}", time.monotonic() - began,
             report["gpu_seconds"], members=len(members), retired=sum(not m["complete"] for m in report["members"]),
             worker=worker, device=torch.cuda.get_device_name(0))
        log(f"worker {worker}: trained {unit_name(recipe, arm, budget)}/group{index}: {report['gpu_seconds']:.0f}s, "
            f"retired {sum(not m['complete'] for m in report['members'])}")
    return 0


def training_complete(output, plan):
    return all(unit_path(output, r, a, b, i).is_file() for r, a, b, i, _ in unit_groups(plan))


def lolo_cells():
    cells = [(recipe, arm, budget, fold, seed) for recipe, arm, budget in UNITS for fold in fit_lineages()
             for seed in SEEDS]
    return sorted(cells, key=lambda c: (-len(ARMS[c[1]]), unit_key(*c[:3]), c[3], c[4]))


def score_lolo(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if not training_complete(output, plan):
        raise ValueError("training incomplete; leave-one-lineage-out scoring needs every member")
    if any(output.glob("records/heldout--*.json")):
        raise ValueError("held-out records exist; LOLO scoring precedes them")
    began = time.monotonic()
    pending = [c for i, c in enumerate(lolo_cells()) if i % workers == worker and not lolo_path(output, *c).is_file()]
    progress = v1.Progress(f"worker {worker} LOLO records", len(pending))
    with harness.deterministic_scoring() as policy:
        scoring = Scoring("fit")
        for cell in pending:
            write_json(lolo_path(output, *cell), lolo_record(output, scoring, *cell, policy))
            progress.step(25)
    wall = time.monotonic() - began
    mark(output, f"score_lolo_worker{worker}", wall, wall, records=len(pending))
    return 0


def lolo_complete(output):
    return all(lolo_path(output, *c).is_file() for c in lolo_cells())


def score_heldout(output, workers, worker):
    output = Path(output)
    load_plan(output)
    if not lolo_complete(output):
        raise ValueError("every LOLO record precedes the held-out look")
    began = time.monotonic()
    cells = [(recipe, arm, budget, seed) for recipe, arm, budget in UNITS for seed in SEEDS]
    pending = [c for i, c in enumerate(cells) if i % workers == worker and not heldout_path(output, *c).is_file()]
    progress = v1.Progress(f"worker {worker} held-out records", len(pending))
    with harness.deterministic_scoring() as policy:
        scoring = Scoring("heldout")
        for cell in pending:
            write_json(heldout_path(output, *cell), heldout_record(output, scoring, *cell, policy))
            progress.step(10)
    wall = time.monotonic() - began
    mark(output, f"score_heldout_worker{worker}", wall, wall, records=len(pending))
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def cohort_rows(output, universe, inventory, cohort):
    """One row per (lineage with a mixed cell, seed); views keyed (cost, unit, request)."""
    rows = []
    lineages = (fit_lineages() if cohort == "lolo"
                else sorted(m for m in universe[inventory] if m in HELD_OUT))
    for lineage in lineages:
        entry = universe[inventory].get(lineage)
        if entry is None or not t96.mixed(entry):
            continue
        for seed in SEEDS:
            views = {}
            for recipe, arm, budget in UNITS:
                if cohort == "lolo":
                    record = read_record(lolo_path(output, recipe, arm, budget, lineage, seed), SCHEMA_LOLO)
                    block = record["inventories"].get(inventory)
                else:
                    record = read_record(heldout_path(output, recipe, arm, budget, seed), SCHEMA_HELDOUT)
                    block = None if record["typed_failure"] else record["inventories"][inventory][lineage]
                for request in arm_requests(arm):
                    rows_ = None if record["typed_failure"] else block[request]["rows"]
                    for cost in COSTS:
                        views[(cost, unit_key(recipe, arm, budget), request)] = v1.cost_view(entry, rows_, cost)
            rows.append({"member": lineage, "seed": seed, "views": views})
    return rows


def comparison_units(recipe, budget, request):
    return {"J": unit_key(recipe, "J", "full"), "D": unit_key(recipe, delta_arm(request), budget),
            "S": unit_key(recipe, specialist_arm(request), budget)}


COMPARISONS = (("J", "S"), ("J", "D"), ("D", "S"))
GROUPS = {**{f"delta_{d}": [r for r in REQUESTS if request_delta(r) == d] for d in DELTAS},
          **{f"alpha_{a}": [r for r in REQUESTS if request_alpha(r) == a] for a in ALPHAS},
          "all": list(REQUESTS)}


def auc_contrasts(rows, cost):
    out = {}
    for recipe in RECIPES:
        for budget in ri.BUDGETS:
            block = {}
            for first, second in COMPARISONS:
                name = f"{first}-{second}"
                per = {r: m99.paired(rows, [(1, (cost, comparison_units(recipe, budget, r)[first], r)),
                                            (-1, (cost, comparison_units(recipe, budget, r)[second], r))])
                       for r in REQUESTS}
                groups = {}
                for group, requests in GROUPS.items():
                    n = len(requests)
                    groups[group] = m99.paired(rows, [(1 / n, (cost, comparison_units(recipe, budget, r)[first], r))
                                                      for r in requests]
                                               + [(-1 / n, (cost, comparison_units(recipe, budget, r)[second], r))
                                                  for r in requests])
                pieces = []
                for d, sign in ((1, 1), (15, -1)):
                    for r in GROUPS[f"delta_{d}"]:
                        units = comparison_units(recipe, budget, r)
                        pieces += [(sign / 3, (cost, units[first], r)), (-sign / 3, (cost, units[second], r))]
                block[name] = {"per_request": per, "groups": groups,
                               "delta1_minus_delta15": m99.paired(rows, pieces)}
            out[f"{recipe}/{budget}"] = block
    return out


def auc_tables(rows):
    units = {}
    for recipe, arm, budget in UNITS:
        for request in arm_requests(arm):
            units[f"{unit_key(recipe, arm, budget)}:{request}"] = v1.ranker_block(
                rows, (PRIMARY_COST, unit_key(recipe, arm, budget), request))
    return {"members": sorted({r["member"] for r in rows}), "cells": len(rows), "units": units,
            "contrasts": auc_contrasts(rows, PRIMARY_COST)}


def rollout_per(output, cohort):
    """{(unit, request): {lineage: [branches per seed or None]}}."""
    per = {}
    for recipe, arm, budget in UNITS:
        key = unit_key(recipe, arm, budget)
        for request in arm_requests(arm):
            per[(key, request)] = {}
        if cohort == "lolo":
            for fold in fit_lineages():
                records = [read_record(lolo_path(output, recipe, arm, budget, fold, s), SCHEMA_LOLO) for s in SEEDS]
                if any(r["rollout"] is None and r["typed_failure"] is None for r in records):
                    continue
                for request in arm_requests(arm):
                    per[(key, request)][fold] = [None if r["typed_failure"] else r["rollout"][request] for r in records]
        else:
            records = [read_record(heldout_path(output, recipe, arm, budget, s), SCHEMA_HELDOUT) for s in SEEDS]
            lineages = sorted({m for r in records for m in r["rollout"]})
            for request in arm_requests(arm):
                per[(key, request)] = {m: [None if r["typed_failure"] else r["rollout"][m][request] for r in records]
                                       for m in lineages}
    return per


def lineage_value(per, unit, request, lineage, t):
    seeds = per[(unit, request)].get(lineage)
    if seeds is None:
        return None
    values = [v1.branch_mean(b, t, METRIC)[0] for b in seeds if b is not None]
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


def ratio_block(per, triples, t):
    """triples [(first unit, second unit, request)]: per lineage the mean over the triples of
    log(first / second); bootstrap over lineages; geometric ratio."""
    lineages = sorted(set.intersection(*(set(per[(u, r)]) & set(per[(w, r)]) for u, w, r in triples)))
    values, clusters = [], []
    for lineage in lineages:
        logs = []
        for first, second, request in triples:
            a = lineage_value(per, first, request, lineage, t)
            b = lineage_value(per, second, request, lineage, t)
            if a is None or b is None or a <= 0 or b <= 0:
                logs = None
                break
            logs.append(math.log(a / b))
        if logs:
            values.append(float(np.mean(logs)))
            clusters.append(lineage)
    block = bootstrap(values, clusters)
    if block is None:
        return None
    return {"ratio": math.exp(block["mean"]), "interval": [math.exp(v) for v in block["interval"]],
            "log_mean": block["mean"], "log_interval": block["interval"], "clusters": block["clusters"],
            "label": block["label"]}


def log_difference_block(per, first_triples, second_triples, t):
    """Per lineage: mean log ratio of the first triples minus that of the second; bootstrap."""
    lineages = sorted(set.intersection(*(set(per[(u, r)]) & set(per[(w, r)])
                                         for u, w, r in first_triples + second_triples)))
    values, clusters = [], []
    for lineage in lineages:
        parts = []
        for triples in (first_triples, second_triples):
            logs = []
            for first, second, request in triples:
                a = lineage_value(per, first, request, lineage, t)
                b = lineage_value(per, second, request, lineage, t)
                if a is None or b is None or a <= 0 or b <= 0:
                    logs = None
                    break
                logs.append(math.log(a / b))
            parts.append(None if not logs else float(np.mean(logs)))
        if None not in parts:
            values.append(parts[0] - parts[1])
            clusters.append(lineage)
    block = bootstrap(values, clusters)
    if block is None:
        return None
    return {"ratio_of_ratios": math.exp(block["mean"]), "interval": [math.exp(v) for v in block["interval"]],
            "log_mean": block["mean"], "log_interval": block["interval"], "clusters": block["clusters"]}


def rollout_tables(per):
    curves = {}
    for (unit, request), lineages in per.items():
        for t in TIMES:
            per_lineage = {lineage: [None if b is None else v1.branch_mean(b, t, METRIC)[0] for b in seeds]
                           for lineage, seeds in lineages.items()}
            curves[f"{unit}:{request}:{t}"] = v1.rollout_block(per_lineage)
    ratios = {}
    for recipe in RECIPES:
        for budget in ri.BUDGETS:
            block = {}
            for first, second in COMPARISONS:
                name = f"{first}/{second}"
                triples = {r: [(comparison_units(recipe, budget, r)[first], comparison_units(recipe, budget, r)[second],
                                r)] for r in REQUESTS}
                block[name] = {
                    "per_request": {r: {str(t): ratio_block(per, triples[r], t) for t in TIMES} for r in REQUESTS},
                    "groups": {g: ratio_block(per, [x for r in rs for x in triples[r]], ENDPOINT)
                               for g, rs in GROUPS.items()},
                    "delta1_over_delta15": log_difference_block(
                        per, [x for r in GROUPS["delta_1"] for x in triples[r]],
                        [x for r in GROUPS["delta_15"] for x in triples[r]], ENDPOINT)}
            ratios[f"{recipe}/{budget}"] = block
    return curves, ratios


def claim_token(block):
    if block is None or block["clusters"] < MIN_CLUSTERS:
        return STOP_TOKEN
    if block["interval"][1] < CLAIM:
        return "supported"
    if block["interval"][0] > CLAIM:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def dispositions(ratios, g4):
    out = {}
    for recipe in RECIPES:
        for budget in ri.BUDGETS:
            key = f"{recipe}/{budget}"
            per = {r: claim_token(ratios[key]["J/S"]["per_request"][r][str(ENDPOINT)]) for r in REQUESTS}
            tokens = set(per.values())
            if not g4[key]["pass"]:
                overall = STOP_TOKEN
            elif tokens == {"supported"}:
                overall = "supported"
            elif "not_supported_by_this_experiment" in tokens:
                overall = "not_supported_by_this_experiment"
            else:
                overall = STOP_TOKEN
            out[key] = {"benchmark": "NovPhy", "per_request": per, "token": overall, "label": "EXPLORATORY (LOLO)",
                        "claim": f"sharing all weights costs no more than {CLAIM}x per-request rollout error at "
                                 f"t = {ENDPOINT} against a matched specialist"}
    return out


# ---- training dynamics ----

def unit_reports(output, plan):
    reports = {}
    for recipe, arm, budget, index, _ in unit_groups(plan):
        reports.setdefault((recipe, arm, budget), []).append(
            read_record(unit_path(output, recipe, arm, budget, index), SCHEMA_UNIT))
    return {k: sorted(v, key=lambda r: r["group"]) for k, v in reports.items()}


def validation_curves(reports):
    """Per unit and request: per probe the mean over LOLO members of each validation term."""
    out = {}
    for (recipe, arm, budget), groups in reports.items():
        probes = len(groups[0]["probes"])
        if any(len(g["probes"]) != probes for g in groups):
            raise ValueError("groups of one unit carry different probe counts")
        for request in arm_requests(arm):
            curve = []
            for p in range(probes):
                head = groups[0]["probes"][p]
                entry = {"update": head["update"], "step": head["step"],
                         "request_updates": head["request_updates"][request]}
                for term in ("local", "long225", "rank"):
                    values = []
                    for group in groups:
                        validation = group["probes"][p]["validation"][request][term]
                        for info, value, member in zip(group["validation_members"], validation, group["members"]):
                            if info["in_sample"] or value is None:
                                continue
                            if term == "rank" and not info["validation_cells"]:
                                continue
                            values.append(value)
                    entry[term] = float(np.mean(values)) if values else None
                    entry[f"{term}_members"] = len(values)
                curve.append(entry)
            out[f"{unit_key(recipe, arm, budget)}:{request}"] = curve
    return out


def gradient_tables(reports):
    """Joint arm per recipe: per-probe mean trunk cosine matrix and pre-clip norms; phase means;
    shares of negative cosines by pair kind."""
    out = {}
    for recipe in RECIPES:
        groups = reports[(recipe, "J", "full")]
        probes = len(groups[0]["probes"])
        per_probe = []
        for p in range(probes):
            grams, norms, step = [], [], groups[0]["probes"][p]["step"]
            for group in groups:
                gradient = group["probes"][p]["gradient"]
                for i, member in enumerate(group["members"]):
                    grams.append(np.array([[np.nan if v is None else v for v in row] for row in gradient["gram"][i]]))
                    norms.append(np.array([np.nan if v is None else v for v in gradient["norm"][i]]))
            grams = np.stack(grams)
            diag_ = np.sqrt(np.clip(np.einsum("mii->mi", grams), 0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                cos = grams / (diag_[:, :, None] * diag_[:, None, :])
            per_probe.append({"step": step, "update": groups[0]["probes"][p]["update"], "cosines": cos,
                              "norms": np.stack(norms)})
        out[recipe] = summarize_cosines(per_probe)
    return out


def pair_kind(i, j):
    a, b = REQUESTS[i], REQUESTS[j]
    if request_delta(a) == request_delta(b):
        return "same_delta"
    if request_alpha(a) == request_alpha(b):
        return "same_alpha"
    return "different"


def summarize_cosines(per_probe):
    trained = [p for p in per_probe if p["update"] > 0]

    def matrix(probes):
        if not probes:
            return None
        stacked = np.concatenate([p["cosines"] for p in probes])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stacked, axis=0)
        return [[None if not np.isfinite(v) else float(v) for v in row] for row in mean]

    def kinds(probes):
        stats = {}
        stacked = np.concatenate([p["cosines"] for p in probes]) if probes else None
        for kind in ("same_delta", "same_alpha", "different"):
            values = []
            if stacked is not None:
                for i in range(9):
                    for j in range(i + 1, 9):
                        if pair_kind(i, j) == kind:
                            v = stacked[:, i, j]
                            values.extend(v[np.isfinite(v)].tolist())
            stats[kind] = {"mean": float(np.mean(values)) if values else None,
                           "negative_share": float(np.mean(np.array(values) < 0)) if values else None,
                           "min": float(np.min(values)) if values else None, "values": len(values)}
        return stats

    phases = {f"{a}-{b}": [p for p in trained if a < p["step"] + 1 <= b] for a, b in PHASES}
    norms = {}
    for p in per_probe:
        with np.errstate(invalid="ignore"):
            norms[str(p["step"]) if p["update"] else "initial"] = [
                None if not np.isfinite(v) else float(v) for v in np.nanmean(p["norms"], axis=0)]
    curve = []
    for p in per_probe:
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(p["cosines"], axis=0)
        off = [mean[i, j] for i in range(9) for j in range(i + 1, 9) if np.isfinite(mean[i, j])]
        curve.append({"step": p["step"], "update": p["update"], "mean_offdiagonal": float(np.mean(off)) if off else None,
                      "min_offdiagonal": float(np.min(off)) if off else None})
    return {"matrix_all": matrix(trained), "matrix_phases": {k: matrix(v) for k, v in phases.items()},
            "kinds_all": kinds(trained), "kinds_phases": {k: kinds(v) for k, v in phases.items()},
            "norm_mean_per_probe": norms, "curve": curve, "requests": list(REQUESTS)}


def grad_norm_tables(reports):
    """Pre-clip update norms per request: joint vs specialist and per-Delta (median over members of
    the member median, and the largest member maximum)."""
    out = {}
    for (recipe, arm, budget), groups in reports.items():
        for request in arm_requests(arm):
            medians, p99, maxes = [], [], []
            for group in groups:
                block = group["grad_norm_summary"][request]
                medians += [v for v in block["median"] if v is not None]
                p99 += [v for v in block["p99"] if v is not None]
                maxes += [v for v in block["max"] if v is not None]
            out[f"{unit_key(recipe, arm, budget)}:{request}"] = {
                "median": float(np.median(medians)) if medians else None,
                "p99": float(np.median(p99)) if p99 else None, "max": float(np.max(maxes)) if maxes else None}
    return out


def retirement_tables(reports):
    """Every retired member and every member with a nonfinite update, attributed to the request of
    that update; the #112 v1 published retirements attributed the same way."""
    here = []
    for (recipe, arm, budget), groups in reports.items():
        for group in groups:
            for event in group["nonfinite_events"]:
                first = event["events"][0]
                here.append({"recipe": recipe, "arm": arm, "budget": budget, "fold": event["fold"],
                             "seed": event["seed"], "retired": event["retired"], "nonfinite_updates": event["count"],
                             "first_update": first["update"], "first_step": first["step"],
                             "first_request": first["request"], "loss_finite": first["loss_finite"],
                             "grad_norm_finite": first["grad_norm_finite"], "horizon": first["horizon"],
                             "requests_nonfinite": sorted({e["request"] for e in event["events"]}),
                             "preceding_max_grad_norm": event["first_preceding_max_grad_norm"]})
    published = []
    for path in published_retirement_paths():
        report = read_json(path)
        for member in report["members"]:
            if member["retired_at"] is None:
                continue
            step = member["retired_at"]
            pair = v1.step_pair(report["family"], step)
            request = (REQUESTS[PAIRS.index(pair)] if report["family"] == "hybrid" else f"C-F-{pair.delta}")
            published.append({"arm": report["arm"], "family": report["family"], "fold": member["fold"],
                              "seed": member["seed"], "update": step, "request": request,
                              "horizon": schedule_values(step)[0]})
    counts = {}
    for row in published:
        key = f"{row['arm']}:{row['family']}:{row['request']}"
        counts[key] = counts.get(key, 0) + 1
    by_unit = {}
    for (recipe, arm, budget), groups in reports.items():
        members = [m for g in groups for m in g["members"]]
        retired = {}
        for m in members:
            if not m["complete"]:
                retired[m["retired_request"]] = retired.get(m["retired_request"], 0) + 1
        by_unit[unit_key(recipe, arm, budget)] = {"members": len(members),
                                                 "retired": sum(not m["complete"] for m in members),
                                                 "retired_by_request": retired,
                                                 "members_with_skips": sum(1 for m in members if m["skipped"]),
                                                 "max_skipped": max(m["skipped"] for m in members),
                                                 "total_skipped": sum(m["skipped"] for m in members)}
    return {"events": here, "issue_112_v1": published, "issue_112_v1_counts": counts, "by_unit": by_unit}


def training_overview(reports):
    out = {}
    for (recipe, arm, budget), groups in reports.items():
        out[unit_key(recipe, arm, budget)] = {
            "gpu_seconds": sum(g["gpu_seconds"] for g in groups), "updates": groups[0]["updates"],
            "peak_memory_gb": max(g["peak_memory_gb"] for g in groups),
            "parameters_per_member": groups[0]["parameters_per_member"]}
    return out


# ---- decision ----

def v4_trigger(auc, ratios, retirements, g4):
    key = f"{GATE_B_RECIPE}/request"
    t1_rows = {r: ratios[key]["J/S"]["per_request"][r][str(ENDPOINT)] for r in REQUESTS}
    t1 = [r for r, b in t1_rows.items() if b is not None and b["clusters"] >= MIN_CLUSTERS and b["interval"][0] > CLAIM]
    grid = auc[PRIMARY]["contrasts"][key]["J-S"]
    mean_block = grid["groups"]["all"]["estimate"]
    unit_block = grid["per_request"][GATE_B_UNIT_REQUEST]["estimate"]
    t2 = [name for name, b in (("request_mean", mean_block), (GATE_B_UNIT_REQUEST, unit_block))
          if b is not None and b["interval"][1] < 0]
    joint = retirements["by_unit"][unit_key(GATE_B_RECIPE, "J", "full")]
    others = {u: b for u, b in retirements["by_unit"].items() if u.startswith(f"{GATE_B_RECIPE}/") and "/J/" not in u}
    t3 = []
    joint_events = [e for e in retirements["events"] if e["recipe"] == GATE_B_RECIPE and e["arm"] == "J"]
    for request in REQUESTS:
        joint_bad = (joint["retired_by_request"].get(request, 0) > 0
                     or any(e["first_request"] == request and e["nonfinite_updates"] > SKIP_CAP for e in joint_events))
        other_bad = any(b["retired_by_request"].get(request, 0) > 0 for b in others.values())
        if joint_bad and not other_bad:
            t3.append(request)
    fired = {"T1": bool(t1), "T2": bool(t2), "T3": bool(t3)}
    guards_ok = g4[key]["pass"]
    problem = guards_ok and any(fired.values())
    return {"recipe": GATE_B_RECIPE, "budget": "request",
            "T1": {"fired": fired["T1"], "requests": t1,
                   "ratios": {r: (None if b is None else {"ratio": b["ratio"], "interval": b["interval"]})
                              for r, b in t1_rows.items()}},
            "T2": {"fired": fired["T2"], "which": t2, "request_mean": mean_block, GATE_B_UNIT_REQUEST: unit_block},
            "T3": {"fired": fired["T3"], "requests": t3, "joint": joint},
            "g4_pass": guards_ok, "interference_problem": problem,
            "action": "build_v4" if problem else "keep_v3",
            "reading": ("a v4 with a per-request mitigation named by the item-3 diagnosis, re-selected on LOLO" if problem
                        else "v3 stays the Gate-B recipe: no pre-declared interference criterion fired")}


def replication_reading(auc, curves):
    """The joint arms retrained here (RTX 5090) next to the published #112 v3 joint arms (RTX 3090)."""
    published = read_json(v3().OUTPUT / "compute.json")
    out = {}
    for recipe, arm112 in (("B", "B"), ("LC3", "LC")):
        unit = unit_key(recipe, "J", "full")
        rows = {}
        for request in REQUESTS:
            ours = auc[PRIMARY]["units"][f"{unit}:{request}"]["auc"]
            theirs = published["lolo"][PRIMARY]["rankers"][PRIMARY_COST][f"{arm112}:{request}"]["auc"]
            ours_r = curves.get(f"{unit}:{request}:{ENDPOINT}")
            theirs_r = published["rollout"]["lolo"].get(f"{arm112}:{request}:{ENDPOINT}:{METRIC}")
            rows[request] = {"grid_auc_here": None if ours is None else ours["mean"],
                             "grid_auc_112_v3": None if theirs is None else theirs["mean"],
                             "rollout_225_here": None if ours_r is None else ours_r["mean"],
                             "rollout_225_112_v3": None if theirs_r is None else theirs_r["mean"]}
        out[recipe] = rows
    return out


def records_overview(output):
    """Scoring policy of every record and the scoring times of each cohort (cached record reads)."""
    policies, times = [], {"lolo": [], "heldout": []}
    for path in sorted(Path(output).glob("records/*.json")):
        cohort = path.name.split("--")[0]
        record = read_record(path, SCHEMA_LOLO if cohort == "lolo" else SCHEMA_HELDOUT)
        policies.append(record.get("scoring_policy") == harness.SCORING_POLICY)
        times[cohort].append(record["scored_at"])
    return all(policies), len(policies), times["lolo"], times["heldout"]


def compute_tables(output, plan, spot):
    output = Path(output)
    _, _, universe = v3().load_universe()
    ledger = load_ledger(output)
    reports = unit_reports(output, plan)
    auc = {inventory: auc_tables(cohort_rows(output, universe, inventory, "lolo")) for inventory in INVENTORIES}
    held_auc = {inventory: auc_tables(cohort_rows(output, universe, inventory, "heldout")) for inventory in INVENTORIES}
    curves, ratios = rollout_tables(rollout_per(output, "lolo"))
    held_curves, held_ratios = rollout_tables(rollout_per(output, "heldout"))
    retirements = retirement_tables(reports)
    policies_ok, scored, lolo_times, held_times = records_overview(output)
    g4 = {}
    for recipe in RECIPES:
        for budget in ri.BUDGETS:
            units = {unit_key(recipe, "J", "full")} | {unit_key(recipe, a, budget) for a in ARMS if a != "J"}
            shares = [b["nonfinite_share"] or 0.0 for k, b in auc[PRIMARY]["units"].items() if k.split(":")[0] in units]
            g4[f"{recipe}/{budget}"] = {"worst_nonfinite_share": max(shares), "pass": max(shares) <= NONFINITE_SHARE_MAX}
    trained_at = [r["trained_at"] for groups in reports.values() for r in groups]
    members = v3().load_training_data_members(v3().OUTPUT)
    complete_units = all(r["updates"] == len(schedule_of(a, b)) for (_, a, b), groups in reports.items()
                         for r in groups)
    guards = {
        "G1": {"observed": {"training_members": members}, "pass": not (set(members) & set(HELD_OUT))},
        "G2": {"observed": {"units": len(reports), "groups": sum(len(g) for g in reports.values()),
                            "complete_schedules": complete_units,
                            "retired": sum(b["retired"] for b in retirements["by_unit"].values())},
               "pass": complete_units and len(reports) == len(UNITS)},
        "G3": {"observed": {"records": scored, "all_carry_policy": policies_ok, "spot_replication": spot},
               "pass": policies_ok and spot is not None and all(s["pass"] for s in spot.values())},
        "G4": {"observed": g4, "pass": True, "scope": "per (recipe, budget) disposition"},
        "chronology": {"observed": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": min(trained_at),
                                    "last_lolo_scored_at": max(lolo_times),
                                    "first_heldout_scored_at": min(held_times) if held_times else None},
                       "pass": (plan["frozen_at"] < min(trained_at) and bool(held_times)
                                and max(lolo_times) <= min(held_times))},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"], "wall_seconds": ledger["wall_seconds_elapsed"]},
                 "pass": (ledger["gpu_seconds_elapsed"] <= plan["caps"]["gpu_seconds"]
                          and ledger["wall_seconds_elapsed"] <= plan["caps"]["wall_seconds"])},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    tokens = dispositions(ratios, g4)
    if not guards_ok:
        tokens = {k: {**v, "token": STOP_TOKEN} for k, v in tokens.items()}
    trigger = v4_trigger(auc, ratios, retirements, g4)
    if not guards_ok:
        trigger = {**trigger, "interference_problem": False, "action": "keep_v3",
                   "reading": f"{STOP_TOKEN}: a guard failed; no evidence for a v4"}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "guards": guards, "guards_pass": guards_ok,
        "dispositions": tokens, "decision": trigger,
        "lolo": {"auc": auc, "rollout": curves, "ratios": ratios},
        "heldout": {"auc": held_auc, "rollout": held_curves, "ratios": held_ratios,
                    "label": "DESCRIPTIVE (exposed lineages, scored once after every LOLO record)"},
        "dynamics": {"validation": validation_curves(reports), "gradients": gradient_tables(reports),
                     "grad_norms": grad_norm_tables(reports), "retirements": retirements},
        "replication_112_v3": replication_reading(auc, curves),
        "training": training_overview(reports),
        "compute": {"phases": ledger["phases"], "gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "wall_seconds": ledger["wall_seconds_elapsed"], "engine_seconds": 0},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def ratio_cell(block):
    if block is None:
        return "n/a"
    return f"{num(block['ratio'], 3)} [{num(block['interval'][0], 3)}, {num(block['interval'][1], 3)}]"


def auc_cell(block):
    return "n/a" if block is None else fmt_interval(block)


def short(request):
    return request[2:]


def findings_md(plan, compute):
    lines = []
    add = lines.append
    add("# Issue #117 — interference between requests in the shared predictor (items 1–3, NovPhy)")
    add("")
    add(f"Frozen plan {plan['frozen_at']} (before training and scoring). Validation: `{VALIDATION_COMMAND}`.")
    guard_text = ", ".join(f"{name} {block['pass']}" for name, block in compute["guards"].items())
    add(f"Guards pass: {compute['guards_pass']} ({guard_text}). Engine seconds 0. LOLO: EXPLORATORY; held-out: "
        "DESCRIPTIVE.")
    add("")
    decision = compute["decision"]
    add(f"**Decision (pre-declared v4 trigger, {decision['recipe']} at the per-request budget): "
        f"{decision['action']}.** T1 {decision['T1']['fired']} ({decision['T1']['requests']}), T2 "
        f"{decision['T2']['fired']} ({decision['T2']['which']}), T3 {decision['T3']['fired']} "
        f"({decision['T3']['requests']}). {decision['reading']}.")
    add("")
    add(f"## 1.5× disposition (joint / specialist rollout error at t = {ENDPOINT}, LOLO)")
    add("")
    add("| recipe / budget | token | " + " | ".join(short(r) for r in REQUESTS) + " |")
    add("| --- | --- | " + " | ".join("---" for _ in REQUESTS) + " |")
    for key, block in compute["dispositions"].items():
        add(f"| {key} | {block['token']} | " + " | ".join(block["per_request"][r].replace(
            "readiness_or_precision_insufficient", "insufficient").replace("not_supported_by_this_experiment",
                                                                           "not_supported")
            for r in REQUESTS) + " |")
    add("")
    add(f"## Item 1. Per-request ratios at t = {ENDPOINT} (geometric mean over lineages [95% interval])")
    add("")
    for key, block in compute["lolo"]["ratios"].items():
        add(f"### {key}")
        add("")
        add("| request | joint / specialist | joint / per-Δ | per-Δ / specialist |")
        add("| --- | --- | --- | --- |")
        for r in REQUESTS:
            add(f"| {r} | " + " | ".join(ratio_cell(block[c]["per_request"][r][str(ENDPOINT)])
                                         for c in ("J/S", "J/D", "D/S")) + " |")
        add("")
    add(f"## Item 1. Rollout error at t = {ENDPOINT} (LOLO mean over lineages)")
    add("")
    for recipe in RECIPES:
        add(f"### {recipe}")
        add("")
        add("| request | J | D req | D full | S req | S full |")
        add("| --- | --- | --- | --- | --- | --- |")
        for r in REQUESTS:
            cells = [compute["lolo"]["rollout"].get(f"{u}:{r}:{ENDPOINT}") for u in (
                unit_key(recipe, "J", "full"), unit_key(recipe, delta_arm(r), "request"),
                unit_key(recipe, delta_arm(r), "full"), unit_key(recipe, specialist_arm(r), "request"),
                unit_key(recipe, specialist_arm(r), "full"))]
            add(f"| {r} | " + " | ".join(num(None if c is None else c["mean"]) for c in cells) + " |")
        add("")
    add("## Item 1. LOLO grid AUC (tie-free) and joint − specialist")
    add("")
    grid = compute["lolo"]["auc"][PRIMARY]
    for recipe in RECIPES:
        add(f"### {recipe} ({len(grid['members'])} lineages, {grid['cells']} cells)")
        add("")
        add("| request | J | S req | S full | J − S req | J − S full |")
        add("| --- | --- | --- | --- | --- | --- |")
        for r in REQUESTS:
            units = [unit_key(recipe, "J", "full"), unit_key(recipe, specialist_arm(r), "request"),
                     unit_key(recipe, specialist_arm(r), "full")]
            aucs = [auc_cell(grid["units"][f"{u}:{r}"]["auc"]) for u in units]
            diffs = [auc_cell(grid["contrasts"][f"{recipe}/{b}"]["J-S"]["per_request"][r]["estimate"])
                     for b in ri.BUDGETS]
            add(f"| {r} | " + " | ".join(aucs + diffs) + " |")
        means = [auc_cell(grid["contrasts"][f"{recipe}/{b}"]["J-S"]["groups"]["all"]["estimate"]) for b in ri.BUDGETS]
        add(f"| request-mean | | | | {means[0]} | {means[1]} |")
        add("")
    add("Every inventory (request-mean paired AUC, J − S / J − D / D − S):")
    add("")
    add("| inventory | recipe / budget | J − S | J − D | D − S |")
    add("| --- | --- | --- | --- | --- |")
    for inventory in INVENTORIES:
        for key, block in compute["lolo"]["auc"][inventory]["contrasts"].items():
            add(f"| {inventory} | {key} | " + " | ".join(auc_cell(block[c]["groups"]["all"]["estimate"])
                                                         for c in ("J-S", "J-D", "D-S")) + " |")
    add("")
    add(f"## Item 2. Asymmetry by Δ and by description (joint / specialist at t = {ENDPOINT}; grid AUC J − S)")
    add("")
    add("| recipe / budget | group | rollout ratio | grid AUC J − S |")
    add("| --- | --- | --- | --- |")
    for key, block in compute["lolo"]["ratios"].items():
        for group in GROUPS:
            add(f"| {key} | {group} | {ratio_cell(block['J/S']['groups'][group])} | "
                f"{auc_cell(grid['contrasts'][key]['J-S']['groups'][group]['estimate'])} |")
        d = block["J/S"]["delta1_over_delta15"]
        dd_ = grid["contrasts"][key]["J-S"]["delta1_minus_delta15"]["estimate"]
        ratio_text = "n/a" if d is None else ratio_cell({"ratio": d["ratio_of_ratios"], "interval": d["interval"]})
        add(f"| {key} | Δ1 vs Δ15 (ratio of ratios; AUC difference) | {ratio_text} | {auc_cell(dd_)} |")
    add("")
    add("## Item 3. Training dynamics of the joint model")
    add("")
    add("Pairwise cosine of per-request gradients on the shared trunk (training probe batch, mean over members and "
        "probes after update 0):")
    add("")
    add("| recipe | phase | same Δ (mean / share < 0) | same description | different both |")
    add("| --- | --- | --- | --- | --- |")
    for recipe, block in compute["dynamics"]["gradients"].items():
        for phase, kinds in [("all", block["kinds_all"]), *block["kinds_phases"].items()]:
            add(f"| {recipe} | {phase} | " + " | ".join(
                f"{num(kinds[k]['mean'], 3)} / {num(kinds[k]['negative_share'], 3)}" for k in
                ("same_delta", "same_alpha", "different")) + " |")
    add("")
    add("Pre-clip gradient norm per update (median over members of the member median; largest member maximum):")
    add("")
    add("| request | " + " | ".join(f"{recipe} J / S req / S full" for recipe in RECIPES) + " |")
    add("| --- | " + " | ".join("---" for _ in RECIPES) + " |")
    norms = compute["dynamics"]["grad_norms"]
    for r in REQUESTS:
        cells = []
        for recipe in RECIPES:
            parts = [norms.get(f"{u}:{r}") for u in (unit_key(recipe, "J", "full"),
                                                     unit_key(recipe, specialist_arm(r), "request"),
                                                     unit_key(recipe, specialist_arm(r), "full"))]
            cells.append(" / ".join(f"{num(p['median'], 3)} ({num(p['max'], 1)})" for p in parts))
        add(f"| {r} | " + " | ".join(cells) + " |")
    add("")
    add("Validation loss on the left-out lineage at the last probe (mean over LOLO members; local / long-horizon "
        "at 225), joint vs per-Δ and specialist at both budgets:")
    add("")
    add("| recipe | request | J | D req | D full | S req | S full |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    curves = compute["dynamics"]["validation"]
    for recipe in RECIPES:
        for r in REQUESTS:
            cells = []
            for u in (unit_key(recipe, "J", "full"), unit_key(recipe, delta_arm(r), "request"),
                      unit_key(recipe, delta_arm(r), "full"), unit_key(recipe, specialist_arm(r), "request"),
                      unit_key(recipe, specialist_arm(r), "full")):
                last = curves[f"{u}:{r}"][-1]
                cells.append(f"{num(last['local'], 3)} / {num(last['long225'], 3)}")
            add(f"| {recipe} | {r} | " + " | ".join(cells) + " |")
    add("")
    ret = compute["dynamics"]["retirements"]
    add("Retired members and nonfinite updates, per unit:")
    add("")
    add("| unit | retired / members | retired by request | members with skips | max skips |")
    add("| --- | --- | --- | --- | --- |")
    affected = [(u, b) for u, b in ret["by_unit"].items() if b["retired"] or b["members_with_skips"]]
    for unit, block in affected:
        add(f"| {unit} | {block['retired']}/{block['members']} | {block['retired_by_request']} | "
            f"{block['members_with_skips']} | {block['max_skipped']} |")
    if not affected:
        add("| none | | | | |")
    add("")
    add("First nonfinite update of every affected member (this run):")
    add("")
    add("| unit | fold | seed | update (step) | request | loss finite | norm finite | horizon | retired |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for e in ret["events"]:
        add(f"| {e['recipe']}/{e['arm']}/{e['budget']} | {e['fold'][-3:]} | {e['seed']} | {e['first_update']} "
            f"({e['first_step']}) | {e['first_request']} | {e['loss_finite']} | {e['grad_norm_finite']} | "
            f"{e['horizon']} | {e['retired']} |")
    if not ret["events"]:
        add("| none | | | | | | | | |")
    add("")
    add(f"#112 v1 published retirements by request: {ret['issue_112_v1_counts']}.")
    add("")
    add("## Cross-hardware replication of the #112 v3 joint arms (LOLO grid AUC / rollout at 225)")
    add("")
    add("| request | B here | B #112 v3 | LC3 here | LC #112 v3 |")
    add("| --- | --- | --- | --- | --- |")
    rep = compute["replication_112_v3"]
    for r in REQUESTS:
        b, lc = rep["B"][r], rep["LC3"][r]
        add(f"| {r} | {num(b['grid_auc_here'])} / {num(b['rollout_225_here'])} | {num(b['grid_auc_112_v3'])} / "
            f"{num(b['rollout_225_112_v3'])} | {num(lc['grid_auc_here'])} / {num(lc['rollout_225_here'])} | "
            f"{num(lc['grid_auc_112_v3'])} / {num(lc['rollout_225_112_v3'])} |")
    add("")
    add("## Held-out (007/008/014/016; scored once after every LOLO record; DESCRIPTIVE)")
    add("")
    add("| recipe / budget | grid AUC J − S (request-mean) | rollout J / S at 225 (all requests) |")
    add("| --- | --- | --- |")
    for key in compute["heldout"]["ratios"]:
        add(f"| {key} | {auc_cell(compute['heldout']['auc'][PRIMARY]['contrasts'][key]['J-S']['groups']['all']['estimate'])} "
            f"| {ratio_cell(compute['heldout']['ratios'][key]['J/S']['groups']['all'])} |")
    add("")
    add("## Compute")
    add("")
    add(f"GPU seconds {compute['compute']['gpu_seconds']:.0f}; wall seconds (summed over workers) "
        f"{compute['compute']['wall_seconds']:.0f}; engine seconds 0.")
    add("")
    add(f"Claim boundary: {plan['claim_boundary']}.")
    add("")
    return "\n".join(lines)


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "cohort", "statistic", "value", "interval_low", "interval_high", "clusters", "label"])
    for cohort, label in (("lolo", "EXPLORATORY"), ("heldout", "DESCRIPTIVE")):
        for inventory, table in compute[cohort]["auc"].items():
            for unit, block in table["units"].items():
                b = block["auc"]
                writer.writerow([f"{cohort}:auc:{inventory}:{unit}", cohort, "auc",
                                 *(("", "", "", "") if b is None else (f"{b['mean']:.6f}", f"{b['interval'][0]:.6f}",
                                                                        f"{b['interval'][1]:.6f}", b["clusters"])),
                                 label])
        for key, block in compute[cohort]["ratios"].items():
            for comparison, values in block.items():
                for request, times in values["per_request"].items():
                    for t, r in times.items():
                        writer.writerow([f"{cohort}:ratio:{key}:{comparison}:{request}:{t}", cohort,
                                         "geometric_ratio", *(("", "", "", "") if r is None else (
                                             f"{r['ratio']:.6f}", f"{r['interval'][0]:.6f}", f"{r['interval'][1]:.6f}",
                                             r["clusters"])), label])
    return buffer.getvalue()


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "question": plan["question"], "guards_pass": compute["guards_pass"],
            "dispositions": {k: v["token"] for k, v in compute["dispositions"].items()},
            "decision": {k: compute["decision"][k] for k in ("recipe", "budget", "interference_problem", "action",
                                                             "reading")},
            "triggers": {k: compute["decision"][k]["fired"] for k in ("T1", "T2", "T3")},
            "claim_boundary": plan["claim_boundary"], "engine_seconds": 0}


def rendered(plan, compute):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute)}


def figures(output, compute):
    """Training-dynamics figures (derived from compute.json; not byte-compared by --validate)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    folder = Path(output) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    curves = compute["dynamics"]["validation"]
    styles = (("J", "full", "joint", "C0", "-"), ("D", "request", "per-Δ (req)", "C1", "-"),
              ("D", "full", "per-Δ (full)", "C1", "--"), ("S", "request", "specialist (req)", "C2", "-"),
              ("S", "full", "specialist (full)", "C2", "--"))
    for recipe in RECIPES:
        for term, title in (("local", "validation #77 local loss"), ("long225", "validation long-horizon loss (225)")):
            fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
            for ax, request in zip(axes.flat, REQUESTS):
                for kind, budget, label, color, style in styles:
                    arm = {"J": "J", "D": delta_arm(request), "S": specialist_arm(request)}[kind]
                    curve = curves.get(f"{unit_key(recipe, arm, budget)}:{request}")
                    if not curve:
                        continue
                    xs = [p["step"] for p in curve if p[term] is not None and p["update"] > 0]
                    ys = [p[term] for p in curve if p[term] is not None and p["update"] > 0]
                    ax.plot(xs, ys, style, color=color, label=label, linewidth=1.2)
                ax.set_yscale("log")
                ax.set_title(request, fontsize=10)
            axes[0, 0].legend(fontsize=8)
            for ax in axes[-1]:
                ax.set_xlabel("joint-schedule step")
            fig.suptitle(f"#117 {recipe}: {title} on the left-out lineage (mean over LOLO members)")
            fig.tight_layout()
            fig.savefig(folder / f"validation-{term}-{recipe}.png", dpi=110, metadata={"Software": None})
            plt.close(fig)
    for recipe, block in compute["dynamics"]["gradients"].items():
        panels = [("all", block["matrix_all"]), *block["matrix_phases"].items()]
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.4))
        fig.subplots_adjust(wspace=0.55, left=0.05, right=0.9, bottom=0.2)
        for ax, (name, matrix) in zip(axes, panels):
            if matrix is None:
                continue
            values = np.array([[np.nan if v is None else v for v in row] for row in matrix])
            image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(9), [short(r) for r in REQUESTS], rotation=90, fontsize=7)
            ax.set_yticks(range(9), [short(r) for r in REQUESTS], fontsize=7)
            ax.set_title(f"steps {name}", fontsize=10)
            for i in range(9):
                for j in range(9):
                    if np.isfinite(values[i, j]):
                        ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=list(axes), shrink=0.8)
        fig.suptitle(f"#117 {recipe} joint: cosine of per-request trunk gradients")
        fig.savefig(folder / f"gradient-cosine-{recipe}.png", dpi=110, metadata={"Software": None})
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = [int(k) for k in block["norm_mean_per_probe"] if k != "initial"]
        for i, request in enumerate(REQUESTS):
            ys = [block["norm_mean_per_probe"][str(s)][i] for s in steps]
            ax.plot(steps, [np.nan if y is None else y for y in ys], label=short(request),
                    color=f"C{DELTAS.index(request_delta(request))}",
                    linestyle=("-", "--", ":")[ALPHAS.index(request_alpha(request))])
        ax.set_yscale("log")
        ax.set_xlabel("joint-schedule step")
        ax.set_ylabel("pre-clip gradient norm (probe batch)")
        ax.legend(fontsize=7, ncol=3)
        ax.set_title(f"#117 {recipe} joint: per-request gradient norm before clipping")
        fig.tight_layout()
        fig.savefig(folder / f"gradient-norm-{recipe}.png", dpi=110, metadata={"Software": None})
        plt.close(fig)


def spot_checks(output):
    """Re-score one LOLO record (the Gate-B recipe's joint) and one held-out record here."""
    out = {}
    fold = fit_lineages()[0]
    scoring = Scoring("fit")
    record = read_record(lolo_path(output, GATE_B_RECIPE, "J", "full", fold, SEEDS[0]), SCHEMA_LOLO)
    model = load_member(output, GATE_B_RECIPE, "J", "full", fold, SEEDS[0])
    if model is not None and PRIMARY in record["inventories"]:
        fresh = endpoint_block(model, REQUESTS, scoring.anchors[fold],
                               scoring.universe[PRIMARY][fold]["inventory"], scoring.costs)
        out["lolo"] = v1._replicated(record["inventories"][PRIMARY], fresh)
    held = Scoring("heldout")
    member = sorted(m for m in held.universe[PRIMARY] if m in HELD_OUT)[0]
    record = read_record(heldout_path(output, GATE_B_RECIPE, specialist_arm(GATE_B_UNIT_REQUEST), "request", SEEDS[0]),
                         SCHEMA_HELDOUT)
    model = load_member(output, GATE_B_RECIPE, specialist_arm(GATE_B_UNIT_REQUEST), "request", "all", SEEDS[0])
    if model is not None:
        fresh = endpoint_block(model, [GATE_B_UNIT_REQUEST], held.anchors[member],
                               held.universe[PRIMARY][member]["inventory"], held.costs)
        out["heldout"] = v1._replicated(record["inventories"][PRIMARY][member], fresh)
    return out


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    with harness.deterministic_scoring():
        spot = spot_checks(output)
    compute = compute_tables(output, plan, spot)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    figures(output, compute)
    log(f"published: decision {compute['decision']['action']}; dispositions "
        f"{ {k: v['token'] for k, v in compute['dispositions'].items()} }; guards pass {compute['guards_pass']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    with harness.deterministic_scoring():
        spot = spot_checks(output)
    fresh = compute_tables(output, plan, spot)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    if not fresh["guards"]["G3"]["pass"]:
        raise ValueError("deterministic re-scoring does not replicate the retained records")
    log(f"validation passed: every table re-derived and byte-compared; spot re-scores replicate "
        f"({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    log(f"  recipes {list(RECIPES)}; arms {list(ARMS)}; units {len(UNITS)}; members per unit "
        f"{len(ensemble_members())}")
    for arm in ARMS:
        log(f"  {arm}: requests {arm_requests(arm)}; updates "
            f"{ {b: len(schedule_of(arm, b)) for b in arm_budgets(arm)} }")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "score-lolo": score_lolo,
         "score-heldout": score_heldout, "publish": publish, "validate": validate}
SHARDED = {"train", "score-lolo", "score-heldout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    try:
        if mode in SHARDED:
            return MODES[mode](args.output, args.workers, args.worker)
        return MODES[mode](args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
