"""Issue-117 item 5: interference between requests on CLEVRER and Physion-Dominoes.

Binding runner module: scripts/run_request_interference_benchmarks.py
Exact validation command: python -u -m scripts.run_request_interference_benchmarks --validate

Items 1-2 of #117 repeated on the two out-of-family benchmarks, with the existing
annotation-derived carriers of #79 (CLEVRER, 12 validation scenes) and #84
(Physion-Dominoes, 12 trials of the release prefix). Neither has a macro level, so the
shared predictor serves six requests: Delta in {1, 5, 15} x description in {continuous,
micro}, one request per update, round-robin (the #79 / #84 hybrid schedule, step % 6).

1. Specialists. Per benchmark and recipe (the #77 recipe B, and LC: the #112 v3
   long-horizon recipe adapted to the 120-frame endpoint, without the ranking term), it
   trains the joint six-request model (J), three per-Delta models (D1, D5, D15) and six
   single-request specialists (S-*). Every arm is the CNNHybridPredictor at 1 837 690
   parameters. Every non-joint arm runs at two budgets: the joint model's per-request
   update count (exactly the joint run's updates of its requests, each at its joint step)
   and the joint model's full update count (9000). Each (benchmark, recipe, arm, budget)
   has 13 folds (each of the 12 scenes left out, plus 'all') x 3 seeds.
2. Asymmetry. Joint / specialist rollout-error ratios by Delta and by description.

Scoring: fixed-pair recursive rollout from the true carrier at contexts 0-3 to the #79 /
#84 endpoints 15, 30, 60 and 120 frames; position MSE (#79 field_errors). Fold-L members
are scored on scene L (leave-one-scene-out, EXPLORATORY); fold-'all' members on all 12
scenes (in-sample, the #79 / #84 and ICLR tab:res:rollout setting, DESCRIPTIVE).
Ranking AUC is unsupported: neither benchmark has an engine-verdicted candidate
inventory. Zero engine seconds; intervals are scene-clustered percentile bootstraps
(10000 draws, PCG64 7201).

Modes: --dry-run --smoke --prepare --train --score --publish --validate. Training and
scoring take --workers N --worker K (one process per GPU, set CUDA_VISIBLE_DEVICES).
Scoring and --validate run under the #111 deterministic_scoring() policy.
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
from types import SimpleNamespace

import numpy as np
import torch

from scripts import run_clevrer_boundary_replication as c79
from scripts import run_launch_power_probe as p94
from scripts import run_lookahead_control as wlc
from scripts import run_third_family_boundary_replication as p84
from world_model.model import Abstraction
from world_model.planning import scoring_harness as harness
from world_model.training import clevrer_boundary as clevrer
from world_model.training import decision_dynamics as dd
from world_model.training import request_interference as ri
from world_model.training import stable_decision_dynamics as sd
from world_model.training.cnn_hybrid import PAIRS, CNNHybridPredictor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-117-item5-benchmarks-v1"
IDENTITY = "issue-117-item5-benchmarks-v1"
SCHEMA_PLAN = "issue_117_item5_plan_v1"
SCHEMA_SMOKE = "issue_117_item5_smoke_v1"
SCHEMA_PREDICTOR = "issue_117_item5_predictor_v1"
SCHEMA_UNIT = "issue_117_item5_training_unit_v1"
SCHEMA_RECORD = "issue_117_item5_record_v1"
SCHEMA_LEDGER = "issue_117_item5_ledger_v1"
SCHEMA_COMPUTE = "issue_117_item5_compute_v1"
SCHEMA_REPORT = "issue_117_item5_report_v1"
SCHEMA_MANIFEST = "issue_117_item5_records_manifest_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_request_interference_benchmarks --validate"
RUNNER = "scripts/run_request_interference_benchmarks.py"

DEVICE = "cuda"
SEEDS = (20260908, 20260909, 20260910)
STEPS = ri.STEPS
PAIRS6 = tuple(p for p in PAIRS if p.abstraction != Abstraction.MACRO)   # the #79 / #84 cycle order
CYCLE = len(PAIRS6)
REQUESTS = tuple(f"F-{p.delta}-{p.abstraction.value}" for p in PAIRS6)
DELTAS = (1, 5, 15)
ALPHAS = ("continuous", "micro")
TIMES = tuple(clevrer.ENDPOINTS)            # 15, 30, 60, 120 (#79 and #84 share the grid)
ENDPOINT = TIMES[-1]
CONTEXTS = tuple(clevrer.EVAL_CONTEXTS)     # 0, 1, 2, 3
METRIC = "position_mse"
STOP_TOKEN = "readiness_or_precision_insufficient"
CLAIM = 1.5
MIN_CLUSTERS = 3
NONFINITE_SHARE_MAX = 0.10
PATIENCE = 10
MEMORY_BUDGET_GB = 26.0
MAX_GROUP = 39
SMOKE_SIZES = (13, 39)
SMOKE_WARMUP = 6
SMOKE_TIMED = 30
SMOKE_EQUIVALENCE_STEPS = 12
EQUIVALENCE_TOLERANCE = 1e-5
SCORER_TOLERANCE = 1e-3
PROGRESS_EVERY = 1000
LOG_EVERY = 1000
WALL_LIMIT_HOURS = 48.0
GPUS = 4

BENCHMARKS = {
    "clevrer": {"label": "CLEVRER", "issue": 79, "runner": c79, "source": c79.OUTPUT, "data": "clevrer-data.json",
                "shard_schema": "issue_79_clevrer_scene_v1", "module": "world_model/training/clevrer_boundary.py",
                "iclr": {"F-1-continuous": {"hybrid": 2.98, "continuous_only": 13.5},
                         "F-15-continuous": {"hybrid": 0.0170, "continuous_only": 0.0116}}},
    "physion": {"label": "Physion-Dominoes", "issue": 84, "runner": p84, "source": p84.OUTPUT,
                "data": "physion-data.json", "shard_schema": "issue_84_physion_scene_v1",
                "module": "world_model/training/third_family_boundary.py",
                "iclr": {"F-1-continuous": {"hybrid": 167.4, "continuous_only": 182.6},
                         "F-15-continuous": {"hybrid": 0.00128, "continuous_only": 0.00114}}},
}
TRAINING = {"steps": STEPS, "batch_size": 64, "learning_rate": 0.0001, "weight_decay": 0.0001, "grad_clip": 1.0}
LONG = {"batch_size": 64, "first": 30, "last": ENDPOINT, "increment": 15, "every": 900, "weight": 1.0,
        "starts_per_scene": 8, "truncation": 45, "seed_offset": 112001,
        "starts": ("8 evenly spread starts s_i = i * L // 7 over [0, L], L = n - 1 - 120 (n: the scene's clip "
                   "frames), so every target s + j * Delta (j * Delta <= 120) is a real frame; CLEVRER (n = 128): "
                   "s = 0..7; Physion-Dominoes: n = 151..242"),
        "targets": "the annotation carrier at every visited position s + j * Delta, j = 1 .. horizon // Delta",
        "adaptation": ("#112 LONG: starts [0, 15, 30, 45, 60] with s + 225 <= 285 (shot length), curriculum 60 -> "
                       "225 by +15 every 500 updates (cap at update 5500 of 9000), weight 1, 64 rows; here the "
                       "endpoint is 120 frames: curriculum 30 -> 120 by +15 every 900 updates (cap at update 5400 "
                       "of 9000, the same 0.6 of training), every horizon a multiple of 15 (so of every Delta); the "
                       "v3 truncation stays 45 transitions (a gradient-path length, not a frame count)")}
RECIPES = {
    "B": {"definition": ("#77 recipe, as #79 / #84 ran it: four local transitions of the requested pair, their "
                         "recursion and the carrier bound on 61-frame windows; micro requests add 0.1 x the "
                         "symbolic losses; 9000 updates, batch 64, AdamW 1e-4 / 1e-4, clip 1; a nonfinite update is "
                         "skipped, retirement after 10 consecutive skips"),
          "long": False},
    "LC": {"definition": ("#112 v3 LC adapted to the 120-frame endpoint: B + long-horizon recursive loss (weight "
                          "1, curriculum 30 -> 120), truncated back-propagation every 45 transitions, skip-on-"
                          "nonfinite (retire after 10 consecutive skips). The decision-contrastive ranking term is "
                          "dropped: it needs engine-verdicted candidate inventories, which neither benchmark has"),
           "long": True},
}
ARMS = {"J": tuple(range(CYCLE)),
        **{f"D{d}": tuple(i for i, p in enumerate(PAIRS6) if p.delta == d) for d in DELTAS},
        **{"S-" + r[2:]: (i,) for i, r in enumerate(REQUESTS)}}
ARM_DEFINITIONS = {"J": "joint: all six requests, round-robin (the #79 / #84 hybrid predictor)",
                   **{f"D{d}": f"per-Delta: the two Delta = {d} requests, shared over the descriptions" for d in DELTAS},
                   **{"S-" + r[2:]: f"specialist: request {r} only" for r in REQUESTS}}
BUDGET_DEFINITIONS = {
    "request": ("the joint model's per-request update count: exactly the joint run's updates of the arm's requests "
                "(1500 per request), each at the joint step that sets the LC horizon curriculum"),
    "full": "the joint model's full update count (9000), cycling the arm's requests; the curriculum follows the arm's own update count",
}
COMPARISONS = (("J", "S"), ("J", "D"), ("D", "S"))
GROUPS = {"all": list(REQUESTS),
          **{f"delta_{d}": [r for r in REQUESTS if int(r.split("-")[1]) == d] for d in DELTAS},
          **{f"alpha_{a}": [r for r in REQUESTS if r.split("-")[2] == a] for a in ALPHAS}}
RANKING = {"token": "unsupported",
           "reason": ("neither CLEVRER nor Physion-Dominoes has an engine-verdicted candidate inventory (a set of "
                      "alternative actions from one state with executed outcomes): CLEVRER ships passive videos "
                      "with annotations, the Physion-Dominoes prefix ships passive trials; the #79 / #84 carriers "
                      "hold the null action only, so no per-inventory ranking AUC can be computed")}

read_json = wlc.read_json
write_json = wlc.write_json
json_text = wlc.json_text
sha256_of = wlc.sha256_of
bootstrap = p94.bootstrap


def log(message):
    print(f"[issue-117-item5] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def num(value, digits=3):
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return format(value, f".{digits}f")


# ---------------------------------------------------------------------------
# units (no statistic)
# ---------------------------------------------------------------------------

def arm_budgets(arm):
    """J's per-request budget is its full budget (1500 updates per request of six)."""
    return ("full",) if arm == "J" else ri.BUDGETS


UNITS = tuple((bench, recipe, arm, budget) for bench in BENCHMARKS for recipe in RECIPES for arm in ARMS
              for budget in arm_budgets(arm))


def unit_key(bench, recipe, arm, budget):
    return f"{bench}/{recipe}/{arm}/{budget}"


def unit_name(bench, recipe, arm, budget):
    return f"{bench}--{recipe}--{arm}--{budget}"


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
    return ri.schedule(ARMS[arm], budget, steps=STEPS, cycle=CYCLE)


def horizon_at(g):
    return dd.horizon_at(g, first=LONG["first"], last=LONG["last"], increment=LONG["increment"], every=LONG["every"])


@functools.lru_cache(maxsize=None)
def source_plan(bench):
    return read_json(BENCHMARKS[bench]["source"] / "plan.json")


def scenes(bench):
    return [str(s["scene_index"]) for s in source_plan(bench)["membership"]["scenes"]]


def folds(bench):
    return (*scenes(bench), "all")


def members_of(bench):
    """Frozen member order of one unit: folds x seeds."""
    return [(fold, seed) for fold in folds(bench) for seed in SEEDS]


def member_groups(bench, size):
    members = members_of(bench)
    groups = math.ceil(len(members) / size)
    size = math.ceil(len(members) / groups)
    return [members[i:i + size] for i in range(0, len(members), size)]


def long_starts(frames):
    last = frames - 1 - ENDPOINT
    if last < 0:
        raise ValueError(f"a {frames}-frame clip cannot host a {ENDPOINT}-frame long window")
    count = LONG["starts_per_scene"]
    return tuple(sorted({i * last // (count - 1) for i in range(count)}))


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def unit_path(output, bench, recipe, arm, budget, index):
    return Path(output) / "training" / f"{unit_name(bench, recipe, arm, budget)}--group{index}.json"


def updates_path(output, bench, recipe, arm, budget, index):
    return Path(output) / "updates" / f"{unit_name(bench, recipe, arm, budget)}--group{index}.pt"


def progress_path(output, bench, recipe, arm, budget, index):
    return Path(output) / "progress" / f"{unit_name(bench, recipe, arm, budget)}--group{index}.pt"


def claim_path(output, bench, recipe, arm, budget, index):
    return Path(output) / "claims" / f"{unit_name(bench, recipe, arm, budget)}--group{index}.claim"


def member_path(output, bench, recipe, arm, budget, fold, seed):
    return Path(output) / "models" / bench / recipe / arm / budget / f"fold-{fold}" / f"seed-{seed}" / "predictor.pt"


def record_path(output, bench, recipe, arm, budget, fold, seed):
    return Path(output) / "records" / f"{unit_name(bench, recipe, arm, budget)}--{fold}--seed{seed}.json"


@functools.lru_cache(maxsize=None)
def _record_json(path, modified):
    return read_json(path)


def read_record(path, schema):
    record = _record_json(str(path), Path(path).stat().st_mtime_ns)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


# ---------------------------------------------------------------------------
# data (the #79 / #84 frozen shards, read-only)
# ---------------------------------------------------------------------------

def shard_paths(bench):
    source = BENCHMARKS[bench]["source"]
    return [source / "shards" / f"scene-{scene}.pt" for scene in scenes(bench)]


class Data:
    """GPU-resident #79 / #84 training windows, clips and long windows, with per-fold pools."""

    WINDOW_KEYS = ("z", "action", "length", "relations", "relations_mask")

    def __init__(self, bench):
        spec, plan = BENCHMARKS[bench], source_plan(bench)
        self.bench = bench
        self.scenes = scenes(bench)
        windows, clips = [], []
        for path in shard_paths(bench):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if shard["schema"] != spec["shard_schema"] or shard["identity"] != plan["representation"]["identity"]:
                raise ValueError(f"{path} source/representation differs from the frozen #{spec['issue']} plan")
            windows.append(shard["windows"])
            clips.append(shard["clip"])
        self.window = {k: torch.cat([w[k] for w in windows]).to(DEVICE) for k in self.WINDOW_KEYS}
        self.window_scenes = [scene for scene, w in zip(self.scenes, windows) for _ in range(len(w["z"]))]
        self.clips = {scene: clip for scene, clip in zip(self.scenes, clips)}
        lengths = [len(clip) for clip in clips]
        self.frames = dict(zip(self.scenes, lengths))
        self.carriers = torch.cat(clips).to(DEVICE)
        self.offsets = torch.tensor(np.cumsum([0] + lengths[:-1]).tolist(), dtype=torch.long, device=DEVICE)
        self.lasts = torch.tensor([n - 1 for n in lengths], dtype=torch.long, device=DEVICE)
        long = [(i, start) for i, n in enumerate(lengths) for start in long_starts(n)]
        self.long = torch.tensor(long, dtype=torch.long, device=DEVICE)
        self.long_scenes = [self.scenes[i] for i, _ in long]

    def pools(self, fold):
        if fold != "all" and fold not in self.scenes:
            raise ValueError(f"unknown fold {fold}")
        pools = {key: torch.tensor([i for i, s in enumerate(names) if s != fold], dtype=torch.long)
                 for key, names in (("window", self.window_scenes), ("long", self.long_scenes))}
        for key, names in (("window", self.window_scenes), ("long", self.long_scenes)):
            if any(names[i] == fold for i in pools[key].tolist()):
                raise ValueError(f"fold {fold} pool holds its own scene")
        return pools

    def pool_scenes(self, pool):
        return sorted({self.window_scenes[i] for i in pool["window"].tolist()}
                      | {self.long_scenes[i] for i in pool["long"].tolist()}, key=self.scenes.index)

    @staticmethod
    def _draw(pools, key, generators, count):
        return torch.stack([pool[key][torch.randint(len(pool[key]), (count,), generator=g)]
                            for pool, g in zip(pools, generators)]).to(DEVICE)

    def windows(self, pools, generators, count, symbolic):
        index = self._draw(pools, "window", generators, count)
        keys = self.WINDOW_KEYS if symbolic else self.WINDOW_KEYS[:3]
        return {k: self.window[k][index] for k in keys}

    def long_batch(self, pools, generators, count, horizon, delta):
        index = self._draw(pools, "long", generators, count)
        scene, start = self.long[index, 0], self.long[index, 1]
        offset, last = self.offsets[scene], self.lasts[scene]
        steps = torch.arange(1, horizon // delta + 1, device=DEVICE) * delta
        positions = start[..., None] + steps
        if bool((positions > last[..., None]).any()):
            raise ValueError("a long-window target leaves its clip")
        return {"start": self.carriers[offset + start], "action": torch.zeros(index.shape + (5,), device=DEVICE),
                "targets": self.carriers[offset[..., None] + positions]}


# ---------------------------------------------------------------------------
# training (its own invocation; torch.compile)
# ---------------------------------------------------------------------------

def make_ensemble(models):
    class Skipping(sd.SkippingEnsemble):
        patience = PATIENCE
    return Skipping(models, learning_rate=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"],
                    grad_clip=TRAINING["grad_clip"], compile=True)


LOG_KEYS = ("loss", "local", "long", "grad_norm")
FLAG_KEYS = ("loss_finite", "norm_finite", "active")


def train_unit(output, bench, recipe, arm, budget, index, members, data, *, steps=None, save=True,
               smoke_full=False, timing_after=None, step_times=False):
    """Train one group of members of one (benchmark, recipe, arm, budget) unit (resumable)."""
    plan = schedule_of(arm, budget)
    total = len(plan) if steps is None else steps
    models = []
    for _, seed in members:
        torch.manual_seed(seed)
        models.append(CNNHybridPredictor().to(DEVICE))
    ensemble = make_ensemble(models)
    del models
    size = ensemble.size
    pools = [data.pools(fold) for fold, _ in members]
    window_generators = [torch.Generator().manual_seed(seed) for _, seed in members]
    long_generators = [torch.Generator().manual_seed(seed + LONG["seed_offset"]) for _, seed in members]
    spec = RECIPES[recipe]
    logs = {k: torch.full((total, size), float("nan"), device=DEVICE) for k in LOG_KEYS}
    flags = {k: torch.zeros((total, size), dtype=torch.bool, device=DEVICE) for k in FLAG_KEYS}
    counts = {request: 0 for request in arm_requests(arm)}
    first, gpu = 0, 0.0
    progress = progress_path(output, bench, recipe, arm, budget, index)
    if save and progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        ensemble.load_state(saved["ensemble"])
        for generators, states in ((window_generators, saved["window"]), (long_generators, saved["long"])):
            for generator, state in zip(generators, states):
                generator.set_state(state)
        first, gpu, counts = saved["step"], saved["gpu_seconds"], saved["counts"]
        for k in LOG_KEYS:
            logs[k][:first] = saved["logs"][k][:first].to(DEVICE)
        for k in FLAG_KEYS:
            flags[k][:first] = saved["flags"][k][:first].to(DEVICE)
        log(f"resumed {unit_name(bench, recipe, arm, budget)}/group{index} at update {first}")
    torch.cuda.synchronize()
    began = time.monotonic()
    segment_began, timed_began, times, horizon = began, None, [], None
    for k in range(first, total):
        if k == timing_after:
            torch.cuda.synchronize()
            timed_began = time.monotonic()
        if step_times:
            torch.cuda.synchronize()
            update_began = time.monotonic()
        pair_index, g = plan[k]
        pair = PAIRS6[pair_index]
        batch = data.windows(pools, window_generators, TRAINING["batch_size"],
                             pair.abstraction != Abstraction.CONTINUOUS)
        local = dd.local_loss(ensemble, batch, pair)
        loss, long_term = local, None
        if spec["long"]:
            horizon = ENDPOINT if smoke_full else horizon_at(g)
            long_batch = data.long_batch(pools, long_generators, LONG["batch_size"], horizon, pair.delta)
            long_term, _ = sd.truncated_long_and_rank_loss(
                ensemble, pair, long=long_batch, rank=None, horizon=horizon, endpoint=ENDPOINT, presence=0,
                center=(0, 0), temperature=1.0, truncation=LONG["truncation"])
            loss = local + LONG["weight"] * long_term
        report = ensemble.update(loss)
        ensemble.mark_retired_step(k, report["newly_retired"])
        logs["loss"][k] = loss.detach()
        logs["local"][k] = local.detach()
        if long_term is not None:
            logs["long"][k] = long_term.detach()
        logs["grad_norm"][k] = report["grad_norm"]
        flags["loss_finite"][k] = torch.isfinite(loss.detach())
        flags["norm_finite"][k] = torch.isfinite(report["grad_norm"])
        flags["active"][k] = report["active"]
        counts[REQUESTS[pair_index]] += 1
        if step_times:
            torch.cuda.synchronize()
            times.append({"pair": pair_index, "seconds": time.monotonic() - update_began})
        if (k + 1) % LOG_EVERY == 0 or k + 1 == total:
            torch.cuda.synchronize()
            now = time.monotonic()
            alive = ~ensemble.retired
            log(f"{unit_name(bench, recipe, arm, budget)}/group{index} update {k + 1}/{total} (step {g}): horizon "
                f"{horizon}, mean loss {float(loss.detach()[alive].mean()) if alive.any() else 'n/a'}, retired "
                f"{int(ensemble.retired.sum())}; {now - began:.0f}s; eta {(now - began) / (k + 1 - first) * (total - k - 1):.0f}s")
        if save and (k + 1) % PROGRESS_EVERY == 0 and k + 1 < total:
            torch.cuda.synchronize()
            now = time.monotonic()
            gpu += now - segment_began
            segment_began = now
            atomic_torch(progress, {"ensemble": ensemble.state(), "step": k + 1, "gpu_seconds": gpu, "counts": counts,
                                    "logs": {key: v[:k + 1].cpu() for key, v in logs.items()},
                                    "flags": {key: v[:k + 1].cpu() for key, v in flags.items()},
                                    "window": [g_.get_state() for g_ in window_generators],
                                    "long": [g_.get_state() for g_ in long_generators]})
    torch.cuda.synchronize()
    gpu += time.monotonic() - segment_began
    timed = None if timed_began is None else (time.monotonic() - timed_began) / (total - timing_after)
    parameters = sum(v[0].numel() for v in ensemble.params.values())
    cpu_logs = {k: v.cpu() for k, v in logs.items()}
    cpu_flags = {k: v.cpu() for k, v in flags.items()}
    skipped = ensemble.skipped.cpu().tolist()
    report = {"schema": SCHEMA_UNIT, "plan_identity": IDENTITY, "benchmark": bench, "recipe": recipe, "arm": arm,
              "budget": budget, "group": index, "requests": arm_requests(arm), "updates": total,
              "update_counts": counts, "trained_at": utc_now(),
              "members": [{"fold": fold, "seed": seed, "complete": ensemble.retired_at[i] is None,
                           "retired_at_update": ensemble.retired_at[i],
                           "retired_request": (REQUESTS[plan[ensemble.retired_at[i]][0]]
                                               if ensemble.retired_at[i] is not None else None),
                           "skipped": int(skipped[i]), "pool_scenes": data.pool_scenes(pools[i])}
                          for i, (fold, seed) in enumerate(members)],
              "nonfinite_events": nonfinite_events(cpu_logs, cpu_flags, plan, members, ensemble.retired_at),
              "final": {k: [float(v) if math.isfinite(v) else None for v in cpu_logs[k][total - 1].double().tolist()]
                        for k in ("loss", "local", "long")},
              "gpu_seconds": gpu, "parameters_per_member": parameters,
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
              "timed_seconds_per_update": timed, "update_times": times or None}
    if save:
        for i, (fold, seed) in enumerate(members):
            atomic_torch(member_path(output, bench, recipe, arm, budget, fold, seed),
                         {"schema": SCHEMA_PREDICTOR, "plan_identity": IDENTITY, "benchmark": bench, "recipe": recipe,
                          "arm": arm, "budget": budget, "fold": fold, "seed": seed, "updates": total,
                          "complete": ensemble.retired_at[i] is None, "retired_at": ensemble.retired_at[i],
                          "skipped": int(skipped[i]), "parameters": parameters, "model": ensemble.member_state(i)})
        atomic_torch(updates_path(output, bench, recipe, arm, budget, index),
                     {"plan_identity": IDENTITY, "members": members, "schedule": plan[:total],
                      "logs": cpu_logs, "flags": cpu_flags})
        write_json(unit_path(output, bench, recipe, arm, budget, index), report)
        progress.unlink(missing_ok=True)
    return report, ensemble


def nonfinite_events(logs, flags, plan, members, retired_at):
    """Per member: the updates (before retirement) whose loss or pre-clip gradient norm was nonfinite, with the
    request each trained."""
    events = []
    total = logs["loss"].shape[0]
    for i, (fold, seed) in enumerate(members):
        last = total if retired_at[i] is None else retired_at[i] + 1
        bad = (~(flags["loss_finite"][:last, i] & flags["norm_finite"][:last, i])).nonzero(as_tuple=True)[0].tolist()
        if bad:
            events.append({"fold": fold, "seed": seed, "count": len(bad), "retired": retired_at[i] is not None,
                           "first_request": REQUESTS[plan[bad[0]][0]],
                           "events": [{"update": k, "step": plan[k][1], "request": REQUESTS[plan[k][0]],
                                       "loss_finite": bool(flags["loss_finite"][k, i]),
                                       "grad_norm_finite": bool(flags["norm_finite"][k, i])} for k in bad[:20]]})
    return events


# ---------------------------------------------------------------------------
# work distribution, ledger
# ---------------------------------------------------------------------------

def claim(path):
    """Exclusive claim of one unit group by this process; a claim of a dead process is stale."""
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
# scoring (inside deterministic_scoring, in processes that did not train)
# ---------------------------------------------------------------------------

def load_member(output, bench, recipe, arm, budget, fold, seed):
    """The trained predictor, or None for a retired (typed-failure) member."""
    payload = torch.load(member_path(output, bench, recipe, arm, budget, fold, seed), map_location="cpu",
                         weights_only=True)
    if (payload.get("schema") != SCHEMA_PREDICTOR or payload.get("plan_identity") != IDENTITY
            or (payload["benchmark"], payload["recipe"], payload["arm"], payload["budget"], payload["fold"],
                payload["seed"]) != (bench, recipe, arm, budget, fold, seed)
            or payload["updates"] != len(schedule_of(arm, budget))):
        raise ValueError(f"predictor {unit_name(bench, recipe, arm, budget)}/{fold}/{seed} binding differs")
    if not payload["complete"]:
        return None
    model = CNNHybridPredictor()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval()


def unit_errors(prediction, target):
    """#79 field_errors, or None where the recursive carrier or a statistic is nonfinite."""
    if not bool(torch.isfinite(prediction).all()):
        return None
    try:
        return clevrer.field_errors(prediction, target)
    except clevrer.ClevrerBoundaryError:
        return None


@torch.no_grad()
def rollout_errors(model, clips, units, pair):
    """Fixed-pair recursive rollout from the true carrier of every (scene, context) unit, batched; errors at every
    endpoint (one shared initial carrier, predictions feed the next transition, no truth resets)."""
    z = torch.stack([clips[scene][context] for scene, context in units]).to(DEVICE)
    null = torch.zeros((len(units), 5), device=DEVICE)
    out = {scene: [] for scene, _ in units}
    errors = [{} for _ in units]
    steps = 0
    for t in TIMES:
        while steps < t:
            z = model.carrier(z, null, pair)
            steps += pair.delta
        if steps != t:
            raise ValueError("fixed pair does not land exactly on the endpoint")
        rows = z.cpu()
        for row, (scene, context) in enumerate(units):
            errors[row][str(t)] = unit_errors(rows[row], clips[scene][context + t])
    for row, (scene, context) in enumerate(units):
        out[scene].append({"context": context, "errors": errors[row]})
    return out


def score_record(output, data, bench, recipe, arm, budget, fold, seed, policy):
    model = load_member(output, bench, recipe, arm, budget, fold, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy, "scored_at": utc_now(),
            "cell": {"benchmark": bench, "recipe": recipe, "arm": arm, "budget": budget, "fold": fold, "seed": seed},
            "cohort": "insample" if fold == "all" else "lolo", "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": "member_retired_nonfinite", "rollout": None}
    units = [(scene, context) for scene in (data.scenes if fold == "all" else [fold]) for context in CONTEXTS]
    rollout = {request: rollout_errors(model, data.clips, units, PAIRS6[REQUESTS.index(request)])
               for request in arm_requests(arm)}
    return {**base, "typed_failure": None, "rollout": rollout}


# ---------------------------------------------------------------------------
# smoke (pre-freeze: nothing kept, no statistic)
# ---------------------------------------------------------------------------

def reference_train(bench, scratch, seed, steps):
    """The #79 / #84 single-model training loop itself (its train_cell, stop_after) for the hybrid arm."""
    runner = BENCHMARKS[bench]["runner"]
    root = Path(scratch) / f"reference-{bench}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").symlink_to(BENCHMARKS[bench]["source"] / "shards")
    args = SimpleNamespace(output=root, device=DEVICE)
    runner.train_cell(args, source_plan(bench), seed, "hybrid", stop_after=steps)
    saved = torch.load(runner.cell(args, seed, "hybrid") / "predictor-progress.pt", map_location="cpu",
                       weights_only=True)
    return saved["model"]


def published_eval(bench, seed, system, scene, context):
    path = BENCHMARKS[bench]["source"] / "eval" / f"seed-{seed}" / system / f"scene-{scene}-t{context}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def published_model(bench, seed):
    value = torch.load(BENCHMARKS[bench]["source"] / f"seed-{seed}" / "hybrid" / "predictor.pt", map_location="cpu",
                       weights_only=True)
    model = CNNHybridPredictor()
    model.load_state_dict(value["model"], strict=True)
    return model.to(DEVICE).eval()


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    checks, timings, memory = [], {}, {}
    joint = schedule_of("J", "full")
    checks.append({"control": "joint_schedule_is_the_issue_79_84_cycle",
                   "ok": joint == [(k % CYCLE, k) for k in range(STEPS)]
                   and PAIRS6 == c79.PAIRS_CLEVRER == p84.PAIRS_FAMILY})
    counts = {arm: {b: len(schedule_of(arm, b)) for b in arm_budgets(arm)} for arm in ARMS}
    restricted = all(schedule_of(arm, "request") == [(p, g) for p, g in joint if p in ARMS[arm]]
                     for arm in ARMS if arm != "J")
    checks.append({"control": "per_request_budget_is_the_joint_runs_updates_of_the_arm", "update_counts": counts,
                   "ok": restricted and all(counts[a].get("request", 1500 * len(ARMS[a])) == 1500 * len(ARMS[a])
                                            and counts[a]["full"] == STEPS for a in ARMS)})
    for bench in BENCHMARKS:
        frozen = source_plan(bench)["design"]["training"]
        checks.append({"control": f"{bench}_recipe_B_is_the_issue_{BENCHMARKS[bench]['issue']}_recipe",
                       "ok": all(frozen[k] == v for k, v in TRAINING.items()) and frozen["unroll"] == dd.LOCAL_STEPS})
    data = {bench: Data(bench) for bench in BENCHMARKS}
    for bench, d in data.items():
        isolation = all(fold not in d.pool_scenes(d.pools(fold)) for fold in d.scenes) \
            and d.pool_scenes(d.pools("all")) == d.scenes
        checks.append({"control": f"{bench}_fold_pools_exclude_their_scene", "windows": len(d.window_scenes),
                       "long_windows": len(d.long_scenes), "frames": d.frames,
                       "ok": isolation and len(d.window_scenes) == 480 and len(d.long_scenes) == 8 * len(d.scenes)})
    # equivalence: the J arm under B makes exactly the #79 / #84 hybrid training updates (fold 'all')
    for bench, d in data.items():
        members = [("all", SEEDS[0]), (d.scenes[0], SEEDS[1])]
        _, ensemble = train_unit(scratch, bench, "B", "J", "full", 0, members, d, steps=SMOKE_EQUIVALENCE_STEPS,
                                 save=False)
        reference = reference_train(bench, scratch, SEEDS[0], SMOKE_EQUIVALENCE_STEPS)
        ours = ensemble.member_state(0)
        delta = max(float((ours[k] - reference[k]).abs().max()) for k in reference)
        checks.append({"control": f"{bench}_B_joint_equals_issue_{BENCHMARKS[bench]['issue']}_hybrid_training",
                       "updates": SMOKE_EQUIVALENCE_STEPS, "max_abs_parameter_delta": delta,
                       "ok": delta <= EQUIVALENCE_TOLERANCE and set(ours) == set(reference)})
        del ensemble
        torch.cuda.empty_cache()
    # cost and memory: the joint arm at the most expensive setting (LC at horizon 120 throughout)
    for recipe in RECIPES:
        for size in SMOKE_SIZES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            members = members_of("clevrer")[:size]
            report, ensemble = train_unit(scratch, "clevrer", recipe, "J", "full", 0, members, data["clevrer"],
                                          steps=SMOKE_WARMUP + SMOKE_TIMED, save=False, smoke_full=True,
                                          timing_after=SMOKE_WARMUP, step_times=True)
            del ensemble
            memory[(recipe, size)] = report["peak_memory_gb"]
            per_pair = {}
            for entry in report["update_times"][SMOKE_WARMUP:]:
                per_pair.setdefault(entry["pair"], []).append(entry["seconds"])
            timings[f"{recipe}_m{size}"] = {str(p): float(np.median(v)) for p, v in sorted(per_pair.items())}
            checks.append({"control": f"{recipe}_m{size}_trains_finite",
                           "retired": sum(not m["complete"] for m in report["members"]),
                           "ok": all(m["complete"] for m in report["members"])})
    # scoring mechanics: the batched scorer reproduces published #79 / #84 evaluation records
    with harness.deterministic_scoring() as policy:
        for bench, d in data.items():
            model = published_model(bench, SEEDS[0])
            worst, compared = 0.0, 0
            began = time.monotonic()
            for system, pair in (("hybrid_continuous_h1", PAIRS6[0]), ("hybrid_micro_h5", PAIRS6[3]),
                                 ("hybrid_continuous_h15", PAIRS6[4])):
                units = [(d.scenes[0], c) for c in CONTEXTS] + [(d.scenes[-1], c) for c in CONTEXTS]
                fresh = rollout_errors(model, d.clips, units, pair)
                for scene, context in units:
                    stored = published_eval(bench, SEEDS[0], system, scene, context)["result"]["errors"]
                    ours = next(u for u in fresh[scene] if u["context"] == context)["errors"]
                    for t in TIMES:
                        a, b = stored[str(t)]["errors"][METRIC], ours[str(t)][METRIC]
                        worst = max(worst, abs(a - b) / max(abs(a), 1e-12))
                        compared += 1
            timings[f"score_{bench}_seconds_24_rollouts"] = time.monotonic() - began
            checks.append({"control": f"{bench}_scorer_reproduces_issue_{BENCHMARKS[bench]['issue']}_eval_records",
                           "policy": policy, "compared": compared, "max_relative_delta": worst,
                           "ok": worst <= SCORER_TOLERANCE})
    projection = project(timings, memory)
    checks.append({"control": "projected_wall_within_48h_on_4_gpus",
                   "projected_wall_hours": projection["wall_hours_4_gpus"],
                   "ok": projection["wall_hours_4_gpus"] <= WALL_LIMIT_HOURS})
    ok = all(c["ok"] for c in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timings": timings, "memory_gb": {f"{r}_m{s}": v for (r, s), v in memory.items()},
                "projection": projection, "device": torch.cuda.get_device_name(0),
                "reading": "infrastructure assertions only; nothing kept; no statistic of any trained arm", "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected GPU hours {projection['gpu_seconds'] / 3600:.1f}; "
        f"wall on 4 GPUs {projection['wall_hours_4_gpus']:.1f} h; group sizes {projection['group_sizes']}")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


def project(timings, memory):
    """Group size per recipe from the memory slope; GPU seconds per unit from per-pair update seconds at that size
    (LC at horizon 120 throughout: an upper bound). Both benchmarks share every tensor shape."""
    small, large = SMOKE_SIZES
    sizes, out = {}, {}
    for recipe in RECIPES:
        slope = max((memory[(recipe, large)] - memory[(recipe, small)]) / (large - small), 1e-3)
        base = memory[(recipe, small)] - small * slope
        size = max(1, min(MAX_GROUP, int((MEMORY_BUDGET_GB - base) // slope)))
        groups = math.ceil(MAX_GROUP / size)
        sizes[recipe] = math.ceil(MAX_GROUP / groups)
        t1, t2 = timings[f"{recipe}_m{small}"], timings[f"{recipe}_m{large}"]
        per_pair = {int(p): t1[p] + (t2[p] - t1[p]) / (large - small) * (sizes[recipe] - small) for p in t1}
        out[recipe] = {"group_size": sizes[recipe], "groups": groups, "gb_per_member": slope, "base_gb": base,
                       "projected_peak_gb": base + slope * sizes[recipe], "seconds_per_update": per_pair}
    units = {}
    for bench, recipe, arm, budget in UNITS:
        p = out[recipe]
        units[unit_key(bench, recipe, arm, budget)] = p["groups"] * sum(p["seconds_per_update"][q]
                                                                        for q, _ in schedule_of(arm, budget))
    gpu = sum(units.values())
    return {"group_sizes": sizes, "per_recipe": out, "unit_seconds": units, "gpu_seconds": gpu,
            "wall_hours_4_gpus": max(gpu / GPUS, max(units.values())) / 3600}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {}
    for bench, spec in BENCHMARKS.items():
        names[f"issue_{spec['issue']}_plan"] = spec["source"] / "plan.json"
        names[f"issue_{spec['issue']}_data"] = spec["source"] / spec["data"]
        names[f"issue_{spec['issue']}_summary"] = spec["source"] / "summary.json"
        names[f"issue_{spec['issue']}_carrier_module"] = ROOT / spec["module"]
        for path in shard_paths(bench):
            names[f"issue_{spec['issue']}_shard_{path.stem}"] = path
    names.update({"cnn_hybrid": ROOT / "world_model/training/cnn_hybrid.py",
                  "decision_dynamics": ROOT / "world_model/training/decision_dynamics.py",
                  "stable_decision_dynamics": ROOT / "world_model/training/stable_decision_dynamics.py",
                  "request_interference_module": ROOT / "world_model/training/request_interference.py",
                  "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
                  "smoke": Path(output) / "smoke.json"})
    return [{"name": name, "artifact": str(Path(path).resolve().relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence):
    projection = smoke_evidence["projection"]
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 117, "item": 5, "parent_issue": 108,
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "engine_seconds": 0,
        "question": ("#117 items 1-2 on CLEVRER and Physion-Dominoes: what does sharing every weight of the six-"
                     "request hybrid predictor (continuous and micro x Delta in {1, 5, 15}) cost each request in "
                     "recursive rollout error against a matched single-request specialist and a per-Delta model, "
                     "under the #77 recipe and the adapted #112 LC recipe, at the per-request and full budgets, and "
                     "is the cost asymmetric across Delta and description?"),
        "disclosure": ("Known before freeze: the #79 / #84 published joint (hybrid) and continuous-only numbers, the "
                       "ICLR tab:res:rollout ratios quoted in #117 (in-sample, 120 frames), and every NovPhy #117 "
                       "item 1-3 result (joint / specialist <= 1.5 at the per-request budget for B / LC1 / LC3; "
                       "Delta = 1 gains most from sharing under LC). No specialist or per-Delta model has been "
                       "trained or scored on either benchmark."),
        "hardware": ("4x RTX 5090 (one training or scoring process per GPU); the #85 rule names a single RTX 3090. "
                     "The joint arm is retrained here with every other arm, not reused from #79 / #84; their "
                     "published in-sample numbers are reported next to it as a replication reading."),
        "data": {bench: {"source": str(spec["source"].relative_to(ROOT)), "carrier": source_plan(bench)["representation"]["identity"],
                         "scenes": scenes(bench), "windows": "40 x 61-frame windows per scene (#79 / #84 frozen starts), 480 in all",
                         "clip_frames": {s: n for s, n in zip(scenes(bench), smoke_frames(smoke_evidence, bench))}}
                 for bench, spec in BENCHMARKS.items()},
        "requests": list(REQUESTS), "pairs": [list(p.identity) for p in PAIRS6],
        "recipes": RECIPES, "training": {**TRAINING, "long_LC": LONG, "patience": PATIENCE,
                                         "initialization": "torch.manual_seed(seed); CNNHybridPredictor() (the #79 / #84 initialization)",
                                         "data_streams": ("per member: windows drawn by torch.Generator(seed) from the fold pool "
                                                          "(fold 'all': exactly the #79 / #84 draws), long windows by seed + 112001"),
                                         "matched": ("every arm: the CNNHybridPredictor architecture and parameter count "
                                                     "(1 837 690); a request's update touches only the parameters it reads"),
                                         "group_sizes": projection["group_sizes"]},
        "arms": {a: {"requests": arm_requests(a), "definition": ARM_DEFINITIONS[a], "budgets": list(arm_budgets(a))}
                 for a in ARMS},
        "budgets": BUDGET_DEFINITIONS, "units": [unit_key(*u) for u in UNITS], "seeds": list(SEEDS),
        "folds": {"rule": ("leave-one-scene-out: fold s trains on the 11 other scenes of its benchmark; fold 'all' on "
                           "all 12 (the #79 / #84 training set)"),
                  "members_per_unit": len(members_of("clevrer"))},
        "scoring": {"rollout": (f"fixed-pair recursive rollout from the true carrier at contexts {list(CONTEXTS)}; "
                                f"errors at t = {list(TIMES)} (#79 field_errors; position MSE under the camera mask on "
                                "CLEVRER, every declared object on Physion-Dominoes); batched over units"),
                    "lolo": "fold-s members on scene s (4 units): EXPLORATORY, the primary cohort",
                    "insample": ("fold-'all' members on all 12 scenes (48 units): the #79 / #84 and ICLR "
                                 "tab:res:rollout setting; DESCRIPTIVE"),
                    "nonfinite": "a nonfinite recursive carrier or statistic is recorded as None and counted (G4)",
                    "ranking": RANKING},
        "statistics": {"declared_endpoint": ENDPOINT,
                       "rollout_error": ("per (unit, request, t, scene): mean over contexts, then mean over the seeds "
                                         "with a finite value; mean over scenes, scene-clustered bootstrap"),
                       "rollout_ratio": ("per (benchmark, recipe, budget, request, t): per scene log(first / second) "
                                         "of the seed-mean errors; scene-clustered bootstrap (10 000 draws, PCG64 7201) "
                                         "of the mean log ratio; geometric ratio exp(mean) [exp(low), exp(high)]; "
                                         "per-seed geometric ratios reported beside (the spread over seeds); "
                                         "comparisons J/S, J/D, D/S"),
                       "asymmetry": ("per recipe and budget: the mean log ratio over the requests of each Delta "
                                     "(2) and of each description (3), and over all six; Delta = 1 over Delta = 15 "
                                     "and continuous over micro as ratios of ratios per scene, bootstrapped"),
                       "labels": "leave-one-scene-out: EXPLORATORY; in-sample: DESCRIPTIVE"},
        "decision_rule": {
            "claim_1_5x": (f"'sharing all weights costs no more than {CLAIM}x per-request rollout error at the endpoint "
                           f"(t = {ENDPOINT}) against a matched specialist', per benchmark, recipe and budget, on the "
                           f"leave-one-scene-out cohort: per request, the joint / specialist ratio at t = {ENDPOINT} is "
                           f"'supported' iff its upper bound < {CLAIM}, 'not_supported_by_this_experiment' iff its lower "
                           f"bound > {CLAIM}, otherwise (or with fewer than {MIN_CLUSTERS} scenes) {STOP_TOKEN}. Per "
                           f"(benchmark, recipe, budget): supported iff every request is supported; not_supported iff "
                           f"any request is not_supported; otherwise {STOP_TOKEN}; {STOP_TOKEN} for every token of a "
                           "(benchmark, recipe, budget) whose guard fails, and for every token if G1, G2, G3 or the "
                           "chronology guard fails"),
            "headline": ("the disposition per benchmark and recipe is the per-request-budget token (the negative-"
                         "transfer contrast: the request's same updates with or without the other requests', as in "
                         "the NovPhy #117 reading); the full-budget token is reported beside it"),
            "ranking": f"ranking AUC: {RANKING['token']} ({RANKING['reason']})",
            "multiplicity": "6 requests x 2 recipes x 2 budgets x 2 benchmarks of per-request intervals; nothing is selected",
            "labels": "leave-one-scene-out: EXPLORATORY; in-sample: DESCRIPTIVE",
        },
        "guards": {
            "G1": "no training window or long window of a fold's left-out scene (pool scenes recorded per member)",
            "G2": "every unit trained its full schedule; every member complete or retired as a typed failure",
            "G3": ("every record carries the #111 SCORING_POLICY; --validate re-scores two records (one per "
                   "benchmark) and they equal the retained records"),
            "G4": (f"per (benchmark, recipe, budget): the share of leave-one-scene-out (member, request) cells with no "
                   f"finite t = {ENDPOINT} error (retired member or nonfinite rollout) <= {NONFINITE_SHARE_MAX}"),
            "chronology": "plan frozen before the first update; scoring after every unit trained",
            "caps": "GPU seconds within the cap",
        },
        "caps": {"gpu_seconds": max(2.0 * projection["gpu_seconds"], 12 * 3600.0), "stop": STOP_TOKEN},
        "projection": projection,
        "prohibited": ["changing any recipe, arm, budget, schedule or rule after training starts",
                       "retrying typed failures", "re-scoring a record", "re-opening any prior disposition"],
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("#117 item 5 only: CLEVRER (#79) and Physion-Dominoes (#84) annotation-derived carriers "
                           "(engine features in, no perception), 12 scenes each, continuous and micro requests (no "
                           "macro level); recursive rollout error only; no ranking, planning or closed-loop claim; "
                           "the in-sample cohort repeats the ICLR setting and is DESCRIPTIVE; zero engine seconds"),
    }


def smoke_frames(evidence, bench):
    check = next(c for c in evidence["checks"] if c["control"] == f"{bench}_fold_pools_exclude_their_scene")
    return [check["frames"][s] for s in scenes(bench)]


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-117 item-5 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
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
    log(f"frozen plan published: {len(UNITS)} units x {len(members_of('clevrer'))} members; projected "
        f"{plan['projection']['gpu_seconds'] / 3600:.1f} GPU hours")
    return 0


# ---------------------------------------------------------------------------
# modes: train, score
# ---------------------------------------------------------------------------

def unit_groups(plan):
    out = []
    for bench, recipe, arm, budget in UNITS:
        size = plan["training"]["group_sizes"][recipe]
        for index, members in enumerate(member_groups(bench, size)):
            out.append((bench, recipe, arm, budget, index, members))
    cost = plan["projection"]["unit_seconds"]
    return sorted(out, key=lambda u: -cost[unit_key(*u[:4])])


def train(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if any(output.glob("records/*.json")):
        raise ValueError("records exist; training precedes scoring")
    torch.backends.cudnn.benchmark = True
    data = {}
    for bench, recipe, arm, budget, index, members in unit_groups(plan):
        if unit_path(output, bench, recipe, arm, budget, index).is_file():
            continue
        if not claim(claim_path(output, bench, recipe, arm, budget, index)):
            continue
        if unit_path(output, bench, recipe, arm, budget, index).is_file():
            continue
        if bench not in data:
            data[bench] = Data(bench)
        began = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        report, ensemble = train_unit(output, bench, recipe, arm, budget, index, members, data[bench])
        del ensemble
        torch.cuda.empty_cache()
        claim_path(output, bench, recipe, arm, budget, index).unlink(missing_ok=True)
        retired = sum(not m["complete"] for m in report["members"])
        mark(output, f"train_{unit_name(bench, recipe, arm, budget)}_group{index}", time.monotonic() - began,
             report["gpu_seconds"], members=len(members), retired=retired, worker=worker,
             device=torch.cuda.get_device_name(0))
        log(f"worker {worker}: trained {unit_name(bench, recipe, arm, budget)}/group{index}: "
            f"{report['gpu_seconds']:.0f}s, retired {retired}")
    return 0


def training_complete(output, plan):
    return all(unit_path(output, *u[:5]).is_file() for u in unit_groups(plan))


def score_cells():
    cells = [(bench, recipe, arm, budget, fold, seed) for bench, recipe, arm, budget in UNITS
             for fold, seed in members_of(bench)]
    return sorted(cells, key=lambda c: (-len(ARMS[c[2]]), unit_key(*c[:4]), c[4], c[5]))


def score(output, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    if not training_complete(output, plan):
        raise ValueError("training incomplete; scoring needs every member")
    began = time.monotonic()
    pending = [c for i, c in enumerate(score_cells()) if i % workers == worker and not record_path(output, *c).is_file()]
    data = {}
    with harness.deterministic_scoring() as policy:
        for done, cell in enumerate(pending, 1):
            if cell[0] not in data:
                data[cell[0]] = Data(cell[0])
            write_json(record_path(output, *cell), score_record(output, data[cell[0]], *cell, policy))
            if done % 200 == 0 or done == len(pending):
                log(f"worker {worker}: scored {done}/{len(pending)}; {time.monotonic() - began:.0f}s")
    wall = time.monotonic() - began
    mark(output, f"score_worker{worker}", wall, wall, records=len(pending))
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def scene_seed_value(rollout, request, scene, t):
    """Mean over contexts of the finite position MSE at t; None if no context is finite."""
    values = [u["errors"][str(t)][METRIC] for u in rollout[request][scene]
              if u["errors"][str(t)] is not None and u["errors"][str(t)][METRIC] is not None]
    return float(np.mean(values)) if values else None


def collect(output):
    """per[(bench, cohort, unit, request)] = {scene: {t: [value per seed or None]}}; nonfinite counts."""
    per, missing = {}, {}
    for bench, recipe, arm, budget in UNITS:
        key = unit_key(bench, recipe, arm, budget)
        for request in arm_requests(arm):
            for cohort in ("lolo", "insample"):
                per[(bench, cohort, key, request)] = {}
            missing[(key, request)] = {"cells": 0, "missing": 0}
        for fold in folds(bench):
            records = [read_record(record_path(output, bench, recipe, arm, budget, fold, s), SCHEMA_RECORD)
                       for s in SEEDS]
            cohort = "insample" if fold == "all" else "lolo"
            covered = scenes(bench) if fold == "all" else [fold]
            for request in arm_requests(arm):
                for scene in covered:
                    per[(bench, cohort, key, request)][scene] = {
                        str(t): [None if r["typed_failure"] else scene_seed_value(r["rollout"], request, scene, t)
                                 for r in records] for t in TIMES}
                if cohort == "lolo":
                    for r in records:
                        missing[(key, request)]["cells"] += 1
                        if r["typed_failure"] or scene_seed_value(r["rollout"], request, fold, ENDPOINT) is None:
                            missing[(key, request)]["missing"] += 1
    return per, missing


def scene_value(per, bench, cohort, unit, request, scene, t):
    seeds = per[(bench, cohort, unit, request)].get(scene)
    if seeds is None:
        return None
    values = [v for v in seeds[str(t)] if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


def rollout_block(per, bench, cohort, unit, request, t):
    values, clusters = [], []
    for scene in scenes(bench):
        v = scene_value(per, bench, cohort, unit, request, scene, t)
        if v is not None:
            values.append(v)
            clusters.append(scene)
    return bootstrap(values, clusters)


def _log_ratios(per, bench, cohort, triples, scene, t, seed_index=None):
    logs = []
    for first, second, request in triples:
        if seed_index is None:
            a = scene_value(per, bench, cohort, first, request, scene, t)
            b = scene_value(per, bench, cohort, second, request, scene, t)
        else:
            sa = per[(bench, cohort, first, request)].get(scene)
            sb = per[(bench, cohort, second, request)].get(scene)
            a = None if sa is None else sa[str(t)][seed_index]
            b = None if sb is None else sb[str(t)][seed_index]
        if a is None or b is None or not a > 0 or not b > 0 or not math.isfinite(a) or not math.isfinite(b):
            return None
        logs.append(math.log(a / b))
    return float(np.mean(logs))


def ratio_block(per, bench, cohort, triples, t):
    """triples [(first unit, second unit, request)]: per scene the mean over the triples of log(first / second);
    scene-clustered bootstrap; geometric ratio; per-seed geometric ratios beside it."""
    values, clusters = [], []
    for scene in scenes(bench):
        v = _log_ratios(per, bench, cohort, triples, scene, t)
        if v is not None:
            values.append(v)
            clusters.append(scene)
    block = bootstrap(values, clusters)
    if block is None:
        return None
    per_seed = []
    for index in range(len(SEEDS)):
        logs = [v for v in (_log_ratios(per, bench, cohort, triples, scene, t, index) for scene in scenes(bench))
                if v is not None]
        per_seed.append(math.exp(float(np.mean(logs))) if logs else None)
    return {"ratio": math.exp(block["mean"]), "interval": [math.exp(v) for v in block["interval"]],
            "log_mean": block["mean"], "log_interval": block["interval"], "clusters": block["clusters"],
            "per_seed": per_seed, "label": block["label"]}


def log_difference_block(per, bench, cohort, first_triples, second_triples, t):
    """Per scene: mean log ratio of the first triples minus that of the second; bootstrap."""
    values, clusters = [], []
    for scene in scenes(bench):
        a = _log_ratios(per, bench, cohort, first_triples, scene, t)
        b = _log_ratios(per, bench, cohort, second_triples, scene, t)
        if a is not None and b is not None:
            values.append(a - b)
            clusters.append(scene)
    block = bootstrap(values, clusters)
    if block is None:
        return None
    return {"ratio_of_ratios": math.exp(block["mean"]), "interval": [math.exp(v) for v in block["interval"]],
            "log_mean": block["mean"], "log_interval": block["interval"], "clusters": block["clusters"]}


def comparison_units(bench, recipe, budget, request):
    return {"J": unit_key(bench, recipe, "J", "full"), "D": unit_key(bench, recipe, delta_arm(request), budget),
            "S": unit_key(bench, recipe, specialist_arm(request), budget)}


def ratio_tables(per, bench, cohort):
    out = {}
    for recipe in RECIPES:
        for budget in ri.BUDGETS:
            block = {}
            for first, second in COMPARISONS:
                triples = {r: [(comparison_units(bench, recipe, budget, r)[first],
                                comparison_units(bench, recipe, budget, r)[second], r)] for r in REQUESTS}
                block[f"{first}/{second}"] = {
                    "per_request": {r: {str(t): ratio_block(per, bench, cohort, triples[r], t) for t in TIMES}
                                    for r in REQUESTS},
                    "groups": {g: ratio_block(per, bench, cohort, [x for r in rs for x in triples[r]], ENDPOINT)
                               for g, rs in GROUPS.items()},
                    "delta1_over_delta15": log_difference_block(
                        per, bench, cohort, [x for r in GROUPS["delta_1"] for x in triples[r]],
                        [x for r in GROUPS["delta_15"] for x in triples[r]], ENDPOINT),
                    "continuous_over_micro": log_difference_block(
                        per, bench, cohort, [x for r in GROUPS["alpha_continuous"] for x in triples[r]],
                        [x for r in GROUPS["alpha_micro"] for x in triples[r]], ENDPOINT)}
            out[f"{recipe}/{budget}"] = block
    return out


def claim_token(block):
    if block is None or block["clusters"] < MIN_CLUSTERS:
        return STOP_TOKEN
    if block["interval"][1] < CLAIM:
        return "supported"
    if block["interval"][0] > CLAIM:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def dispositions(ratios, guards):
    out = {}
    for bench in BENCHMARKS:
        for recipe in RECIPES:
            for budget in ri.BUDGETS:
                key = f"{recipe}/{budget}"
                blocks = {r: ratios[bench]["lolo"][key]["J/S"]["per_request"][r][str(ENDPOINT)] for r in REQUESTS}
                per = {r: claim_token(b) for r, b in blocks.items()}
                tokens = set(per.values())
                if not global_pass(guards) or not guards["G4"]["per_scope"][f"{bench}/{key}"]["pass"]:
                    overall = STOP_TOKEN  # the frozen rule: every token of the failed scope stops
                    per = {r: STOP_TOKEN for r in per}
                elif tokens == {"supported"}:
                    overall = "supported"
                elif "not_supported_by_this_experiment" in tokens:
                    overall = "not_supported_by_this_experiment"
                else:
                    overall = STOP_TOKEN
                uppers = [b["interval"][1] for b in blocks.values() if b is not None]
                out[f"{bench}/{key}"] = {
                    "benchmark": BENCHMARKS[bench]["label"], "recipe": recipe, "budget": budget,
                    "per_request": per, "token": overall, "label": "EXPLORATORY (leave-one-scene-out)",
                    "largest_upper_bound": max(uppers) if uppers else None,
                    "rule": (f"supported iff every request's joint / specialist upper bound at t = {ENDPOINT} < {CLAIM}"),
                    "claim": (f"sharing all weights costs no more than {CLAIM}x per-request rollout error at t = "
                              f"{ENDPOINT} against a matched specialist")}
    return out


def headline(disp):
    return {f"{bench}/{recipe}": {"token": disp[f"{bench}/{recipe}/request"]["token"],
                                  "full_budget_token": disp[f"{bench}/{recipe}/full"]["token"]}
            for bench in BENCHMARKS for recipe in RECIPES}


def unit_reports(output, plan):
    reports = {}
    for bench, recipe, arm, budget, index, _ in unit_groups(plan):
        reports.setdefault((bench, recipe, arm, budget), []).append(
            read_record(unit_path(output, bench, recipe, arm, budget, index), SCHEMA_UNIT))
    return {k: sorted(v, key=lambda r: r["group"]) for k, v in reports.items()}


def training_tables(reports):
    units, failures, skips = {}, [], []
    for (bench, recipe, arm, budget), groups in sorted(reports.items()):
        members = [m for g in groups for m in g["members"]]
        units[unit_key(bench, recipe, arm, budget)] = {
            "members": len(members), "complete": sum(m["complete"] for m in members),
            "skipped_updates": sum(m["skipped"] for m in members),
            "members_with_skips": sum(m["skipped"] > 0 for m in members),
            "updates": groups[0]["updates"], "update_counts": groups[0]["update_counts"],
            "gpu_seconds": sum(g["gpu_seconds"] for g in groups),
            "peak_memory_gb": max(g["peak_memory_gb"] for g in groups),
            "parameters_per_member": groups[0]["parameters_per_member"]}
        for m in members:
            if not m["complete"]:
                failures.append({"unit": unit_key(bench, recipe, arm, budget), "fold": m["fold"], "seed": m["seed"],
                                 "typed_failure": "member_retired_nonfinite",
                                 "retired_at_update": m["retired_at_update"], "request": m["retired_request"]})
        for g in groups:
            for event in g["nonfinite_events"]:
                skips.append({"unit": unit_key(bench, recipe, arm, budget), "fold": event["fold"],
                              "seed": event["seed"], "count": event["count"], "retired": event["retired"],
                              "first_request": event["first_request"]})
    return {"units": units, "typed_failures": failures, "nonfinite_members": skips}


def guard_tables(output, plan, reports, missing, spot):
    g1 = all(m["fold"] == "all" and m["pool_scenes"] == scenes(bench) or
             m["fold"] != "all" and m["fold"] not in m["pool_scenes"] and len(m["pool_scenes"]) == len(scenes(bench)) - 1
             for (bench, *_), groups in reports.items() for g in groups for m in g["members"])
    g2 = all(len(groups) == len(member_groups(bench, plan["training"]["group_sizes"][recipe]))
             and all(g["updates"] == len(schedule_of(arm, budget)) for g in groups)
             and sorted((m["fold"], m["seed"]) for g in groups for m in g["members"])
             == sorted((f, s) for f, s in members_of(bench))
             for (bench, recipe, arm, budget), groups in reports.items()) and len(reports) == len(UNITS)
    policies = all(read_record(record_path(output, *c), SCHEMA_RECORD)["scoring_policy"] == harness.SCORING_POLICY
                   for c in score_cells())
    g3 = policies and bool(spot) and all(v["equal"] for v in spot.values())
    scope = {}
    for bench, recipe, arm, budget in UNITS:
        for b in ri.BUDGETS:
            if budget != b and arm != "J":
                continue
            entry = scope.setdefault(f"{bench}/{recipe}/{b}", {"cells": 0, "missing": 0})
            for request in arm_requests(arm):
                entry["cells"] += missing[(unit_key(bench, recipe, arm, budget), request)]["cells"]
                entry["missing"] += missing[(unit_key(bench, recipe, arm, budget), request)]["missing"]
    for entry in scope.values():
        entry["share"] = entry["missing"] / entry["cells"] if entry["cells"] else None
        entry["pass"] = entry["share"] is not None and entry["share"] <= NONFINITE_SHARE_MAX
    trained = [read_json(unit_path(output, *u[:5]))["trained_at"] for u in unit_groups(plan)]
    scored = [read_record(record_path(output, *c), SCHEMA_RECORD)["scored_at"] for c in score_cells()]
    chronology = plan["frozen_at"] <= min(trained) and max(trained) <= min(scored)
    ledger = load_ledger(output)
    caps = ledger["gpu_seconds_elapsed"] <= plan["caps"]["gpu_seconds"]
    out = {"G1": {"pass": g1}, "G2": {"pass": g2}, "G3": {"pass": g3, "policies": policies, "spot": spot},
           "G4": {"per_scope": scope, "pass": all(e["pass"] for e in scope.values())},
           "chronology": {"pass": chronology, "frozen_at": plan["frozen_at"], "last_trained": max(trained),
                          "first_scored": min(scored)},
           "caps": {"pass": caps, "gpu_seconds": ledger["gpu_seconds_elapsed"], "cap": plan["caps"]["gpu_seconds"]}}
    out["global_pass"] = global_pass(out)
    return out


GLOBAL_GUARDS = ("G1", "G2", "G3", "chronology", "caps")


def global_pass(guards):
    """Every frozen guard that is not per scope, the GPU-seconds cap included (G4 is per scope)."""
    return all(guards[k]["pass"] for k in GLOBAL_GUARDS)


def guards_pass(guards):
    return global_pass(guards) and guards["G4"]["pass"]


def iclr_reference(per):
    """In-sample J under B (fold 'all' on the 12 training scenes, the #79 / #84 setting) next to the published
    joint (hybrid) numbers; DESCRIPTIVE."""
    out = {}
    for bench, spec in BENCHMARKS.items():
        summary = read_json(spec["source"] / "summary.json")
        rows = {}
        for request in REQUESTS:
            delta, alpha = request_delta(request), request_alpha(request)
            published = [summary["per_seed"][str(s)][f"hybrid_{alpha}_h{delta}"]["curves"][str(ENDPOINT)][METRIC]
                         for s in SEEDS]
            ours = rollout_block(per, bench, "insample", unit_key(bench, "B", "J", "full"), request, ENDPOINT)
            rows[request] = {"published_hybrid_mean": float(np.mean(published)), "published_per_seed": published,
                             "retrained_J_B": ours}
        out[bench] = {"rows": rows, "iclr_quoted": spec["iclr"],
                      "reading": ("fold-'all' J under B retrains the #79 / #84 hybrid on the same 12 scenes and "
                                  "draws; the published number is the mean over 48 in-sample units and 3 seeds")}
    return out


def records_overview(output):
    present = sorted(Path(output).glob("records/*.json"))
    expected = sorted(record_path(output, *c) for c in score_cells())
    return {"records": len(present), "expected": len(expected), "exact": present == expected}


def manifest(output):
    """Content hashes of every raw record, training report and update log."""
    output = Path(output)
    overview = records_overview(output)
    if not overview["exact"]:
        raise ValueError(f"records/ holds {overview['records']} files, expected exactly {overview['expected']}")
    files = sorted(output.glob("records/*.json")) + sorted(output.glob("training/*.json")) \
        + sorted(output.glob("updates/*.pt"))
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "count": len(files),
            "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in files}}


def compute_tables(output, plan, spot):
    output = Path(output)
    reports = unit_reports(output, plan)
    per, missing = collect(output)
    guards = guard_tables(output, plan, reports, missing, spot)
    rollout = {bench: {cohort: {f"{unit_key(*u)}:{r}:{t}": rollout_block(per, bench, cohort, unit_key(*u), r, t)
                                for u in UNITS if u[0] == bench for r in arm_requests(u[2]) for t in TIMES}
                       for cohort in ("lolo", "insample")} for bench in BENCHMARKS}
    ratios = {bench: {cohort: ratio_tables(per, bench, cohort) for cohort in ("lolo", "insample")}
              for bench in BENCHMARKS}
    disp = dispositions(ratios, guards)
    training = training_tables(reports)
    ledger = load_ledger(output)
    train_gpu = sum(u["gpu_seconds"] for u in training["units"].values())
    score_wall = sum(v["wall_seconds"] for k, v in ledger["phases"].items() if k.startswith("score_"))
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"], "engine_seconds": 0,
        "records": records_overview(output), "guards": guards, "guards_pass": guards_pass(guards),
        "training": training, "rollout": rollout, "ratios": ratios, "dispositions": disp, "headline": headline(disp),
        "ranking": RANKING, "iclr_reference": iclr_reference(per),
        "compute": {"training_gpu_seconds": train_gpu, "scoring_wall_seconds": score_wall,
                    "gpu_seconds": ledger["gpu_seconds_elapsed"], "wall_seconds_summed_over_workers": ledger["wall_seconds_elapsed"],
                    "projected_gpu_seconds": plan["projection"]["gpu_seconds"], "engine_seconds": 0,
                    "device": "4x NVIDIA RTX 5090 (#85 names one RTX 3090)"},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def ratio_cell(block):
    if block is None:
        return "n/a"
    return f"{num(block['ratio'], 2)} [{num(block['interval'][0], 2)}, {num(block['interval'][1], 2)}]"


def ror_cell(block):
    if block is None:
        return "n/a"
    return f"{num(block['ratio_of_ratios'], 2)} [{num(block['interval'][0], 2)}, {num(block['interval'][1], 2)}]"


def seeds_cell(block):
    if block is None:
        return "n/a"
    return ", ".join(num(v, 2) for v in block["per_seed"])


def error_cell(block):
    if block is None:
        return "n/a"
    return f"{num(block['mean'])} [{num(block['interval'][0])}, {num(block['interval'][1])}]"


def short(request):
    return request[2:].replace("continuous", "cont")


def findings_md(plan, compute):
    lines = ["# Issue #117 item 5 - request interference on CLEVRER and Physion-Dominoes", "",
             f"Plan frozen {plan['frozen_at']}; validation `{VALIDATION_COMMAND}`. Zero engine seconds.",
             "Guards: " + ", ".join(f"{k} {v['pass']}" for k, v in compute["guards"].items()
                                    if isinstance(v, dict) and "pass" in v) + ".",
             "Leave-one-scene-out (LOSO) results are EXPLORATORY; in-sample (fold 'all', the ICLR setting) results "
             "are DESCRIPTIVE. Intervals: scene-clustered percentile bootstrap (10 000 draws, PCG64 7201) of seed-mean "
             "errors; per-seed geometric ratios are listed beside the interval.", "",
             f"## 1.5x disposition (joint / specialist at t = {ENDPOINT}, LOSO)", "",
             f"Rule: supported iff every request's upper bound < {CLAIM}; not_supported iff some lower bound > {CLAIM}; "
             "otherwise readiness_or_precision_insufficient. Headline = per-request budget.", "",
             "| benchmark | recipe | per-request budget | full budget | largest upper bound (request / full) |",
             "| --- | --- | --- | --- | --- |"]
    disp = compute["dispositions"]
    for bench in BENCHMARKS:
        for recipe in RECIPES:
            a, b = disp[f"{bench}/{recipe}/request"], disp[f"{bench}/{recipe}/full"]
            lines.append(f"| {BENCHMARKS[bench]['label']} | {recipe} | {a['token']} | {b['token']} | "
                         f"{num(a['largest_upper_bound'], 2)} / {num(b['largest_upper_bound'], 2)} |")
    lines += ["", "Per-request tokens:", ""]
    for key, entry in disp.items():
        lines.append(f"- {key}: " + ", ".join(f"{short(r)} {t}" for r, t in entry["per_request"].items()))
    lines += ["", f"## Joint / specialist ratio at t = {ENDPOINT} (LOSO; geometric, request-mean and per request)", ""]
    for bench in BENCHMARKS:
        lines += [f"### {BENCHMARKS[bench]['label']}", "",
                  "| recipe / budget | all six | " + " | ".join(short(r) for r in REQUESTS) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in REQUESTS) + " |"]
        for key, block in compute["ratios"][bench]["lolo"].items():
            js = block["J/S"]
            lines.append(f"| {key} | {ratio_cell(js['groups']['all'])} | "
                         + " | ".join(ratio_cell(js["per_request"][r][str(ENDPOINT)]) for r in REQUESTS) + " |")
        lines += ["", "Per-seed joint / specialist ratios at t = 120 (seeds 20260908, 20260909, 20260910):", "",
                  "| recipe / budget | " + " | ".join(short(r) for r in REQUESTS) + " |",
                  "| --- | " + " | ".join("---" for _ in REQUESTS) + " |"]
        for key, block in compute["ratios"][bench]["lolo"].items():
            lines.append(f"| {key} | " + " | ".join(seeds_cell(block["J/S"]["per_request"][r][str(ENDPOINT)])
                                                    for r in REQUESTS) + " |")
        lines.append("")
    lines += ["## Joint / specialist ratio at every endpoint (LOSO)", ""]
    for bench in BENCHMARKS:
        for key, block in compute["ratios"][bench]["lolo"].items():
            lines += [f"### {BENCHMARKS[bench]['label']} {key}", "",
                      "| request | " + " | ".join(f"t = {t}" for t in TIMES) + " |",
                      "| --- | " + " | ".join("---" for _ in TIMES) + " |"]
            for r in REQUESTS:
                lines.append(f"| {short(r)} | " + " | ".join(ratio_cell(block["J/S"]["per_request"][r][str(t)])
                                                            for t in TIMES) + " |")
            lines.append("")
    lines += ["## Asymmetry by Delta and description (joint / specialist at t = 120, LOSO)", "",
              "| benchmark | recipe / budget | Delta = 1 | Delta = 5 | Delta = 15 | continuous | micro | "
              "Delta1 / Delta15 | continuous / micro |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for bench in BENCHMARKS:
        for key, block in compute["ratios"][bench]["lolo"].items():
            g = block["J/S"]["groups"]
            lines.append(f"| {BENCHMARKS[bench]['label']} | {key} | {ratio_cell(g['delta_1'])} | "
                         f"{ratio_cell(g['delta_5'])} | {ratio_cell(g['delta_15'])} | {ratio_cell(g['alpha_continuous'])} | "
                         f"{ratio_cell(g['alpha_micro'])} | {ror_cell(block['J/S']['delta1_over_delta15'])} | "
                         f"{ror_cell(block['J/S']['continuous_over_micro'])} |")
    lines += ["", "## Joint / per-Delta and per-Delta / specialist at t = 120 (LOSO, by Delta)", "",
              "| benchmark | recipe / budget | J/D Delta = 1 | J/D 5 | J/D 15 | D/S Delta = 1 | D/S 5 | D/S 15 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for bench in BENCHMARKS:
        for key, block in compute["ratios"][bench]["lolo"].items():
            lines.append(f"| {BENCHMARKS[bench]['label']} | {key} | "
                         + " | ".join(ratio_cell(block[c]["groups"][f"delta_{d}"]) for c in ("J/D", "D/S")
                                      for d in DELTAS) + " |")
    lines += ["", "## Per-request rollout error (position MSE, LOSO; mean over scenes)", ""]
    for bench in BENCHMARKS:
        for recipe in RECIPES:
            lines += [f"### {BENCHMARKS[bench]['label']} {recipe}", "",
                      "| request | arm | budget | " + " | ".join(f"t = {t}" for t in TIMES) + " |",
                      "| --- | --- | --- | " + " | ".join("---" for _ in TIMES) + " |"]
            for r in REQUESTS:
                for arm, budget in (("J", "full"), (delta_arm(r), "request"), (delta_arm(r), "full"),
                                    (specialist_arm(r), "request"), (specialist_arm(r), "full")):
                    unit = unit_key(bench, recipe, arm, budget)
                    lines.append(f"| {short(r)} | {arm} | {budget} | "
                                 + " | ".join(error_cell(compute["rollout"][bench]["lolo"][f"{unit}:{r}:{t}"])
                                              for t in TIMES) + " |")
            lines.append("")
    lines += ["## In-sample (ICLR setting, DESCRIPTIVE)", "",
              "Retrained joint under B on all 12 scenes, scored on them, next to the published #79 / #84 hybrid "
              f"(t = {ENDPOINT}, position MSE):", "",
              "| benchmark | request | published hybrid (mean of 3 seeds) | retrained J-B |", "| --- | --- | --- | --- |"]
    for bench, block in compute["iclr_reference"].items():
        for r, row in block["rows"].items():
            lines.append(f"| {BENCHMARKS[bench]['label']} | {short(r)} | {num(row['published_hybrid_mean'])} | "
                         f"{error_cell(row['retrained_J_B'])} |")
    lines += ["", "In-sample joint / specialist ratio at t = 120 (request-mean, geometric):", "",
              "| benchmark | " + " | ".join(f"{r} / {b}" for r in RECIPES for b in ri.BUDGETS) + " |",
              "| --- | " + " | ".join("---" for _ in RECIPES for _ in ri.BUDGETS) + " |"]
    for bench in BENCHMARKS:
        lines.append(f"| {BENCHMARKS[bench]['label']} | "
                     + " | ".join(ratio_cell(compute["ratios"][bench]["insample"][f"{r}/{b}"]["J/S"]["groups"]["all"])
                                  for r in RECIPES for b in ri.BUDGETS) + " |")
    lines += ["", "## Ranking AUC", "", f"`{compute['ranking']['token']}`: {compute['ranking']['reason']}.", "",
              "## Training", "",
              f"Typed failures (retired members): {len(compute['training']['typed_failures'])}; members with a "
              f"nonfinite update: {len(compute['training']['nonfinite_members'])}.", ""]
    for failure in compute["training"]["typed_failures"]:
        lines.append(f"- {failure['unit']} fold {failure['fold']} seed {failure['seed']}: retired at update "
                     f"{failure['retired_at_update']} on {failure['request']}")
    nonfinite = {}
    for entry in compute["training"]["nonfinite_members"]:
        nonfinite.setdefault(entry["unit"], []).append(entry)
    for unit, entries in nonfinite.items():
        lines.append(f"- {unit}: {len(entries)} members with nonfinite updates "
                     f"({sum(e['count'] for e in entries)} updates; first requests "
                     f"{sorted({e['first_request'] for e in entries})})")
    g4 = compute["guards"]["G4"]["per_scope"]
    lines += ["", "G4 (share of LOSO cells without a finite t = 120 error): "
              + ", ".join(f"{k} {num(v['share'], 3)}" for k, v in g4.items()), "",
              "## Compute", "",
              f"Training {compute['compute']['training_gpu_seconds']:.0f} GPU s; ledger total "
              f"{compute['compute']['gpu_seconds']:.0f} GPU s (projection {compute['compute']['projected_gpu_seconds']:.0f}); "
              f"engine seconds 0; {compute['compute']['device']}.", ""]
    return "\n".join(lines)


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["benchmark", "cohort", "recipe_budget", "comparison", "request_or_group", "t", "ratio",
                     "interval_low", "interval_high", "clusters", "per_seed"])
    for bench in BENCHMARKS:
        for cohort in ("lolo", "insample"):
            for key, block in compute["ratios"][bench][cohort].items():
                for comparison, entry in block.items():
                    rows = [(r, t, entry["per_request"][r][str(t)]) for r in REQUESTS for t in TIMES]
                    rows += [(g, ENDPOINT, b) for g, b in entry["groups"].items()]
                    for name, t, b in rows:
                        if b is None:
                            writer.writerow([bench, cohort, key, comparison, name, t, "", "", "", "", ""])
                        else:
                            writer.writerow([bench, cohort, key, comparison, name, t, f"{b['ratio']:.6g}",
                                             f"{b['interval'][0]:.6g}", f"{b['interval'][1]:.6g}", b["clusters"],
                                             ";".join("" if v is None else f"{v:.6g}" for v in b["per_seed"])])
    return buffer.getvalue()


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "question": plan["question"],
            "guards_pass": compute["guards_pass"], "headline": compute["headline"],
            "dispositions": {k: v["token"] for k, v in compute["dispositions"].items()},
            "joint_over_specialist_request_mean": {
                f"{bench}/{key}": compute["ratios"][bench]["lolo"][key]["J/S"]["groups"]["all"]
                for bench in BENCHMARKS for key in compute["ratios"][bench]["lolo"]},
            "ranking": compute["ranking"], "typed_failures": len(compute["training"]["typed_failures"]),
            "gpu_seconds": compute["compute"]["gpu_seconds"], "engine_seconds": 0,
            "claim_boundary": plan["claim_boundary"]}


def rendered(plan, compute, output):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute), "records-manifest.json": json_text(manifest(output))}


def spot_checks(output):
    """Re-score two retained records (LC joint, first LOSO fold, CLEVRER; B Delta = 1 specialist, fold 'all',
    Physion-Dominoes) and compare them with the stored rollout errors."""
    out = {}
    for name, cell in (("clevrer_LC_J_lolo", ("clevrer", "LC", "J", "full", scenes("clevrer")[0], SEEDS[0])),
                       ("physion_B_S1_insample", ("physion", "B", "S-1-continuous", "request", "all", SEEDS[0]))):
        record = read_record(record_path(output, *cell), SCHEMA_RECORD)
        fresh = score_record(output, Data(cell[0]), *cell, harness.SCORING_POLICY)
        out[name] = {"cell": list(cell), "equal": json.loads(json.dumps(fresh["rollout"])) == record["rollout"]}
    return out


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    with harness.deterministic_scoring():
        spot = spot_checks(output)
    compute = compute_tables(output, plan, spot)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute, output).items():
        (output / name).write_bytes(text.encode())
    log(f"published: headline {compute['headline']}; guards pass {compute['guards_pass']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    with harness.deterministic_scoring():
        spot = spot_checks(output)
    fresh = compute_tables(output, plan, spot)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation (compared by value)")
    for name, text in rendered(plan, fresh, output).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    if not fresh["guards"]["G3"]["pass"]:
        raise ValueError("deterministic re-scoring does not replicate the retained records")
    log(f"validation passed: every table re-derived; compute.json compared by value, the summary, the findings, "
        f"comparisons.csv and the records manifest byte-compared; spot re-scores replicate "
        f"({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    log(f"  benchmarks {list(BENCHMARKS)}; recipes {list(RECIPES)}; arms {list(ARMS)}; units {len(UNITS)}; "
        f"members per unit {len(members_of('clevrer'))}")
    for arm in ARMS:
        log(f"  {arm}: requests {arm_requests(arm)}; updates { {b: len(schedule_of(arm, b)) for b in arm_budgets(arm)} }")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "score": score,
         "publish": publish, "validate": validate}
SHARDED = {"train", "score"}


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
