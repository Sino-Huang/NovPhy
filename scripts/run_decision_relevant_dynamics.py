"""Issue-112: decision-relevant dynamics on readable carriers (the Gate-B model).

Binding runner module: scripts/run_decision_relevant_dynamics.py
Exact validation command: python -u -m scripts.run_decision_relevant_dynamics --validate

The #99 spatial slot parser (E99) reads the pig and the micro relations, but predictors
retrained on its carriers rank held-out launches no better than on the blind parser's.
The #77 recipe supervises at most four transitions inside 60-frame windows, while every
ranking is scored after a 225-frame rollout. This runner

1. diagnoses the frozen #99 E predictors on fit lineages (zero training): action
   sensitivity of the predicted endpoint against the engine spread, per-request rollout
   error growth from 15 to 225 frames, and whether the anchor (start carrier) or the
   action explains the endpoint variance;
2. trains four arms of the same two predictor families on E99 carriers of the 12 fit
   lineages (B: #77 recipe; L: + long-horizon recursive supervision to 225 frames with a
   60 -> 225 horizon curriculum; C: + a decision-contrastive pairwise ranking loss on
   the tie-free cost of fit-lineage anchors; LC: both), each for 13 folds (leave one fit
   lineage out, and all 12) x 3 seeds, with the #77 update budget;
3. scores every fold model on its left-out lineage (leave-one-lineage-out validation,
   grid/offset AUC under the tie-free cost, and the #100 engine-referenced rollout
   error) and selects one (arm, request) by the frozen rule, which freezes the recipe;
4. scores the held-out lineages 007/008/014/016 exactly once after that freeze
   (descriptive; these lineages are exposed by earlier tickets);
5. hands the frozen recipe and its checkpoints to Gate B (#104 sealed split).

Modes: --dry-run --smoke --prepare --data --diagnose --train --score-lolo --select
--score-heldout --publish --validate. Training runs in its own mode invocation
(torch.compile, autotuning allowed); every scoring mode and --validate runs under the
#111 deterministic_scoring() policy and writes it into each record. Zero engine
seconds; every interval is a member-clustered percentile bootstrap (10000 draws, PCG64
7201), DESCRIPTIVE; development (leave-one-lineage-out) results are EXPLORATORY.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import io
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_issue_77_n1_diagnostic as diag
from scripts import run_launch_power_probe as p94
from scripts import run_lookahead_control as wlc
from scripts import run_readout_fidelity_jepa as r100
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import decision_dynamics as dd

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-112-decision-relevant-dynamics-v1"
SLOT_FIX = s99.OUTPUT
READOUT = r100.OUTPUT
ATTRIBUTION = m99.OUTPUT
CAMPAIGN = s99.CAMPAIGN
DYNAMICS = t96.DYNAMICS

IDENTITY = "issue-112-decision-relevant-dynamics-v1"
SCHEMA_PLAN = "issue_112_plan_v1"
SCHEMA_SMOKE = "issue_112_smoke_v1"
SCHEMA_DATA = "issue_112_training_data_v1"
SCHEMA_PREDICTOR = "issue_112_predictor_v1"
SCHEMA_GROUP = "issue_112_training_group_v1"
SCHEMA_DIAG = "issue_112_diagnostic_record_v1"
SCHEMA_ENDPOINTS = "issue_112_diagnostic_endpoints_v1"
SCHEMA_LOLO = "issue_112_lolo_record_v1"
SCHEMA_HELDOUT = "issue_112_heldout_record_v1"
SCHEMA_SELECTION = "issue_112_selection_v1"
SCHEMA_HANDOFF = "issue_112_gate_b_handoff_v1"
SCHEMA_LEDGER = "issue_112_ledger_v1"
SCHEMA_COMPUTE = "issue_112_compute_v1"
SCHEMA_REPORT = "issue_112_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_decision_relevant_dynamics --validate"
RUNNER = "scripts/run_decision_relevant_dynamics.py"
MODEL_MODULE = "world_model/training/decision_dynamics.py"

DEVICE = "cuda"
SEEDS = t96.SEEDS
ENDPOINT = t96.ENDPOINT
INVENTORIES = t96.INVENTORIES
PRIMARY = t96.PRIMARY
SELECTION_INVENTORIES = ("grid", "offset")
HELD_OUT = m99.HELD_OUT
FAMILIES = ("hybrid", "continuous")
REQUESTS = s99.REQUESTS
COSTS = harness.COSTS
PRIMARY_COST = harness.PRIMARY_COST
TIMES = (15, 30, 60, 150, 225)
ARMS = ("B", "L", "C", "LC")
ARM_DEFINITIONS = {
    "B": "#77 recipe only (the E99 baseline predictor recipe, retrained here per fold)",
    "L": "#77 recipe + long-horizon recursive supervision to 225 frames (horizon curriculum 60 -> 225)",
    "C": "#77 recipe + decision-contrastive pairwise ranking loss on fit-lineage anchors",
    "LC": "#77 recipe + long-horizon supervision + decision-contrastive loss",
}
REFERENCE = "E99"  # the frozen #99 E predictors (held-out reference only)

LONG = {"starts": [0, 15, 30, 45, 60], "batch_size": 64, "first": 60, "last": ENDPOINT, "increment": 15,
        "every": 500, "weight": 1.0, "truncation": "none (full back-propagation through the unrolled recursion)",
        "targets": "the E99 parsed carrier at every visited position start + j * delta, terminal absorption past "
                   "the shot's last frame", "seed_offset": 112001}
RANK = {"cells_per_step": 2, "candidate_pad": 20, "temperature": 0.1, "weight": 0.1, "start": 1000, "ramp": 1000,
        "inventories": list(INVENTORIES), "seed_offset": 112002,
        "cost": "tie_free (scoring_harness.tie_free on tensors): pig presence - 0.1 x pig-slot center displacement "
                "between the anchor carrier and the rollout endpoint at 225"}

TARGET_AUC = m99.TARGET_AUC
TARGET_LOWER = m99.TARGET_LOWER
DELTA = 0.02
NONFINITE_SHARE_MAX = 0.10
GPU_CAP_SECONDS = 16 * 3600.0
WALL_CAP_SECONDS = 30 * 3600.0
STOP_TOKEN = t96.STOP_TOKEN
MEMORY_BUDGET_GB = 20.0
MAX_GROUP = 39
WINDOW_POSITIONS = s99.WINDOW_LENGTH + 1
PROGRESS_EVERY = 1000
TRACE_EVERY = 900
SMOKE_STEPS = 12
PLACEHOLDER_MARKERS = m99.PLACEHOLDER_MARKERS

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
bootstrap = t96.bootstrap
fmt = t96.fmt
fmt_interval = t96.fmt_interval
atomic_torch = s99.atomic_torch


def log(message):
    print(f"[issue-112-dynamics] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Progress:
    """Foreground progress with ETA."""

    def __init__(self, label, total):
        self.label, self.total, self.done, self.began = label, total, 0, time.monotonic()

    def step(self, every=1):
        self.done += 1
        if self.done % every == 0 or self.done == self.total:
            elapsed = time.monotonic() - self.began
            log(f"{self.label} {self.done}/{self.total}; elapsed {elapsed:.0f}s; "
                f"eta {elapsed / self.done * (self.total - self.done):.0f}s")


# ---------------------------------------------------------------------------
# membership (no statistic)
# ---------------------------------------------------------------------------

def fit_lineages():
    return s99.fit_members()


def folds():
    return (*fit_lineages(), "all")


def ensemble_members():
    """Frozen member order of one (arm, family) ensemble: folds x seeds."""
    return [(fold, seed) for fold in folds() for seed in SEEDS]


def member_groups(size):
    members = ensemble_members()
    return [members[i:i + size] for i in range(0, len(members), size)]


def family_requests(family):
    return [r for r in REQUESTS if t96.arm_family(r) == family]


def family_pairs(family):
    from world_model.training.cnn_hybrid import PAIRS
    from world_model.training.matched_dynamics import CONTINUOUS_PAIRS
    return PAIRS if family == "hybrid" else CONTINUOUS_PAIRS


def request_pair(request):
    return family_pairs(t96.arm_family(request))[t96.arm_spec(request)["pair_index"]]


def step_pair(family, step):
    """#77 schedule: PAIRS[step % 9]; the continuous family uses the same horizon."""
    from world_model.model import Abstraction, PredictionPair
    from world_model.training.cnn_hybrid import PAIRS
    selected = PAIRS[step % 9]
    return selected if family == "hybrid" else PredictionPair(selected.delta, Abstraction.CONTINUOUS)


def load_universe():
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    return sources, members, universe


def fit_cells(universe):
    """Every mixed-verdict (member, inventory) cell of a fit lineage: the ranking-loss units."""
    fitted = set(fit_lineages())
    return [(member, inventory) for inventory in INVENTORIES for member in sorted(universe[inventory])
            if member in fitted and t96.mixed(universe[inventory][member])]


def rollout_samples_all():
    """#77 diagnostic sample rule (every coverage-admissible N1 branch, candidate order) for
    every lineage; the held-out lineages reproduce the #100 rollout samples exactly."""
    campaign = read_json(CAMPAIGN / "plan.json")
    coverage = read_json(CAMPAIGN / "coverage.json")
    plan_branches = {b["identity"]: b for b in campaign["branches"]}
    samples = {}
    for member in sorted(campaign["members"], key=lambda m: m["identity"]):
        admissible = [b for b in sorted(campaign["branches"], key=lambda b: b["candidate_ordinal"])
                      if b["source_member_identity"] == member["identity"]
                      and coverage["branches"][b["identity"]]["status"] == "admissible"]
        samples[member["identity"]] = [diag.candidate_source(b, plan_branches) for b in admissible]
    held = {s["state"]["identity"]: [x["identity"] for x in s["sources"]] for s in r100.rollout_samples()}
    mine = {m: [x["identity"] for x in v] for m, v in samples.items() if m in HELD_OUT and v}
    if held != mine:
        raise ValueError("held-out rollout samples differ from the #100 rollout samples")
    return samples


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def data_path(output):
    return Path(output) / "data" / "training-data.pt"


def anchors_path(output, cohort):
    return Path(output) / "data" / f"anchors-{cohort}.pt"


def contexts_path(output, cohort):
    return Path(output) / "data" / f"contexts-{cohort}.pt"


def member_path(output, arm, family, fold, seed):
    return Path(output) / "models" / arm / family / f"fold-{fold}" / f"seed-{seed}" / "predictor.pt"


def group_path(output, arm, family, index):
    return Path(output) / "training" / f"{arm}--{family}--group{index}.json"


def progress_path(output, arm, family, index):
    return Path(output) / "training" / f"{arm}--{family}--group{index}.progress.pt"


def diag_path(output, family, seed):
    return Path(output) / "records" / f"diag--{family}--seed{seed}.json"


def endpoints_path(output):
    return Path(output) / "records" / "diag-endpoints.json"


def lolo_path(output, arm, family, fold, seed):
    return Path(output) / "records" / f"lolo--{arm}--{family}--{fold}--seed{seed}.json"


def heldout_path(output, arm, family, seed):
    return Path(output) / "records" / f"heldout--{arm}--{family}--seed{seed}.json"


@functools.lru_cache(maxsize=None)
def _record_json(path, modified):
    return read_json(path)


def read_record(path, schema):
    """Retained records are immutable once written; a changed file is re-read (mtime key)."""
    record = _record_json(str(path), Path(path).stat().st_mtime_ns)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


# ---------------------------------------------------------------------------
# training data (outcome-free windows; fit-lineage verdicts for the ranking loss)
# ---------------------------------------------------------------------------

def load_shot(key):
    data = torch.load(s99.shot_path(SLOT_FIX, key), map_location="cpu", weights_only=True, mmap=True)
    carriers = torch.load(s99.carrier_path(SLOT_FIX, "E", key), map_location="cpu", weights_only=True)
    if carriers.get("plan_identity") != s99.IDENTITY or carriers.get("arm") != "E":
        raise ValueError(f"E99 carriers of {key} binding differs")
    carriers = carriers["carriers"]
    if len(carriers) != data["end"] + 1:
        raise ValueError(f"E99 carriers of {key} do not cover positions 0..end")
    return data, carriers


def build_training_data(shots, universe, anchors):
    """#99 windows (s99.windows, identical order to s99.window_data), long windows, and the
    ranking cells of the fit lineages."""
    rows, window_shot, carriers_all, offsets, lasts, actions = [], [], [], [], [], []
    total = 0
    progress = Progress("training data", len(shots))
    for index, shot in enumerate(shots):
        data, carriers = load_shot(shot["key"])
        windows = s99.windows(data, carriers)
        rows.extend(windows)
        window_shot.extend([index] * len(windows))
        offsets.append(total)
        lasts.append(int(data["end"]))
        actions.append(data["action"].clone())
        carriers_all.append(carriers)
        total += len(carriers)
        progress.step(50)
    window = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    long_windows = [(index, start) for index, last in enumerate(lasts) for start in LONG["starts"] if start < last]
    cells = []
    for member, inventory in fit_cells(universe):
        entry = universe[inventory][member]
        items = [item for item in entry["inventory"] if item["ordinal"] in entry["verdicts"]]
        if len(items) > RANK["candidate_pad"]:
            raise ValueError("a ranking cell exceeds the candidate pad")
        action = torch.zeros(RANK["candidate_pad"], 5)
        verdict = torch.full((RANK["candidate_pad"],), -1, dtype=torch.long)
        action[:len(items)] = torch.cat([wlc.action_tensor(item["action"], "cpu") for item in items])
        verdict[:len(items)] = torch.tensor([int(entry["verdicts"][item["ordinal"]]) for item in items])
        cells.append({"member": member, "inventory": inventory, "ordinals": [item["ordinal"] for item in items],
                      "z0": anchors[member], "action": action, "verdict": verdict})
    members = [s["member"] for s in shots]
    return {"schema": SCHEMA_DATA, "plan_identity": IDENTITY, **window,
            "window_shot": torch.tensor(window_shot), "shot_keys": [s["key"] for s in shots],
            "shot_members": members, "carriers": torch.cat(carriers_all), "offsets": torch.tensor(offsets),
            "lasts": torch.tensor(lasts), "shot_actions": torch.stack(actions),
            "long_windows": torch.tensor(long_windows),
            "cell_members": [c["member"] for c in cells], "cell_inventories": [c["inventory"] for c in cells],
            "cell_ordinals": [c["ordinals"] for c in cells],
            "cell_z0": torch.stack([c["z0"] for c in cells]), "cell_action": torch.stack([c["action"] for c in cells]),
            "cell_verdict": torch.stack([c["verdict"] for c in cells])}


def parse_anchors(adapter, members):
    return {member: wlc.encode_carrier(adapter, p94.frozen_anchor(member), DEVICE).float().cpu()
            for member in members}


def parse_contexts(adapter, samples, members):
    return {member: r100.context_carrier(adapter, CAMPAIGN / samples[member][0]["attempt"]).float().cpu()
            for member in members if samples.get(member)}


class TrainData:
    """GPU-resident training tensors and per-fold pools."""

    WINDOW_KEYS = ("z", "action", "length")
    SYMBOL_KEYS = ("relations", "relations_mask", "macros", "macros_mask")

    def __init__(self, payload, family):
        keys = self.WINDOW_KEYS + (self.SYMBOL_KEYS if family == "hybrid" else ())
        self.window = {k: payload[k].to(DEVICE) for k in keys}
        self.shot_members = payload["shot_members"]
        self.window_members = [self.shot_members[i] for i in payload["window_shot"].tolist()]
        self.carriers = payload["carriers"].to(DEVICE)
        self.offsets = payload["offsets"].to(DEVICE)
        self.lasts = payload["lasts"].to(DEVICE)
        self.shot_actions = payload["shot_actions"].to(DEVICE)
        self.long = payload["long_windows"].to(DEVICE)
        self.long_members = [self.shot_members[i] for i in payload["long_windows"][:, 0].tolist()]
        self.cell_members = payload["cell_members"]
        self.cell_z0 = payload["cell_z0"].to(DEVICE)
        self.cell_action = payload["cell_action"].to(DEVICE)
        self.cell_verdict = payload["cell_verdict"].to(DEVICE)

    @staticmethod
    def _pool(members, fold):
        if set(members) & set(HELD_OUT):
            raise ValueError("a held-out lineage entered the training data")
        return torch.tensor([i for i, m in enumerate(members) if m != fold], dtype=torch.long)

    def pools(self, fold):
        pools = {"window": self._pool(self.window_members, fold), "long": self._pool(self.long_members, fold),
                 "cell": self._pool(self.cell_members, fold)}
        if fold != "all" and any(m == fold for key, members in (("window", self.window_members),
                                                                 ("long", self.long_members),
                                                                 ("cell", self.cell_members))
                                 for m in (members[i] for i in pools[key].tolist())):
            raise ValueError(f"fold {fold} pool holds its own lineage")
        return pools

    @staticmethod
    def _draw(pools, key, generators, count):
        return torch.stack([pool[key][torch.randint(len(pool[key]), (count,), generator=g)]
                            for pool, g in zip(pools, generators)]).to(DEVICE)

    def windows(self, pools, generators, count):
        index = self._draw(pools, "window", generators, count)
        return {k: v[index] for k, v in self.window.items()}

    def long_batch(self, pools, generators, count, horizon, delta):
        index = self._draw(pools, "long", generators, count)
        shot, start = self.long[index, 0], self.long[index, 1]
        offset, last = self.offsets[shot], self.lasts[shot]
        steps = torch.arange(1, horizon // delta + 1, device=DEVICE) * delta
        positions = torch.minimum(start[..., None] + steps, last[..., None])
        return {"start": self.carriers[offset + start], "action": self.shot_actions[shot],
                "targets": self.carriers[offset[..., None] + positions]}

    def rank_batch(self, pools, generators, count):
        index = self._draw(pools, "cell", generators, count)
        return {"z0": self.cell_z0[index], "action": self.cell_action[index], "verdict": self.cell_verdict[index]}


# ---------------------------------------------------------------------------
# training (its own mode invocation; torch.compile and autotuning allowed)
# ---------------------------------------------------------------------------

def recipe():
    return s99.training_recipe()


def pig_columns():
    costs = harness.EndpointCosts(wlc.load_objective())
    return costs.presence, costs.center


def train_group(output, arm, family, index, members, data, *, steps=None, save=True, smoke_full=False,
                timing_after=None):
    """Train one group of ensemble members (resumable); returns the group report.

    smoke_full (smoke only): the ranking term from update 0 and the long horizon at 225
    from update 0, the most expensive update of each arm; timing_after: report the mean
    wall seconds per update after that many updates (compile warm-up excluded)."""
    frozen = recipe()
    total = steps or frozen["steps"]
    models = []
    for fold, seed in members:
        torch.manual_seed(seed)
        models.append(s99.new_predictor(family).to(DEVICE))
    ensemble = dd.Ensemble(models, learning_rate=frozen["learning_rate"], weight_decay=frozen["weight_decay"],
                           grad_clip=frozen["grad_clip"], compile=True)
    del models
    pools = [data.pools(fold) for fold, _ in members]
    window_generators = [torch.Generator().manual_seed(seed) for _, seed in members]
    long_generators = [torch.Generator().manual_seed(seed + LONG["seed_offset"]) for _, seed in members]
    rank_generators = [torch.Generator().manual_seed(seed + RANK["seed_offset"]) for _, seed in members]
    presence, center = pig_columns()
    uses_long, uses_rank = arm in ("L", "LC"), arm in ("C", "LC")
    first, trace, gpu = 0, [], 0.0
    progress = progress_path(output, arm, family, index)
    if save and progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        ensemble.load_state(saved["ensemble"])
        for generators, states in ((window_generators, saved["window"]), (long_generators, saved["long"]),
                                   (rank_generators, saved["rank"])):
            for generator, state in zip(generators, states):
                generator.set_state(state)
        first, trace, gpu = saved["step"], saved["trace"], saved["gpu_seconds"]
        log(f"resumed {arm}/{family}/group{index} at update {first}")
    torch.cuda.synchronize()
    began = time.monotonic()
    segment_began, timed_began = began, None
    for step in range(first, total):
        if step == timing_after:
            torch.cuda.synchronize()
            timed_began = time.monotonic()
        pair = step_pair(family, step)
        batch = data.windows(pools, window_generators, frozen["batch_size"])
        local = dd.local_loss(ensemble, batch, pair)
        loss, long_term, rank_term = local, None, None
        weight = RANK["weight"] * (1.0 if smoke_full else dd.ramp_at(step, start=RANK["start"], length=RANK["ramp"]))
        horizon = ENDPOINT if smoke_full else dd.horizon_at(step, first=LONG["first"], last=LONG["last"],
                                                            increment=LONG["increment"], every=LONG["every"])
        long_batch = data.long_batch(pools, long_generators, LONG["batch_size"], horizon, pair.delta) \
            if uses_long else None
        rank_batch = data.rank_batch(pools, rank_generators, RANK["cells_per_step"]) if uses_rank else None
        if uses_long or (uses_rank and weight > 0):
            long_term, rank_term = dd.long_and_rank_loss(
                ensemble, pair, long=long_batch, rank=rank_batch if weight > 0 else None, horizon=horizon,
                endpoint=ENDPOINT, presence=presence, center=center, temperature=RANK["temperature"])
            if uses_long:
                loss = loss + LONG["weight"] * long_term
            if uses_rank and weight > 0:
                loss = loss + weight * rank_term
        report = ensemble.update(loss)
        ensemble.mark_retired_step(step, report["newly_retired"])
        if (step + 1) % TRACE_EVERY == 0 or step + 1 == total:
            trace.append({"step": step + 1, "horizon": horizon, "rank_weight": weight,
                          "loss": loss.detach().cpu().tolist(), "local": local.detach().cpu().tolist(),
                          "long": None if long_term is None else long_term.detach().cpu().tolist(),
                          "rank": None if rank_term is None else rank_term.detach().cpu().tolist(),
                          "grad_norm": report["grad_norm"].cpu().tolist(),
                          "retired": ensemble.retired.cpu().tolist()})
            torch.cuda.synchronize()
            now = time.monotonic()
            log(f"{arm}/{family}/group{index} update {step + 1}/{total}: horizon {horizon}, "
                f"mean loss {float(loss.detach()[~ensemble.retired].mean()) if (~ensemble.retired).any() else 'n/a'}, "
                f"retired {int(ensemble.retired.sum())}; {now - began:.0f}s; "
                f"eta {(now - began) / (step + 1 - first) * (total - step - 1):.0f}s")
        if save and (step + 1) % PROGRESS_EVERY == 0 and step + 1 < total:
            torch.cuda.synchronize()
            now = time.monotonic()
            gpu += now - segment_began
            segment_began = now
            atomic_torch(progress, {"ensemble": ensemble.state(), "step": step + 1, "trace": trace,
                                    "gpu_seconds": gpu,
                                    "window": [g.get_state() for g in window_generators],
                                    "long": [g.get_state() for g in long_generators],
                                    "rank": [g.get_state() for g in rank_generators]})
    torch.cuda.synchronize()
    gpu += time.monotonic() - segment_began
    timed = None if timed_began is None else (time.monotonic() - timed_began) / (total - timing_after)
    parameters = sum(v[0].numel() for v in ensemble.params.values())
    report = {"schema": SCHEMA_GROUP, "plan_identity": IDENTITY, "arm": arm, "family": family, "group": index,
              "members": [{"fold": fold, "seed": seed, "retired_at": ensemble.retired_at[i],
                           "complete": ensemble.retired_at[i] is None} for i, (fold, seed) in enumerate(members)],
              "steps": total, "gpu_seconds": gpu, "parameters_per_member": parameters,
              "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
              "pool_sizes": [{k: len(v) for k, v in pool.items()} for pool in pools], "trace": trace,
              "timed_seconds_per_update": timed}
    if save:
        for i, (fold, seed) in enumerate(members):
            atomic_torch(member_path(output, arm, family, fold, seed),
                         {"schema": SCHEMA_PREDICTOR, "plan_identity": IDENTITY, "arm": arm, "family": family,
                          "fold": fold, "seed": seed, "steps": total, "complete": ensemble.retired_at[i] is None,
                          "retired_at": ensemble.retired_at[i], "parameters": parameters,
                          "model": ensemble.member_state(i)})
        write_json(group_path(output, arm, family, index), report)
        progress.unlink(missing_ok=True)
    return report, ensemble


def load_member(output, arm, family, fold, seed):
    """The trained predictor, or None for a retired (typed-failure) member."""
    if arm == REFERENCE:
        return s99.load_predictor(SLOT_FIX, "E", seed, family)
    payload = torch.load(member_path(output, arm, family, fold, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_PREDICTOR or payload.get("plan_identity") != IDENTITY
            or (payload["arm"], payload["family"], payload["fold"], payload["seed"]) != (arm, family, fold, seed)
            or payload["steps"] != recipe()["steps"]):
        raise ValueError(f"predictor {arm}/{family}/{fold}/{seed} binding differs")
    if not payload["complete"]:
        return None
    model = s99.new_predictor(family)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# scoring blocks (always inside deterministic_scoring)
# ---------------------------------------------------------------------------

def endpoint_block(model, family, z0, items, costs):
    actions = m99.action_batch(items)
    out = {}
    for request in family_requests(family):
        z_end, timing = m99.rollout_endpoints(model, None, z0.to(DEVICE), actions, t96.arm_spec(request))
        out[request] = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu())
                                 for slot, item in enumerate(items)], "candidates": len(items), **timing}
    return out


@torch.no_grad()
def rollout_curves(model, family, context, sources):
    """Batched fixed-request rollouts from the lineage context over its branch actions;
    carriers at every time in TIMES (None where nonfinite: a typed failure)."""
    actions = torch.cat([diag.n1_action_tensor(s["action"], DEVICE) for s in sources])
    out = {}
    for request in family_requests(family):
        pair = request_pair(request)
        z = context.to(DEVICE)[None].expand(len(sources), -1).clone()
        elapsed, curve = 0, {}
        for t in TIMES:
            while elapsed < t:
                z = model.carrier(z, actions, pair)
                elapsed += pair.delta
            curve[t] = z.cpu()
        out[request] = curve
    return out


@functools.lru_cache(maxsize=None)
def audit_truth(identity):
    """The #100 audit engine labels (presence, projected centers) and E99 carriers of one N1 branch."""
    labels = r100.load_labels(READOUT, identity)
    return {"presence": labels["presence"], "centers": labels["centers"]}, r100.load_carriers(READOUT, "E99", identity)


def curve_errors(curves, sources, objective):
    """#100 engine-referenced metric (engine_position_mse) and the own-parse position MSE;
    a nonfinite prediction or error statistic is a typed failure (None)."""
    out = {request: [] for request in curves}
    for index, source in enumerate(sources):
        labels, carriers = audit_truth(source["identity"])
        last = len(carriers) - 1
        for request, curve in curves.items():
            errors = {}
            for t in TIMES:
                prediction = curve[t][index]
                values = prediction.tolist()
                try:
                    if not bool(torch.isfinite(prediction).all()):
                        raise ValueError("nonfinite prediction")
                    field = diag.field_errors(values, carriers[min(t, last)].tolist(), objective)
                    engine = r100.engine_position_mse(values, labels, min(t, last))
                    if engine is not None and not math.isfinite(engine):
                        raise ValueError("nonfinite engine error")
                except ValueError:
                    errors[str(t)] = None
                    continue
                errors[str(t)] = {"engine_position_mse": engine, "position_mse": field["position_mse"]}
            out[request].append({"identity": source["identity"], "errors": errors})
    return out


def lineage_rollout(model, family, context, sources, objective):
    return curve_errors(rollout_curves(model, family, context, sources), sources, objective)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def lolo_record(output, universe, costs, objective, anchors, contexts, samples, arm, family, fold, seed, policy):
    model = load_member(output, arm, family, fold, seed)
    base = {"schema": SCHEMA_LOLO, "plan_identity": IDENTITY, "scoring_policy": policy,
            "cell": {"arm": arm, "family": family, "fold": fold, "seed": seed}, "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": "member_retired_nonfinite", "inventories": {}, "rollout": None}
    inventories = {}
    for inventory in INVENTORIES:
        entry = universe[inventory].get(fold)
        if entry is not None:
            inventories[inventory] = endpoint_block(model, family, anchors[fold], entry["inventory"], costs)
    rollout = lineage_rollout(model, family, contexts[fold], samples[fold], objective) if fold in contexts else None
    return {**base, "typed_failure": None, "inventories": inventories, "rollout": rollout}


def heldout_record(output, universe, costs, objective, anchors, contexts, samples, arm, family, seed, policy):
    model = load_member(output, arm, family, "all", seed)
    base = {"schema": SCHEMA_HELDOUT, "plan_identity": IDENTITY, "scoring_policy": policy, "scored_at": utc_now(),
            "cell": {"arm": arm, "family": family, "seed": seed}, "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": "member_retired_nonfinite", "inventories": {}, "rollout": {}}
    inventories = {inventory: {member: endpoint_block(model, family, anchors[member],
                                                      universe[inventory][member]["inventory"], costs)
                               for member in sorted(m for m in universe[inventory] if m in HELD_OUT)}
                   for inventory in INVENTORIES}
    rollout = {member: lineage_rollout(model, family, contexts[member], samples[member], objective)
               for member in sorted(contexts)}
    return {**base, "typed_failure": None, "inventories": inventories, "rollout": rollout}


def diag_record(universe, costs, objective, anchors, contexts, samples, family, seed, policy):
    """Frozen #99 E predictors on every fit-lineage cell (every inventory item) and N1 sample."""
    model = load_member(None, REFERENCE, family, "all", seed)
    fitted = set(fit_lineages())
    cells = {inventory: {member: endpoint_block(model, family, anchors[member],
                                                universe[inventory][member]["inventory"], costs)
                         for member in sorted(m for m in universe[inventory] if m in fitted)}
             for inventory in INVENTORIES}
    rollout = {member: lineage_rollout(model, family, contexts[member], samples[member], objective)
               for member in sorted(contexts)}
    return {"schema": SCHEMA_DIAG, "plan_identity": IDENTITY, "scoring_policy": policy,
            "cell": {"model": REFERENCE, "family": family, "seed": seed}, "cells": cells, "rollout": rollout,
            "engine_seconds": 0}


def diag_endpoints(universe, costs, policy):
    """Engine endpoint (#99 attribution evidence: engine-projected carriers at 0 and 225) and
    the E99 parsed endpoint (#99 fix carriers) of every executed fit-lineage candidate."""
    fitted = set(fit_lineages())
    out = {}
    for inventory in INVENTORIES:
        out[inventory] = {}
        for member in sorted(m for m in universe[inventory] if m in fitted):
            engine, parsed = [], []
            for ordinal in sorted(universe[inventory][member]["verdicts"]):
                evidence = m99.load_evidence(ATTRIBUTION, inventory, member, ordinal)
                engine.append(costs.row(ordinal, evidence["engine_end"], evidence["engine_start"]))
                _, carriers = load_shot(f"inv--{m99.candidate_key(inventory, member, ordinal)}")
                parsed.append(costs.row(ordinal, carriers[min(ENDPOINT, len(carriers) - 1)], carriers[0]))
            out[inventory][member] = {"engine": engine, "parsed": parsed}
    return {"schema": SCHEMA_ENDPOINTS, "plan_identity": IDENTITY, "scoring_policy": policy, "cells": out,
            "engine_seconds": 0}


# ---------------------------------------------------------------------------
# smoke (pre-freeze: nothing kept, no statistic, no held-out access)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    sources, _, universe = load_universe()
    shots = s99.training_shots(sources, universe)
    checks, timings = [], {}
    fitted = fit_lineages()
    # training data on a small fit subset
    chosen = [s for s in shots if s["member"] in fitted[:2]][:6] + [s for s in shots if s["member"] == fitted[2]][:2]
    with harness.deterministic_scoring():
        adapter = s99.load_adapter(SLOT_FIX, "E")
        anchors = parse_anchors(adapter, sorted({m for m, _ in fit_cells(universe)}))
        again = parse_anchors(adapter, sorted(anchors)[:1])
    key = sorted(again)[0]
    checks.append({"control": "anchor_parse_is_deterministic",
                   "replication": harness.carrier_replication(anchors[key], again[key]),
                   "ok": bool(torch.equal(anchors[key], again[key]))})
    began = time.monotonic()
    payload = build_training_data(chosen, universe, anchors)
    timings["data_seconds_per_shot"] = (time.monotonic() - began) / len(chosen)
    reference = s99.window_data(SLOT_FIX, "E", chosen)
    checks.append({"control": "windows_equal_issue_99_window_data",
                   "windows": len(payload["z"]),
                   "ok": all(torch.equal(payload[k], reference[k]) for k in reference)})
    checks.append({"control": "ranking_cells_are_fit_mixed_cells",
                   "cells": len(payload["cell_members"]),
                   "ok": (not set(payload["cell_members"]) & set(HELD_OUT)
                          and all(((v == 1).any() and (v == 0).any()) for v in payload["cell_verdict"]))})
    projection, memory = {}, {}
    for family in FAMILIES:
        data = TrainData(payload, family)
        # masked #77 loss equals the reference loss on a real batch (member 0, eager)
        from scripts.run_issue_74_matched_dynamics import loss_for
        torch.manual_seed(SEEDS[0])
        single = s99.new_predictor(family).to(DEVICE)
        eager = dd.Ensemble([single], learning_rate=1e-4, weight_decay=1e-4, grad_clip=1.0, compile=False)
        pools = [data.pools(chosen[0]["member"])]
        batch = data.windows(pools, [torch.Generator().manual_seed(1)], 16)
        worst = 0.0
        for step in range(9):
            pair = step_pair(family, step)
            ours = dd.local_loss(eager, batch, pair)
            theirs = loss_for(single, {k: v[0] for k, v in batch.items()}, pair)
            worst = max(worst, abs(float(ours[0]) - float(theirs)) / max(abs(float(theirs)), 1e-8))
        checks.append({"control": f"{family}_masked_loss_equals_issue_77_loss", "max_relative_delta": worst,
                       "ok": worst <= 1e-4})
        compiled = dd.Ensemble([single], learning_rate=1e-4, weight_decay=1e-4, grad_clip=1.0, compile=True)
        z = batch["z"][:, :, 0]
        delta = max(float((compiled.transition(p, z, batch["action"]) - eager.transition(p, z, batch["action"]))
                          .abs().max()) for p in family_pairs(family))
        checks.append({"control": f"{family}_compiled_transition_equals_eager", "max_abs_delta": delta,
                       "ok": delta <= 1e-4})
        del compiled, eager, single
        extra = full_data_gb(shots, family) - payload_gb(payload, family)
        for arm in ARMS:
            for size in (2, 4):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                members = [(chosen[0]["member"], SEEDS[i % 3]) for i in range(size)]
                # 9 warm-up updates compile every pair; then SMOKE_STEPS timed updates, all at the
                # most expensive setting (horizon 225, ranking term on)
                report, _ = train_group(scratch, arm, family, 0, members, data, steps=9 + SMOKE_STEPS, save=False,
                                        smoke_full=True, timing_after=9)
                memory[(arm, family, size)] = report["peak_memory_gb"]
                timings[f"{arm}_{family}_m{size}_seconds_per_update"] = report["timed_seconds_per_update"]
                checks.append({"control": f"{arm}_{family}_m{size}_trains_finite",
                               "retired": sum(not m["complete"] for m in report["members"]),
                               "final_loss": report["trace"][-1]["loss"],
                               "ok": all(m["complete"] for m in report["members"])})
            slope = max((memory[(arm, family, 4)] - memory[(arm, family, 2)]) / 2, 1e-3)
            base = memory[(arm, family, 2)] - 2 * slope + extra
            size = max(1, min(MAX_GROUP, int((MEMORY_BUDGET_GB - base) // slope)))
            groups = math.ceil(len(ensemble_members()) / size)
            size = math.ceil(len(ensemble_members()) / groups)
            t2 = timings[f"{arm}_{family}_m2_seconds_per_update"]
            t4 = timings[f"{arm}_{family}_m4_seconds_per_update"]
            per_update = t2 + (t4 - t2) / 2 * (size - 2)
            projection[f"{arm}_{family}"] = {"group_size": size, "groups": groups, "gb_per_member": slope,
                                             "base_gb_with_full_data": base, "projected_peak_gb": base + slope * size,
                                             "seconds_per_update_at_group_size": per_update,
                                             "training_seconds_upper": groups * recipe()["steps"] * per_update}
        del data
        torch.cuda.empty_cache()
    # scoring mechanics on a fit lineage (rows only; no AUC, no verdict joined)
    samples = rollout_samples_all()
    costs = harness.EndpointCosts(wlc.load_objective())
    objective = diag.base.TaskObjective.from_vocabulary(s99.vocabulary())
    member = next(m for m in fitted if m in universe[PRIMARY])
    with harness.deterministic_scoring() as policy:
        contexts = parse_contexts(adapter, samples, [member])
        model = s99.load_predictor(SLOT_FIX, "E", SEEDS[0], "hybrid")
        began = time.monotonic()
        first = endpoint_block(model, "hybrid", anchors[member], universe[PRIMARY][member]["inventory"], costs)
        curves = lineage_rollout(model, "hybrid", contexts[member], samples[member], objective)
        timings["score_seconds_per_hybrid_record"] = time.monotonic() - began
        second = endpoint_block(model, "hybrid", anchors[member], universe[PRIMARY][member]["inventory"], costs)
    replication = [harness.endpoint_replication(first[r]["rows"], second[r]["rows"]) for r in first]
    checks.append({"control": "scoring_is_bit_reproducible_in_process", "policy": policy,
                   "worst": max(max(r["max_abs_delta"].values()) for r in replication),
                   "ok": all(r["pass"] and max(r["max_abs_delta"].values()) == 0.0 for r in replication)})
    checks.append({"control": "rollout_error_mechanics", "requests": len(curves),
                   "branches": len(samples[member]), "ok": len(curves) == len(family_requests("hybrid"))})
    checks.append({"control": "held_out_isolation", "training_members": sorted({s["member"] for s in shots}),
                   "ok": not ({s["member"] for s in shots} & set(HELD_OUT))})
    group = {key: value["group_size"] for key, value in projection.items()}
    training_seconds = sum(value["training_seconds_upper"] for value in projection.values())
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timings": timings, "memory_gb": {f"{a}_{f}_m{s}": v for (a, f, s), v in memory.items()},
                "group_sizes": group, "projection": {"training_seconds": training_seconds, **projection},
                "reading": "infrastructure assertions only; nothing kept; no statistic; no held-out access",
                "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected training {training_seconds / 3600:.2f} h; "
        f"group sizes {group}")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


def _bytes(windows, positions, long_windows, family):
    per_window = (WINDOW_POSITIONS * 236 * 4 + 5 * 4 + 8
                  + (WINDOW_POSITIONS * (18 * 18 * 2 * 2 + 2 * 2) if family == "hybrid" else 0))
    return (windows * per_window + positions * (236 * 4) + long_windows * 16) / 1e9


def payload_gb(payload, family):
    return _bytes(len(payload["z"]), len(payload["carriers"]), len(payload["long_windows"]), family)


def full_data_gb(shots, family):
    """GPU bytes of the full training tensors, from the E99 carrier lengths (outcome-free)."""
    windows = positions = long_windows = 0
    for shot in shots:
        last = len(torch.load(s99.carrier_path(SLOT_FIX, "E", shot["key"]), weights_only=True)["carriers"]) - 1
        windows += sum(1 for start in s99.WINDOW_STARTS if start < last)
        long_windows += sum(1 for start in LONG["starts"] if start < last)
        positions += last + 1
    return _bytes(windows, positions, long_windows, family)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_99_fix_plan": SLOT_FIX / "plan.json",
        "issue_99_fix_encoder": SLOT_FIX / "encoder" / "encoder.pt",
        **{f"issue_99_E_{family}_seed{seed}": s99.predictor_path(SLOT_FIX, "E", seed, family)
           for seed in SEEDS for family in FAMILIES},
        "issue_99_attribution_plan": ATTRIBUTION / "plan.json",
        "issue_100_plan": READOUT / "plan.json",
        "issue_100_v2_plan": ROOT / ".local-artifacts/issue-100-guard-revision-v2/plan.json",
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "issue_77_n1_plan": CAMPAIGN / "plan.json",
        "issue_77_diagnostic_plan": diag.OUTPUT / "plan.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "model_module": ROOT / MODEL_MODULE,
        "smoke": Path(output) / "smoke.json",
    }
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence, shots, universe):
    samples = rollout_samples_all()
    group_sizes = smoke_evidence["group_sizes"]
    cells = fit_cells(universe)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 112,
        "role": ("development protocol under the #87 versioned-protocol policy: leave-one-lineage-out results are "
                 "EXPLORATORY; the recipe freeze is the --select record; held-out scored once after it"),
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True, "issue_64_authorized": False,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "model_module": MODEL_MODULE, "engine_seconds": 0,
        "question": ("on E99 carriers, does long-horizon recursive supervision to the 225-frame decision endpoint "
                     "and/or a decision-contrastive ranking loss make the request rollout order a left-out fit "
                     "lineage's candidates better than the #77 recipe, and which (arm, request) goes to Gate B?"),
        "disclosure": ("Known before freeze: every #99, #100, #109 and #96 table, including the #99 E held-out "
                       "stage-c AUCs (E best request C-F-1 0.7259; E - R0 0.0025) and the #100 held-out rollout "
                       "errors. Held-out lineages 007/008/014/016 are exposed; nothing here reads a held-out "
                       "record before the --select freeze. Fit-lineage inventory verdicts are training labels of "
                       "the ranking loss and the leave-one-lineage-out validation targets; no fold model is "
                       "trained on its validation lineage. The smoke computes no statistic."),
        "encoder": {"identity": "E99: the frozen #99 spatial slot parser (issue-99-slot-encoder-fix-v1 arm E)",
                    "carriers": "the #99 fix E carriers of the 745 fit shots (positions 0..min(last, 285))",
                    "anchors": "E99 batch-1 parse of each member's frozen sealed anchor (p94.frozen_anchor)"},
        "arms": ARM_DEFINITIONS,
        "families": list(FAMILIES), "requests": list(REQUESTS), "seeds": list(SEEDS),
        "folds": {"fit_lineages": fit_lineages(), "folds": list(folds()),
                  "rule": ("fold L trains on every fit shot, long window and ranking cell whose lineage is not L; "
                           "fold 'all' trains on all 12 fit lineages (the Gate-B checkpoints)"),
                  "members_per_ensemble": len(ensemble_members())},
        "training": {
            "recipe": recipe(), "initialization": "torch.manual_seed(seed); s99.new_predictor(family) (from scratch)",
            "schedule": "PAIRS[step % 9]; the continuous family uses the same horizon",
            "local_loss": ("#74/#77 loss (run_issue_74_matched_dynamics.loss_for) in masked form "
                           "(decision_dynamics.local_loss) on the #99 windows: starts 0..225 step 15, 60 transitions, "
                           "terminal padding masked; batch of 64 windows sampled uniformly with replacement from the "
                           "member's pool by torch.Generator().manual_seed(seed)"),
            "long_horizon": {**LONG, "arms": ["L", "LC"],
                             "definition": ("windows at starts 0, 15, 30, 45, 60 of every fit shot (start below the "
                                            "shot's last position); the step's pair is unrolled horizon // delta "
                                            "transitions; loss = mean over visited positions of row-mean carrier MSE "
                                            "+ 0.01 x the #77 carrier-bound penalty; horizon = min(225, 60 + 15 x "
                                            "floor(update / 500))")},
            "ranking": {**RANK, "arms": ["C", "LC"],
                        "definition": ("per update 2 mixed fit-lineage cells (member x inventory, sampled uniformly "
                                       "with replacement from the member's pool); every verdict-bearing candidate "
                                       "action is rolled out with the step's pair from the member's E99 anchor to "
                                       "225; loss = mean over cells of the mean over (success, failure) pairs of "
                                       "softplus((cost_success - cost_failure) / 0.1); weight 0.1 x ramp, ramp 0 "
                                       "before update 1000 and linear to 1 by update 2000"),
                        "cells": [{"member": m, "inventory": i} for m, i in cells],
                        "verdict_source": ("fit-lineage inventory executions only (#93 grid/offset, #94 power, "
                                           "#87/#89 angle oracle verdicts); no held-out verdict is read")},
            "update": ("one AdamW over stacked member parameters (elementwise, so per member); per-member gradient "
                       "clipping to grad_clip; a member with a nonfinite loss or gradient norm is retired at that "
                       "update (typed terminal failure, never scored)"),
            "execution": ("members of one (arm, family) are trained in lock-step groups by a vmapped, "
                          "torch.compile'd transition (training process only; autotuning allowed)"),
            "group_sizes": group_sizes,
            "matched": "every arm: same architectures and parameter counts, same 9000 updates of 64 #77 windows",
        },
        "diagnostics": {
            "models": "the frozen #99 E hybrid and continuous predictors, 3 seeds (zero training)",
            "cohort": "fit lineages only (in-sample for the #99 E one-step training windows)",
            "action_sensitivity": ("per fit member x inventory cell and request: SD over executed candidates of the "
                                   "predicted endpoint pig presence, pig displacement and tie-free cost against the "
                                   "engine endpoint SD (#99 attribution engine-projected carriers at 0 and 225); SD "
                                   "ratio and Spearman correlation; the E99 parsed endpoint as the perception "
                                   "reference"),
            "error_growth": ("per request: the #100 engine-referenced rollout error (engine_position_mse) and the "
                             "own-parse position MSE at t = 15, 30, 60, 150, 225 on every fit lineage's N1 samples "
                             "(#77 diagnostic sample rule, #100 context rule)"),
            "variance": ("grid and offset (identical action sets across members): two-way decomposition of the "
                         "endpoint quantity over (anchor member, action) on members whose every candidate was "
                         "executed; anchor, action and residual shares; predicted, engine and parsed"),
        },
        "validation": {
            "lolo": ("fold L model on lineage L: every inventory cell of L (all items) x the family's requests "
                     "(m99.rollout_endpoints from L's E99 anchor to 225; EndpointCosts rows) and L's N1 rollout "
                     "error; AUC per mixed cell under each cost (m99.view)"),
            "selection_metric": ("S(arm, request) = mean of the grid and offset leave-one-lineage-out AUC means "
                                 "(member-clustered bootstrap mean over (lineage, seed) cells) under the tie-free "
                                 "cost"),
            "selection_rule": ("the (arm, request) with the largest S; ties broken by arm order B, L, C, LC then "
                               "request order; the frozen recipe is that arm's training recipe, the Gate-B request "
                               "is that request, the checkpoints are its fold-'all' members (3 seeds)"),
            "heldout": ("after the selection freeze, once: every arm's fold-'all' members and the frozen #99 E "
                        "predictors (reference) on the held-out lineages, every inventory and request, plus the "
                        "#100 held-out rollout error; DESCRIPTIVE, no token"),
        },
        "decision_rule": {
            "Q1_long_horizon": (f"request-mean paired (L - B) leave-one-lineage-out grid AUC, tie-free: supported "
                                f"iff mean >= {DELTA} and lower > 0; not_supported_by_this_experiment iff upper < "
                                f"{DELTA}; otherwise {STOP_TOKEN}; (LC - C) reported alongside"),
            "Q2_contrastive": ("request-mean paired (C - B) leave-one-lineage-out grid AUC, tie-free, same rule; "
                               "(LC - L) reported alongside"),
            "Q3_selected_target": (f"the selected (arm, request): supported iff its leave-one-lineage-out grid AUC "
                                   f">= {TARGET_AUC} with lower bound > {TARGET_LOWER}; "
                                   f"not_supported_by_this_experiment otherwise; {STOP_TOKEN} if a guard fails"),
            "multiplicity": ("selection scans 4 arms x 12 requests, so the selected unit's leave-one-lineage-out "
                             "estimate is optimistic; Gate B on the #104 sealed split is the unbiased test"),
            "labels": "leave-one-lineage-out: EXPLORATORY; held-out: DESCRIPTIVE",
        },
        "guards": {
            "G1": "no training window, long window or ranking cell of a held-out lineage; no fold pool holds its lineage",
            "G2": "every member of every arm reached 9000 updates without retirement",
            "G3": ("every scoring record carries the #111 SCORING_POLICY; --validate re-scores one diagnostic, one "
                   "leave-one-lineage-out and one held-out record and endpoint_replication passes; stored anchors "
                   "and contexts pass carrier_replication against a fresh parse"),
            "G4": f"nonfinite share <= {NONFINITE_SHARE_MAX} on leave-one-lineage-out grid cells, every arm and request",
            "caps": f"GPU seconds <= {GPU_CAP_SECONDS}, wall seconds <= {WALL_CAP_SECONDS}",
        },
        "universe": {"fit_shots": len(shots), "ranking_cells": len(cells),
                     "lolo_mixed": {i: sorted(m for m, e in universe[i].items() if m in fit_lineages()
                                              and t96.mixed(e)) for i in INVENTORIES},
                     "n1_samples": {m: len(v) for m, v in samples.items()},
                     "lolo_records": len(ARMS) * len(FAMILIES) * len(fit_lineages()) * len(SEEDS),
                     "heldout_records": (len(ARMS) + 1) * len(FAMILIES) * len(SEEDS)},
        "prohibited": ["reading a held-out record before the --select freeze", "changing any arm, loss, weight, "
                       "window or recipe after training starts", "retrying typed failures",
                       "re-opening any prior disposition (#15-#100, #109, #111)"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("decision-only ranking of retained engine-verdicted executions; fit lineages by "
                           "leave-one-lineage-out, the 4 exposed held-out lineages once, descriptively; E99 encoder "
                           "fixed; no engine access, rendering or capture; no closed-loop or competence claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-112 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    blob = json_text({k: v for k, v in plan.items() if k not in ("smoke_evidence", "disclosure")})
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
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
    for name in ("data", "models", "training", "records"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    sources, _, universe = load_universe()
    shots = s99.training_shots(sources, universe)
    plan = frozen_plan(output, utc_now(), evidence, shots, universe)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {len(shots)} fit shots, {plan['universe']['ranking_cells']} ranking cells, "
        f"{plan['universe']['lolo_records']} leave-one-lineage-out records scheduled")
    return 0


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "phases": {}, "gpu_seconds_elapsed": 0.0,
            "wall_seconds_elapsed": 0.0}


def mark(output, phase, wall, gpu, **extra):
    ledger = load_ledger(output)
    block = ledger["phases"].get(phase, {"wall_seconds": 0.0, "gpu_seconds": 0.0})
    ledger["phases"][phase] = {**block, "wall_seconds": block["wall_seconds"] + wall,
                               "gpu_seconds": block["gpu_seconds"] + gpu, "last_at": utc_now(), **extra}
    ledger["gpu_seconds_elapsed"] += gpu
    ledger["wall_seconds_elapsed"] += wall
    write_json(Path(output) / "ledger.json", ledger)
    if ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS or ledger["wall_seconds_elapsed"] > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


# ---------------------------------------------------------------------------
# modes: data, diagnose, train, score-lolo, select, score-heldout
# ---------------------------------------------------------------------------

def data(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    sources, _, universe = load_universe()
    shots = s99.training_shots(sources, universe)
    samples = rollout_samples_all()
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(SLOT_FIX, "E")
        if not anchors_path(output, "fit").is_file():
            atomic_torch(anchors_path(output, "fit"), {"scoring_policy": policy,
                                                      "anchors": parse_anchors(adapter, fit_lineages_in(universe))})
        if not contexts_path(output, "fit").is_file():
            atomic_torch(contexts_path(output, "fit"), {"scoring_policy": policy,
                                                       "contexts": parse_contexts(adapter, samples, fit_lineages())})
    if not data_path(output).is_file():
        anchors = torch.load(anchors_path(output, "fit"), weights_only=True)["anchors"]
        atomic_torch(data_path(output), build_training_data(shots, universe, anchors))
    mark(output, "data", time.monotonic() - began, 0.0)
    log(f"training data ready ({time.monotonic() - began:.0f}s)")
    return 0


def fit_lineages_in(universe):
    fitted = set(fit_lineages())
    return sorted({m for inventory in INVENTORIES for m in universe[inventory] if m in fitted})


def load_training_data(output):
    payload = torch.load(data_path(output), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_DATA or payload.get("plan_identity") != IDENTITY:
        raise ValueError("training data binding differs")
    return payload


def diagnose(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    _, _, universe = load_universe()
    samples = rollout_samples_all()
    anchors = torch.load(anchors_path(output, "fit"), weights_only=True)["anchors"]
    contexts = torch.load(contexts_path(output, "fit"), weights_only=True)["contexts"]
    costs = harness.EndpointCosts(wlc.load_objective())
    objective = diag.base.TaskObjective.from_vocabulary(s99.vocabulary())
    with harness.deterministic_scoring() as policy:
        if not endpoints_path(output).is_file():
            write_json(endpoints_path(output), diag_endpoints(universe, costs, policy))
        for family in FAMILIES:
            for seed in SEEDS:
                if not diag_path(output, family, seed).is_file():
                    write_json(diag_path(output, family, seed),
                               diag_record(universe, costs, objective, anchors, contexts, samples, family, seed,
                                           policy))
                    log(f"diagnostic {family} seed {seed} written")
    wall = time.monotonic() - began
    mark(output, "diagnose", wall, wall)
    return 0


def train(output):
    output = Path(output)
    plan = load_plan(output)
    torch.backends.cudnn.benchmark = True
    payload = load_training_data(output)
    for family in FAMILIES:
        data_on_gpu = None
        for arm in ARMS:
            size = plan["training"]["group_sizes"][f"{arm}_{family}"]
            for index, members in enumerate(member_groups(size)):
                if group_path(output, arm, family, index).is_file():
                    continue
                data_on_gpu = data_on_gpu or TrainData(payload, family)
                began = time.monotonic()
                report, ensemble = train_group(output, arm, family, index, members, data_on_gpu)
                del ensemble
                torch.cuda.empty_cache()
                mark(output, f"train_{arm}_{family}_group{index}", time.monotonic() - began, report["gpu_seconds"],
                     members=len(members), retired=sum(not m["complete"] for m in report["members"]))
                log(f"trained {arm}/{family}/group{index}: {report['gpu_seconds']:.0f}s, retired "
                    f"{sum(not m['complete'] for m in report['members'])}")
        del data_on_gpu
        torch.cuda.empty_cache()
    return 0


def training_complete(output, plan):
    for family in FAMILIES:
        for arm in ARMS:
            size = plan["training"]["group_sizes"][f"{arm}_{family}"]
            for index in range(len(member_groups(size))):
                if not group_path(output, arm, family, index).is_file():
                    return False
    return True


def score_lolo(output):
    output = Path(output)
    plan = load_plan(output)
    if not training_complete(output, plan):
        raise ValueError("training incomplete; leave-one-lineage-out scoring needs every member")
    began = time.monotonic()
    _, _, universe = load_universe()
    samples = rollout_samples_all()
    anchors = torch.load(anchors_path(output, "fit"), weights_only=True)["anchors"]
    contexts = torch.load(contexts_path(output, "fit"), weights_only=True)["contexts"]
    costs = harness.EndpointCosts(wlc.load_objective())
    objective = diag.base.TaskObjective.from_vocabulary(s99.vocabulary())
    cells = [(arm, family, fold, seed) for arm in ARMS for family in FAMILIES for fold in fit_lineages()
             for seed in SEEDS]
    pending = [c for c in cells if not lolo_path(output, *c).is_file()]
    progress = Progress("leave-one-lineage-out records", len(pending))
    with harness.deterministic_scoring() as policy:
        for arm, family, fold, seed in pending:
            record = lolo_record(output, universe, costs, objective, anchors, contexts, samples, arm, family, fold,
                                 seed, policy)
            write_json(lolo_path(output, arm, family, fold, seed), record)
            progress.step(12)
    wall = time.monotonic() - began
    mark(output, "score_lolo", wall, wall, records=len(cells))
    return 0


def select(output):
    output = Path(output)
    plan = load_plan(output)
    path = output / "selection.json"
    if path.is_file():
        selection = read_json(path)
        if selection != selection_record(output, plan, selection["frozen_at"]):
            raise ValueError("selection.json differs from the frozen rule applied to the retained records")
        log("existing selection validated")
        return 0
    if any(output.glob("records/heldout--*.json")):
        raise ValueError("held-out records exist before the selection freeze")
    selection = selection_record(output, plan, utc_now())
    write_json(path, selection)
    write_json(output / "gate_b_handoff.json", handoff(output, plan, selection))
    log(f"recipe frozen: arm {selection['selected']['arm']}, request {selection['selected']['request']} "
        f"(S = {selection['selected']['score']:.4f})")
    return 0


def score_heldout(output):
    output = Path(output)
    plan = load_plan(output)
    selection = read_json(output / "selection.json")
    if selection.get("schema") != SCHEMA_SELECTION:
        raise ValueError("the selection freeze is missing; held-out scoring comes after it")
    if selection != selection_record(output, plan, selection["frozen_at"]):
        raise ValueError("selection.json differs from the frozen rule; refusing held-out scoring")
    began = time.monotonic()
    _, _, universe = load_universe()
    samples = rollout_samples_all()
    costs = harness.EndpointCosts(wlc.load_objective())
    objective = diag.base.TaskObjective.from_vocabulary(s99.vocabulary())
    held = sorted({m for inventory in INVENTORIES for m in universe[inventory] if m in HELD_OUT})
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(SLOT_FIX, "E")
        if not anchors_path(output, "heldout").is_file():
            atomic_torch(anchors_path(output, "heldout"), {"scoring_policy": policy,
                                                          "anchors": parse_anchors(adapter, held)})
        if not contexts_path(output, "heldout").is_file():
            atomic_torch(contexts_path(output, "heldout"), {"scoring_policy": policy,
                                                           "contexts": parse_contexts(adapter, samples, HELD_OUT)})
        anchors = torch.load(anchors_path(output, "heldout"), weights_only=True)["anchors"]
        contexts = torch.load(contexts_path(output, "heldout"), weights_only=True)["contexts"]
        cells = [(arm, family, seed) for arm in (*ARMS, REFERENCE) for family in FAMILIES for seed in SEEDS]
        pending = [c for c in cells if not heldout_path(output, *c).is_file()]
        progress = Progress("held-out records", len(pending))
        for arm, family, seed in pending:
            record = heldout_record(output, universe, costs, objective, anchors, contexts, samples, arm, family,
                                    seed, policy)
            write_json(heldout_path(output, arm, family, seed), record)
            progress.step(3)
    wall = time.monotonic() - began
    mark(output, "score_heldout", wall, wall, records=len(cells))
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def cost_view(entry, rows, cost):
    if rows is None:
        return {"auc": None, "typed_failure": True, "nonfinite": 0, "verdict_slots": len(entry["verdicts"]),
                "all_tied": False, "pairs": 0, "tied_pairs": 0, "top1_hit": None, "chance_top1": None,
                "resolved": 0}
    return m99.view(entry, rows, cost)


def lolo_rows(output, universe, inventory):
    """One row per (fold lineage with a mixed cell, seed); views keyed (cost, arm, request)."""
    rows = []
    for fold in fit_lineages():
        entry = universe[inventory].get(fold)
        if entry is None or not t96.mixed(entry):
            continue
        for seed in SEEDS:
            views = {}
            for arm in ARMS:
                for family in FAMILIES:
                    record = read_record(lolo_path(output, arm, family, fold, seed), SCHEMA_LOLO)
                    block = record["inventories"].get(inventory)
                    for request in family_requests(family):
                        rows_ = None if record["typed_failure"] else block[request]["rows"]
                        for cost in COSTS:
                            views[(cost, arm, request)] = cost_view(entry, rows_, cost)
            rows.append({"member": fold, "seed": seed, "views": views})
    return rows


def heldout_rows(output, universe, inventory):
    rows = []
    for member in sorted(m for m, e in universe[inventory].items() if m in HELD_OUT and t96.mixed(e)):
        entry = universe[inventory][member]
        for seed in SEEDS:
            views = {}
            for arm in (*ARMS, REFERENCE):
                for family in FAMILIES:
                    record = read_record(heldout_path(output, arm, family, seed), SCHEMA_HELDOUT)
                    for request in family_requests(family):
                        rows_ = None if record["typed_failure"] else record["inventories"][inventory][member][
                            request]["rows"]
                        for cost in COSTS:
                            views[(cost, arm, request)] = cost_view(entry, rows_, cost)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def ranker_block(rows, key):
    views = [r["views"][key] for r in rows]
    pairs = sum(v["pairs"] for v in views)
    slots = sum(v["verdict_slots"] for v in views)
    return {"auc": m99.block_of(rows, key),
            "cells": len(views), "typed_failures": sum(1 for v in views if v.get("typed_failure")),
            "tied_cells": sum(v["all_tied"] for v in views),
            "tied_pair_share": sum(v["tied_pairs"] for v in views) / pairs if pairs else None,
            "top1_hits": sum(1 for v in views if v["top1_hit"]),
            "nonfinite_share": sum(v["nonfinite"] for v in views) / slots if slots else None}


def contrast(rows, cost, first, second, requests=REQUESTS):
    n = len(requests)
    out = {"request_mean": m99.paired(rows, [(1 / n, (cost, first, r)) for r in requests]
                                      + [(-1 / n, (cost, second, r)) for r in requests])}
    for request in requests:
        out[request] = m99.paired(rows, [(1, (cost, first, request)), (-1, (cost, second, request))])
    return out


CONTRASTS = (("L", "B"), ("LC", "C"), ("C", "B"), ("LC", "L"), ("LC", "B"))


def inventory_tables(rows, arms):
    return {"members": sorted({r["member"] for r in rows}), "cells": len(rows),
            "rankers": {cost: {f"{arm}:{request}": ranker_block(rows, (cost, arm, request))
                               for arm in arms for request in REQUESTS} for cost in COSTS},
            "contrasts": {cost: {f"{a}-{b}": contrast(rows, cost, a, b) for a, b in CONTRASTS} for cost in COSTS}}


def selection_scores(lolo):
    scores = []
    for arm in ARMS:
        for request in REQUESTS:
            blocks = [lolo[i]["rankers"][PRIMARY_COST][f"{arm}:{request}"]["auc"] for i in SELECTION_INVENTORIES]
            score = None if any(b is None for b in blocks) else float(np.mean([b["mean"] for b in blocks]))
            scores.append({"arm": arm, "request": request, "score": score,
                           **{f"{i}_auc": b for i, b in zip(SELECTION_INVENTORIES, blocks)}})
    return scores


def select_unit(scores):
    best = None
    for row in scores:  # frozen order: arms B, L, C, LC, then requests; strict > keeps the earlier unit on ties
        if row["score"] is not None and (best is None or row["score"] > best["score"]):
            best = row
    if best is None:
        raise ValueError("no (arm, request) has a defined selection score")
    return best


def lolo_tables(output, universe):
    return {inventory: inventory_tables(lolo_rows(output, universe, inventory), ARMS) for inventory in INVENTORIES}


def member_hashes(output, arm, family):
    return {str(seed): sha256_of(member_path(output, arm, family, "all", seed)) for seed in SEEDS}


def selection_record(output, plan, frozen_at):
    _, _, universe = load_universe()
    lolo = lolo_tables(output, universe)
    scores = selection_scores(lolo)
    best = select_unit(scores)
    family = t96.arm_family(best["request"])
    lolo_files = sorted(output.glob("records/lolo--*.json"))
    return {"schema": SCHEMA_SELECTION, "plan_identity": IDENTITY, "frozen_at": frozen_at,
            "rule": plan["validation"]["selection_rule"], "metric": plan["validation"]["selection_metric"],
            "scores": scores, "selected": {"arm": best["arm"], "request": best["request"], "family": family,
                                           "score": best["score"]},
            "recipe": {"arm": best["arm"], "definition": ARM_DEFINITIONS[best["arm"]],
                       "training": plan["training"]},
            "checkpoints": {"arm": best["arm"], "family": family, "fold": "all",
                            "sha256": member_hashes(output, best["arm"], family)},
            "lolo_records_sha256": sha256_text("".join(sha256_of(p) for p in lolo_files)),
            "lolo_records": len(lolo_files)}


def sha256_text(text):
    from hashlib import sha256
    return f"sha256:{sha256(text.encode()).hexdigest()}"


def handoff(output, plan, selection):
    chosen = selection["selected"]
    return {"schema": SCHEMA_HANDOFF, "plan_identity": IDENTITY, "frozen_at": selection["frozen_at"],
            "gate": ("Gate B (#108): the frozen recipe on the #104 sealed held-out split; grid AUC >= "
                     f"{TARGET_AUC} with lower bound > {TARGET_LOWER} under the tie-free cost"),
            "encoder": {"identity": "E99", "checkpoint": str((SLOT_FIX / "encoder" / "encoder.pt").relative_to(ROOT)),
                        "sha256": sha256_of(SLOT_FIX / "encoder" / "encoder.pt"),
                        "load": "scripts.run_slot_encoder_fix.load_adapter(SLOT_FIX, 'E')"},
            "recipe": selection["recipe"], "request": chosen["request"], "family": chosen["family"],
            "checkpoints": [{"seed": seed, "path": str(member_path(output, chosen["arm"], chosen["family"], "all",
                                                                   seed).relative_to(ROOT)),
                             "sha256": selection["checkpoints"]["sha256"][str(seed)]} for seed in SEEDS],
            "load_checkpoint": ("scripts.run_decision_relevant_dynamics.load_member(OUTPUT, arm, family, 'all', "
                                "seed)"),
            "retrain": ("to retrain on a new fit split, run train_group with the frozen recipe (plan.json training "
                        "block, the selected arm) on that split's E99 carriers, windows and fit-lineage cells"),
            "scoring": ("anchor carrier = E99 batch-1 parse of the member's sealed anchor; per candidate "
                        "m99.rollout_endpoints(model, None, z0, actions, t96.arm_spec(request)); cost = "
                        "scoring_harness.EndpointCosts(...).row(...)['tie_free']; cell AUC = m99.view on mixed "
                        "cells; member-clustered bootstrap (10000 draws, PCG64 7201); every scoring phase inside "
                        "scoring_harness.deterministic_scoring() in a process that did not train"),
            "seed_aggregation": "cells are (member, seed); the bootstrap resamples members",
            "selection_evidence": {"score": chosen["score"], "lolo_records_sha256": selection["lolo_records_sha256"]}}


def rollout_block(per_lineage):
    """per_lineage: {lineage: [value per seed or None]} -> seed-mean per lineage, bootstrap over lineages."""
    values, clusters = [], []
    for lineage, seeds in sorted(per_lineage.items()):
        finite = [v for v in seeds if v is not None]
        if finite and len(finite) == len(seeds):
            values.append(float(np.mean(finite)))
            clusters.append(lineage)
    return bootstrap(values, clusters)


def branch_mean(branches, t, metric):
    values = [b["errors"][str(t)][metric] for b in branches
              if b["errors"].get(str(t)) is not None and b["errors"][str(t)][metric] is not None]
    nonfinite = sum(1 for b in branches if b["errors"].get(str(t)) is None)
    return (float(np.mean(values)) if values else None), nonfinite


METRICS = ("engine_position_mse", "position_mse")


def rollout_tables(per_record):
    """per_record: {(arm, request): {lineage: [rollout branches per seed or None]}}."""
    table, nonfinite = {}, {}
    for (arm, request), lineages in per_record.items():
        for t in TIMES:
            for metric in METRICS:
                per_lineage, bad = {}, 0
                for lineage, seeds in lineages.items():
                    per_lineage[lineage] = []
                    for branches in seeds:
                        if branches is None:
                            per_lineage[lineage].append(None)
                            continue
                        value, count = branch_mean(branches, t, metric)
                        per_lineage[lineage].append(value)
                        bad += count
                table[f"{arm}:{request}:{t}:{metric}"] = rollout_block(per_lineage)
                nonfinite[f"{arm}:{request}:{t}"] = bad
    return table, nonfinite


def rollout_contrasts(per_record, arms, baseline):
    out = {}
    for arm in arms:
        if arm == baseline:
            continue
        for request in REQUESTS:
            for metric in METRICS:
                values, clusters = [], []
                a, b = per_record[(arm, request)], per_record[(baseline, request)]
                for lineage in sorted(a):
                    pa = [None if x is None else branch_mean(x, ENDPOINT, metric)[0] for x in a[lineage]]
                    pb = [None if x is None else branch_mean(x, ENDPOINT, metric)[0] for x in b[lineage]]
                    if all(v is not None for v in pa + pb):
                        values.append(float(np.mean(pa) - np.mean(pb)))
                        clusters.append(lineage)
                out[f"{arm}-{baseline}:{request}:{metric}"] = bootstrap(values, clusters)
    return out


def lolo_rollout(output):
    per = {}
    for arm in ARMS:
        for family in FAMILIES:
            for request in family_requests(family):
                per[(arm, request)] = {}
            for fold in fit_lineages():
                seeds = [read_record(lolo_path(output, arm, family, fold, seed), SCHEMA_LOLO) for seed in SEEDS]
                if any(r["rollout"] is None and r["typed_failure"] is None for r in seeds):
                    continue  # a lineage without N1 samples
                for request in family_requests(family):
                    per[(arm, request)][fold] = [None if r["typed_failure"] else r["rollout"][request] for r in seeds]
    return per


def heldout_rollout(output):
    per = {}
    for arm in (*ARMS, REFERENCE):
        for family in FAMILIES:
            records = [read_record(heldout_path(output, arm, family, seed), SCHEMA_HELDOUT) for seed in SEEDS]
            lineages = sorted({m for r in records for m in r["rollout"]})
            for request in family_requests(family):
                per[(arm, request)] = {m: [None if r["typed_failure"] else r["rollout"][m][request] for r in records]
                                       for m in lineages}
    return per


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


QUANTITIES = ("pig_presence", "pig_displacement", PRIMARY_COST)


def shares(matrix):
    """Two-way decomposition (rows = anchors, columns = actions) -> anchor/action/residual shares."""
    x = np.asarray(matrix, dtype=float)
    grand = x.mean()
    total = ((x - grand) ** 2).sum()
    if total <= 0:
        return None
    anchor = x.shape[1] * ((x.mean(1) - grand) ** 2).sum()
    action = x.shape[0] * ((x.mean(0) - grand) ** 2).sum()
    return {"anchor": float(anchor / total), "action": float(action / total),
            "residual": float((total - anchor - action) / total)}


def by_ordinal(rows, quantity):
    """One member's row of the (anchor x action) matrix: the quantity in candidate-ordinal order
    (the grid and offset action of an ordinal is the same for every member)."""
    return [row[quantity] for row in sorted(rows, key=lambda r: r["ordinal"])]


def diagnostic_tables(output, universe):
    endpoints = read_record(endpoints_path(output), SCHEMA_ENDPOINTS)["cells"]
    records = {(f, s): read_record(diag_path(output, f, s), SCHEMA_DIAG) for f in FAMILIES for s in SEEDS}
    sensitivity, variance = {}, {}
    # engine / parsed reference variance
    for inventory in ("grid", "offset"):
        complete = [m for m, block in endpoints[inventory].items()
                    if len(block["engine"]) == len(universe[inventory][m]["inventory"])
                    and all(not r["excluded"] for r in block["engine"] + block["parsed"])]
        sizes = {len(universe[inventory][m]["inventory"]) for m in complete}
        if len(sizes) != 1:
            complete = [m for m in complete if len(universe[inventory][m]["inventory"]) == max(sizes)]
        variance[inventory] = {"members": complete, "reference": {}, "predicted": {}}
        for source in ("engine", "parsed"):
            variance[inventory]["reference"][source] = {
                q: shares([by_ordinal(endpoints[inventory][m][source], q) for m in complete]) for q in QUANTITIES}
        for request in REQUESTS:
            family = t96.arm_family(request)
            per_seed = []
            for seed in SEEDS:
                rows = {m: records[(family, seed)]["cells"][inventory][m][request]["rows"] for m in complete}
                if any(r["excluded"] for m in complete for r in rows[m]):
                    per_seed.append(None)
                    continue
                per_seed.append({q: shares([by_ordinal(rows[m], q) for m in complete]) for q in QUANTITIES})
            variance[inventory]["predicted"][request] = {
                q: ({k: float(np.mean([s[q][k] for s in per_seed])) for k in ("anchor", "action", "residual")}
                    if all(s is not None and s[q] is not None for s in per_seed) else None) for q in QUANTITIES}
    # action sensitivity: per cell SD ratio and Spearman vs engine
    for request in (*REQUESTS, "parsed"):
        per_q = {q: {"sd_ratio": [], "spearman": [], "clusters_ratio": [], "clusters_rho": [], "sd_predicted": [],
                     "sd_engine": []} for q in QUANTITIES}
        for inventory in INVENTORIES:
            for member, block in endpoints[inventory].items():
                ordinals = [r["ordinal"] for r in block["engine"]]
                engine = {r["ordinal"]: r for r in block["engine"]}
                if len(ordinals) < 3:
                    continue
                if request == "parsed":
                    predicted_sets = [{r["ordinal"]: r for r in block["parsed"]}]
                else:
                    family = t96.arm_family(request)
                    predicted_sets = [{r["ordinal"]: r for r in records[(family, seed)]["cells"][inventory][member][
                        request]["rows"]} for seed in SEEDS]
                for q in QUANTITIES:
                    e = [engine[o][q] for o in ordinals]
                    if any(v is None for v in e):
                        continue
                    ratios, rhos, sds = [], [], []
                    for predicted in predicted_sets:
                        p = [predicted[o][q] for o in ordinals]
                        if any(v is None for v in p):
                            continue
                        sds.append(float(np.std(p)))
                        if np.std(e) > 0:
                            ratios.append(float(np.std(p) / np.std(e)))
                        rho = spearman(p, e)
                        if rho is not None:
                            rhos.append(rho)
                    if ratios:
                        per_q[q]["sd_ratio"].append(float(np.mean(ratios)))
                        per_q[q]["clusters_ratio"].append(member)
                    if rhos:
                        per_q[q]["spearman"].append(float(np.mean(rhos)))
                        per_q[q]["clusters_rho"].append(member)
                    if sds:
                        per_q[q]["sd_predicted"].append(float(np.mean(sds)))
                        per_q[q]["sd_engine"].append(float(np.std(e)))
        sensitivity[request] = {q: {"sd_ratio": bootstrap(v["sd_ratio"], v["clusters_ratio"]),
                                    "spearman": bootstrap(v["spearman"], v["clusters_rho"]),
                                    "mean_sd_predicted": float(np.mean(v["sd_predicted"])) if v["sd_predicted"] else None,
                                    "mean_sd_engine": float(np.mean(v["sd_engine"])) if v["sd_engine"] else None,
                                    "cells": len(v["sd_predicted"])} for q, v in per_q.items()}
    # error growth on fit-lineage N1 samples
    per = {}
    for family in FAMILIES:
        lineages = sorted(records[(family, SEEDS[0])]["rollout"])
        for request in family_requests(family):
            per[(REFERENCE, request)] = {m: [records[(family, s)]["rollout"][m][request] for s in SEEDS]
                                         for m in lineages}
    growth, nonfinite = rollout_tables(per)
    ratios = {}
    for request in REQUESTS:
        for metric in METRICS:
            a = growth[f"{REFERENCE}:{request}:60:{metric}"]
            b = growth[f"{REFERENCE}:{request}:{ENDPOINT}:{metric}"]
            ratios[f"{request}:{metric}"] = (b["mean"] / a["mean"]) if a and b and a["mean"] else None
    return {"sensitivity": sensitivity, "variance": variance, "growth": growth, "growth_ratio_225_over_60": ratios,
            "growth_nonfinite": nonfinite}


def training_summary(output, plan):
    groups, members = [], []
    for family in FAMILIES:
        for arm in ARMS:
            size = plan["training"]["group_sizes"][f"{arm}_{family}"]
            for index in range(len(member_groups(size))):
                report = read_record(group_path(output, arm, family, index), SCHEMA_GROUP)
                last = report["trace"][-1]
                groups.append({"arm": arm, "family": family, "group": index, "members": len(report["members"]),
                               "gpu_seconds": report["gpu_seconds"], "peak_memory_gb": report["peak_memory_gb"],
                               "parameters_per_member": report["parameters_per_member"]})
                for i, member in enumerate(report["members"]):
                    members.append({"arm": arm, "family": family, **member,
                                    "final_loss": last["loss"][i], "final_local": last["local"][i],
                                    "final_long": None if last["long"] is None else last["long"][i],
                                    "final_rank": None if last["rank"] is None else last["rank"][i]})
    per_arm = {}
    for arm in ARMS:
        for family in FAMILIES:
            mine = [m for m in members if m["arm"] == arm and m["family"] == family and m["complete"]]
            per_arm[f"{arm}:{family}"] = {
                "members": sum(1 for m in members if m["arm"] == arm and m["family"] == family),
                "complete": len(mine),
                "gpu_seconds": sum(g["gpu_seconds"] for g in groups if g["arm"] == arm and g["family"] == family),
                "parameters_per_member": next(g["parameters_per_member"] for g in groups
                                              if g["arm"] == arm and g["family"] == family),
                **{f"mean_final_{k}": (float(np.mean([m[f"final_{k}"] for m in mine]))
                                       if mine and mine[0][f"final_{k}"] is not None else None)
                   for k in ("loss", "local", "long", "rank")}}
    return {"groups": groups, "members": members, "per_arm": per_arm}


def token(block, guards_ok):
    if not guards_ok or block is None:
        return STOP_TOKEN
    low, high = block["interval"]
    if block["mean"] >= DELTA and low > 0:
        return "supported"
    if high < DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def meets(block):
    return bool(block and block["mean"] >= TARGET_AUC and block["interval"][0] > TARGET_LOWER)


def scoring_policies(output):
    names = sorted(output.glob("records/*.json"))
    return all(read_json(p).get("scoring_policy") == harness.SCORING_POLICY for p in names), len(names)


def compute_tables(output, plan, spot=None):
    output = Path(output)
    _, _, universe = load_universe()
    ledger = load_ledger(output)
    selection = read_json(output / "selection.json")
    diagnostics = diagnostic_tables(output, universe)
    lolo = lolo_tables(output, universe)
    heldout = {inventory: inventory_tables(heldout_rows(output, universe, inventory), (*ARMS, REFERENCE))
               for inventory in INVENTORIES}
    lolo_per, held_per = lolo_rollout(output), heldout_rollout(output)
    lolo_curves, lolo_nonfinite = rollout_tables(lolo_per)
    held_curves, held_nonfinite = rollout_tables(held_per)
    training = training_summary(output, plan)
    scores = selection_scores(lolo)
    best = select_unit(scores)
    policies_ok, scored = scoring_policies(output)
    worst_nonfinite = max(b["nonfinite_share"] or 0.0 for b in lolo[PRIMARY]["rankers"][PRIMARY_COST].values())
    heldout_times = [read_json(p)["scored_at"] for p in output.glob("records/heldout--*.json")]
    guards = {
        "G1": {"observed": {"training_members": sorted({m for p in [load_training_data_members(output)] for m in p}),
                            "held_out": list(HELD_OUT)},
               "pass": not (set(load_training_data_members(output)) & set(HELD_OUT))},
        "G2": {"observed": {"members": len(training["members"]),
                            "retired": sum(not m["complete"] for m in training["members"])},
               "pass": all(m["complete"] for m in training["members"])},
        "G3": {"observed": {"records_with_policy": scored, "all_carry_policy": policies_ok,
                            "spot_replication": spot},
               "pass": policies_ok and (spot is None or all(s["pass"] for s in spot.values()))},
        "G4": {"observed": {"worst_nonfinite_share": worst_nonfinite}, "pass": worst_nonfinite <= NONFINITE_SHARE_MAX},
        "chronology": {"observed": {"plan_frozen_at": plan["frozen_at"], "selection_frozen_at": selection["frozen_at"],
                                    "first_heldout_scored_at": min(heldout_times) if heldout_times else None},
                       "pass": bool(heldout_times) and plan["frozen_at"] < selection["frozen_at"] <= min(heldout_times)},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"], "wall_seconds": ledger["wall_seconds_elapsed"]},
                 "pass": (ledger["gpu_seconds_elapsed"] <= GPU_CAP_SECONDS
                          and ledger["wall_seconds_elapsed"] <= WALL_CAP_SECONDS)},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    grid = lolo[PRIMARY]["contrasts"][PRIMARY_COST]
    q1 = grid["L-B"]["request_mean"]["estimate"]
    q2 = grid["C-B"]["request_mean"]["estimate"]
    selected_block = lolo[PRIMARY]["rankers"][PRIMARY_COST][f"{best['arm']}:{best['request']}"]["auc"]
    q3 = STOP_TOKEN if not guards_ok else ("supported" if meets(selected_block) else "not_supported_by_this_experiment")
    held_selected = heldout[PRIMARY]["rankers"][PRIMARY_COST][f"{best['arm']}:{best['request']}"]["auc"]
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "guards": guards, "guards_pass": guards_ok,
        "Q1_long_horizon": {"estimate": q1, "with_contrastive_LC_minus_C": grid["LC-C"]["request_mean"]["estimate"],
                            "token": token(q1, guards_ok), "label": "EXPLORATORY"},
        "Q2_contrastive": {"estimate": q2, "with_long_horizon_LC_minus_L": grid["LC-L"]["request_mean"]["estimate"],
                           "token": token(q2, guards_ok), "label": "EXPLORATORY"},
        "Q3_selected_target": {"selected": {"arm": best["arm"], "request": best["request"], "score": best["score"]},
                               "lolo_grid_auc": selected_block, "token": q3, "label": "EXPLORATORY"},
        "selection": {"scores": scores, "frozen_at": selection["frozen_at"],
                      "matches_record": selection["selected"] == {"arm": best["arm"], "request": best["request"],
                                                                  "family": t96.arm_family(best["request"]),
                                                                  "score": best["score"]}},
        "heldout_reading": {"selected_grid_auc": held_selected, "meets_target": meets(held_selected),
                            "requests_meeting_target": {arm: [r for r in REQUESTS if meets(
                                heldout[PRIMARY]["rankers"][PRIMARY_COST][f"{arm}:{r}"]["auc"])]
                                for arm in (*ARMS, REFERENCE)},
                            "label": "DESCRIPTIVE (exposed lineages, scored once after the selection freeze)"},
        "diagnostics": diagnostics,
        "lolo": lolo, "heldout": heldout,
        "rollout": {"lolo": lolo_curves, "lolo_nonfinite": lolo_nonfinite,
                    "lolo_contrasts": rollout_contrasts(lolo_per, ARMS, "B"),
                    "heldout": held_curves, "heldout_nonfinite": held_nonfinite,
                    "heldout_contrasts": rollout_contrasts(held_per, (*ARMS, REFERENCE), "B")},
        "training": training,
        "compute": {"phases": ledger["phases"], "gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "wall_seconds": ledger["wall_seconds_elapsed"], "engine_seconds": 0},
    }


def load_training_data_members(output):
    payload = torch.load(data_path(output), map_location="cpu", weights_only=True, mmap=True)
    return sorted(set(payload["shot_members"]) | set(payload["cell_members"]))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def auc_cell(block):
    return "n/a" if block is None else fmt_interval(block)


def num(value, digits=4):
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return format(value, f".{digits}f") if abs(value) >= 1e-3 or value == 0 else f"{value:.2e}"


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "cohort", "inventory", "cost", "unit", "statistic", "value", "interval_low",
                     "interval_high", "clusters", "label"])

    def row(identifier, cohort, inventory, cost, unit, statistic, block, label):
        if block is None:
            writer.writerow([identifier, cohort, inventory, cost, unit, statistic, "", "", "", "", label])
        else:
            writer.writerow([identifier, cohort, inventory, cost, unit, statistic, f"{block['mean']:.6f}",
                             f"{block['interval'][0]:.6f}", f"{block['interval'][1]:.6f}", block["clusters"], label])

    for cohort, tables, label in (("lolo", compute["lolo"], "EXPLORATORY"), ("heldout", compute["heldout"],
                                                                             "DESCRIPTIVE")):
        for inventory, table in tables.items():
            for cost in COSTS:
                for unit, block in table["rankers"][cost].items():
                    row(f"{cohort}:{inventory}:{cost}:{unit}", cohort, inventory, cost, unit, "auc", block["auc"],
                        label)
                for name, contrast_block in table["contrasts"][cost].items():
                    for request, value in contrast_block.items():
                        row(f"{cohort}:{inventory}:{cost}:{name}:{request}", cohort, inventory, cost,
                            f"{name}:{request}", "paired_auc_difference", value["estimate"], label)
    for cohort in ("lolo", "heldout"):
        for key, block in compute["rollout"][cohort].items():
            row(f"rollout:{cohort}:{key}", cohort, "n1", "", key, "rollout_error", block,
                "EXPLORATORY" if cohort == "lolo" else "DESCRIPTIVE")
    for key, block in compute["diagnostics"]["growth"].items():
        row(f"diagnostic:growth:{key}", "fit", "n1", "", key, "rollout_error", block, "DESCRIPTIVE")
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    sel = compute["Q3_selected_target"]["selected"]
    add("# Issue #112 — decision-relevant dynamics on E99 carriers (Gate-B model)")
    add("")
    add(f"Frozen plan {plan['frozen_at']} (before training and scoring); recipe frozen by selection "
        f"{compute['selection']['frozen_at']}; held-out scored once after it. Validation: `{VALIDATION_COMMAND}`.")
    guard_text = ", ".join(f"{name} {block['pass']}" for name, block in compute["guards"].items())
    add(f"Guards pass: {compute['guards_pass']} ({guard_text}). Engine seconds 0.")
    add("")
    add("| question | estimate | token |")
    add("| --- | --- | --- |")
    add(f"| Q1 long horizon: L − B (grid, request-mean, LOLO) | {auc_cell(compute['Q1_long_horizon']['estimate'])} | "
        f"{compute['Q1_long_horizon']['token']} |")
    add(f"| Q1 alongside: LC − C | {auc_cell(compute['Q1_long_horizon']['with_contrastive_LC_minus_C'])} | |")
    add(f"| Q2 contrastive: C − B | {auc_cell(compute['Q2_contrastive']['estimate'])} | "
        f"{compute['Q2_contrastive']['token']} |")
    add(f"| Q2 alongside: LC − L | {auc_cell(compute['Q2_contrastive']['with_long_horizon_LC_minus_L'])} | |")
    add(f"| Q3 selected {sel['arm']}:{sel['request']} LOLO grid AUC (target {TARGET_AUC}, lower > {TARGET_LOWER}) | "
        f"{auc_cell(compute['Q3_selected_target']['lolo_grid_auc'])} | {compute['Q3_selected_target']['token']} |")
    held = compute["heldout_reading"]
    add(f"| held-out (descriptive, once) {sel['arm']}:{sel['request']} grid AUC | {auc_cell(held['selected_grid_auc'])} "
        f"| meets target: {held['meets_target']} |")
    add("")
    add("All leave-one-lineage-out (LOLO) numbers are EXPLORATORY; held-out numbers are DESCRIPTIVE.")
    add("")
    add("## 1. Diagnostics (frozen #99 E predictors, fit lineages, zero training)")
    add("")
    add("Action sensitivity: SD across an anchor's executed candidates of the predicted endpoint quantity divided by "
        "the engine SD, and the Spearman correlation with the engine value (member-clustered).")
    add("")
    add("| request | presence SD ratio | presence ρ | displacement SD ratio | displacement ρ | cost ρ |")
    add("| --- | --- | --- | --- | --- | --- |")
    sens = compute["diagnostics"]["sensitivity"]
    for request in (*REQUESTS, "parsed"):
        s = sens[request]
        add(f"| {request if request != 'parsed' else 'E99 parsed endpoint'} | {auc_cell(s['pig_presence']['sd_ratio'])} | "
            f"{auc_cell(s['pig_presence']['spearman'])} | {auc_cell(s['pig_displacement']['sd_ratio'])} | "
            f"{auc_cell(s['pig_displacement']['spearman'])} | {auc_cell(s[PRIMARY_COST]['spearman'])} |")
    add("")
    add("Rollout error growth (#100 engine-referenced metric, fit-lineage N1 samples):")
    add("")
    add("| request | t=15 | t=60 | t=150 | t=225 | 225 / 60 |")
    add("| --- | --- | --- | --- | --- | --- |")
    growth = compute["diagnostics"]["growth"]
    for request in REQUESTS:
        cells = [num((growth[f"{REFERENCE}:{request}:{t}:engine_position_mse"] or {}).get("mean"))
                 for t in (15, 60, 150, 225)]
        add(f"| {request} | {' | '.join(cells)} | "
            f"{num(compute['diagnostics']['growth_ratio_225_over_60'][f'{request}:engine_position_mse'])} |")
    add("")
    add("Endpoint variance shares (anchor / action / residual), tie-free cost:")
    add("")
    add("| inventory | source | anchor | action | residual |")
    add("| --- | --- | --- | --- | --- |")
    for inventory, block in compute["diagnostics"]["variance"].items():
        for source, values in block["reference"].items():
            v = values[PRIMARY_COST]
            if v:
                add(f"| {inventory} ({len(block['members'])} members) | {source} | {num(v['anchor'])} | "
                    f"{num(v['action'])} | {num(v['residual'])} |")
        for request, values in block["predicted"].items():
            v = values[PRIMARY_COST]
            if v:
                add(f"| {inventory} | {request} | {num(v['anchor'])} | {num(v['action'])} | {num(v['residual'])} |")
    add("")
    add("## 2. Leave-one-lineage-out AUC (tie-free cost; EXPLORATORY)")
    add("")
    for inventory in INVENTORIES:
        table = compute["lolo"][inventory]
        add(f"### {inventory} ({len(table['members'])} lineages, {table['cells']} cells)")
        add("")
        add("| request | " + " | ".join(ARMS) + " |")
        add("| --- | " + " | ".join("---" for _ in ARMS) + " |")
        for request in REQUESTS:
            add(f"| {request} | " + " | ".join(auc_cell(table["rankers"][PRIMARY_COST][f"{a}:{request}"]["auc"])
                                               for a in ARMS) + " |")
        add("")
        add("| contrast (request-mean) | estimate |")
        add("| --- | --- |")
        for name, block in table["contrasts"][PRIMARY_COST].items():
            add(f"| {name} | {auc_cell(block['request_mean']['estimate'])} |")
        add("")
    add("### Selection (S = mean of grid and offset LOLO AUC)")
    add("")
    top = sorted((s for s in compute["selection"]["scores"] if s["score"] is not None), key=lambda s: -s["score"])[:8]
    add("| rank | arm | request | S |")
    add("| --- | --- | --- | --- |")
    for rank, row in enumerate(top, 1):
        add(f"| {rank} | {row['arm']} | {row['request']} | {num(row['score'])} |")
    add("")
    add(f"Selected: **{sel['arm']} : {sel['request']}** (S = {num(sel['score'])}).")
    add("")
    add("## 3. Rollout error at t = 225 (#100 engine-referenced metric)")
    add("")
    add("| request | " + " | ".join(f"{a} LOLO" for a in ARMS) + " | " + " | ".join(f"{a} held-out"
                                                                          for a in (*ARMS, REFERENCE)) + " |")
    add("| --- | " + " | ".join("---" for _ in range(2 * len(ARMS) + 1)) + " |")
    for request in REQUESTS:
        lo = [num((compute["rollout"]["lolo"][f"{a}:{request}:{ENDPOINT}:engine_position_mse"] or {}).get("mean"))
              for a in ARMS]
        he = [num((compute["rollout"]["heldout"][f"{a}:{request}:{ENDPOINT}:engine_position_mse"] or {}).get("mean"))
              for a in (*ARMS, REFERENCE)]
        add(f"| {request} | " + " | ".join(lo + he) + " |")
    add("")
    add("## 4. Held-out (007/008/014/016; scored once after the freeze; DESCRIPTIVE)")
    add("")
    for inventory in INVENTORIES:
        table = compute["heldout"][inventory]
        add(f"### {inventory} ({len(table['members'])} members, {table['cells']} cells)")
        add("")
        add("| request | " + " | ".join((*ARMS, REFERENCE)) + " |")
        add("| --- | " + " | ".join("---" for _ in range(len(ARMS) + 1)) + " |")
        for request in REQUESTS:
            add(f"| {request} | " + " | ".join(auc_cell(table["rankers"][PRIMARY_COST][f"{a}:{request}"]["auc"])
                                               for a in (*ARMS, REFERENCE)) + " |")
        add("")
    add(f"Requests meeting the target on the held-out grid: {held['requests_meeting_target']}.")
    add("")
    add("## 5. Training and compute")
    add("")
    add("| arm:family | members complete | GPU s | params / member | final local | final long | final rank |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for key, block in compute["training"]["per_arm"].items():
        add(f"| {key} | {block['complete']}/{block['members']} | {block['gpu_seconds']:.0f} | "
            f"{block['parameters_per_member']} | {num(block['mean_final_local'])} | {num(block['mean_final_long'])} | "
            f"{num(block['mean_final_rank'])} |")
    add("")
    add(f"Total GPU seconds {compute['compute']['gpu_seconds']:.0f}; wall seconds {compute['compute']['wall_seconds']:.0f}; "
        "engine seconds 0.")
    add("")
    add(f"Claim boundary: {plan['claim_boundary']}.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "selection_frozen_at": compute["selection"]["frozen_at"], "validation_command": VALIDATION_COMMAND,
            "question": plan["question"], "disclosure": plan["disclosure"], "guards": compute["guards"],
            "guards_pass": compute["guards_pass"],
            "questions": {k: {"token": compute[k]["token"], "label": compute[k]["label"]}
                          for k in ("Q1_long_horizon", "Q2_contrastive", "Q3_selected_target")},
            "selected": compute["Q3_selected_target"]["selected"], "heldout_reading": compute["heldout_reading"],
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False, "engine_seconds": 0}


def rendered(plan, compute):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute)}


def spot_checks(output):
    """Re-score one diagnostic, one leave-one-lineage-out and one held-out record in this
    process and compare parsed quantities (#111 endpoint_replication); re-parse one stored
    anchor and context (#111 carrier_replication)."""
    _, _, universe = load_universe()
    samples = rollout_samples_all()
    costs = harness.EndpointCosts(wlc.load_objective())
    out = {}
    adapter = s99.load_adapter(SLOT_FIX, "E")
    member = fit_lineages_in(universe)[0]
    stored = torch.load(anchors_path(output, "fit"), weights_only=True)["anchors"][member]
    out["anchor"] = harness.carrier_replication(stored, parse_anchors(adapter, [member])[member])
    stored = torch.load(contexts_path(output, "fit"), weights_only=True)["contexts"][member]
    out["context"] = harness.carrier_replication(stored, parse_contexts(adapter, samples, [member])[member])
    anchors = torch.load(anchors_path(output, "fit"), weights_only=True)["anchors"]
    record = read_record(diag_path(output, "hybrid", SEEDS[0]), SCHEMA_DIAG)
    fresh = endpoint_block(load_member(output, REFERENCE, "hybrid", "all", SEEDS[0]), "hybrid", anchors[member],
                           universe[PRIMARY][member]["inventory"], costs)
    out["diagnostic"] = _replicated(record["cells"][PRIMARY][member], fresh)
    arm = "LC"
    record = read_record(lolo_path(output, arm, "hybrid", member, SEEDS[0]), SCHEMA_LOLO)
    model = load_member(output, arm, "hybrid", member, SEEDS[0])
    if model is not None:
        fresh = endpoint_block(model, "hybrid", anchors[member], universe[PRIMARY][member]["inventory"], costs)
        out["lolo"] = _replicated(record["inventories"][PRIMARY], fresh)
    held = sorted(m for m in universe[PRIMARY] if m in HELD_OUT)[0]
    record = read_record(heldout_path(output, arm, "continuous", SEEDS[0]), SCHEMA_HELDOUT)
    model = load_member(output, arm, "continuous", "all", SEEDS[0])
    if model is not None:
        anchors = torch.load(anchors_path(output, "heldout"), weights_only=True)["anchors"]
        fresh = endpoint_block(model, "continuous", anchors[held], universe[PRIMARY][held]["inventory"], costs)
        out["heldout"] = _replicated(record["inventories"][PRIMARY][held], fresh)
    return out


def _replicated(stored, fresh):
    checks = [harness.endpoint_replication(stored[r]["rows"], fresh[r]["rows"]) for r in fresh]
    worst = {k: max(c["max_abs_delta"][k] for c in checks) for k in harness.PARSED_QUANTITIES}
    return {"requests": len(checks), "max_abs_delta": worst, "pass": all(c["pass"] for c in checks)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    with harness.deterministic_scoring():
        spot = spot_checks(output)
    compute = compute_tables(output, plan, spot)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    log(f"published: Q1 {compute['Q1_long_horizon']['token']}, Q2 {compute['Q2_contrastive']['token']}, "
        f"Q3 {compute['Q3_selected_target']['token']}; guards pass {compute['guards_pass']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    with harness.deterministic_scoring():
        selection = read_json(output / "selection.json")
        if selection != selection_record(output, plan, selection["frozen_at"]):
            raise ValueError("selection.json differs from the frozen rule applied to the retained records")
        if read_json(output / "gate_b_handoff.json") != handoff(output, plan, selection):
            raise ValueError("gate_b_handoff.json differs from the frozen selection")
        spot = spot_checks(output)
        fresh = compute_tables(output, plan, spot)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    if not fresh["guards"]["G3"]["pass"]:
        raise ValueError("deterministic re-scoring does not replicate the retained records")
    log(f"validation passed: selection, handoff, every table re-derived and byte-compared; spot re-scores "
        f"replicate ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    sources, _, universe = load_universe()
    shots = s99.training_shots(sources, universe)
    samples = rollout_samples_all()
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    log(f"  fit lineages {fit_lineages()}; fit shots {len(shots)}; ranking cells {len(fit_cells(universe))}")
    log(f"  folds {len(folds())} x seeds {len(SEEDS)} = {len(ensemble_members())} members per (arm, family); "
        f"arms {ARMS}; families {FAMILIES}")
    log(f"  N1 samples per lineage {({m: len(v) for m, v in samples.items()})}")
    log(f"  recipe {recipe()}; long {LONG}; rank {RANK}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": (dry_run, False), "smoke": (smoke, True), "prepare": (prepare, False), "data": (data, True),
         "diagnose": (diagnose, True), "train": (train, True), "score-lolo": (score_lolo, True),
         "select": (select, False), "score-heldout": (score_heldout, True), "publish": (publish, True),
         "validate": (validate, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    function, locked = MODES[mode]
    try:
        if locked:
            with wlc.GPULock():
                return function(args.output)
        return function(args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
