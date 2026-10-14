"""Issue-103: engine-truth compute-quality frontier of the request policies.

Binding runner module: scripts/run_engine_truth_frontier.py
Exact validation command: python -u -m scripts.run_engine_truth_frontier --validate

#103 replaces the ICLR compute comparison ("matches a horizon-only adaptive baseline at a third of its
compute", measured with the #74 count-cost proxy) by a Pareto frontier of ranking quality against
compute on engine verdicts. This runner

1. assembles every request policy of the #108 program on the Gate-B LC predictors: the nine fixed
   requests, the continuous-only fixed requests C-F-{1,5,15} on C*, the horizon-only adaptive C-J
   (8 lambdas), J (8 lambdas), J-cont (8), the restricted T-alpha / D-Delta arms, the decision-aware
   J_dec (4) and J_pg (4), the factorized / open-loop arms, the #105 tied-axes restrictions, the
   hierarchical levels and the #78 ports, and the no-model action prior (71 policies);
2. primary set (the #104 sealed #109 evaluation split, Gate-B cells, grid primary): reuses the retained
   #101 / #114 / #105 records, which score exactly these cells under the #111 contract; it checks reuse
   fidelity by re-scoring every policy on every cell of seed 20260908 in a uniform timing pass (one
   process per GPU, warm-up cell discarded), which is also the wall-time axis;
3. declared secondary set (the #93 / #94 / #87 engine-verdicted inventories of the #96 universe, mixed
   cells): scores every policy except the primary-only open-loop arms (``--anchors``, ``--score``);
4. publishes AUC and top-3 against linear MACs per candidate-state and against wall seconds per state,
   the point Pareto set, each policy's member-clustered bootstrap frontier share, tie counts, a one-line
   verdict per policy, and the assessment of the ICLR sentence (``--publish``).

Modes: --dry-run --smoke --prepare --anchors --score --publish --validate. --score takes --workers N
--worker K (one process per GPU; CUDA_VISIBLE_DEVICES set by the caller). Zero engine seconds.
Intervals: member (level)-clustered percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_decision_aware_policy as dap
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_hierarchy_baselines as hb
from scripts import run_horizon_vs_joint as hvj
from scripts import run_launch_power_probe as p94
from scripts import run_lookahead_control as wlc
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-103-engine-truth-frontier-v1"
P101 = dap.OUTPUT
P114 = hvj.OUTPUT
P105 = hb.OUTPUT
GATE_B = gb.OUTPUT
V1 = v1.OUTPUT  # #112 v1: E99 parses of the #96 universe anchors (p94.frozen_anchor)

IDENTITY = "issue-103-engine-truth-frontier-v1"
SCHEMA_PLAN = "issue_103_engine_truth_frontier_plan_v1"
SCHEMA_SMOKE = "issue_103_engine_truth_frontier_smoke_v1"
SCHEMA_ANCHORS = "issue_103_secondary_f77_anchors_v1"
SCHEMA_TIMING = "issue_103_timing_record_v1"
SCHEMA_SECONDARY = "issue_103_secondary_record_v1"
SCHEMA_COMPUTE = "issue_103_compute_v1"
SCHEMA_REPORT = "issue_103_report_v1"
SCHEMA_MANIFEST = "issue_103_record_manifest_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_engine_truth_frontier --validate"
RUNNER = "scripts/run_engine_truth_frontier.py"

DEVICE = "cuda"
SEEDS = tuple(dap.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
TIMING_SEED = SEEDS[0]  # smoke sample and validation spot check
TIMING_SEEDS = SEEDS  # post-freeze amendment (AMENDMENT): the timing pass covers every seed
ICLR_LAMBDA = dap.ICLR_LAMBDA
SETS = ("primary", "secondary")
PRIOR = "prior"
GPU_CAP_SECONDS = 6 * 3600.0
WALL_CAP_SECONDS = 12 * 3600.0  # summed over processes (4 GPUs, one process each)
SMOKE_GRID_CELLS = 2

ICLR_SENTENCE = "matches a horizon-only adaptive baseline at a third of its compute"
ICLR_J = f"J-{dap.tag(ICLR_LAMBDA)}"
ICLR_CJ = f"CJ-{dap.tag(ICLR_LAMBDA)}"
THIRD_TOLERANCE = 0.40  # "a third" read generously: J's MACs at most 0.40 x C-J's
HISTORICAL_74 = {
    "source": ".local-artifacts/issue-74-matched-dynamics-v1/readiness.json (count-cost proxy; earlier predictors)",
    "hybrid_adaptive_macs_per_candidate_state": 118.7e6, "continuous_adaptive_macs_per_candidate_state": 356.0e6,
    "ratio": 118.7 / 356.0,
}

COMBOS = {  # frontier: (x key, minimized; y field, maximized)
    "auc_macs": ("mean_macs_per_candidate", "auc"),
    "auc_wall": ("seconds_per_state", "auc"),
    "top3_macs": ("mean_macs_per_candidate", "top3_hit"),
    "top3_wall": ("seconds_per_state", "top3_hit"),
}
HEADLINE = "auc_macs"
Y_BLOCK = {"auc": "auc", "top3_hit": "top3"}

GROUPS = {
    "fixed": "fixed request (hybrid)",
    "fixed_continuous": "continuous-only fixed Delta (C*)",
    "horizon_adaptive": "horizon-only adaptive C-J (C*)",
    "J": "J (prediction-error DP)",
    "J_cont": "J-cont (continuous requests, hybrid)",
    "J_restricted": "J restricted (T-alpha / D-Delta)",
    "J_dec": "J_dec (decision-valued DP)",
    "J_pg": "J_pg (policy gradient)",
    "factorized_open_loop": "factorized / open-loop (#101)",
    "tied_axes": "#105 tied-axes restriction",
    "hierarchical": "#105 hierarchical level",
    "external_78": "#78 port (#71 carrier)",
    "prior": "no-model action prior",
}
GROUP_STYLE = {
    "fixed": ("tab:gray", "x"), "fixed_continuous": ("black", "D"), "horizon_adaptive": ("tab:green", "^"),
    "J": ("tab:blue", "o"), "J_cont": ("tab:orange", "s"), "J_restricted": ("tab:red", "*"),
    "J_dec": ("tab:purple", "P"), "J_pg": ("tab:pink", "X"), "factorized_open_loop": ("tab:olive", "v"),
    "tied_axes": ("tab:cyan", "<"), "hierarchical": ("tab:brown", ">"), "external_78": ("gold", "h"),
    "prior": ("red", "_"),
}
PRIMARY_ONLY = ("OL-pe", "OL-dec")
AMENDMENT = {
    "at": "2026-10-14T15:18:02Z",
    "after": ("the first publication of the frozen runner (all 70 seed-20260908 timing and 204 secondary records "
              "complete; compute.json, summary.json and findings.md written once)"),
    "what": ("the primary timing pass (wall axis, and G1 reuse fidelity) is extended from seed 20260908 to all three "
             "seeds; the wall column pools every seed's timing records, as the MAC column pools every seed's records"),
    "why": ("each seed is a separately trained policy, so seed 20260908 alone does not represent the all-seed MAC column "
            "of the same policy: grid transition calls per candidate at seed 20260908 vs pooled were 196 vs 121 for "
            "J-1e-04 and 21 vs 124.5 for CJ-1e-04; the frozen seed-only wall gave a J / C-J wall ratio of 16.7 that "
            "describes seed 20260908, not the policies"),
    "unchanged": ("cells, retained and secondary records, AUC and top-3, MACs, every MAC-axis frontier, bootstrap share "
                  "and verdict, and the ICLR-sentence token (its rule uses MACs); only the wall axis (AUC x wall and "
                  "top-3 x wall frontiers, wall columns and the wall ratio) and the G1 coverage change"),
    "superseded": "superseded/ keeps the first publication's findings.md, summary.json and compute.json",
    "execution": ("one additional launch of --score on 4 GPUs (one process each) for the seed-20260909/10 timing "
                  "tasks; existing records are skipped, nothing is overwritten; PIDs in logs/score-pids.txt"),
}
EXCLUDED = {
    "J-as-cont (#114)": "a counterfactual replay of J's recorded schedules; it needs J to run first, so it is not a "
                        "deployable policy",
    "wrong-anchor arms (#101/#114/#105)": "controls scored from the wrong anchor, not policies",
    "OL-pe / OL-dec on the secondary set": "their open-loop streams are frozen to the primary cell order and J's "
                                           "primary request frequencies (#101); no secondary analogue was frozen",
}

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval
tag = dap.tag


def log(message):
    print(f"[issue-103] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(value):
    """sha256 of the canonical JSON text of ``value`` (rows / schedules of one cell)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def catalogue():
    """Every frontier policy: id, group, source ticket and the source record arm (frozen order)."""
    out = []

    def add(pid, group, source, arm):
        out.append({"id": pid, "group": group, "source": source, "arm": arm})

    for arm in t96.FIXED_ARMS:
        add(arm, "fixed", "101", arm)
    for arm in t96.CONTINUOUS_FIXED_ARMS:
        add(arm, "fixed_continuous", "101" if arm == dap.C_STAR else "114", arm)
    for lam in hvj.LAMBDAS:
        add(f"CJ-{tag(lam)}", "horizon_adaptive", "114", hvj.family_arm("CJ", lam))
    for lam in hvj.LAMBDAS:
        if lam in dap.LAMBDAS["pe"]:
            add(f"J-{tag(lam)}", "J", "101", dap.policy_arm("pe", lam))
        else:
            add(f"J-{tag(lam)}", "J", "114", hvj.family_arm("J", lam))
    for lam in hvj.LAMBDAS:
        add(f"Jcont-{tag(lam)}", "J_cont", "114", hvj.family_arm("Jcont", lam))
    for arm in hvj.RESTRICTED_ARMS:
        add(arm, "J_restricted", "114", arm)
    for lam in dap.LAMBDAS["dec"]:
        add(dap.policy_arm("dec", lam), "J_dec", "101", dap.policy_arm("dec", lam))
    for lam in dap.LAMBDAS["pg"]:
        add(dap.policy_arm("pg", lam), "J_pg", "101", dap.policy_arm("pg", lam))
    for arm in ("M-pe", "M-dec", *PRIMARY_ONLY):
        add(arm, "factorized_open_loop", "101", arm)
    for arm in hb.POLICY_ARMS:
        if arm not in ("J", "J_dec"):  # = #101 J-pe-1e-04 / J-dec-1e-01 (cross-checked, not duplicated)
            add(arm, "tied_axes", "105", arm)
    for arm in hb.H_ARMS:
        add(arm, "hierarchical", "105", arm)
    for arm in hb.EXTERNAL_ARMS:
        add(arm, "external_78", "105", arm)
    return tuple(out)


def entries_for(set_name):
    return [e for e in catalogue() if set_name == "primary" or e["id"] not in PRIMARY_ONLY]


def entry_of(pid):
    return next(e for e in catalogue() if e["id"] == pid)


def policy_ids(set_name):
    return [e["id"] for e in entries_for(set_name)] + [PRIOR]


def group_of(pid):
    return "prior" if pid == PRIOR else entry_of(pid)["group"]


def retained_path(source, arm, seed):
    if source == "101":
        return dap.record_path(P101, arm, seed)
    if source == "114":
        return hvj.record_path(P114, arm, seed)
    if source == "105":
        return hb.record_path(P105, arm, seed)
    if source == "gate_b":
        return gb.record_path(GATE_B, arm, seed)
    raise ValueError(f"unknown source {source}")


def retained(source, arm, seed):
    """A retained record, through its source runner's binding check."""
    if source == "101":
        return dap.read_record(P101, arm, seed)
    if source == "114":
        return hvj.read_record(P114, arm, seed)
    if source == "105":
        return hb.read_record(P105, arm, seed)
    record = read_json(gb.record_path(GATE_B, arm, seed))
    if record.get("schema") != gb.SCHEMA_RECORD or record.get("plan_identity") != gb.IDENTITY:
        raise ValueError(f"Gate-B record {arm}/{seed} binding differs")
    return record


@functools.lru_cache(maxsize=None)
def selection():
    return dap.load_selection(P101)


def cross_pairs():
    """Retained arms recorded by more than one source (they must be identical)."""
    pairs = []
    for arm in t96.FIXED_ARMS:
        pairs += [(("101", arm), ("114", arm)), (("101", arm), ("105", arm))]
    pairs += [(("101", dap.C_STAR), ("114", dap.C_STAR)), (("101", dap.C_STAR), ("105", hb.C_STAR))]
    pairs += [(("101", dap.policy_arm("pe", lam)), ("114", hvj.family_arm("J", lam))) for lam in dap.LAMBDAS["pe"]]
    pairs += [(("101", dap.J_ICLR), ("105", "J")), (("101", selection()["kinds"]["dec"]["arm"]), ("105", "J_dec"))]
    pairs += [(("101", "F-15-macro"), ("gate_b", "unit")), (("101", dap.C_STAR), ("gate_b", "c_star"))]
    return pairs


def load_arm(entry, seed, cells):
    """Model, policy, schedule spec and carrier of one policy at one seed, through its source runner's
    loaders (hash-checked there) and its source runner's scoring block."""
    source, arm = entry["source"], entry["arm"]
    base = {"block": None, "model": None, "policy": None, "spec": None, "carrier": "E99", "schedules": False,
            "typed_failure": None, "binding": {}, "open_loop": None}
    if source == "105":
        model, policy, spec, carrier, extra = hb.arm_setup(P105, arm, seed)
        failure = extra.pop("typed_failure", None)
        return {**base, "block": hb.block, "model": model, "policy": policy, "spec": spec, "carrier": carrier,
                "schedules": arm in hb.SCHEDULED, "typed_failure": failure, "binding": extra}
    if source == "114":
        name = hvj.predictor_name(arm)
        out = {**base, "block": hvj.block, "model": dap.load_predictor(seed, name),
               "binding": {"predictor_sha256": sha256_of(gb.member_path(name, seed))}}
        if arm in hvj.C_FIXED_ARMS:
            return {**out, "spec": t96.arm_spec(arm)}
        family, lam = (("J", hvj.ICLR_LAMBDA) if arm in hvj.RESTRICTED_ARMS else hvj.parse_arm(arm))
        out["binding"]["policy_sha256"] = hvj.policy_sha(P114, family, lam, seed)
        return {**out, "policy": hvj.load_policy(P114, family, lam, seed), "schedules": True,
                "spec": hvj.restricted_spec(arm) if arm in hvj.RESTRICTED_ARMS else {"kind": "joint"}}
    name = "c_star" if arm == dap.C_STAR else "unit"
    out = {**base, "block": dap.block, "model": dap.load_predictor(seed, name),
           "binding": {"predictor_sha256": sha256_of(gb.member_path(name, seed))}}
    if arm in dap.FIXED_ARMS or arm == dap.C_STAR:
        return {**out, "spec": t96.arm_spec(arm)}
    if arm in dap.PHASE_2:
        source_arm = dap.J_ICLR if arm == "OL-pe" else selection()["kinds"]["dec"]["arm"]
        return {**out, "schedules": True, "spec": "open_loop",
                "open_loop": dap.ol_schedules(cells, dap.j_counts(P101, source_arm))}
    kind, lam, schedule_kind = dap.arm_policy(arm, selection())
    out["binding"]["policy_sha256"] = sha256_of(dap.policy_path(P101, kind, lam, seed))
    return {**out, "policy": dap.load_policy(P101, kind, lam, seed), "spec": {"kind": schedule_kind},
            "schedules": True}


def run_cell(arm, inventory, member, entry, anchors, costs, seed):
    spec = arm["spec"]
    if spec == "open_loop":
        spec = {"kind": "schedule", "schedule": arm["open_loop"][(inventory, member, seed)]}
    return arm["block"](arm["model"], arm["policy"], spec, anchors[arm["carrier"]][member], entry["inventory"], costs,
                        arm["schedules"])


def equal_block(fresh, stored):
    return bool(fresh["rows"] == stored["rows"] and fresh.get("schedules") == stored.get("schedules")
                and all(fresh[k] == stored[k] for k in ("linear_macs", "transition_calls", "controller_calls",
                                                        "candidates")))


# ---------------------------------------------------------------------------
# cells and anchors
# ---------------------------------------------------------------------------

def primary_cells():
    return gb.evaluation_set()[1]


@functools.lru_cache(maxsize=None)
def primary_anchors():
    return {"E99": gb.load_anchors(GATE_B)["anchors"], "F77": hb.load_f77_anchors(P105)["anchors"]}


@functools.lru_cache(maxsize=None)
def secondary_universe():
    _, _, universe = v1.load_universe()
    return universe


def secondary_cells():
    """The #96 universe (#93 grid / offset, #94 power, #87 + #89 angle): mixed member cells."""
    universe = secondary_universe()
    return {inventory: {"all": dict(universe[inventory]),
                        "mixed": {m: e for m, e in sorted(universe[inventory].items()) if t96.mixed(e)}}
            for inventory in INVENTORIES}


def secondary_members():
    cells = secondary_cells()
    return sorted({m for inventory in INVENTORIES for m in cells[inventory]["mixed"]})


def secondary_e99_paths():
    return [v1.anchors_path(V1, cohort) for cohort in ("fit", "heldout")]


def secondary_e99():
    """E99 batch-1 parses of each member's sealed #96 anchor (p94.frozen_anchor), retained by #112 v1."""
    out = {}
    for path in secondary_e99_paths():
        store = torch.load(path, map_location="cpu", weights_only=True)
        if store.get("scoring_policy") != harness.SCORING_POLICY:
            raise ValueError(f"{path}: E99 anchors were not parsed under deterministic scoring")
        out.update(store["anchors"])
    return out


def f77_store_path(output):
    return Path(output) / "data" / "anchors-secondary-f77.pt"


def load_secondary_anchors(output):
    store = torch.load(f77_store_path(output), map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_ANCHORS or store.get("plan_identity") != IDENTITY:
        raise ValueError("secondary F77 anchor store binding differs")
    return {"E99": secondary_e99(), "F77": store["anchors"]}


def frozen_anchor(member):
    anchor = p94.frozen_anchor(member)
    if sha256_of(anchor["frame_path"]) != anchor["sha256"]:
        raise ValueError(f"{STOP_TOKEN}: anchor frame of {member} changed")
    return anchor


def ordered(cells):
    return [(inventory, member, entry) for inventory in INVENTORIES
            for member, entry in sorted(cells[inventory]["mixed"].items())]


# ---------------------------------------------------------------------------
# paths, ledger
# ---------------------------------------------------------------------------

def timing_path(output, pid, seed):
    return Path(output) / "records" / f"timing--{pid}--seed{seed}.json"


def secondary_path(output, pid, seed):
    return Path(output) / "records" / f"secondary--{pid}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def mark(output, phase, wall, gpu, **extra):
    write_json(ledger_path(output, phase), {"phase": phase, "wall_seconds": wall, "gpu_seconds": gpu,
                                            "at": utc_now(), **extra})


def ledger_totals(output):
    phases = ledger_phases(output)
    return {"gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
            "wall_seconds": sum(b["wall_seconds"] for b in phases.values())}


# ---------------------------------------------------------------------------
# anchors (secondary #71 carrier for the #78 ports)
# ---------------------------------------------------------------------------

def anchors(output):
    output = Path(output)
    load_plan(output)
    if f77_store_path(output).is_file():
        log("secondary F77 anchors already parsed")
        return 0
    began = time.monotonic()
    members = secondary_members()
    with harness.deterministic_scoring() as policy:
        adapter = wlc.load_adapter(DEVICE)
        store = {m: wlc.encode_carrier(adapter, frozen_anchor(m), DEVICE).float().cpu() for m in members}
    atomic_torch(f77_store_path(output), {"schema": SCHEMA_ANCHORS, "plan_identity": IDENTITY, "scoring_policy": policy,
                                          "parsed_at": utc_now(), "anchors": store})
    wall = time.monotonic() - began
    mark(output, "anchors-secondary-f77", wall, wall, members=len(store))
    log(f"secondary F77 anchors parsed: {len(store)} members ({wall:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# scoring: primary timing pass (reuse fidelity + wall time) and the secondary set
# ---------------------------------------------------------------------------

def record_base(schema, entry, seed, arm, policy_state, set_name):
    return {"schema": schema, "plan_identity": IDENTITY, "scoring_policy": policy_state, "scored_at": utc_now(),
            "policy": entry["id"], "group": entry["group"], "source": entry["source"], "arm": entry["arm"],
            "seed": seed, "set": set_name, "carrier": arm["carrier"], "typed_failure": arm["typed_failure"],
            "binding": arm["binding"], "engine_seconds": 0, "device": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def timing_record(entry, seed, policy_state):
    """Every primary cell of one seed re-scored in one process per GPU after a discarded warm-up cell:
    synchronized rollout seconds per state, MACs and calls, and digests of rows and schedules (compared
    with the retained record at publication)."""
    cells = primary_cells()
    store = primary_anchors()
    costs = dap.endpoint_costs()
    began = time.monotonic()
    arm = load_arm(entry, seed, cells)
    base = record_base(SCHEMA_TIMING, entry, seed, arm, policy_state, "primary-timing")
    if arm["model"] is None:
        return {**base, "inventories": {}, "seconds": time.monotonic() - began}
    order = ordered(cells)
    run_cell(arm, *order[0], store, costs, seed)  # warm-up, discarded
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, cell in order:
        block = run_cell(arm, inventory, member, cell, store, costs, seed)
        inventories[inventory][member] = {
            "seconds": block["gpu_seconds"], "candidates": block["candidates"], "linear_macs": block["linear_macs"],
            "transition_calls": block["transition_calls"], "controller_calls": block["controller_calls"],
            "rows_sha256": digest(block["rows"]), "schedules_sha256": digest(block.get("schedules"))}
    torch.cuda.synchronize()
    return {**base, "warmup": [order[0][0], order[0][1]], "inventories": inventories,
            "seconds": time.monotonic() - began}


def secondary_record(output, entry, seed, policy_state):
    cells = secondary_cells()
    store = load_secondary_anchors(output)
    costs = dap.endpoint_costs()
    began = time.monotonic()
    arm = load_arm(entry, seed, cells)
    base = record_base(SCHEMA_SECONDARY, entry, seed, arm, policy_state, "secondary")
    if arm["model"] is None:
        return {**base, "inventories": {}, "seconds": time.monotonic() - began}
    order = ordered(cells)
    run_cell(arm, *order[0], store, costs, seed)  # warm-up, discarded
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, cell in order:
        inventories[inventory][member] = run_cell(arm, inventory, member, cell, store, costs, seed)
    torch.cuda.synchronize()
    return {**base, "warmup": [order[0][0], order[0][1]], "inventories": inventories,
            "seconds": time.monotonic() - began}


def read_timing(output, pid, seed):
    record = read_json(timing_path(output, pid, seed))
    if (record.get("schema") != SCHEMA_TIMING or record.get("plan_identity") != IDENTITY or record["policy"] != pid
            or record["seed"] != seed):
        raise ValueError(f"timing record {pid}/{seed} binding differs")
    return record


def read_secondary(output, pid, seed):
    record = read_json(secondary_path(output, pid, seed))
    if (record.get("schema") != SCHEMA_SECONDARY or record.get("plan_identity") != IDENTITY
            or record["policy"] != pid or record["seed"] != seed):
        raise ValueError(f"secondary record {pid}/{seed} binding differs")
    return record


def score_tasks():
    timing = [("timing", e["id"], s) for s in TIMING_SEEDS for e in entries_for("primary")]
    secondary = [("secondary", e["id"], s) for e in entries_for("secondary") for s in SEEDS]
    return timing, secondary


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    if not f77_store_path(output).is_file():
        raise ValueError("secondary F77 anchors missing; run --anchors first")
    timing, secondary = score_tasks()
    for kind, pid, seed in timing[worker::workers] + secondary[worker::workers]:
        path = timing_path(output, pid, seed) if kind == "timing" else secondary_path(output, pid, seed)
        if path.is_file():
            continue
        totals = ledger_totals(output)
        if totals["gpu_seconds"] > GPU_CAP_SECONDS or totals["wall_seconds"] > WALL_CAP_SECONDS:
            log(f"worker {worker}: compute cap reached; stopping ({totals})")
            return 1
        began = time.monotonic()
        with harness.deterministic_scoring() as policy_state:
            entry = entry_of(pid)
            record = (timing_record(entry, seed, policy_state) if kind == "timing"
                      else secondary_record(output, entry, seed, policy_state))
        write_json(path, record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - began
        mark(output, f"{kind}-{pid}-seed{seed}", wall, wall, worker=worker)
        log(f"worker {worker}: {kind} {pid} seed {seed} ({wall:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# frontier statistics
# ---------------------------------------------------------------------------

def pareto_mask(x, y):
    """On-frontier flags (x minimized, y maximized): i is dominated iff some j has x_j <= x_i and
    y_j >= y_i with at least one strict. Undefined (non-finite) points are off the frontier and
    dominate nothing; identical points do not dominate each other."""
    return pareto_batch(np.asarray(x, dtype=float), np.asarray(y, dtype=float)[None])[0]


def pareto_batch(x, ys):
    """``pareto_mask`` for each row of ``ys`` [draws, policies] with fixed ``x`` [policies]."""
    x = np.asarray(x, dtype=float)
    ys = np.asarray(ys, dtype=float)
    valid = np.isfinite(ys) & np.isfinite(x)[None]
    le = (x[None, :] <= x[:, None])[None]          # [1, i, j]: x_j <= x_i
    lt = (x[None, :] < x[:, None])[None]
    ge = ys[:, None, :] >= ys[:, :, None]          # [d, i, j]: y_j >= y_i
    gt = ys[:, None, :] > ys[:, :, None]
    dominated = (le & ge & (lt | gt) & valid[:, None, :]).any(axis=2)
    return valid & ~dominated


def cluster_matrix(rows, pids, field):
    """Per member (sorted) and policy: the mean of the defined cell values and a defined flag."""
    members = sorted({r["member"] for r in rows})
    index = {m: i for i, m in enumerate(members)}
    sums = np.zeros((len(members), len(pids)))
    counts = np.zeros((len(members), len(pids)))
    for r in rows:
        i = index[r["member"]]
        for j, pid in enumerate(pids):
            value = r["views"][pid][field]
            if value is not None:
                sums[i, j] += float(value)
                counts[i, j] += 1
    defined = counts > 0
    return members, np.where(defined, sums / np.maximum(counts, 1), 0.0), defined


def frontier_shares(rows, pids, x, field, draws=p94.BOOTSTRAP_DRAWS, chunk=500):
    """Share of member-clustered bootstrap draws (members resampled jointly for all policies, the p94
    generator and draw matrix) in which each policy is on the Pareto frontier; x (compute) held fixed.
    A policy's draw value is the mean of its defined cluster means among the drawn members."""
    members, means, defined = cluster_matrix(rows, pids, field)
    generator = np.random.default_rng(np.random.PCG64(p94.BOOTSTRAP_SEED))
    draw_index = generator.integers(0, len(members), size=(draws, len(members)))
    on = np.zeros(len(pids))
    weights_defined = defined.astype(float)
    for start in range(0, draws, chunk):
        idx = draw_index[start:start + chunk]
        weight = np.zeros((len(idx), len(members)))
        np.add.at(weight, (np.arange(len(idx))[:, None], idx), 1.0)
        den = weight @ weights_defined
        ys = np.where(den > 0, (weight @ means) / np.where(den > 0, den, 1.0), np.nan)
        on += pareto_batch(x, ys).sum(axis=0)
    return on / draws


def paired(rows, pieces, field):
    """m99.paired for any per-cell view field (AUC or top-3 hit)."""
    values, clusters = [], []
    for r in rows:
        parts = [r["views"][key][field] for _, key in pieces]
        if any(p is None for p in parts):
            continue
        values.append(float(sum(w * p for (w, _), p in zip(pieces, parts))))
        clusters.append(r["member"])
    return {"cells": len(values), "members": len(set(clusters)), "estimate": p94.bootstrap(values, clusters)}


def delta_token(block, guards_ok):
    return c2.token(block) if guards_ok else STOP_TOKEN


def view_rows(records, cells, inventory, stats, pids):
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = hvj.view(entry, gb.prior_rows(entry, stats))
        for seed in SEEDS:
            views = {}
            for pid in pids:
                record = records[(pid, seed)]
                rows_ = None if record.get("typed_failure") else record["inventories"][inventory][member]["rows"]
                views[pid] = hvj.view(entry, rows_)
            views[PRIOR] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def ranker(rows, pid):
    views = [r["views"][pid] for r in rows]
    chance = [v["chance_top3"] for v in views if v["chance_top3"] is not None]
    pairs = sum(v["pairs"] for v in views)
    slots = sum(v["verdict_slots"] for v in views)
    return {"auc": m99.block_of(rows, pid), "top3": m99.block_of(rows, pid, "top3_hit"),
            "chance_top3": float(np.mean(chance)) if chance else None, "cells": len(views),
            "typed_failures": sum(1 for v in views if v.get("typed_failure")),
            "tied_cells": sum(1 for v in views if v["all_tied"] and not v.get("typed_failure")),
            "tied_pairs": sum(v["tied_pairs"] for v in views), "pairs": pairs,
            "nonfinite": sum(v["nonfinite"] for v in views), "verdict_slots": slots}


def compute_columns(records, inventory, pid, wall_source):
    results = [r["inventories"][inventory][m] for s in SEEDS for r in [records[(pid, s)]] if not r.get("typed_failure")
               for m in r["inventories"][inventory]]
    candidates = sum(r["candidates"] for r in results)
    out = {"mean_macs_per_candidate": sum(r["linear_macs"] for r in results) / candidates if candidates else None,
           "mean_transitions_per_candidate": sum(r["transition_calls"] for r in results) / candidates
           if candidates else None,
           "mean_policy_calls_per_candidate": sum(r["controller_calls"] for r in results) / candidates
           if candidates else None,
           "recorded_seconds_per_state": float(np.mean([r["gpu_seconds"] for r in results])) if results else None,
           "states": len(results), "candidates": candidates}
    if wall_source is None:
        out["seconds_per_state"] = out["recorded_seconds_per_state"]
        out["seconds_source"] = "this scoring (one process per GPU, warm-up discarded), all seeds"
    else:
        timed = [t for record in wall_source if not record.get("typed_failure")
                 for t in record["inventories"][inventory].values()]
        out["seconds_per_state"] = float(np.mean([t["seconds"] for t in timed])) if timed else None
        out["seconds_per_candidate"] = (sum(t["seconds"] for t in timed) / sum(t["candidates"] for t in timed)
                                        if timed else None)
        out["seconds_source"] = "uniform timing pass, all seeds (one process per GPU, warm-up discarded)"
    return out


PRIOR_COMPUTE = {"mean_macs_per_candidate": 0.0, "mean_transitions_per_candidate": 0.0,
                 "mean_policy_calls_per_candidate": 0.0, "recorded_seconds_per_state": None,
                 "seconds_per_state": 0.0, "seconds_source": "no model work (a table lookup; not timed, plotted at 0)"}


def frontier_block(rows, pids, policies, combo, guards_ok):
    xkey, field = COMBOS[combo]
    yblock = Y_BLOCK[field]
    x = np.array([policies[p][xkey] if policies[p][xkey] is not None else np.nan for p in pids], dtype=float)
    y = np.array([policies[p][yblock]["mean"] if policies[p][yblock] else np.nan for p in pids], dtype=float)
    on = pareto_mask(x, y)
    shares = frontier_shares(rows, pids, x, field)
    verdicts = {}
    for i, pid in enumerate(pids):
        if not np.isfinite(y[i]) or not np.isfinite(x[i]):
            verdicts[pid] = {"on_frontier": False, "undefined": True, "text": "undefined (no defined cell)"}
            continue
        if on[i]:
            verdicts[pid] = {"on_frontier": True, "share": float(shares[i]),
                             "text": f"on the frontier (bootstrap share {shares[i]:.2f})"}
            continue
        dominators = [j for j in range(len(pids)) if np.isfinite(y[j]) and x[j] <= x[i] and y[j] >= y[i]
                      and (x[j] < x[i] or y[j] > y[i])]
        best = max(dominators, key=lambda j: (y[j], -x[j], -j))
        contrast = paired(rows, [(1, pids[best]), (-1, pid)], field)
        token = delta_token(contrast["estimate"], guards_ok)
        verdicts[pid] = {"on_frontier": False, "share": float(shares[i]), "dominated_by": pids[best],
                         "dominators": [pids[j] for j in dominators], "contrast": contrast, "token": token,
                         "text": (f"dominated by {pids[best]} ({len(dominators)} dominating; {pids[best]} - {pid} = "
                                  f"{fmt_interval(contrast['estimate'])}, {token}; bootstrap frontier share "
                                  f"{shares[i]:.2f})")}
    return {"x": xkey, "y": field, "on": [p for p, f in zip(pids, on) if f],
            "shares": {p: float(s) for p, s in zip(pids, shares)}, "verdicts": verdicts}


def inventory_table(records, cells, inventory, stats, set_name, timing, guards_ok):
    pids = policy_ids(set_name)
    models = pids[:-1]
    rows = view_rows(records, cells, inventory, stats, models)
    policies = {}
    for pid in pids:
        block = ranker(rows, pid)
        if pid == PRIOR:
            block.update(PRIOR_COMPUTE)
        else:
            block.update(compute_columns(records, inventory, pid,
                                         [timing[(pid, s)] for s in TIMING_SEEDS] if timing is not None else None))
        policies[pid] = {"group": group_of(pid), **block}
    frontiers = {combo: frontier_block(rows, pids, policies, combo, guards_ok) for combo in COMBOS}
    return {"cells": len(rows), "members": len({r["member"] for r in rows}), "policies": policies,
            "frontiers": frontiers}, rows


def iclr_assessment(table, rows, guards_ok):
    """The ICLR sentence on engine truth: J at the ICLR lambda 1e-4 against the horizon-only adaptive
    C-J at lambda 1e-4 (the transferred ICLR operating recipe)."""
    policies = table["policies"]
    j, cj = policies[ICLR_J], policies[ICLR_CJ]
    ratio = j["mean_macs_per_candidate"] / cj["mean_macs_per_candidate"]
    wall_ratio = (j["seconds_per_state"] / cj["seconds_per_state"]
                  if j["seconds_per_state"] and cj["seconds_per_state"] else None)
    contrast = paired(rows, [(1, ICLR_J), (-1, ICLR_CJ)], "auc")
    low, high = contrast["estimate"]["interval"]
    a_third, noninferior = ratio <= THIRD_TOLERANCE, low > -DELTA
    if not guards_ok:
        token = STOP_TOKEN
    elif a_third and noninferior:
        token = "supported"
    elif not a_third or high < -DELTA:
        token = "not_supported_by_this_experiment"
    else:
        token = STOP_TOKEN
    reading = []
    for lam in hvj.LAMBDAS:
        pid = f"J-{tag(lam)}"
        block = paired(rows, [(1, pid), (-1, ICLR_CJ)], "auc")
        reading.append({"policy": pid, "mac_ratio_to_CJ_1e-04": policies[pid]["mean_macs_per_candidate"]
                        / cj["mean_macs_per_candidate"], "auc_minus_CJ_1e-04": block,
                        "noninferior": bool(block["estimate"]["interval"][0] > -DELTA)})
    cheapest = min((r for r in reading if r["noninferior"]), key=lambda r: r["mac_ratio_to_CJ_1e-04"], default=None)
    headline = table["frontiers"][HEADLINE]["verdicts"]
    return {"sentence": ICLR_SENTENCE, "historical_issue_74": HISTORICAL_74,
            "operating_points": {"J": ICLR_J, "horizon_only_adaptive": ICLR_CJ},
            "mac_ratio": ratio, "wall_ratio": wall_ratio, "auc_contrast_J_minus_CJ": contrast,
            "a_third": bool(a_third), "noninferior": bool(noninferior), "token": token,
            "rule": plan_rules()["iclr_sentence"],
            "lambda_reading": reading, "cheapest_noninferior_J": cheapest,
            "frontier_status": {ICLR_J: headline[ICLR_J]["text"], ICLR_CJ: headline[ICLR_CJ]["text"]}}


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def reuse_fidelity(primary, timing):
    """Timing-pass digests, MACs and calls equal the retained record on every cell of every timed seed."""
    checked, mismatches = 0, []
    for entry in entries_for("primary"):
        pid = entry["id"]
        for seed in TIMING_SEEDS:
            stored, fresh = primary[(pid, seed)], timing[(pid, seed)]
            if fresh.get("typed_failure") or stored.get("typed_failure"):
                if bool(fresh.get("typed_failure")) != bool(stored.get("typed_failure")):
                    mismatches.append({"policy": pid, "seed": seed, "cell": "typed_failure"})
                continue
            for inventory in INVENTORIES:
                for member, cell in stored["inventories"][inventory].items():
                    f = fresh["inventories"][inventory].get(member)
                    checked += 1
                    same = (f is not None and f["rows_sha256"] == digest(cell["rows"])
                            and f["schedules_sha256"] == digest(cell.get("schedules"))
                            and all(f[k] == cell[k] for k in ("linear_macs", "transition_calls", "controller_calls",
                                                              "candidates")))
                    if not same:
                        mismatches.append({"policy": pid, "seed": seed, "inventory": inventory, "member": member})
    return {"cells_checked": checked, "mismatches": mismatches[:20], "mismatch_count": len(mismatches),
            "pass": bool(checked and not mismatches)}


def cross_source(cache):
    checked, mismatches = 0, []
    for (sa, aa), (sb, ab) in cross_pairs():
        for seed in SEEDS:
            a, b = cache[(sa, aa, seed)], cache[(sb, ab, seed)]
            for inventory in INVENTORIES:
                for member, cell in a["inventories"][inventory].items():
                    other = b["inventories"][inventory][member]
                    checked += 1
                    same = cell["rows"] == other["rows"] and (
                        "schedules" not in cell or "schedules" not in other or cell["schedules"] == other["schedules"])
                    if not same:
                        mismatches.append({"a": f"{sa}:{aa}", "b": f"{sb}:{ab}", "seed": seed, "member": member})
    return {"pairs": [[f"{sa}:{aa}", f"{sb}:{ab}"] for (sa, aa), (sb, ab) in cross_pairs()], "cells_checked": checked,
            "mismatches": mismatches[:20], "mismatch_count": len(mismatches), "pass": bool(checked and not mismatches)}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def load_retained():
    """Canonical records of every primary policy, plus the arms recorded by a second source."""
    cache = {}
    for entry in entries_for("primary"):
        for seed in SEEDS:
            cache[(entry["source"], entry["arm"], seed)] = retained(entry["source"], entry["arm"], seed)
    for pair in cross_pairs():
        for source, arm in pair:
            for seed in SEEDS:
                if (source, arm, seed) not in cache:
                    cache[(source, arm, seed)] = retained(source, arm, seed)
    return cache


def record_manifest(output):
    """Content hashes of every raw record and anchor store the published numbers come from."""
    paths = []
    for entry in entries_for("primary"):
        paths += [("retained-primary", retained_path(entry["source"], entry["arm"], s)) for s in SEEDS]
    for pair in cross_pairs():
        paths += [("retained-cross-check", retained_path(source, arm, s)) for source, arm in pair for s in SEEDS]
    paths += [("new-timing", timing_path(output, e["id"], s)) for s in TIMING_SEEDS for e in entries_for("primary")]
    paths += [("new-secondary", secondary_path(output, e["id"], s)) for e in entries_for("secondary") for s in SEEDS]
    paths += [("anchors", p) for p in (gb.anchors_path(GATE_B), hb.f77_anchors_path(P105), *secondary_e99_paths(),
                                        f77_store_path(output))]
    paths += [("verdicts", gb.FIT_VERDICTS), ("verdicts", gb.CAMPAIGN / "branches.json")]
    seen, files = set(), []
    for role, path in paths:
        rel = str(Path(path).relative_to(ROOT))
        if rel in seen:
            continue
        seen.add(rel)
        files.append({"path": rel, "role": role, "bytes": Path(path).stat().st_size, "sha256": sha256_of(path)})
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "files": files,
            "roles": {role: sum(1 for f in files if f["role"] == role) for role in sorted({f["role"] for f in files})}}


def compute_tables(output, plan):
    output = Path(output)
    cache = load_retained()
    primary = {(e["id"], s): cache[(e["source"], e["arm"], s)] for e in entries_for("primary") for s in SEEDS}
    timing = {(e["id"], s): read_timing(output, e["id"], s) for e in entries_for("primary") for s in TIMING_SEEDS}
    secondary = {(e["id"], s): read_secondary(output, e["id"], s) for e in entries_for("secondary") for s in SEEDS}
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    new_records = list(timing.values()) + list(secondary.values())
    scored_at = sorted(r["scored_at"] for r in new_records)
    universe_counts = {i: len(c["mixed"]) for i, c in secondary_cells().items()}
    store = load_secondary_anchors(output)
    guards = {
        "G1_reuse_fidelity": reuse_fidelity(primary, timing),
        "G2_cross_source_identity": cross_source(cache),
        "G3_scoring_policy": {"pass": all(r.get("scoring_policy") == harness.SCORING_POLICY
                                          for r in list(cache.values()) + new_records)},
        "G4_input_binding": {"pass": True, "inputs": len(plan["inputs"]),
                             "note": "load_plan re-hashed every frozen input before this computation"},
        "G5_chronology": {"plan_frozen_at": plan["frozen_at"], "first_new_record": scored_at[0],
                          "pass": plan["frozen_at"] <= scored_at[0]},
        "G6_secondary_inputs": {"mixed_cells": universe_counts, "frozen": plan["secondary"]["mixed_cells"],
                                "pass": bool(universe_counts == plan["secondary"]["mixed_cells"]
                                             and all(m in store["E99"] and m in store["F77"]
                                                     for m in secondary_members()))},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    sets = {"primary": {}, "secondary": {}}
    rows_of = {}
    primary_cells_ = primary_cells()
    secondary_cells_ = secondary_cells()
    for inventory in INVENTORIES:
        sets["primary"][inventory], rows_of[("primary", inventory)] = inventory_table(
            primary, primary_cells_, inventory, stats[inventory], "primary", timing, guards_ok)
        sets["secondary"][inventory], rows_of[("secondary", inventory)] = inventory_table(
            secondary, secondary_cells_, inventory, stats[inventory], "secondary", None, guards_ok)
    iclr = {s: iclr_assessment(sets[s][PRIMARY], rows_of[(s, PRIMARY)], guards_ok) for s in SETS}
    phases = ledger_phases(output)
    smoke = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    grouped = {}
    for name, block in all_phases.items():
        key = name.split("-")[0] if name.startswith(("timing", "secondary")) else name
        g = grouped.setdefault(key, {"wall_seconds": 0.0, "gpu_seconds": 0.0, "phases": 0})
        g["wall_seconds"] += block["wall_seconds"]
        g["gpu_seconds"] += block["gpu_seconds"]
        g["phases"] += 1
    reused = {source: sum(cache[(src, arm, s)].get("seconds", 0.0) for (src, arm, s) in cache if src == source)
              for source in ("101", "114", "105", "gate_b")}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"], "guards": guards,
        "guards_pass": guards_ok, "policies": [dict(e) for e in catalogue()] + [{"id": PRIOR, "group": "prior",
                                                                                 "source": "gate_b rule", "arm": None}],
        "excluded": EXCLUDED, "amendment": AMENDMENT, "evaluation": gb.evaluation_accounting(*gb.evaluation_set()),
        "secondary_mixed_cells": universe_counts, "sets": sets, "iclr_sentence": iclr,
        "compute": {"phases": grouped, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "reused_record_scoring_seconds": reused,
                    "hardware": "4 x RTX 5090 (one process per GPU) instead of the #85 RTX 3090"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def mmacs(value):
    return "n/a" if value is None else ("0" if value == 0 else f"{value / 1e6:.1f}M")


def secs(value):
    return "n/a" if value is None else ("0" if value == 0 else f"{value:.3f}")


def cell(block):
    return fmt_interval(block) if block else "n/a"


def short_verdict(v):
    if v.get("undefined"):
        return "undefined"
    if v["on_frontier"]:
        return f"**on the frontier** ({v['share']:.2f})"
    return f"dominated by {v['dominated_by']} ({fmt_interval(v['contrast']['estimate'])}, {v['token']}; {v['share']:.2f})"


def policy_table(add, table):
    head = table["frontiers"][HEADLINE]
    add("| Policy | Group | AUC | Top-3 | MACs / cand. | Wall s / state | All-tied cells | Tied pairs | "
        "Frontier (AUC×MACs, share; AUC×wall; top-3×MACs; top-3×wall) | Verdict (AUC × MACs) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for pid, p in table["policies"].items():
        flags = []
        for combo in COMBOS:
            f = table["frontiers"][combo]
            flags.append(("**on**" if pid in f["on"] else "off") + (f" {f['shares'][pid]:.2f}" if combo == HEADLINE else ""))
        add(f"| {pid} | {GROUPS[p['group']]} | {cell(p['auc'])} | {cell(p['top3'])} | "
            f"{mmacs(p['mean_macs_per_candidate'])} | {secs(p['seconds_per_state'])} | {p['tied_cells']}/{p['cells']} | "
            f"{p['tied_pairs']}/{p['pairs']} | {'; '.join(flags)} | {short_verdict(head['verdicts'][pid])} |")


def findings_md(plan, compute):
    lines = []
    add = lines.append
    grid = compute["sets"]["primary"][PRIMARY]
    second = compute["sets"]["secondary"][PRIMARY]
    add("# Issue #103: engine-truth compute–quality frontier of the request policies")
    add("")
    add(f"Frozen {plan['frozen_at']}. Validation: `{VALIDATION_COMMAND}`. Guards "
        f"{'all pass' if compute['guards_pass'] else 'FAILED: every token is ' + STOP_TOKEN}. Zero engine seconds. "
        "Intervals are member (level)-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. "
        f"δ = {DELTA}: `supported` iff mean ≥ δ and lower > 0; `not_supported_by_this_experiment` iff upper < δ; "
        f"otherwise `{STOP_TOKEN}`. A positive difference below δ is not a win.")
    add("")
    add("## The ICLR sentence on engine truth")
    add("")
    add(f"Sentence: “{ICLR_SENTENCE}”. The #74 proxy basis: hybrid adaptive 118.7M vs continuous adaptive 356.0M MACs "
        "per candidate-state (ratio 0.33), regret on the construct-invalidated count cost.")
    add("")
    for set_name in SETS:
        a = compute["iclr_sentence"][set_name]
        add(f"- **{set_name}** ({'Gate-B grid' if set_name == 'primary' else '#93 grid'}): {ICLR_J} / {ICLR_CJ} MAC "
            f"ratio {a['mac_ratio']:.3f}, wall ratio {num(a['wall_ratio'], 3)}; AUC {ICLR_J} − {ICLR_CJ} = "
            f"{fmt_interval(a['auc_contrast_J_minus_CJ']['estimate'])}; a third: {a['a_third']}; non-inferior "
            f"(lower > −δ): {a['noninferior']} → **`{a['token']}`**.")
        add(f"  - Frontier status: {ICLR_J} {a['frontier_status'][ICLR_J]}; {ICLR_CJ} {a['frontier_status'][ICLR_CJ]}.")
        c = a["cheapest_noninferior_J"]
        add(f"  - Other λ (descriptive): the cheapest J point non-inferior to {ICLR_CJ} is "
            + (f"{c['policy']} at MAC ratio {c['mac_ratio_to_CJ_1e-04']:.3f} "
               f"(AUC difference {fmt_interval(c['auc_minus_CJ_1e-04']['estimate'])})." if c else "none."))
    add("")
    add(f"Rule (frozen): {plan['rules']['iclr_sentence']}")
    add("")
    add(f"## Primary set: #109 evaluation split, Gate-B grid ({grid['members']} levels × {len(SEEDS)} seeds)")
    add("")
    add("MACs: linear MACs per candidate rollout to step 225 (predictor + policy calls), retained records, all seeds. "
        "Wall: synchronized rollout seconds per state (all candidates batched), uniform timing pass over all three seeds, "
        "one process per RTX 5090, warm-up cell discarded. Frontier column: on/off for AUC×MACs (with the bootstrap "
        "frontier share), AUC×wall, top-3×MACs, top-3×wall. Verdict: the strongest dominator (highest AUC among "
        "policies with no more MACs), dominator − policy AUC, δ token, bootstrap frontier share.")
    add("")
    policy_table(add, grid)
    add("")
    add("## Frontier sets per inventory and axis")
    add("")
    add("| Set | Inventory | Cells | AUC × MACs | AUC × wall | Top-3 × MACs | Top-3 × wall |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for set_name in SETS:
        for inventory in INVENTORIES:
            t = compute["sets"][set_name][inventory]
            add(f"| {set_name} | {inventory} | {t['members']} × {len(SEEDS)} | "
                + " | ".join(", ".join(t["frontiers"][c]["on"]) for c in COMBOS) + " |")
    add("")
    add(f"## Declared secondary set: #93 / #94 / #87 inventories, #93 grid ({second['members']} members × {len(SEEDS)} "
        "seeds; DESCRIPTIVE)")
    add("")
    add("Wall: this scoring's synchronized rollout seconds per state, all seeds. OL-pe / OL-dec are primary-only.")
    add("")
    policy_table(add, second)
    add("")
    add("## Guards")
    add("")
    for name, g in compute["guards"].items():
        extra = {k: v for k, v in g.items() if k in ("cells_checked", "mismatch_count", "first_new_record", "inputs")}
        add(f"- {name}: {'pass' if g['pass'] else 'FAIL'} {json.dumps(extra) if extra else ''}".rstrip())
    add("")
    add("## Compute")
    add("")
    c = compute["compute"]
    phases = ", ".join(f"{k} {v['gpu_seconds']:.0f}s" for k, v in c["phases"].items())
    add(f"- New: {c['gpu_seconds']:.0f} GPU-allocation seconds, {c['wall_seconds']:.0f} process-wall seconds "
        f"(phases: {phases}); engine 0 s.")
    add("- Reused retained scoring (not re-charged): "
        + ", ".join(f"#{k if k != 'gate_b' else 'Gate-B'} {v:.0f}s" for k, v in c["reused_record_scoring_seconds"].items())
        + ".")
    add(f"- Hardware: {c['hardware']}.")
    add("")
    add("## Post-freeze amendment (wall axis only)")
    add("")
    for key in ("at", "after", "what", "why", "unchanged", "superseded", "execution"):
        add(f"- {key}: {AMENDMENT[key]}.")
    add("")
    add("## Excluded")
    add("")
    for name, why in EXCLUDED.items():
        add(f"- {name}: {why}.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    def verdicts(set_name):
        t = compute["sets"][set_name][PRIMARY]
        return {pid: t["frontiers"][HEADLINE]["verdicts"][pid]["text"] for pid in t["policies"]}

    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "guards_pass": compute["guards_pass"],
            "amendment": {k: AMENDMENT[k] for k in ("at", "what")},
            "iclr_sentence": {s: {k: compute["iclr_sentence"][s][k] for k in ("token", "mac_ratio", "wall_ratio",
                                                                               "a_third", "noninferior")}
                              | {"auc_J_minus_CJ": compute["iclr_sentence"][s]["auc_contrast_J_minus_CJ"]["estimate"]}
                              for s in SETS},
            "frontiers": {s: {i: {c: compute["sets"][s][i]["frontiers"][c]["on"] for c in COMBOS} for i in INVENTORIES}
                          for s in SETS},
            "verdicts_grid": {s: verdicts(s) for s in SETS},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")}}


CSV_FIELDS = ("set", "inventory", "policy", "group", "source", "auc", "auc_low", "auc_high", "top3", "top3_low",
              "top3_high", "chance_top3", "macs_per_candidate", "transitions_per_candidate", "policy_calls_per_candidate",
              "wall_seconds_per_state", "recorded_seconds_per_state", "cells", "all_tied_cells", "tied_pairs", "pairs",
              "nonfinite_slots", "typed_failures", *(f"on_{c}" for c in COMBOS), *(f"share_{c}" for c in COMBOS),
              "verdict_auc_macs")


def table_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    sources = {e["id"]: e["source"] for e in catalogue()}
    for set_name in SETS:
        for inventory in INVENTORIES:
            t = compute["sets"][set_name][inventory]
            for pid, p in t["policies"].items():
                auc, top3 = p["auc"] or {}, p["top3"] or {}
                writer.writerow([
                    set_name, inventory, pid, p["group"], sources.get(pid, "gate_b rule"),
                    auc.get("mean"), (auc.get("interval") or [None, None])[0], (auc.get("interval") or [None, None])[1],
                    top3.get("mean"), (top3.get("interval") or [None, None])[0], (top3.get("interval") or [None, None])[1],
                    p["chance_top3"], p["mean_macs_per_candidate"], p["mean_transitions_per_candidate"],
                    p["mean_policy_calls_per_candidate"], p["seconds_per_state"], p["recorded_seconds_per_state"],
                    p["cells"], p["tied_cells"], p["tied_pairs"], p["pairs"], p["nonfinite"], p["typed_failures"],
                    *(int(pid in t["frontiers"][c]["on"]) for c in COMBOS),
                    *(t["frontiers"][c]["shares"][pid] for c in COMBOS),
                    t["frontiers"][HEADLINE]["verdicts"][pid]["text"]])
    return buffer.getvalue()


def frontier_axes(ax, table, combo, title, label_frontier=True):
    from matplotlib import ticker
    xkey, field = COMBOS[combo]
    yblock = Y_BLOCK[field]
    policies = table["policies"]
    positive = [p[xkey] for p in policies.values() if p[xkey]]
    floor = min(positive) / 4
    on = table["frontiers"][combo]["on"]
    seen = set()
    for pid, p in policies.items():
        if p[xkey] is None or p[yblock] is None:
            continue
        color, marker = GROUP_STYLE[p["group"]]
        x = p[xkey] if p[xkey] > 0 else floor
        lo, hi = p[yblock]["interval"]
        ax.errorbar([x], [p[yblock]["mean"]], yerr=[[p[yblock]["mean"] - lo], [hi - p[yblock]["mean"]]], fmt="none",
                    ecolor=color, alpha=0.35, linewidth=0.8)
        filled = marker not in ("x", "_", "+")
        ax.scatter([x], [p[yblock]["mean"]], color=color, marker=marker, s=60 if pid in on else 22,
                   edgecolors=("black" if pid in on else "none") if filled else None,
                   linewidths=0.8 if filled else 1.4,
                   label=GROUPS[p["group"]] if p["group"] not in seen else None, zorder=3)
        seen.add(p["group"])
        if pid in on and label_frontier:
            ax.annotate(pid, (x, p[yblock]["mean"]), fontsize=6, xytext=(4, 4 + 9 * on.index(pid)),
                        textcoords="offset points", arrowprops={"arrowstyle": "-", "linewidth": 0.4})
    pts = sorted((policies[p][xkey] if policies[p][xkey] > 0 else floor, policies[p][yblock]["mean"]) for p in on)
    if pts:
        ax.step([q[0] for q in pts], [q[1] for q in pts], where="post", color="black", linewidth=0.8, zorder=2)
    ax.axvline(floor * 1.6, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(("linear MACs per candidate-state" if xkey == "mean_macs_per_candidate" else "wall seconds per state")
                  + " (left of the dotted line: 0, prior)")
    ax.set_ylabel("tie-free AUC" if field == "auc" else "top-3 hit rate")
    ax.set_title(title, fontsize=9)


def figure_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    return buffer.getvalue()


def figure_set(compute, set_name):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    table = compute["sets"][set_name][PRIMARY]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    for ax, combo in zip(axes.flat, ("auc_macs", "auc_wall", "top3_macs", "top3_wall")):
        frontier_axes(ax, table, combo, f"{combo.replace('_', ' × ')}; frontier outlined, step line")
    axes[0][0].legend(fontsize=6, loc="lower right")
    name = ("primary: #109 evaluation split, Gate-B grid" if set_name == "primary"
            else "secondary: #93 grid (#96 universe), DESCRIPTIVE")
    fig.suptitle(f"#103 engine-truth frontier, {name} ({table['members']} members × {len(SEEDS)} seeds); "
                 "bars = DESCRIPTIVE 95% member-clustered intervals")
    fig.tight_layout()
    out = figure_png(fig)
    plt.close(fig)
    return out


def figure_inventories(compute):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=100)
    for r, set_name in enumerate(SETS):
        for c, inventory in enumerate(INVENTORIES):
            t = compute["sets"][set_name][inventory]
            frontier_axes(axes[r][c], t, HEADLINE, f"{set_name} {inventory} ({t['members']} members): AUC × MACs")
    axes[0][0].legend(fontsize=6, loc="lower right")
    fig.suptitle("#103 AUC × MACs frontier per inventory (row 1: #109 evaluation split; row 2: #93/#94/#87 inventories)")
    fig.tight_layout()
    out = figure_png(fig)
    plt.close(fig)
    return out


def rendered(output, plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "frontier-table.csv": table_csv(compute),
            "frontier-primary.png": figure_set(compute, "primary"),
            "frontier-secondary.png": figure_set(compute, "secondary"),
            "frontier-inventories.png": figure_inventories(compute),
            "manifest.json": json_text(record_manifest(output))}


def as_bytes(value):
    return value if isinstance(value, bytes) else value.encode()


# ---------------------------------------------------------------------------
# smoke (pre-freeze; retained primary outputs and secondary mechanics, no new outcome statistic)
# ---------------------------------------------------------------------------

def synthetic_checks():
    """Pareto and bootstrap-share mechanics on synthetic data with known answers."""
    x = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 1.0])
    y = np.array([0.5, 0.7, 0.7, 0.8, 0.6, 0.7])
    on = pareto_mask(x, y).tolist()
    expected = [True, True, False, True, False, True]
    rows = [{"member": f"m{k}", "seed": 0, "views": {"a": {"auc": 0.8 + 0.01 * k}, "b": {"auc": 0.6 + 0.01 * k}}}
            for k in range(6)]
    shares = frontier_shares(rows, ["a", "b"], np.array([1.0, 2.0]), "auc", draws=200).tolist()
    single = frontier_shares(rows, ["a"], np.array([1.0]), "auc", draws=200).tolist()
    return {"control": "frontier_mechanics_synthetic", "pareto": on, "expected": expected, "shares": shares,
            "ok": bool(on == expected and shares == [1.0, 0.0] and single == [1.0])}


def smoke_cells(cells):
    grid = sorted(cells[PRIMARY]["mixed"])[:SMOKE_GRID_CELLS]
    return [(PRIMARY, m) for m in grid] + [(i, sorted(cells[i]["mixed"])[0]) for i in INVENTORIES if i != PRIMARY]


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    checks = [synthetic_checks()]
    cells = primary_cells()
    sample = smoke_cells(cells)
    store = primary_anchors()
    reuse = []
    with harness.deterministic_scoring():
        costs = dap.endpoint_costs()
        for entry in entries_for("primary"):
            arm = load_arm(entry, TIMING_SEED, cells)
            record = retained(entry["source"], entry["arm"], TIMING_SEED)
            same, seconds = arm["model"] is not None, 0.0
            for inventory, member in sample:
                fresh = run_cell(arm, inventory, member, cells[inventory]["mixed"][member], store, costs, TIMING_SEED)
                same &= equal_block(fresh, record["inventories"][inventory][member])
                seconds += fresh["gpu_seconds"]
            reuse.append({"policy": entry["id"], "source": entry["source"], "arm": entry["arm"], "equal": bool(same),
                          "seconds": seconds})
            torch.cuda.empty_cache()
        checks.append({"control": "reuse_equality_sample", "seed": TIMING_SEED,
                       "cells": [list(c) for c in sample], "policies": len(reuse),
                       "unequal": [r["policy"] for r in reuse if not r["equal"]],
                       "ok": all(r["equal"] for r in reuse)})
        # secondary set: inputs and mechanics only (no verdict is read)
        scells = secondary_cells()
        members = secondary_members()
        e99 = secondary_e99()
        first = members[0]
        adapter_e = s99.load_adapter(v1.SLOT_FIX, "E")
        reparsed = wlc.encode_carrier(adapter_e, frozen_anchor(first), DEVICE).float().cpu()
        difference = float((reparsed - e99[first]).abs().max())
        checks.append({"control": "secondary_e99_anchor_reparse", "member": first, "max_abs_difference": difference,
                       "members_with_e99": sum(m in e99 for m in members), "members": len(members),
                       "ok": bool(difference <= harness.PARSE_TOLERANCE and all(m in e99 for m in members))})
        adapter_f = wlc.load_adapter(DEVICE)
        f1 = wlc.encode_carrier(adapter_f, frozen_anchor(first), DEVICE).float().cpu()
        f2 = wlc.encode_carrier(adapter_f, frozen_anchor(first), DEVICE).float().cpu()
        checks.append({"control": "secondary_f77_parse_deterministic", "member": first,
                       "ok": bool(torch.equal(f1, f2) and f1.shape == e99[first].shape)})
        temp = {"E99": e99, "F77": {first: f1}}
        inventory = next(i for i in INVENTORIES if first in scells[i]["mixed"])
        mechanics = []
        for pid in ("F-15-macro", ICLR_J, ICLR_CJ, "J-dec-1e-01", "J_diag", "H-5-micro", "TAWM-15", "TAWM-adaptive"):
            entry = entry_of(pid)
            arm = load_arm(entry, TIMING_SEED, scells)
            block = run_cell(arm, inventory, first, scells[inventory]["mixed"][first], temp, costs, TIMING_SEED)
            mechanics.append({"policy": pid, "candidates": block["candidates"],
                              "rows": len(block["rows"]), "seconds": block["gpu_seconds"]})
        checks.append({"control": "secondary_mechanics", "member": first, "inventory": inventory,
                       "policies": mechanics, "reading": "rollouts and costs only; no verdict or AUC computed",
                       "ok": all(m["rows"] == m["candidates"] for m in mechanics)})
    counts = {i: len(c["mixed"]) for i, c in scells.items()}
    checks.append({"control": "secondary_universe", "mixed_cells": counts, "members": len(members),
                   "ok": bool(all(counts.values()))})
    cache = load_retained()
    cross = cross_source(cache)
    checks.append({"control": "cross_source_identity_retained", "cells_checked": cross["cells_checked"],
                   "mismatch_count": cross["mismatch_count"], "ok": cross["pass"]})
    del cache
    per_cell = {r["policy"]: r["seconds"] / len(sample) for r in reuse}
    primary_cells_ = sum(len(cells[i]["mixed"]) for i in INVENTORIES)
    secondary_cells_ = sum(counts.values())
    timing_seconds = sum(per_cell.values()) * primary_cells_
    secondary_seconds = sum(s for p, s in per_cell.items() if p not in PRIMARY_ONLY) * secondary_cells_ * len(SEEDS)
    projection = {"timing_rollout_seconds": timing_seconds, "secondary_rollout_seconds": secondary_seconds,
                  "overhead_seconds": 15.0 * (len(per_cell) + len(per_cell) * len(SEEDS)),
                  "gpus": 4}
    projection["total_gpu_seconds"] = sum(v for k, v in projection.items() if k.endswith("seconds"))
    projection["elapsed_hours_on_4_gpus"] = projection["total_gpu_seconds"] / 4 / 3600
    ok = all(c["ok"] for c in checks)
    wall = time.monotonic() - began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks, "reuse": reuse,
                "projection": projection, "wall_seconds": wall,
                "reading": ("re-scored retained primary outputs (published before this ticket) on a cell sample; "
                            "secondary-set anchors and rollout mechanics without verdicts; synthetic frontier checks"),
                "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s); projection {projection['total_gpu_seconds']:.0f} GPU s")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: no freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "runner_issue_101": ROOT / dap.RUNNER, "runner_issue_114": ROOT / hvj.RUNNER,
        "runner_issue_105": ROOT / hb.RUNNER, "runner_gate_b": ROOT / gb.RUNNER, "runner_issue_112_v1": ROOT / v1.RUNNER,
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "request_policy_module": ROOT / dap.MODULE, "request_selectors_module": ROOT / hvj.SELECTORS,
        "hierarchical_module": ROOT / hb.MODULE, "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "gate_b_anchors": gb.anchors_path(GATE_B), "issue_105_f77_anchors": hb.f77_anchors_path(P105),
        **{f"issue_112_v1_e99_anchors_{i}": p for i, p in enumerate(secondary_e99_paths())},
        "issue_112_fit_verdicts": gb.FIT_VERDICTS, "issue_112_retrain_handoff": gb.HANDOFF,
        "issue_104_campaign_branches": gb.CAMPAIGN / "branches.json", "issue_101_selection": dap.selection_path(P101),
        "issue_96_anchor_retention": p94.NINETY_THREE / "anchor_retention.json",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_{n}_{c['seed']}": gb.member_path(n, c["seed"]) for n in gb.UNITS for c in gb.checkpoints(n)})
    names.update({f"policy_101_{k}_{tag(lam)}_{s}": dap.policy_path(P101, k, lam, s) for k, lam, s in dap.groups()})
    names.update({f"policy_114_{f}_{tag(lam)}_{s}": hvj.policy_path(P114, f, lam, s) for f, lam, s in hvj.groups()})
    names.update({f"hierarchical_{s}": hb.member_path(P105, s) for s in SEEDS})
    names.update(hb.external_files())
    for entry in entries_for("primary"):
        for s in SEEDS:
            names[f"record_{entry['source']}_{entry['arm']}_{s}"] = retained_path(entry["source"], entry["arm"], s)
    for pair in cross_pairs():
        for source, arm in pair:
            for s in SEEDS:
                names[f"record_{source}_{arm}_{s}"] = retained_path(source, arm, s)
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def plan_rules():
    return {
        "axes": ("x: linear MACs per candidate rollout to step 225 (predictor transitions + policy calls; #96/#101 "
                 "accounting, nonlinear work excluded), pooled over seeds and cells; or wall seconds per state = "
                 "CUDA-synchronized host elapsed seconds of one state's batched rollout (all candidates), measured in a "
                 f"uniform timing pass (primary: seed {TIMING_SEED}, every cell; secondary: this scoring, all seeds), one "
                 "process per RTX 5090 after a discarded warm-up cell. The prior has 0 model MACs and is not timed (0). "
                 "y: tie-free AUC (member-mean of cell AUCs, m99.view) or top-3 hit rate (#114 view)."),
        "pareto": ("point Pareto set per (set, inventory, axis pair): a policy is dominated iff another has x <= and y >= "
                   "with at least one strict; identical points do not dominate each other."),
        "bootstrap_frontier_share": ("members (levels) resampled jointly for all policies with the p94 generator "
                                     "(PCG64 7201, 10000 draws); each policy's draw value = mean of its defined "
                                     "cluster means among the drawn members; x held fixed (compute uncertainty not "
                                     "modelled); share = fraction of draws on the Pareto set. DESCRIPTIVE."),
        "verdict": ("per policy (AUC x MACs, and stored for every axis pair): 'on the frontier' with its share, or "
                    "'dominated by Q' with Q = the dominator with the highest AUC (ties: fewer MACs), the paired "
                    "Q - P difference (member-clustered) and its delta token; the delta rule is the only wording for "
                    "differences."),
        "iclr_sentence": (f"'{ICLR_SENTENCE}' on the primary grid with the transferred ICLR operating recipe: "
                          f"J = {ICLR_J} against the horizon-only adaptive C-J = {ICLR_CJ}. 'a third' holds iff J's MACs "
                          f"per candidate-state <= {THIRD_TOLERANCE} x C-J's; 'matches' holds iff the paired J - C-J AUC "
                          f"lower bound > -{DELTA} (non-inferiority at delta). supported iff both; "
                          f"not_supported_by_this_experiment iff 'a third' fails or the upper bound < -{DELTA}; else "
                          f"{STOP_TOKEN}; any failed guard -> {STOP_TOKEN}. The other J lambdas and the frontier status "
                          "are reported DESCRIPTIVE. The same rule is reported on the secondary set (DESCRIPTIVE). If "
                          "the primary token is not 'supported', a MANUSCRIPT ticket is opened (#103 acceptance)."),
        "guards": ("G1 reuse fidelity: the timing pass reproduces every retained primary record on every seed-"
                   f"{TIMING_SEED} cell (rows, schedules, MACs, calls); G2 arms recorded by two sources are identical; "
                   "G3 every record under scoring_harness.SCORING_POLICY; G4 frozen inputs unchanged; G5 freeze before "
                   "the first new record; G6 secondary inputs equal the frozen counts and anchors. A failed guard turns "
                   f"every token (dominance contrasts and the ICLR sentence) into {STOP_TOKEN}."),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    branches, cells = gb.evaluation_set()
    scells = secondary_cells()
    timing, secondary = score_tasks()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 103,
        "related_issues": [74, 87, 93, 94, 96, 101, 104, 105, 108, 109, 111, 112, 114],
        "frozen_at": frozen_at, "frozen_before_new_scoring": True,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "validation_command": VALIDATION_COMMAND,
        "smoke_evidence": {"ok": smoke_evidence["ok"], "run_at": smoke_evidence["run_at"],
                           "checks": [{"control": c["control"], "ok": c["ok"]} for c in smoke_evidence["checks"]],
                           "projection": smoke_evidence["projection"]},
        "primary": {"cells": "Gate-B mixed cells of the #109 evaluation split (#104 campaign pig-removal verdicts)",
                    "anchors": "Gate-B anchor store (E99); #105 F77 store for the #78 ports",
                    "mixed_cells": {i: len(cells[i]["mixed"]) for i in INVENTORIES}, "primary_inventory": PRIMARY,
                    "descriptive": [i for i in INVENTORIES if i != PRIMARY],
                    "reuse": ("every policy's canonical retained record (source per policy in 'policies'); retained "
                              "records score exactly these cells, anchors, predictors and policies under the #111 "
                              "contract; reuse fidelity is G1"),
                    "accounting": gb.evaluation_accounting(branches, cells)},
        "secondary": {"cells": ("#96 universe: #93 Arm B grid (16), #93 Arm A offset (11), #87 + #89 angle (9-13; "
                                "#89 typed-unmeasurable states excluded), #94 power (20); first_shot_success engine "
                                "verdicts; mixed member cells"),
                      "anchors": ("E99: #112 v1 parses of p94.frozen_anchor (retained); F77 (#78 ports): parsed here "
                                  "(--anchors) from the same frames"),
                      "mixed_cells": {i: len(scells[i]["mixed"]) for i in INVENTORIES},
                      "members": secondary_members(), "descriptive": True,
                      "disclosure": ("N1 lineages: the E99 encoder (#99) was fitted on the 11 non-held-out lineages and "
                                     "the #78 ports were trained on N1 lineages; the LC predictors and all policies "
                                     "never saw them"),
                      "prior": "Gate-B rule (fit-split success frequency of the identical action; unseen -> 0)"},
        "policies": [dict(e) for e in catalogue()], "prior": "Gate-B no-model action prior", "excluded": EXCLUDED,
        "groups": GROUPS, "tasks": {"timing": len(timing), "secondary": len(secondary)},
        "statistics": {"cost": PRIMARY_COST, "bootstrap": "member-clustered percentile, 10000 draws, PCG64 7201",
                       "delta": DELTA, "label": "DESCRIPTIVE"},
        "rules": plan_rules(),
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "engine_seconds": 0,
                 "stop": STOP_TOKEN},
        "known_before_freeze": ("every primary AUC, MAC and recorded wall value (published by #101/#114/#105 and Gate "
                                "B), the #74 proxy table; no policy output on the secondary set with the Gate-B "
                                "predictors existed; the frontier, shares, verdicts and the ICLR-sentence token "
                                "had not been computed"),
        "hardware": "4 x RTX 5090 (one process per GPU) instead of the #85 RTX 3090",
        "inputs": input_bindings(output),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-103 protocol")
    if not plan.get("frozen_before_new_scoring") or not plan["smoke_evidence"].get("ok"):
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
    if (output / "records").exists():
        raise ValueError("records exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    (output / "plan.json").write_text(json_text(frozen_plan(output, utc_now(), evidence)))
    load_plan(output)
    log("frozen plan published")
    return 0


# ---------------------------------------------------------------------------
# publication and validation
# ---------------------------------------------------------------------------

def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, content in rendered(output, plan, compute).items():
        (output / name).write_bytes(as_bytes(content))
    log(f"published: guards {'pass' if compute['guards_pass'] else 'FAIL'}; ICLR sentence "
        f"{compute['iclr_sentence']['primary']['token']}; grid frontier "
        f"{compute['sets']['primary'][PRIMARY]['frontiers'][HEADLINE]['on']}")
    return 0


def spot_checks(output):
    """Re-score the secondary records of J and C-J (lambda 1e-4), first seed, in this process (exact)."""
    cells = secondary_cells()
    store = load_secondary_anchors(output)
    same = True
    with harness.deterministic_scoring():
        costs = dap.endpoint_costs()
        for pid in (ICLR_J, ICLR_CJ):
            arm = load_arm(entry_of(pid), TIMING_SEED, cells)
            record = read_secondary(output, pid, TIMING_SEED)
            for inventory, member, cell_ in ordered(cells):
                fresh = run_cell(arm, inventory, member, cell_, store, costs, TIMING_SEED)
                same &= equal_block(fresh, record["inventories"][inventory][member])
    return {"policies": [ICLR_J, ICLR_CJ], "seed": TIMING_SEED, "pass": bool(same)}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, content in rendered(output, plan, compute).items():
        if (output / name).read_bytes() != as_bytes(content):
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the secondary records: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): tables, CSV, figures and record manifest re-derived and "
        f"byte-compared; {', '.join(spot['policies'])} secondary seed {spot['seed']} re-scored exactly; ICLR sentence "
        f"{compute['iclr_sentence']['primary']['token']}")
    return 0


def dry_run(output):
    timing, secondary = score_tasks()
    cells = primary_cells()
    log(f"output {Path(output)}; policies {len(catalogue())} + prior; timing tasks {len(timing)}, secondary tasks "
        f"{len(secondary)}; primary mixed {({i: len(cells[i]['mixed']) for i in INVENTORIES})}; secondary mixed "
        f"{({i: len(c['mixed']) for i, c in secondary_cells().items()})}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "anchors": anchors, "score": score,
         "publish": publish, "validate": validate}
SHARDED = ("score",)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode in SHARDED:
        return MODES[mode](OUTPUT, args.workers, args.worker)
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
