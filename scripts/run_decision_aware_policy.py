"""Issue-101: decision-aware request policy, scored once on the #104 sealed #109 evaluation split.

Binding runner module: scripts/run_decision_aware_policy.py
Exact validation command: python -u -m scripts.run_decision_aware_policy --validate

#101 asks whether a request policy trained on decision value (launch ordering) instead of prediction
error makes jointness (J vs M) and state dependence (J vs OL) carry ranking value, and whether it beats
the best fixed request in hindsight. This runner

1. builds the controller pool from the #109 ``policy`` split captured by the #104 campaign (stream
   ``issue109-policy``: 120 levels, 3480 complete branches, grid + angle, engine pig-removal verdicts):
   E99 carriers of every shot (``--data``), each level's E99 anchor and the mixed grid / angle cells
   (``--assemble``). The fit split stays predictor-only; the evaluation split is held out;
2. trains, per #112 v3 LC hybrid predictor (fold 'all', seeds 20260908/09/10, the Gate-B
   checkpoints), three policy kinds with the shared module ``world_model/training/request_policy.py``
   over a lambda sweep (``--train``, sharded over the GPUs):
   - ``pe``: ICLR prediction-error DP labels (#77 path: first 60-frame window of every pool shot,
     2 DAgger rounds x 1800 steps, batch 128, AdamW 1e-3);
   - ``dec``: decision-valued labels (per visited rollout state from the level anchor, the change in
     the cell's pairwise success-vs-failure ranking loss of each request + lambda MAC; continuation
     F-15-macro in round 0 and the round-0 policy in the DAgger round), same MLP and recipe;
   - ``pg``: REINFORCE on per-cell AUC minus lambda MAC over imagined rollouts (no engine);
3. selects the primary lambda of ``dec`` and ``pg`` on the policy split only (``--select``);
4. scores once, in processes that did not train, under ``scoring_harness.deterministic_scoring()``:
   the nine fixed requests, C* (LC continuous C-F-15, reference), every policy (joint argmax = J),
   M and OL of J (ICLR, lambda 1e-4) and J_dec, the cross-fit r-dagger and F_hindsight contrasts, the
   no-model action prior and J_dec's wrong-anchor control, on the Gate-B evaluation anchors (``--score``);
5. publishes compute.json, summary.json and findings.md.

Modes: --dry-run --smoke --prepare --data --assemble --train --select --score --publish --validate.
--data, --train and --score take --workers N --worker K (one process per GPU; CUDA_VISIBLE_DEVICES set
by the caller). Zero engine seconds. Intervals: member (level)-clustered percentile bootstrap (10000
draws, PCG64 7201), DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import functools
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_lookahead_control as wlc
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import request_policy as rp
from world_model.training.cnn_hybrid import PAIRS

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-101-decision-aware-policy-v1"
CAMPAIGN = rt.CAMPAIGN
STREAM = "issue109-policy"
GATE_B = gb.OUTPUT
RETRAIN = rt.OUTPUT
ISSUE77 = ROOT / ".local-artifacts/issue-77-n1-dynamics-v1"

IDENTITY = "issue-101-decision-aware-policy-v1"
SCHEMA_PLAN = "issue_101_decision_aware_policy_plan_v1"
SCHEMA_SMOKE = "issue_101_decision_aware_policy_smoke_v1"
SCHEMA_SHOT = "issue_101_policy_shot_v1"
SCHEMA_POOL = "issue_101_policy_pool_v1"
SCHEMA_POLICY = "issue_101_request_policy_v1"
SCHEMA_REPORT_TRAIN = "issue_101_training_report_v1"
SCHEMA_SELECTION = "issue_101_selection_v1"
SCHEMA_RECORD = "issue_101_record_v1"
SCHEMA_COMPUTE = "issue_101_compute_v1"
SCHEMA_REPORT = "issue_101_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_decision_aware_policy --validate"
RUNNER = "scripts/run_decision_aware_policy.py"
MODULE = "world_model/training/request_policy.py"

DEVICE = "cuda"
SEEDS = tuple(t96.SEEDS)
INVENTORIES = gb.INVENTORIES
PRIMARY = gb.GATE_INVENTORY
POLICY_INVENTORIES = rt.FIT_INVENTORIES
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
FIXED_ARMS = t96.FIXED_ARMS
C_STAR = "C-F-15"
KINDS = ("pe", "dec", "pg")
KIND_NAMES = {"pe": "prediction-error DP (ICLR eq. dp)", "dec": "decision-valued DP",
              "pg": "policy gradient on per-cell AUC"}
LAMBDAS = {"pe": (1e-5, 1e-4, 1e-3, 1e-2), "dec": (1e-4, 1e-3, 1e-2, 1e-1), "pg": (1e-4, 1e-3, 1e-2, 1e-1)}
ICLR_LAMBDA = rp.ICLR_COMPUTE_WEIGHT
WINDOW_LENGTH = 60
BASE_CONTINUATION = "F-15-macro"  # round-0 continuation of the decision labels (the Gate-B unit request)
PG = {"steps": 3600, "cells_per_step": 8, "samples": 8, "learning_rate": 3e-4, "grad_clip": 1.0}
OL_SEED = t96.OL_SEED
GPU_CAP_SECONDS = 40 * 3600.0
WALL_CAP_SECONDS = 60 * 3600.0  # summed over processes (4 GPUs: about 15 h elapsed)
REPLICATION_LINEAGES = ("issue-77-n1-005", "issue-77-n1-010", "issue-77-n1-015")
REPLICATION_SEED = 20260908
SMOKE_LEVELS = 4

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-101] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tag(lam):
    return f"{lam:.0e}"


def policy_arm(kind, lam):
    return f"J-{kind}-{tag(lam)}"


POLICY_ARMS = tuple(policy_arm(k, lam) for k in KINDS for lam in LAMBDAS[k])
J_ICLR = policy_arm("pe", ICLR_LAMBDA)
PHASE_1 = (*FIXED_ARMS, C_STAR, *POLICY_ARMS, "M-pe", "M-dec")
PHASE_2 = ("OL-pe", "OL-dec")
ARMS = (*PHASE_1, *PHASE_2)


def groups():
    """Training groups (kind, lambda, seed), kind-major; worker k trains groups[k::workers]."""
    return [(k, lam, s) for k in KINDS for lam in LAMBDAS[k] for s in SEEDS]


# ---------------------------------------------------------------------------
# membership: the #109 policy split as captured by the #104 campaign
# ---------------------------------------------------------------------------

def policy_levels():
    return [level for family in rt.families() for level in rt.cohort()["splits"]["policy"][family]]


@functools.lru_cache(maxsize=None)
def stream_rows():
    """The campaign report rows of the policy stream only (other streams are dropped at load)."""
    return {row["identity"]: row for row in read_json(CAMPAIGN / "branches.json") if row["stream"] == STREAM}


def policy_branches():
    """Frozen order: families, roster order, candidate suffix order (the #109 training candidates)."""
    rows = stream_rows()
    out = []
    for level in policy_levels():
        for candidate in rt.candidates():
            key = f"{level['identity']}-{candidate['suffix']}"
            out.append({"key": key, "member": level["identity"], "family": level["generator_family"],
                        "candidate": candidate, "status": rows[key]["status"],
                        "failure_class": rows[key]["failure_class"],
                        "attempt": str(CAMPAIGN / STREAM / "attempts" / key)})
    if len(out) != len(rows):
        raise ValueError("the policy-stream rows differ from the #109 policy roster x training candidates")
    return out


def complete_branches():
    return [b for b in policy_branches() if b["status"] == "complete"]


def policy_verdicts():
    return {key: bool(row["pig_removed"]) for key, row in stream_rows().items() if row["status"] == "complete"}


def policy_cell_entries():
    """(level, inventory) -> gb-style entry of the mixed policy cells, levels in roster order."""
    branches = complete_branches()
    verdicts = policy_verdicts()
    out = {}
    for inventory in POLICY_INVENTORIES:
        for member, entry in gb.mixed_cells(gb.inventory_entries(policy_levels(), branches, verdicts,
                                                                 inventory)).items():
            out[(member, inventory)] = entry
    order = {level["identity"]: i for i, level in enumerate(policy_levels())}
    return dict(sorted(out.items(), key=lambda kv: (order[kv[0][0]], POLICY_INVENTORIES.index(kv[0][1]))))


def pool_accounting():
    branches = policy_branches()
    cells = policy_cell_entries()
    rosters = {split: {lv["identity"] for f in rt.families() for lv in rt.cohort()["splits"][split][f]}
               for split in ("fit", "evaluation", "policy")}
    return {"levels": len(policy_levels()), "branches": len(branches),
            "complete": sum(b["status"] == "complete" for b in branches),
            "typed_failures": [{"key": b["key"], "failure_class": b["failure_class"]}
                               for b in branches if b["status"] != "complete"],
            "mixed_cells": dict(Counter(i for _, i in cells)),
            "cell_levels": len({m for m, _ in cells}),
            "disjoint_from_fit_and_evaluation": not (rosters["policy"] & (rosters["fit"] | rosters["evaluation"])),
            "pig_removed": sum(policy_verdicts().values())}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def shot_path(output, key):
    return Path(output) / "shots" / f"{key}.pt"


def pool_path(output):
    return Path(output) / "data" / "pool.pt"


def group_root(output, kind, lam, seed):
    return Path(output) / "models" / kind / f"lam-{tag(lam)}" / f"seed-{seed}"


def policy_path(output, kind, lam, seed):
    return group_root(output, kind, lam, seed) / "policy.pt"


def report_path(output, kind, lam, seed):
    return group_root(output, kind, lam, seed) / "report.json"


def selection_path(output):
    return Path(output) / "selection.json"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def mark(output, phase, wall, gpu, **extra):
    write_json(ledger_path(output, phase), {"phase": phase, "wall_seconds": wall, "gpu_seconds": gpu,
                                            "at": utc_now(), **extra})
    totals = ledger_totals(output)
    if totals["gpu_seconds"] > GPU_CAP_SECONDS or totals["wall_seconds"] > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded ({totals})")


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def ledger_totals(output):
    phases = ledger_phases(output)
    return {"gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
            "wall_seconds": sum(b["wall_seconds"] for b in phases.values())}


# ---------------------------------------------------------------------------
# predictors, costs, policies
# ---------------------------------------------------------------------------

def load_predictor(seed, name="unit"):
    """The Gate-B handoff checkpoint (hash-checked by run_gate_b.load_model)."""
    model, failure = gb.load_model(name, seed)
    if model is None:
        raise ValueError(f"predictor {name}/{seed}: {failure}")
    return model


@functools.lru_cache(maxsize=None)
def endpoint_costs():
    return harness.EndpointCosts(wlc.load_objective())


def tie_free_tensor(z_end, z0):
    """The #111 tie-free cost on tensors (presence - 0.1 x pig-centre displacement); nonfinite stays."""
    costs = endpoint_costs()
    displacement = torch.hypot(z_end[:, costs.center[0]]-z0[:, costs.center[0]],
                               z_end[:, costs.center[1]]-z0[:, costs.center[1]])
    return harness.tie_free(z_end[:, costs.presence], displacement)


def save_policy(output, kind, lam, seed, policy, extra):
    atomic_torch(policy_path(output, kind, lam, seed), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "kind": kind, "lambda": lam, "seed": seed,
        "pairs": [str(p.identity) for p in policy.pairs], "width": policy.layers[0].out_features,
        "remaining_feature": policy.remaining_feature, "model": policy.state_dict(), **extra})


def load_policy(output, kind, lam, seed):
    payload = torch.load(policy_path(output, kind, lam, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_POLICY or payload.get("plan_identity") != IDENTITY
            or (payload["kind"], payload["lambda"], payload["seed"]) != (kind, lam, seed)):
        raise ValueError(f"policy {kind}/{lam}/{seed} binding differs")
    policy = rp.RequestPolicy(PAIRS, payload["width"], payload["remaining_feature"])
    if payload["pairs"] != [str(p.identity) for p in policy.pairs]:
        raise ValueError("policy request set differs")
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# data: policy-split shot carriers (E99) and the pool payload
# ---------------------------------------------------------------------------

def shot_entry(branch, adapter):
    _, _, frames, _ = s99.shot_frames(branch["attempt"])
    end = min(len(frames)-1, s99.MAX_POSITION)
    segment = read_json(Path(branch["attempt"]) / "shot-1" / "segment.json")
    action = s99.action_of(segment)
    if not torch.equal(action, wlc.action_tensor(branch["candidate"]["action"], "cpu")[0]):
        raise ValueError(f"executed action of {branch['key']} differs from its inventory action")
    return {"schema": SCHEMA_SHOT, "plan_identity": IDENTITY, "key": branch["key"], "member": branch["member"],
            "end": end, "frame_count": len(frames), "action": action,
            "carriers": s99.carriers_for(adapter, branch["attempt"], list(range(end+1)))}


def read_shot(output, key):
    entry = torch.load(shot_path(output, key), map_location="cpu", weights_only=True)
    if entry.get("schema") != SCHEMA_SHOT or entry.get("plan_identity") != IDENTITY or entry["key"] != key:
        raise ValueError(f"shot {key} binding differs")
    return entry


def data(output, workers, worker):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    shard = complete_branches()[worker::workers]
    pending = [b for b in shard if not shot_path(output, b["key"]).is_file()]
    log(f"worker {worker}/{workers}: {len(pending)} of {len(shard)} policy shots to parse")
    with harness.deterministic_scoring():
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        for done, branch in enumerate(pending, 1):
            atomic_torch(shot_path(output, branch["key"]), shot_entry(branch, adapter))
            if done % 100 == 0:
                elapsed = time.monotonic()-began
                log(f"worker {worker}: {done}/{len(pending)} shots, {elapsed:.0f}s, "
                    f"eta {elapsed/done*(len(pending)-done):.0f}s")
    wall = time.monotonic()-began
    mark(output, f"data-worker{worker}", wall, wall, shots=len(shard))
    return 0


def policy_cells(entries, anchors):
    """Training cells for the decision labels / policy gradient: anchor, candidate actions, verdicts."""
    return [{"member": member, "inventory": inventory, "ordinals": [it["ordinal"] for it in entry["inventory"]],
             "z0": anchors[member], "actions": m99.action_batch(entry["inventory"]).cpu(),
             "verdicts": torch.tensor([entry["verdicts"][it["ordinal"]] for it in entry["inventory"]])}
            for (member, inventory), entry in entries.items()]


def assemble(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    branches = complete_branches()
    first = rt.anchor_branches(branches)
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        anchors = {member: rt.anchor_carrier(adapter, b["attempt"]) for member, b in first.items()}
    shots = [read_shot(output, b["key"]) for b in branches]
    windows = [rp.first_window(s["carriers"], s["action"], s["end"], WINDOW_LENGTH) for s in shots]
    cells = policy_cells(policy_cell_entries(), anchors)
    atomic_torch(pool_path(output), {
        "schema": SCHEMA_POOL, "plan_identity": IDENTITY, "scoring_policy": policy, "assembled_at": utc_now(),
        "levels": [lv["identity"] for lv in policy_levels()], "anchors": anchors,
        "anchor_frames": {m: b["key"] for m, b in first.items()},
        "shot_keys": [s["key"] for s in shots], "shot_members": [s["member"] for s in shots],
        "shot_ends": [int(s["end"]) for s in shots],
        "window_z": torch.stack([w["z"] for w in windows]), "window_action": torch.stack([w["action"] for w in windows]),
        "window_length": torch.stack([w["length"] for w in windows]), "cells": cells})
    wall = time.monotonic()-began
    mark(output, "assemble", wall, wall, shots=len(shots), cells=len(cells))
    log(f"pool assembled: {len(shots)} shots, {len(cells)} mixed cells ({wall:.0f}s)")
    return 0


@functools.lru_cache(maxsize=None)
def load_pool(output):
    payload = torch.load(pool_path(output), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_POOL or payload.get("plan_identity") != IDENTITY:
        raise ValueError("pool binding differs")
    return payload


def window_units(pool):
    """Per level (roster order): its shots' first windows (candidate suffix order)."""
    by_level = {}
    for i, member in enumerate(pool["shot_members"]):
        by_level.setdefault(member, []).append({"z": pool["window_z"][i], "action": pool["window_action"][i],
                                                "length": pool["window_length"][i]})
    return [by_level[m] for m in pool["levels"] if m in by_level]


def cell_units(pool):
    """Unit index of each training cell (levels with a mixed cell, roster order)."""
    members = [m for m in pool["levels"] if any(c["member"] == m for c in pool["cells"])]
    index = {m: i for i, m in enumerate(members)}
    return [index[c["member"]] for c in pool["cells"]], len(members)


# ---------------------------------------------------------------------------
# training (one group = kind x lambda x seed)
# ---------------------------------------------------------------------------

def split_rows(rows, units, count):
    """Decision rows grouped by the unit of their cell."""
    unit_of = torch.tensor(units)[rows["cell"]]
    return [{k: rows[k][unit_of == u] for k in ("z", "action", "remaining", "labels")} for u in range(count)]


def label_round(kind, model, policy, pool, lam, round_index):
    """Rows per unit for one DAgger round."""
    if kind == "pe":
        return [rp.window_rows(model, policy, unit, aggregated=bool(round_index), compute_weight=lam, device=DEVICE)
                for unit in window_units(pool)], {}
    units, count = cell_units(pool)
    continuation = ({"kind": "fixed", "pair_index": t96.PAIR_NAMES.index(BASE_CONTINUATION[2:])}
                    if round_index == 0 else {"kind": "policy"})
    rows = rp.decision_rows(model, policy, pool["cells"], continuation=continuation, cost_fn=tie_free_tensor,
                            compute_weight=lam, device=DEVICE)
    stats = {"rows": int(len(rows["labels"])),
             "label_counts": torch.bincount(rows["labels"], minlength=len(PAIRS)).tolist()}
    return split_rows(rows, units, count), stats


def rows_stats(history):
    labels = torch.cat([u["labels"] for rounds in history for u in rounds])
    return {"rows": int(len(labels)), "label_counts": torch.bincount(labels, minlength=len(PAIRS)).tolist()}


@torch.no_grad()
def policy_split_evaluation(model, policy, pool):
    """In-sample (policy split) greedy-J rollout from each cell's anchor: tie-free AUC per inventory,
    mean MACs per candidate rollout, request frequencies. Training data only."""
    costs = endpoint_costs()
    entries = policy_cell_entries()
    aucs = {i: [] for i in POLICY_INVENTORIES}
    macs = candidates = 0
    counts = [0]*len(PAIRS)
    for cell in pool["cells"]:
        entry = entries[(cell["member"], cell["inventory"])]
        z_end, timing = rp.rollout_endpoints(model, policy, cell["z0"].to(DEVICE), cell["actions"].to(DEVICE),
                                             {"kind": "joint"}, select=t96.select_pairs, record_schedules=True)
        rows = [costs.row(o, z_end[k].cpu(), cell["z0"]) for k, o in enumerate(cell["ordinals"])]
        auc = m99.view(entry, rows, PRIMARY_COST)["auc"]
        if auc is not None:
            aucs[cell["inventory"]].append(auc)
        macs += timing["linear_macs"]
        candidates += len(rows)
        for i, c in enumerate(rp.schedule_counts(timing["schedules"], len(PAIRS))):
            counts[i] += c
    pooled = [a for values in aucs.values() for a in values]
    return {"auc": {i: float(np.mean(v)) if v else None for i, v in aucs.items()},
            "auc_pooled": float(np.mean(pooled)) if pooled else None, "cells": len(pooled),
            "mean_macs_per_candidate": macs/candidates, "pair_counts": counts}


def train_group(output, kind, lam, seed):
    root = group_root(output, kind, lam, seed)
    if report_path(output, kind, lam, seed).is_file():
        log(f"{kind}/{tag(lam)}/{seed}: done")
        return
    began = time.monotonic()
    pool = load_pool(output)
    model = load_predictor(seed)
    with harness.deterministic_scoring() as scoring_policy:
        policy = rp.new_policy(PAIRS, seed, device=DEVICE)
        stats = {}
        if kind in ("pe", "dec"):
            history = []
            for round_index in range(rp.DISTIL_RECIPE["rounds"]):
                checkpoint = root / f"round-{round_index}.pt"
                labels = root / f"labels-{round_index}.pt"
                if labels.is_file():
                    history.append(torch.load(labels, map_location="cpu", weights_only=True)["units"])
                else:
                    t0 = time.monotonic()
                    rows, label_stats = label_round(kind, model, policy, pool, lam, round_index)
                    atomic_torch(labels, {"units": rows, "stats": label_stats})
                    history.append(rows)
                    stats[f"label_seconds_{round_index}"] = time.monotonic()-t0
                if checkpoint.is_file():
                    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                    continue
                t0 = time.monotonic()
                losses = rp.distil_round(policy, rp.aggregate_rounds(history), round_index, seed=seed, device=DEVICE)
                atomic_torch(checkpoint, {"model": policy.state_dict(), "losses": losses})
                stats[f"distil_seconds_{round_index}"] = time.monotonic()-t0
                stats[f"final_loss_{round_index}"] = float(np.mean(losses[-60:]))
                log(f"{kind}/{tag(lam)}/{seed}: round {round_index} rows {sum(len(u['labels']) for u in history[-1])}, "
                    f"loss {stats[f'final_loss_{round_index}']:.4f}")
            stats["labels"] = {f"round_{r}": rows_stats([h]) for r, h in enumerate(history)}
        else:
            t0 = time.monotonic()
            trace = rp.policy_gradient(model, policy, pool["cells"], seed=seed, cost_fn=tie_free_tensor,
                                       compute_weight=lam, steps=PG["steps"], cells_per_step=PG["cells_per_step"],
                                       samples=PG["samples"], learning_rate=PG["learning_rate"],
                                       grad_clip=PG["grad_clip"], device=DEVICE,
                                       on_step=lambda r: r["step"] % 300 == 0 and log(
                                           f"pg/{tag(lam)}/{seed}: step {r['step']} reward {r['reward']:.4f} "
                                           f"auc {r['auc']:.4f} work {r['work_units']:.1f}"))
            stats["pg_seconds"] = time.monotonic()-t0
            stats["pg_curve"] = [{k: float(np.mean([t[k] for t in trace[i:i+300]])) for k in ("reward", "auc",
                                                                                                "work_units", "loss")}
                                 for i in range(0, len(trace), 300)]
        policy.eval()
        evaluation = policy_split_evaluation(model, policy, pool)
    save_policy(output, kind, lam, seed, policy, {"scoring_policy": scoring_policy})
    wall = time.monotonic()-began
    write_json(report_path(output, kind, lam, seed), {
        "schema": SCHEMA_REPORT_TRAIN, "plan_identity": IDENTITY, "kind": kind, "lambda": lam, "seed": seed,
        "policy_sha256": sha256_of(policy_path(output, kind, lam, seed)), "trained_at": utc_now(),
        "predictor_sha256": sha256_of(gb.member_path("unit", seed)), "stats": stats,
        "policy_split": evaluation, "wall_seconds": wall})
    mark(output, f"train-{kind}-{tag(lam)}-seed{seed}", wall, wall)
    log(f"{kind}/{tag(lam)}/{seed}: trained ({wall:.0f}s); policy-split AUC {evaluation['auc_pooled']:.4f}, "
        f"MACs/candidate {evaluation['mean_macs_per_candidate']:.3g}")


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    load_pool(output)
    for kind, lam, seed in groups()[worker::workers]:
        train_group(output, kind, lam, seed)
    return 0


# ---------------------------------------------------------------------------
# selection (policy split only)
# ---------------------------------------------------------------------------

def load_report(output, kind, lam, seed):
    report = read_json(report_path(output, kind, lam, seed))
    if report.get("schema") != SCHEMA_REPORT_TRAIN or report.get("plan_identity") != IDENTITY:
        raise ValueError("training report binding differs")
    return report


def selection_table(output):
    """Primary lambda of dec and pg: the largest seed-mean policy-split pooled AUC (ties to the larger
    lambda); pe is fixed at the ICLR lambda."""
    table = {}
    for kind in KINDS:
        scores = {tag(lam): float(np.mean([load_report(output, kind, lam, s)["policy_split"]["auc_pooled"]
                                           for s in SEEDS])) for lam in LAMBDAS[kind]}
        if kind == "pe":
            chosen = ICLR_LAMBDA
        else:
            chosen = max(LAMBDAS[kind], key=lambda lam: (scores[tag(lam)], lam))
        table[kind] = {"policy_split_auc": scores, "lambda": chosen, "arm": policy_arm(kind, chosen)}
    return table


def select(output):
    output = Path(output)
    load_plan(output)
    if selection_path(output).is_file():
        log("selection exists")
        return 0
    if any(Path(output, "records").glob("*.json")):
        raise ValueError("records exist before the selection; refusing")
    table = selection_table(output)
    write_json(selection_path(output), {"schema": SCHEMA_SELECTION, "plan_identity": IDENTITY,
                                        "selected_at": utc_now(), "rule": frozen_rules()["selection"],
                                        "kinds": table})
    log(f"selection: { {k: v['arm'] for k, v in table.items()} }")
    return 0


def load_selection(output):
    selection = read_json(selection_path(output))
    if selection.get("schema") != SCHEMA_SELECTION or selection.get("plan_identity") != IDENTITY:
        raise ValueError("selection binding differs")
    return selection


def primary_arms(selection):
    return {"J": J_ICLR, "J_dec": selection["kinds"]["dec"]["arm"], "J_pg": selection["kinds"]["pg"]["arm"]}


def arm_policy(arm, selection):
    """(kind, lambda, schedule kind) of a controller arm."""
    if arm in POLICY_ARMS:
        kind, text = arm.split("-")[1], arm.split("-", 2)[2]
        return kind, next(lam for lam in LAMBDAS[kind] if tag(lam) == text), "joint"
    if arm in ("M-pe", "M-dec"):
        kind = arm[2:]
        return kind, selection["kinds"][kind]["lambda"], "factorized"
    return None


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def rle_schedules(items, timing):
    return {str(item["ordinal"]): t96.rle(timing["schedules"][k]) for k, item in enumerate(items)}


def block(model, policy, spec, z0, items, costs, schedules):
    actions = m99.action_batch(items)
    z_end, timing = rp.rollout_endpoints(model, policy, z0.to(DEVICE), actions, spec, select=t96.select_pairs,
                                         record_schedules=schedules)
    out = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if schedules:
        out["schedules"] = rle_schedules(items, timing)
    return out


def ordered_cells(cells):
    """Frozen scoring order: inventory, member (sorted), as Gate B."""
    return [(inventory, member, entry) for inventory in INVENTORIES
            for member, entry in sorted(cells[inventory]["mixed"].items())]


def j_counts(output, arm):
    """Per seed, the pooled request counts of ``arm`` over every cell of every inventory and every
    candidate (outcome-free; #96 j_frequencies)."""
    counts = {}
    for seed in SEEDS:
        record = read_record(output, arm, seed)
        total = [0]*len(PAIRS)
        for members in record["inventories"].values():
            for result in members.values():
                for text in result["schedules"].values():
                    for index in t96.unrle(text):
                        total[index] += 1
        counts[str(seed)] = total
    return counts


def ol_schedules(cells, counts):
    """#96 OL: one PCG64(7202) stream per OL arm, consumed in the frozen order (inventory, member, seed)."""
    generator = np.random.Generator(np.random.PCG64(OL_SEED))
    return {(inventory, member, seed): rp.open_loop_schedule(counts[str(seed)], PAIRS, generator)
            for inventory, member, _ in ordered_cells(cells) for seed in SEEDS}


def score_record(output, cells, anchors, arm, seed, selection, policy_state):
    costs = endpoint_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": arm, "seed": seed, "engine_seconds": 0}
    began = time.monotonic()
    model = load_predictor(seed, "c_star" if arm == C_STAR else "unit")
    policy, schedules, extra = None, False, {}
    if arm in FIXED_ARMS or arm == C_STAR:
        spec = t96.arm_spec(arm)
    elif arm in PHASE_2:
        kind = arm[3:]
        source = J_ICLR if kind == "pe" else selection["kinds"]["dec"]["arm"]
        counts = j_counts(output, source)
        schedules_by_cell = ol_schedules(cells, counts)
        extra = {"ol_source": source, "ol_counts": counts[str(seed)]}
        schedules = True
    else:
        kind, lam, schedule_kind = arm_policy(arm, selection)
        policy = load_policy(output, kind, lam, seed)
        spec = {"kind": schedule_kind}
        schedules = True
        extra = {"policy": {"kind": kind, "lambda": lam,
                            "sha256": sha256_of(policy_path(output, kind, lam, seed))}}
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in ordered_cells(cells):
        if arm in PHASE_2:
            spec = {"kind": "schedule", "schedule": schedules_by_cell[(inventory, member, seed)]}
        inventories[inventory][member] = block(model, policy, spec, anchors[member], entry["inventory"], costs,
                                               schedules)
    torch.cuda.synchronize()
    return {**base, **extra, "predictor_sha256": sha256_of(gb.member_path("c_star" if arm == C_STAR else "unit", seed)),
            "inventories": inventories, "seconds": time.monotonic()-began}


def wrong_record(output, seed, selection, policy_state):
    """J_dec from every fit-split anchor over the shared grid candidates (Gate-B wrong-anchor rule)."""
    costs = endpoint_costs()
    kind, lam = "dec", selection["kinds"]["dec"]["lambda"]
    model = load_predictor(seed)
    policy = load_policy(output, kind, lam, seed)
    items = gb.inventory_items(PRIMARY)
    anchors = gb.fit_anchors()
    began = time.monotonic()
    rows = {member: block(model, policy, {"kind": "joint"}, anchors[member], items, costs, False)
            for member in sorted(anchors)}
    return {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": "wrong-anchor", "seed": seed, "engine_seconds": 0,
            "policy": {"kind": kind, "lambda": lam, "sha256": sha256_of(policy_path(output, kind, lam, seed))},
            "wrong_anchor": rows, "seconds": time.monotonic()-began}


def read_record(output, arm, seed):
    record = read_json(record_path(output, arm, seed))
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY or record["seed"] != seed:
        raise ValueError(f"record {arm}/{seed} binding differs")
    return record


def score_tasks():
    return ([(arm, s) for arm in PHASE_1 for s in SEEDS] + [("wrong-anchor", s) for s in SEEDS],
            [(arm, s) for arm in PHASE_2 for s in SEEDS])


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    selection = load_selection(output)
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    phase_1, phase_2 = score_tasks()
    for phase, tasks in (("1", phase_1), ("2", phase_2)):
        if phase == "2":
            missing = [a for a in (J_ICLR, selection["kinds"]["dec"]["arm"]) for s in SEEDS
                       if not record_path(output, a, s).is_file()]
            if missing:
                log(f"worker {worker}: phase 2 waits for the J records ({len(missing)} missing)")
                return 0
        for arm, seed in tasks[worker::workers]:
            if record_path(output, arm, seed).is_file():
                continue
            t0 = time.monotonic()
            with harness.deterministic_scoring() as policy_state:
                record = (wrong_record(output, seed, selection, policy_state) if arm == "wrong-anchor"
                          else score_record(output, cells, anchors, arm, seed, selection, policy_state))
            write_json(record_path(output, arm, seed), record)
            torch.cuda.empty_cache()
            wall = time.monotonic()-t0
            mark(output, f"score-{arm}-seed{seed}", wall, wall)
            log(f"worker {worker}: scored {arm} seed {seed} ({wall:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def load_records(output):
    return {(arm, s): read_record(output, arm, s) for arm in ARMS for s in SEEDS}


def view_rows(records, cells, inventory, stats):
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, stats), PRIMARY_COST)
        for seed in SEEDS:
            views = {arm: v1.cost_view(entry, records[(arm, seed)]["inventories"][inventory][member]["rows"],
                                       PRIMARY_COST) for arm in ARMS}
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def member_means(rows):
    return {arm: t96.member_mean([{"member": r["member"], "auc": r["views"][arm]["auc"]} for r in rows], "auc")
            for arm in ARMS}


def arm_compute(records, inventory, arm):
    results = [records[(arm, s)]["inventories"][inventory][m] for s in SEEDS
               for m in records[(arm, s)]["inventories"][inventory]]
    candidates = sum(r["candidates"] for r in results)
    return {"mean_macs_per_candidate": sum(r["linear_macs"] for r in results)/candidates if candidates else None,
            "mean_transitions_per_candidate": sum(r["transition_calls"] for r in results)/candidates
            if candidates else None,
            "gpu_seconds": sum(r["gpu_seconds"] for r in results)}


def schedule_diagnostics(records, inventory, arm):
    """Pooled request counts, modal request and non-modal step share, share of cells whose candidates'
    schedules are not all identical (state dependence), over seeds."""
    counts = [0]*len(PAIRS)
    differing = cells = 0
    for seed in SEEDS:
        for result in records[(arm, seed)]["inventories"][inventory].values():
            texts = list(result["schedules"].values())
            cells += 1
            differing += len(set(texts)) > 1
            for text in texts:
                for index in t96.unrle(text):
                    counts[index] += 1
    total = sum(counts)
    modal = max(range(len(PAIRS)), key=lambda k: (counts[k], -k))
    return {"pair_counts": counts, "modal_pair": t96.PAIR_NAMES[modal],
            "non_modal_step_share": 1-counts[modal]/total if total else None,
            "cells_with_differing_schedules": differing, "cells": cells}


def contrast_block(rows, first, second):
    block = m99.paired(rows, [(1, first), (-1, second)])
    return {**block, "first": first, "second": second, "token": c2.token(block["estimate"])}


def inventory_tables(records, rows, means_all, inventory, selection):
    arms = primary_arms(selection)

    def cross_fit(names):
        scores = {}
        for arm in names:
            others = [means_all[o][arm] for o in INVENTORIES if o != inventory and means_all[o][arm] is not None]
            scores[arm] = float(np.mean(others)) if others else None
        return max(names, key=lambda a: (-math.inf if scores[a] is None else scores[a], -names.index(a))), scores

    f_star, f_scores = cross_fit(list(FIXED_ARMS))
    means = means_all[inventory]
    hindsight = max(FIXED_ARMS, key=lambda a: (-math.inf if means[a] is None else means[a], -FIXED_ARMS.index(a)))
    j, jd, jp = arms["J"], arms["J_dec"], arms["J_pg"]
    pairs = {
        "J_dec-F_hindsight": (jd, hindsight), "J_dec-F*": (jd, f_star), "J_dec-M_dec": (jd, "M-dec"),
        "J_dec-OL_dec": (jd, "OL-dec"), "J_dec-J": (jd, j), "J_dec-prior": (jd, "prior"), "J_dec-C*": (jd, C_STAR),
        "J-F_hindsight": (j, hindsight), "J-F*": (j, f_star), "J-M": (j, "M-pe"), "J-OL": (j, "OL-pe"),
        "J_pg-F_hindsight": (jp, hindsight), "J_pg-F*": (jp, f_star), "J_pg-J_dec": (jp, jd), "J_pg-J": (jp, j),
    }
    rankers = {arm: {**v1.ranker_block(rows, arm), **(arm_compute(records, inventory, arm) if arm != "prior" else {})}
               for arm in (*ARMS, "prior")}
    sweep = {kind: [{"lambda": lam, "arm": policy_arm(kind, lam), "auc": rankers[policy_arm(kind, lam)]["auc"],
                     "mean_macs_per_candidate": rankers[policy_arm(kind, lam)]["mean_macs_per_candidate"],
                     "minus_F_hindsight": contrast_block(rows, policy_arm(kind, lam), hindsight)["estimate"]}
                    for lam in LAMBDAS[kind]] for kind in KINDS}
    controller_arms = (*POLICY_ARMS, "M-pe", "M-dec", "OL-pe", "OL-dec")
    return {"cells": len(rows), "members": len({r["member"] for r in rows}),
            "F_star": f_star, "F_star_scores": f_scores, "F_hindsight": hindsight,
            "member_mean_auc": means, "rankers": rankers,
            "contrasts": {name: contrast_block(rows, a, b) for name, (a, b) in pairs.items()},
            "lambda_sweep": sweep,
            "schedules": {arm: schedule_diagnostics(records, inventory, arm) for arm in controller_arms}}


def wrong_anchor_table(output, records, cells, selection):
    jd = primary_arms(selection)["J_dec"]
    rows = []
    for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
        for seed in SEEDS:
            right = records[(jd, seed)]["inventories"][PRIMARY][member]["rows"]
            wrong_record_ = read_record(output, "wrong-anchor", seed)["wrong_anchor"]
            wrong = [wrong_record_[a]["rows"] for a in sorted(wrong_record_)]
            wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
            rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong) if r is not None]
            rows.append({"member": member, "seed": seed, "views": {"J_dec": {
                "right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    block = c2.difference_block(rows, [(1, "J_dec", "right"), (-1, "J_dec", "wrong")])
    return {"arm": jd, "cells": len(rows), "wrong_anchors": len(gb.fit_anchors()),
            "right": c2.field_block(rows, "J_dec", "right"), "wrong": c2.field_block(rows, "J_dec", "wrong"),
            "right_minus_wrong": block, "stability": c2.field_block(rows, "J_dec", "stability"),
            "token": c2.token(block)}


def gate_b_replication(records):
    """F-15-macro and C-F-15 rows equal the Gate-B unit / C* records on every cell and seed."""
    same = True
    checked = 0
    for name, arm in (("unit", gb.UNITS["unit"]["request"]), ("c_star", C_STAR)):
        for seed in SEEDS:
            reference = read_json(gb.record_path(GATE_B, name, seed))["inventories"]
            mine = records[(arm, seed)]["inventories"]
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    same &= mine[inventory][member]["rows"] == result["rows"]
                    checked += 1
    return {"blocks": checked, "pass": bool(same)}


def training_table(output):
    out = {}
    for kind, lam, seed in groups():
        report = load_report(output, kind, lam, seed)
        out.setdefault(kind, {}).setdefault(tag(lam), {})[str(seed)] = {
            "policy_split": report["policy_split"], "labels": report["stats"].get("labels"),
            "pg_curve_last": (report["stats"].get("pg_curve") or [None])[-1], "wall_seconds": report["wall_seconds"],
            "policy_sha256": report["policy_sha256"]}
    return out


def compute_tables(output, plan):
    output = Path(output)
    selection = load_selection(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output)
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    rows = {i: view_rows(records, cells, i, stats[i]) for i in INVENTORIES}
    means = {i: member_means(rows[i]) for i in INVENTORIES}
    tables = {i: inventory_tables(records, rows[i], means, i, selection) for i in INVENTORIES}
    wrong = wrong_anchor_table(output, records, cells, selection)
    replication = gate_b_replication(records)
    training = training_table(output)
    pool = load_pool(output)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    policy_roster = {lv["identity"] for lv in policy_levels()}
    hashes_ok = all(records[(arm, s)]["policy"]["sha256"] == training[r[0]][tag(r[1])][str(s)]["policy_sha256"]
                    for arm in ARMS for s in SEEDS for r in [arm_policy(arm, selection)] if r is not None)
    trained_at = sorted(load_report(output, k, lam, s)["trained_at"] for k, lam, s in groups())
    scored_at = sorted(r["scored_at"] for r in records.values())
    guards = {
        "G1_isolation": {"pass": bool(set(pool["levels"]) == policy_roster
                                      and not (policy_roster & (eval_roster | fit_roster))
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES))},
        "G2_scoring_policy": {"pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained_at[0],
                          "last_trained_at": trained_at[-1], "selected_at": selection["selected_at"],
                          "first_scored_at": scored_at[0],
                          "pass": plan["frozen_at"] <= trained_at[0] and trained_at[-1] <= selection["selected_at"]
                          <= scored_at[0]},
        "G4_checkpoint_binding": {"pass": bool(hashes_ok)},
        "G5_gate_b_replication": replication,
    }
    guards_ok = all(g["pass"] for g in guards.values())
    grid = tables[PRIMARY]
    primary = grid["contrasts"]["J_dec-F_hindsight"]
    phases = ledger_phases(output)
    smoke = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "selection": selection["kinds"], "primary_arms": primary_arms(selection),
        "evaluation": gb.evaluation_accounting(branches, cells), "pool": plan["pool"]["accounting"],
        "guards": guards,
        "primary": {"contrast": "J_dec - F_hindsight (grid)", "estimate": primary["estimate"],
                    "cells": primary["cells"], "members": primary["members"], "F_hindsight": grid["F_hindsight"],
                    "token": c2.token(primary["estimate"]) if guards_ok else STOP_TOKEN,
                    "rule": plan["decision_rule"]["primary"]},
        "inventories": tables, "wrong_anchor": wrong, "training": training,
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "hardware": "4 x RTX 5090 (one process per GPU)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def cell(block):
    return fmt_interval(block) if block else "n/a"


def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def gmacs(value):
    return "n/a" if value is None else f"{value/1e6:.1f}M"


def findings_md(plan, compute):
    lines = []
    add = lines.append
    arms = compute["primary_arms"]
    grid = compute["inventories"][PRIMARY]
    p = compute["primary"]
    add("# Issue #101: decision-aware request policy on the #104 sealed #109 evaluation split")
    add("")
    add(f"Plan frozen {plan['frozen_at']} after a passed smoke, before any training or evaluation score; scored once "
        f"in processes that did not train, under `deterministic_scoring()`. Validation: `{VALIDATION_COMMAND}`.")
    add("")
    add(f"**Primary (grid): J_dec − F_hindsight = {cell(p['estimate'])} over {p['cells']} cells, {p['members']} "
        f"levels; {p['token']}** (δ = {DELTA}; F_hindsight = {p['F_hindsight']}). J_dec = `{arms['J_dec']}`, "
        f"J (ICLR) = `{arms['J']}`, J_pg = `{arms['J_pg']}`.")
    add("")
    add("## Contrasts per inventory (paired AUC differences, level-clustered, DESCRIPTIVE)")
    add("")
    names = ("J_dec-F_hindsight", "J_dec-F*", "J_dec-M_dec", "J_dec-OL_dec", "J_dec-J", "J-F_hindsight", "J-M", "J-OL",
             "J_pg-F_hindsight", "J_pg-J_dec", "J_dec-prior", "J_dec-C*")
    add("| contrast | " + " | ".join(INVENTORIES) + " |")
    add("| --- | " + " | ".join("---" for _ in INVENTORIES) + " |")
    for name in names:
        add(f"| {name.replace('-', ' − ', 1)} | " + " | ".join(
            f"{cell(compute['inventories'][i]['contrasts'][name]['estimate'])} {compute['inventories'][i]['contrasts'][name]['token']}"
            for i in INVENTORIES) + " |")
    add("")
    add("Targets: " + "; ".join(f"{i}: F_hindsight {compute['inventories'][i]['F_hindsight']}, F* (cross-fit r†) "
                                 f"{compute['inventories'][i]['F_star']}" for i in INVENTORIES) + ".")
    add("")
    add("## Rankers (grid)")
    add("")
    add("| arm | AUC | MACs / candidate | tied pair share | nonfinite share |")
    add("| --- | --- | --- | --- | --- |")
    for arm in (*FIXED_ARMS, C_STAR, arms["J"], "M-pe", "OL-pe", arms["J_dec"], "M-dec", "OL-dec", arms["J_pg"], "prior"):
        r = grid["rankers"][arm]
        add(f"| {arm} | {cell(r['auc'])} | {gmacs(r.get('mean_macs_per_candidate'))} | {num(r['tied_pair_share'])} | "
            f"{num(r['nonfinite_share'])} |")
    add("")
    add("## λ sweep (grid): MAC vs AUC for each policy type")
    add("")
    add("| type | λ | AUC | MACs / candidate | − F_hindsight | policy-split AUC (seed mean) |")
    add("| --- | --- | --- | --- | --- | --- |")
    for kind in KINDS:
        for entry in grid["lambda_sweep"][kind]:
            add(f"| {kind} | {tag(entry['lambda'])} | {cell(entry['auc'])} | {gmacs(entry['mean_macs_per_candidate'])} | "
                f"{cell(entry['minus_F_hindsight'])} | {num(compute['selection'][kind]['policy_split_auc'][tag(entry['lambda'])])} |")
    add("")
    add("## Schedules (grid; pooled over seeds)")
    add("")
    add("| arm | modal request | non-modal step share | cells with candidate-dependent schedules |")
    add("| --- | --- | --- | --- |")
    for arm in (arms["J"], arms["J_dec"], arms["J_pg"], "M-pe", "M-dec"):
        s = grid["schedules"][arm]
        add(f"| {arm} | {s['modal_pair']} | {num(s['non_modal_step_share'], 3)} | "
            f"{s['cells_with_differing_schedules']}/{s['cells']} |")
    add("")
    w = compute["wrong_anchor"]
    add(f"Wrong-anchor control of J_dec ({w['wrong_anchors']} fit-split anchors): right {cell(w['right'])}, wrong "
        f"{cell(w['wrong'])}, right − wrong {cell(w['right_minus_wrong'])} ({w['token']}); stability {cell(w['stability'])}.")
    add("")
    add("## Guards")
    add("")
    for name, guard in compute["guards"].items():
        add(f"- {name}: {guard['pass']}")
    add("")
    c = compute["compute"]
    add(f"Compute: {c['gpu_seconds']/3600:.2f} GPU h, {c['wall_seconds']/3600:.2f} process wall h ({c['hardware']}); "
        "0 engine seconds.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    def contrasts(names):
        return {i: {n: {"estimate": compute["inventories"][i]["contrasts"][n]["estimate"],
                        "token": compute["inventories"][i]["contrasts"][n]["token"]} for n in names}
                for i in INVENTORIES}
    grid = compute["inventories"][PRIMARY]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "primary": compute["primary"],
            "primary_arms": compute["primary_arms"],
            "contrasts": contrasts(("J_dec-F_hindsight", "J_dec-F*", "J_dec-M_dec", "J_dec-OL_dec", "J_dec-J",
                                    "J-F_hindsight", "J-M", "J-OL", "J_pg-F_hindsight", "J_pg-J_dec")),
            "lambda_sweep_grid": {k: [{"lambda": e["lambda"], "auc": e["auc"]["mean"] if e["auc"] else None,
                                       "mean_macs_per_candidate": e["mean_macs_per_candidate"]}
                                      for e in grid["lambda_sweep"][k]] for k in KINDS},
            "F_hindsight": {i: compute["inventories"][i]["F_hindsight"] for i in INVENTORIES},
            "F_star": {i: compute["inventories"][i]["F_star"] for i in INVENTORIES},
            "wrong_anchor": {k: compute["wrong_anchor"][k] for k in ("right_minus_wrong", "token")},
            "guards": {k: v["pass"] for k, v in compute["guards"].items()},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")},
            "shared_module": MODULE, "claim_boundary": plan["claim_boundary"]}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute)}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; #77 artefacts, the fit split and a few policy levels; no evaluation score)
# ---------------------------------------------------------------------------

def issue77_args(output):
    return argparse.Namespace(output=Path(output), device=DEVICE, issue71=None, parser=None)


def replication_checks():
    """The shared module reproduces the #77 controller-label path and training loop bit for bit on
    the #77 seed-20260908 hybrid checkpoint and its three N1 controller lineages (this hardware)."""
    from scripts import run_issue_77_n1_train as n77
    from world_model.training.matched_dynamics import MatchedController
    plan77 = read_json(ISSUE77 / "plan.json")
    checks = []
    args = issue77_args(ISSUE77)
    with harness.deterministic_scoring():
        model, _ = n77.load_predictor(args, plan77, REPLICATION_SEED, "hybrid")
        torch.manual_seed(REPLICATION_SEED+rp.INIT_OFFSET)
        matched = MatchedController(False).to(DEVICE)
        mine = rp.new_policy(PAIRS, REPLICATION_SEED, device=DEVICE)
        init_equal = all(torch.equal(a, b) for a, b in zip(matched.state_dict().values(), mine.state_dict().values()))
        trained = torch.load(ISSUE77 / f"seed-{REPLICATION_SEED}/hybrid/controller-round-0.pt", map_location="cpu",
                             weights_only=True)["model"]
        matched.load_state_dict(trained)
        mine.load_state_dict(trained)
        probe = torch.Generator().manual_seed(0)
        z = torch.randn(64, 236, generator=probe).to(DEVICE)
        a = torch.rand(64, 5, generator=probe).to(DEVICE)
        remaining = torch.randint(1, 226, (64,), generator=probe).to(DEVICE)
        checks.append({"control": "request_policy_equals_matched_controller", "init_equal": bool(init_equal),
                       "ok": bool(init_equal and torch.equal(matched(z, a, remaining), mine(z, a, remaining)))})
        label_equal, stored_equal, stored_mismatch = True, True, 0
        for aggregated in (False, True):
            for lineage in REPLICATION_LINEAGES:
                reference = n77.label_lineage(args, plan77, None, model, matched, lineage, aggregated)
                data = n77.shard(args, plan77, None, lineage, pure=True)
                windows = [{"z": data["z"][i], "action": data["action"][i], "length": data["length"][i]}
                           for i in n77.first_windows(args, plan77, lineage)]
                ours = rp.window_rows(model, mine, windows, aggregated=aggregated, device=DEVICE)
                label_equal &= all(torch.equal(reference[k], ours[k]) for k in reference)
                stored = torch.load(n77.label_path(ISSUE77 / f"seed-{REPLICATION_SEED}" / "hybrid", int(aggregated),
                                                   lineage), map_location="cpu", weights_only=True)["tensors"]
                stored_equal &= all(torch.equal(stored[k], ours[k]) for k in stored)
                stored_mismatch += int((stored["labels"] != ours["labels"]).sum())
        checks.append({"control": "window_rows_equal_issue77_label_lineage", "lineages": list(REPLICATION_LINEAGES),
                       "rounds": [0, 1], "ok": bool(label_equal),
                       "informational_equal_to_stored_issue77_labels": bool(stored_equal),
                       "informational_stored_label_mismatches": stored_mismatch,
                       "note": "stored #77 labels were computed on the #85 RTX 3090; the gating comparison is the "
                               "#77 code path re-run on this hardware"})
        # training loop: #77 train_control on the stored label files of the three lineages
        with tempfile.TemporaryDirectory(prefix="issue101-replication-") as temporary:
            temp = Path(temporary)
            cell = temp / f"seed-{REPLICATION_SEED}" / "hybrid"
            source = ISSUE77 / f"seed-{REPLICATION_SEED}" / "hybrid"
            for r in (0, 1):
                (cell / f"labels-{r}").mkdir(parents=True)
                for lineage in REPLICATION_LINEAGES:
                    path = n77.label_path(source, r, lineage)
                    (cell / f"labels-{r}" / path.name).symlink_to(path)
            (cell / "predictor.pt").symlink_to(source / "predictor.pt")
            n77.train_control(issue77_args(temp), plan77, None, REPLICATION_SEED, "hybrid",
                              indices=list(REPLICATION_LINEAGES))
            reference = [torch.load(cell / f"controller-round-{r}.pt", map_location="cpu", weights_only=True)["model"]
                         for r in (0, 1)]
            history = [[torch.load(n77.label_path(source, r, lineage), map_location="cpu", weights_only=True)["tensors"]
                        for lineage in REPLICATION_LINEAGES] for r in (0, 1)]
        policy = rp.new_policy(PAIRS, REPLICATION_SEED, device=DEVICE)
        equal = True
        for r in (0, 1):
            rp.distil_round(policy, rp.aggregate_rounds(history[:r+1]), r, seed=REPLICATION_SEED,
                            steps=plan77["controller"]["steps_per_round"], batch_size=plan77["controller"]["batch_size"],
                            learning_rate=plan77["controller"]["learning_rate"], device=DEVICE)
            equal &= all(torch.equal(v.cpu(), reference[r][k]) for k, v in policy.state_dict().items())
        checks.append({"control": "distil_round_equals_issue77_train_control", "lineages": list(REPLICATION_LINEAGES),
                       "steps_per_round": plan77["controller"]["steps_per_round"], "ok": bool(equal)})
    return checks


@torch.no_grad()
def naive_q(model, policy, cell, candidate, position, z_state, request, partner_costs, sign, lam, continuation):
    """Independent single-row recomputation of one decision-label Q value (smoke reference)."""
    pairs = policy.pairs
    reference = rp.work(model, rp.REFERENCE_PAIR)
    z = z_state[None].clone()
    a = cell["actions"][candidate][None].to(DEVICE)
    t, spent, first = position, 0.0, True
    while t < rp.ENDPOINT:
        remaining = rp.ENDPOINT-t
        if first:
            k, first = request, False
        elif continuation["kind"] == "fixed":
            base = pairs[continuation["pair_index"]]
            options = [j for j, p in enumerate(pairs) if p.abstraction == base.abstraction and p.delta <= remaining]
            k = continuation["pair_index"] if base.delta <= remaining else max(options, key=lambda j: pairs[j].delta)
        else:
            k = int(policy(z, a, torch.tensor([remaining], device=DEVICE)).argmax(-1))
        z = model.carrier(z, a, pairs[k])
        t += pairs[k].delta
        spent += float(rp.work(model, pairs[k], policy))
    cost = float(tie_free_tensor(z, cell["z0"][None].to(DEVICE))[0])
    terms = [math.log1p(math.exp(sign*(cost-p)/rp.RANK_TEMPERATURE)) for p in partner_costs]
    return sum(terms)/len(terms)+lam*spent/reference


def decision_label_checks(model, cells, lam_grid):
    """Decision labels on a few policy cells: brute-force Q equality on sampled rows, label
    distribution and loss scale per lambda, and one DAgger-continuation pass (timing)."""
    out = {"per_lambda": {}}
    policy = rp.new_policy(PAIRS, SEEDS[0], device=DEVICE)
    continuation = {"kind": "fixed", "pair_index": t96.PAIR_NAMES.index(BASE_CONTINUATION[2:])}
    t0 = time.monotonic()
    rows = rp.decision_rows(model, policy, cells, continuation=continuation, cost_fn=tie_free_tensor,
                            compute_weight=lam_grid[1], device=DEVICE)
    out["round0_seconds"] = time.monotonic()-t0
    q = rows["q"].double()
    finite = torch.isfinite(q)
    spread = (q.masked_fill(~finite, -math.inf).max(1).values-q.masked_fill(~finite, math.inf).min(1).values)
    out["rows"] = len(q)
    out["q_spread_quantiles_at_lambda"] = {"lambda": lam_grid[1], "q10_50_90": torch.quantile(
        spread.float(), torch.tensor([0.1, 0.5, 0.9])).tolist()}
    # brute force on sampled rows (independent single-row loop)
    generator = torch.Generator().manual_seed(1)
    picks = torch.randint(len(q), (12,), generator=generator).tolist()
    base = rows["continuation_costs"]
    verdicts = torch.cat([c["verdicts"] for c in cells]).bool()
    cell_of = rows["cell_of_candidate"]
    worst = 0.0
    for pick in picks:
        candidate_global = int(rows["candidate"][pick])
        position = int(rows["position"][pick])
        c = int(cell_of[candidate_global])
        first = int(sum(len(cells[j]["actions"]) for j in range(c)))
        sign = 1.0 if bool(verdicts[candidate_global]) else -1.0
        partners = [float(base[first+j]) for j in range(len(cells[c]["actions"]))
                    if bool(verdicts[first+j]) != bool(verdicts[candidate_global]) and math.isfinite(float(base[first+j]))]
        for request in range(len(PAIRS)):
            if PAIRS[request].delta > rp.ENDPOINT-position:
                continue
            value = naive_q(model, policy, cells[c], candidate_global-first, position, rows["z"][pick].to(DEVICE),
                            request, partners, sign, lam_grid[1], continuation)
            worst = max(worst, abs(value-float(q[pick, request])))
    out["brute_force"] = {"rows_checked": len(picks), "max_abs_q_delta": worst}
    for lam in lam_grid:
        labels = rp.decision_rows(model, policy, cells, continuation=continuation, cost_fn=tie_free_tensor,
                                  compute_weight=lam, device=DEVICE)["labels"]
        out["per_lambda"][tag(lam)] = torch.bincount(labels, minlength=len(PAIRS)).tolist()
    random_policy = rp.new_policy(PAIRS, SEEDS[1], device=DEVICE)
    t0 = time.monotonic()
    dagger = rp.decision_rows(model, random_policy, cells, continuation={"kind": "policy"}, cost_fn=tie_free_tensor,
                              compute_weight=lam_grid[1], device=DEVICE)
    out["dagger_round_seconds"] = time.monotonic()-t0
    out["dagger_rows"] = len(dagger["labels"])
    t0 = time.monotonic()
    trace = rp.policy_gradient(model, rp.new_policy(PAIRS, SEEDS[0], device=DEVICE), cells, seed=SEEDS[0],
                               cost_fn=tie_free_tensor, compute_weight=lam_grid[1], steps=3,
                               cells_per_step=PG["cells_per_step"], samples=PG["samples"],
                               learning_rate=PG["learning_rate"], device=DEVICE)
    out["pg_seconds_per_step"] = (time.monotonic()-t0)/3
    out["pg_finite"] = all(math.isfinite(t["loss"]) for t in trace)
    out["ok"] = bool(worst <= 1e-4 and out["pg_finite"])
    return out


def rollout_checks(model_unit):
    """rp.rollout_endpoints equals m99.rollout_endpoints on three fit grid cells: fixed requests and a
    (random-init) policy under the joint and factorized kinds; a constant schedule equals fixed."""
    fit_cells = gb.mixed_cells(gb.inventory_entries(rt.fit_levels(), rt.complete_branches(), rt.fit_verdicts(), PRIMARY))
    stored = gb.fit_anchors()
    policy = rp.new_policy(PAIRS, SEEDS[2], device=DEVICE).eval()
    equal = True
    for member, entry in sorted(fit_cells.items())[:3]:
        actions = m99.action_batch(entry["inventory"])
        z0 = stored[member].to(DEVICE)
        for arm in ("F-15-macro", "F-1-continuous"):
            a, _ = m99.rollout_endpoints(model_unit, None, z0, actions, t96.arm_spec(arm))
            b, _ = rp.rollout_endpoints(model_unit, None, z0, actions, t96.arm_spec(arm), select=t96.select_pairs)
            equal &= bool(torch.equal(a, b))
            sched = [t96.arm_spec(arm)["pair_index"]]*(rp.ENDPOINT//PAIRS[t96.arm_spec(arm)["pair_index"]].delta)
            c, _ = rp.rollout_endpoints(model_unit, None, z0, actions, {"kind": "schedule", "schedule": sched},
                                        select=t96.select_pairs)
            equal &= bool(torch.equal(a, c))
        for kind in ("joint", "factorized"):
            a, ta = m99.rollout_endpoints(model_unit, policy, z0, actions, {"kind": kind})
            b, tb = rp.rollout_endpoints(model_unit, policy, z0, actions, {"kind": kind}, select=t96.select_pairs,
                                         record_schedules=True)
            equal &= bool(torch.equal(a, b)) and ta["linear_macs"] == tb["linear_macs"]
    return {"control": "rollout_endpoints_equals_m99", "cells": 3, "ok": bool(equal)}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    checks = replication_checks()
    with harness.deterministic_scoring():
        model = load_predictor(SEEDS[0])
        checks.append(rollout_checks(model))
        branches = complete_branches()
        first = rt.anchor_branches(branches)
        levels = [lv["identity"] for lv in policy_levels()]
        entries = policy_cell_entries()
        probe_levels = [m for m in levels if any(k[0] == m for k in entries)][:SMOKE_LEVELS]
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        anchors = {m: rt.anchor_carrier(adapter, first[m]["attempt"]) for m in probe_levels}
        t0 = time.monotonic()
        shot = shot_entry(first[probe_levels[0]], adapter)
        parse_seconds = time.monotonic()-t0
        del adapter
        cells = policy_cells({k: v for k, v in entries.items() if k[0] in probe_levels}, anchors)
        decision = decision_label_checks(model, cells, LAMBDAS["dec"])
        checks.append({"control": "decision_labels_brute_force_and_scale", **decision})
        window = rp.first_window(shot["carriers"], shot["action"], shot["end"], WINDOW_LENGTH)
        t0 = time.monotonic()
        pe_rows = rp.window_rows(model, rp.new_policy(PAIRS, SEEDS[0], device=DEVICE), [window] * 8, aggregated=True,
                                 device=DEVICE)
        pe_seconds = (time.monotonic()-t0)/8
    accounting = pool_accounting()
    checks.append({"control": "policy_pool_accounting", **accounting,
                   "ok": bool(accounting["levels"] == 120 and accounting["complete"] == 3480
                              and accounting["disjoint_from_fit_and_evaluation"])})
    store = gb.load_anchors(GATE_B)
    gate = read_json(GATE_B / "summary.json")
    checks.append({"control": "gate_b_inputs_bound", "gate_b": gate["gate_b"], "anchors": len(store["anchors"]),
                   "ok": bool(gate["gate_b"] == "pass" and store["scoring_policy"] == harness.SCORING_POLICY)})
    ok = all(c["ok"] for c in checks)
    timings = {"policy_shot_parse_seconds": parse_seconds, "pe_window_seconds_round1": pe_seconds,
               "pe_rows_per_window": len(pe_rows["labels"]) / 8,
               "dec_round0_seconds_per_cell": decision["round0_seconds"]/len(cells),
               "dec_dagger_seconds_per_cell": decision["dagger_round_seconds"]/len(cells),
               "pg_seconds_per_step": decision["pg_seconds_per_step"], "smoke_cells": len(cells)}
    wall = time.monotonic()-began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "pool_accounting": accounting, "timings": timings, "projection": projection(timings, accounting),
                "wall_seconds": wall,
                "reading": ("#77 artefacts, fit-split cells and anchors, and the first policy levels with a mixed cell "
                            "(training data); no evaluation-split model output"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s)")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: no freeze")
    return 0


def projection(timings, accounting):
    """Projected GPU seconds per phase from the smoke timings (DAgger rounds assumed 3x round 0 for dec)."""
    cells = sum(accounting["mixed_cells"].values())
    pe = 2*accounting["complete"]*timings["pe_window_seconds_round1"]+2*120
    dec = cells*(timings["dec_round0_seconds_per_cell"]+timings["dec_dagger_seconds_per_cell"])+2*120
    pg = PG["steps"]*timings["pg_seconds_per_step"]
    per_kind = {"pe": pe, "dec": dec, "pg": pg}
    training = sum(per_kind[k]*len(LAMBDAS[k])*len(SEEDS) for k in KINDS)
    data_ = accounting["complete"]*timings["policy_shot_parse_seconds"]
    return {"per_group_seconds": per_kind, "training_gpu_seconds": training, "data_gpu_seconds": data_,
            "scoring_gpu_seconds_estimate": 3*3600.0, "total_gpu_seconds": training+data_+3*3600.0,
            "elapsed_hours_on_4_gpus": (training+data_+3*3600.0)/4/3600}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_fit_verdicts": gb.FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(RETRAIN),
        "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
        "gate_b_compute": GATE_B / "compute.json", "gate_b_anchors": gb.anchors_path(GATE_B),
        **{f"gate_b_record_{n}_{s}": gb.record_path(GATE_B, n, s) for n in gb.UNITS for s in SEEDS},
        "gate_b_runner": ROOT / gb.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "request_policy_module": ROOT / MODULE, "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_{n}_{c['seed']}": gb.member_path(n, c["seed"]) for n in gb.UNITS for c in gb.checkpoints(n)})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "selection": ("pe: the ICLR lambda 1e-4 (no selection). dec and pg: the lambda of the sweep with the largest "
                      "seed-mean policy-split pooled AUC (greedy J rollouts from each policy level's anchor over the "
                      "mixed grid and angle cells; training data only), ties to the larger lambda; selected after "
                      "every group is trained and before any evaluation score"),
        "primary": (f"J_dec - F_hindsight paired grid AUC per (level, seed): supported iff mean >= {DELTA} with lower "
                    f"bound > 0; not_supported_by_this_experiment iff upper < {DELTA}; otherwise {STOP_TOKEN}; any "
                    f"failed guard -> {STOP_TOKEN}"),
        "secondary": ("J_dec - M_dec, J_dec - OL_dec, J_dec - F*, J_dec - J, J - F_hindsight, J - M, J - OL, J_pg - "
                      "F_hindsight, J_pg - J_dec, J_dec - prior, J_dec - C* per inventory with the same token rule; "
                      "offset, angle, power DESCRIPTIVE"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    rules = frozen_rules()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 101,
        "related_issues": [96, 104, 108, 109, 111, 112, 114, 116, 119, 121],
        "role": "train request policies on decision value vs prediction error; score once on the sealed #109 evaluation split",
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "shared_module": {"path": MODULE, "sha256_at_freeze": sha256_of(ROOT / MODULE)},
        "engine_seconds": 0,
        "question": ("does a request policy trained on decision value (J_dec) rank the candidate launches of the sealed "
                     "#109 evaluation split better than the best fixed request in hindsight, and do jointness (J_dec - "
                     "M) and state dependence (J_dec - OL) then carry ranking value?"),
        "predictors": {"unit": "LC hybrid fold 'all' (Gate-B handoff checkpoints, hash-checked), seeds "
                               f"{list(SEEDS)}; every policy pairs predictor seed s with policy seed s",
                       "c_star": "LC continuous C-F-15 (Gate-B C*), scored as a fixed reference arm only"},
        "pool": {
            "split": ("#109 policy split (issue-109-sealed-cohort-v1 splits.policy, 60 levels per family), captured "
                      f"by the #104 campaign stream {STREAM}; split_roles.policy = controller (policy) training. It "
                      "replaces the 3 #77 controller lineages (owner comment on #101). The fit split stays "
                      "predictor-only; the evaluation split is held out"),
            "accounting": smoke_evidence["pool_accounting"],
            "shots": ("E99 carriers (s99.carriers_for, positions 0..min(frames-1, 285)) of every complete branch, "
                      "under deterministic_scoring()"),
            "anchors": "per level: E99 batch-1 parse of shot frame 0 of its first complete candidate (retrain rule)",
            "cells": "per level x {grid, angle}: complete verdict-bearing candidates, ordinal order, mixed only",
        },
        "request_set": list(t96.PAIR_NAMES),
        "policy": {"architecture": "Linear(242,128)-SiLU-Linear(128,9) (= MatchedController(False))",
                   "remaining_feature": "clamp60: min(remaining, 60) / 60 (ICLR)",
                   "init": "torch.manual_seed(seed + 7700)"},
        "kinds": {
            "pe": {"name": KIND_NAMES["pe"], "labels": (
                "ICLR eq. dp (matched_dynamics.controller_labels with lambda as a parameter): the first 60-frame "
                "window of every pool shot (3480), Delta x one-step carrier MSE + lambda x work / work(15,cont), "
                "backward DP; DAgger round 1 on contexts visited by the round-0 policy"),
                   "lambdas": list(LAMBDAS["pe"])},
            "dec": {"name": KIND_NAMES["dec"], "labels": (
                "request_policy.decision_rows: per mixed policy cell, every candidate rolled out from the level anchor "
                f"under the continuation ({BASE_CONTINUATION} in round 0, the round-0 policy in the DAgger round); at "
                "each visited state and each feasible request: deviate, continue to 225, Q = the candidate's pairwise "
                f"softplus ranking loss (T = {rp.RANK_TEMPERATURE}) against the other candidates' continuation costs "
                "+ lambda x work of the remaining transitions / work(15,cont); label = argmin Q"),
                    "lambdas": list(LAMBDAS["dec"])},
            "pg": {"name": KIND_NAMES["pg"], "training": (
                f"request_policy.policy_gradient: {PG['steps']} AdamW updates (lr {PG['learning_rate']}, clip "
                f"{PG['grad_clip']}), {PG['cells_per_step']} cells x {PG['samples']} sampled schedule draws per update; "
                "reward = cell tie-free AUC - lambda x mean work units; advantage vs the cell's mean draw"),
                   "lambdas": list(LAMBDAS["pg"])},
            "distillation": {**rp.DISTIL_RECIPE, "unit": "policy level (roster order), step s -> unit s % units",
                             "sampler": "CPU Generator(seed + 7700 + round)", "loss": "cross-entropy"},
        },
        "arms": {"fixed": list(FIXED_ARMS), "c_star": C_STAR, "policies": list(POLICY_ARMS),
                 "J": J_ICLR, "M": "factorized selector (t96) on J / J_dec weights", "OL": (
                     "t96 open loop: per seed the pooled request frequencies of J / J_dec over every evaluation cell "
                     "of all four inventories and every candidate; one PCG64(7202) stream per OL arm in the frozen "
                     "order (inventory, level, seed)"),
                 "F_star": "cross-fit r-dagger: fixed arm with the best mean member-mean AUC over the other three inventories",
                 "F_hindsight": "in-inventory best fixed arm by member-mean AUC (outcome-selected contrast target)",
                 "controls": "no-model action prior (Gate-B rule, fit-verdicts.json); wrong-anchor control of J_dec "
                             "(480 fit anchors, grid)"},
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "anchors": "the Gate-B anchor store (issue-108-gate-b-v1/data/anchors-evaluation.pt)",
                       "primary_inventory": PRIMARY, "descriptive": list(gb.DESCRIPTIVE_INVENTORIES),
                       "cost": "tie_free (EndpointCosts.row against the anchor)",
                       "cell_auc": "m99.view; cells (level, seed); bootstrap over levels (10000 draws, PCG64 7201)"},
        "decision_rule": rules,
        "guards": {"G1_isolation": "pool = policy roster, disjoint from fit and evaluation rosters; scored cells are evaluation levels",
                   "G2_scoring_policy": "every record carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= first training <= last training <= selection <= first score",
                   "G4_checkpoint_binding": "every scored policy's sha256 equals its training report",
                   "G5_gate_b_replication": "F-15-macro and C-F-15 rows equal the Gate-B unit and C* records exactly",
                   "failure": f"a failed guard turns every token into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training or selection", "re-scoring after publication",
                       "changing a lambda, rule or arm after the freeze", "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"], "scope_rule": (
                     "if the projected elapsed wall exceeds 36 h, cut to the acceptance minimum (pe and dec at 4 "
                     "lambdas, pg dropped) and record why on #101")},
        "disclosure": ("Known before the freeze: Gate B (pass; unit 0.7444, C* 0.7834, prior 0.7203 on grid), every "
                       "#96/#100/#112 result. No policy has been trained on the #112 predictors; no policy output on an "
                       "evaluation capture exists. Hardware: RTX 5090 instead of the #85 RTX 3090. The decision-label "
                       "lambda grid was set from the smoke's label distribution on four policy levels (training data)."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split for request policies trained on the #109 "
                           "policy split; grid ranking primary, other inventories descriptive; tie-free cost; "
                           "DESCRIPTIVE intervals; no closed-loop claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-101 protocol")
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
    for name in ("records", "models", "data", "shots"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
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
    for name, text in rendered(plan, compute).items():
        (output / name).write_text(text)
    log(f"published: primary {compute['primary']['token']} ({fmt_interval(compute['primary']['estimate'])})")
    return 0


def spot_checks(output):
    """Re-score J_dec's first seed on every grid cell in this process (exact row equality)."""
    selection = load_selection(output)
    arm = primary_arms(selection)["J_dec"]
    seed = SEEDS[0]
    record = read_record(output, arm, seed)
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    costs = endpoint_costs()
    kind, lam, schedule_kind = arm_policy(arm, selection)
    same = True
    with harness.deterministic_scoring():
        model = load_predictor(seed)
        policy = load_policy(output, kind, lam, seed)
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            fresh = block(model, policy, {"kind": schedule_kind}, anchors[member], entry["inventory"], costs, True)
            same &= fresh["rows"] == record["inventories"][PRIMARY][member]["rows"]
            same &= fresh["schedules"] == record["inventories"][PRIMARY][member]["schedules"]
    return {"arm": arm, "seed": seed, "pass": bool(same)}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    if selection_table(output) != load_selection(output)["kinds"]:
        raise ValueError("the selection differs from the training reports")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic()-began:.0f}s): tables re-derived and byte-compared, selection "
        f"re-derived, {spot['arm']} seed {spot['seed']} re-scored exactly; primary {compute['primary']['token']}")
    return 0


def dry_run(output):
    accounting = pool_accounting()
    log(f"output {Path(output)}; pool {accounting}")
    log(f"groups {len(groups())}; arms {len(ARMS)} x seeds {len(SEEDS)}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "data": data, "assemble": assemble,
         "train": train, "select": select, "score": score, "publish": publish, "validate": validate}
SHARDED = ("data", "train", "score")


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
