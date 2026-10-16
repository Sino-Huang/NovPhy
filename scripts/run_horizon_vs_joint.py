"""Issue-114: horizon-only versus joint request selection at matched compute, scored once on the #104
sealed #109 evaluation split.

Binding runner module: scripts/run_horizon_vs_joint.py
Exact validation command: python -u -m scripts.run_horizon_vs_joint --validate

#114 separates the value of choosing the description alpha from the value of choosing the horizon
Delta at matched compute. This runner

1. scores restricted-argmax arms of the #101 J policy (ICLR prediction-error DP, lambda 1e-4, the
   #109 policy split, LC hybrid Gate-B predictors) without training: T-{continuous, micro, macro}
   (argmax over Delta at a fixed alpha) and D-{1, 5, 15} (argmax over alpha at a fixed Delta), with the
   shared selector module ``world_model/training/request_selectors.py``;
2. trains J-cont: the #101 J recipe (pool, seeds, 2 DAgger rounds x 1800 steps, batch 128, AdamW 1e-3,
   the same 242 -> 128 -> 9 policy and initialization) with the #77 DP labels restricted to
   R_cont = {(1, cont), (5, cont), (15, cont)} and the other six logits masked;
3. sweeps lambda over 8 values for J (four reused from #101, four trained here with the same path),
   J-cont and C-J (the horizon-only adaptive policy on the LC continuous C* predictor,
   ``MatchedController(True)`` architecture) and reports tie-free AUC, top-3, MACs per candidate and
   state, GPU seconds per state and ties; the matched-compute contrasts J - J-cont and J - C-J
   interpolate each frontier (piecewise linear in log10 MACs) at the MAC budgets of J's and C-J's
   lambda = 1e-4 points;
4. decomposes the MAC difference per candidate and state into predictor calls (horizon effect),
   per-call alpha cost (alpha effect; for C-J also the trunk) and policy calls, and scores the
   counterfactual "J's schedule with every alpha replaced by continuous" (per-row replay);
5. reports the no-model action prior and the wrong-anchor control in every table.

Modes: --dry-run --smoke --prepare --train --score --publish --validate. --train and --score take
--workers N --worker K (CUDA_VISIBLE_DEVICES set by the caller). Zero engine seconds. Intervals:
member (level)-clustered percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts import post_freeze_amendments as amendments
from scripts import run_decision_aware_policy as dap
from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_gate_b as gb
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import request_policy as rp
from world_model.training import request_selectors as rs
from world_model.training.cnn_hybrid import PAIRS, linear_macs
from world_model.training.matched_dynamics import CONTINUOUS_PAIRS

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-114-horizon-vs-joint-v1"
P101 = dap.OUTPUT
GATE_B = gb.OUTPUT

IDENTITY = "issue-114-horizon-vs-joint-v1"
SCHEMA_PLAN = "issue_114_horizon_vs_joint_plan_v1"
SCHEMA_SMOKE = "issue_114_horizon_vs_joint_smoke_v1"
SCHEMA_POLICY = "issue_114_request_policy_v1"
SCHEMA_REPORT_TRAIN = "issue_114_training_report_v1"
SCHEMA_RECORD = "issue_114_record_v1"
SCHEMA_COMPUTE = "issue_114_compute_v1"
SCHEMA_REPORT = "issue_114_report_v1"
SCHEMA_FRONTIER = "issue_114_frontier_points_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_horizon_vs_joint --validate"
RUNNER = "scripts/run_horizon_vs_joint.py"
SELECTORS = "world_model/training/request_selectors.py"
MODULE = dap.MODULE
FIGURE = "frontier-grid.png"
FRONTIER_FILE = "frontier.json"

DEVICE = "cuda"
SEEDS = dap.SEEDS
INVENTORIES = dap.INVENTORIES
PRIMARY = dap.PRIMARY
PRIMARY_COST = harness.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
FIXED_ARMS = t96.FIXED_ARMS
C_FIXED_ARMS = t96.CONTINUOUS_FIXED_ARMS
ICLR_LAMBDA = rp.ICLR_COMPUTE_WEIGHT
LAMBDAS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
PROBE_LAMBDAS = (0.0, 1e-6, 0.3, 1.0, 10.0)  # smoke only: saturation of the label mix outside the grid
REUSED_J = dap.LAMBDAS["pe"]  # #101 J-pe policies (same pool, seeds, recipe)
R_CONT = tuple(i for i, p in enumerate(PAIRS) if p in CONTINUOUS_PAIRS)
FAMILIES = ("J", "Jcont", "CJ")
FAMILY_NAMES = {"J": "J (all nine requests, hybrid trunk)", "Jcont": "J-cont (R_cont on the hybrid trunk, six logits masked)",
                "CJ": "C-J (horizon-only adaptive on the LC continuous C* predictor)"}
CJ_WIDTH = 131  # MatchedController(True)
ALPHA_ARMS = tuple(f"T-{alpha}" for alpha in t96.ALPHAS)
DELTA_ARMS = tuple(f"D-{delta}" for delta in t96.DELTAS)
RESTRICTED_ARMS = (*ALPHA_ARMS, *DELTA_ARMS)
GPU_CAP_SECONDS = 12 * 3600.0
WALL_CAP_SECONDS = 24 * 3600.0  # summed over processes (2 GPUs x 2 processes)
SMOKE_UNITS = 3

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval
tag = dap.tag


def log(message):
    print(f"[issue-114] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def family_arm(family, lam):
    return f"{family}-{tag(lam)}"


def ascont_arm(lam):
    return f"Jascont-{tag(lam)}"


J_ICLR = family_arm("J", ICLR_LAMBDA)
JCONT_ICLR = family_arm("Jcont", ICLR_LAMBDA)
CJ_ICLR = family_arm("CJ", ICLR_LAMBDA)
POLICY_ARMS = tuple(family_arm(f, lam) for f in FAMILIES for lam in LAMBDAS)
ASCONT_ARMS = tuple(ascont_arm(lam) for lam in LAMBDAS)
PHASE_1 = (*FIXED_ARMS, *C_FIXED_ARMS, *POLICY_ARMS, *RESTRICTED_ARMS)
PHASE_2 = ASCONT_ARMS
ARMS = (*PHASE_1, *PHASE_2)
WRONG_ARMS = (J_ICLR, JCONT_ICLR, CJ_ICLR)


def groups():
    """Training groups (family, lambda, seed) not reused from #101; worker k trains groups[k::workers]."""
    return [(f, lam, s) for f in FAMILIES for lam in LAMBDAS for s in SEEDS
            if not (f == "J" and lam in REUSED_J)]


def parse_arm(arm):
    """(family, lambda) of a policy or as-continuous arm."""
    family, text = arm.split("-", 1)
    return family, next(lam for lam in LAMBDAS if tag(lam) == text)


def predictor_name(arm):
    return "c_star" if arm in C_FIXED_ARMS or arm.startswith("CJ-") else "unit"


# ---------------------------------------------------------------------------
# paths and ledger
# ---------------------------------------------------------------------------

def group_root(output, family, lam, seed):
    return Path(output) / "models" / family / f"lam-{tag(lam)}" / f"seed-{seed}"


def policy_path(output, family, lam, seed):
    if family == "J" and lam in REUSED_J:
        return dap.policy_path(P101, "pe", lam, seed)
    return group_root(output, family, lam, seed) / "policy.pt"


def report_path(output, family, lam, seed):
    if family == "J" and lam in REUSED_J:
        return dap.report_path(P101, "pe", lam, seed)
    return group_root(output, family, lam, seed) / "report.json"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def ledger_path(output, phase):
    return Path(output) / "ledger" / f"{phase}.json"


def ledger_phases(output):
    root = Path(output) / "ledger"
    return {p.stem: read_json(p) for p in sorted(root.glob("*.json"))} if root.is_dir() else {}


def mark(output, phase, wall, gpu, **extra):
    write_json(ledger_path(output, phase), {"phase": phase, "wall_seconds": wall, "gpu_seconds": gpu,
                                            "at": utc_now(), **extra})
    phases = ledger_phases(output)
    totals = {"gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
              "wall_seconds": sum(b["wall_seconds"] for b in phases.values())}
    if totals["gpu_seconds"] > GPU_CAP_SECONDS or totals["wall_seconds"] > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded ({totals})")


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------

def new_family_policy(family, seed):
    if family == "J":
        return rp.new_policy(PAIRS, seed, device=DEVICE)
    if family == "Jcont":
        return rs.new_masked_policy(PAIRS, R_CONT, seed, device=DEVICE)
    return rp.new_policy(CONTINUOUS_PAIRS, seed, width=CJ_WIDTH, device=DEVICE)


def save_policy(output, family, lam, seed, policy, extra):
    atomic_torch(policy_path(output, family, lam, seed), {
        "schema": SCHEMA_POLICY, "plan_identity": IDENTITY, "family": family, "lambda": lam, "seed": seed,
        "pairs": [str(p.identity) for p in policy.pairs], "allowed": list(getattr(policy, "allowed", range(len(policy.pairs)))),
        "width": policy.layers[0].out_features, "remaining_feature": policy.remaining_feature,
        "model": policy.state_dict(), **extra})


def load_policy(output, family, lam, seed):
    if family == "J" and lam in REUSED_J:
        return dap.load_policy(P101, "pe", lam, seed)
    payload = torch.load(policy_path(output, family, lam, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_POLICY or payload.get("plan_identity") != IDENTITY
            or (payload["family"], payload["lambda"], payload["seed"]) != (family, lam, seed)):
        raise ValueError(f"policy {family}/{lam}/{seed} binding differs")
    if family == "Jcont":
        policy = rs.MaskedRequestPolicy(PAIRS, payload["allowed"], payload["width"], payload["remaining_feature"])
    else:
        policy = rp.RequestPolicy(PAIRS if family == "J" else CONTINUOUS_PAIRS, payload["width"],
                                  payload["remaining_feature"])
    if payload["pairs"] != [str(p.identity) for p in policy.pairs]:
        raise ValueError("policy request set differs")
    policy.load_state_dict(payload["model"], strict=True)
    return policy.to(DEVICE).eval()


def policy_sha(output, family, lam, seed):
    return sha256_of(policy_path(output, family, lam, seed))


# ---------------------------------------------------------------------------
# training (one group = family x lambda x seed; the #101 pe path)
# ---------------------------------------------------------------------------

def family_rows(family, model, policy, unit, aggregated, lam):
    if family == "Jcont":
        return rs.restricted_window_rows(model, policy, unit, R_CONT, aggregated=aggregated, compute_weight=lam,
                                         device=DEVICE)
    return rp.window_rows(model, policy, unit, aggregated=aggregated, compute_weight=lam, device=DEVICE)


def train_group(output, family, lam, seed):
    root = group_root(output, family, lam, seed)
    if report_path(output, family, lam, seed).is_file():
        log(f"{family}/{tag(lam)}/{seed}: done")
        return
    began = time.monotonic()
    pool = dap.load_pool(P101)
    units = dap.window_units(pool)
    model = dap.load_predictor(seed, "c_star" if family == "CJ" else "unit")
    with harness.deterministic_scoring() as scoring_policy:
        policy = new_family_policy(family, seed)
        stats, history = {}, []
        for round_index in range(rp.DISTIL_RECIPE["rounds"]):
            checkpoint = root / f"round-{round_index}.pt"
            labels = root / f"labels-{round_index}.pt"
            if labels.is_file():
                history.append(torch.load(labels, map_location="cpu", weights_only=True)["units"])
            else:
                t0 = time.monotonic()
                rows = [family_rows(family, model, policy, unit, bool(round_index), lam) for unit in units]
                atomic_torch(labels, {"units": rows})
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
            log(f"{family}/{tag(lam)}/{seed}: round {round_index} rows {sum(len(u['labels']) for u in history[-1])}, "
                f"loss {stats[f'final_loss_{round_index}']:.4f}")
        stats["labels"] = {f"round_{r}": dap.rows_stats([h]) for r, h in enumerate(history)}
        policy.eval()
        evaluation = dap.policy_split_evaluation(model, policy, pool)
    save_policy(output, family, lam, seed, policy, {"scoring_policy": scoring_policy})
    wall = time.monotonic()-began
    write_json(report_path(output, family, lam, seed), {
        "schema": SCHEMA_REPORT_TRAIN, "plan_identity": IDENTITY, "family": family, "lambda": lam, "seed": seed,
        "policy_sha256": policy_sha(output, family, lam, seed), "trained_at": utc_now(),
        "predictor_sha256": sha256_of(gb.member_path("c_star" if family == "CJ" else "unit", seed)),
        "stats": stats, "policy_split": evaluation, "wall_seconds": wall})
    mark(output, f"train-{family}-{tag(lam)}-seed{seed}", wall, wall)
    log(f"{family}/{tag(lam)}/{seed}: trained ({wall:.0f}s); policy-split AUC {evaluation['auc_pooled']:.4f}, "
        f"MACs/candidate {evaluation['mean_macs_per_candidate']:.3g}")


def train(output, workers, worker):
    output = Path(output)
    load_plan(output)
    dap.load_pool(P101)
    for family, lam, seed in groups()[worker::workers]:
        train_group(output, family, lam, seed)
    return 0


def load_report(output, family, lam, seed):
    report = read_json(report_path(output, family, lam, seed))
    if family == "J" and lam in REUSED_J:
        if report.get("schema") != dap.SCHEMA_REPORT_TRAIN or (report["kind"], report["lambda"]) != ("pe", lam):
            raise ValueError("#101 training report binding differs")
        return report
    if report.get("schema") != SCHEMA_REPORT_TRAIN or report.get("plan_identity") != IDENTITY:
        raise ValueError("training report binding differs")
    return report


# ---------------------------------------------------------------------------
# scoring (deterministic_scoring, processes that did not train)
# ---------------------------------------------------------------------------

def restricted_spec(arm):
    if arm in ALPHA_ARMS:
        return rs.temporal_spec(PAIRS, arm[2:])
    return rs.description_spec(PAIRS, int(arm[2:]))


def block(model, policy, spec, z0, items, costs, schedules):
    actions = m99.action_batch(items)
    z_end, timing = rs.rollout_endpoints(model, policy, z0.to(DEVICE), actions, spec, select=t96.select_pairs,
                                         record_schedules=schedules)
    out = {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
           "candidates": len(items), **{k: v for k, v in timing.items() if k != "schedules"}}
    if schedules:
        out["schedules"] = dap.rle_schedules(items, timing)
    return out


def as_continuous(sequence):
    """Every request (Delta, alpha) -> (Delta, continuous) (PAIRS index 3 * (k // 3))."""
    return [3*(k//3) for k in sequence]


def replay_block(model, source, z0, items, costs):
    """J's per-candidate schedules with alpha replaced by continuous; a candidate whose J endpoint is a
    typed failure (nonfinite, truncated schedule) keeps that failure and is not replayed."""
    complete = [k for k, item in enumerate(items) if not source["rows"][k]["excluded"]]
    rows = [dict(r) for r in source["rows"]]
    out = {"candidates": len(items), "transition_calls": 0, "controller_calls": 0, "linear_macs": 0,
           "gpu_seconds": 0.0, "replayed": len(complete)}
    if complete:
        chosen = [items[k] for k in complete]
        schedules = [as_continuous(t96.unrle(source["schedules"][str(item["ordinal"])])) for item in chosen]
        z_end, timing = rs.rollout_endpoints(model, None, z0.to(DEVICE), m99.action_batch(chosen),
                                             rs.replay_spec(schedules), select=t96.select_pairs)
        for slot, k in enumerate(complete):
            rows[k] = costs.row(items[k]["ordinal"], z_end[slot].cpu(), z0.cpu())
        out.update({k: timing[k] for k in ("transition_calls", "controller_calls", "linear_macs", "gpu_seconds")})
    out["rows"] = rows
    return out


def score_record(output, cells, anchors, arm, seed, policy_state):
    costs = dap.endpoint_costs()
    began = time.monotonic()
    name = predictor_name(arm)
    model = dap.load_predictor(seed, name)
    policy, schedules, extra, spec, source = None, False, {}, None, None
    if arm in FIXED_ARMS or arm in C_FIXED_ARMS:
        spec = t96.arm_spec(arm)
    elif arm in PHASE_2:
        _, lam = parse_arm(arm)
        source = read_record(output, family_arm("J", lam), seed)
        extra = {"replay_of": family_arm("J", lam)}
    else:
        family, lam = (("J", ICLR_LAMBDA) if arm in RESTRICTED_ARMS else parse_arm(arm))
        policy = load_policy(output, family, lam, seed)
        spec = restricted_spec(arm) if arm in RESTRICTED_ARMS else {"kind": "joint"}
        schedules = True
        extra = {"policy": {"family": family, "lambda": lam, "sha256": policy_sha(output, family, lam, seed)},
                 "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}}
    inventories = {i: {} for i in INVENTORIES}
    for inventory, member, entry in dap.ordered_cells(cells):
        if source is not None:
            inventories[inventory][member] = replay_block(model, source["inventories"][inventory][member],
                                                          anchors[member], entry["inventory"], costs)
        else:
            inventories[inventory][member] = block(model, policy, spec, anchors[member], entry["inventory"], costs,
                                                   schedules)
    torch.cuda.synchronize()
    return {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": arm, "seed": seed, "engine_seconds": 0, **extra,
            "predictor_sha256": sha256_of(gb.member_path(name, seed)), "inventories": inventories,
            "seconds": time.monotonic()-began}


def wrong_record(output, arm, seed, policy_state):
    """The arm's policy from every fit-split anchor over the shared grid candidates (Gate-B rule)."""
    costs = dap.endpoint_costs()
    family, lam = parse_arm(arm)
    model = dap.load_predictor(seed, predictor_name(arm))
    policy = load_policy(output, family, lam, seed)
    items = gb.inventory_items(PRIMARY)
    anchors = gb.fit_anchors()
    began = time.monotonic()
    rows = {member: block(model, policy, {"kind": "joint"}, anchors[member], items, costs, False)
            for member in sorted(anchors)}
    return {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy_state,
            "scored_at": utc_now(), "arm": f"wrong-{arm}", "seed": seed, "engine_seconds": 0,
            "policy": {"family": family, "lambda": lam, "sha256": policy_sha(output, family, lam, seed)},
            "wrong_anchor": rows, "seconds": time.monotonic()-began}


def read_record(output, arm, seed):
    record = read_json(record_path(output, arm, seed))
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY or record["seed"] != seed:
        raise ValueError(f"record {arm}/{seed} binding differs")
    return record


def score_tasks():
    """Phase 1 heaviest first (Delta = 1 policies and fixed requests), so the shards balance."""
    phase_1 = ([(arm, s) for arm in PHASE_1 for s in SEEDS] + [(f"wrong-{a}", s) for a in WRONG_ARMS for s in SEEDS])
    return phase_1, [(arm, s) for arm in PHASE_2 for s in SEEDS]


def score(output, workers, worker):
    output = Path(output)
    load_plan(output)
    missing = [g for g in groups() if not report_path(output, *g).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} training groups missing; scoring follows training")
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    phase_1, phase_2 = score_tasks()
    for phase, tasks in (("1", phase_1), ("2", phase_2)):
        if phase == "2":
            needed = [family_arm("J", lam) for lam in LAMBDAS]
            if any(not record_path(output, a, s).is_file() for a in needed for s in SEEDS):
                log(f"worker {worker}: phase 2 waits for the J records")
                return 0
        for arm, seed in tasks[worker::workers]:
            if record_path(output, arm, seed).is_file():
                continue
            t0 = time.monotonic()
            with harness.deterministic_scoring() as policy_state:
                record = (wrong_record(output, arm[6:], seed, policy_state) if arm.startswith("wrong-")
                          else score_record(output, cells, anchors, arm, seed, policy_state))
            write_json(record_path(output, arm, seed), record)
            torch.cuda.empty_cache()
            wall = time.monotonic()-t0
            mark(output, f"score-{arm}-seed{seed}", wall, wall)
            log(f"worker {worker}: scored {arm} seed {seed} ({wall:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def view(entry, rows):
    """v1.cost_view plus top-3: hit iff a success is among the three lowest-cost admissible candidates
    (cost, then ordinal); chance3 = 1 - C(n - k, 3) / C(n, 3)."""
    base = dict(v1.cost_view(entry, rows, PRIMARY_COST))
    if rows is None:
        return {**base, "top3_hit": None, "chance_top3": None}
    values = {row["ordinal"]: row[PRIMARY_COST] for row in rows}
    verdicts = entry["verdicts"]
    admissible = sorted(((o, values[o]) for o in verdicts if values.get(o) is not None), key=lambda p: (p[1], p[0]))
    n, k = len(admissible), sum(1 for o, _ in admissible if verdicts[o])
    if not n:
        return {**base, "top3_hit": None, "chance_top3": None}
    chance = 1.0 if n <= 3 else 1.0-math.comb(n-k, 3)/math.comb(n, 3)
    return {**base, "top3_hit": float(any(verdicts[o] for o, _ in admissible[:3])), "chance_top3": chance}


def load_records(output):
    return {(arm, s): read_record(output, arm, s) for arm in ARMS for s in SEEDS}


def view_rows(records, cells, inventory, stats):
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = view(entry, gb.prior_rows(entry, stats))
        for seed in SEEDS:
            views = {arm: view(entry, records[(arm, seed)]["inventories"][inventory][member]["rows"]) for arm in ARMS}
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def compute_constants():
    """Per-call MACs of each request on each predictor, and per policy call (from the checkpoints)."""
    hybrid = dap.load_predictor(SEEDS[0], "unit")
    continuous = dap.load_predictor(SEEDS[0], "c_star")
    return {"hybrid": {t96.PAIR_NAMES[k]: linear_macs(hybrid, p) for k, p in enumerate(PAIRS)},
            "continuous": {f"{p.delta}-continuous": linear_macs(continuous, p) for p in CONTINUOUS_PAIRS},
            "policy": {"J": (242*128+128*9), "Jcont": (242*128+128*9), "CJ": (242*CJ_WIDTH+CJ_WIDTH*3)}}


def arm_compute(records, inventory, arm):
    results = [records[(arm, s)]["inventories"][inventory][m] for s in SEEDS
               for m in records[(arm, s)]["inventories"][inventory]]
    candidates = sum(r["candidates"] for r in results)
    return {"mean_macs_per_candidate": sum(r["linear_macs"] for r in results)/candidates,
            "mean_transitions_per_candidate": sum(r["transition_calls"] for r in results)/candidates,
            "mean_policy_calls_per_candidate": sum(r["controller_calls"] for r in results)/candidates,
            "gpu_seconds_per_state": sum(r["gpu_seconds"] for r in results)/len(results),
            "gpu_seconds": sum(r["gpu_seconds"] for r in results), "states": len(results),
            "candidates": candidates}


def schedule_diagnostics(records, inventory, arm):
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
    names = (t96.PAIR_NAMES if not arm.startswith("CJ-") else
             tuple(f"{d}-continuous" for d in t96.DELTAS))
    modal = max(range(len(counts)), key=lambda k: (counts[k], -k))
    return {"pair_counts": counts[:len(names)], "pair_names": list(names), "modal_pair": names[modal],
            "non_modal_step_share": 1-counts[modal]/total if total else None,
            "cells_with_differing_schedules": differing, "cells": cells}


def ranker(rows, records, inventory, arm):
    views = [r["views"][arm] for r in rows]
    pairs = sum(v["pairs"] for v in views)
    out = {**v1.ranker_block(rows, arm), "tied_pairs": sum(v["tied_pairs"] for v in views), "pairs": pairs,
           "top3": m99.block_of(rows, arm, "top3_hit"),
           "chance_top3": float(np.mean([v["chance_top3"] for v in views if v["chance_top3"] is not None]))}
    if arm != "prior":
        out.update(arm_compute(records, inventory, arm))
    return out


def contrast_block(rows, pieces, label):
    block = m99.paired(rows, pieces)
    return {**block, "contrast": label, "pieces": [[w, k] for w, k in pieces], "token": c2.token(block["estimate"])}


def interpolation(points, budget):
    """Pieces [(weight, arm)] of the frontier's AUC at ``budget`` MACs: piecewise linear in log10 MACs
    through every lambda point sorted by MACs (ties by lambda). Above the frontier's largest MACs the
    policy cannot spend the budget: its largest-MAC point is used (it spends less; flagged by the caller).
    Below the smallest MACs the policy cannot meet the budget: None (no extrapolation)."""
    ordered = sorted(points, key=lambda p: (p["mean_macs_per_candidate"], p["lambda"]))
    macs = [p["mean_macs_per_candidate"] for p in ordered]
    if budget < macs[0]:
        return None
    if budget >= macs[-1]:
        return [(1.0, ordered[-1]["arm"])]
    for p in ordered:
        if p["mean_macs_per_candidate"] == budget:
            return [(1.0, p["arm"])]
    for a, b in zip(ordered, ordered[1:]):
        if a["mean_macs_per_candidate"] < budget < b["mean_macs_per_candidate"]:
            la, lb = math.log10(a["mean_macs_per_candidate"]), math.log10(b["mean_macs_per_candidate"])
            w = (lb-math.log10(budget))/(lb-la)
            return [(w, a["arm"]), (1-w, b["arm"])]
    raise AssertionError("budget bracketing failed")


def frontier_tables(rows, rankers):
    families = {}
    for family in FAMILIES:
        points = []
        for lam in LAMBDAS:
            arm = family_arm(family, lam)
            r = rankers[arm]
            points.append({"lambda": lam, "arm": arm, "auc": r["auc"], "top3": r["top3"], "tied_cells": r["tied_cells"],
                           "tied_pairs": r["tied_pairs"], "pairs": r["pairs"],
                           **{k: r[k] for k in ("mean_macs_per_candidate", "mean_transitions_per_candidate",
                                                "mean_policy_calls_per_candidate", "gpu_seconds_per_state")}})
        families[family] = points
    budgets = {"J_lambda_1e-04": rankers[J_ICLR]["mean_macs_per_candidate"],
               "CJ_lambda_1e-04": rankers[CJ_ICLR]["mean_macs_per_candidate"]}
    matched = {}
    for name, budget in budgets.items():
        pieces = {f: interpolation(families[f], budget) for f in FAMILIES}
        entry = {"budget_macs_per_candidate": budget,
                 "pieces": {f: ([[w, a] for w, a in p] if p else None) for f, p in pieces.items()},
                 "budget_above_frontier": {f: budget > max(p["mean_macs_per_candidate"] for p in families[f])
                                           for f in FAMILIES},
                 "auc": {f: (m99.paired(rows, p)["estimate"] if p else None) for f, p in pieces.items()},
                 "contrasts": {}}
        for other in ("Jcont", "CJ"):
            label = f"J - {'J-cont' if other == 'Jcont' else 'C-J'}"
            if pieces["J"] is None or pieces[other] is None:
                entry["contrasts"][other] = {"contrast": label, "estimate": None, "token": STOP_TOKEN,
                                             "reason": "budget below a frontier's smallest MACs (no extrapolation)"}
            else:
                entry["contrasts"][other] = contrast_block(rows, pieces["J"] + [(-w, a) for w, a in pieces[other]],
                                                           label)
        matched[name] = entry
    return {"points": families, "budgets": budgets, "matched": matched}


def decomposition(rankers, constants, first, second):
    """MAC difference per candidate and state of ``first`` minus ``second``: (a) predictor calls at the
    second arm's per-call cost (horizon effect), (b) the first arm's calls times the per-call cost
    difference (alpha effect; for a continuous-trunk second arm split into alpha and trunk), (c) policy
    calls. (a) + (b) + (c) equals the difference exactly."""
    def parts(arm):
        r = rankers[arm]
        family = arm.split("-", 1)[0]
        policy = constants["policy"].get(family, 0)
        n, p = r["mean_transitions_per_candidate"], r["mean_policy_calls_per_candidate"]
        predictor = r["mean_macs_per_candidate"]-p*policy
        return {"calls": n, "policy_calls": p, "predictor_macs": predictor, "per_call": predictor/n,
                "policy_macs": p*policy, "total": r["mean_macs_per_candidate"]}
    a, b = parts(first), parts(second)
    calls = (a["calls"]-b["calls"])*b["per_call"]
    per_call = a["calls"]*(a["per_call"]-b["per_call"])
    policy = a["policy_macs"]-b["policy_macs"]
    out = {"first": first, "second": second, "first_parts": a, "second_parts": b,
           "difference": a["total"]-b["total"], "a_calls_horizon": calls, "b_per_call": per_call,
           "c_policy": policy, "residual": a["total"]-b["total"]-calls-per_call-policy}
    if second.startswith("CJ-"):
        cont = constants["hybrid"]["1-continuous"]
        out["b_alpha"] = a["calls"]*(a["per_call"]-cont)
        out["b_trunk"] = a["calls"]*(cont-b["per_call"])
    else:
        out["b_alpha"], out["b_trunk"] = per_call, 0.0
    return out


def wrong_anchor_table(output, records, cells, arm):
    rows = []
    wrong_records = {s: read_record(output, f"wrong-{arm}", s)["wrong_anchor"] for s in SEEDS}
    for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
        for seed in SEEDS:
            right = records[(arm, seed)]["inventories"][PRIMARY][member]["rows"]
            wrong = [wrong_records[seed][a]["rows"] for a in sorted(wrong_records[seed])]
            wrong_views = [view(entry, w) for w in wrong]
            wrong_auc = [v["auc"] for v in wrong_views if v["auc"] is not None]
            wrong_top3 = [v["top3_hit"] for v in wrong_views if v["top3_hit"] is not None]
            rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong) if r is not None]
            rows.append({"member": member, "seed": seed, "views": {"x": {
                "right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                "wrong_top3": float(np.mean(wrong_top3)) if wrong_top3 else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    block = c2.difference_block(rows, [(1, "x", "right"), (-1, "x", "wrong")])
    measured = [result for members in wrong_records.values() for result in members.values()]
    candidates = sum(result["candidates"] for result in measured)
    return {"arm": arm, "cells": len(rows), "wrong_anchors": len(gb.fit_anchors()),
            "right": c2.field_block(rows, "x", "right"), "wrong": c2.field_block(rows, "x", "wrong"),
            "top3": c2.field_block(rows, "x", "wrong_top3"),
            "mean_macs_per_candidate": sum(result["linear_macs"] for result in measured)/candidates,
            "gpu_seconds_per_state": sum(result["gpu_seconds"] for result in measured)/len(measured),
            "right_minus_wrong": block, "stability": c2.field_block(rows, "x", "stability"), "token": c2.token(block)}


CONTRASTS = {
    "J-Jcont": (J_ICLR, JCONT_ICLR), "J-T-continuous": (J_ICLR, "T-continuous"), "J-CJ": (J_ICLR, CJ_ICLR),
    "J-Jascont": (J_ICLR, ascont_arm(ICLR_LAMBDA)), "J-T-micro": (J_ICLR, "T-micro"), "J-T-macro": (J_ICLR, "T-macro"),
    "J-D-1": (J_ICLR, "D-1"), "J-D-5": (J_ICLR, "D-5"), "J-D-15": (J_ICLR, "D-15"),
    "Jcont-T-continuous": (JCONT_ICLR, "T-continuous"), "J-prior": (J_ICLR, "prior"), "Jcont-prior": (JCONT_ICLR, "prior"),
    "CJ-prior": (CJ_ICLR, "prior"),
}


def inventory_tables(records, rows, inventory, constants):
    rankers = {arm: ranker(rows, records, inventory, arm) for arm in (*ARMS, "prior")}
    contrasts = {name: contrast_block(rows, [(1, a), (-1, b)], f"{a} - {b}") for name, (a, b) in CONTRASTS.items()}
    frontier = frontier_tables(rows, rankers)
    decomp = {tag(lam): {"J-Jcont": decomposition(rankers, constants, family_arm("J", lam), family_arm("Jcont", lam)),
                         "J-CJ": decomposition(rankers, constants, family_arm("J", lam), family_arm("CJ", lam)),
                         "J-Jascont": decomposition(rankers, constants, family_arm("J", lam), ascont_arm(lam))}
              for lam in LAMBDAS}
    ascont = {tag(lam): contrast_block(rows, [(1, family_arm("J", lam)), (-1, ascont_arm(lam))],
                                       f"{family_arm('J', lam)} - {ascont_arm(lam)}") for lam in LAMBDAS}
    controller = (*POLICY_ARMS, *RESTRICTED_ARMS)
    return {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": rankers,
            "contrasts": contrasts, "frontier": frontier, "decomposition": decomp, "J_minus_ascont": ascont,
            "schedules": {arm: schedule_diagnostics(records, inventory, arm) for arm in controller}}


def disposition(matched, other):
    """supported iff J - other is supported at both declared budgets; not_supported_by_this_experiment
    iff its upper bound is < delta at either budget; otherwise readiness_or_precision_insufficient."""
    tokens = [m["contrasts"][other]["token"] for m in matched.values()]
    if all(t == "supported" for t in tokens):
        return "supported"
    if any(t == "not_supported_by_this_experiment" for t in tokens):
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def replication_guard(records):
    """Fixed-request and reused-J rows equal the #101 records exactly (and through #101 the Gate-B unit
    and C* records)."""
    same, checked = True, 0
    for mine, theirs in [(a, a) for a in (*FIXED_ARMS, "C-F-15")] + [(family_arm("J", lam), dap.policy_arm("pe", lam))
                                                                      for lam in REUSED_J]:
        for seed in SEEDS:
            reference = dap.read_record(P101, theirs, seed)["inventories"]
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    ours = records[(mine, seed)]["inventories"][inventory][member]
                    same &= ours["rows"] == result["rows"]
                    if "schedules" in result:
                        same &= ours["schedules"] == result["schedules"]
                    checked += 1
    for name, arm in (("unit", gb.UNITS["unit"]["request"]), ("c_star", "C-F-15")):
        for seed in SEEDS:
            reference = read_json(gb.record_path(GATE_B, name, seed))["inventories"]
            for inventory in INVENTORIES:
                for member, result in reference[inventory].items():
                    same &= records[(arm, seed)]["inventories"][inventory][member]["rows"] == result["rows"]
                    checked += 1
    return {"blocks": checked, "pass": bool(same)}


def subset_guard(records):
    """Every restricted / J-cont schedule uses only its allowed requests."""
    ok = True
    for arm in (*RESTRICTED_ARMS, *(family_arm("Jcont", lam) for lam in LAMBDAS)):
        allowed = set(restricted_spec(arm)["allowed"]) if arm in RESTRICTED_ARMS else set(R_CONT)
        for seed in SEEDS:
            for members in records[(arm, seed)]["inventories"].values():
                for result in members.values():
                    for text in result["schedules"].values():
                        ok &= set(t96.unrle(text)) <= allowed
    return {"pass": bool(ok)}


def training_table(output):
    out = {}
    for family in FAMILIES:
        for lam in LAMBDAS:
            for seed in SEEDS:
                report = load_report(output, family, lam, seed)
                out.setdefault(family, {}).setdefault(tag(lam), {})[str(seed)] = {
                    "policy_split": report["policy_split"], "labels": report["stats"].get("labels"),
                    "wall_seconds": report["wall_seconds"], "policy_sha256": report["policy_sha256"],
                    "source": "issue-101 pe" if family == "J" and lam in REUSED_J else IDENTITY}
    return out


GUARD_FAILURE_NOTE = (f"the frozen rule turns the primary and secondary dispositions into {STOP_TOKEN}; every other "
                      "token in this publication (fixed-lambda, matched-budget, restricted-arm, inventory and "
                      "wrong-anchor contrasts) is not interpretable either and must not be cited as a result")


def guard_failure(guards):
    """Explicit report of a failed guard (phase-A review): the frozen wording invalidates only the two
    dispositions, so the publication states that its secondary tokens are not interpretable. None if
    every guard passes (the publication is then unchanged)."""
    failed = sorted(name for name, g in guards.items() if not g["pass"])
    if not failed:
        return None
    return {"failed_guards": failed, "secondary_tokens_interpretable": False, "note": GUARD_FAILURE_NOTE}


def compute_tables(output, plan):
    output = Path(output)
    branches, cells = gb.evaluation_set()
    records = load_records(output)
    fit = read_json(gb.FIT_VERDICTS)["branches"]
    constants = plan["per_call_macs"]
    stats = {i: gb.prior_stats(fit, i) for i in INVENTORIES}
    rows = {i: view_rows(records, cells, i, stats[i]) for i in INVENTORIES}
    tables = {i: inventory_tables(records, rows[i], i, constants) for i in INVENTORIES}
    wrong = {arm: wrong_anchor_table(output, records, cells, arm) for arm in WRONG_ARMS}
    training = training_table(output)
    incident_path = output / "execution-incident.json"
    incident = read_json(incident_path) if incident_path.is_file() else None
    pool = dap.load_pool(P101)
    eval_roster = {lv["identity"] for lv in gb.evaluation_levels()}
    fit_roster = {lv["identity"] for lv in rt.fit_levels()}
    policy_roster = {lv["identity"] for lv in dap.policy_levels()}
    hashes_ok = all(records[(arm, s)]["policy"]["sha256"] == training[f][tag(lam)][str(s)]["policy_sha256"]
                    for arm in (*POLICY_ARMS, *RESTRICTED_ARMS) for s in SEEDS
                    for f, lam in [("J", ICLR_LAMBDA) if arm in RESTRICTED_ARMS else parse_arm(arm)])
    trained_at = sorted(load_report(output, *g)["trained_at"] for g in groups())
    scored_at = sorted(r["scored_at"] for r in records.values())
    guards = {
        "G1_isolation": {"pass": bool(set(pool["levels"]) == policy_roster and not (policy_roster & (eval_roster | fit_roster))
                                      and all(set(cells[i]["mixed"]) <= eval_roster for i in INVENTORIES))},
        "G2_scoring_policy": {"pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "first_trained_at": trained_at[0],
                          "last_trained_at": trained_at[-1], "first_scored_at": scored_at[0],
                          "pass": plan["frozen_at"] <= trained_at[0] and trained_at[-1] <= scored_at[0]},
        "G4_checkpoint_binding": {"pass": bool(hashes_ok)},
        "G5_replication_101_gate_b": replication_guard(records),
        "G6_restricted_schedules": subset_guard(records),
    }
    guards_ok = all(g["pass"] for g in guards.values())
    failure = guard_failure(guards)
    grid = tables[PRIMARY]
    matched = grid["frontier"]["matched"]
    primary = {"claim": "adaptive alpha adds decision value beyond adaptive Delta at matched compute",
               "contrast": "J - J-cont (grid AUC), frontiers interpolated at the MAC budgets of J's and C-J's lambda = 1e-4 points",
               "estimates": {name: m["contrasts"]["Jcont"] for name, m in matched.items()},
               "token": disposition(matched, "Jcont") if guards_ok else STOP_TOKEN,
               "rule": plan["decision_rule"]["primary"]}
    secondary = {"claim": "the hybrid J keeps its advantage over the horizon-only adaptive C-J at matched compute",
                 "contrast": "J - C-J (grid AUC) at the same two budgets",
                 "estimates": {name: m["contrasts"]["CJ"] for name, m in matched.items()},
                 "token": disposition(matched, "CJ") if guards_ok else STOP_TOKEN}
    phases = ledger_phases(output)
    smoke = read_json(output / "smoke.json")
    all_phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
                  **{k: {"wall_seconds": v["wall_seconds"], "gpu_seconds": v["gpu_seconds"]} for k, v in phases.items()}}
    reused = sum(training["J"][tag(lam)][str(s)]["wall_seconds"] for lam in REUSED_J for s in SEEDS)
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "evaluation": gb.evaluation_accounting(branches, cells), "guards": guards,
        "primary": primary, "secondary": secondary, "per_call_macs": constants,
        "inventories": tables, "wrong_anchor": wrong, "training": training,
        "execution_integrity": {"literal_single_score_pass": incident is None,
                                "frozen_statistics_and_token_rules_unchanged": True,
                                "incident": incident},
        **({"guard_failure": failure} if failure else {}),
        "compute": {"phases": all_phases, "gpu_seconds": sum(b["gpu_seconds"] for b in all_phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in all_phases.values()), "engine_seconds": 0,
                    "reused_issue_101_training_gpu_seconds": reused,
                    "hardware": "RTX 5090 (GPUs 0 and 1, two processes per GPU)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def cell(block):
    return fmt_interval(block) if block else "n/a"


def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def mmacs(value):
    return "n/a" if value is None else f"{value/1e6:.1f}M"


def arm_label(arm):
    for family, label in (("Jascont", "J-as-cont"), ("Jcont", "J-cont"), ("CJ", "C-J"), ("J", "J")):
        if arm.startswith(family + "-") and arm.split("-", 1)[1][0].isdigit():
            return f"{label} λ={arm.split('-', 1)[1]}"
    return arm


def findings_md(plan, compute):
    lines = []
    add = lines.append
    grid = compute["inventories"][PRIMARY]
    p, s = compute["primary"], compute["secondary"]
    add("# Issue #114: horizon-only versus joint request selection at matched compute (#104 sealed #109 evaluation split)")
    add("")
    if "guard_failure" in compute:
        add(f"**Guard failure ({', '.join(compute['guard_failure']['failed_guards'])}): "
            f"{compute['guard_failure']['note']}.**")
        add("")
    add(f"Plan frozen {plan['frozen_at']} after a passed smoke, before training and scoring. Retained scores use "
        f"processes that did not train, `deterministic_scoring()` and tie-free cost. Validation: `{VALIDATION_COMMAND}`.")
    execution = compute["execution_integrity"]
    if not execution["literal_single_score_pass"]:
        add("")
        add("**Execution deviation:** accidentally duplicated fixed-arm scoring workers were terminated. Literal "
            "scored-once compliance failed; deterministic recomputation is not new independent evidence. Numerical "
            "dispositions below use the untouched frozen G1–G6 and token rules, not a post hoc relaxed rule. See "
            "`execution-incident.json`. Timings include concurrent-worker contention and are not isolated-device latency.")
    add("")
    add(f"**Disposition — {p['claim']}: {p['token']}** ({p['contrast']}; δ = {DELTA}).")
    add("")
    add("| budget (MACs / candidate) | J − J-cont | token | J − C-J | token |")
    add("| --- | --- | --- | --- | --- |")
    for name, m in grid["frontier"]["matched"].items():
        jc, cj = m["contrasts"]["Jcont"], m["contrasts"]["CJ"]
        add(f"| {name} ({mmacs(m['budget_macs_per_candidate'])}) | {cell(jc.get('estimate'))} | {jc['token']} | "
            f"{cell(cj.get('estimate'))} | {cj['token']} |")
    add(f"| control: no-model action prior AUC | {cell(grid['rankers']['prior']['auc'])} | reference | — | — |")
    add(f"| control: wrong-anchor J AUC (grid) | {cell(compute['wrong_anchor'][J_ICLR]['wrong'])} | reference | — | — |")
    add("")
    add(f"Secondary — {s['claim']}: **{s['token']}**.")
    add("")
    add("## Per-inventory AUC (tie-free; level-clustered, DESCRIPTIVE)")
    add("")
    arms = (J_ICLR, JCONT_ICLR, *RESTRICTED_ARMS, CJ_ICLR, ascont_arm(ICLR_LAMBDA), *FIXED_ARMS, *C_FIXED_ARMS, "prior")
    add("| arm | " + " | ".join(INVENTORIES) + " |")
    add("| --- | " + " | ".join("---" for _ in INVENTORIES) + " |")
    for arm in arms:
        add(f"| {arm_label(arm)} | " + " | ".join(cell(compute["inventories"][i]["rankers"][arm]["auc"]) for i in INVENTORIES)
            + " |")
    w = compute["wrong_anchor"][J_ICLR]
    add(f"| wrong-anchor J λ=1e-04 (grid only) | {cell(w['wrong'])} | — | — | — |")
    add("")
    add("## Contrasts (paired by level and seed; δ rule)")
    add("")
    add("| contrast | " + " | ".join(INVENTORIES) + " |")
    add("| --- | " + " | ".join("---" for _ in INVENTORIES) + " |")
    for name in CONTRASTS:
        add(f"| {compute['inventories'][PRIMARY]['contrasts'][name]['contrast']} | " + " | ".join(
            f"{cell(compute['inventories'][i]['contrasts'][name]['estimate'])} {compute['inventories'][i]['contrasts'][name]['token']}"
            for i in INVENTORIES) + " |")
    add("| control: no-model action prior AUC | " + " | ".join(
        cell(compute["inventories"][i]["rankers"]["prior"]["auc"]) for i in INVENTORIES) + " |")
    add(f"| control: wrong-anchor J AUC | {cell(compute['wrong_anchor'][J_ICLR]['wrong'])} | grid only | grid only | grid only |")
    add("")
    add("## λ-swept frontiers (grid)")
    add("")
    add("| policy | λ | AUC | top-3 | MACs / candidate | calls / candidate | GPU s / state | tied cells | tied pairs |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for family in FAMILIES:
        for pt in grid["frontier"]["points"][family]:
            add(f"| {FAMILY_NAMES[family].split(' (')[0]} | {tag(pt['lambda'])} | {cell(pt['auc'])} | {cell(pt['top3'])} | "
                f"{mmacs(pt['mean_macs_per_candidate'])} | {num(pt['mean_transitions_per_candidate'], 1)} | "
                f"{num(pt['gpu_seconds_per_state'], 3)} | {pt['tied_cells']} | {pt['tied_pairs']}/{pt['pairs']} |")
    pr = grid["rankers"]["prior"]
    add(f"| no-model action prior | — | {cell(pr['auc'])} | {cell(pr['top3'])} | 0 | 0 | 0 | {pr['tied_cells']} | "
        f"{pr['tied_pairs']}/{pr['pairs']} |")
    for control_arm in WRONG_ARMS:
        control = compute["wrong_anchor"][control_arm]
        add(f"| wrong-anchor {arm_label(control_arm)} | 1e-04 | {cell(control['wrong'])} | {cell(control['top3'])} | "
            f"{mmacs(control['mean_macs_per_candidate'])} | — | {num(control['gpu_seconds_per_state'], 3)} | — | — |")
    add("")
    add(f"Top-3 chance on grid: {num(grid['rankers'][J_ICLR]['chance_top3'], 3)}. Frontier figure: `{FIGURE}`; "
        f"machine-readable points: `{FRONTIER_FILE}`.")
    add("")
    add("Matched-compute interpolation (grid AUC at each budget; † = budget above the frontier's largest MACs, the "
        "largest-MAC point is used):")
    add("")
    for name, m in grid["frontier"]["matched"].items():
        add(f"- {name} ({mmacs(m['budget_macs_per_candidate'])}): " + "; ".join(
            f"{FAMILY_NAMES[f].split(' (')[0]}{'†' if m['budget_above_frontier'][f] else ''} {cell(m['auc'][f])} "
            f"({' + '.join(f'{w:.3f}·{arm_label(a)}' for w, a in m['pieces'][f]) if m['pieces'][f] else 'below frontier'})"
            for f in FAMILIES) + ".")
    add("")
    add("An endpoint clamp is an **at-most-budget contrast, not exact equal expenditure**. Both declared budgets "
        "can reuse the same endpoint pair; equal estimates at the two budgets are not independent replications. "
        "The frozen interpolation passes through every lambda point without outcome-based pruning; it is not "
        "the best-AUC-at-or-below-budget envelope and does not establish Pareto dominance. The numerical token "
        "must be read with these limitations and the execution-integrity disclosure above.")
    add("")
    add("## Compute decomposition (grid, MACs per candidate and state; first − second)")
    add("")
    add("| λ | contrast | Δ MACs | (a) calls (horizon) | (b) per-call α | (b') trunk | (c) policy |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for lam in LAMBDAS:
        for name, d in grid["decomposition"][tag(lam)].items():
            add(f"| {tag(lam)} | {arm_label(d['first'])} − {arm_label(d['second'])} | {mmacs(d['difference'])} | "
                f"{mmacs(d['a_calls_horizon'])} | {mmacs(d['b_alpha'])} | {mmacs(d['b_trunk'])} | {mmacs(d['c_policy'])} |")
    add(f"| control | no-model action prior: AUC {cell(pr['auc'])}, 0 model MACs | — | — | — | — | — |")
    add(f"| control | wrong-anchor J: AUC {cell(compute['wrong_anchor'][J_ICLR]['wrong'])} | — | — | — | — | — |")
    add("")
    add("Counterfactual J-as-cont (J's schedule with every α replaced by continuous, same calls, no policy calls), grid:")
    add("")
    add("| λ | J AUC | J-as-cont AUC | J − J-as-cont | J MACs | J-as-cont MACs |")
    add("| --- | --- | --- | --- | --- | --- |")
    for lam in LAMBDAS:
        j, a = grid["rankers"][family_arm("J", lam)], grid["rankers"][ascont_arm(lam)]
        c = grid["J_minus_ascont"][tag(lam)]
        add(f"| {tag(lam)} | {cell(j['auc'])} | {cell(a['auc'])} | {cell(c['estimate'])} {c['token']} | "
            f"{mmacs(j['mean_macs_per_candidate'])} | {mmacs(a['mean_macs_per_candidate'])} |")
    add(f"| control | no-model action prior {cell(pr['auc'])} | — | — | 0 | — |")
    add(f"| control | wrong-anchor J {cell(compute['wrong_anchor'][J_ICLR]['wrong'])} | — | — | — | — |")
    add("")
    add("## Schedules (grid, pooled over seeds)")
    add("")
    add("| arm | modal request | non-modal step share | cells with candidate-dependent schedules |")
    add("| --- | --- | --- | --- |")
    for arm in (J_ICLR, JCONT_ICLR, CJ_ICLR, *RESTRICTED_ARMS):
        sd = grid["schedules"][arm]
        add(f"| {arm_label(arm)} | {sd['modal_pair']} | {num(sd['non_modal_step_share'], 3)} | "
            f"{sd['cells_with_differing_schedules']}/{sd['cells']} |")
    add(f"| control: no-model action prior AUC {cell(pr['auc'])} | no predictor | — | — |")
    add(f"| control: wrong-anchor J AUC {cell(compute['wrong_anchor'][J_ICLR]['wrong'])} | wrong-anchor diagnostic | — | — |")
    add("")
    add("## Controls")
    add("")
    add("| policy (λ = 1e-4) | right anchor | wrong anchors (480 fit) | right − wrong | token | stability |")
    add("| --- | --- | --- | --- | --- | --- |")
    for arm in WRONG_ARMS:
        w = compute["wrong_anchor"][arm]
        add(f"| {arm_label(arm)} | {cell(w['right'])} | {cell(w['wrong'])} | {cell(w['right_minus_wrong'])} | "
            f"{w['token']} | {cell(w['stability'])} |")
    add(f"| no-model action prior | {cell(pr['auc'])} | anchor independent | — | reference | — |")
    add("")
    add(f"No-model action prior (grid): {cell(pr['auc'])}.")
    add("")
    add("## Guards")
    add("")
    for name, guard in compute["guards"].items():
        add(f"- {name}: {guard['pass']}")
    add("")
    c = compute["compute"]
    add(f"Compute: {c['gpu_seconds']/3600:.2f} GPU h, {c['wall_seconds']/3600:.2f} process wall h ({c['hardware']}); "
        f"reused #101 J training {c['reused_issue_101_training_gpu_seconds']/3600:.2f} GPU h; 0 engine seconds.")
    if not execution["literal_single_score_pass"]:
        add("Discarded-run compute is an explicitly labelled conservative allocated-process upper bound in "
            "`ledger/operational-discarded-upper-bound.json`, not measured CUDA active time.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    grid = compute["inventories"][PRIMARY]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND,
            "primary": {k: compute["primary"][k] for k in ("claim", "contrast", "token")},
            "primary_estimates": {n: e.get("estimate") for n, e in compute["primary"]["estimates"].items()},
            "secondary": {k: compute["secondary"][k] for k in ("claim", "contrast", "token")},
            "secondary_estimates": {n: e.get("estimate") for n, e in compute["secondary"]["estimates"].items()},
            "budgets": grid["frontier"]["budgets"],
            "frontier_interpretation": ("Endpoint clamps compare policies at most at the budget, not exact equal "
                                        "expenditure. Reused endpoint pairs are not independent replications. "
                                        "The all-lambda interpolant is not a best-AUC-under-budget envelope; "
                                        "its frozen token does not establish Pareto dominance."),
            "contrasts": {i: {n: {"estimate": compute["inventories"][i]["contrasts"][n]["estimate"],
                                  "token": compute["inventories"][i]["contrasts"][n]["token"]} for n in CONTRASTS}
                          for i in INVENTORIES},
            "auc_grid": {arm: grid["rankers"][arm]["auc"]["mean"] if grid["rankers"][arm]["auc"] else None
                         for arm in (*ARMS, "prior")},
            "wrong_anchor": {a: {k: compute["wrong_anchor"][a][k] for k in ("right_minus_wrong", "token")}
                             for a in WRONG_ARMS},
            "guards": {k: v["pass"] for k, v in compute["guards"].items()},
            **({"guard_failure": compute["guard_failure"]} if "guard_failure" in compute else {}),
            "execution_integrity": compute["execution_integrity"],
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")},
            "selector_module": SELECTORS, "claim_boundary": plan["claim_boundary"]}


def frontier_export(plan, compute):
    """Machine-readable frontier points for #103 (every inventory; J, J-cont, C-J; fixed references)."""
    out = {"schema": SCHEMA_FRONTIER, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
           "cost": PRIMARY_COST, "macs": "linear MACs per candidate rollout to step 225 (predictor + policy calls)",
           "wall": "CUDA-synchronized GPU seconds per state (all candidates of one cell), RTX 5090",
           "policies": {}, "fixed": {}, "restricted": {}, "prior": {}}
    for family in FAMILIES:
        out["policies"][family] = {"name": FAMILY_NAMES[family], "points": []}
        for lam in LAMBDAS:
            out["policies"][family]["points"].append({
                "lambda": lam, "arm": family_arm(family, lam),
                "checkpoints": {str(s): {"path": str(policy_path(OUTPUT, family, lam, s).relative_to(ROOT)),
                                         "sha256": compute["training"][family][tag(lam)][str(s)]["policy_sha256"]}
                                for s in SEEDS},
                "inventories": {i: {k: compute["inventories"][i]["rankers"][family_arm(family, lam)][k]
                                    for k in ("auc", "top3", "tied_cells", "tied_pairs", "pairs", "mean_macs_per_candidate",
                                              "mean_transitions_per_candidate", "mean_policy_calls_per_candidate",
                                              "gpu_seconds_per_state")} for i in INVENTORIES}})
    for arm in (*FIXED_ARMS, *C_FIXED_ARMS, *RESTRICTED_ARMS):
        out["restricted" if arm in RESTRICTED_ARMS else "fixed"][arm] = {
            i: {k: compute["inventories"][i]["rankers"][arm][k] for k in ("auc", "top3", "mean_macs_per_candidate",
                                                                          "gpu_seconds_per_state")} for i in INVENTORIES}
    out["prior"] = {i: {k: compute["inventories"][i]["rankers"]["prior"][k] for k in ("auc", "top3")} for i in INVENTORIES}
    out["matched"] = {i: compute["inventories"][i]["frontier"]["matched"] for i in INVENTORIES}
    out["wrong_anchor_controls"] = compute["wrong_anchor"]
    out["execution_integrity"] = compute["execution_integrity"]
    return out


def figure_bytes(compute):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    grid = compute["inventories"][PRIMARY]
    styles = {"J": ("tab:blue", "o"), "Jcont": ("tab:orange", "s"), "CJ": ("tab:green", "^")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=100)
    for col, (xkey, xlabel) in enumerate((("mean_macs_per_candidate", "linear MACs per candidate and state"),
                                          ("gpu_seconds_per_state", "GPU seconds per state (RTX 5090)"))):
        for row, (ykey, ylabel) in enumerate((("auc", "tie-free AUC (grid)"), ("top3", "top-3 hit rate (grid)"))):
            ax = axes[row][col]
            for family in FAMILIES:
                pts = sorted(grid["frontier"]["points"][family], key=lambda p: p[xkey])
                color, marker = styles[family]
                ax.plot([p[xkey] for p in pts], [p[ykey]["mean"] for p in pts], marker=marker, color=color,
                        label=FAMILY_NAMES[family].split(" (")[0])
                for p in pts:
                    ax.annotate(tag(p["lambda"]), (p[xkey], p[ykey]["mean"]), fontsize=6, color=color)
            for arm in (*FIXED_ARMS, *C_FIXED_ARMS):
                r = grid["rankers"][arm]
                ax.scatter([r[xkey]], [r[ykey]["mean"]], color="grey" if arm in FIXED_ARMS else "black",
                           marker="x" if arm in FIXED_ARMS else "D", s=14)
            r = grid["rankers"]["T-continuous"]
            ax.scatter([r[xkey]], [r[ykey]["mean"]], color="tab:red", marker="*", s=50, label="T-cont (J restricted)")
            ax.axhline(grid["rankers"]["prior"][ykey]["mean"], color="grey", linestyle="--", linewidth=0.8,
                       label="no-model action prior")
            if ykey == "auc":
                ax.axhline(compute["wrong_anchor"][J_ICLR]["wrong"]["mean"], color="tab:purple", linestyle="--",
                           linewidth=0.8, label="wrong-anchor J (grid)")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            for name, m in grid["frontier"]["matched"].items():
                if col == 0:
                    ax.axvline(m["budget_macs_per_candidate"], color="purple", linestyle=":", linewidth=0.8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle("#114 λ-swept frontiers on the #109 evaluation split (grid; x = fixed hybrid, ◆ = fixed C*)")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    plt.close(fig)
    return buffer.getvalue()


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), FRONTIER_FILE: json_text(frontier_export(plan, compute)),
            FIGURE: figure_bytes(compute)}


def as_bytes(value):
    return value if isinstance(value, bytes) else value.encode()


# ---------------------------------------------------------------------------
# smoke (pre-freeze; #101 artefacts, fit-split cells, policy-split training data; no evaluation score)
# ---------------------------------------------------------------------------

def selector_checks(model, policy):
    """On three fit grid cells with the #101 J (lambda 1e-4) policy: restricted-to-all-nine equals the
    joint rollout of request_policy; T-alpha / D-Delta equal t96 temporal / description; replay of J's
    schedules equals J; J-cont's masked policy only requests R_cont."""
    fit_cells = gb.mixed_cells(gb.inventory_entries(rt.fit_levels(), rt.complete_branches(), rt.fit_verdicts(), PRIMARY))
    stored = gb.fit_anchors()
    equal = True
    for member, entry in sorted(fit_cells.items())[:3]:
        actions = m99.action_batch(entry["inventory"])
        z0 = stored[member].to(DEVICE)
        a, ta = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                     record_schedules=True)
        b, tb = rs.rollout_endpoints(model, policy, z0, actions, rs.restricted_spec(PAIRS, PAIRS),
                                     record_schedules=True)
        equal &= bool(torch.equal(a, b)) and ta["schedules"] == tb["schedules"] and ta["linear_macs"] == tb["linear_macs"]
        for k in range(3):
            a, ta = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "temporal", "alpha_index": k},
                                         select=t96.select_pairs, record_schedules=True)
            b, tb = rs.rollout_endpoints(model, policy, z0, actions, rs.temporal_spec(PAIRS, t96.ALPHAS[k]),
                                         record_schedules=True)
            equal &= bool(torch.equal(a, b)) and ta["schedules"] == tb["schedules"]
            a, ta = rp.rollout_endpoints(model, policy, z0, actions, {"kind": "description", "delta_index": k},
                                         select=t96.select_pairs, record_schedules=True)
            b, tb = rs.rollout_endpoints(model, policy, z0, actions, rs.description_spec(PAIRS, t96.DELTAS[k]),
                                         record_schedules=True)
            equal &= bool(torch.equal(a, b)) and ta["schedules"] == tb["schedules"]
        j, tj = rs.rollout_endpoints(model, policy, z0, actions, {"kind": "joint"}, select=t96.select_pairs,
                                     record_schedules=True)
        r, tr = rs.rollout_endpoints(model, None, z0, actions, rs.replay_spec(tj["schedules"]))
        equal &= bool(torch.equal(j, r)) and tr["transition_calls"] == tj["transition_calls"]
        _, tc = rs.rollout_endpoints(model, None, z0, actions, rs.replay_spec([as_continuous(s) for s in tj["schedules"]]))
        equal &= tc["transition_calls"] == tj["transition_calls"]
    masked = rs.new_masked_policy(PAIRS, R_CONT, SEEDS[0], device=DEVICE).eval()
    member, entry = sorted(fit_cells.items())[0]
    _, tm = rs.rollout_endpoints(model, masked, stored[member].to(DEVICE), m99.action_batch(entry["inventory"]),
                                 {"kind": "joint"}, select=t96.select_pairs, record_schedules=True)
    only_cont = all(set(s) <= set(R_CONT) for s in tm["schedules"])
    return {"control": "selectors_equal_request_policy_and_t96_kinds", "cells": 3, "ok": bool(equal and only_cont)}


def training_checks(model):
    """The training path reproduces #101's J (lambda 1e-4, seed 20260908): round-0 labels of the first
    units equal #101's stored labels; both distillation rounds on #101's stored labels reproduce #101's
    round checkpoints bit for bit. restricted_window_rows with all nine requests equals window_rows."""
    seed, lam = SEEDS[0], ICLR_LAMBDA
    pool = dap.load_pool(P101)
    units = dap.window_units(pool)
    root = dap.group_root(P101, "pe", lam, seed)
    stored = [torch.load(root / f"labels-{r}.pt", map_location="cpu", weights_only=True)["units"] for r in (0, 1)]
    policy = rp.new_policy(PAIRS, seed, device=DEVICE)
    labels_equal = all(all(torch.equal(a[k], b[k]) for k in a)
                       for a, b in zip((family_rows("J", model, policy, u, False, lam) for u in units[:SMOKE_UNITS]),
                                       stored[0][:SMOKE_UNITS]))
    all_nine = all(all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(
        (rs.restricted_window_rows(model, policy, u, range(9), aggregated=True, compute_weight=lam, device=DEVICE)
         for u in units[:SMOKE_UNITS]),
        (rp.window_rows(model, policy, u, aggregated=True, compute_weight=lam, device=DEVICE) for u in units[:SMOKE_UNITS])))
    distil_equal = True
    t0 = time.monotonic()
    for r in (0, 1):
        rp.distil_round(policy, rp.aggregate_rounds(stored[:r+1]), r, seed=seed, device=DEVICE)
        reference = torch.load(root / f"round-{r}.pt", map_location="cpu", weights_only=True)["model"]
        distil_equal &= all(torch.equal(v.cpu(), reference[k]) for k, v in policy.state_dict().items())
    distil_seconds = time.monotonic()-t0
    return {"control": "training_path_reproduces_issue101_J", "units": SMOKE_UNITS, "round0_labels_equal": bool(labels_equal),
            "restricted_all_nine_equals_window_rows": bool(all_nine), "distil_rounds_equal": bool(distil_equal),
            "distil_seconds_two_rounds": distil_seconds, "ok": bool(labels_equal and all_nine and distil_equal)}


def continuous_policy_check():
    from world_model.training.matched_dynamics import MatchedController
    torch.manual_seed(SEEDS[0]+rp.INIT_OFFSET)
    matched = MatchedController(True)
    mine = rp.new_policy(CONTINUOUS_PAIRS, SEEDS[0], width=CJ_WIDTH, device="cpu")
    same = all(torch.equal(a, b) for a, b in zip(matched.state_dict().values(), mine.state_dict().values()))
    z, a = torch.randn(8, 236), torch.rand(8, 5)
    remaining = torch.tensor([1, 3, 5, 10, 15, 30, 60, 225])
    return {"control": "C-J policy equals MatchedController(True)", "ok": bool(same and torch.equal(matched(z, a, remaining),
                                                                                                    mine(z, a, remaining)))}


def label_distributions(unit_model, cstar_model):
    """Round-0 label request mix per lambda on the first policy units (training data): coverage of the
    MAC range by the lambda grid; plus the per-unit labelling time."""
    pool = dap.load_pool(P101)
    units = dap.window_units(pool)[:SMOKE_UNITS]
    out = {}
    t0 = time.monotonic()
    for family, model in (("J", unit_model), ("Jcont", unit_model), ("CJ", cstar_model)):
        policy = new_family_policy(family, SEEDS[0])
        out[family] = {}
        for lam in (*LAMBDAS, *PROBE_LAMBDAS):
            rows = [family_rows(family, model, policy, u, False, lam) for u in units]
            labels = torch.cat([r["labels"] for r in rows])
            early = torch.cat([r["labels"][r["remaining"] >= 55] for r in rows])  # window positions 0..5
            out[family][tag(lam)] = {"all_positions": torch.bincount(labels, minlength=len(policy.pairs)).tolist(),
                                     "positions_0_to_5": torch.bincount(early, minlength=len(policy.pairs)).tolist()}
    seconds = (time.monotonic()-t0)/(3*(len(LAMBDAS)+len(PROBE_LAMBDAS))*len(units))
    t0 = time.monotonic()
    policy = new_family_policy("Jcont", SEEDS[0])
    family_rows("Jcont", unit_model, policy, units[0], True, ICLR_LAMBDA)
    dagger = time.monotonic()-t0
    return {"per_family_lambda": out, "units": len(units), "round0_seconds_per_unit": seconds,
            "round1_seconds_per_unit": dagger}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    checks = []
    with harness.deterministic_scoring():
        unit = dap.load_predictor(SEEDS[0], "unit")
        cstar = dap.load_predictor(SEEDS[0], "c_star")
        j = dap.load_policy(P101, "pe", ICLR_LAMBDA, SEEDS[0])
        checks.append(selector_checks(unit, j))
        checks.append(training_checks(unit))
        checks.append(continuous_policy_check())
        distributions = label_distributions(unit, cstar)
    accounting = dap.pool_accounting()
    checks.append({"control": "policy_pool_accounting", **accounting,
                   "ok": bool(accounting["levels"] == 120 and accounting["complete"] == 3480
                              and accounting["disjoint_from_fit_and_evaluation"])})
    store = gb.load_anchors(GATE_B)
    gate = read_json(GATE_B / "summary.json")
    checks.append({"control": "gate_b_inputs_bound", "gate_b": gate["gate_b"], "anchors": len(store["anchors"]),
                   "ok": bool(gate["gate_b"] == "pass" and store["scoring_policy"] == harness.SCORING_POLICY)})
    reused = [dap.report_path(P101, "pe", lam, s) for lam in REUSED_J for s in SEEDS]
    checks.append({"control": "issue101_J_policies_present", "reports": len(reused),
                   "ok": all(p.is_file() for p in reused)})
    ok = all(c["ok"] for c in checks)
    wall = time.monotonic()-began
    per_group = read_json(dap.report_path(P101, "pe", ICLR_LAMBDA, SEEDS[0]))["wall_seconds"]
    projection = {"training_groups": len(groups()), "issue101_pe_group_seconds": per_group,
                  "training_gpu_seconds": len(groups())*per_group,
                  "scoring_gpu_seconds_estimate": len(ARMS)*len(SEEDS)*60.0 + len(WRONG_ARMS)*len(SEEDS)*15.0}
    projection["total_gpu_seconds"] = projection["training_gpu_seconds"]+projection["scoring_gpu_seconds_estimate"]
    projection["elapsed_hours_on_2_gpus"] = projection["total_gpu_seconds"]/2/3600
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "label_distributions": distributions, "pool_accounting": accounting, "projection": projection,
                "per_call_macs": compute_constants(), "wall_seconds": wall,
                "reading": ("#101 artefacts, fit-split cells and anchors, policy-split windows (training data); no "
                            "evaluation-split model output"), "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'} ({wall:.0f}s)")
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
        "issue_112_retrain_handoff": gb.HANDOFF, "issue_112_fit_verdicts": gb.FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(rt.OUTPUT),
        "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
        "gate_b_anchors": gb.anchors_path(GATE_B),
        **{f"gate_b_record_{n}_{s}": gb.record_path(GATE_B, n, s) for n in gb.UNITS for s in SEEDS},
        "issue_101_plan": P101 / "plan.json", "issue_101_compute": P101 / "compute.json",
        "issue_101_pool": dap.pool_path(P101),
        **{f"issue_101_J_{tag(lam)}_{s}_{what}": path for lam in REUSED_J for s in SEEDS
           for what, path in (("policy", dap.policy_path(P101, "pe", lam, s)),
                              ("report", dap.report_path(P101, "pe", lam, s)))},
        **{f"issue_101_record_{arm}_{s}": dap.record_path(P101, arm, s)
           for arm in (*FIXED_ARMS, "C-F-15", *(dap.policy_arm("pe", lam) for lam in REUSED_J)) for s in SEEDS},
        "issue_101_runner": ROOT / dap.RUNNER, "gate_b_runner": ROOT / gb.RUNNER,
        "issue_112_retrain_runner": ROOT / rt.RUNNER, "issue_112_v1_runner": ROOT / v1.RUNNER,
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_campaign_branches": dap.CAMPAIGN / "branches.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "request_policy_module": ROOT / MODULE, "request_selectors_module": ROOT / SELECTORS,
        "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_{n}_{c['seed']}": gb.member_path(n, c["seed"]) for n in gb.UNITS for c in gb.checkpoints(n)})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "primary": (f"claim 'adaptive alpha adds decision value beyond adaptive Delta at matched compute': J - J-cont "
                    "paired grid AUC per (level, seed), each frontier interpolated piecewise linearly in log10 mean "
                    "MACs per candidate (every lambda point, sorted by MACs, no outcome-based pruning, no "
                    "extrapolation) at two budgets: the MACs of J's and of C-J's lambda = 1e-4 points. Per budget "
                    f"the delta rule: supported iff mean >= {DELTA} and lower > 0; not_supported_by_this_experiment "
                    f"iff upper < {DELTA}; else {STOP_TOKEN}. A budget above a frontier's largest MACs uses that "
                    "frontier's largest-MAC point (the policy cannot spend more; flagged); a budget below its smallest "
                    f"MACs gives {STOP_TOKEN} (no extrapolation). Disposition: supported iff supported at both budgets; "
                    f"not_supported_by_this_experiment iff not supported at either; else {STOP_TOKEN}; any failed "
                    f"guard -> {STOP_TOKEN}"),
        "secondary": ("J - C-J at the same budgets with the same rule (does the hybrid keep its advantage over the "
                      "horizon-only adaptive baseline at matched compute?); at lambda = 1e-4 (unmatched): J - J-cont, "
                      "J - T-cont, J - C-J, J - J-as-cont, J - T-micro/macro, J - D-1/5/15, J-cont - T-cont and each "
                      "policy - prior, every inventory, same token rule; offset, angle, power DESCRIPTIVE"),
    }


def frozen_plan(output, frozen_at, smoke_evidence):
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 114,
        "related_issues": [74, 96, 100, 101, 103, 104, 105, 108, 109, 111, 112],
        "role": "horizon-only versus joint request selection at matched compute; one score on the sealed #109 evaluation split",
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "selector_module": {"path": SELECTORS, "sha256_at_freeze": sha256_of(ROOT / SELECTORS)},
        "engine_seconds": 0,
        "question": ("does choosing the description alpha add ranking value beyond choosing the horizon Delta when the "
                     "horizon-only policies get the same compute?"),
        "predictors": {"hybrid": "LC hybrid fold 'all' Gate-B unit checkpoints (J, J-cont, restricted arms, J-as-cont)",
                       "continuous": "LC continuous fold 'all' Gate-B C* checkpoints (C-J, C-F-*)",
                       "seeds": list(SEEDS), "pairing": "policy seed s uses predictor seed s"},
        "pool": {"source": "the #101 pool (issue-101 data/pool.pt): #109 policy split, 120 levels, 3480 shots, first "
                           "60-frame window per shot; unit = policy level", "accounting": smoke_evidence["pool_accounting"]},
        "policies": {
            "J": {"name": FAMILY_NAMES["J"], "request_set": list(t96.PAIR_NAMES),
                  "reused_from_issue_101": [tag(lam) for lam in REUSED_J],
                  "trained_here": [tag(lam) for lam in LAMBDAS if lam not in REUSED_J],
                  "labels": "#77 prediction-error DP (request_policy.window_rows) with lambda as the compute weight"},
            "Jcont": {"name": FAMILY_NAMES["Jcont"], "request_set": [t96.PAIR_NAMES[i] for i in R_CONT],
                      "policy": "request_selectors.MaskedRequestPolicy: the 242 -> 128 -> 9 policy, logits outside "
                                "R_cont = -inf (same parameters and initialization as J)",
                      "labels": "request_selectors.restricted_window_rows: the same DP and DAgger contexts over R_cont "
                                "only; the work term counts the full policy"},
            "CJ": {"name": FAMILY_NAMES["CJ"], "request_set": [f"{d}-continuous" for d in t96.DELTAS],
                   "policy": f"242 -> {CJ_WIDTH} -> 3 (= MatchedController(True))",
                   "labels": "#77 prediction-error DP on the C* predictor"},
            "recipe": {**rp.DISTIL_RECIPE, "init": "torch.manual_seed(seed + 7700)",
                       "sampler": "CPU Generator(seed + 7700 + round)", "unit": "policy level",
                       "remaining_feature": "clamp60"},
            "lambdas": [tag(lam) for lam in LAMBDAS],
            "lambda_grid_reason": ("8 values from 1e-5 to 1e-1 (>= 5 required) including the ICLR 1e-4 and the four #101 "
                                   "J values; the ICLR operating points (#74: hybrid 118.7M, continuous_adaptive 356.0M "
                                   "MACs per candidate and state, both at lambda 1e-4) are reproduced by recipe at 1e-4. "
                                   "Smoke label mixes on policy units (training data; probes 0, 1e-6, 0.3, 1, 10): below "
                                   "1e-5 the labels equal 1e-5's (the DP is error-dominated) and above 1e-1 they saturate "
                                   "(Delta = 15 wherever the window allows), so the grid spans each policy's reachable "
                                   "MAC range; resolution is densest where J's labels change (1e-4 .. 1e-1)"),
        },
        "arms": {"fixed": list(FIXED_ARMS), "continuous_fixed": list(C_FIXED_ARMS),
                 "policies": list(POLICY_ARMS), "restricted": {arm: [t96.PAIR_NAMES[i] for i in restricted_spec(arm)["allowed"]]
                                                               for arm in RESTRICTED_ARMS},
                 "restricted_policy": f"#101 J (pe, lambda {tag(ICLR_LAMBDA)}); argmax over the allowed subset of the "
                                      "same logits (request_selectors.restricted_spec), no retraining",
                 "as_continuous": ("per lambda: J's recorded per-candidate schedule with every (Delta, alpha) replaced "
                                   "by (Delta, continuous), replayed on the hybrid trunk without policy calls; a candidate "
                                   "whose J endpoint is a typed failure keeps the failure"),
                 "controls": "no-model action prior (Gate-B rule, fit-verdicts.json) in every table; wrong-anchor control "
                             f"(480 fit anchors x grid candidates) of {', '.join(WRONG_ARMS)}"},
        "metrics": {"auc": "tie-free cost, m99.view; per (level, seed); member-clustered bootstrap 10000 draws PCG64 7201",
                    "top3": "hit iff a success is among the three lowest-cost admissible candidates (cost, ordinal)",
                    "ties": "all-tied cells and tied success-failure pairs per arm",
                    "macs": "linear MACs per candidate rollout (predictor calls x per-call MACs + policy calls x policy MACs)",
                    "wall": "CUDA-synchronized GPU seconds per state (per cell rollout of all candidates)",
                    "decomposition": ("first - second = (a) (n1 - n2) c2 [calls, horizon] + (b) n1 (c1 - c2) [per-call; "
                                      "for C-J split into alpha n1 (c1 - c_cont,hybrid) and trunk n1 (c_cont,hybrid - c2)] "
                                      "+ (c) p1 m1 - p2 m2 [policy]; n calls, c mean predictor MACs per call, p policy calls, "
                                      "m policy MACs per call")},
        "per_call_macs": smoke_evidence["per_call_macs"],
        "evaluation": {"split": "the #109 evaluation split as in Gate B (stream issue109-evaluation, mixed cells)",
                       "anchors": "the Gate-B anchor store", "primary_inventory": PRIMARY,
                       "descriptive": list(gb.DESCRIPTIVE_INVENTORIES)},
        "decision_rule": frozen_rules(),
        "guards": {"G1_isolation": "pool = policy roster, disjoint from fit and evaluation rosters",
                   "G2_scoring_policy": "every record carries the #111 SCORING_POLICY",
                   "G3_chronology": "frozen_at <= first training <= last training <= first score",
                   "G4_checkpoint_binding": "every scored policy's sha256 equals its training report (#101 for reused J)",
                   "G5_replication_101_gate_b": "fixed and reused-J rows (and schedules) equal the #101 records; F-15-macro "
                                                "and C-F-15 equal the Gate-B unit and C* records",
                   "G6_restricted_schedules": "restricted and J-cont schedules use only their allowed requests",
                   "failure": f"a failed guard turns the dispositions into {STOP_TOKEN}"},
        "prohibited": ["reading evaluation captures for training", "re-scoring after publication",
                       "changing a lambda, rule or arm after the freeze", "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds_summed_over_processes": WALL_CAP_SECONDS,
                 "projection": smoke_evidence["projection"],
                 "scope_rule": ("if the projected elapsed wall exceeds 24 h on 2 GPUs, cut to the acceptance minimum "
                                "(5 lambdas per family) and record why on #114")},
        "disclosure": ("Known before the freeze: Gate B (unit 0.7444, C* 0.7834, prior 0.7203 on grid), the #101 results "
                       "(J at lambdas 1e-5..1e-2 on this split: grid AUC 0.704/0.706/0.734/0.713 at 181M/184M/34M/50M "
                       "MACs; F-15-continuous 0.762, F-1-continuous 0.677) and every #96/#100/#112 result. No J-cont "
                       "or C-J policy and no restricted-argmax arm has been scored on an evaluation capture. Hardware: "
                       "RTX 5090 instead of the #85 RTX 3090."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split for policies trained on the #109 policy split; "
                           "grid ranking primary, other inventories descriptive; tie-free cost; DESCRIPTIVE intervals; "
                           "MACs are linear MACs, wall times are RTX 5090 GPU seconds; no closed-loop claim"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-114 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if not amendments.matches(entry["artifact"], entry["sha256"], sha256_of(ROOT / entry["artifact"])):
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("records", "models"):
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
    for name, content in rendered(plan, compute).items():
        (output / name).write_bytes(as_bytes(content))
    log(f"published: primary {compute['primary']['token']}, secondary {compute['secondary']['token']}")
    return 0


def spot_checks(output):
    """Re-score J-cont (lambda 1e-4), T-continuous and J-as-cont (lambda 1e-4) of the first seed on every
    grid cell in this process (exact rows and schedules)."""
    seed = SEEDS[0]
    _, cells = gb.evaluation_set()
    anchors = gb.load_anchors(GATE_B)["anchors"]
    costs = dap.endpoint_costs()
    same = True
    with harness.deterministic_scoring():
        unit = dap.load_predictor(seed, "unit")
        jcont = load_policy(output, "Jcont", ICLR_LAMBDA, seed)
        j = load_policy(output, "J", ICLR_LAMBDA, seed)
        records = {arm: read_record(output, arm, seed) for arm in (JCONT_ICLR, "T-continuous", J_ICLR, ascont_arm(ICLR_LAMBDA))}
        for member, entry in sorted(cells[PRIMARY]["mixed"].items()):
            for arm, policy, spec in ((JCONT_ICLR, jcont, {"kind": "joint"}), ("T-continuous", j, restricted_spec("T-continuous"))):
                fresh = block(unit, policy, spec, anchors[member], entry["inventory"], costs, True)
                stored = records[arm]["inventories"][PRIMARY][member]
                same &= fresh["rows"] == stored["rows"] and fresh["schedules"] == stored["schedules"]
            fresh = replay_block(unit, records[J_ICLR]["inventories"][PRIMARY][member], anchors[member],
                                 entry["inventory"], costs)
            same &= fresh["rows"] == records[ascont_arm(ICLR_LAMBDA)]["inventories"][PRIMARY][member]["rows"]
    return {"arms": [JCONT_ICLR, "T-continuous", ascont_arm(ICLR_LAMBDA)], "seed": seed, "pass": bool(same)}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, content in rendered(plan, compute).items():
        if (output / name).read_bytes() != as_bytes(content):
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic()-began:.0f}s): tables, frontier export and figure re-derived and "
        f"byte-compared; {', '.join(spot['arms'])} seed {spot['seed']} re-scored exactly; primary {compute['primary']['token']}")
    return 0


def dry_run(output):
    log(f"output {Path(output)}; pool {dap.pool_accounting()}")
    log(f"training groups {len(groups())}; arms {len(ARMS)} x seeds {len(SEEDS)} + wrong-anchor {len(WRONG_ARMS)} x {len(SEEDS)}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "train": train, "score": score,
         "publish": publish, "validate": validate}
SHARDED = ("train", "score")


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
