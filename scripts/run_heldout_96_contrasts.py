"""Issue-104 work item 3: the #96 contrast set on the sealed #109 evaluation split (held out), scored once.

Binding runner module: scripts/run_heldout_96_contrasts.py
Exact validation command: python -u -m scripts.run_heldout_96_contrasts --validate

#104 asks for the #96 contrast set "on the held-out split only", with the #96 precision guard G6 passing
or an explicit statement of the achieved half-width. This runner is the faithful rerun:

- predictor and controller: the ICLR #77 N1 checkpoints (hybrid and continuous, seeds 20260908/09/10;
  ``run_lookahead_control.load_predictor`` / ``load_controller``), exactly the #96 stack;
- arms: the 18 #96 arms (nine fixed requests F, J, M, T, D, OL, C-F-1/5/15, C-J) with the #96 selector
  ``t96.select_pairs`` and ``t96.arm_spec``; T/D coordinates and the OL draw re-derived from this run's
  J traces with the #96 rules (outcome-free); F* (cross-fit r-dagger), F_hindsight and C-F* by the #96
  rules;
- cells: the Gate-B cells (``run_gate_b.evaluation_set``): per inventory the mixed-verdict levels of the
  #104 campaign stream ``issue109-evaluation`` with the engine pig-removal verdicts (grid 311, offset 76,
  angle 78, power 47 levels), each scored for every seed;
- anchors: the #105 F77 store (the #71 repaired-adapter batch-1 parse of each level's decision frame,
  parsed under ``deterministic_scoring()``), hash-pinned;
- costs: both #111 costs from the same endpoint (``EndpointCosts.row``); the tie-free cost is primary,
  the #96 count cost secondary (pre-declared);
- statistics: the #96 cell AUC (ties 0.5), the #96 contrasts (paired by cell, level-clustered percentile
  bootstrap, 10000 draws, PCG64 7201, DESCRIPTIVE), the #96 guards G1-G5 and caps, and G6 as #104 reads
  it: the primary interval's half-width <= delta = 0.02 (the #96 plan's literal G6 bound 0.05 is reported
  alongside).

The secondary reading, the #112 Gate-B predictor's #96-style contrasts published by #101, is re-tabulated
from the #101 compute.json with half-widths; nothing of it is re-scored.

Modes: --dry-run --smoke --prepare --run --frequencies --publish --validate. Zero training, zero engine
seconds. ``--run`` and ``--validate`` score under ``scoring_harness.deterministic_scoring()``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts import post_freeze_amendments as amendments
from scripts import run_lookahead_control as wlc
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import request_policy as rp
from world_model.training.matched_dynamics import pairs_for

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-104-heldout-contrasts-v1"
RUNNER = "scripts/run_heldout_96_contrasts.py"
IDENTITY = "issue-104-heldout-contrasts-v1"
SCHEMA_SMOKE = "issue_104_heldout_contrasts_smoke_v1"
SCHEMA_PLAN = "issue_104_heldout_contrasts_plan_v1"
SCHEMA_CELLS = "issue_104_heldout_contrasts_cells_v1"
SCHEMA_RECORD = "issue_104_heldout_contrasts_record_v1"
SCHEMA_FREQUENCIES = "issue_104_heldout_contrasts_frequencies_v1"
SCHEMA_LEDGER = "issue_104_heldout_contrasts_ledger_v1"
SCHEMA_COMPUTE = "issue_104_heldout_contrasts_compute_v1"
SCHEMA_REPORT = "issue_104_heldout_contrasts_report_v1"
SCHEMA_MANIFEST = "issue_104_heldout_contrasts_record_manifest_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_heldout_96_contrasts --validate"

DEVICE = "cuda"
SEEDS = t96.SEEDS
INVENTORIES = t96.INVENTORIES
PRIMARY = t96.PRIMARY
ARMS = t96.ARMS
PHASES = {1: t96.PHASE_1, 2: t96.PHASE_2}
FIXED_ARMS = t96.FIXED_ARMS
CONTINUOUS_FIXED_ARMS = t96.CONTINUOUS_FIXED_ARMS
CONTROLLER_ARMS = t96.CONTROLLER_ARMS
PAIR_NAMES = t96.PAIR_NAMES
DELTAS, ALPHAS = t96.DELTAS, t96.ALPHAS
COSTS = harness.COSTS
PRIMARY_COST = harness.PRIMARY_COST
SECONDARY_COST = harness.SECONDARY_COST
CONTRAST_ORDER = t96.CONTRAST_ORDER
GRID_SECONDARIES = ("J-M", "J-OL")

DELTA = t96.DELTA                       # 0.02, the #96 margin
G6_HALF_WIDTH = DELTA                   # #104: "G6 ... half-width 0.0747 against the pre-declared margin 0.02"
HALF_WIDTH_96_LITERAL = t96.HALF_WIDTH_MAX   # the #96 plan's literal G6 bound (0.05), reported alongside
DESIGN_DELTA = {"grid": 0.02, "offset": 0.05, "angle": 0.05, "power": 0.05}   # #104 frozen depth rule
EXPECTED_MIXED = {"grid": 311, "offset": 76, "angle": 78, "power": 47}         # #104 yield / Gate-B cells
NONFINITE_SHARE_MAX = t96.NONFINITE_SHARE_MAX
TIE_SHARE_MAX = t96.TIE_SHARE_MAX
NONMODAL_SHARE_MIN = t96.NONMODAL_SHARE_MIN
RANK_DIVERGENCE_SHARE_MIN = t96.RANK_DIVERGENCE_SHARE_MIN
COST_TOLERANCE = t96.COST_TOLERANCE
STOP_TOKEN = t96.STOP_TOKEN
GPU_CAP_SECONDS = 6 * 3600.0
WALL_CAP_SECONDS = 6 * 3600.0

# smoke: the #96 records must reproduce on their original cells (an exact record with >= 2 distinct costs, and
# the #96 G1 criterion on the #96 G1 cells); the G1 member's seed-20260910 J record is reported as schedule evidence
G1_MEMBER_J_RECORD = ("angle", "issue-77-n1-001", 20260910, "J")
G1_CELLS_96 = (("angle", t96.G1_MEMBER, t96.G1_SEED, "J"), ("angle", t96.G1_MEMBER, t96.G1_SEED, "F-15-continuous"))
ANCHOR_REPARSE_LEVELS = 8

ISSUE_101 = ROOT / ".local-artifacts/issue-101-decision-aware-policy-v1"
ISSUE_105 = ROOT / ".local-artifacts/issue-105-hierarchy-baselines-v1"
F77_ANCHORS = ISSUE_105 / "data" / "anchors-f77-evaluation.pt"
F77_SCHEMA = "issue_105_f77_anchors_v1"
F77_IDENTITY = "issue-105-hierarchy-baselines-v1"
CAMPAIGN = ROOT / ".local-artifacts/issue-104-power-cohort-campaign-v1"
COHORT = ROOT / ".local-artifacts/issue-109-sealed-cohort-v1/plan.json"
POWER_PLAN = ROOT / ".local-artifacts/issue-104-power-cohort-v1/plan.json"
PLAN_87 = t96.wp1.EIGHTY_SEVEN / "plan.json"
SECONDARY_101 = {   # the #101 #96-style contrasts (published names) re-tabulated, never re-scored
    "J-F*": "J - F* (cross-inventory r-dagger)", "J-F_hindsight": "J - F_hindsight", "J-M": "J - M",
    "J-OL": "J - OL", "J_dec-F*": "J_dec - F*", "J_dec-F_hindsight": "J_dec - F_hindsight",
    "J_dec-M_dec": "J_dec - M_dec", "J_dec-OL_dec": "J_dec - OL_dec",
}
NOT_IN_101 = ("J-F(1,macro)", "J-F(1,continuous)", "J-T", "J-D", "C-J-C-F*")

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
bootstrap = t96.bootstrap
fmt = t96.fmt
fmt_interval = t96.fmt_interval


def log(message):
    print(f"[issue-104-heldout-contrasts] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# cells (Gate-B rule) and anchors (#105 F77 store)
# ---------------------------------------------------------------------------

def cells_path(output):
    return Path(output) / "data" / "cells.json"


def derive_cells():
    """The Gate-B mixed cells per inventory (run_gate_b.evaluation_set), in a JSON-stable form."""
    from scripts import run_gate_b as gb
    if gb.INVENTORIES != INVENTORIES:
        raise ValueError("Gate-B inventory order differs from the #96 order")
    _, cells = gb.evaluation_set()
    return {"schema": SCHEMA_CELLS, "identity": IDENTITY,
            "rule": ("run_gate_b.evaluation_set: per #109 evaluation level the complete verdict-bearing candidates "
                     "of the inventory in ordinal order with the engine pig-removal verdicts of the #104 campaign "
                     "stream issue109-evaluation; mixed-verdict levels only"),
            "inventories": {inventory: {level: {
                "inventory": [{"ordinal": item["ordinal"], "action": item["action"], "key": item["key"]}
                              for item in entry["inventory"]],
                "verdicts": {str(o): bool(v) for o, v in sorted(entry["verdicts"].items())}}
                for level, entry in sorted(cells[inventory]["mixed"].items())} for inventory in INVENTORIES}}


def load_cells(output):
    payload = read_json(cells_path(output))
    if payload.get("schema") != SCHEMA_CELLS or payload.get("identity") != IDENTITY:
        raise ValueError("cells.json binding differs")
    return {inventory: {level: {"inventory": entry["inventory"],
                                "verdicts": {int(o): v for o, v in entry["verdicts"].items()}}
                        for level, entry in payload["inventories"][inventory].items()}
            for inventory in INVENTORIES}


def scheduled_cells(cells):
    """Frozen cell order (the #96 order): inventory, level, seed."""
    return [(inventory, level, seed) for inventory in INVENTORIES for level in sorted(cells[inventory])
            for seed in SEEDS]


def load_anchor_store():
    store = torch.load(F77_ANCHORS, map_location="cpu", weights_only=True)
    if store.get("schema") != F77_SCHEMA or store.get("plan_identity") != F77_IDENTITY:
        raise ValueError("#105 F77 anchor store binding differs")
    return store


# ---------------------------------------------------------------------------
# models and the scoring rollout
# ---------------------------------------------------------------------------

def checkpoint_paths():
    plan87 = read_json(PLAN_87)
    out = {}
    for seed in SEEDS:
        for family in ("hybrid", "continuous"):
            out[f"predictor_{family}_{seed}"] = Path(plan87["dynamics_checkpoints"][str(seed)][family]["path"])
            out[f"controller_{family}_{seed}"] = t96.DYNAMICS / f"seed-{seed}" / family / "controller.pt"
    return out


class Stack:
    """The #77 N1 predictors and controllers, loaded once per (seed, family)."""

    def __init__(self):
        self.plan87 = read_json(PLAN_87)
        self.models, self.controllers = {}, {}
        t96.check_pair_order()

    def model(self, seed, family):
        if (seed, family) not in self.models:
            self.models[(seed, family)] = wlc.load_predictor(self.plan87, seed, family, DEVICE)
        return self.models[(seed, family)]

    def controller(self, seed, family):
        if (seed, family) not in self.controllers:
            self.controllers[(seed, family)] = wlc.load_controller(seed, family, DEVICE)
        return self.controllers[(seed, family)]

    def for_arm(self, arm, seed):
        family = t96.arm_family(arm)
        return self.model(seed, family), (self.controller(seed, family) if arm in CONTROLLER_ARMS else None)


def score_cell(model, controller, z0, items, spec, costs):
    """One cell: the #96 rollout (request_policy.rollout_endpoints = t96.rollout's ops in order) from the
    anchor over the items' actions; both #111 costs of each endpoint against the anchor."""
    z0 = z0.to(DEVICE)
    actions = torch.cat([wlc.action_tensor(item["action"], DEVICE) for item in items], 0)
    z, timing = rp.rollout_endpoints(model, controller, z0, actions, spec, select=t96.select_pairs,
                                     record_schedules=True)
    pairs = controller.pairs if controller is not None else pairs_for(model)
    sequences = timing.pop("schedules")
    rows = []
    for slot, item in enumerate(items):
        row = costs.row(item["ordinal"], z[slot], z0)
        row["steps"] = sum(pairs[k].delta for k in sequences[slot])
        rows.append(row)
    return rows, sequences, timing


def choose(rows, cost):
    return wlc.choose([{"ordinal": r["ordinal"], "predicted_cost": r[cost]} for r in rows])


# ---------------------------------------------------------------------------
# smoke (pre-freeze): replication of #96 records on their original cells, anchors, cells, determinism
# ---------------------------------------------------------------------------

def compare_96(rows, sequences, timing, reference, arm, ol_rle):
    """A fresh #96-cell block against the retained #96 record (count cost)."""
    mine = {r["ordinal"]: r[SECONDARY_COST] for r in rows}
    theirs = {r["ordinal"]: r["predicted_cost"] for r in reference["ranking"]}
    order = lambda costs: sorted((o for o in costs if costs[o] is not None), key=lambda o: (costs[o], o))
    chosen, _ = choose(rows, SECONDARY_COST)
    names = PAIR_NAMES if t96.arm_family(arm) == "hybrid" else tuple(f"{d}-continuous" for d in DELTAS)
    pair_counts = Counter(k for sequence in sequences for k in sequence)
    schedules = ({str(r["ordinal"]): t96.rle(s) for r, s in zip(rows, sequences)} if arm in CONTROLLER_ARMS else None)
    finite = [abs(mine[o] - theirs[o]) for o in theirs if mine.get(o) is not None and theirs[o] is not None]
    checks = {
        "costs_bit_equal": mine == theirs,
        "chosen_equal": chosen == reference["chosen_ordinal"],
        "order_equal": order(mine) == order(theirs),
        "steps_equal": [r["steps"] for r in rows] == [r["steps"] for r in reference["ranking"]],
        "schedules_equal": schedules == reference["schedules"],
        "counts_equal": (timing["transition_calls"] == reference["transition_calls"]
                         and timing["controller_calls"] == reference["controller_calls"]
                         and timing["linear_macs"] == reference["linear_macs"]
                         and {names[k]: pair_counts[k] for k in sorted(pair_counts)} == reference["pair_step_counts"]),
        "ol_schedule_equal": ol_rle is None or reference.get("ol_schedule") == ol_rle,
    }
    return {**checks, "exact": all(checks.values()),
            "selected_cost_delta": (None if chosen is None or reference["selected_predicted_cost"] is None
                                    else abs(mine[chosen] - reference["selected_predicted_cost"])),
            "max_abs_count_delta": max(finite) if finite else 0.0,
            "distinct_reference_costs": len({c for c in theirs.values() if c is not None})}


def smoke_census(stack, costs):
    """Every #96 record (3186) re-scored on its original cell with the faithful path."""
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    anchors, anchor_evidence = t96.member_anchors(sources, members)
    frequencies = read_json(t96.OUTPUT / "frequencies.json")
    schedules = t96.ol_schedules(universe, frequencies)
    adapter = wlc.load_adapter(DEVICE)
    carriers = {member: wlc.encode_carrier(adapter, anchors[member], DEVICE).float() for member in members}
    results, gpu_by_arm, candidates_by_arm = {}, Counter(), Counter()
    began = time.monotonic()
    for inventory, member, seed in t96.scheduled_cells(universe):
        entry = universe[inventory][member]
        for arm in ARMS:
            spec = t96.arm_spec(arm, frequencies["coordinates"][str(seed)], schedules[(inventory, member, seed)])
            model, controller = stack.for_arm(arm, seed)
            rows, sequences, timing = score_cell(model, controller, carriers[member], entry["inventory"], spec, costs)
            reference = t96.read_record(t96.OUTPUT, inventory, member, seed, arm)["decision"]
            ol_rle = t96.rle(schedules[(inventory, member, seed)]) if arm == "OL" else None
            results[(inventory, member, seed, arm)] = compare_96(rows, sequences, timing, reference, arm, ol_rle)
            gpu_by_arm[arm] += timing["gpu_seconds"]
            candidates_by_arm[arm] += len(entry["inventory"])
    wall = time.monotonic() - began
    exact_nontrivial = sorted((k for k, r in results.items() if r["exact"] and r["distinct_reference_costs"] >= 2),
                              key=lambda k: t96.record_identity(*k))
    g1 = []
    for key in G1_CELLS_96:
        r = results[key]
        g1.append({"record": t96.record_identity(*key), "chosen_equal": r["chosen_equal"],
                   "selected_cost_delta": r["selected_cost_delta"], "tolerance": COST_TOLERANCE,
                   "ok": bool(r["chosen_equal"] and r["selected_cost_delta"] is not None
                              and r["selected_cost_delta"] <= COST_TOLERANCE)})
    keys = ("exact", "costs_bit_equal", "chosen_equal", "order_equal", "steps_equal", "schedules_equal",
            "counts_equal", "ol_schedule_equal")
    per_arm = {arm: {"records": sum(1 for k in results if k[3] == arm),
                     **{key: sum(1 for k, r in results.items() if k[3] == arm and r[key]) for key in keys},
                     "within_cost_tolerance": sum(1 for k, r in results.items() if k[3] == arm
                                                  and r["max_abs_count_delta"] <= COST_TOLERANCE),
                     "gpu_seconds": gpu_by_arm[arm], "candidate_rollouts": candidates_by_arm[arm]}
               for arm in ARMS}
    totals = {key: sum(r[key] for r in results.values()) for key in keys}
    totals["within_cost_tolerance"] = sum(r["max_abs_count_delta"] <= COST_TOLERANCE for r in results.values())
    totals["exact_with_two_or_more_distinct_costs"] = sum(
        r["exact"] and r["distinct_reference_costs"] >= 2 for r in results.values())
    chosen_differs = sorted(t96.record_identity(*k) for k, r in results.items() if not r["chosen_equal"])
    return {
        "control": "S1_S2_replication_of_96_records",
        "records": len(results), "totals": totals, "per_arm": per_arm,
        "chosen_differs": chosen_differs,
        "exact_nontrivial_records": [t96.record_identity(*k) for k in exact_nontrivial],
        "exact_record": ({"record": t96.record_identity(*exact_nontrivial[0]), **results[exact_nontrivial[0]]}
                         if exact_nontrivial else None),
        "exact_record_ok": bool(exact_nontrivial),
        "g1_member_J_record": {"record": t96.record_identity(*G1_MEMBER_J_RECORD), **results[G1_MEMBER_J_RECORD]},
        "g1_96_criterion": g1, "g1_96_ok": all(c["ok"] for c in g1),
        "anchor_g2_96_ok": all(r["ok"] for r in anchor_evidence),
        "hardware_note": ("#96 records were scored on the RTX 3090 workstation; this server is RTX 5090 "
                          "(SERVER_SETUP.md step 4: cross-hardware float replication does not hold bit for bit)"),
        "wall_seconds": wall,
    }


def smoke_cells_and_anchors():
    from scripts import run_gate_b as gb
    from scripts import run_hierarchy_baselines as hb
    from scripts import run_decision_dynamics_retrain as rt
    payload = derive_cells()
    counts = {inventory: len(payload["inventories"][inventory]) for inventory in INVENTORIES}
    store = load_anchor_store()
    levels = sorted(set().union(*(payload["inventories"][i].keys() for i in INVENTORIES)))
    branches, _ = gb.evaluation_set()
    first = rt.anchor_branches(gb.complete_branches(branches))
    reparse = levels[:ANCHOR_REPARSE_LEVELS]
    adapter = wlc.load_adapter(DEVICE)
    equal = {level: bool(torch.equal(hb.f77_anchor(adapter, first[level]["attempt"]), store["anchors"][level]))
             for level in reparse}
    frames_match = all(store["frames"][level] == first[level]["key"] for level in levels)
    return {
        "control": "S3_S4_cells_and_anchors",
        "mixed_levels": counts, "expected": EXPECTED_MIXED, "cells_ok": counts == EXPECTED_MIXED,
        "anchor_store_sha256": sha256_of(F77_ANCHORS),
        "anchor_store_policy_ok": store["scoring_policy"] == harness.SCORING_POLICY,
        "anchor_store_covers_mixed_levels": all(level in store["anchors"] for level in levels),
        "anchor_frames_follow_gate_b_rule": frames_match,
        "reparsed_levels": equal, "reparse_bit_equal": all(equal.values()),
        "ok": bool(counts == EXPECTED_MIXED and store["scoring_policy"] == harness.SCORING_POLICY
                   and all(level in store["anchors"] for level in levels) and frames_match and all(equal.values())),
    }


def smoke_determinism(stack, costs):
    """One #96 cell re-scored twice in this process: rows and schedules bit-identical."""
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    anchors, _ = t96.member_anchors(sources, [t96.G1_MEMBER])
    adapter = wlc.load_adapter(DEVICE)
    z0 = wlc.encode_carrier(adapter, anchors[t96.G1_MEMBER], DEVICE).float()
    entry = universe["angle"][t96.G1_MEMBER]
    model, controller = stack.for_arm("J", t96.G1_SEED)
    first = score_cell(model, controller, z0, entry["inventory"], {"kind": "joint"}, costs)
    again = score_cell(model, controller, z0, entry["inventory"], {"kind": "joint"}, costs)
    return {"control": "S5_in_process_determinism", "ok": bool(first[0] == again[0] and first[1] == again[1])}


def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        stack, costs = Stack(), harness.EndpointCosts(wlc.load_objective())
        identities = t96.checkpoint_identities(stack.plan87)
        determinism = smoke_determinism(stack, costs)
        cells_anchors = smoke_cells_and_anchors()
        census = smoke_census(stack, costs)
    per_candidate = {arm: census["per_arm"][arm]["gpu_seconds"] / census["per_arm"][arm]["candidate_rollouts"]
                     for arm in ARMS}
    candidates = sum(EXPECTED_MIXED[i] * {"grid": 16, "offset": 11, "angle": 13, "power": 20}[i] for i in INVENTORIES)
    projected = sum(per_candidate[arm] * candidates * len(SEEDS) for arm in ARMS)
    checks = {"S1_exact_96_record": census["exact_record_ok"], "S2_96_g1_criterion": census["g1_96_ok"],
              "S3_S4_cells_and_anchors": cells_anchors["ok"], "S5_determinism": determinism["ok"],
              "S6_checkpoint_identity": all(r["ok"] for r in identities),
              "S7_96_anchor_binding": census["anchor_g2_96_ok"]}
    evidence = {
        "schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "scoring_policy": policy,
        "device": torch.cuda.get_device_name(0), "checks": checks, "ok": all(checks.values()),
        "census": census, "cells_and_anchors": cells_anchors, "determinism": determinism,
        "checkpoints": identities,
        "projection": {"evaluation_candidate_rollouts_per_arm_and_seed": candidates,
                       "projected_gpu_seconds": projected, "per_candidate_gpu_seconds": per_candidate},
        "wall_seconds": time.monotonic() - began,
        "disclosure": ("before the smoke, an exploratory probe of the 54 records of the #96 G1 member "
                       "(angle/issue-77-n1-001) on this RTX 5090 found bit-exact reproduction mainly where the count "
                       "cost saturates (all candidates tied); a first smoke run named such a tied record "
                       "(angle/001/seed20260910/J) for S1, so S1 was revised before the freeze to require an exact "
                       "record with >= 2 distinct costs and the smoke was re-run; no evaluation-split cell was scored "
                       "before the freeze"),
    }
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'passed' if evidence['ok'] else 'FAILED'}: {checks}; census exact "
        f"{census['totals']['exact']}/{census['records']}, chosen equal {census['totals']['chosen_equal']}; "
        f"projected {projected:.0f} GPU s")
    return 0 if evidence["ok"] else 1


# ---------------------------------------------------------------------------
# frozen plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    from scripts import run_gate_b as gb
    names = {
        "issue_96_plan": t96.OUTPUT / "plan.json", "issue_96_compute": t96.OUTPUT / "compute.json",
        "issue_96_frequencies": t96.OUTPUT / "frequencies.json", "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
        "issue_94_runner_bootstrap_and_cell_auc": ROOT / "scripts/run_launch_power_probe.py",
        "issue_92_runner_loaders": ROOT / "scripts/run_lookahead_control.py",
        "issue_77_dynamics_plan": t96.DYNAMICS / "plan.json", "issue_87_plan": PLAN_87,
        "issue_105_f77_anchor_store": F77_ANCHORS, "gate_b_runner": ROOT / gb.RUNNER,
        "issue_104_campaign_branches": CAMPAIGN / "branches.json", "issue_109_cohort_plan": COHORT,
        "issue_104_power_plan": POWER_PLAN,
        "issue_101_compute": ISSUE_101 / "compute.json", "issue_101_summary": ISSUE_101 / "summary.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "request_policy_module": ROOT / "world_model/training/request_policy.py",
        "matched_dynamics": ROOT / "world_model/training/matched_dynamics.py",
        "cnn_hybrid": ROOT / "world_model/training/cnn_hybrid.py",
        "smoke": Path(output) / "smoke.json", "cells": cells_path(output),
        **{f"checkpoint_{name}": path for name, path in checkpoint_paths().items()},
    }
    return [{"name": name, "artifact": str(Path(path).resolve().relative_to(ROOT.resolve())), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_rules():
    return {
        "primary_cost": (f"{PRIMARY_COST} (#111 PRIMARY_COST, pre-declared); the #96 {SECONDARY_COST} cost is "
                         "secondary, scored from the same endpoints and reported in full with its own guards"),
        "primary": (f"theta_grid = AUC_J - AUC_F* over the mixed grid cells (tie-free cost); supported iff all guards "
                    f"pass AND mean >= {DELTA} AND lower bound > 0; not_supported_by_this_experiment iff all guards "
                    f"pass AND upper bound < {DELTA}; otherwise {STOP_TOKEN}"),
        "secondary_tokens": (f"J - M and J - OL on the grid: the same rule with guards G1-G5, G7 and caps and the "
                             f"contrast's own half-width <= {G6_HALF_WIDTH} in place of G6 (the #96 rule)"),
        "replication": ("J - F* on offset / angle / power reported verbatim with the count meeting mean >= delta AND "
                        "lower > 0 (the #96 rule, DESCRIPTIVE); not interpretable if any guard fails"),
        "g6": (f"G6 (#104 reading): primary interval half-width <= delta = {G6_HALF_WIDTH}; reported per inventory "
               f"for J - F* and per contrast; the #96 plan's literal G6 bound {HALF_WIDTH_96_LITERAL} and the #104 "
               "design delta (grid 0.02, union inventories 0.05) are reported alongside; where G6 fails the "
               "achieved half-width is stated with the mixed-level count m and m x (half-width / delta)^2"),
        "selection": ("F* (cross-fit r-dagger) = the fixed pair with the highest mean, over the other three "
                      "inventories, of the member-mean AUC; F_hindsight = in-inventory best fixed pair; C-F* likewise "
                      "over C-F; ties by arm order; selected per cost from that cost's AUCs (#96 rules)"),
        "secondary_reading": ("the #112 Gate-B predictor's #96-style contrasts published by #101 (compute.json), "
                              "re-tabulated with half-widths and the G6 reading; not re-scored"),
        "guard_failure": f"any failed guard turns every token of that cost into {STOP_TOKEN}",
    }


def frozen_plan(output, frozen_at, evidence):
    cells = load_cells(output)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 104,
        "related_issues": [96, 101, 105, 108, 109, 111, 112],
        "role": "#104 work item 3: the #96 contrast set re-run on the sealed #109 evaluation split, scored once",
        "frozen_at": frozen_at, "frozen_before_scoring": True,
        "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "validation_command": VALIDATION_COMMAND,
        "stack": {"predictor": "#77 N1 hybrid / continuous checkpoints bound by the #87 plan (the #96 stack)",
                  "controller": "#77 N1 MatchedController per seed and family (the #96 controllers)",
                  "anchors": ("#105 F77 store: the #71 repaired-adapter batch-1 parse of each evaluation level's "
                              "decision frame (shot frame 0 of its first complete candidate, the Gate-B rule), "
                              "parsed under deterministic_scoring()"),
                  "rollout": "request_policy.rollout_endpoints with t96.select_pairs / t96.arm_spec, endpoint 225",
                  "costs": {cost: harness.COST_DEFINITIONS[cost] for cost in COSTS}},
        "arms": {arm: t96.ARM_DEFINITIONS[arm] for arm in ARMS},
        "arm_rule_changes_vs_96": ("J step-pair frequencies (alpha_T, Delta_D, OL weights) pool this run's J traces "
                                   "over the scheduled cells, which are the mixed cells only; the OL draw uses one "
                                   "PCG64(7202) stream in the frozen order (inventory, level, seed)"),
        "cells": {inventory: len(cells[inventory]) for inventory in INVENTORIES},
        "scheduled_cells": len(scheduled_cells(cells)), "records": len(ARMS) * len(SEEDS),
        "seeds": list(SEEDS), "inventories": list(INVENTORIES), "primary_inventory": PRIMARY,
        "contrasts": list(CONTRAST_ORDER),
        "statistics": {"cell_auc": "t96.cell_view (P(success cost < failure cost), ties 0.5, admissible candidates)",
                       "interval": "t96.contrast: paired per (level, seed); level-clustered percentile bootstrap of "
                                   "member means, 10000 draws, PCG64 7201, [0.025, 0.975], DESCRIPTIVE",
                       "half_width": "(upper - lower) / 2"},
        "rules": frozen_rules(), "delta": DELTA, "g6_half_width": G6_HALF_WIDTH,
        "g6_96_literal": HALF_WIDTH_96_LITERAL, "design_delta": DESIGN_DELTA,
        "guards": {
            "G1": "smoke: at least one #96 record with >= 2 distinct costs reproduces exactly on its original cell "
                  "(bit-equal costs, chosen ordinal, ranking order, steps, schedules, calls, MACs; census over all "
                  "3186 #96 records reported), and the #96 G1 criterion (chosen ordinal equal, selected cost within "
                  "1e-3) holds on the #96 G1 J and F-15-continuous cells",
            "G2": "anchor store hash equals the pinned #105 hash, parsed under SCORING_POLICY, covers every scored "
                  "level, and the smoke re-parse is bit-identical",
            "G3": f"verdict resolution 1.0 on admissible candidates of mixed grid cells; nonfinite share <= {NONFINITE_SHARE_MAX}",
            "G4": f"<= {TIE_SHARE_MAX} of mixed grid cells have J or F* fully tied",
            "G5": (f"J non-modal step share >= {NONMODAL_SHARE_MIN} on mixed grid cells AND Kendall tau-b(J, F(modal)) < 1 "
                   f"on >= {RANK_DIVERGENCE_SHARE_MIN} of them"),
            "G6": f"primary (grid J - F*) half-width <= {G6_HALF_WIDTH}",
            "G7": "every record carries the #111 SCORING_POLICY",
            "caps": f"summed worker GPU seconds <= {GPU_CAP_SECONDS}, elapsed scoring wall <= {WALL_CAP_SECONDS} s",
        },
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "engine_seconds": 0},
        "launch": "phase 1 (F x9, J, M, C-F x3, C-J) on 4 GPUs once; --frequencies once; phase 2 (T, D, OL) once",
        "inputs": input_bindings(output),
        "smoke_evidence": {"ok": evidence["ok"], "checks": evidence["checks"], "run_at": evidence["run_at"],
                           "projection": evidence["projection"]["projected_gpu_seconds"],
                           "census_totals": evidence["census"]["totals"],
                           "exact_record": evidence["census"]["exact_record"]["record"]},
        "claim_boundary": ("decision-only rollouts of the frozen ICLR #77 N1 checkpoints and controllers from the #105 "
                           "F77 anchors of the sealed #109 evaluation levels to endpoint 225 on the Gate-B mixed cells; "
                           "inventories never pooled; every interval DESCRIPTIVE; zero engine seconds; no training"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-104 held-out contrast protocol")
    if not plan.get("frozen_before_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    for entry in plan["inputs"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    if not amendments.matches(RUNNER, plan["runner_sha256_at_freeze"], sha256_of(ROOT / RUNNER)):
        raise ValueError("runner changed after the freeze without a registered amendment")
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
    cells_path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(cells_path(output), derive_cells())
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log("frozen plan published")
    return 0


# ---------------------------------------------------------------------------
# scoring (phase 1, frequencies, phase 2)
# ---------------------------------------------------------------------------

def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def read_record(output, arm, seed):
    record = read_json(record_path(output, arm, seed))
    if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {arm}/{seed} binding differs")
    return record


def ledger_dir(output):
    return Path(output) / "ledger"


def ledger_totals(output):
    files = sorted(ledger_dir(output).glob("*.json"))
    entries = [read_json(path) for path in files]
    return {"gpu_seconds": sum(e["gpu_seconds"] for e in entries),
            "process_wall_seconds": sum(e["wall_seconds"] for e in entries),
            "first_started_at": min((e["started_at"] for e in entries), default=None),
            "last_finished_at": max((e["finished_at"] for e in entries if e["finished_at"]), default=None),
            "workers": [{k: e[k] for k in ("phase", "worker", "pid", "device", "started_at", "finished_at",
                                         "wall_seconds", "gpu_seconds", "status", "tasks")} for e in entries]}


def elapsed_seconds(totals):
    if not totals["first_started_at"] or not totals["last_finished_at"]:
        return 0.0
    parse = lambda text: datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (parse(totals["last_finished_at"]) - parse(totals["first_started_at"])).total_seconds()


def phase_inputs(output, cells, phase):
    if phase == 1:
        return None, None
    frequencies = read_json(Path(output) / "frequencies.json")
    if frequencies != json.loads(json_text(j_frequencies(output, cells))):
        raise ValueError("frequencies.json differs from the J records")
    return frequencies, t96.ol_schedules(cells, frequencies)


def score_arm(output, plan, stack, costs, anchors, cells, arm, seed, frequencies, schedules, policy):
    model, controller = stack.for_arm(arm, seed)
    coordinates = None if frequencies is None else frequencies["coordinates"][str(seed)]
    inventories, totals = {}, Counter()
    began = time.monotonic()
    for inventory in INVENTORIES:
        inventories[inventory] = {}
        for level in sorted(cells[inventory]):
            schedule = None if schedules is None else schedules[(inventory, level, seed)]
            spec = t96.arm_spec(arm, coordinates, schedule)
            rows, sequences, timing = score_cell(model, controller, anchors[level], cells[inventory][level]["inventory"],
                                                 spec, costs)
            inventories[inventory][level] = {
                "rows": rows,
                "schedules": ({str(r["ordinal"]): t96.rle(s) for r, s in zip(rows, sequences)}
                              if arm in CONTROLLER_ARMS else None),
                "ol_schedule": t96.rle(schedule) if arm == "OL" else None,
                **timing}
            totals.update({k: timing[k] for k in ("transition_calls", "controller_calls", "linear_macs", "gpu_seconds")})
    extra = {}
    if arm == "T":
        extra = {"alpha_T": coordinates["alpha_T"]}
    if arm == "D":
        extra = {"delta_D": coordinates["delta_D"]}
    paths = checkpoint_paths()
    family = t96.arm_family(arm)
    hashes = {"predictor_sha256": sha256_of(paths[f"predictor_{family}_{seed}"])}
    if arm in CONTROLLER_ARMS:
        hashes["controller_sha256"] = sha256_of(paths[f"controller_{family}_{seed}"])
    return {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "plan_version": plan["version"],
            "scoring_policy": policy, "scored_at": utc_now(),
            "cell": {"arm": arm, "seed": seed, "family": family, "endpoint": t96.ENDPOINT},
            **extra, **hashes, "inventories": inventories, "totals": dict(totals),
            "wall_seconds": time.monotonic() - began, "engine_seconds": 0}


def verify_checkpoints(plan):
    pinned = {entry["name"]: entry["sha256"] for entry in plan["inputs"]}
    for name, path in checkpoint_paths().items():
        if sha256_of(path) != pinned[f"checkpoint_{name}"]:
            raise ValueError(f"checkpoint {name} differs from the frozen hash")


def run(output, phase, workers, worker):
    output = Path(output)
    plan = load_plan(output)
    cells = load_cells(output)
    verify_checkpoints(plan)
    store = load_anchor_store()
    anchors = store["anchors"]
    frequencies, schedules = phase_inputs(output, cells, phase)
    tasks = [(arm, seed) for arm in PHASES[phase] for seed in SEEDS][worker::workers]
    ledger_dir(output).mkdir(parents=True, exist_ok=True)
    (output / "records").mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir(output) / f"phase{phase}-worker{worker}.json"
    if ledger_path.is_file():
        raise ValueError(f"{ledger_path.name} exists: this stage was launched already; never relaunch")
    entry = {"schema": SCHEMA_LEDGER, "phase": phase, "worker": worker, "workers": workers, "pid": os.getpid(),
             "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_at": utc_now(), "finished_at": None,
             "wall_seconds": 0.0, "gpu_seconds": 0.0, "status": "running", "tasks": []}
    write_json(ledger_path, entry)
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        stack, costs = Stack(), harness.EndpointCosts(wlc.load_objective())
        for arm, seed in tasks:
            path = record_path(output, arm, seed)
            if path.is_file():
                raise ValueError(f"record {path.name} exists; scoring is once")
            totals = ledger_totals(output)
            if totals["gpu_seconds"] > GPU_CAP_SECONDS:
                entry.update(status=f"{STOP_TOKEN}: GPU cap", finished_at=utc_now())
                write_json(ledger_path, entry)
                raise ValueError(f"{STOP_TOKEN}: GPU cap reached before {arm}/{seed}")
            record = score_arm(output, plan, stack, costs, anchors, cells, arm, seed, frequencies, schedules, policy)
            write_json(path, record)
            entry["tasks"].append(f"{arm}--seed{seed}")
            entry["gpu_seconds"] += record["totals"]["gpu_seconds"]
            entry["wall_seconds"] = time.monotonic() - began
            write_json(ledger_path, entry)
            log(f"phase {phase} worker {worker}: {arm} seed {seed} ({record['wall_seconds']:.0f}s)")
    entry.update(status="terminal", finished_at=utc_now(), wall_seconds=time.monotonic() - began)
    write_json(ledger_path, entry)
    return 0


def j_frequencies(output, cells):
    """J step-pair counts per seed over every scheduled cell and candidate (the #96 rule, outcome-free)."""
    counts = {seed: [0] * len(PAIR_NAMES) for seed in SEEDS}
    for seed in SEEDS:
        record = read_record(output, "J", seed)
        for inventory, level, _ in [c for c in scheduled_cells(cells) if c[2] == seed]:
            for text in record["inventories"][inventory][level]["schedules"].values():
                for index in t96.unrle(text):
                    counts[seed][index] += 1
    coordinates = {}
    for seed in SEEDS:
        by_alpha = [sum(counts[seed][d * len(ALPHAS) + a] for d in range(len(DELTAS))) for a in range(len(ALPHAS))]
        by_delta = [sum(counts[seed][d * len(ALPHAS) + a] for a in range(len(ALPHAS))) for d in range(len(DELTAS))]
        alpha = max(range(len(ALPHAS)), key=lambda a: (by_alpha[a], -a))
        delta = max(range(len(DELTAS)), key=lambda d: (by_delta[d], -d))
        coordinates[str(seed)] = {"alpha_T_index": alpha, "alpha_T": ALPHAS[alpha], "delta_D_index": delta,
                                  "delta_D": DELTAS[delta], "alpha_marginal_steps": by_alpha,
                                  "delta_marginal_steps": by_delta}
    return {"schema": SCHEMA_FREQUENCIES, "identity": IDENTITY,
            "rule": ("J step-pair counts pooled per seed over every scheduled (mixed) cell of all four inventories "
                     "and every candidate (outcome-free); alpha_T / Delta_D = modal marginal, ties to the lower index"),
            "pairs": list(PAIR_NAMES), "counts": {str(seed): counts[seed] for seed in SEEDS},
            "coordinates": coordinates}


def frequencies_mode(output):
    output = Path(output)
    load_plan(output)
    cells = load_cells(output)
    if any(not record_path(output, arm, seed).is_file() for arm in PHASES[1] for seed in SEEDS):
        raise ValueError("phase 1 is not complete")
    if any(record_path(output, arm, seed).is_file() for arm in PHASES[2] for seed in SEEDS):
        raise ValueError("phase 2 records exist before frequencies.json")
    value = j_frequencies(output, cells)
    path = output / "frequencies.json"
    if path.is_file():
        if read_json(path) != json.loads(json_text(value)):
            raise ValueError("frequencies.json exists and differs")
        log("frequencies.json already written and equal")
        return 0
    write_json(path, value)
    log(f"frequencies written: {value['coordinates']}")
    return 0


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def load_records(output):
    return {(arm, seed): read_record(output, arm, seed) for arm in ARMS for seed in SEEDS}


def record_manifest(output):
    output = Path(output)
    paths = sorted(record_path(output, arm, seed) for arm in ARMS for seed in SEEDS)
    present = sorted(output.glob("records/*.json"))
    if paths != present:
        raise ValueError(f"records/ holds {len(present)} files, expected exactly the {len(paths)} named records")
    return {"schema": SCHEMA_MANIFEST, "identity": IDENTITY, "count": len(paths),
            "records": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256_of(p)} for p in paths}}


def inventory_rows(cells, records, inventory, cost):
    """One row per mixed level x seed with the #96 cell view of every arm under ``cost``."""
    rows = []
    for level in sorted(cells[inventory]):
        entry = cells[inventory][level]
        for seed in SEEDS:
            arms = {}
            for arm in ARMS:
                block = records[(arm, seed)]["inventories"][inventory][level]
                pseudo = {"decision": {"ranking": [{"ordinal": r["ordinal"], "predicted_cost": r[cost]}
                                                   for r in block["rows"]]}}
                view = t96.cell_view(entry, pseudo)
                if arm == "J":
                    view["schedules"] = block["schedules"]
                if arm == "J" or arm.startswith("F-"):
                    view["costs"] = {r["ordinal"]: r[cost] for r in block["rows"]}
                arms[arm] = view
            rows.append({"member": level, "seed": seed, "arms": arms})
    return rows


def half_width_statement(block, members, threshold):
    if block is None:
        return None
    hw = block["half_width"]
    return {"half_width": hw, "threshold": threshold, "pass": hw <= threshold, "mixed_levels": members,
            "levels_for_threshold_by_sqrt_scaling": math.ceil(members * (hw / threshold) ** 2)}


def cost_tables(cells, records, cost, design):
    """The #96 per-inventory tables, guard inputs, guards G3-G6 and tokens for one cost."""
    tables = {inventory: inventory_rows(cells, records, inventory, cost) for inventory in INVENTORIES}
    member_means = {inventory: {arm: t96.member_mean([{"member": r["member"], "auc": r["arms"][arm]["auc"]}
                                                      for r in tables[inventory]], "auc") for arm in ARMS}
                    for inventory in INVENTORIES}

    def cross_fit(inventory, arms):
        scores = {}
        for arm in arms:
            others = [member_means[other][arm] for other in INVENTORIES
                      if other != inventory and member_means[other][arm] is not None]
            scores[arm] = float(np.mean(others)) if others else None
        best = max(arms, key=lambda arm: (-math.inf if scores[arm] is None else scores[arm], -arms.index(arm)))
        return best, scores

    per_inventory = {}
    for inventory in INVENTORIES:
        rows = tables[inventory]
        members = len({r["member"] for r in rows})
        f_star, f_scores = cross_fit(inventory, list(FIXED_ARMS))
        c_star, c_scores = cross_fit(inventory, list(CONTINUOUS_FIXED_ARMS))
        hindsight = max(FIXED_ARMS, key=lambda arm: (
            -math.inf if member_means[inventory][arm] is None else member_means[inventory][arm], -FIXED_ARMS.index(arm)))
        contrasts = {
            "J-F*": t96.contrast(rows, "J", f_star), "J-F_hindsight": t96.contrast(rows, "J", hindsight),
            "J-F(1,macro)": t96.contrast(rows, "J", "F-1-macro"),
            "J-F(1,continuous)": t96.contrast(rows, "J", "F-1-continuous"),
            "J-M": t96.contrast(rows, "J", "M"), "J-OL": t96.contrast(rows, "J", "OL"),
            "J-T": t96.contrast(rows, "J", "T"), "J-D": t96.contrast(rows, "J", "D"),
            "C-J-C-F*": t96.contrast(rows, "C-J", c_star),
        }
        for name, block in contrasts.items():
            est = block["estimate"]
            block["half_width_002"] = half_width_statement(est, members, G6_HALF_WIDTH)
            block["half_width_design"] = half_width_statement(est, members, DESIGN_DELTA[inventory])
            block["design_required_mixed_members"] = design[inventory].get(name)
        top1 = t96.contrast(rows, "J", f_star, key="top1_hit")
        spread_values = []
        for row in rows:
            aucs = [row["arms"][arm]["auc"] for arm in FIXED_ARMS if row["arms"][arm]["auc"] is not None]
            if aucs:
                spread_values.append({"member": row["member"], "spread": max(aucs) - min(aucs)})
        spread = bootstrap([r["spread"] for r in spread_values], [r["member"] for r in spread_values])
        arm_auc = {arm: bootstrap([r["arms"][arm]["auc"] for r in rows if r["arms"][arm]["auc"] is not None],
                                  [r["member"] for r in rows if r["arms"][arm]["auc"] is not None]) for arm in ARMS}
        hits = {arm: sum(1 for r in rows if r["arms"][arm]["top1_hit"]) for arm in ("J", f_star)}
        chance = [r["arms"]["J"]["chance_top1"] for r in rows if r["arms"]["J"]["chance_top1"] is not None]
        resolved = sum(r["arms"][arm]["resolved"] for r in rows for arm in ARMS)
        slots = sum(r["arms"][arm]["verdict_slots"] for r in rows for arm in ARMS)
        nonfinite = sum(r["arms"][arm]["nonfinite"] for r in rows for arm in ARMS)
        tied = sum(1 for r in rows if r["arms"]["J"]["all_tied"] or r["arms"][f_star]["all_tied"])
        steps = Counter()
        for r in rows:
            for text in r["arms"]["J"]["schedules"].values():
                steps.update(t96.unrle(text))
        total_steps = sum(steps.values())
        modal_index = max(range(len(PAIR_NAMES)), key=lambda k: (steps[k], -k)) if total_steps else 0
        modal_arm = f"F-{PAIR_NAMES[modal_index]}"
        divergent = 0
        for r in rows:
            j_costs, f_costs = r["arms"]["J"]["costs"], r["arms"][modal_arm]["costs"]
            common = sorted(o for o in j_costs if o in cells[inventory][r["member"]]["verdicts"]
                            and j_costs[o] is not None and f_costs.get(o) is not None)
            tau = t96.kendall([j_costs[o] for o in common], [f_costs[o] for o in common])
            divergent += int(tau is not None and tau < 1.0 - 1e-12)
        primary = contrasts["J-F*"]["estimate"]
        per_inventory[inventory] = {
            "mixed_levels": members, "mixed_cells": len(rows),
            "F_star": f_star, "F_star_cross_fit_scores": f_scores, "C_F_star": c_star,
            "C_F_star_cross_fit_scores": c_scores, "F_hindsight": hindsight,
            "arm_member_mean_auc": member_means[inventory], "arm_auc": arm_auc, "contrasts": contrasts,
            "g6": {"contrast": "J-F*", **(half_width_statement(primary, members, G6_HALF_WIDTH) or {}),
                   "pass_96_literal": primary is not None and primary["half_width"] <= HALF_WIDTH_96_LITERAL,
                   "pass_design_delta": primary is not None and primary["half_width"] <= DESIGN_DELTA[inventory],
                   "design_delta": DESIGN_DELTA[inventory]},
            "top1": {"J_hits": hits["J"], "F_star_hits": hits[f_star], "cells": len(rows),
                     "chance_mean": float(np.mean(chance)) if chance else None, "difference": top1},
            "fixed_pair_spread": spread,
            "guard_inputs": {
                "verdict_resolution": resolved / slots if slots else None,
                "nonfinite_share": nonfinite / slots if slots else None,
                "tied_cells": tied, "tied_share": tied / len(rows) if rows else None,
                "J_modal_pair": PAIR_NAMES[modal_index],
                "J_step_counts": {PAIR_NAMES[k]: steps[k] for k in sorted(steps)},
                "J_nonmodal_step_share": 1 - steps[modal_index] / total_steps if total_steps else None,
                "rank_divergent_cells": divergent, "rank_divergent_share": divergent / len(rows) if rows else None},
        }
    return per_inventory


def cost_guards(per_inventory, shared):
    grid = per_inventory[PRIMARY]
    gi = grid["guard_inputs"]
    primary = grid["contrasts"]["J-F*"]["estimate"]
    guards = {
        **{name: shared[name] for name in ("G1", "G2")},
        "G3": {"observed": {"verdict_resolution": gi["verdict_resolution"], "nonfinite_share": gi["nonfinite_share"]},
               "pass": gi["verdict_resolution"] == 1.0 and gi["nonfinite_share"] <= NONFINITE_SHARE_MAX},
        "G4": {"observed": {"tied_cells": gi["tied_cells"], "cells": grid["mixed_cells"], "tied_share": gi["tied_share"]},
               "pass": gi["tied_share"] is not None and gi["tied_share"] <= TIE_SHARE_MAX},
        "G5": {"observed": {"J_modal_pair": gi["J_modal_pair"], "J_nonmodal_step_share": gi["J_nonmodal_step_share"],
                            "rank_divergent_share": gi["rank_divergent_share"]},
               "pass": (gi["J_nonmodal_step_share"] is not None and gi["J_nonmodal_step_share"] >= NONMODAL_SHARE_MIN
                        and gi["rank_divergent_share"] >= RANK_DIVERGENCE_SHARE_MIN)},
        "G6": {"observed": {"half_width": None if primary is None else primary["half_width"],
                            "threshold": G6_HALF_WIDTH},
               "pass": primary is not None and primary["half_width"] <= G6_HALF_WIDTH},
        **{name: shared[name] for name in ("G7", "caps")},
    }
    return guards


def cost_tokens(per_inventory, guards):
    """Tokens of one cost. Frozen guards.failure rule: any failed guard turns every token of the cost into the stop
    token. The #96 secondary rule (guards except G6, plus the contrast's own half-width) and the #96 plan's literal
    G6 bound are kept as DESCRIPTIVE readings beside the tokens, never as tokens."""
    grid = per_inventory[PRIMARY]
    primary = grid["contrasts"]["J-F*"]["estimate"]
    guards_ok = all(g["pass"] for g in guards.values())
    base_ok = all(g["pass"] for name, g in guards.items() if name != "G6")
    literal_ok = base_ok and primary is not None and primary["half_width"] <= HALF_WIDTH_96_LITERAL
    secondary = {}
    for name in GRID_SECONDARIES:
        block = grid["contrasts"][name]["estimate"]
        own = block is not None and block["half_width"] <= G6_HALF_WIDTH
        own_literal = block is not None and block["half_width"] <= HALF_WIDTH_96_LITERAL
        secondary[name] = {"token": t96.token(block, guards_ok and own),
                           "own_half_width": None if block is None else block["half_width"],
                           "own_half_width_pass": own,
                           "reading_by_96_secondary_rule": t96.token(block, base_ok and own),
                           "reading_at_96_literal_bound": t96.token(block, base_ok and own_literal)}
    replication = {}
    for inventory in INVENTORIES:
        if inventory == PRIMARY:
            continue
        block = per_inventory[inventory]["contrasts"]["J-F*"]["estimate"]
        replication[inventory] = {"estimate": block, "meets_rule": bool(
            block is not None and block["mean"] >= DELTA and block["interval"][0] > 0)}
    spread = grid["fixed_pair_spread"]["mean"] if grid["fixed_pair_spread"] else None
    return {
        "guards_pass": guards_ok,
        "failed_guards": [name for name, g in guards.items() if not g["pass"]],
        "primary": {"estimand": "theta_grid = AUC_J - AUC_F* over mixed grid cells", "F_star": grid["F_star"],
                    "estimate": primary, "delta": DELTA, "token": t96.token(primary, guards_ok),
                    "reading_at_96_literal_bound": t96.token(primary, literal_ok)},
        "secondary_tokens": secondary,
        "replication": {"inventories": replication, "count": sum(r["meets_rule"] for r in replication.values()),
                        "of": len(replication), "interpretable": guards_ok},
        "fixed_pair_spread_grid_member_mean": spread,
        "near_inert": spread is not None and spread < DELTA,
    }


def design_requirements():
    """#104 power design: required mixed members per inventory and contrast at the design delta."""
    power = read_json(POWER_PLAN)["power"]["contrasts"]
    return {inventory: {name: block["required_mixed_members"][str(DESIGN_DELTA[inventory])]["design"]
                        for name, block in power[inventory].items()} for inventory in INVENTORIES}


def secondary_101():
    """#101's #96-style contrasts of the #112 Gate-B predictor, re-tabulated (never re-scored)."""
    compute = read_json(ISSUE_101 / "compute.json")
    out = {}
    for inventory in INVENTORIES:
        block = compute["inventories"][inventory]
        members = block["members"]
        rows = {}
        for name in SECONDARY_101:
            c = block["contrasts"][name]
            est = c["estimate"]
            hw = (est["interval"][1] - est["interval"][0]) / 2
            rows[name] = {"first": c["first"], "second": c["second"], "cells": c["cells"], "mean": est["mean"],
                          "interval": est["interval"], "half_width": hw, "token_101": c["token"],
                          "half_width_002": half_width_statement({"half_width": hw}, members, G6_HALF_WIDTH),
                          "half_width_design": half_width_statement({"half_width": hw}, members,
                                                                    DESIGN_DELTA[inventory])}
        out[inventory] = {"members": members, "cells": block["cells"], "F_star": block["F_star"],
                          "F_hindsight": block["F_hindsight"], "contrasts": rows,
                          "g6": {"contrast": "J-F*", "pass": rows["J-F*"]["half_width"] <= G6_HALF_WIDTH,
                                 "half_width": rows["J-F*"]["half_width"]}}
    return {"source": ".local-artifacts/issue-101-decision-aware-policy-v1/compute.json",
            "predictor": ("#112 v3 LC hybrid Gate-B handoff checkpoints (fold 'all', seeds 20260908/09/10), E99 Gate-B "
                          "anchors, tie-free cost; J = J-pe-1e-04 (ICLR prediction-error DP), J_dec = J-dec-1e-01"),
            "guards_101": {name: g["pass"] for name, g in compute["guards"].items()},
            "not_in_101": list(NOT_IN_101), "inventories": out}


def compute_tables(output, plan):
    output = Path(output)
    cells = load_cells(output)
    records = load_records(output)
    frequencies = read_json(output / "frequencies.json")
    if frequencies != json.loads(json_text(j_frequencies(output, cells))):
        raise ValueError("frequencies.json differs from the retained J records")
    schedules = t96.ol_schedules(cells, frequencies)
    accounting = {arm: {"records": 0, "transition_calls": 0, "controller_calls": 0, "linear_macs": 0,
                        "gpu_seconds": 0.0, "candidate_rollouts": 0} for arm in ARMS}
    nonfinite = {cost: 0 for cost in COSTS}
    candidates = 0
    for (arm, seed), record in records.items():
        if record["scoring_policy"] != harness.SCORING_POLICY:
            raise ValueError(f"record {arm}/{seed} was not scored under the #111 policy")
        if arm == "T" and record["alpha_T"] != frequencies["coordinates"][str(seed)]["alpha_T"]:
            raise ValueError("T coordinate differs from frequencies.json")
        if arm == "D" and record["delta_D"] != frequencies["coordinates"][str(seed)]["delta_D"]:
            raise ValueError("D coordinate differs from frequencies.json")
        block = accounting[arm]
        block["records"] += 1
        for inventory, level, s in scheduled_cells(cells):
            if s != seed:
                continue
            cell = record["inventories"][inventory][level]
            if arm == "OL" and cell["ol_schedule"] != t96.rle(schedules[(inventory, level, seed)]):
                raise ValueError(f"OL schedule of {inventory}/{level}/{seed} differs from the frozen draw")
            for key in ("transition_calls", "controller_calls", "linear_macs", "gpu_seconds"):
                block[key] += cell[key]
            block["candidate_rollouts"] += len(cell["rows"])
            candidates += len(cell["rows"])
            for cost in COSTS:
                nonfinite[cost] += sum(r[cost] is None for r in cell["rows"])
    ledger = ledger_totals(output)
    elapsed = elapsed_seconds(ledger)
    status_ok = all(w["status"] == "terminal" for w in ledger["workers"])
    smoke = read_json(output / "smoke.json")
    store = load_anchor_store()
    levels = sorted(set().union(*(cells[i].keys() for i in INVENTORIES)))
    shared = {
        "G1": {"observed": {"exact_record": (smoke["census"]["exact_record"] or {}).get("record"),
                            "exact": smoke["checks"]["S1_exact_96_record"],
                            "g1_96_criterion": smoke["census"]["g1_96_criterion"]},
               "pass": bool(smoke["checks"]["S1_exact_96_record"] and smoke["checks"]["S2_96_g1_criterion"])},
        "G2": {"observed": {"store_sha256": sha256_of(F77_ANCHORS), "levels": len(levels),
                            "store_policy": store["scoring_policy"],
                            "reparse_bit_equal": smoke["cells_and_anchors"]["reparse_bit_equal"]},
               "pass": bool(store["scoring_policy"] == harness.SCORING_POLICY
                            and all(level in store["anchors"] for level in levels)
                            and smoke["cells_and_anchors"]["reparse_bit_equal"])},
        "G7": {"observed": {"records": len(records)},
               "pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds"], "elapsed_wall_seconds": elapsed,
                              "workers_terminal": status_ok},
                 "pass": bool(status_ok and ledger["gpu_seconds"] <= GPU_CAP_SECONDS and elapsed <= WALL_CAP_SECONDS)},
    }
    design = design_requirements()
    costs_block = {}
    for cost in COSTS:
        per_inventory = cost_tables(cells, records, cost, design)
        guards = cost_guards(per_inventory, shared)
        costs_block[cost] = {"role": "primary" if cost == PRIMARY_COST else "secondary", "guards": guards,
                             **cost_tokens(per_inventory, guards), "inventories": per_inventory,
                             "nonfinite_candidates": nonfinite[cost]}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "structure": {"records": len(records), "scheduled_cells": len(scheduled_cells(cells)), "arms": len(ARMS),
                      "mixed_levels": {i: len(cells[i]) for i in INVENTORIES}, "candidate_rollouts": candidates},
        "frequencies": frequencies, "primary_cost": PRIMARY_COST, "costs": costs_block,
        "g6_reading": {"threshold": G6_HALF_WIDTH, "literal_96_threshold": HALF_WIDTH_96_LITERAL,
                       "design_delta": DESIGN_DELTA},
        "issue_96_reference": {"primary": read_json(t96.OUTPUT / "compute.json")["primary"],
                               "mixed_cells": read_json(t96.OUTPUT / "compute.json")["structure"]["mixed_cells"]},
        "secondary_101": secondary_101(),
        "compute": {"per_arm": accounting, "run_gpu_seconds": ledger["gpu_seconds"],
                    "process_wall_seconds": ledger["process_wall_seconds"], "elapsed_wall_seconds": elapsed,
                    "workers": ledger["workers"], "engine_seconds": 0,
                    "smoke_census_wall_seconds": smoke["census"]["wall_seconds"],
                    "smoke_wall_seconds": smoke["wall_seconds"],
                    "smoke_projected_gpu_seconds": smoke["projection"]["projected_gpu_seconds"]},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def label(block, name):
    if name == "J-F*":
        return f"J - F* [{block['F_star']}]"
    if name == "J-F_hindsight":
        return f"J - F_hindsight [{block['F_hindsight']}]"
    if name == "C-J-C-F*":
        return f"C-J - C-F* [{block['C_F_star']}]"
    return name.replace("-", " - ", 1)


def hw(block):
    return "n/a" if block is None else fmt(block["half_width"])


def cost_section(compute, cost, lines):
    c = compute["costs"][cost]
    p = c["primary"]
    lines += [f"## {cost} cost ({c['role']})", "",
              f"- primary theta_grid = AUC_J - AUC_F* (F* = {p['F_star']}): **{fmt_interval(p['estimate'])}**, "
              f"half-width {hw(p['estimate'])} -> **{p['token']}** (delta {DELTA}; G6 half-width <= {G6_HALF_WIDTH}; "
              f"DESCRIPTIVE reading with the #96 plan's literal G6 bound {HALF_WIDTH_96_LITERAL}: "
              f"{p['reading_at_96_literal_bound']})"]
    for name, s in c["secondary_tokens"].items():
        lines.append(f"- secondary {name} (grid): {fmt_interval(c['inventories'][PRIMARY]['contrasts'][name]['estimate'])}, "
                     f"own half-width {fmt(s['own_half_width'])} -> **{s['token']}** (DESCRIPTIVE readings, not "
                     f"tokens: by the #96 secondary rule, which skips the primary's G6, "
                     f"{s['reading_by_96_secondary_rule']}; at the literal bound {s['reading_at_96_literal_bound']})")
    r = c["replication"]
    lines.append(f"- replication of J - F* on offset/angle/power (mean >= delta AND lower > 0, DESCRIPTIVE): "
                 f"{r['count']}/{r['of']}" + ("" if r["interpretable"] else " (not interpretable: a guard failed)"))
    if not c["guards_pass"]:
        lines.append(f"- failed guards: {', '.join(c['failed_guards'])}; every token of this cost is {STOP_TOKEN}")
    lines += ["", "### Guards", "", "| guard | pass | observed |", "|---|---|---|"]
    for name, g in c["guards"].items():
        lines.append(f"| {name} | {g['pass']} | `{json.dumps(g['observed'], sort_keys=True)}` |")
    lines += ["", "### G6 per inventory (J - F*)", "",
              f"| inventory | mixed levels / cells | half-width | <= {G6_HALF_WIDTH} | <= {HALF_WIDTH_96_LITERAL} (#96 literal) "
              "| <= design delta | levels for 0.02 by 1/sqrt(m) |", "|---|---|---|---|---|---|---|"]
    for inventory in INVENTORIES:
        b = c["inventories"][inventory]
        g = b["g6"]
        lines.append(f"| {inventory} | {b['mixed_levels']} / {b['mixed_cells']} | {fmt(g.get('half_width'))} | "
                     f"{g.get('pass')} | {g['pass_96_literal']} | {g['pass_design_delta']} ({g['design_delta']}) | "
                     f"{g.get('levels_for_threshold_by_sqrt_scaling')} |")
    lines += ["", "### Contrasts per inventory (AUC difference, paired by level x seed, level-clustered; never pooled)", "",
              "| inventory | contrast | estimate [95% DESCRIPTIVE] | half-width | <= 0.02 | <= design delta | "
              "#104 design members | levels for 0.02 |", "|---|---|---|---|---|---|---|---|"]
    for inventory in INVENTORIES:
        b = c["inventories"][inventory]
        for name in CONTRAST_ORDER:
            k = b["contrasts"][name]
            s2, sd = k["half_width_002"], k["half_width_design"]
            lines.append(f"| {inventory} | {label(b, name)} | {fmt_interval(k['estimate'])} | {hw(k['estimate'])} | "
                         f"{None if s2 is None else s2['pass']} | {None if sd is None else sd['pass']} | "
                         f"{k['design_required_mixed_members']} | {None if s2 is None else s2['levels_for_threshold_by_sqrt_scaling']} |")
    lines += ["", "### Per inventory", "",
              "| inventory | F* (cross-fit) | F_hindsight | C-F* | AUC J | AUC F* | top-1 J / F* / chance | "
              "fixed-pair spread | J modal pair, non-modal share | tied share |", "|---|---|---|---|---|---|---|---|---|---|"]
    for inventory in INVENTORIES:
        b = c["inventories"][inventory]
        gi, t = b["guard_inputs"], b["top1"]
        lines.append(f"| {inventory} | {b['F_star']} | {b['F_hindsight']} | {b['C_F_star']} | "
                     f"{fmt_interval(b['arm_auc']['J'])} | {fmt_interval(b['arm_auc'][b['F_star']])} | "
                     f"{t['J_hits']}/{t['cells']} / {t['F_star_hits']}/{t['cells']} / {fmt(t['chance_mean'])} | "
                     f"{fmt_interval(b['fixed_pair_spread'])} | {gi['J_modal_pair']}, {fmt(gi['J_nonmodal_step_share'])} | "
                     f"{fmt(gi['tied_share'])} |")
    lines += ["", "### Per-arm member-mean AUC (mixed cells)", "", "| arm | " + " | ".join(INVENTORIES) + " |",
              "|---|" + "---|" * len(INVENTORIES)]
    for arm in ARMS:
        lines.append(f"| {arm} | " + " | ".join(fmt(c["inventories"][i]["arm_member_mean_auc"][arm]) for i in INVENTORIES) + " |")
    lines.append("")


def findings_md(plan, compute):
    lines = ["# Issue-104 work item 3: the #96 contrast set on the held-out #109 evaluation split — findings", "",
             f"- identity `{IDENTITY}`, plan v{plan['version']} frozen {plan['frozen_at']} before any evaluation score",
             f"- validation command: `{VALIDATION_COMMAND}`; zero engine seconds; endpoint 225; every interval "
             "DESCRIPTIVE (level-clustered, 10000 draws, PCG64 7201)",
             f"- stack: {plan['stack']['predictor']}; {plan['stack']['controller']}; anchors: {plan['stack']['anchors']}",
             "- cells (Gate-B, mixed only): " + ", ".join(f"{i} {compute['structure']['mixed_levels'][i]}" for i in INVENTORIES)
             + f" levels x 3 seeds; {compute['structure']['records']} records, "
               f"{compute['structure']['candidate_rollouts']} candidate rollouts",
             f"- primary cost: {PRIMARY_COST} (pre-declared); secondary: {SECONDARY_COST} (#96's cost)",
             f"- #96 reference (development, count cost): theta_grid "
             f"{fmt_interval(compute['issue_96_reference']['primary']['estimate'])} over "
             f"{compute['issue_96_reference']['mixed_cells']['grid']} cells, "
             f"half-width {fmt(compute['issue_96_reference']['primary']['estimate']['half_width'])}",
             "- post-freeze amendment A1 (reporting / guard propagation, after scoring; registered in "
             "`scripts/post_freeze_amendments.py`): the frozen plan's guard_failure rule ('any failed guard turns "
             "every token of that cost into the stop token') is applied to the grid secondaries J - M and J - OL too; "
             "the frozen #96 secondary rule (own half-width in place of G6) and the #96 literal G6 bound are kept as "
             "DESCRIPTIVE readings beside the tokens; no record, estimate or interval changed", ""]
    for cost in (PRIMARY_COST, SECONDARY_COST):
        cost_section(compute, cost, lines)
    s = compute["secondary_101"]
    lines += ["## Secondary reading: the #112 Gate-B predictor's #96-style contrasts (#101, re-tabulated, not re-scored)", "",
              f"- source `{s['source']}`; {s['predictor']}; #101 guards {s['guards_101']}",
              f"- #96 contrasts not in the #101 publication: {', '.join(s['not_in_101'])}", "",
              "| inventory | contrast | estimate [95% DESCRIPTIVE] | half-width | <= 0.02 | <= design delta | #101 token |",
              "|---|---|---|---|---|---|---|"]
    for inventory in INVENTORIES:
        b = s["inventories"][inventory]
        for name, text in SECONDARY_101.items():
            r = b["contrasts"][name]
            second = f" [{r['second']}]" if name.endswith(("F*", "F_hindsight")) else ""
            lines.append(f"| {inventory} | {text}{second} | {fmt(r['mean'])} [{fmt(r['interval'][0])}, "
                         f"{fmt(r['interval'][1])}] | {fmt(r['half_width'])} | {r['half_width_002']['pass']} | "
                         f"{r['half_width_design']['pass']} | {r['token_101']} |")
    lines += ["", "## Compute", "", "| arm | records | candidate rollouts | transition calls | controller calls | "
              "linear MACs | GPU s |", "|---|---|---|---|---|---|---|"]
    for arm, a in compute["compute"]["per_arm"].items():
        lines.append(f"| {arm} | {a['records']} | {a['candidate_rollouts']} | {a['transition_calls']} | "
                     f"{a['controller_calls']} | {a['linear_macs']} | {a['gpu_seconds']:.1f} |")
    k = compute["compute"]
    lines += ["", f"- scoring: {k['run_gpu_seconds']:.0f} GPU s summed over workers, {k['process_wall_seconds']:.0f} s "
              f"process wall, {k['elapsed_wall_seconds']:.0f} s elapsed; smoke {k['smoke_wall_seconds']:.0f} s "
              f"(census of the 3186 #96 records {k['smoke_census_wall_seconds']:.0f} s); engine seconds 0", "",
              "## Claim boundary", "", plan["claim_boundary"], ""]
    return "\n".join(lines)


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["source", "cost", "inventory", "contrast", "first", "second", "cells", "levels", "mean", "lower",
                     "upper", "half_width", "half_width_le_0.02", "half_width_le_design_delta", "token"])
    for cost in COSTS:
        c = compute["costs"][cost]
        for inventory in INVENTORIES:
            b = c["inventories"][inventory]
            for name in CONTRAST_ORDER:
                k = b["contrasts"][name]
                e = k["estimate"]
                token = ""
                if inventory == PRIMARY and name == "J-F*":
                    token = c["primary"]["token"]
                elif inventory == PRIMARY and name in c["secondary_tokens"]:
                    token = c["secondary_tokens"][name]["token"]
                writer.writerow(["issue-104-faithful-77", cost, inventory, name, k["first"], k["second"], k["cells"],
                                 k["members"], *(["", "", "", ""] if e is None else
                                                 [f"{e['mean']:.6f}", f"{e['interval'][0]:.6f}",
                                                  f"{e['interval'][1]:.6f}", f"{e['half_width']:.6f}"]),
                                 "" if k["half_width_002"] is None else k["half_width_002"]["pass"],
                                 "" if k["half_width_design"] is None else k["half_width_design"]["pass"], token])
    for inventory, b in compute["secondary_101"]["inventories"].items():
        for name, r in b["contrasts"].items():
            writer.writerow(["issue-101-gate-b-predictor", PRIMARY_COST, inventory, name, r["first"], r["second"],
                             r["cells"], b["members"], f"{r['mean']:.6f}", f"{r['interval'][0]:.6f}",
                             f"{r['interval'][1]:.6f}", f"{r['half_width']:.6f}", r["half_width_002"]["pass"],
                             r["half_width_design"]["pass"], r["token_101"]])
    return buffer.getvalue()


def summary(plan, compute):
    def brief(cost):
        c = compute["costs"][cost]
        return {"role": c["role"], "guards_pass": c["guards_pass"], "failed_guards": c["failed_guards"],
                "primary": {"F_star": c["primary"]["F_star"], "estimate": c["primary"]["estimate"],
                            "token": c["primary"]["token"],
                            "reading_at_96_literal_bound": c["primary"]["reading_at_96_literal_bound"]},
                "secondary_tokens": {n: s["token"] for n, s in c["secondary_tokens"].items()},
                "secondary_readings_by_96_secondary_rule": {n: s["reading_by_96_secondary_rule"]
                                                            for n, s in c["secondary_tokens"].items()},
                "replication": f"{c['replication']['count']}/{c['replication']['of']}",
                "g6_per_inventory": {i: {k: c["inventories"][i]["g6"].get(k) for k in
                                         ("half_width", "pass", "pass_96_literal", "pass_design_delta", "mixed_levels",
                                          "levels_for_threshold_by_sqrt_scaling")} for i in INVENTORIES},
                "max_half_width_per_inventory": {i: max(k["estimate"]["half_width"] for k in
                                                        c["inventories"][i]["contrasts"].values() if k["estimate"])
                                                 for i in INVENTORIES}}
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "issue": 104, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "primary_cost": PRIMARY_COST,
            "disposition": compute["costs"][PRIMARY_COST]["primary"]["token"],
            "costs": {cost: brief(cost) for cost in COSTS},
            "secondary_101_g6": {i: b["g6"] for i, b in compute["secondary_101"]["inventories"].items()},
            "compute": {k: compute["compute"][k] for k in ("run_gpu_seconds", "process_wall_seconds",
                                                          "elapsed_wall_seconds", "engine_seconds")},
            "claim_boundary": plan["claim_boundary"]}


def rendered(plan, compute, output):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute), "comparisons.csv": comparisons_csv(compute),
            "records-manifest.json": json_text(record_manifest(output))}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    with harness.deterministic_scoring():
        compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        (output / name).write_text(text)
    c = compute["costs"][PRIMARY_COST]
    log(f"published: primary {fmt_interval(c['primary']['estimate'])} -> {c['primary']['token']}")
    return 0


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def rescore_check(output, plan, arm="J", seed=SEEDS[0]):
    """Re-score one record from scratch on every cell: rows and schedules must be bit-identical."""
    cells = load_cells(output)
    verify_checkpoints(plan)
    anchors = load_anchor_store()["anchors"]
    retained = read_record(output, arm, seed)
    stack, costs = Stack(), harness.EndpointCosts(wlc.load_objective())
    model, controller = stack.for_arm(arm, seed)
    cells_checked = 0
    for inventory in INVENTORIES:
        for level in sorted(cells[inventory]):
            rows, sequences, _ = score_cell(model, controller, anchors[level], cells[inventory][level]["inventory"],
                                            t96.arm_spec(arm), costs)
            block = retained["inventories"][inventory][level]
            schedules = ({str(r["ordinal"]): t96.rle(s) for r, s in zip(rows, sequences)}
                         if arm in CONTROLLER_ARMS else None)
            if json.loads(json_text(rows)) != block["rows"] or schedules != block["schedules"]:
                raise ValueError(f"re-scored {arm}/{seed} {inventory}/{level} differs from the retained record")
            cells_checked += 1
    return cells_checked


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    if read_json(cells_path(output)) != json.loads(json_text(derive_cells())):
        raise ValueError("cells.json differs from the Gate-B cell rule re-derivation")
    with harness.deterministic_scoring():
        checked = rescore_check(output, plan)
        compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute, output).items():
        if (output / name).read_text() != text:
            raise ValueError(f"{name} differs from the fresh recomputation")
    log(f"validation passed ({time.monotonic() - began:.0f}s): cells re-derived, J seed {SEEDS[0]} re-scored on "
        f"{checked} cells bit-identically, records manifest and every published file regenerated byte for byte")
    return 0


def dry_run(output):
    payload = derive_cells()
    for inventory in INVENTORIES:
        levels = payload["inventories"][inventory]
        log(f"{inventory}: {len(levels)} mixed levels, "
            f"{sum(len(e['inventory']) for e in levels.values())} candidates per seed")
    log(f"records to score: {len(ARMS) * len(SEEDS)} ({len(ARMS)} arms x {len(SEEDS)} seeds)")
    return 0


MODES = ("dry-run", "smoke", "prepare", "run", "frequencies", "publish", "validate")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        group.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.run:
            return run(args.output, args.phase, args.workers, args.worker)
        mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
        return {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "frequencies": frequencies_mode,
                "publish": publish, "validate": validate}[mode](args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"STOP: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
