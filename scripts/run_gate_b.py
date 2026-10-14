"""Issue-108 Gate B: the #112 v3 Gate-B unit on the #104 sealed evaluation split, scored once.

Binding runner module: scripts/run_gate_b.py
Exact validation command: python -u -m scripts.run_gate_b --validate

Gate B (#108) asks whether the frozen #112 v3 recipe, retrained on the #109 fit split captured by
the #104 campaign (``issue-112-v3-retrain-v1/gate_b_handoff.json``), ranks the candidate launches
of the sealed #109 evaluation split: pooled grid AUC >= 0.65 with lower bound > 0.55 under the
tie-free cost. This runner

1. builds the evaluation set from the #104 campaign stream ``issue109-evaluation`` only: per
   level the complete verdict-bearing candidates of each inventory in ordinal order with the
   engine pig-removal verdicts; mixed-verdict cells only. Anchors follow the training rule (E99
   batch-1 parse of shot frame 0 of the level's first complete candidate in suffix order;
   ``run_decision_dynamics_retrain.anchor_carrier`` / ``anchor_branches``);
2. smoke (pre-freeze, fit split only): the cell construction reproduces the retrain's ranking
   cells, the anchor rule reproduces its stored anchors, the scoring path reproduces the v1
   ``endpoint_block`` rows, and the statistics reproduce the v3 LOLO numbers;
3. freezes plan.json (handoff and checkpoint hashes, units, thresholds, bootstrap, disposition
   rule, the pre-declared unit - C* contrast, the controls) before any evaluation score;
4. scores once, in a process that did not train, under scoring_harness.deterministic_scoring():
   the unit (LC : F-15-macro, hybrid) and C* (LC : C-F-15, continuous), every usable seed, plus
   the no-model action prior (fit-verdicts.json) and the wrong-anchor control of the unit;
5. publishes compute.json, summary.json (top-level ``gate_b``: pass / fail) and findings.md.

Modes: --dry-run --smoke --prepare --score --publish --validate. Zero training, zero engine
seconds. Every interval is a member-clustered percentile bootstrap (10000 draws, PCG64 7201),
DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import functools
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_dynamics_versions as versions
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_lookahead_control as wlc
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-108-gate-b-v1"
RETRAIN = rt.OUTPUT
HANDOFF = RETRAIN / "gate_b_handoff.json"
FIT_VERDICTS = RETRAIN / "fit-verdicts.json"
CAMPAIGN = rt.CAMPAIGN
STREAM = "issue109-evaluation"

IDENTITY = "issue-108-gate-b-v1"
SCHEMA_PLAN = "issue_108_gate_b_plan_v1"
SCHEMA_SMOKE = "issue_108_gate_b_smoke_v1"
SCHEMA_RECORD = "issue_108_gate_b_record_v1"
SCHEMA_ANCHORS = "issue_108_gate_b_anchors_v1"
SCHEMA_COMPUTE = "issue_108_gate_b_compute_v1"
SCHEMA_REPORT = "issue_108_gate_b_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_gate_b --validate"
RUNNER = "scripts/run_gate_b.py"

DEVICE = "cuda"
ARM = rt.ARM
UNITS = {"unit": {"family": "hybrid", "request": rt.UNITS["hybrid"]},
         "c_star": {"family": "continuous", "request": rt.UNITS["continuous"]}}
GATE_INVENTORY = "grid"
DESCRIPTIVE_INVENTORIES = ("offset", "angle", "power")
INVENTORIES = (GATE_INVENTORY, *DESCRIPTIVE_INVENTORIES)
PRIMARY_COST = harness.PRIMARY_COST
TARGET_AUC = m99.TARGET_AUC
TARGET_LOWER = m99.TARGET_LOWER
DELTA = v1.DELTA
NONFINITE_SHARE_MAX = v1.NONFINITE_SHARE_MAX
TOLERANCE = harness.PARSE_TOLERANCE
STOP_TOKEN = v1.STOP_TOKEN
GPU_CAP_SECONDS = 2 * 3600.0
WALL_CAP_SECONDS = 4 * 3600.0
SMOKE_CELLS = 3

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-108-gate-b] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# membership: the #109 evaluation split as captured by the #104 campaign
# ---------------------------------------------------------------------------

def evaluation_levels():
    return [level for family in rt.families() for level in rt.cohort()["splits"]["evaluation"][family]]


def evaluation_candidates():
    return rt.cohort()["candidate_inventories"]["sets"]["evaluation"]["candidates"]


@functools.lru_cache(maxsize=None)
def stream_rows():
    """The campaign report rows of the evaluation stream only (other streams are dropped at load)."""
    return {row["identity"]: row for row in read_json(CAMPAIGN / "branches.json") if row["stream"] == STREAM}


def evaluation_branches():
    """Every scheduled evaluation branch in frozen order (families, roster order, candidate
    suffix order); levels outside the union subset carry the grid candidates only."""
    rows = stream_rows()
    out = []
    for level in evaluation_levels():
        for candidate in evaluation_candidates():
            key = f"{level['identity']}-{candidate['suffix']}"
            row = rows.get(key)
            if row is None:
                continue
            out.append({"key": key, "member": level["identity"], "family": level["generator_family"],
                        "candidate": candidate, "status": row["status"], "failure_class": row["failure_class"],
                        "attempt": str(CAMPAIGN / STREAM / "attempts" / key)})
    if len(out) != len(rows):
        raise ValueError("the evaluation-stream rows differ from the #109 evaluation roster x evaluation candidates")
    return out


def complete_branches(branches):
    return [b for b in branches if b["status"] == "complete"]


def evaluation_verdicts():
    """Engine pig-removal verdict of every complete evaluation branch (#104 report, #85 channel)."""
    return {key: bool(row["pig_removed"]) for key, row in stream_rows().items() if row["status"] == "complete"}


def inventory_entries(levels, branches, verdicts, inventory):
    """Per level with the inventory: its verdict-bearing candidates (complete branches, i.e. keys
    in ``verdicts``) in inventory-ordinal order, and their verdicts by ordinal. The cell rule of
    ``run_decision_dynamics_retrain.ranking_cells`` without the padding."""
    by_level = {}
    for branch in branches:
        by_level.setdefault(branch["member"], []).append(branch)
    out = {}
    for level in levels:
        items = sorted(({"ordinal": entry["ordinal"], "action": branch["candidate"]["action"], "key": branch["key"]}
                        for branch in by_level.get(level["identity"], []) if branch["key"] in verdicts
                        for entry in branch["candidate"]["inventory_entries"] if entry["inventory"] == inventory),
                       key=lambda item: item["ordinal"])
        if items:
            out[level["identity"]] = {"inventory": items,
                                      "verdicts": {item["ordinal"]: verdicts[item["key"]] for item in items}}
    return out


def mixed_cells(entries):
    return {member: entry for member, entry in entries.items() if t96.mixed(entry)}


@functools.lru_cache(maxsize=None)
def evaluation_set():
    """inventory -> {all: level entries, mixed: mixed-verdict cells}; plus branch accounting."""
    branches = evaluation_branches()
    verdicts = evaluation_verdicts()
    levels = evaluation_levels()
    cells = {}
    for inventory in INVENTORIES:
        entries = inventory_entries(levels, branches, verdicts, inventory)
        cells[inventory] = {"all": entries, "mixed": mixed_cells(entries)}
    return branches, cells


def inventory_items(inventory):
    """The inventory's (ordinal, action) items of the frozen evaluation candidate set."""
    return sorted(({"ordinal": entry["ordinal"], "action": candidate["action"]} for candidate in evaluation_candidates()
                   for entry in candidate["inventory_entries"] if entry["inventory"] == inventory),
                  key=lambda item: item["ordinal"])


# ---------------------------------------------------------------------------
# handoff, checkpoints, anchors
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def handoff():
    return read_json(HANDOFF)


def checkpoints(name):
    return handoff()[name]["checkpoints"]


def seeds():
    """Seeds of the handoff checkpoints (union of both units, in order)."""
    return sorted({c["seed"] for name in UNITS for c in checkpoints(name)})


def member_path(name, seed):
    return rt.implementation().member_path(RETRAIN, ARM, UNITS[name]["family"], "all", seed)


def load_model(name, seed):
    """The handoff checkpoint, or a typed failure: unusable in the handoff, hash differs, retired."""
    entry = next((c for c in checkpoints(name) if c["seed"] == seed), None)
    if entry is None or not entry["usable"]:
        return None, "unusable_in_handoff"
    if sha256_of(member_path(name, seed)) != entry["sha256"]:
        raise ValueError(f"{name} seed {seed}: checkpoint differs from the handoff hash")
    model = rt.implementation().load_member(RETRAIN, ARM, UNITS[name]["family"], "all", seed)
    return (None, "member_retired_nonfinite") if model is None else (model, None)


def fit_anchors():
    return torch.load(rt.anchors_path(RETRAIN), map_location="cpu", weights_only=True)["anchors"]


def anchors_path(output):
    return Path(output) / "data" / "anchors-evaluation.pt"


def record_path(output, name, seed):
    return Path(output) / "records" / f"{name}--seed{seed}.json"


def ledger_path(output):
    return Path(output) / "ledger.json"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_path(output)
    ledger = read_json(path) if path.is_file() else {"phases": {}}
    block = ledger["phases"].get(phase, {"wall_seconds": 0.0, "gpu_seconds": 0.0})
    ledger["phases"][phase] = {**block, "wall_seconds": block["wall_seconds"] + wall,
                               "gpu_seconds": block["gpu_seconds"] + gpu, "last_at": utc_now(), **extra}
    write_json(path, ledger)
    if (sum(b["gpu_seconds"] for b in ledger["phases"].values()) > GPU_CAP_SECONDS
            or sum(b["wall_seconds"] for b in ledger["phases"].values()) > WALL_CAP_SECONDS):
        raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")


# ---------------------------------------------------------------------------
# scoring (always inside deterministic_scoring)
# ---------------------------------------------------------------------------

def request_block(model, request, z0, items, costs):
    """v1.endpoint_block for one request: m99.rollout_endpoints from z0 over the items' actions,
    EndpointCosts rows against z0."""
    actions = m99.action_batch(items)
    z_end, timing = m99.rollout_endpoints(model, None, z0.to(DEVICE), actions, t96.arm_spec(request))
    return {"rows": [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)],
            "candidates": len(items), **timing}


def score_record(cells, anchors, wrong_anchors, costs, name, seed, policy):
    """Rows of every mixed cell of every inventory from its own anchor; for the unit also the
    grid candidates from every fit-split anchor (wrong-anchor control)."""
    request = UNITS[name]["request"]
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy, "scored_at": utc_now(),
            "cell": {"name": name, "arm": ARM, **UNITS[name], "seed": seed}, "engine_seconds": 0}
    began = time.monotonic()
    model, failure = load_model(name, seed)
    if model is None:
        return {**base, "typed_failure": failure, "inventories": {}, "wrong_anchor": None}
    inventories = {inventory: {member: request_block(model, request, anchors[member], entry["inventory"], costs)
                               for member, entry in sorted(cells[inventory]["mixed"].items())}
                   for inventory in INVENTORIES}
    wrong = None
    if name == "unit":
        items = inventory_items(GATE_INVENTORY)
        wrong = {member: request_block(model, request, wrong_anchors[member], items, costs)
                 for member in sorted(wrong_anchors)}
    torch.cuda.synchronize()
    return {**base, "typed_failure": None, "checkpoint_sha256": sha256_of(member_path(name, seed)),
            "inventories": inventories, "wrong_anchor": wrong, "seconds": time.monotonic() - began}


# ---------------------------------------------------------------------------
# no-model action prior (fit-verdicts.json)
# ---------------------------------------------------------------------------

def action_key(action):
    return (action["drag_x"], action["drag_y"], action["release_time_ms"], action["tap_time_ms"])


def prior_stats(fit_branches, inventory):
    """Engine success counts of each action over the fit split's executions in the inventory."""
    stats = {}
    for branch in fit_branches:
        if any(e["inventory"] == inventory for e in branch["inventory_entries"]):
            s = stats.setdefault(action_key(branch["action"]), [0, 0])
            s[0] += int(branch["pig_removed"])
            s[1] += 1
    return stats


def prior_rows(entry, stats):
    """c2.prior_tables (held-out cohort): cost = minus the fit success frequency of the identical
    action; an action never executed in the fit split costs 0."""
    return [{"ordinal": item["ordinal"], "excluded": False, "count": 0.0,
             PRIMARY_COST: -(stats[action_key(item["action"])][0] / stats[action_key(item["action"])][1])
             if action_key(item["action"]) in stats else 0.0} for item in entry["inventory"]]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def view_rows(records, cells, inventory, stats, seed_list):
    """One row per (member, seed): views 'unit' and 'c_star' of that seed, 'prior' (seed-free;
    the same value per seed keeps the v1 cell weighting, as c2.prior_tables)."""
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, prior_rows(entry, stats), PRIMARY_COST)
        for seed in seed_list:
            views = {}
            for name in UNITS:
                record = records[(name, seed)]
                found = None if record["typed_failure"] else record["inventories"][inventory][member]["rows"]
                views[name] = v1.cost_view(entry, found, PRIMARY_COST)
            views["prior"] = prior
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def ranking_tables(rows, seed_list, names=("unit", "c_star", "prior")):
    """Pooled (cells = (member, seed), bootstrap over members) and per-seed AUC of each named
    ranker, the pre-declared unit - C* contrast and the unit - prior contrast."""
    out = {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": {}, "contrasts": {}}
    for name in names:
        out["rankers"][name] = {**v1.ranker_block(rows, name),
                                "per_seed": {str(s): m99.block_of([r for r in rows if r["seed"] == s], name)
                                             for s in seed_list} if name != "prior" else None}
    for first, second in (("unit", "c_star"), ("unit", "prior")):
        if first in names and second in names:
            block = m99.paired(rows, [(1, first), (-1, second)])
            out["contrasts"][f"{first}-{second}"] = {**block, "token": c2.token(block["estimate"])}
    return out


def anchor_rows(records, cells, seed_list):
    """Wrong-anchor control of the unit (v2 held-out definition): per (cell, seed) the right-anchor
    AUC, the mean AUC from every fit-split anchor, and the mean Spearman of the candidate costs."""
    rows = []
    for member, entry in sorted(cells[GATE_INVENTORY]["mixed"].items()):
        for seed in seed_list:
            record = records[("unit", seed)]
            if record["typed_failure"]:
                rows.append({"member": member, "seed": seed, "views": {"unit": None}})
                continue
            right = record["inventories"][GATE_INVENTORY][member]["rows"]
            wrong = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
            wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
            rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong) if r is not None]
            rows.append({"member": member, "seed": seed, "views": {"unit": {
                "right": c2.auc(entry, right), "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                "stability": float(np.mean(rhos)) if rhos else None}}})
    return rows


def anchor_table(rows, wrong_anchors):
    block = c2.difference_block(rows, [(1, "unit", "right"), (-1, "unit", "wrong")])
    return {"cells": len(rows), "wrong_anchors": wrong_anchors,
            "right": c2.field_block(rows, "unit", "right"), "wrong": c2.field_block(rows, "unit", "wrong"),
            "right_minus_wrong": block, "stability": c2.field_block(rows, "unit", "stability"),
            "token": c2.token(block)}


def gate_decision(block, nonfinite_share):
    """pass iff the pooled unit grid AUC >= 0.65 with lower bound > 0.55 and at most 10 % of the
    unit's verdict slots nonfinite (v1 G4); otherwise fail."""
    ok = v1.meets(block) and (nonfinite_share or 0.0) <= NONFINITE_SHARE_MAX
    return "pass" if ok else "fail"


def load_records(output):
    records = {}
    for name in UNITS:
        for seed in seeds():
            record = read_json(record_path(output, name, seed))
            if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY:
                raise ValueError(f"record {name}/{seed} binding differs")
            records[(name, seed)] = record
    return records


def evaluation_accounting(branches, cells):
    complete = complete_branches(branches)
    return {"levels": len(evaluation_levels()), "branches": len(branches), "complete": len(complete),
            "typed_failures": [{"key": b["key"], "failure_class": b["failure_class"]}
                               for b in branches if b["status"] != "complete"],
            "incomplete_branches": [b["key"] for b in branches if b["status"] not in ("complete", "failed")],
            "levels_per_inventory": {i: len(cells[i]["all"]) for i in INVENTORIES},
            "mixed_cells": {i: len(cells[i]["mixed"]) for i in INVENTORIES},
            "verdict_slots_mixed": {i: sum(len(e["verdicts"]) for e in cells[i]["mixed"].values()) for i in INVENTORIES}}


def compute_tables(output, plan):
    output = Path(output)
    branches, cells = evaluation_set()
    records = load_records(output)
    seed_list = seeds()
    fit = read_json(FIT_VERDICTS)["branches"]
    stats = {i: prior_stats(fit, i) for i in INVENTORIES}
    tables = {i: ranking_tables(view_rows(records, cells, i, stats[i], seed_list), seed_list) for i in INVENTORIES}
    overlap = {i: float(np.mean([action_key(item["action"]) in stats[i] for item in inventory_items(i)]))
               for i in INVENTORIES}
    wrong_pool = sorted(next((r["wrong_anchor"] for (n, _), r in records.items() if n == "unit" and r["wrong_anchor"]),
                             {}))
    anchors = anchor_table(anchor_rows(records, cells, seed_list), len(wrong_pool))
    anchor_store = torch.load(anchors_path(output), map_location="cpu", weights_only=True)
    roster = {level["identity"] for level in evaluation_levels()}
    fit_roster = {level["identity"] for level in rt.fit_levels()}
    scored_at = sorted(r["scored_at"] for r in records.values())
    hashes = {f"{n}/{s}": r.get("checkpoint_sha256") == next(c["sha256"] for c in checkpoints(n) if c["seed"] == s)
              for (n, s), r in records.items() if not r["typed_failure"]}
    guards = {
        "G1_isolation": {"pass": bool(all(Path(b["attempt"]).parent.parent.name == STREAM and b["member"] in roster
                                          for b in branches)
                                      and all(set(cells[i]["mixed"]) <= roster for i in INVENTORIES)
                                      and set(wrong_pool) <= fit_roster)},
        "G2_scoring_policy": {"records": len(records), "anchor_store": True,
                              "pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in records.values())
                              and anchor_store["scoring_policy"] == harness.SCORING_POLICY},
        "G3_chronology": {"plan_frozen_at": plan["frozen_at"], "anchors_parsed_at": anchor_store["parsed_at"],
                          "first_record_scored_at": scored_at[0],
                          "pass": plan["frozen_at"] <= anchor_store["parsed_at"] <= scored_at[0]},
        "G4_checkpoint_binding": {"records": hashes, "pass": all(hashes.values())},
    }
    if not all(g["pass"] for g in guards.values()):
        raise ValueError(f"a Gate-B guard failed: {guards}")
    grid = tables[GATE_INVENTORY]
    unit = grid["rankers"]["unit"]
    decision = gate_decision(unit["auc"], unit["nonfinite_share"])
    ledger = read_json(ledger_path(output))
    smoke = read_json(output / "smoke.json")
    phases = {"smoke": {"wall_seconds": smoke["wall_seconds"], "gpu_seconds": smoke["wall_seconds"]},
              **ledger["phases"]}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "units": {n: {"arm": ARM, **UNITS[n], "seeds": seed_list,
                      "typed_failures": {str(s): records[(n, s)]["typed_failure"] for s in seed_list}} for n in UNITS},
        "evaluation": evaluation_accounting(branches, cells),
        "guards": guards,
        "gate_b": {"decision": decision, "rule": plan["decision_rule"]["gate_b"], "pooled_grid_auc": unit["auc"],
                   "per_seed_grid_auc": unit["per_seed"], "nonfinite_share": unit["nonfinite_share"],
                   "target_auc": TARGET_AUC, "target_lower": TARGET_LOWER},
        "contrast_unit_minus_c_star": {**grid["contrasts"]["unit-c_star"], "label": "DESCRIPTIVE (pre-declared)"},
        "controls": {"action_prior": {"grid_auc": grid["rankers"]["prior"]["auc"], "action_overlap": overlap,
                                      "unit_minus_prior": grid["contrasts"]["unit-prior"]},
                     "wrong_anchor": anchors},
        "inventories": tables,
        "descriptive_inventories": list(DESCRIPTIVE_INVENTORIES),
        "compute": {"phases": phases, "gpu_seconds": sum(b["gpu_seconds"] for b in phases.values()),
                    "wall_seconds": sum(b["wall_seconds"] for b in phases.values()), "engine_seconds": 0,
                    "hardware": "1 x RTX 5090 (one scoring process)"},
    }


# ---------------------------------------------------------------------------
# renderings
# ---------------------------------------------------------------------------

def cell(block):
    return fmt_interval(block)


def num(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def findings_md(plan, compute):
    lines = []
    add = lines.append
    gate = compute["gate_b"]
    ev = compute["evaluation"]
    grid = compute["inventories"][GATE_INVENTORY]
    add("# Issue #108 Gate B: the #112 v3 unit on the #104 sealed evaluation split")
    add("")
    add(f"Plan frozen {plan['frozen_at']} after a passed smoke, before any evaluation score; scored once in a process "
        f"that did not train, under `deterministic_scoring()`. Validation: `{VALIDATION_COMMAND}`.")
    add("")
    add(f"**Gate B: {gate['decision']}.** Unit LC : {UNITS['unit']['request']} (hybrid, E99), pooled grid AUC "
        f"**{cell(gate['pooled_grid_auc'])}** over {grid['cells']} cells ({grid['members']} levels x "
        f"{len(compute['units']['unit']['seeds'])} seeds); target >= {TARGET_AUC} with lower bound > {TARGET_LOWER}.")
    add("")
    add("## Evaluation set (stream `issue109-evaluation`)")
    add("")
    add(f"- {ev['levels']} levels, {ev['branches']} branches, {ev['complete']} complete; typed failures "
        f"{len(ev['typed_failures'])}, incomplete branches {len(ev['incomplete_branches'])}.")
    add("- Mixed-verdict cells (levels with the inventory): "
        + ", ".join(f"{i} {ev['mixed_cells'][i]} ({ev['levels_per_inventory'][i]})" for i in INVENTORIES) + ".")
    add("- Anchors: E99 batch-1 parse of shot frame 0 of each level's first complete candidate in suffix order "
        "(the training rule).")
    add("")
    add("## Grid (Gate B)")
    add("")
    add("| ranker | pooled AUC | " + " | ".join(f"seed {s}" for s in compute["units"]["unit"]["seeds"])
        + " | tied pair share | nonfinite share | typed failures |")
    add("| --- | --- | " + " | ".join("---" for _ in compute["units"]["unit"]["seeds"]) + " | --- | --- | --- |")
    labels = {"unit": f"unit LC : {UNITS['unit']['request']}", "c_star": f"C* LC : {UNITS['c_star']['request']}",
              "prior": "no-model action prior"}
    for name, block in grid["rankers"].items():
        per_seed = block["per_seed"] or {}
        add(f"| {labels[name]} | {cell(block['auc'])} | "
            + " | ".join(cell(per_seed.get(str(s))) if per_seed else "seed-free"
                         for s in compute["units"]["unit"]["seeds"])
            + f" | {num(block['tied_pair_share'])} | {num(block['nonfinite_share'])} | {block['typed_failures']} |")
    add("")
    contrast = compute["contrast_unit_minus_c_star"]
    add(f"- Pre-declared contrast unit − C* (paired by level and seed): {cell(contrast['estimate'])} over "
        f"{contrast['cells']} cells, {contrast['members']} levels; {contrast['token']}.")
    prior = compute["controls"]["action_prior"]
    add(f"- Unit − no-model action prior (paired): {cell(prior['unit_minus_prior']['estimate'])}; "
        f"{prior['unit_minus_prior']['token']}. The prior ranks by each grid action's fit-split success frequency "
        f"(action overlap {num(prior['action_overlap'][GATE_INVENTORY], 2)}).")
    wrong = compute["controls"]["wrong_anchor"]
    add(f"- Wrong-anchor control of the unit ({wrong['wrong_anchors']} fit-split anchors per cell): right "
        f"{cell(wrong['right'])}, wrong {cell(wrong['wrong'])}, right − wrong {cell(wrong['right_minus_wrong'])} "
        f"({wrong['token']}); rank stability (Spearman right vs wrong) {cell(wrong['stability'])}.")
    add("")
    add("## Other inventories (DESCRIPTIVE; not part of the gate)")
    add("")
    add("| inventory | cells | unit | C* | action prior (overlap) | unit − C* |")
    add("| --- | --- | --- | --- | --- | --- |")
    for inventory in DESCRIPTIVE_INVENTORIES:
        t = compute["inventories"][inventory]
        add(f"| {inventory} | {t['cells']} | {cell(t['rankers']['unit']['auc'])} | "
            f"{cell(t['rankers']['c_star']['auc'])} | {cell(t['rankers']['prior']['auc'])} "
            f"({num(prior['action_overlap'][inventory], 2)}) | {cell(t['contrasts']['unit-c_star']['estimate'])} |")
    add("")
    add("## Guards")
    add("")
    for name, guard in compute["guards"].items():
        add(f"- {name}: {guard['pass']}")
    add("")
    c = compute["compute"]
    add(f"Compute: {c['gpu_seconds']:.0f} GPU s, {c['wall_seconds']:.0f} wall s ({c['hardware']}); 0 engine seconds; "
        "0 training.")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    grid = compute["inventories"][GATE_INVENTORY]
    controls = compute["controls"]
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND,
            "gate_b": compute["gate_b"]["decision"],
            "gate_b_rule": plan["decision_rule"]["gate_b"],
            "unit": {"arm": ARM, **UNITS["unit"], "encoder": "E99"},
            "c_star": {"arm": ARM, **UNITS["c_star"]},
            "pooled_grid_auc": compute["gate_b"]["pooled_grid_auc"],
            "per_seed_grid_auc": compute["gate_b"]["per_seed_grid_auc"],
            "c_star_pooled_grid_auc": grid["rankers"]["c_star"]["auc"],
            "c_star_per_seed_grid_auc": grid["rankers"]["c_star"]["per_seed"],
            "contrast_unit_minus_c_star": compute["contrast_unit_minus_c_star"],
            "controls": {"action_prior_grid_auc": controls["action_prior"]["grid_auc"],
                         "unit_minus_action_prior": controls["action_prior"]["unit_minus_prior"],
                         "wrong_anchor": {k: controls["wrong_anchor"][k]
                                          for k in ("right", "wrong", "right_minus_wrong", "stability", "token",
                                                    "wrong_anchors")}},
            "cells": {"grid": grid["cells"], "grid_levels": grid["members"],
                      "mixed_cells": compute["evaluation"]["mixed_cells"]},
            "typed_failures": {"capture": compute["evaluation"]["typed_failures"],
                               "incomplete_branches": compute["evaluation"]["incomplete_branches"],
                               "checkpoints": {n: compute["units"][n]["typed_failures"] for n in UNITS},
                               "nonfinite_share_unit_grid": compute["gate_b"]["nonfinite_share"]},
            "guards": {k: v["pass"] for k, v in compute["guards"].items()},
            "compute": {k: compute["compute"][k] for k in ("gpu_seconds", "wall_seconds", "engine_seconds")},
            "claim_boundary": plan["claim_boundary"]}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute)}


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit split and v3 records only; no evaluation score)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    began = time.monotonic()
    checks = []
    impl = rt.implementation()

    # 1. cell construction reproduces the retrain's ranking cells (fit stream, grid and angle)
    fit_branches = rt.complete_branches()
    fit_cells = {(m, i): e for i in rt.FIT_INVENTORIES
                 for m, e in mixed_cells(inventory_entries(rt.fit_levels(), fit_branches, rt.fit_verdicts(), i)).items()}
    payload = rt.load_training_data(RETRAIN)
    stored = fit_anchors()
    same = len(fit_cells) == len(payload["cell_members"])
    for k, (member, inventory) in enumerate(zip(payload["cell_members"], payload["cell_inventories"])):
        entry = fit_cells.get((member, inventory))
        if entry is None:
            same = False
            break
        n = len(entry["inventory"])
        same &= [it["ordinal"] for it in entry["inventory"]] == payload["cell_ordinals"][k]
        same &= payload["cell_verdict"][k, :n].tolist() == [int(entry["verdicts"][it["ordinal"]])
                                                             for it in entry["inventory"]]
        same &= bool(torch.equal(payload["cell_action"][k, :n], m99.action_batch(entry["inventory"]).cpu()))
        same &= bool(torch.equal(payload["cell_z0"][k], stored[member]))
    checks.append({"control": "cells_reproduce_retrain_ranking_cells", "cells": len(fit_cells),
                   "per_inventory": dict(Counter(i for _, i in fit_cells)), "ok": bool(same)})
    del payload

    # 2. the anchor rule reproduces the retrain's stored anchors (two fit levels)
    first = rt.anchor_branches(fit_branches)
    probe = [fit_branches[0]["member"], fit_branches[-1]["member"]]
    with harness.deterministic_scoring():
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        fresh = {m: rt.anchor_carrier(adapter, first[m]["attempt"]) for m in probe}
    checks.append({"control": "anchor_rule_reproduces_retrain_anchors", "levels": probe,
                   "max_abs_delta": max(float((fresh[m] - stored[m]).abs().max()) for m in probe),
                   "ok": all(torch.equal(fresh[m], stored[m]) for m in probe)})
    del adapter

    # 3. the scoring path reproduces v1 endpoint_block rows on fit-split grid cells
    costs = harness.EndpointCosts(wlc.load_objective())
    grid = sorted((m, e) for (m, i), e in fit_cells.items() if i == GATE_INVENTORY)[:SMOKE_CELLS]
    seed = seeds()[0]
    timings = {}
    with harness.deterministic_scoring():
        for name, unit in UNITS.items():
            model, failure = load_model(name, seed)
            if model is None:
                raise ValueError(f"smoke model {name}/{seed}: {failure}")
            equal = True
            t0 = time.monotonic()
            for member, entry in grid:
                mine = request_block(model, unit["request"], stored[member], entry["inventory"], costs)
                again = request_block(model, unit["request"], stored[member], entry["inventory"], costs)
                reference = impl.endpoint_block(model, unit["family"], stored[member], entry["inventory"], costs)
                equal &= mine["rows"] == reference[unit["request"]]["rows"] == again["rows"]
            timings[f"{name}_seconds_per_cell"] = (time.monotonic() - t0) / (3 * len(grid))
            checks.append({"control": f"{name}_scoring_path_equals_v1_endpoint_block", "seed": seed,
                           "cells": [m for m, _ in grid], "ok": bool(equal)})
            del model
            torch.cuda.empty_cache()

    # 4. the statistics reproduce the v3 LOLO numbers (pooled unit AUC, per-seed pooling, unit - C*)
    v3 = versions.implementation(3)
    _, _, universe = v3.load_universe()
    v3_rows = v3.lolo_rows(v3.OUTPUT, universe, GATE_INVENTORY)
    mapped = [{"member": r["member"], "seed": r["seed"],
               "views": {n: r["views"][(PRIMARY_COST, ARM, UNITS[n]["request"])] for n in UNITS}} for r in v3_rows]
    ours = ranking_tables(mapped, v3.SEEDS, names=tuple(UNITS))
    v3_compute = read_json(v3.OUTPUT / "compute.json")
    checks.append({"control": "statistics_reproduce_v3_lolo",
                   "unit_auc": ours["rankers"]["unit"]["auc"],
                   "unit_minus_c_star": ours["contrasts"]["unit-c_star"]["estimate"],
                   "ok": (ours["rankers"]["unit"]["auc"] == v3_compute["Q3_selected_target"]["lolo_grid_auc"]
                          and {k: v for k, v in ours["contrasts"]["unit-c_star"].items() if k != "token"}
                          == v3_compute["baseline_contrast"]["grid"])})

    # 5. evaluation inventory (no model, no anchor): accounting, isolation, shared grid actions
    branches, cells = evaluation_set()
    accounting = evaluation_accounting(branches, cells)
    grid_actions = {it["ordinal"]: it["action"] for it in inventory_items(GATE_INVENTORY)}
    fit_grid = {e["ordinal"]: c["action"] for c in rt.candidates() for e in c["inventory_entries"]
                if e["inventory"] == GATE_INVENTORY}
    shared = all(it["action"] == grid_actions[it["ordinal"]]
                 for e in cells[GATE_INVENTORY]["all"].values() for it in e["inventory"])
    checks.append({"control": "grid_actions_shared_and_equal_fit_grid", "ordinals": len(grid_actions),
                   "ok": bool(shared and grid_actions == fit_grid and len(grid_actions) == 16)})
    roster = {level["identity"] for level in evaluation_levels()}
    attempts = all(Path(b["attempt"]).is_dir() for b in complete_branches(branches))
    checks.append({"control": "evaluation_stream_isolation", "branches": len(branches),
                   "ok": bool(attempts and all(b["member"] in roster for b in branches)
                              and not (roster & {level["identity"] for level in rt.fit_levels()}))})
    checks.append({"control": "handoff_checkpoints_bound", "token": handoff()["token"],
                   "ok": bool(handoff()["token"] == "supported"
                              and all(sha256_of(member_path(n, c["seed"])) == c["sha256"]
                                      for n in UNITS for c in checkpoints(n))
                              and all(handoff()[n]["request"] == UNITS[n]["request"]
                                      and handoff()[n]["family"] == UNITS[n]["family"] for n in UNITS))})
    ok = all(c["ok"] for c in checks)
    wall = time.monotonic() - began
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "evaluation_accounting": accounting, "timings": timings, "wall_seconds": wall,
                "reading": ("fit split and v3 records only; no evaluation-split anchor parse or model score; the "
                            "evaluation accounting reads engine verdicts (mixed-cell counts, published by #104)"),
                "ok": ok}
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
        "issue_112_retrain_handoff": HANDOFF, "issue_112_retrain_plan": RETRAIN / "plan.json",
        "issue_112_retrain_compute": RETRAIN / "compute.json", "issue_112_fit_verdicts": FIT_VERDICTS,
        "issue_112_fit_anchors": rt.anchors_path(RETRAIN),
        "issue_112_v3_compute": versions.implementation(3).OUTPUT / "compute.json",
        "issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_versions_runner": ROOT / versions.RUNNER,
        "issue_112_controls_runner": ROOT / c2.RUNNER, "issue_112_retrain_runner": ROOT / rt.RUNNER,
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
        "issue_109_cohort_plan": rt.COHORT, "issue_104_plan": rt.POWER_PLAN,
        "issue_104_campaign_branches": CAMPAIGN / "branches.json",
        "issue_104_campaign_summary": CAMPAIGN / "summary.json",
        "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
        "smoke": Path(output) / "smoke.json",
    }
    names.update({f"checkpoint_{n}_{c['seed']}": member_path(n, c["seed"]) for n in UNITS for c in checkpoints(n)})
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence):
    accounting = smoke_evidence["evaluation_accounting"]
    h = handoff()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 108, "related_issues": [104, 109, 112],
        "role": "Gate B: score the frozen #112 v3 Gate-B unit once on the #104 sealed #109 evaluation split",
        "frozen_at": frozen_at, "frozen_before_evaluation_scoring": True, "issue_64_authorized": False,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "engine_seconds": 0, "training": "none",
        "question": ("does the #112 v3 Gate-B unit, retrained on the #109 fit split, rank the candidate launches of "
                     "the sealed #109 evaluation split: pooled grid AUC >= 0.65 with lower bound > 0.55 (tie-free)?"),
        "handoff": {"path": str(HANDOFF.relative_to(ROOT)), "sha256": sha256_of(HANDOFF), "token": h["token"],
                    "plan_identity": h["plan_identity"], "supersedes": h["supersedes"]},
        "units": {n: {"arm": ARM, **UNITS[n], "encoder": "E99", "usable_seeds": h[n]["usable_seeds"],
                      "checkpoints": [{"seed": c["seed"], "path": c["path"], "sha256": c["sha256"],
                                       "usable": c["usable"]} for c in checkpoints(n)],
                      "role": ("the Gate-B unit (hybrid; #112 v3 selection)" if n == "unit" else
                               "C*: the pre-declared matched baseline, the continuous-family unit with the largest "
                               "v3 selection score S (never a Gate-B candidate)")} for n in UNITS},
        "evaluation": {
            "split": ("#109 evaluation split (issue-109-sealed-cohort-v1 splits.evaluation, 200 levels per family, "
                      f"families {rt.families()}), captured by the #104 campaign stream {STREAM}; never fitted"),
            "accounting": accounting,
            "inventories": {GATE_INVENTORY: "Gate B (16 candidates, every evaluation level)",
                            **{i: "DESCRIPTIVE only (the 120 levels captured with all 60 candidates)"
                               for i in DESCRIPTIVE_INVENTORIES}},
            "cells": ("per level x inventory: its complete verdict-bearing candidates in inventory-ordinal order "
                      "(run_decision_dynamics_retrain.ranking_cells rule); mixed-verdict cells only (t96.mixed)"),
            "verdicts": (f"engine pig-removal verdict of every complete {STREAM} branch from the #104 campaign "
                         "report (branches.json, #85 channel); rows of other streams are dropped at load"),
            "anchors": ("per level: E99 batch-1 parse (wlc.encode_carrier) of shot frame 0 of its first complete "
                        "candidate in suffix order (run_decision_dynamics_retrain.anchor_branches / anchor_carrier), "
                        "under deterministic_scoring(), parsed after the freeze"),
            "typed_failures": ("capture typed failures and incomplete branches are recorded and carry no verdict; "
                               "never retried or replaced"),
            "excluded": {"issue109-policy": "controller training split; never read",
                         "issue110-*": "not part of Gate B; never read",
                         "issue109-fit": ("read only as the frozen fit-verdicts.json (action prior) and the retrain's "
                                          "fit anchors (wrong-anchor pool)")},
        },
        "scoring": {
            "operation": ("per cell and checkpoint: m99.rollout_endpoints(model, None, z0, actions, "
                          "t96.arm_spec(request)) from the level anchor to 225 over the cell's candidate actions "
                          "(v1 endpoint_block for the unit's request only)"),
            "cost": "tie_free (scoring_harness.EndpointCosts.row against the anchor used)",
            "cell_auc": ("m99.view: P(a pig-removing candidate has lower cost than a failing one), ties 0.5; "
                         "nonfinite endpoints are excluded rows (typed failures) and counted"),
            "pooling": ("cells are (level, seed) over the usable seeds; the pooled AUC is the mean of per-level seed "
                        "means; per-seed AUCs use that seed's cells (v3 seed_aggregation)"),
            "bootstrap": {"kind": "member(level)-clustered percentile bootstrap", "draws": t96.BOOTSTRAP_DRAWS,
                          "generator": f"PCG64 {t96.BOOTSTRAP_SEED}", "quantiles": list(t96.INTERVAL_QUANTILES),
                          "label": "DESCRIPTIVE"},
            "policy": ("every anchor parse and rollout inside scoring_harness.deterministic_scoring(), in one "
                       "process that did not train; each checkpoint is hash-checked against the handoff before "
                       "loading"),
        },
        "decision_rule": {
            "gate_b": (f"pass iff the unit's pooled grid AUC >= {TARGET_AUC} and its lower bound > {TARGET_LOWER}, and "
                       f"at most {NONFINITE_SHARE_MAX:.0%} of the unit's grid verdict slots are nonfinite (v1 G4); "
                       "otherwise fail. Grid only."),
            "pre_declared_contrast": (f"unit - C* paired grid AUC per (level, seed); supported iff mean >= {DELTA} with "
                                      f"lower > 0; not_supported_by_this_experiment iff upper < {DELTA}; otherwise "
                                      f"{STOP_TOKEN}. Reported, not part of the gate."),
            "controls": ("reported, not part of the gate; the unit - prior contrast and the wrong-anchor right - wrong "
                         "use the same token rule"),
            "descriptive_inventories": "offset, angle, power: unit, C*, prior AUC and unit - C*; DESCRIPTIVE only",
        },
        "controls": {
            "action_prior": ("no model (c2.prior_tables, held-out cohort): cost = minus the engine success frequency of "
                             "the identical action over the fit split's executions in the same inventory "
                             "(issue-112-v3-retrain-v1/fit-verdicts.json); an action never executed there costs 0"),
            "wrong_anchor": ("the unit's checkpoints roll the shared grid candidates out from every fit-split anchor "
                             "(the retrain's anchors-fit.pt, the anchors the unit trained with; v2 held-out "
                             "definition); per cell: right-anchor AUC minus the mean wrong-anchor AUC, and the mean "
                             "Spearman of the candidate costs right vs wrong (rank stability)"),
            "unit_minus_prior": "paired grid AUC per (level, seed), the prior value repeated per seed",
        },
        "guards": {
            "G1_isolation": (f"every scored branch is a {STREAM} capture of an evaluation-roster level; every wrong "
                             "anchor is a fit-roster level"),
            "G2_scoring_policy": "every record and the anchor store carry the #111 SCORING_POLICY",
            "G3_chronology": "plan frozen_at <= anchors parsed_at <= first record scored_at",
            "G4_checkpoint_binding": "every scored checkpoint's sha256 equals the handoff",
            "failure": "a failed guard refuses publication (no gate value)",
            "validate": (f"--validate re-derives every table from the records and byte-compares the renderings; it "
                         f"re-scores the unit's first seed on every grid cell and re-parses one anchor "
                         f"(endpoint_replication / carrier_replication, tolerance {TOLERANCE})"),
        },
        "prohibited": ["training", "reading issue109-policy or issue110 captures", "re-scoring after publication",
                       "changing a unit, threshold, rule or control after the freeze",
                       "retrying or replacing a typed failure"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "disclosure": ("Known before the freeze: every #112 v1/v2/v3, #117 and retrain result, and the #104 campaign "
                       "accounting with its mixed-verdict yield per inventory on this split (grid 311, offset 76, "
                       "angle 78, power 47). No model output on any evaluation capture has been computed. Hardware: "
                       "RTX 5090 instead of the #85 RTX 3090. Inputs bind the current bytes after the SERVER_SETUP "
                       "path rewrite (the #104 plan.json differs from its committed bytes by that rewrite)."),
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("one look at the sealed #109 evaluation split for the frozen #112 v3 unit; grid ranking of "
                           "16 fixed candidates per level under the tie-free cost; DESCRIPTIVE intervals; no claim "
                           "about closed-loop play, other inventories (descriptive) or novelty cells"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-108 Gate-B protocol")
    if not plan.get("frozen_before_evaluation_scoring") or not plan["smoke_evidence"].get("ok"):
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
    for name in ("records", "data"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now(), evidence)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {plan['evaluation']['accounting']['mixed_cells']} mixed cells")
    return 0


# ---------------------------------------------------------------------------
# score once
# ---------------------------------------------------------------------------

def parse_anchors(output):
    branches, _ = evaluation_set()
    first = rt.anchor_branches(complete_branches(branches))
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        anchors = {member: rt.anchor_carrier(adapter, b["attempt"]) for member, b in first.items()}
    atomic_torch(anchors_path(output), {"schema": SCHEMA_ANCHORS, "plan_identity": IDENTITY, "scoring_policy": policy,
                                        "parsed_at": utc_now(), "anchors": anchors,
                                        "frames": {m: b["key"] for m, b in first.items()}})


def load_anchors(output):
    store = torch.load(anchors_path(output), map_location="cpu", weights_only=True)
    if store.get("schema") != SCHEMA_ANCHORS or store.get("plan_identity") != IDENTITY:
        raise ValueError("anchor store binding differs")
    return store


def score(output):
    output = Path(output)
    load_plan(output)
    began = time.monotonic()
    if not anchors_path(output).is_file():
        parse_anchors(output)
        mark(output, "anchors", time.monotonic() - began, time.monotonic() - began,
             levels=len(load_anchors(output)["anchors"]))
    anchors = load_anchors(output)["anchors"]
    _, cells = evaluation_set()
    wrong = fit_anchors()
    costs = harness.EndpointCosts(wlc.load_objective())
    pending = [(n, s) for n in UNITS for s in seeds() if not record_path(output, n, s).is_file()]
    for name, seed in pending:
        t0 = time.monotonic()
        with harness.deterministic_scoring() as policy:
            record = score_record(cells, anchors, wrong, costs, name, seed, policy)
        write_json(record_path(output, name, seed), record)
        torch.cuda.empty_cache()
        wall = time.monotonic() - t0
        mark(output, f"score-{name}-seed{seed}", wall, wall)
        log(f"scored {name} seed {seed} ({wall:.0f}s){'; typed failure ' + record['typed_failure'] if record['typed_failure'] else ''}")
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
    log(f"published compute.json, summary.json, findings.md: Gate B {compute['gate_b']['decision']} "
        f"({fmt_interval(compute['gate_b']['pooled_grid_auc'])})")
    return 0


def spot_checks(output):
    """Re-score the unit's first seed on every grid cell and re-parse one anchor in this process."""
    records = load_records(output)
    seed = seeds()[0]
    record = records[("unit", seed)]
    store = load_anchors(output)
    _, cells = evaluation_set()
    costs = harness.EndpointCosts(wlc.load_objective())
    worst = dict.fromkeys(harness.PARSED_QUANTITIES, 0.0)
    passed = True
    with harness.deterministic_scoring():
        model, failure = load_model("unit", seed)
        if model is not None:
            for member, entry in sorted(cells[GATE_INVENTORY]["mixed"].items()):
                fresh = request_block(model, UNITS["unit"]["request"], store["anchors"][member], entry["inventory"], costs)
                check = harness.endpoint_replication(record["inventories"][GATE_INVENTORY][member]["rows"], fresh["rows"])
                passed &= check["pass"]
                for k in worst:
                    worst[k] = max(worst[k], check["max_abs_delta"][k])
        member, key = next(iter(sorted(store["frames"].items())))
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        anchor = rt.anchor_carrier(adapter, str(CAMPAIGN / STREAM / "attempts" / key))
    parse = harness.carrier_replication(store["anchors"][member], anchor)
    return {"rescored": {"seed": seed, "failure": failure, "max_abs_delta": worst, "pass": bool(passed)},
            "anchor": parse, "pass": bool(passed and parse["pass"])}


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_checks(output)
    if not spot["pass"]:
        raise ValueError(f"deterministic re-scoring does not replicate the retained records: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): every table re-derived and byte-compared; unit seed "
        f"{spot['rescored']['seed']} re-scored (max delta {spot['rescored']['max_abs_delta']}), anchor re-parsed; "
        f"Gate B {compute['gate_b']['decision']}")
    return 0


def dry_run(output):
    branches, cells = evaluation_set()
    accounting = evaluation_accounting(branches, cells)
    log(f"output {Path(output)}; levels {accounting['levels']}; branches {accounting['branches']}, complete "
        f"{accounting['complete']}; levels per inventory {accounting['levels_per_inventory']}")
    log(f"units {UNITS}; seeds {seeds()}; handoff token {handoff()['token']}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "score": score, "publish": publish,
         "validate": validate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    return MODES[mode](OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
