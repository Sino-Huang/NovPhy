"""Issue-112 v2: anchor and action-prior controls for the v1 ranking gains.

Binding runner module: scripts/run_decision_dynamics_controls.py
Exact validation command: python -u -m scripts.run_decision_dynamics_controls --validate

The v1 run (issue-112-decision-relevant-dynamics-v1) found that the arms trained with the
decision-contrastive loss (C, LC) rank a left-out lineage's candidates well above the #77
recipe (B), while the C arm's AUC is identical across all nine hybrid requests on several
inventories. Grid and offset share one action set across members, so a predictor could rank
by the action alone, learned from the fit lineages' verdicts, without using the anchor
state. That would pass a ranking gate without carrying decision-relevant dynamics.

v2 asks, with every v1 model and record unchanged:

- Q5 anchor dependence: the same fold model ranks the same candidates from the wrong
  anchors (every other fit member's E99 anchor carrier). Right-anchor minus mean
  wrong-anchor AUC, and the Spearman correlation of the candidate costs between the right
  and each wrong anchor.
- An action-only reference ranker with no model: each candidate action's engine success
  frequency over the other fit lineages' executions of the identical action.
- Endpoint variance shares (anchor / action / residual) of every arm's fold-'all'
  predictions over the fit anchors, as in the v1 diagnostic for the frozen #99 E models.

v2 was frozen AFTER every v1 outcome was known; that is disclosed in plan.json and every
rendering. Zero training, zero engine seconds; rollouts are m99.rollout_endpoints (the v1
scoring operation) inside the #111 deterministic_scoring(). Every interval is a
member-clustered percentile bootstrap (10000 draws, PCG64 7201), EXPLORATORY for fit
lineages and DESCRIPTIVE for the exposed held-out lineages.

Modes: --dry-run --prepare --run --publish --validate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_lookahead_control as wlc
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-112-controls-v2"
V1 = v1.OUTPUT
IDENTITY = "issue-112-controls-v2"
SCHEMA_PLAN = "issue_112_controls_plan_v2"
SCHEMA_RECORD = "issue_112_controls_record_v2"
SCHEMA_COMPUTE = "issue_112_controls_compute_v2"
SCHEMA_REPORT = "issue_112_controls_report_v2"
VALIDATION_COMMAND = "python -u -m scripts.run_decision_dynamics_controls --validate"
RUNNER = "scripts/run_decision_dynamics_controls.py"

INVENTORIES = ("grid", "offset")  # the inventories whose action set is identical across members
PRIMARY = v1.PRIMARY
ARMS = v1.ARMS
FAMILIES = v1.FAMILIES
SEEDS = v1.SEEDS
REQUESTS = v1.REQUESTS
PRIMARY_COST = v1.PRIMARY_COST
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
TOLERANCE = harness.PARSE_TOLERANCE
V1_PUBLISHED = ("plan.json", "smoke.json", "selection.json", "gate_b_handoff.json", "compute.json", "summary.json",
                "findings.md", "comparisons.csv")

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
bootstrap = v1.bootstrap
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-112-controls] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_control(path, schema):
    """v2 control records (cached like v1 records: immutable once written)."""
    record = v1._record_json(str(path), Path(path).stat().st_mtime_ns)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


def record_path(output, arm, family, fold, seed):
    return Path(output) / "records" / f"anchors--{arm}--{family}--{fold}--seed{seed}.json"


def action_key(item):
    a = item["action"]
    return (a["drag_x"], a["drag_y"], a["release_time_ms"], a["tap_time_ms"])


def check_shared_actions(universe):
    for inventory in INVENTORIES:
        sets = {tuple(action_key(i) for i in e["inventory"]) for e in universe[inventory].values()}
        if len(sets) != 1:
            raise ValueError(f"{inventory} action sets differ across members")


def anchor_members(universe, inventory):
    """Fit members with an entry in the inventory: the pool of wrong anchors."""
    fitted = set(v1.fit_lineages())
    return sorted(m for m in universe[inventory] if m in fitted)


def targets(universe, fold, inventory):
    if fold == "all":
        return sorted(m for m in universe[inventory] if m in v1.HELD_OUT)
    return [fold] if fold in universe[inventory] else []


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def control_record(universe, costs, anchors, arm, family, fold, seed, policy):
    """Rows of every target member's candidates from its own anchor and from every fit anchor;
    fold 'all' also rolls the shared action set out from every fit anchor (variance shares)."""
    model = v1.load_member(V1, arm, family, fold, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "scoring_policy": policy,
            "cell": {"arm": arm, "family": family, "fold": fold, "seed": seed}, "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": "member_retired_nonfinite", "inventories": {}}
    out = {}
    for inventory in INVENTORIES:
        block = {}
        pool = anchor_members(universe, inventory)
        for target in targets(universe, fold, inventory):
            items = universe[inventory][target]["inventory"]
            block[target] = {anchor: v1.endpoint_block(model, family, anchors[anchor], items, costs)
                             for anchor in sorted({target, *pool})}
        if fold == "all":
            items = universe[inventory][pool[0]]["inventory"]
            block["_fit_anchors"] = {anchor: v1.endpoint_block(model, family, anchors[anchor], items, costs)
                                     for anchor in pool}
        out[inventory] = block
    return {**base, "typed_failure": None, "inventories": out}


def load_anchors():
    fit = torch.load(v1.anchors_path(V1, "fit"), weights_only=True)["anchors"]
    held = torch.load(v1.anchors_path(V1, "heldout"), weights_only=True)["anchors"]
    return {**fit, **held}


def cells():
    return [(arm, family, fold, seed) for arm in ARMS for family in FAMILIES for fold in v1.folds() for seed in SEEDS]


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def frozen_plan(frozen_at):
    selection = read_json(V1 / "selection.json")
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 2, "issue": 112, "frozen_at": frozen_at,
        "role": "post-hoc controls of v1 (v1 unchanged and re-validated at every publish and validate)",
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER, "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "disclosure": ("Frozen AFTER every v1 outcome was known, including the v1 held-out tables and the observation "
                       "that the C arm's leave-one-lineage-out and held-out AUCs are identical across the nine hybrid "
                       "requests on several inventories. The questions below were chosen because of that observation. "
                       "They do not change any v1 token, the v1 selection or the Gate-B handoff."),
        "v1": {"selected": selection["selected"],
               "artifacts": [{"name": n, "sha256": sha256_of(V1 / n)} for n in V1_PUBLISHED],
               "runner_sha256": sha256_of(ROOT / v1.RUNNER), "module_sha256": sha256_of(ROOT / v1.MODEL_MODULE)},
        "inventories": list(INVENTORIES),
        "controls": {
            "wrong_anchor": ("for every v1 model (every arm, family, fold, seed) and every target member (the fold "
                             "lineage; held-out members for fold 'all'): the target's candidates rolled out with each "
                             "request from the target's own E99 anchor (right) and from every fit member's E99 anchor "
                             "other than the target (wrong); costs are EndpointCosts rows against the anchor used"),
            "right_minus_wrong": ("per cell (target, inventory, seed): right-anchor AUC minus the mean over wrong "
                                  "anchors of the wrong-anchor AUC, tie-free cost; member-clustered bootstrap"),
            "rank_stability": ("per cell: mean over wrong anchors of the Spearman correlation between the right-anchor "
                               "and wrong-anchor candidate costs (1 = the ranking ignores the anchor)"),
            "action_prior": ("no model: cost = minus the engine success frequency of the identical action over the "
                             "other fit lineages' verdict-bearing executions (leave-one-lineage-out) or over all fit "
                             "lineages (held-out); an action never executed elsewhere costs 0"),
            "variance": ("fold-'all' members: the shared action set rolled out from every fit anchor; two-way shares "
                         "(anchor, action, residual) of the tie-free cost over the fit members with every candidate "
                         "executed (the v1 diagnostic membership), per arm and request, seed-mean"),
        },
        "decision_rule": {
            "Q5_anchor_dependence": (f"the v1-selected unit ({selection['selected']['arm']}:"
                                     f"{selection['selected']['request']}), leave-one-lineage-out grid: supported "
                                     f"(its ranking depends on the anchor) iff right - wrong >= {DELTA} with lower > 0; "
                                     f"not_supported_by_this_experiment iff upper < {DELTA}; otherwise {STOP_TOKEN}"),
            "labels": "leave-one-lineage-out: EXPLORATORY; held-out: DESCRIPTIVE",
        },
        "guards": {
            "G5": (f"every right-anchor row replicates the v1 record row (endpoint_replication, tolerance {TOLERANCE}) "
                   "- the same model, anchor, actions and operation"),
            "G6": "every record carries the #111 SCORING_POLICY",
        },
        "claim_boundary": v1.read_json(V1 / "plan.json")["claim_boundary"],
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-112 controls protocol")
    for entry in plan["v1"]["artifacts"]:
        if sha256_of(V1 / entry["name"]) != entry["sha256"]:
            raise ValueError(f"v1 artifact {entry['name']} changed after the v2 freeze")
    if sha256_of(ROOT / v1.RUNNER) != plan["v1"]["runner_sha256"]:
        raise ValueError("the v1 runner changed after the v2 freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    if (output / "records").exists():
        raise ValueError("records exist before the freeze; refusing")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json_text(frozen_plan(utc_now())))
    load_plan(output)
    log("frozen plan published")
    return 0


def run(output):
    output = Path(output)
    load_plan(output)
    _, _, universe = v1.load_universe()
    check_shared_actions(universe)
    anchors = load_anchors()
    costs = harness.EndpointCosts(wlc.load_objective())
    pending = [c for c in cells() if not record_path(output, *c).is_file()]
    progress = v1.Progress("control records", len(pending))
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        for cell in pending:
            write_json(record_path(output, *cell), control_record(universe, costs, anchors, *cell, policy))
            progress.step(13)
    log(f"records complete ({time.monotonic() - began:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def auc(entry, rows):
    return m99.view(entry, rows, PRIMARY_COST)["auc"]


def spearman_rows(a, b):
    x = {r["ordinal"]: r[PRIMARY_COST] for r in a if not r["excluded"]}
    y = {r["ordinal"]: r[PRIMARY_COST] for r in b if not r["excluded"]}
    common = sorted(x.keys() & y.keys())
    return v1.spearman([x[o] for o in common], [y[o] for o in common])


def anchor_rows(output, universe, inventory, lolo):
    """One row per (target, seed) cell: per (arm, request) right AUC, mean wrong AUC, stability."""
    rows = []
    folds = v1.fit_lineages() if lolo else ["all"]
    for fold in folds:
        for target in targets(universe, fold, inventory):
            entry = universe[inventory][target]
            if not t96.mixed(entry):
                continue
            for seed in SEEDS:
                views = {}
                for arm in ARMS:
                    for family in FAMILIES:
                        record = read_control(record_path(output, arm, family, fold, seed), SCHEMA_RECORD)
                        for request in v1.family_requests(family):
                            if record["typed_failure"]:
                                views[(arm, request)] = None
                                continue
                            block = record["inventories"][inventory][target]
                            right = block[target][request]["rows"]
                            wrong = [block[a][request]["rows"] for a in sorted(block) if a != target]
                            wrong_auc = [a for a in (auc(entry, w) for w in wrong) if a is not None]
                            rhos = [spearman_rows(right, w) for w in wrong]
                            rhos = [r for r in rhos if r is not None]
                            views[(arm, request)] = {
                                "right": auc(entry, right),
                                "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                                "stability": float(np.mean(rhos)) if rhos else None, "wrong_anchors": len(wrong)}
                rows.append({"member": target, "seed": seed, "views": views})
    return rows


def field_block(rows, key, field):
    values = [(r["views"][key][field], r["member"]) for r in rows
              if r["views"][key] is not None and r["views"][key][field] is not None]
    return bootstrap([v for v, _ in values], [m for _, m in values])


def difference_block(rows, pieces):
    """pieces: list of (weight, key, field); a cell enters when every piece is defined."""
    values, clusters = [], []
    for r in rows:
        parts = [None if r["views"][k] is None else r["views"][k][f] for _, k, f in pieces]
        if any(p is None for p in parts):
            continue
        values.append(float(sum(w * p for (w, _, _), p in zip(pieces, parts))))
        clusters.append(r["member"])
    return bootstrap(values, clusters)


def anchor_tables(rows):
    n = len(REQUESTS)
    out = {"members": sorted({r["member"] for r in rows}), "cells": len(rows), "units": {}, "arms": {}}
    for arm in ARMS:
        for request in REQUESTS:
            key = (arm, request)
            out["units"][f"{arm}:{request}"] = {
                "right": field_block(rows, key, "right"), "wrong": field_block(rows, key, "wrong"),
                "right_minus_wrong": difference_block(rows, [(1, key, "right"), (-1, key, "wrong")]),
                "stability": field_block(rows, key, "stability")}
        out["arms"][arm] = {
            "right_minus_wrong": difference_block(rows, [(1 / n, (arm, r), "right") for r in REQUESTS]
                                                  + [(-1 / n, (arm, r), "wrong") for r in REQUESTS]),
            "stability": difference_block(rows, [(1 / n, (arm, r), "stability") for r in REQUESTS])}
    return out


def prior_tables(universe):
    fit = v1.fit_lineages()
    out = {}
    for inventory in (*INVENTORIES, "angle", "power"):
        for cohort in ("lolo", "heldout"):
            values, clusters, overlap = [], [], []
            for member in sorted(universe[inventory]):
                entry = universe[inventory][member]
                if not t96.mixed(entry) or (member in fit) != (cohort == "lolo"):
                    continue
                pool = [m for m in fit if m != member]
                stats = {}
                for other in pool:
                    e = universe[inventory].get(other)
                    for item in (e or {"inventory": []})["inventory"]:
                        if item["ordinal"] in e["verdicts"]:
                            s = stats.setdefault(action_key(item), [0, 0])
                            s[0] += int(e["verdicts"][item["ordinal"]])
                            s[1] += 1
                rows = [{"ordinal": item["ordinal"], "excluded": False, "count": 0.0,
                         PRIMARY_COST: -(stats[action_key(item)][0] / stats[action_key(item)][1])
                         if action_key(item) in stats else 0.0} for item in entry["inventory"]]
                overlap.append(float(np.mean([action_key(i) in stats for i in entry["inventory"]])))
                for _ in SEEDS:  # the same (seed-free) value per seed keeps the v1 cell weighting
                    values.append(auc(entry, rows))
                    clusters.append(member)
            out[f"{inventory}:{cohort}"] = {"auc": bootstrap(values, clusters),
                                            "action_overlap": float(np.mean(overlap)) if overlap else None}
    return out


def variance_tables(output, universe):
    endpoints = v1.read_record(v1.endpoints_path(V1), v1.SCHEMA_ENDPOINTS)["cells"]
    out = {}
    for inventory in INVENTORIES:
        complete = [m for m, block in endpoints[inventory].items()
                    if len(block["engine"]) == len(universe[inventory][m]["inventory"])
                    and all(not r["excluded"] for r in block["engine"] + block["parsed"])]
        engine = v1.shares([v1.by_ordinal(endpoints[inventory][m]["engine"], PRIMARY_COST) for m in complete])
        arms = {}
        for arm in ARMS:
            for family in FAMILIES:
                for request in v1.family_requests(family):
                    per_seed = []
                    for seed in SEEDS:
                        record = read_control(record_path(output, arm, family, "all", seed), SCHEMA_RECORD)
                        if record["typed_failure"]:
                            continue
                        rows = {m: record["inventories"][inventory]["_fit_anchors"][m][request]["rows"]
                                for m in complete}
                        if any(r["excluded"] for m in complete for r in rows[m]):
                            continue
                        share = v1.shares([v1.by_ordinal(rows[m], PRIMARY_COST) for m in complete])
                        if share is not None:
                            per_seed.append(share)
                    arms[f"{arm}:{request}"] = ({k: float(np.mean([s[k] for s in per_seed]))
                                                 for k in ("anchor", "action", "residual")} | {"seeds": len(per_seed)}
                                                if per_seed else None)
        out[inventory] = {"members": complete, "engine": engine, "arms": arms}
    return out


def replication_guard(output, universe):
    worst = dict.fromkeys(harness.PARSED_QUANTITIES, 0.0)
    compared, passed = 0, True
    for arm, family, fold, seed in cells():
        record = read_control(record_path(output, arm, family, fold, seed), SCHEMA_RECORD)
        if record["typed_failure"]:
            continue
        if fold == "all":
            reference = v1.read_record(v1.heldout_path(V1, arm, family, seed), v1.SCHEMA_HELDOUT)["inventories"]
        else:
            reference = {i: {fold: b} for i, b in
                         v1.read_record(v1.lolo_path(V1, arm, family, fold, seed), v1.SCHEMA_LOLO)["inventories"].items()}
        for inventory in INVENTORIES:
            for target, block in record["inventories"][inventory].items():
                if target == "_fit_anchors":
                    continue
                for request, value in block[target].items():
                    check = harness.endpoint_replication(reference[inventory][target][request]["rows"], value["rows"])
                    compared += 1
                    passed &= check["pass"]
                    for k in worst:
                        worst[k] = max(worst[k], check["max_abs_delta"][k])
    return {"compared_request_blocks": compared, "max_abs_delta": worst, "tolerance": TOLERANCE, "pass": passed}


def token(block):
    if block is None:
        return STOP_TOKEN
    low, high = block["interval"]
    if block["mean"] >= DELTA and low > 0:
        return "supported"
    if high < DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def compute_tables(output, plan):
    _, _, universe = v1.load_universe()
    lolo = {i: anchor_tables(anchor_rows(output, universe, i, True)) for i in INVENTORIES}
    held = {i: anchor_tables(anchor_rows(output, universe, i, False)) for i in INVENTORIES}
    records = sorted(Path(output).glob("records/*.json"))
    policies = all(read_json(p).get("scoring_policy") == harness.SCORING_POLICY for p in records)
    guards = {"G5": replication_guard(output, universe),
              "G6": {"records": len(records), "pass": policies and len(records) == len(cells())}}
    guards_ok = all(g["pass"] for g in guards.values())
    selected = plan["v1"]["selected"]
    unit = lolo[PRIMARY]["units"][f"{selected['arm']}:{selected['request']}"]
    return {"schema": SCHEMA_COMPUTE, "identity": IDENTITY, "guards": guards, "guards_pass": guards_ok,
            "Q5_anchor_dependence": {"unit": f"{selected['arm']}:{selected['request']}",
                                     "estimate": unit["right_minus_wrong"], "stability": unit["stability"],
                                     "token": token(unit["right_minus_wrong"]) if guards_ok else STOP_TOKEN,
                                     "label": "EXPLORATORY (post hoc)"},
            "lolo": lolo, "heldout": held, "action_prior": prior_tables(universe),
            "variance": variance_tables(output, universe), "engine_seconds": 0}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def cell(block):
    return "n/a" if block is None else fmt_interval(block)


def findings_md(plan, compute):
    lines = []
    add = lines.append
    q5 = compute["Q5_anchor_dependence"]
    add("# Issue #112 v2 — anchor and action-prior controls (post hoc)")
    add("")
    add(f"Frozen {plan['frozen_at']}, after every v1 outcome was known. {plan['disclosure']}")
    add(f"Guards pass: {compute['guards_pass']} (G5 right-anchor replication of v1 "
        f"{compute['guards']['G5']['pass']}, max {max(compute['guards']['G5']['max_abs_delta'].values()):.2e}; "
        f"G6 {compute['guards']['G6']['pass']}). Validation: `{VALIDATION_COMMAND}`.")
    add("")
    add(f"Q5 anchor dependence of the v1-selected unit {q5['unit']} (LOLO grid, right − wrong anchor AUC): "
        f"{cell(q5['estimate'])}; rank stability {cell(q5['stability'])}; token **{q5['token']}** "
        f"({q5['label']}).")
    add("")
    for cohort, tables, label in (("Leave-one-lineage-out (EXPLORATORY)", compute["lolo"], "lolo"),
                                  ("Held-out, fold-'all' models (DESCRIPTIVE)", compute["heldout"], "heldout")):
        add(f"## {cohort}")
        add("")
        for inventory, table in tables.items():
            add(f"### {inventory} ({len(table['members'])} members, {table['cells']} cells)")
            add("")
            add("| arm | request-mean right − wrong | request-mean rank stability |")
            add("| --- | --- | --- |")
            for arm, block in table["arms"].items():
                add(f"| {arm} | {cell(block['right_minus_wrong'])} | {cell(block['stability'])} |")
            add("")
            add("| unit | right anchor | wrong anchors | right − wrong | rank stability |")
            add("| --- | --- | --- | --- | --- |")
            for key, block in table["units"].items():
                add(f"| {key} | {cell(block['right'])} | {cell(block['wrong'])} | {cell(block['right_minus_wrong'])} "
                    f"| {cell(block['stability'])} |")
            add("")
            prior = compute["action_prior"][f"{inventory}:{label}"]
            add(f"No-model action prior on these cells: AUC {cell(prior['auc'])} (action overlap "
                f"{prior['action_overlap']}).")
            add("")
    add("## Action prior on the other inventories")
    add("")
    add("| inventory:cohort | AUC | action overlap |")
    add("| --- | --- | --- |")
    for key, block in compute["action_prior"].items():
        add(f"| {key} | {cell(block['auc'])} | {block['action_overlap']} |")
    add("")
    add("## Endpoint variance shares of the fold-'all' models over the fit anchors (tie-free cost, in-sample)")
    add("")
    add("| inventory | unit | anchor | action | residual | seeds |")
    add("| --- | --- | --- | --- | --- | --- |")
    for inventory, block in compute["variance"].items():
        e = block["engine"]
        add(f"| {inventory} ({len(block['members'])} members) | engine | {e['anchor']:.4f} | {e['action']:.4f} | "
            f"{e['residual']:.4f} | |")
        for key, v in block["arms"].items():
            if v is not None:
                add(f"| {inventory} | {key} | {v['anchor']:.4f} | {v['action']:.4f} | {v['residual']:.4f} | "
                    f"{v['seeds']} |")
    add("")
    add(f"Claim boundary: {plan['claim_boundary']}.")
    add("")
    return "\n".join(lines)


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "cohort", "inventory", "unit", "statistic", "value", "interval_low", "interval_high",
                     "clusters", "label"])
    for cohort, tables, label in (("lolo", compute["lolo"], "EXPLORATORY"), ("heldout", compute["heldout"],
                                                                             "DESCRIPTIVE")):
        for inventory, table in tables.items():
            for key, block in table["units"].items():
                for statistic, value in block.items():
                    if value is None:
                        writer.writerow([f"{cohort}:{inventory}:{key}:{statistic}", cohort, inventory, key, statistic,
                                         "", "", "", "", label])
                    else:
                        writer.writerow([f"{cohort}:{inventory}:{key}:{statistic}", cohort, inventory, key, statistic,
                                         f"{value['mean']:.6f}", f"{value['interval'][0]:.6f}",
                                         f"{value['interval'][1]:.6f}", value["clusters"], label])
    return buffer.getvalue()


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "disclosure": plan["disclosure"], "guards": compute["guards"],
            "guards_pass": compute["guards_pass"], "Q5_anchor_dependence": compute["Q5_anchor_dependence"],
            "claim_boundary": plan["claim_boundary"], "engine_seconds": 0}


def rendered(plan, compute):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    if v1.validate(V1) != 0:
        raise ValueError("v1 --validate failed")
    compute = compute_tables(output, plan)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    log(f"published: Q5 {compute['Q5_anchor_dependence']['token']}; guards pass {compute['guards_pass']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    if v1.validate(V1) != 0:
        raise ValueError("v1 --validate failed")
    with harness.deterministic_scoring() as policy:
        _, _, universe = v1.load_universe()
        arm, family, fold, seed = ARMS[-1], "continuous", v1.fit_lineages()[0], SEEDS[1]
        fresh = control_record(universe, harness.EndpointCosts(wlc.load_objective()), load_anchors(), arm, family,
                               fold, seed, policy)
    stored = read_control(record_path(output, arm, family, fold, seed), SCHEMA_RECORD)
    if fresh["typed_failure"] is None:
        for inventory, block in fresh["inventories"].items():
            for target, anchors in block.items():
                for anchor, requests in anchors.items():
                    for request, value in requests.items():
                        check = harness.endpoint_replication(stored["inventories"][inventory][target][anchor][request]
                                                             ["rows"], value["rows"])
                        if not check["pass"]:
                            raise ValueError(f"re-scored control rows differ ({inventory}/{target}/{anchor}/{request})")
    fresh_compute = compute_tables(output, plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh_compute)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh_compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"validation passed: v1 re-validated, one record re-scored, every table re-derived and byte-compared "
        f"({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    _, _, universe = v1.load_universe()
    check_shared_actions(universe)
    log(f"dry run (no write, no statistic); output {Path(output)}; {len(cells())} control records; "
        f"wrong-anchor pools {({i: len(anchor_members(universe, i)) for i in INVENTORIES})}")
    return 0


MODES = {"dry-run": (dry_run, False), "prepare": (prepare, False), "run": (run, True), "publish": (publish, True),
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
