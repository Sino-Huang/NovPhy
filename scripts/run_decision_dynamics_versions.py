"""Issue-112 v3+: versioned development of the Gate-B predictor recipe (leave-one-lineage-out only).

Binding runner module: scripts/run_decision_dynamics_versions.py
Exact validation command: python -u -m scripts.run_decision_dynamics_versions --version N --validate

v1 (scripts/run_decision_relevant_dynamics.py) is hash-bound by its frozen plan and by the
v2 controls plan, so it is never edited. Each version N >= 3 here loads a private copy of the
v1 implementation (its own module object; v1 imported elsewhere is unaffected), applies the
frozen spec ``VERSIONS[N]`` (output root, identity, arms, loss constants, the stability module)
and reuses every v1 step: data, smoke, training, leave-one-lineage-out (LOLO) scoring and
selection. The spec is stored in each version's plan and compared at every load.

Held-out lineages 007/008/014/016 are NOT scored by any version here: v1 used their single
post-freeze look. Every version is selected and reported on LOLO only (EXPLORATORY); the test
is Gate B on the #104 sealed split.

Version policy (declared in the v3 plan, binding for v4-v6):
- a version changes only what a pre-declared diagnosis of the previous version names;
- if a version's G2 (training stability) fails, the next version may change stability only;
- once G2 passes, a further version needs a named deficiency and stops the series when its
  best selection score S improves by less than 0.02 over the previous version's; v6 is the last;
- the frozen Gate-B recipe is the selected unit of the last version whose G2 passed.

Modes: --dry-run --smoke --prepare --data --train --score-lolo --select --score-controls
--publish --validate (each with --version N). Training runs in its own invocation; every
scoring mode and --validate runs under the #111 deterministic_scoring() policy.
"""
from __future__ import annotations

import argparse
import csv
import functools
import importlib.util
import io
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from scripts import run_decision_dynamics_controls as c2
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_lookahead_control as wlc
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.planning import scoring_harness as harness
from world_model.training import decision_dynamics as dd
from world_model.training import stable_decision_dynamics as sd

ROOT = v1.ROOT
RUNNER = "scripts/run_decision_dynamics_versions.py"
STABLE_MODULE = "world_model/training/stable_decision_dynamics.py"
STEPS = 9000

VERSIONS = {
    3: {
        "parent": 1,
        "diagnosis": ("v1 retired 28/312 members (L 5/78, LC 23/78, B and C 0), every one on a Delta = 1 update at "
                      "curriculum horizon >= 165. An instrumented replay of LC/continuous fold 006 seed 20260909 "
                      "measured per-term gradient norms on Delta = 1 updates of up to 2.2e3 (long-horizon) and "
                      "1.1e4 (ranking) against at most 1.2 (#77 local): exploding gradients through up to 225 "
                      "recursive transitions; an overflow retires the member. No Delta = 5 update (45 recursive "
                      "transitions) of any of the 312 members was nonfinite."),
        "change": ("truncated back-propagation: the long-horizon and ranking recursions are detached after every "
                   "45 transitions (Delta = 5 and 15 unchanged; Delta = 1 in 45-frame gradient segments); a "
                   "nonfinite update is skipped per member (parameters and AdamW moments untouched), and a member "
                   "is retired only after 10 consecutive skips"),
        "truncation": 45, "patience": 10, "skip_cap": 45,
        "arms": ["B", "L", "C", "LC"], "long": {}, "rank": {},
    },
}
STOP_DELTA = 0.02
LAST_VERSION = 6

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
bootstrap = v1.bootstrap
fmt_interval = v1.fmt_interval


def log(message):
    print(f"[issue-112-versions] {message}", flush=True)


# ---------------------------------------------------------------------------
# the per-version implementation
# ---------------------------------------------------------------------------

def arm_definitions(spec):
    tail = (f" (v{spec['version']}: truncated back-propagation every {spec['truncation']} transitions, skip-on-"
            f"nonfinite with retirement after {spec['patience']} consecutive skips)")
    return {arm: v1.ARM_DEFINITIONS[arm] + (tail if arm != "B" else " (identical recipe to v1 B)")
            for arm in spec["arms"]}


def spec_of(version):
    if version not in VERSIONS:
        raise ValueError(f"version {version} is not declared")
    return {"version": version, **VERSIONS[version]}


@functools.lru_cache(maxsize=None)
def implementation(version):
    spec = spec_of(version)
    loader = importlib.util.spec_from_file_location(f"issue112_v{version}_implementation", ROOT / v1.RUNNER)
    impl = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(impl)
    impl.VERSION = version
    impl.OUTPUT = ROOT / f".local-artifacts/issue-112-decision-dynamics-v{version}"
    impl.IDENTITY = f"issue-112-decision-dynamics-v{version}"
    for name in dir(impl):
        if name.startswith("SCHEMA_"):
            setattr(impl, name, getattr(v1, name).replace("_v1", f"_v{version}"))
    impl.VALIDATION_COMMAND = f"python -u -m scripts.run_decision_dynamics_versions --version {version} --validate"
    impl.RUNNER = RUNNER
    impl.MODEL_MODULE = STABLE_MODULE
    impl.ARMS = tuple(spec["arms"])
    impl.ARM_DEFINITIONS = arm_definitions(spec)
    impl.LONG = {**v1.LONG, **spec["long"],
                 "truncation": f"every {spec['truncation']} recursive transitions (stable_decision_dynamics)"}
    impl.RANK = {**v1.RANK, **spec["rank"]}

    class Ensemble(sd.SkippingEnsemble):
        patience = spec["patience"]

    impl.dd = types.SimpleNamespace(
        Ensemble=Ensemble, local_loss=dd.local_loss, horizon_at=dd.horizon_at, ramp_at=dd.ramp_at,
        long_and_rank_loss=functools.partial(sd.truncated_long_and_rank_loss, truncation=spec["truncation"]))
    original_plan, original_bindings = impl.frozen_plan, impl.input_bindings

    def input_bindings(output):
        extra = {"issue_112_v1_runner": ROOT / v1.RUNNER, "issue_112_v1_module": ROOT / v1.MODEL_MODULE,
                 "issue_112_v1_selection": v1.OUTPUT / "selection.json",
                 "issue_112_v1_compute": v1.OUTPUT / "compute.json"}
        if spec["parent"] > 1:
            parent = implementation(spec["parent"]).OUTPUT
            extra.update({"parent_plan": parent / "plan.json", "parent_compute": parent / "compute.json"})
        return original_bindings(output) + [{"name": n, "artifact": str(Path(p).relative_to(ROOT)),
                                             "sha256": sha256_of(p)} for n, p in extra.items()]

    def frozen_plan(output, frozen_at, smoke_evidence, shots, universe):
        plan = original_plan(output, frozen_at, smoke_evidence, shots, universe)
        plan.update({
            "version": version, "version_spec": spec,
            "role": ("development version under the #87 versioned-protocol policy; selection on leave-one-lineage-"
                     "out only (EXPLORATORY); held-out lineages are not scored by this version"),
            "version_policy": policy_text(),
            "disclosure": (plan["disclosure"] + f" Frozen after every outcome of v1 (and v2 controls) and of every "
                           f"version before v{version}: the change is chosen from the named training diagnosis, not "
                           "from an AUC; held-out records exist only from v1 and are not read here."),
        })
        plan["validation"]["heldout"] = "not scored (v1 used the single post-freeze look)"
        plan["guards"]["G2"] = (f"every member reached {STEPS} updates without retirement and skipped at most "
                                f"{spec['skip_cap']} updates")
        plan["universe"]["heldout_records"] = 0
        plan["prohibited"].append("scoring or reading any held-out record")
        return plan

    impl.input_bindings = input_bindings
    impl.frozen_plan = frozen_plan
    return impl


def policy_text():
    return {"rules": [line.strip("- ") for line in __doc__.split("Version policy")[1].split("Modes:")[0]
                      .strip().split("\n") if line.startswith("- ")],
            "stop_delta": STOP_DELTA, "last_version": LAST_VERSION}


def load_plan(version):
    impl = implementation(version)
    plan = impl.load_plan(impl.OUTPUT)
    if plan.get("version_spec") != json.loads(json_text(spec_of(version))):
        raise ValueError(f"version {version} spec differs from the frozen plan")
    return impl, plan


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def dry_run(version):
    impl = implementation(version)
    log(f"v{version}: output {impl.OUTPUT}; arms {impl.ARMS}; spec {spec_of(version)}")
    return impl.dry_run(impl.OUTPUT)


def smoke(version):
    impl = implementation(version)
    impl.OUTPUT.mkdir(parents=True, exist_ok=True)
    return impl.smoke(impl.OUTPUT)


def prepare(version):
    impl = implementation(version)
    return impl.prepare(impl.OUTPUT)


def data(version):
    impl, _ = load_plan(version)
    return impl.data(impl.OUTPUT)


def skip_path(impl, arm, family, index):
    return impl.OUTPUT / "training" / f"{arm}--{family}--group{index}.skips.json"


def train(version):
    impl, plan = load_plan(version)
    output = impl.OUTPUT
    torch.backends.cudnn.benchmark = True
    payload = impl.load_training_data(output)
    for family in impl.FAMILIES:
        data_on_gpu = None
        for arm in impl.ARMS:
            size = plan["training"]["group_sizes"][f"{arm}_{family}"]
            for index, members in enumerate(impl.member_groups(size)):
                if impl.group_path(output, arm, family, index).is_file():
                    continue
                data_on_gpu = data_on_gpu or impl.TrainData(payload, family)
                began = time.monotonic()
                report, ensemble = impl.train_group(output, arm, family, index, members, data_on_gpu)
                write_json(skip_path(impl, arm, family, index),
                           {"members": [{"fold": f, "seed": s, "skipped": int(ensemble.skipped[i])}
                                        for i, (f, s) in enumerate(members)]})
                del ensemble
                torch.cuda.empty_cache()
                impl.mark(output, f"train_{arm}_{family}_group{index}", time.monotonic() - began,
                          report["gpu_seconds"], members=len(members),
                          retired=sum(not m["complete"] for m in report["members"]))
                log(f"v{version} trained {arm}/{family}/group{index}: {report['gpu_seconds']:.0f}s")
        del data_on_gpu
        torch.cuda.empty_cache()
    return 0


def score_lolo(version):
    impl, _ = load_plan(version)
    return impl.score_lolo(impl.OUTPUT)


def select(version):
    impl, _ = load_plan(version)
    return impl.select(impl.OUTPUT)


def controls_path(impl, fold, seed):
    return impl.OUTPUT / "records" / f"controls--{fold}--seed{seed}.json"


def score_controls(version):
    """Wrong-anchor control of the selected unit's (arm, family) members on grid and offset (v2 definition)."""
    impl, _ = load_plan(version)
    selection = read_json(impl.OUTPUT / "selection.json")
    arm, family = selection["selected"]["arm"], selection["selected"]["family"]
    _, _, universe = impl.load_universe()
    anchors = torch.load(impl.anchors_path(impl.OUTPUT, "fit"), weights_only=True)["anchors"]
    costs = harness.EndpointCosts(wlc.load_objective())
    cells = [(fold, seed) for fold in impl.fit_lineages() for seed in impl.SEEDS]
    pending = [c for c in cells if not controls_path(impl, *c).is_file()]
    progress = impl.Progress(f"v{version} control records", len(pending))
    with harness.deterministic_scoring() as policy:
        for fold, seed in pending:
            model = impl.load_member(impl.OUTPUT, arm, family, fold, seed)
            record = {"schema": f"issue_112_controls_record_v{version}", "plan_identity": impl.IDENTITY,
                      "scoring_policy": policy, "cell": {"arm": arm, "family": family, "fold": fold, "seed": seed},
                      "typed_failure": None if model is not None else "member_retired_nonfinite", "inventories": {}}
            if model is not None:
                for inventory in c2.INVENTORIES:
                    if fold in universe[inventory]:
                        items = universe[inventory][fold]["inventory"]
                        record["inventories"][inventory] = {
                            anchor: impl.endpoint_block(model, family, anchors[anchor], items, costs)
                            for anchor in sorted({fold, *c2.anchor_members(universe, inventory)})}
            write_json(controls_path(impl, fold, seed), record)
            progress.step(6)
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def control_rows(impl, universe, inventory, arm, family):
    rows = []
    for fold in impl.fit_lineages():
        entry = universe[inventory].get(fold)
        if entry is None or not t96.mixed(entry):
            continue
        for seed in impl.SEEDS:
            record = impl.read_record(controls_path(impl, fold, seed), f"issue_112_controls_record_v{impl.VERSION}")
            views = {}
            for request in impl.family_requests(family):
                if record["typed_failure"]:
                    views[(arm, request)] = None
                    continue
                block = record["inventories"][inventory]
                right = block[fold][request]["rows"]
                wrong = [block[a][request]["rows"] for a in sorted(block) if a != fold]
                wrong_auc = [a for a in (c2.auc(entry, w) for w in wrong) if a is not None]
                rhos = [r for r in (c2.spearman_rows(right, w) for w in wrong) if r is not None]
                views[(arm, request)] = {"right": c2.auc(entry, right),
                                         "wrong": float(np.mean(wrong_auc)) if wrong_auc else None,
                                         "stability": float(np.mean(rhos)) if rhos else None}
            rows.append({"member": fold, "seed": seed, "views": views})
    return rows


def controls_table(impl, universe, selection):
    arm, family, request = (selection["selected"][k] for k in ("arm", "family", "request"))
    out = {}
    for inventory in c2.INVENTORIES:
        rows = control_rows(impl, universe, inventory, arm, family)
        key = (arm, request)
        out[inventory] = {"unit": f"{arm}:{request}", "cells": len(rows),
                          "right_minus_wrong": c2.difference_block(rows, [(1, key, "right"), (-1, key, "wrong")]),
                          "stability": c2.field_block(rows, key, "stability")}
    return out


def skip_counts(impl, plan):
    members = []
    for family in impl.FAMILIES:
        for arm in impl.ARMS:
            size = plan["training"]["group_sizes"][f"{arm}_{family}"]
            for index in range(len(impl.member_groups(size))):
                for member in read_json(skip_path(impl, arm, family, index))["members"]:
                    members.append({"arm": arm, "family": family, **member})
    per_arm = {f"{a}:{f}": {"members_with_skips": sum(1 for m in members if (m["arm"], m["family"]) == (a, f)
                                                      and m["skipped"]),
                            "total_skipped": sum(m["skipped"] for m in members if (m["arm"], m["family"]) == (a, f)),
                            "max_skipped": max(m["skipped"] for m in members if (m["arm"], m["family"]) == (a, f))}
               for a in impl.ARMS for f in impl.FAMILIES}
    return members, per_arm


def previous_best(version):
    parent = spec_of(version)["parent"]
    if parent == 1:
        selection = read_json(v1.OUTPUT / "selection.json")
        return {"version": 1, **selection["selected"]}
    compute = read_json(implementation(parent).OUTPUT / "compute.json")
    return {"version": parent, **compute["Q3_selected_target"]["selected"]}


def spot_check(impl, universe):
    """Re-score one complete LOLO record of the selected arm in this process (endpoint rows)."""
    selection = read_json(impl.OUTPUT / "selection.json")
    arm, family = selection["selected"]["arm"], selection["selected"]["family"]
    anchors = torch.load(impl.anchors_path(impl.OUTPUT, "fit"), weights_only=True)["anchors"]
    costs = harness.EndpointCosts(wlc.load_objective())
    for fold in impl.fit_lineages():
        for seed in impl.SEEDS:
            model = impl.load_member(impl.OUTPUT, arm, family, fold, seed)
            if model is None or fold not in universe[impl.PRIMARY]:
                continue
            record = impl.read_record(impl.lolo_path(impl.OUTPUT, arm, family, fold, seed), impl.SCHEMA_LOLO)
            fresh = impl.endpoint_block(model, family, anchors[fold], universe[impl.PRIMARY][fold]["inventory"], costs)
            return {"cell": [arm, family, fold, seed], **impl._replicated(record["inventories"][impl.PRIMARY], fresh)}
    return None


def compute_tables(version, spot):
    impl, plan = load_plan(version)
    output = impl.OUTPUT
    spec = spec_of(version)
    _, _, universe = impl.load_universe()
    ledger = impl.load_ledger(output)
    selection = read_json(output / "selection.json")
    lolo = impl.lolo_tables(output, universe)
    scores = impl.selection_scores(lolo)
    best = impl.select_unit(scores)
    per = impl.lolo_rollout(output)
    curves, nonfinite = impl.rollout_tables(per)
    training = impl.training_summary(output, plan)
    skips, skips_per_arm = skip_counts(impl, plan)
    records = sorted(output.glob("records/*.json"))
    policies = all(read_json(p).get("scoring_policy") == harness.SCORING_POLICY for p in records)
    worst_nonfinite = max(b["nonfinite_share"] or 0.0 for b in lolo[impl.PRIMARY]["rankers"][impl.PRIMARY_COST].values())
    members = impl.load_training_data_members(output)
    guards = {
        "G1": {"observed": {"training_members": members}, "pass": not (set(members) & set(impl.HELD_OUT))},
        "G2": {"observed": {"members": len(training["members"]),
                            "retired": sum(not m["complete"] for m in training["members"]),
                            "max_skipped": max(m["skipped"] for m in skips), "skip_cap": spec["skip_cap"]},
               "pass": (all(m["complete"] for m in training["members"])
                        and all(m["skipped"] <= spec["skip_cap"] for m in skips))},
        "G3": {"observed": {"records": len(records), "all_carry_policy": policies, "spot_replication": spot},
               "pass": policies and spot is not None and spot["pass"]},
        "G4": {"observed": {"worst_nonfinite_share": worst_nonfinite}, "pass": worst_nonfinite <= impl.NONFINITE_SHARE_MAX},
        "no_heldout": {"observed": {"heldout_records": len(list(output.glob("records/heldout--*.json")))},
                       "pass": not any(output.glob("records/heldout--*.json"))},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"], "wall_seconds": ledger["wall_seconds_elapsed"]},
                 "pass": (ledger["gpu_seconds_elapsed"] <= impl.GPU_CAP_SECONDS
                          and ledger["wall_seconds_elapsed"] <= impl.WALL_CAP_SECONDS)},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    grid = lolo[impl.PRIMARY]["contrasts"][impl.PRIMARY_COST]
    arms = set(impl.ARMS)
    q1 = grid["L-B"]["request_mean"]["estimate"] if {"L", "B"} <= arms else None
    q2 = grid["C-B"]["request_mean"]["estimate"] if {"C", "B"} <= arms else None
    selected_block = lolo[impl.PRIMARY]["rankers"][impl.PRIMARY_COST][f"{best['arm']}:{best['request']}"]["auc"]
    q3 = impl.STOP_TOKEN if not guards_ok else ("supported" if impl.meets(selected_block)
                                                 else "not_supported_by_this_experiment")
    controls = controls_table(impl, universe, selection)
    q5 = controls[impl.PRIMARY]["right_minus_wrong"]
    previous = previous_best(version)
    gain = best["score"] - previous["score"]
    prior = c2.prior_tables(universe)
    return {
        "schema": f"issue_112_compute_v{version}", "identity": impl.IDENTITY, "version": version, "spec": spec,
        "guards": guards, "guards_pass": guards_ok,
        "Q1_long_horizon": {"estimate": q1, "token": impl.token(q1, guards_ok), "label": "EXPLORATORY"},
        "Q2_contrastive": {"estimate": q2, "token": impl.token(q2, guards_ok), "label": "EXPLORATORY"},
        "Q3_selected_target": {"selected": {"arm": best["arm"], "request": best["request"],
                                            "family": t96.arm_family(best["request"]), "score": best["score"]},
                               "lolo_grid_auc": selected_block, "token": q3, "label": "EXPLORATORY"},
        "Q5_anchor_dependence": {"unit": controls[impl.PRIMARY]["unit"], "estimate": q5,
                                 "stability": controls[impl.PRIMARY]["stability"],
                                 "token": impl.token(q5, guards_ok), "label": "EXPLORATORY"},
        "controls": controls,
        "action_prior_lolo": {k: v for k, v in prior.items() if k.endswith(":lolo")},
        "series": {"previous": previous, "gain_in_S": gain, "stop_delta": STOP_DELTA,
                   "g2_passed": guards["G2"]["pass"],
                   "continue": (version < LAST_VERSION and (not guards["G2"]["pass"] or gain >= STOP_DELTA))},
        "selection": {"scores": scores, "frozen_at": selection["frozen_at"]},
        "lolo": lolo, "rollout": {"lolo": curves, "lolo_nonfinite": nonfinite,
                                   "lolo_contrasts": impl.rollout_contrasts(per, impl.ARMS, "B")},
        "training": {**training, "skips": skips, "skips_per_arm": skips_per_arm},
        "compute": {"phases": ledger["phases"], "gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "wall_seconds": ledger["wall_seconds_elapsed"], "engine_seconds": 0},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def cell(block):
    return "n/a" if block is None else fmt_interval(block)


def findings_md(version, plan, compute):
    impl = implementation(version)
    lines = []
    add = lines.append
    spec, series = compute["spec"], compute["series"]
    sel = compute["Q3_selected_target"]["selected"]
    add(f"# Issue #112 v{version} — {spec['change'].split(':')[0]}")
    add("")
    add(f"Frozen {plan['frozen_at']}; selection {compute['selection']['frozen_at']}. Validation: "
        f"`{impl.VALIDATION_COMMAND}`. Leave-one-lineage-out (LOLO) only; EXPLORATORY; held-out not scored.")
    add("")
    add(f"Diagnosis (from v{spec['parent']}): {spec['diagnosis']}")
    add("")
    add(f"Change: {spec['change']}.")
    add("")
    guard_text = ", ".join(f"{name} {block['pass']}" for name, block in compute["guards"].items())
    add(f"Guards pass: {compute['guards_pass']} ({guard_text}).")
    g2 = compute["guards"]["G2"]["observed"]
    add(f"Training stability: {g2['retired']} of {g2['members']} members retired; max skipped updates per member "
        f"{g2['max_skipped']} (cap {g2['skip_cap']}).")
    add("")
    add("| question | estimate | token |")
    add("| --- | --- | --- |")
    add(f"| Q1 long horizon L − B (grid, request-mean) | {cell(compute['Q1_long_horizon']['estimate'])} | "
        f"{compute['Q1_long_horizon']['token']} |")
    add(f"| Q2 contrastive C − B (grid, request-mean) | {cell(compute['Q2_contrastive']['estimate'])} | "
        f"{compute['Q2_contrastive']['token']} |")
    add(f"| Q3 selected {sel['arm']}:{sel['request']} grid AUC (target {impl.TARGET_AUC}, lower > {impl.TARGET_LOWER}) "
        f"| {cell(compute['Q3_selected_target']['lolo_grid_auc'])} | {compute['Q3_selected_target']['token']} |")
    q5 = compute["Q5_anchor_dependence"]
    add(f"| Q5 anchor dependence {q5['unit']} (grid, right − wrong anchor) | {cell(q5['estimate'])} | {q5['token']} |")
    add("")
    add(f"Series: selected S = {sel['score']:.4f}; previous (v{series['previous']['version']} "
        f"{series['previous']['arm']}:{series['previous']['request']}) S = {series['previous']['score']:.4f}; gain "
        f"{series['gain_in_S']:+.4f} (stop below {series['stop_delta']}); G2 passed {series['g2_passed']}; "
        f"continue: {series['continue']}.")
    add("")
    for inventory, table in compute["lolo"].items():
        add(f"## {inventory} ({len(table['members'])} lineages, {table['cells']} cells), tie-free AUC")
        add("")
        add("| request | " + " | ".join(impl.ARMS) + " |")
        add("| --- | " + " | ".join("---" for _ in impl.ARMS) + " |")
        for request in impl.REQUESTS:
            add(f"| {request} | " + " | ".join(cell(table["rankers"][impl.PRIMARY_COST][f"{a}:{request}"]["auc"])
                                               for a in impl.ARMS) + " |")
        add("")
        add("| contrast (request-mean) | estimate |")
        add("| --- | --- |")
        for name, block in table["contrasts"][impl.PRIMARY_COST].items():
            add(f"| {name} | {cell(block['request_mean']['estimate'])} |")
        add("")
        prior = compute["action_prior_lolo"].get(f"{inventory}:lolo")
        if prior:
            add(f"No-model action prior (LOLO): {cell(prior['auc'])}.")
            add("")
    add("## Selection (S = mean of grid and offset LOLO AUC), top 8")
    add("")
    add("| rank | arm | request | S |")
    add("| --- | --- | --- | --- |")
    top = sorted((s for s in compute["selection"]["scores"] if s["score"] is not None), key=lambda s: -s["score"])[:8]
    for rank, row in enumerate(top, 1):
        add(f"| {rank} | {row['arm']} | {row['request']} | {row['score']:.4f} |")
    add("")
    add("## Wrong-anchor control of the selected (arm, family)")
    add("")
    add("| inventory | unit | right − wrong | rank stability |")
    add("| --- | --- | --- | --- |")
    for inventory, block in compute["controls"].items():
        add(f"| {inventory} | {block['unit']} | {cell(block['right_minus_wrong'])} | {cell(block['stability'])} |")
    add("")
    add("## Rollout error at t = 225 (#100 engine-referenced metric, LOLO)")
    add("")
    add("| request | " + " | ".join(impl.ARMS) + " |")
    add("| --- | " + " | ".join("---" for _ in impl.ARMS) + " |")
    for request in impl.REQUESTS:
        add(f"| {request} | " + " | ".join(
            impl.num((compute["rollout"]["lolo"][f"{a}:{request}:{impl.ENDPOINT}:engine_position_mse"] or {}).get("mean"))
            for a in impl.ARMS) + " |")
    add("")
    add("## Training")
    add("")
    add("| arm:family | complete | members with skips | max skips | GPU s | final local | final long | final rank |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, block in compute["training"]["per_arm"].items():
        s = compute["training"]["skips_per_arm"][key]
        add(f"| {key} | {block['complete']}/{block['members']} | {s['members_with_skips']} | {s['max_skipped']} | "
            f"{block['gpu_seconds']:.0f} | {impl.num(block['mean_final_local'])} | {impl.num(block['mean_final_long'])} "
            f"| {impl.num(block['mean_final_rank'])} |")
    add("")
    add(f"GPU seconds {compute['compute']['gpu_seconds']:.0f}; wall {compute['compute']['wall_seconds']:.0f}; "
        "engine seconds 0.")
    add("")
    return "\n".join(lines)


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "inventory", "unit", "statistic", "value", "interval_low", "interval_high", "clusters", "label"])
    for inventory, table in compute["lolo"].items():
        for cost, rankers in table["rankers"].items():
            for unit, block in rankers.items():
                b = block["auc"]
                writer.writerow([f"lolo:{inventory}:{cost}:{unit}", inventory, unit, "auc",
                                 *(("", "", "", "") if b is None else (f"{b['mean']:.6f}", f"{b['interval'][0]:.6f}",
                                                                        f"{b['interval'][1]:.6f}", b["clusters"])),
                                 "EXPLORATORY"])
    return buffer.getvalue()


def summary(version, plan, compute):
    return {"schema": f"issue_112_report_v{version}", "identity": compute["identity"], "version": version,
            "frozen_at": plan["frozen_at"], "spec": compute["spec"], "guards_pass": compute["guards_pass"],
            "questions": {k: {"token": compute[k]["token"], "label": compute[k]["label"]}
                          for k in ("Q1_long_horizon", "Q2_contrastive", "Q3_selected_target", "Q5_anchor_dependence")},
            "selected": compute["Q3_selected_target"]["selected"], "series": compute["series"], "engine_seconds": 0}


def rendered(version, plan, compute):
    return {"summary.json": json_text(summary(version, plan, compute)),
            "findings.md": findings_md(version, plan, compute), "comparisons.csv": comparisons_csv(compute)}


def publish(version):
    impl, plan = load_plan(version)
    _, _, universe = impl.load_universe()
    with harness.deterministic_scoring():
        spot = spot_check(impl, universe)
    compute = compute_tables(version, spot)
    write_json(impl.OUTPUT / "compute.json", compute)
    for name, text in rendered(version, plan, compute).items():
        (impl.OUTPUT / name).write_bytes(text.encode())
    log(f"v{version} published: Q1 {compute['Q1_long_horizon']['token']}, Q2 {compute['Q2_contrastive']['token']}, "
        f"Q3 {compute['Q3_selected_target']['token']}, Q5 {compute['Q5_anchor_dependence']['token']}; "
        f"guards pass {compute['guards_pass']}; continue {compute['series']['continue']}")
    return 0


def validate(version):
    impl, plan = load_plan(version)
    began = time.monotonic()
    with harness.deterministic_scoring():
        selection = read_json(impl.OUTPUT / "selection.json")
        if selection != impl.selection_record(impl.OUTPUT, plan, selection["frozen_at"]):
            raise ValueError("selection.json differs from the frozen rule applied to the retained records")
        _, _, universe = impl.load_universe()
        spot = spot_check(impl, universe)
    fresh = compute_tables(version, spot)
    if read_json(impl.OUTPUT / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(version, plan, fresh).items():
        if (impl.OUTPUT / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"v{version} validation passed: selection re-derived, one record re-scored, tables byte-compared "
        f"({time.monotonic() - began:.1f}s)")
    return 0


MODES = {"dry-run": (dry_run, False), "smoke": (smoke, True), "prepare": (prepare, False), "data": (data, True),
         "train": (train, True), "score-lolo": (score_lolo, True), "select": (select, False),
         "score-controls": (score_controls, True), "publish": (publish, True), "validate": (validate, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    function, locked = MODES[mode]
    try:
        if locked:
            with wlc.GPULock():
                return function(args.version)
        return function(args.version)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
