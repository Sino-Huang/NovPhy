"""Issue-109 work item 1: #96 contrasts re-scored on held-out N1 members only.

Binding runner module: scripts/run_heldout_rescore.py
Exact validation command: python -u -m scripts.run_heldout_rescore --validate

Question: does the submitted #96 reading (primary J - F* on the grid
readiness_or_precision_insufficient; secondaries J - M and J - OL
readiness_or_precision_insufficient) change when the scoring is restricted to
the N1 members that no #77 predictor or controller was fit on
(issue-77-n1-007, -008, -014, -016)?

This is a sanity check of submitted results, not a new claim. Zero engine
seconds, zero GPU seconds, no rollout: every number is re-derived from the
retained, published #96 decision records (read-only) and the #96 verdict
universe (scripts.run_tau_ad_within_checkpoint.build_universe), filtered by
pruning the member set per inventory. Three cohorts, never pooled with each
other, and four inventories, never pooled with each other:

- all       the published 15-member #96 universe. Guard: must reproduce the #96
            per-cell rows exactly and the published per-arm member-mean AUCs,
            contrasts, guard inputs and tokens to the printed 4 decimals, else
            the run stops with an error.
- held_out  issue-77-n1-007, -008, -014, -016.
- exposed   the fit complement: #77 predictor and controller lineages present in
            the universe (context only).

The F* / F_hindsight / C-F* arm choices are the PUBLISHED #96 choices, frozen and
never reselected in-cohort: the #96 cross-fit already used every member, so an
in-cohort reselection would not make the held-out estimate any cleaner. The
in-cohort F_hindsight is reported as a descriptive extra only.

Modes (mutually exclusive):
- --dry-run   structural inventory only (cohorts, members, mixed cells); computes
              no statistic and writes nothing.
- --prepare   writes the frozen plan.json once; refuses to overwrite.
- --publish   compute.json / summary.json / comparisons.csv / findings.md and the
              wall-clock ledger.json.
- --validate  recomputes every table and byte-compares the published files.

Every interval is DESCRIPTIVE (member-clustered percentile bootstrap, PCG64 seed
7201, 10000 draws, exactly as #96).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts import run_tau_ad_within_checkpoint as t96

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-109-heldout-rescore-v1"
SOURCE = t96.OUTPUT
DYNAMICS_PLAN = t96.DYNAMICS / "plan.json"

IDENTITY = "issue-109-heldout-rescore-v1"
SCHEMA_PLAN = "issue_109_heldout_rescore_plan_v1"
SCHEMA_COMPUTE = "issue_109_heldout_rescore_compute_v1"
SCHEMA_REPORT = "issue_109_heldout_rescore_report_v1"
SCHEMA_LEDGER = "issue_109_heldout_rescore_ledger_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_heldout_rescore --validate"
RUNNER = "scripts/run_heldout_rescore.py"

HELD_OUT = ("issue-77-n1-007", "issue-77-n1-008", "issue-77-n1-014", "issue-77-n1-016")
COHORTS = ("all", "held_out", "exposed")
TOKEN_COHORTS = ("all", "held_out")
MIN_CLUSTERS = 3
UNINFORMATIVE = "interval_uninformative"
INVENTORIES = t96.INVENTORIES
PRIMARY = t96.PRIMARY
ARMS = t96.ARMS
FIXED_ARMS = t96.FIXED_ARMS
PAIR_NAMES = t96.PAIR_NAMES
SEEDS = t96.SEEDS
DELTA = t96.DELTA
STOP_TOKEN = t96.STOP_TOKEN
PUBLISHED_CONTRASTS = t96.CONTRAST_ORDER
IN_COHORT_HINDSIGHT = "J-F_hindsight_in_cohort"
CONTRASTS = (*PUBLISHED_CONTRASTS, IN_COHORT_HINDSIGHT)
SECONDARIES = ("J-M", "J-OL")
COHORT_GUARDS = ("G3", "G4", "G5", "G6")
INHERITED_GUARDS = ("G1", "G2", "caps")
GUARD_INPUT_KEYS = ("verdict_resolution", "nonfinite_share", "tied_cells", "tied_share",
                    "J_modal_pair", "J_nonmodal_step_share", "rank_divergent_cells",
                    "rank_divergent_share")
PLAN_MARKERS = (*t96.PLACEHOLDER_MARKERS, "~", "e.g.")

PINNED = {
    "issue_96_plan": SOURCE / "plan.json",
    "issue_96_compute": SOURCE / "compute.json",
    "issue_96_summary": SOURCE / "summary.json",
    "issue_96_findings": SOURCE / "findings.md",
    "issue_96_comparisons": SOURCE / "comparisons.csv",
    "issue_96_frequencies": SOURCE / "frequencies.json",
    "issue_96_ledger": SOURCE / "ledger.json",
    "issue_96_runner": ROOT / "scripts/run_tau_ad_within_checkpoint.py",
    "issue_94_runner_bootstrap_and_cell_auc": ROOT / "scripts/run_launch_power_probe.py",
    "issue_77_dynamics_plan": DYNAMICS_PLAN,
}

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
bootstrap = t96.bootstrap
fmt = t96.fmt
fmt_interval = t96.fmt_interval


def log(message):
    print(f"[issue-109-heldout-rescore] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative(path):
    return str(Path(path).relative_to(ROOT))


# ---------------------------------------------------------------------------
# membership (no statistic)
# ---------------------------------------------------------------------------

def fit_lineages():
    lineages = read_json(DYNAMICS_PLAN)["n1_lineages"]
    return {"predictor": sorted(lineages["predictor"]), "controller": sorted(lineages["controller"])}


def cohort_members(members, fit):
    """all / held_out / exposed member sets; held_out and exposed partition the universe."""
    members = sorted(members)
    fitted = set(fit["predictor"]) | set(fit["controller"])
    missing = sorted(set(HELD_OUT) - set(members))
    if missing:
        raise ValueError(f"held-out members {missing} are absent from the #96 universe")
    leaked = sorted(set(HELD_OUT) & fitted)
    if leaked:
        raise ValueError(f"held-out members {leaked} are #77 fit lineages")
    exposed = [m for m in members if m in fitted]
    unassigned = sorted(set(members) - set(HELD_OUT) - set(exposed))
    if unassigned:
        raise ValueError(f"universe members {unassigned} are neither held out nor fit lineages")
    return {"all": members, "held_out": sorted(HELD_OUT), "exposed": exposed}


def prune(universe, keep):
    keep = set(keep)
    return {inventory: {m: e for m, e in entries.items() if m in keep}
            for inventory, entries in universe.items()}


def structure(universe, cohorts):
    """Per cohort and inventory: members, mixed members, mixed cells (no statistic)."""
    out = {}
    for cohort in COHORTS:
        pruned = prune(universe, cohorts[cohort])
        out[cohort] = {}
        for inventory in INVENTORIES:
            entries = pruned[inventory]
            mixed_members = sorted(m for m, e in entries.items() if t96.mixed(e))
            out[cohort][inventory] = {"members": sorted(entries), "mixed_members": mixed_members,
                                      "mixed_cells": len(mixed_members) * len(SEEDS)}
    return out


def load_universe():
    sources = t96.load_sources()
    members, universe, excluded = t96.build_universe(sources)
    return members, universe, excluded


def published_choices(compute96):
    return {inventory: {key: compute96["inventories"][inventory][key]
                        for key in ("F_star", "F_hindsight", "C_F_star")}
            for inventory in INVENTORIES}


def published_tokens(compute96):
    return {"J-F*": compute96["primary"]["token"],
            **{name: compute96["secondary_tokens"][name]["token"] for name in SECONDARIES}}


def records_digest():
    names = sorted(path.name for path in (SOURCE / "records").glob("*.json"))
    digest = sha256()
    for name in names:
        digest.update(name.encode())
        digest.update(sha256_of(SOURCE / "records" / name).encode())
    return {"records": len(names), "sha256_of_sorted_name_and_sha256": "sha256:" + digest.hexdigest()}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def frozen_plan(frozen_at, members, universe, excluded):
    compute96 = read_json(SOURCE / "compute.json")
    plan96 = read_json(SOURCE / "plan.json")
    fit = fit_lineages()
    cohorts = cohort_members(members, fit)
    layout = structure(universe, cohorts)
    for inventory in INVENTORIES:
        if layout["all"][inventory]["mixed_members"] != plan96["universe"]["mixed_members"][inventory]:
            raise ValueError(f"{inventory}: all-cohort mixed members differ from the #96 plan")
    choices = published_choices(compute96)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "issue": 109, "work_item": 1,
        "frozen_at": frozen_at, "frozen_before_scoring_run": True,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER),
        "engine_seconds": 0, "gpu_seconds": 0, "issue_64_authorized": False,
        "role": ("sanity check of the submitted #96 results, not a new claim; the #96 disposition "
                 "(primary and secondary tokens) is unchanged regardless of the held-out result"),
        "question": ("Does the submitted #96 reading (primary J - F* grid "
                     "readiness_or_precision_insufficient; J - M, J - OL "
                     "readiness_or_precision_insufficient) change when restricted to held-out "
                     "members?"),
        "disclosure": ("Designed after #96 was published. The #96 all-member numbers and the "
                       "retained per-cell rows of #96 compute.json (from which member subsets are "
                       "derivable) were public before this freeze; no held-out-only or "
                       "exposed-only statistic was computed before this freeze. The published F*, "
                       "F_hindsight and C-F* arm choices were selected on all 15 members (the F* "
                       "cross-fit uses the other inventories' all-member means, which include the "
                       "held-out members), so held-out outcomes already informed the arm choice; "
                       "reselecting in-cohort would not remove that dependence and is not done."),
        "membership": {
            "source_universe": "scripts.run_tau_ad_within_checkpoint.build_universe (15 members)",
            "fit_lineages": fit,
            "fit_lineages_source": relative(DYNAMICS_PLAN) + " n1_lineages/{predictor,controller}",
            "cohorts": cohorts,
            "cohort_rules": {
                "all": "the published #96 universe, every member",
                "held_out": "issue-77-n1-007, -008, -014, -016: in no #77 predictor or controller fit",
                "exposed": ("the fit complement: #77 predictor and controller lineages present in "
                            "the universe; context only"),
                "pooling": "cohorts are never pooled with each other; inventories are never pooled",
            },
            "excluded": excluded,
            "per_inventory": layout,
            "expected_mixed_cells": {cohort: {i: layout[cohort][i]["mixed_cells"] for i in INVENTORIES}
                                     for cohort in COHORTS},
        },
        "published_arm_choices": choices,
        "published_arm_choice_rule": ("F*, F_hindsight and C-F* are the #96 published choices per "
                                      "inventory, frozen for every cohort; never reselected "
                                      "in-cohort"),
        "published_tokens": published_tokens(compute96),
        "estimands": {
            "cell_auc": "the #96 cell AUC (run_tau_ad_within_checkpoint.cell_view), unchanged",
            "rows": ("run_tau_ad_within_checkpoint.inventory_rows over the cohort's mixed members; "
                     "cells = mixed member x 3 seeds"),
            "per_arm": "member-mean AUC (run_tau_ad_within_checkpoint.member_mean) for all 18 arms",
            "contrasts": {
                **{name: "run_tau_ad_within_checkpoint.contrast with the published arm choice"
                   for name in PUBLISHED_CONTRASTS},
                IN_COHORT_HINDSIGHT: ("J - F_h(cohort), F_h(cohort) = the fixed pair with the "
                                      "highest cohort member-mean AUC, ties by PAIRS order; "
                                      "descriptive extra, no token"),
            },
            "fixed_pair_spread": "member-clustered mean of per-cell max_F - min_F AUC, as #96",
            "top1": "J and F* top-1 hit counts over the cohort's mixed cells, and mean chance",
            "interval": (f"member-clustered percentile bootstrap, {t96.BOOTSTRAP_DRAWS} draws, PCG64 "
                         f"seed {t96.BOOTSTRAP_SEED}, quantiles {list(t96.INTERVAL_QUANTILES)}, "
                         "DESCRIPTIVE"),
            "cluster_flag": (f"a contrast with fewer than {MIN_CLUSTERS} member clusters is flagged "
                             f"{UNINFORMATIVE}: the point estimate is reported and the interval is "
                             "still computed but carries no precision reading"),
            "exposure_optimism": ("per inventory, J - F* point estimate held_out minus exposed; "
                                  "descriptive, no interval"),
        },
        "guards": {
            "all_member_reproduction": ("the all cohort must reproduce the #96 compute.json per-cell "
                                        "rows exactly, and the published per-arm member-mean AUCs "
                                        "(18 arms x 4 inventories), all nine published contrasts "
                                        "(estimate and interval endpoints), fixed-pair spread, "
                                        "F_hindsight, top-1 counts, grid guard inputs and the three "
                                        "tokens to 4 decimals; any mismatch stops the run with an "
                                        "error and nothing is published"),
            "structure": ("cohort mixed cells must equal expected_mixed_cells; held_out and exposed "
                          "must partition the universe; no held-out member is a fit lineage"),
            "inputs": "every pinned input and the #96 record digest are re-verified at every load",
        },
        "decision_rule": {
            "token": (f"run_tau_ad_within_checkpoint.token with delta {DELTA}: supported if all "
                      "guards pass AND estimate >= delta AND lower > 0; "
                      "not_supported_by_this_experiment if all guards pass AND upper < delta; "
                      "readiness_or_precision_insufficient otherwise"),
            "guards_per_cohort": ("G1, G2 and caps are cohort-independent and inherited from the "
                                  "published #96 guards; G3 (verdict resolution 1.0, nonfinite "
                                  f"share <= {t96.NONFINITE_SHARE_MAX}), G4 (tied share <= "
                                  f"{t96.TIE_SHARE_MAX}), G5 (J non-modal step share >= "
                                  f"{t96.NONMODAL_SHARE_MIN} AND rank-divergent share >= "
                                  f"{t96.RANK_DIVERGENCE_SHARE_MIN}) and G6 (primary half-width <= "
                                  f"{t96.HALF_WIDTH_MAX}) are recomputed on the cohort's mixed grid "
                                  "cells with the #96 formulas"),
            "primary": "J - F* on the grid: all guards AND not interval_uninformative",
            "secondaries": ("J - M and J - OL on the grid: G1-G5 and caps, the secondary's own "
                            f"half-width <= {t96.HALF_WIDTH_MAX}, and not interval_uninformative"),
            "cohorts_tokened": list(TOKEN_COHORTS),
            "record": ("the held-out token is recorded next to the published #96 token with a "
                       "changed flag; the #96 disposition is unchanged regardless"),
        },
        "stop_rules": ["all-member reproduction mismatch -> error, nothing published",
                       "pinned input or #96 record digest changed -> error",
                       "cohort structure differs from the frozen membership -> error",
                       "CUDA initialized during scoring -> error (GPU seconds must be 0)"],
        "inputs": [{"name": name, "artifact": relative(path), "sha256": sha256_of(path)}
                   for name, path in PINNED.items()],
        "issue_96_records": records_digest(),
        "claim_boundary": ("a held-out-member re-scoring of the retained #96 decision records; the "
                           "held-out cohort has 2 to 3 member clusters per inventory, so its "
                           "intervals are descriptive and wide; the arm choices were made on all "
                           "members; nothing here re-opens or edits the #96 disposition, the "
                           "ICLR 2026 submission or any prior artifact; zero engine and GPU "
                           "seconds; cohorts and inventories are never pooled"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-109 held-out re-score protocol")
    if not plan.get("frozen_before_scoring_run"):
        raise ValueError("plan.json does not declare a pre-scoring freeze")
    blob = json_text(plan)
    for marker in PLAN_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a disallowed marker {marker!r}")
    for entry in plan["inputs"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"pinned input {entry['name']} changed after the freeze")
    if records_digest() != plan["issue_96_records"]:
        raise ValueError("the #96 decision records changed after the freeze")
    t96.load_plan(SOURCE)
    compute96 = read_json(SOURCE / "compute.json")
    if published_choices(compute96) != plan["published_arm_choices"]:
        raise ValueError("published #96 arm choices differ from the frozen copy")
    if published_tokens(compute96) != plan["published_tokens"]:
        raise ValueError("published #96 tokens differ from the frozen copy")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").exists():
        raise ValueError("plan.json already frozen; refusing to overwrite")
    members, universe, excluded = load_universe()
    plan = frozen_plan(utc_now(), members, universe, excluded)
    blob = json_text(plan)
    for marker in PLAN_MARKERS:
        if marker in blob:
            raise ValueError(f"plan contains a disallowed marker {marker!r}; not frozen")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(blob)
    load_plan(output)
    log(f"frozen plan published at {plan['frozen_at']}: cohorts "
        f"{ {c: len(plan['membership']['cohorts'][c]) for c in COHORTS} }; no statistic computed")
    return 0


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def flag(block):
    clusters = block["members"]
    return block | {"clusters": clusters, UNINFORMATIVE: clusters < MIN_CLUSTERS}


def guard_inputs(rows, entries, f_star):
    """The #96 grid guard ingredients (compute_tables), on an arbitrary row subset."""
    resolved = sum(r["arms"][arm]["resolved"] for r in rows for arm in ARMS)
    slots = sum(r["arms"][arm]["verdict_slots"] for r in rows for arm in ARMS)
    nonfinite = sum(r["arms"][arm]["nonfinite"] for r in rows for arm in ARMS)
    tied = sum(1 for r in rows if r["arms"]["J"]["all_tied"] or r["arms"][f_star]["all_tied"])
    steps = Counter()
    for r in rows:
        for text in r["arms"]["J"]["schedules"].values():
            steps.update(t96.unrle(text))
    total = sum(steps.values())
    modal = max(range(len(PAIR_NAMES)), key=lambda k: (steps[k], -k)) if total else 0
    modal_arm = f"F-{PAIR_NAMES[modal]}"
    divergent = 0
    for r in rows:
        j_costs, f_costs = r["arms"]["J"]["costs"], r["arms"][modal_arm]["costs"]
        common = sorted(o for o in j_costs if o in entries[r["member"]]["verdicts"]
                        and j_costs[o] is not None and f_costs.get(o) is not None)
        tau = t96.kendall([j_costs[o] for o in common], [f_costs[o] for o in common])
        divergent += int(tau is not None and tau < 1.0 - 1e-12)
    return {"verdict_resolution": resolved / slots if slots else None,
            "nonfinite_share": nonfinite / slots if slots else None,
            "tied_cells": tied, "tied_share": tied / len(rows) if rows else None,
            "J_modal_pair": PAIR_NAMES[modal],
            "J_nonmodal_step_share": 1 - steps[modal] / total if total else None,
            "rank_divergent_cells": divergent,
            "rank_divergent_share": divergent / len(rows) if rows else None}


def inventory_block(rows, entries, choice):
    member_means = {arm: t96.member_mean([{"member": r["member"], "auc": r["arms"][arm]["auc"]}
                                          for r in rows], "auc") for arm in ARMS}
    hindsight = max(FIXED_ARMS, key=lambda arm: (
        -math.inf if member_means[arm] is None else member_means[arm], -FIXED_ARMS.index(arm)))
    f_star, c_star = choice["F_star"], choice["C_F_star"]
    pairs = {"J-F*": ("J", f_star), "J-F_hindsight": ("J", choice["F_hindsight"]),
             "J-F(1,macro)": ("J", "F-1-macro"), "J-F(1,continuous)": ("J", "F-1-continuous"),
             "J-M": ("J", "M"), "J-OL": ("J", "OL"), "J-T": ("J", "T"), "J-D": ("J", "D"),
             "C-J-C-F*": ("C-J", c_star), IN_COHORT_HINDSIGHT: ("J", hindsight)}
    contrasts = {name: flag(t96.contrast(rows, *pairs[name])) for name in CONTRASTS}
    spread_rows = []
    for r in rows:
        aucs = [r["arms"][arm]["auc"] for arm in FIXED_ARMS if r["arms"][arm]["auc"] is not None]
        if aucs:
            spread_rows.append((r["member"], max(aucs) - min(aucs)))
    spread = bootstrap([s for _, s in spread_rows], [m for m, _ in spread_rows])
    chance = [r["arms"]["J"]["chance_top1"] for r in rows if r["arms"]["J"]["chance_top1"] is not None]
    return {
        "mixed_members": sorted({r["member"] for r in rows}), "mixed_cells": len(rows),
        "F_star": f_star, "F_hindsight": choice["F_hindsight"], "C_F_star": c_star,
        "F_hindsight_in_cohort": hindsight,
        "arm_member_mean_auc": member_means,
        "contrasts": contrasts,
        "top1": {"J_hits": sum(1 for r in rows if r["arms"]["J"]["top1_hit"]),
                 "F_star_hits": sum(1 for r in rows if r["arms"][f_star]["top1_hit"]),
                 "cells": len(rows), "chance_mean": float(np.mean(chance)) if chance else None},
        "fixed_pair_spread": spread,
        "guard_inputs": guard_inputs(rows, entries, f_star),
    }


def cohort_tokens(grid, published_guards):
    gi = grid["guard_inputs"]
    primary = grid["contrasts"]["J-F*"]
    estimate = primary["estimate"]
    guards = {name: published_guards[name]["pass"] for name in INHERITED_GUARDS}
    guards["G3"] = gi["verdict_resolution"] == 1.0 and gi["nonfinite_share"] <= t96.NONFINITE_SHARE_MAX
    guards["G4"] = gi["tied_share"] is not None and gi["tied_share"] <= t96.TIE_SHARE_MAX
    guards["G5"] = (gi["J_nonmodal_step_share"] is not None
                    and gi["J_nonmodal_step_share"] >= t96.NONMODAL_SHARE_MIN
                    and gi["rank_divergent_share"] >= t96.RANK_DIVERGENCE_SHARE_MIN)
    guards["G6"] = estimate is not None and estimate["half_width"] <= t96.HALF_WIDTH_MAX
    all_ok = all(guards.values())
    base_ok = all(guards[g] for g in ("G1", "G2", "G3", "G4", "G5", "caps"))
    tokens = {"J-F*": {"token": t96.token(estimate, all_ok and not primary[UNINFORMATIVE]),
                       "half_width": None if estimate is None else estimate["half_width"],
                       UNINFORMATIVE: primary[UNINFORMATIVE]}}
    for name in SECONDARIES:
        block = grid["contrasts"][name]
        est = block["estimate"]
        own = est is not None and est["half_width"] <= t96.HALF_WIDTH_MAX
        tokens[name] = {"token": t96.token(est, base_ok and own and not block[UNINFORMATIVE]),
                        "half_width": None if est is None else est["half_width"],
                        "own_half_width_pass": own, UNINFORMATIVE: block[UNINFORMATIVE]}
    return {"guards": guards, "guards_pass": all_ok, "tokens": tokens}


def published_rows(rows):
    return [{"member": r["member"], "seed": r["seed"],
             "auc": {arm: r["arms"][arm]["auc"] for arm in ARMS},
             "top1_hit": {arm: r["arms"][arm]["top1_hit"] for arm in ARMS},
             "admissible": len(r["arms"]["J"]["admissible"])} for r in rows]


def reproduction_check(all_blocks, all_rows, all_tokens, compute96):
    """All-member guard: #96 rows exactly; published numbers to the printed 4 decimals."""
    mismatches, compared = [], 0

    def same(label, mine, theirs, formatter=fmt):
        nonlocal compared
        compared += 1
        if formatter(mine) != formatter(theirs):
            mismatches.append(f"{label}: {formatter(mine)} != published {formatter(theirs)}")

    ident = str
    for inventory in INVENTORIES:
        pub = compute96["inventories"][inventory]
        mine = all_blocks[inventory]
        same(f"{inventory} rows", published_rows(all_rows[inventory]) == pub["rows"], True, ident)
        same(f"{inventory} mixed members", mine["mixed_members"], pub["mixed_members"], ident)
        same(f"{inventory} F_hindsight in-cohort", mine["F_hindsight_in_cohort"], pub["F_hindsight"], ident)
        for arm in ARMS:
            same(f"{inventory} AUC {arm}", mine["arm_member_mean_auc"][arm], pub["arm_member_mean_auc"][arm])
        for name in PUBLISHED_CONTRASTS:
            a, b = mine["contrasts"][name], pub["contrasts"][name]
            same(f"{inventory} {name} cells", (a["cells"], a["members"], a["undefined_cells"]),
                 (b["cells"], b["members"], b["undefined_cells"]), ident)
            same(f"{inventory} {name}", a["estimate"], b["estimate"], fmt_interval)
        same(f"{inventory} spread", mine["fixed_pair_spread"], pub["fixed_pair_spread"], fmt_interval)
        for key in ("J_hits", "F_star_hits", "cells"):
            same(f"{inventory} top1 {key}", mine["top1"][key], pub["top1"][key], ident)
        same(f"{inventory} top1 chance", mine["top1"]["chance_mean"], pub["top1"]["chance_mean"])
        for key in GUARD_INPUT_KEYS:
            theirs = pub["guard_inputs"][key]
            same(f"{inventory} guard input {key}", mine["guard_inputs"][key], theirs,
                 ident if isinstance(theirs, (str, int)) else fmt)
    for name, token in published_tokens(compute96).items():
        same(f"token {name}", all_tokens["tokens"][name]["token"], token, ident)
    for name in COHORT_GUARDS:
        same(f"guard {name}", all_tokens["guards"][name], compute96["guards"][name]["pass"], ident)
    if mismatches:
        raise ValueError("all-member cohort does not reproduce #96: " + "; ".join(mismatches))
    return {"pass": True, "values_compared": compared,
            "rule": ("#96 compute.json rows exact; per-arm member-mean AUCs, nine contrasts, spread, "
                     "top-1, F_hindsight, grid guard inputs, G3-G6 and three tokens to 4 decimals")}


def compute_tables(plan):
    import torch
    members, universe, _ = load_universe()
    cohorts = cohort_members(members, fit_lineages())
    if cohorts != plan["membership"]["cohorts"]:
        raise ValueError("cohort membership differs from the frozen plan")
    layout = structure(universe, cohorts)
    if layout != plan["membership"]["per_inventory"]:
        raise ValueError("cohort structure differs from the frozen plan")
    compute96 = read_json(SOURCE / "compute.json")
    choices = plan["published_arm_choices"]
    blocks, rows_all, tokens = {}, {}, {}
    for cohort in COHORTS:
        pruned = prune(universe, cohorts[cohort])
        blocks[cohort] = {}
        for inventory in INVENTORIES:
            rows = t96.inventory_rows(SOURCE, pruned, inventory)
            if len(rows) != plan["membership"]["expected_mixed_cells"][cohort][inventory]:
                raise ValueError(f"{cohort}/{inventory}: mixed cells differ from the frozen count")
            if cohort == "all":
                rows_all[inventory] = rows
            blocks[cohort][inventory] = inventory_block(rows, pruned[inventory], choices[inventory])
        if cohort in TOKEN_COHORTS:
            tokens[cohort] = cohort_tokens(blocks[cohort][PRIMARY], compute96["guards"])
    reproduction = reproduction_check(blocks["all"], rows_all, tokens["all"], compute96)
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized during scoring; GPU seconds must be 0")
    optimism = {}
    for inventory in INVENTORIES:
        held = blocks["held_out"][inventory]["contrasts"]["J-F*"]["estimate"]
        exposed = blocks["exposed"][inventory]["contrasts"]["J-F*"]["estimate"]
        optimism[inventory] = {
            "held_out": None if held is None else held["mean"],
            "exposed": None if exposed is None else exposed["mean"],
            "held_out_minus_exposed": (None if held is None or exposed is None
                                       else held["mean"] - exposed["mean"]),
            "label": "DESCRIPTIVE, no interval"}
    published = plan["published_tokens"]
    disposition = {
        "question": plan["question"],
        "rows": {name: {"published_token": published[name],
                        "held_out_token": tokens["held_out"]["tokens"][name]["token"],
                        "changed": tokens["held_out"]["tokens"][name]["token"] != published[name]}
                 for name in ("J-F*", *SECONDARIES)},
        "issue_96_disposition": "unchanged (sanity check, not a new claim)",
    }
    return {"schema": SCHEMA_COMPUTE, "identity": IDENTITY, "plan_frozen_at": plan["frozen_at"],
            "cohorts": cohorts, "structure": layout, "published_arm_choices": choices,
            "inventories": {inventory: {cohort: blocks[cohort][inventory] for cohort in COHORTS}
                            for inventory in INVENTORIES},
            "tokens": tokens, "reproduction": reproduction, "exposure_optimism": optimism,
            "disposition": disposition}


def with_accounting(compute, ledger):
    return compute | {"compute": {"wall_seconds_publish": ledger["wall_seconds_publish"],
                                  "gpu_seconds": 0, "engine_seconds": 0,
                                  "rollouts": 0, "source_records_read": ledger["source_records_read"]}}


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
    if name == IN_COHORT_HINDSIGHT:
        return "J - F_h(cohort)"
    return name.replace("-", " - ", 1)


def clusters_note(contrast):
    note = f"{contrast['clusters']} clusters, {contrast['cells']} cells"
    return note + f"; {UNINFORMATIVE}" if contrast[UNINFORMATIVE] else note


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "inventory", "cohort", "statistic", "value", "interval_low",
                     "interval_high", "interval_label", "cells", "clusters", "flag", "detail"])

    def put(identity, inventory, cohort, statistic, value, block=None, cells="", clusters="",
            flag_text="", detail=""):
        writer.writerow([identity, inventory, cohort, statistic, "" if value is None else value,
                         block["interval"][0] if block else "", block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", cells, clusters, flag_text, detail])

    for cohort in TOKEN_COHORTS:
        for name, block in compute["tokens"][cohort]["tokens"].items():
            put(f"token_{cohort}_{name}", PRIMARY, cohort, f"{name} token", block["token"],
                detail=json.dumps(block, sort_keys=True))
        put(f"guards_{cohort}", PRIMARY, cohort, "guards pass", compute["tokens"][cohort]["guards_pass"],
            detail=json.dumps(compute["tokens"][cohort]["guards"], sort_keys=True))
    for inventory in INVENTORIES:
        for cohort in COHORTS:
            inv = compute["inventories"][inventory][cohort]
            for name in CONTRASTS:
                c = inv["contrasts"][name]
                est = c["estimate"]
                put(f"{inventory}_{cohort}_{name}", inventory, cohort, label(inv, name),
                    est and est["mean"], est, c["cells"], c["clusters"],
                    UNINFORMATIVE if c[UNINFORMATIVE] else "", f"undefined {c['undefined_cells']}")
            s = inv["fixed_pair_spread"]
            put(f"{inventory}_{cohort}_spread", inventory, cohort, "fixed-pair spread max_F - min_F",
                s and s["mean"], s, inv["mixed_cells"], len(inv["mixed_members"]))
            t = inv["top1"]
            put(f"{inventory}_{cohort}_top1", inventory, cohort, f"top-1 hits J / F* [{inv['F_star']}]",
                f"{t['J_hits']}/{t['F_star_hits']}", cells=t["cells"],
                clusters=len(inv["mixed_members"]), detail=f"chance {t['chance_mean']}")
            for arm in ARMS:
                put(f"{inventory}_{cohort}_auc_{arm}", inventory, cohort, f"member-mean AUC {arm}",
                    inv["arm_member_mean_auc"][arm], cells=inv["mixed_cells"],
                    clusters=len(inv["mixed_members"]))
        o = compute["exposure_optimism"][inventory]
        put(f"{inventory}_exposure_optimism", inventory, "held_out-exposed",
            "J - F* point estimate, held_out minus exposed", o["held_out_minus_exposed"],
            detail="DESCRIPTIVE, no interval")
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    cohorts = compute["cohorts"]
    add("# Issue-109 item 1: #96 contrasts re-scored on held-out N1 members — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v{plan['version']} frozen {plan['frozen_at']} before any "
        "held-out statistic was computed")
    add(f"- role: {plan['role']}")
    add(f"- source: the retained #96 decision records (`{relative(SOURCE)}`, read-only, "
        f"{plan['issue_96_records']['records']} records); zero rollouts")
    add(f"- cohorts (never pooled): all = {len(cohorts['all'])} members; held_out = "
        f"{', '.join(cohorts['held_out'])}; exposed (fit lineages in the universe) = "
        f"{', '.join(cohorts['exposed'])}")
    add(f"- arm choices: {plan['published_arm_choice_rule']}")
    add(f"- disclosure: {plan['disclosure']}")
    add(f"- intervals: {plan['estimands']['interval']}; {plan['estimands']['cluster_flag']}")
    add("")
    add("## Disposition (sanity check, not a new claim)")
    add("")
    add(f"Question: {compute['disposition']['question']}")
    add("")
    add("| grid contrast | published #96 token | held-out token | changed | held-out estimate (clusters) |")
    add("|---|---|---|---|---|")
    grid_held = compute["inventories"][PRIMARY]["held_out"]
    for name, row in compute["disposition"]["rows"].items():
        c = grid_held["contrasts"][name]
        add(f"| {label(grid_held, name)} | {row['published_token']} | {row['held_out_token']} | "
            f"{row['changed']} | {fmt_interval(c['estimate'])} ({clusters_note(c)}) |")
    add("")
    held_guards = compute["tokens"]["held_out"]["guards"]
    add("- held-out grid guards (G1, G2, caps inherited from #96; G3-G6 recomputed on held-out grid "
        "cells): " + ", ".join(f"{g} {held_guards[g]}" for g in (*INHERITED_GUARDS, *COHORT_GUARDS)))
    gi = grid_held["guard_inputs"]
    add(f"- held-out grid guard inputs: tied {gi['tied_cells']}/{grid_held['mixed_cells']} "
        f"(share {fmt(gi['tied_share'])}); J modal pair {gi['J_modal_pair']}, non-modal share "
        f"{fmt(gi['J_nonmodal_step_share'])}; rank-divergent share {fmt(gi['rank_divergent_share'])}; "
        f"primary half-width {fmt(compute['tokens']['held_out']['tokens']['J-F*']['half_width'])}")
    add(f"- the #96 disposition is {compute['disposition']['issue_96_disposition']}")
    add("")
    add("## Contrasts: published all-member vs held-out vs exposed (AUC difference, member-clustered)")
    add("")
    add("| inventory | contrast | published all-member [interval] | held-out [interval] (clusters) | "
        "exposed [interval] |")
    add("|---|---|---|---|---|")
    for inventory in INVENTORIES:
        block = compute["inventories"][inventory]
        for name in CONTRASTS:
            a, h, e = (block[c]["contrasts"][name] for c in COHORTS)
            name_label = label(block["all"], name)
            if name == IN_COHORT_HINDSIGHT:
                name_label = ("J - F_h(cohort) [" + " / ".join(block[c]["F_hindsight_in_cohort"]
                                                               for c in COHORTS) + "] (descriptive)")
            add(f"| {inventory} | {name_label} | {fmt_interval(a['estimate'])} | "
                f"{fmt_interval(h['estimate'])} ({clusters_note(h)}) | "
                f"{fmt_interval(e['estimate'])} ({e['clusters']} clusters) |")
    add("")
    add("The all-member column is recomputed here and equals the published #96 value to 4 decimals "
        f"(reproduction guard: {compute['reproduction']['values_compared']} values compared, pass "
        f"{compute['reproduction']['pass']}). F_h(cohort) lists the in-cohort hindsight arm for "
        "all / held_out / exposed.")
    add("")
    add("A k-cluster member bootstrap has C(2k-1, k) distinct resample multisets (" + ", ".join(
        f"k={k}: {math.comb(2 * k - 1, k)}" for k in (2, 3)) + "), so the held-out percentile "
        "intervals are coarse and understate uncertainty; a narrow held-out interval is not "
        "precision.")
    add("")
    add("## Exposure optimism (J - F\\* point estimate, held_out minus exposed; descriptive, no interval)")
    add("")
    add("| inventory | held-out | exposed | held_out - exposed |")
    add("|---|---|---|---|")
    for inventory in INVENTORIES:
        o = compute["exposure_optimism"][inventory]
        add(f"| {inventory} | {fmt(o['held_out'])} | {fmt(o['exposed'])} | "
            f"{fmt(o['held_out_minus_exposed'])} |")
    add("")
    add("## Cohort structure, spread and top-1")
    add("")
    add("| inventory | cohort | mixed members / cells | fixed-pair spread | top-1 J / F\\* / chance |")
    add("|---|---|---|---|---|")
    for inventory in INVENTORIES:
        for cohort in COHORTS:
            inv = compute["inventories"][inventory][cohort]
            t = inv["top1"]
            add(f"| {inventory} | {cohort} | {len(inv['mixed_members'])} / {inv['mixed_cells']} | "
                f"{fmt_interval(inv['fixed_pair_spread'])} | {t['J_hits']}/{t['cells']} / "
                f"{t['F_star_hits']}/{t['cells']} / {fmt(t['chance_mean'])} |")
    add("")
    add("- held-out mixed members: " + "; ".join(
        f"{i} {', '.join(compute['inventories'][i]['held_out']['mixed_members']) or 'none'}"
        for i in INVENTORIES))
    add("")
    add("## Per-arm member-mean AUC, held-out vs all (mixed cells)")
    add("")
    add("| arm | " + " | ".join(f"{i} all | {i} held-out" for i in INVENTORIES) + " |")
    add("|---|" + "---|---|" * len(INVENTORIES))
    for arm in ARMS:
        add(f"| {arm} | " + " | ".join(
            f"{fmt(compute['inventories'][i]['all']['arm_member_mean_auc'][arm])} | "
            f"{fmt(compute['inventories'][i]['held_out']['arm_member_mean_auc'][arm])}"
            for i in INVENTORIES) + " |")
    add("")
    add("## Compute accounting")
    add("")
    acc = compute["compute"]
    add(f"- wall {acc['wall_seconds_publish']:.1f} s (publish run); GPU seconds {acc['gpu_seconds']}; "
        f"engine seconds {acc['engine_seconds']}; rollouts {acc['rollouts']}; #96 records read "
        f"{acc['source_records_read']}")
    add("")
    add("## Protocol deviations")
    add("")
    add(f"- after the plan freeze at {plan['frozen_at']} the runner was edited only to add the "
        "k-cluster bootstrap resample note to the findings rendering; no estimand, membership, "
        "statistic or token logic changed; plan.json runner_sha256_at_freeze therefore records the "
        "pre-note runner")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ".")
    add("")
    add("## Reproduction")
    add("")
    add("```")
    for mode in ("dry-run", "prepare", "publish"):
        add(f"python -u -m scripts.run_heldout_rescore --{mode}")
    add(VALIDATION_COMMAND)
    add("```")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "role": plan["role"],
            "question": plan["question"], "disclosure": plan["disclosure"],
            "cohorts": compute["cohorts"], "published_arm_choices": compute["published_arm_choices"],
            "disposition": compute["disposition"], "tokens": compute["tokens"],
            "reproduction": compute["reproduction"],
            "exposure_optimism": compute["exposure_optimism"],
            "inventories": {inventory: {cohort: {k: v for k, v in block.items() if k != "guard_inputs"}
                                        for cohort, block in cohorts.items()}
                            for inventory, cohorts in compute["inventories"].items()},
            "compute": compute["compute"], "claim_boundary": plan["claim_boundary"],
            "issue_64_authorized": False}


def rendered(plan, compute):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute)}


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def publish(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    compute = compute_tables(plan)
    ledger = {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "status": "published",
              "published_at": utc_now(), "wall_seconds_publish": time.monotonic() - began,
              "gpu_seconds": 0, "engine_seconds": 0,
              "source_records_read": plan["issue_96_records"]["records"]}
    write_json(output / "ledger.json", ledger)
    compute = with_accounting(compute, ledger)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    held = compute["disposition"]["rows"]
    log("published: " + "; ".join(f"{name} held-out {row['held_out_token']} (published "
                                  f"{row['published_token']})" for name, row in held.items()))
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    ledger = read_json(output / "ledger.json")
    if ledger.get("schema") != SCHEMA_LEDGER or ledger.get("gpu_seconds") != 0:
        raise ValueError("ledger.json is not a zero-GPU issue-109 ledger")
    fresh = with_accounting(compute_tables(plan), ledger)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"exact recomputation validation passed: all-member reproduction "
        f"{fresh['reproduction']['values_compared']} values; compute.json, summary.json, "
        f"comparisons.csv and findings.md byte-compared ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    members, universe, excluded = load_universe()
    fit = fit_lineages()
    cohorts = cohort_members(members, fit)
    layout = structure(universe, cohorts)
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    log(f"  fit lineages: predictor {fit['predictor']}; controller {fit['controller']}")
    for cohort in COHORTS:
        log(f"  cohort {cohort} ({len(cohorts[cohort])}): {', '.join(cohorts[cohort])}")
        for inventory in INVENTORIES:
            block = layout[cohort][inventory]
            log(f"    {inventory}: {len(block['members'])} members, {len(block['mixed_members'])} "
                f"mixed {block['mixed_members']} -> {block['mixed_cells']} mixed cells")
    for row in excluded:
        log(f"  excluded {row['inventory']}/{row['member']}: {row['rule']}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": dry_run, "prepare": prepare, "publish": publish, "validate": validate}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    try:
        return MODES[mode](args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
