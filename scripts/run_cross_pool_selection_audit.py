"""Issue-92 WP3 ADD-EXP: cross-family / cross-split selection-validity audit.

Binding runner module: scripts/run_cross_pool_selection_audit.py
Exact validation command: python -u -m scripts.run_cross_pool_selection_audit --validate

Analysis/audit ticket over published artifacts only. Nothing is re-rendered,
retrained, recaptured, or re-executed; every input is READ-ONLY (the
bounded-transfer, n2-appearance and n2n oracle pools; the issue-77 n2-eval
selection pool; the issue-92 WP1 inputs for the N1 constant-ordinal table),
CPU-only, bounded wall+memory, zero GPU work.

Frozen-protocol chronology (binding shared rule):
  --dry-run   no-write, NO-SCORING structural inventory (allowed pre/post freeze)
  --prepare   writes plan.json with sha256_of(input) per input, no statistics
  --run       computes compute.json + cross_split_cells.csv + constant_ordinal.csv
  --publish   derives summary.json + comparisons.csv + findings.md + receipts.json
  --validate  recomputes every published artifact from frozen inputs and
              byte-compares; exit 0 only on exact reproduction

Audited objects (all numbers pre-declared in the frozen plan as expectations):
  Q1 (bounded-transfer breadth): the 46-measurable-member ceiling recomputed
     from retained native truth: 25/46 member-unit, 32 admissible successful
     cells with the ordinal histogram, the constant-ordinal policies and the
     inventory-matched uniform reference; the issue-76-bounded-transfer-002-a10
     typed-failure cell is recomputed from its retained native chunks and
     published with its evidence and the exclusion rule (coverage-admissible
     join), with the typed-failure outcome-blindness disclosure.
  Q2 (cross-split selection validity): the 17-state x 12-system x 3-seed x
     2-condition n2-eval pool (1,224 selection cells) joined to the owning
     pools' engine truth; top-1/top-3/AUC with the same member-cluster
     bootstrap and frozen margins as WP1, split by condition (zero-shot and
     few-shot are never pooled; only the zero-shot arm is a frozen-system
     result); the per-state localization; the constant-ordinal reference
     tables on N1 and on the cross-split.
  Q3 (N2/N2n typed coverage): the measured pool outcomes published as typed
     coverage, never as breadth.
"""
import argparse
import gzip
import json
import re
import resource
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

from scripts import run_selection_validity_restatement as wp1

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / ".local-artifacts"
BOUNDED = ARTIFACTS / "issue-76-bounded-transfer-v1"
N2_POOL = ARTIFACTS / "issue-77-n2-appearance-v1"
N2N_POOL = ARTIFACTS / "issue-77-n2n-v1"
EVAL_POOL = ARTIFACTS / "issue-77-n2-eval-v1"
OUTPUT = ARTIFACTS / "issue-92-cross-pool-audit-v1"

IDENTITY = "issue-92-cross-pool-audit-v1"
SCHEMA_PLAN = "issue_92_cross_pool_audit_plan_v1"
SCHEMA_COMPUTE = "issue_92_cross_pool_audit_compute_v1"
SCHEMA_SUMMARY = "issue_92_cross_pool_audit_summary_v1"
SCHEMA_RECEIPTS = "issue_92_cross_pool_audit_receipts_v1"

WALL_CAP_SECONDS = 300.0
DERIVED_BYTES_BUDGET = 50 * 1024 * 1024
STOP_TOKEN = "readiness_or_precision_insufficient"
INTERVAL_LABEL = "DESCRIPTIVE"

CONDITIONS = ("zero-shot", "few-shot")
SEEDS = (20260908, 20260909, 20260910)
SYSTEMS = ("continuous_h1", "continuous_h5", "continuous_h15",
           "hybrid_continuous_h1", "hybrid_continuous_h5", "hybrid_continuous_h15",
           "hybrid_micro_h1", "hybrid_micro_h5", "hybrid_micro_h15",
           "hybrid_macro_h1", "hybrid_macro_h5", "hybrid_macro_h15")
ANALOGUE_SYSTEMS = ("continuous_h1", "continuous_h5", "hybrid_continuous_h1")
ORDINALS = tuple(range(13))
BRANCH_SUFFIX = re.compile(r"-a(\d{2})$")
RESOLUTION_CELL = "issue-76-bounded-transfer-002-a10"

# Pre-declared expectations (frozen; the audit verifies every one and any
# delta is published, never silently absorbed).
EXPECTED = {
    "bounded": {
        "measurable_members": 46,
        "admissible_cells": 575,
        "failed_cells": 41,
        "member_ceiling": 25,
        "ceiling_interval": [0.3913, 0.6957],
        "success_cells": 32,
        "ordinal_histogram": {"0": 1, "1": 1, "2": 3, "3": 3, "4": 5, "5": 3,
                              "6": 4, "10": 3, "11": 3, "12": 6},
        "typed_failures_with_success": 1,
    },
    "n2": {
        "admissible": 186, "executed": 208, "measurable_members": 15,
        "member_ceiling": 5, "ceiling_interval": [0.1333, 0.6000],
        "success_cells": 7, "type010102_measurable": 7,
        "type010102_successes": 0,
    },
    "n2n": {
        "admissible": 34, "executed": 104, "typed_failures": 70,
        "measurable_members": 5, "member_ceiling": 0, "success_cells": 0,
    },
    "cross_split": {
        "cells": 1224, "cells_per_condition": 612, "states": 17,
        "top1_pooled": 42,
        "per_state_top1": {
            "issue-77-n2-007": 12, "issue-76-bounded-transfer-032": 13,
            "issue-76-bounded-transfer-038": 12,
            "issue-76-bounded-transfer-037": 4,
            "issue-76-bounded-transfer-033": 1},
        "zero_shot_top1": 13, "few_shot_top1": 29,
        "analogue_top1_per_condition": 2,
        "uniform_chance": 0.0407,
        "pooled_top1_at_ordinal_12": 37,
        "frozen_top1_at_ordinal_12": 11,
        "frozen_success_states": 4,
    },
    "constant_ordinal_n1": {
        "a04": [5, 12], "a03": [2, 12], "a02": [2, 12], "a06": [2, 12],
        "a05": [2, 12], "a12": [0, 12], "a08": [0, 12]},
    "constant_ordinal_cross_split": {
        "a12": [3, 15], "a04": [1, 14], "a02": [2, 13], "a03": [1, 13],
        "a06": [1, 16], "a10": [1, 15], "a08": [0, 17]},
}

CLAIM_BOUNDARY = (
    "cross-pool audit over retained artifacts only; #76's, #77's and the n2-eval "
    "pool's published verdicts are inputs and are never recomputed, amended, or "
    "reinterpreted beyond the frozen recomputation declared here; #64/#65 stay "
    "sealed; the n2-eval system set (12 systems incl. adapted predictors and "
    "micro/macro ablations) is never mixed with #87's 4-system set in one rate; "
    "zero-shot and few-shot are reported separately and never pooled; no engine "
    "access, no rendering, no new data; intervals are DESCRIPTIVE")


def log(message):
    print(f"[issue-92-cross-pool] {message}", flush=True)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(path, value):
    wp1.write_json(Path(path), value)


def write_text(path, text):
    wp1.write_text(Path(path), text)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def file_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def sha256_of(path):
    return wp1.sha256_of(path)


GPULock = wp1.GPULock
bootstrap_interval = wp1.bootstrap_interval


def branch_ordinal(identity):
    match = BRANCH_SUFFIX.search(identity)
    if match is None:
        raise ValueError(f"branch identity without -aNN suffix: {identity}")
    return int(match.group(1))


def branch_member(identity):
    return BRANCH_SUFFIX.sub("", identity)


def declared_sources():
    return [
        {"artifact": str(BOUNDED / "plan.json"), "role": "bounded membership"},
        {"artifact": str(BOUNDED / "coverage.json"), "role": "bounded coverage"},
        {"artifact": str(BOUNDED / "results"), "role": "bounded engine truth"},
        {"artifact": str(N2_POOL / "plan.json"), "role": "n2 membership"},
        {"artifact": str(N2_POOL / "coverage.json"), "role": "n2 coverage"},
        {"artifact": str(N2_POOL / "results"), "role": "n2 engine truth"},
        {"artifact": str(N2N_POOL / "plan.json"), "role": "n2n membership"},
        {"artifact": str(N2N_POOL / "coverage.json"), "role": "n2n coverage"},
        {"artifact": str(N2N_POOL / "results"), "role": "n2n engine truth"},
        {"artifact": str(EVAL_POOL / "plan.json"), "role": "eval membership"},
        {"artifact": str(EVAL_POOL / "fixed"), "role": "eval selection cells"},
        {"artifact": str(ARTIFACTS / "issue-87-closed-loop-oracle-v1"),
         "role": "N1 constant-ordinal input (via WP1 loaders)"},
        {"artifact": str(ARTIFACTS / "issue-89-oracle-completion-v1"),
         "role": "N1 constant-ordinal input (via WP1 loaders)"},
    ]


def frozen_plan(frozen_at):
    return {
        "identity": IDENTITY,
        "schema": SCHEMA_PLAN,
        "version": 1,
        "role": "terminal",
        "frozen_at": frozen_at,
        "frozen_before_scoring_run": True,
        "validation_command":
            "python -u -m scripts.run_cross_pool_selection_audit --validate",
        "protocol_policy": ("single terminal protocol version: the full protocol "
                            "is frozen with actual numeric margins and every "
                            "pre-declared expectation before the scoring run and "
                            "executed exactly once; the dry run computes no "
                            "statistic; --validate recomputes every published "
                            "table"),
        "claim_boundary": CLAIM_BOUNDARY,
        "caps": {
            "derived_bytes_budget": DERIVED_BYTES_BUDGET,
            "gpu_cap_seconds": 0.0,
            "gpu_scope": ("no GPU work of any kind in this ticket; the shared "
                          "flock is still held for every wall-time-measured phase"),
            "stop": STOP_TOKEN,
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "wall_scope": ("cumulative measured wall of the run plus publish plus "
                           "validate phases; the dry run and the freeze publish "
                           "no wall time"),
        },
        "questions": {
            "q1_bounded_breadth": ("at the frozen bounded-transfer window, what "
                                   "are the member-unit ceiling, the ordinal "
                                   "support, the constant-ordinal policies and "
                                   "the uniform reference, with the 002-a10 "
                                   "typed-failure resolution published?"),
            "q2_cross_split_selection": ("on the 17-state cross-split, do the "
                                         "frozen (zero-shot) systems select "
                                         "engine-truth successes at or below "
                                         "inventory-matched uniform chance?"),
            "q3_n2_typed_coverage": ("what are the N2 and N2n measured pool "
                                     "outcomes, published as typed coverage?"),
        },
        "unit_definitions": {
            "member": ("physical initial state; PRIMARY unit of analysis for "
                       "every ceiling headline"),
            "selection_cell": ("one (condition, seed, state, system) decision "
                               "cell of the n2-eval pool; 1,224 cells over 17 "
                               "states; selection = argmin predicted cost, ties "
                               "to the lower ordinal (WP1's documented rule)"),
            "admissible_branch": ("oracle branch whose coverage status is "
                                  "'admissible'; engine truth = pig_removed "
                                  "event count >= 1 in the retained segment "
                                  "summary; typed failures carry no success "
                                  "value and are excluded, never worst-cased"),
        },
        "statistics": {
            "bootstrap": ("percentile bootstrap, 10000 draws, PCG64 seed 7201, "
                          "quantiles [0.025, 0.975], labelled DESCRIPTIVE; "
                          "member-clustered variant resamples members with "
                          "every cell of a member moving together and averages "
                          "per-member means (identical construction to WP1, "
                          "same code path)"),
            "member_ceiling": ("indicator per measurable member = any admissible "
                               "successful branch; prevalence over measurable "
                               "members with a member-resampled interval"),
            "top_k": ("per selection cell: top-1 hit, top-3 hit; "
                      "inventory-matched per-cell chance k_s/n_s and "
                      "1 - C(n_s-k_s,3)/C(n_s,3); paired per-cell difference "
                      "with a member-clustered interval"),
            "auc": ("per scored selection cell (>=1 success and >=1 failure "
                    "among candidates): P(successful candidate receives a lower "
                    "predicted cost than a failure candidate), ties 0.5; cell "
                    "mean with cell-resampled interval; member-clustered = mean "
                    "of per-member means with member-resampled interval"),
            "constant_ordinal": ("per ordinal k: successes over opportunities "
                                 "among admissible branches, on N1 (12 sealed "
                                 "ceiling states, WP1 verdicts), on the "
                                 "cross-split (17 states), and on the full "
                                 "bounded pool (46 measurable members); the "
                                 "inventory-matched uniform reference is the "
                                 "only admissible chance baseline"),
            "typed_failure_audit": ("per pool: typed-failure count and the "
                                    "count of typed failures whose retained "
                                    "truth carries pig_removed >= 1; the 002-a10 "
                                    "cell is recomputed from its retained native "
                                    "chunks (manifest + chunk events on "
                                    "runtime:pig:0000) and published with its "
                                    "evidence and the exclusion rule"),
        },
        "margins": {
            "chance_margin": 0.0,
            "rule": ("numeric margins frozen before any statistic; never re-cut "
                     "after seeing the results"),
            "rule_text": ("q2's frozen margin is 0.0: the frozen (zero-shot) "
                          "arm is at-or-below chance iff its top-1 rate is "
                          "strictly below the inventory-matched uniform "
                          "reference with the member-clustered paired-difference "
                          "interval entirely below 0"),
        },
        "precision_guards": {
            "cells_total": 1224,
            "cells_per_condition": 612,
            "cross_split_states": 17,
            "eval_join_min_fraction": 0.99,
            "oracle_pools": 3,
            "rule": ("each guard is evaluated on the frozen tables; any failed "
                     "guard maps every disposition to "
                     "readiness_or_precision_insufficient before the question "
                     "rules are read"),
            "sparse_localization_floor": 3,
        },
        "verdict_mapping": {
            "q1_bounded_breadth": {
                "supported": ("every recomputed bounded-pool quantity equals its "
                              "pre-declared expectation (measurable members, "
                              "admissible/failed cells, member ceiling and its "
                              "interval at 4 decimals, success cells, ordinal "
                              "histogram, typed failures with success) AND the "
                              "002-a10 chunk recomputation confirms a "
                              "pig_removed event on runtime:pig:0000"),
                "not_supported_by_this_experiment": ("any recomputed quantity "
                                                     "differs from its "
                                                     "pre-declared expectation; "
                                                     "the delta is published"),
                "readiness_or_precision_insufficient": ("a precision guard "
                                                        "tripped, or the 002-a10 "
                                                        "chunks are unreadable"),
            },
            "q2_cross_split_selection": {
                "supported": ("the zero-shot top-1 rate is strictly below the "
                              "inventory-matched uniform reference (margin "
                              "0.0) AND the member-clustered paired-difference "
                              "interval lies entirely below 0 AND at least 3 "
                              "states carry a top-1 success"),
                "not_supported_by_this_experiment": ("the zero-shot top-1 rate "
                                                     "meets or exceeds the "
                                                     "uniform reference"),
                "readiness_or_precision_insufficient": ("the paired-difference "
                                                        "interval covers 0, or "
                                                        "fewer than 3 states "
                                                        "carry a success "
                                                        "(split-level rate only, "
                                                        "no generality "
                                                        "conclusion), or a "
                                                        "precision guard "
                                                        "tripped"),
            },
            "q3_n2_typed_coverage": {
                "supported": ("every recomputed N2/N2n coverage quantity equals "
                              "its pre-declared expectation"),
                "not_supported_by_this_experiment": ("any recomputed quantity "
                                                     "differs; the delta is "
                                                     "published"),
                "readiness_or_precision_insufficient": "a precision guard tripped",
            },
        },
        "expectations": EXPECTED,
        "disclosures": {
            "resolution_cell": ("issue-76-bounded-transfer-002-a10: coverage "
                                "status failed (operationally_valid false) yet "
                                "its retained segment summary carries "
                                "pig_removed = 1; under the #85/#87 semantics "
                                "the cell carries no success value and is "
                                "excluded; the cell, its chunk-recomputed "
                                "evidence and the exclusion rule are published; "
                                "the bounded pool's typed failures are "
                                "therefore not outcome-blind (1 of 41 carries "
                                "an engine-truth success), the same class of "
                                "finding #90 published for #87's timeouts"),
            "claim_split": ("the absolute zero-selection sentence is valid only "
                            "on the #87 N1 24-state membership (0 of 144 bound "
                            "cells); on the broader type010101 split the same "
                            "family of frozen systems selects engine-truth "
                            "successes at 3.4 % overall, localized to 5 of 17 "
                            "states; selection validity is published "
                            "state-conditional, never universal"),
            "condition_boundary": ("zero-shot and few-shot subtrees are disjoint "
                                   "and never pooled; only the zero-shot arm is "
                                   "a frozen-system result; the adapted arm "
                                   "exceeding chance is not a frozen-system "
                                   "result"),
            "ordinal_prior_retired": ("a constant-ordinal-8 policy scores 0/17 "
                                      "on the cross-split and is retired as any "
                                      "kind of baseline; the only admissible "
                                      "chance reference is inventory-matched "
                                      "uniform selection"),
        },
        "limitations": [
            "descriptive audit over retained artifacts; no inferential claim",
            "the n2-eval split uses a different system set and membership than "
            "#87; the two system sets are never mixed in one rate",
            "the cross-split chance reference is inventory-matched uniform "
            "selection; no other baseline is admissible",
            "typed failures are excluded and disclosed, never worst-cased",
        ],
        "inputs": declared_sources(),
        "published_context": {
            "issue_92_wp1": ("WP1's published N1 measurements are cited verbatim "
                             "as context; the N1 constant-ordinal table is "
                             "recomputed from WP1's own verdict inputs via its "
                             "published loader code path"),
        },
    }


def load_plan(path):
    plan = read_json(path)
    if plan["identity"] != IDENTITY or plan["schema"] != SCHEMA_PLAN:
        raise ValueError("plan binding differs")
    return plan


def load_terminal_plan(output):
    plan_path = Path(output) / "plan.json"
    if not plan_path.is_file():
        raise ValueError("plan.json missing; run --prepare before --run")
    return load_plan(plan_path)


# ---------------------------------------------------------------- pool loaders

def oracle_census(pool_root):
    """Per-branch engine truth joined with coverage; member rollup."""
    coverage = read_json(pool_root / "coverage.json")["branches"]
    plan = read_json(pool_root / "plan.json")
    members = {member["identity"]: member for member in plan["members"]}
    rows = {}
    for identity, entry in coverage.items():
        if entry["status"] != "admissible":
            rows[identity] = {"identity": identity, "member": branch_member(identity),
                              "ordinal": branch_ordinal(identity),
                              "status": entry["status"], "success": None}
            continue
        result_path = pool_root / "results" / f"{identity}.json"
        if not result_path.is_file():
            raise ValueError(f"admissible branch without a result file: {identity}")
        result = read_json(result_path)
        removed = sum(segment["summary"]["event_counts"].get("pig_removed", 0)
                      for segment in result["segments"])
        rows[identity] = {"identity": identity, "member": branch_member(identity),
                          "ordinal": branch_ordinal(identity),
                          "status": "admissible", "success": bool(removed >= 1)}
    return rows, members, coverage


def pool_rollup(rows, members):
    admissible = [row for row in rows.values() if row["status"] == "admissible"]
    failed = [row for row in rows.values() if row["status"] == "failed"]
    by_member = defaultdict(list)
    for row in admissible:
        by_member[row["member"]].append(row)
    measurable = sorted(member for member in by_member
                        if member in members)
    ceiling_members = sorted(member for member in measurable
                             if any(row["success"] for row in by_member[member]))
    histogram = Counter(str(row["ordinal"]) for row in admissible if row["success"])
    typed_with_success = 0
    for row in rows.values():
        if row["status"] == "failed":
            result_path = None
            for root in (BOUNDED, N2_POOL, N2N_POOL):
                candidate = root / "results" / f"{row['identity']}.json"
                if candidate.is_file():
                    result_path = candidate
                    break
            if result_path is None:
                continue
            result = read_json(result_path)
            removed = sum(segment["summary"]["event_counts"].get("pig_removed", 0)
                          for segment in result.get("segments", []))
            if removed >= 1:
                typed_with_success += 1
    prevalence = bootstrap_interval(
        [1.0 if member in ceiling_members else 0.0 for member in measurable])
    return {
        "measurable_members": len(measurable),
        "admissible_cells": len(admissible),
        "failed_cells": len(failed),
        "member_ceiling": len(ceiling_members),
        "ceiling_members": ceiling_members,
        "member_prevalence": prevalence,
        "success_cells": sum(1 for row in admissible if row["success"]),
        "ordinal_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        "typed_failures_with_success": typed_with_success,
        "by_member": {member: sorted((row["ordinal"], row["success"])
                                     for row in by_member[member])
                      for member in measurable},
    }


def recompute_resolution_cell():
    """Chunk-level recomputation of 002-a10's engine truth (bounded pool)."""
    coverage = read_json(BOUNDED / "coverage.json")["branches"][RESOLUTION_CELL]
    result = read_json(BOUNDED / "results" / f"{RESOLUTION_CELL}.json")
    segment = result["segments"][0]
    native_root = Path(segment["native_root"])
    manifest = read_json(native_root / "native-manifest.json")
    events = []
    for chunk in manifest["chunks"]:
        if chunk.get("event_count", 0) < 1:
            continue
        with gzip.open(native_root / chunk["path"], "rt") as handle:
            payload = json.load(handle)
        for event in payload.get("events", []):
            events.append(event)
    pig_events = [event for event in events
                  if "pig" in json.dumps(event) and event.get("kind") in
                  ("pig_removed", "entity_death", "entity_destroyed")]
    removed = [event for event in events
               if event.get("kind") == "pig_removed"
               and "runtime:pig:0000" in json.dumps(event)]
    return {
        "identity": RESOLUTION_CELL,
        "coverage": coverage,
        "result_summary_event_counts": segment["summary"]["event_counts"],
        "result_summary_failure": segment["summary"]["failure"],
        "manifest_status": manifest.get("status"),
        "manifest_failure": manifest.get("failure"),
        "chunk_events": events,
        "pig_related_events": pig_events,
        "chunk_recomputation": {
            "pig_removed_on_runtime_pig": len(removed) >= 1,
            "pig_removed_events": len(removed),
        },
        "exclusion_rule": ("coverage-admissible join: under the #85/#87 semantics "
                           "the cell carries no success value and is excluded "
                           "from the ceiling; the cell, its evidence and this "
                           "rule are published, never assumed away"),
    }


# ---------------------------------------------------------- cross-split loader

def load_eval_cells(rows_by_pool, notify=None):
    """One row per (condition, seed, state, system) selection cell."""
    fixed = EVAL_POOL / "fixed"
    cells = []
    done = 0
    for condition in CONDITIONS:
        for seed in SEEDS:
            seed_root = fixed / condition / f"seed-{seed}"
            for state_root in sorted(seed_root.iterdir()):
                state = state_root.name[len("state-"):]
                for system in SYSTEMS:
                    system_root = state_root / system
                    candidates = []
                    for candidate_path in sorted(system_root.glob("candidate-*.json")):
                        payload = read_json(candidate_path)
                        identity = payload["candidate_identity"]
                        pool_rows = rows_by_pool[pool_of(identity)]
                        branch = pool_rows.get(identity)
                        if branch is None:
                            raise ValueError(f"eval candidate without an oracle "
                                             f"branch: {identity}")
                        candidates.append({
                            "identity": identity,
                            "ordinal": branch_ordinal(identity),
                            "cost": float(payload["cost"]),
                            "admissible": branch["status"] == "admissible",
                            "success": bool(branch["success"]),
                        })
                    cells.append({"condition": condition, "seed": seed,
                                  "state": state, "system": system,
                                  "candidates": candidates})
                    done += 1
                    if notify is not None and done % 200 == 0:
                        notify(done)
    return cells


def pool_of(identity):
    if identity.startswith("issue-76-bounded-transfer-"):
        return "bounded"
    if identity.startswith("issue-77-n2n-"):
        return "n2n"
    if identity.startswith("issue-77-n2-"):
        return "n2"
    raise ValueError(f"unknown oracle pool for {identity}")


def choose(candidates):
    """argmin predicted cost; ties break toward the lower ordinal (WP1 rule)."""
    ordered = sorted(candidates, key=lambda item: (item["cost"], item["ordinal"]))
    return ordered


def constant_ordinal_table(states_rows):
    """Per ordinal: successes / opportunities over admissible branches."""
    table = {}
    for ordinal in ORDINALS:
        opportunities = successes = 0
        for rows in states_rows:
            for row in rows:
                if row["ordinal"] == ordinal:
                    opportunities += 1
                    successes += int(row["success"])
        table[f"a{ordinal:02d}"] = [successes, opportunities]
    return table


def n1_constant_table():
    """N1 table over the 12 sealed ceiling states via WP1's verdict path."""
    plan87, _, oracles87 = wp1.load_issue_87()
    summary89, records89 = wp1.load_issue_89()
    verdicts, _, _, _ = wp1.build_verdicts(plan87, oracles87, records89)
    members = wp1.member_tables(plan87, summary89, verdicts)
    sealed = [state for member in members.values() for state in member["states"]
              if state["ceiling_indicator"] and not state["typed_unmeasurable"]]
    table = {}
    for ordinal in ORDINALS:
        opportunities = successes = 0
        for state in sealed:
            key = (state["identity"], ordinal)
            if key in verdicts:
                opportunities += 1
                successes += int(verdicts[key])
        table[f"a{ordinal:02d}"] = [successes, opportunities]
    return table, len(sealed)


# ------------------------------------------------------------------ statistics

def cross_split_tables(cells):
    per_condition = {}
    for condition in CONDITIONS:
        subset = [cell for cell in cells if cell["condition"] == condition]
        rows = []
        for cell in subset:
            admissible = [item for item in cell["candidates"] if item["admissible"]]
            if not admissible:
                continue
            ordered = choose(admissible)
            successes = [item for item in admissible if item["success"]]
            n_s, k_s = len(admissible), len(successes)
            top1 = ordered[0]
            top3 = ordered[:3]
            chance1 = k_s / n_s
            chance3 = (1.0 if n_s < 3 else
                       1.0 - (comb(n_s - k_s, 3) / comb(n_s, 3))
                       if n_s - k_s >= 3 else 1.0)
            rows.append({
                "condition": condition, "seed": cell["seed"],
                "state": cell["state"], "system": cell["system"],
                "candidates": n_s, "successes": k_s,
                "chosen_ordinal": top1["ordinal"],
                "top1_hit": bool(top1["success"]),
                "top3_hit": any(item["success"] for item in top3),
                "chance1": chance1, "chance3": chance3,
                "diff1": float(top1["success"]) - chance1,
                "diff3": float(any(item["success"] for item in top3)) - chance3,
                "auc": (cell_auc(ordered) if 0 < k_s < n_s else None),
            })
        per_condition[condition] = rows
    return per_condition


def cell_auc(ordered):
    """P(successful candidate beats failure candidate on cost); ties 0.5."""
    wins = total = 0
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if first["success"] == second["success"]:
                continue
            total += 1
            wins += 1 if first["success"] else 0
            if first["cost"] == second["cost"]:
                wins -= 0.5
    return wins / total if total else None


def condition_summary(rows, states):
    top1 = [row["top1_hit"] for row in rows]
    chance = sum(row["chance1"] for row in rows) / len(rows)
    diff = [row["diff1"] for row in rows]
    clusters = [row["state"] for row in rows]
    per_state = {state: {"cells": 0, "top1": 0} for state in states}
    for row in rows:
        per_state[row["state"]]["cells"] += 1
        per_state[row["state"]]["top1"] += int(row["top1_hit"])
    analogue = [row for row in rows if row["system"] in ANALOGUE_SYSTEMS]
    scored = [row for row in rows if row["auc"] is not None]
    ordinal12 = sum(1 for row in rows if row["top1_hit"] and row["chosen_ordinal"] == 12)
    success_states = sorted({row["state"] for row in rows if row["top1_hit"]})
    return {
        "cells": len(rows),
        "top1_hits": sum(top1),
        "top1_rate": sum(top1) / len(rows),
        "uniform_chance": chance,
        "paired_diff": bootstrap_interval(diff, clusters),
        "per_state": {state: [entry["top1"], entry["cells"]]
                      for state, entry in sorted(per_state.items())},
        "states_with_top1": len(success_states),
        "success_states": success_states,
        "top1_at_ordinal_12": ordinal12,
        "analogue": {"cells": len(analogue),
                     "top1_hits": sum(row["top1_hit"] for row in analogue)},
        "top3": {
            "hits": sum(row["top3_hit"] for row in rows),
            "rate": sum(row["top3_hit"] for row in rows) / len(rows),
            "chance": sum(row["chance3"] for row in rows) / len(rows),
            "paired_diff": bootstrap_interval([row["diff3"] for row in rows], clusters),
        },
        "auc": {
            "scored_cells": len(scored),
            "cell_unit": bootstrap_interval([row["auc"] for row in scored],
                                            [row["state"] for row in scored])
            if scored else None,
        },
        "chosen_ordinal_histogram": dict(sorted(
            Counter(row["chosen_ordinal"] for row in rows).items())),
    }


# ---------------------------------------------------------------------- report

def verify_expectations(compute):
    """Every pre-declared expectation is checked; deltas are published."""
    checks = []

    def check(path, actual, expected):
        checks.append({"path": path, "expected": expected, "actual": actual,
                       "match": actual == expected})

    bounded = compute["bounded"]
    exp = EXPECTED["bounded"]
    check("bounded.measurable_members", bounded["measurable_members"],
          exp["measurable_members"])
    check("bounded.admissible_cells", bounded["admissible_cells"],
          exp["admissible_cells"])
    check("bounded.failed_cells", bounded["failed_cells"], exp["failed_cells"])
    check("bounded.member_ceiling", bounded["member_ceiling"], exp["member_ceiling"])
    check("bounded.ceiling_interval",
          [round(value, 4) for value in
           bounded["member_prevalence"]["interval"]], exp["ceiling_interval"])
    check("bounded.success_cells", bounded["success_cells"], exp["success_cells"])
    check("bounded.ordinal_histogram", bounded["ordinal_histogram"],
          exp["ordinal_histogram"])
    check("bounded.typed_failures_with_success",
          bounded["typed_failures_with_success"], exp["typed_failures_with_success"])

    n2 = compute["n2"]
    exp_n2 = EXPECTED["n2"]
    check("n2.admissible_cells", n2["admissible_cells"], exp_n2["admissible"])
    check("n2.measurable_members", n2["measurable_members"],
          exp_n2["measurable_members"])
    check("n2.member_ceiling", n2["member_ceiling"], exp_n2["member_ceiling"])
    check("n2.ceiling_interval",
          [round(value, 4) for value in n2["member_prevalence"]["interval"]],
          exp_n2["ceiling_interval"])
    check("n2.success_cells", n2["success_cells"], exp_n2["success_cells"])
    check("n2.type010102_measurable", n2["type010102_measurable"],
          exp_n2["type010102_measurable"])
    check("n2.type010102_successes", n2["type010102_successes"],
          exp_n2["type010102_successes"])

    n2n = compute["n2n"]
    exp_n2n = EXPECTED["n2n"]
    check("n2n.admissible_cells", n2n["admissible_cells"], exp_n2n["admissible"])
    check("n2n.failed_cells", n2n["failed_cells"], exp_n2n["typed_failures"])
    check("n2n.measurable_members", n2n["measurable_members"],
          exp_n2n["measurable_members"])
    check("n2n.member_ceiling", n2n["member_ceiling"], exp_n2n["member_ceiling"])
    check("n2n.success_cells", n2n["success_cells"], exp_n2n["success_cells"])

    cross = compute["cross_split"]
    exp_cross = EXPECTED["cross_split"]
    total = sum(entry["cells"] for entry in cross["per_condition"].values())
    check("cross_split.cells", total, exp_cross["cells"])
    pooled = sum(entry["top1_hits"] for entry in cross["per_condition"].values())
    check("cross_split.top1_pooled", pooled, exp_cross["top1_pooled"])
    check("cross_split.zero_shot_top1",
          cross["per_condition"]["zero-shot"]["top1_hits"],
          exp_cross["zero_shot_top1"])
    check("cross_split.few_shot_top1",
          cross["per_condition"]["few-shot"]["top1_hits"],
          exp_cross["few_shot_top1"])
    for condition in CONDITIONS:
        check(f"cross_split.analogue_top1.{condition}",
              cross["per_condition"][condition]["analogue"]["top1_hits"],
              exp_cross["analogue_top1_per_condition"])
    check("cross_split.uniform_chance",
          round(cross["per_condition"]["zero-shot"]["uniform_chance"], 4),
          exp_cross["uniform_chance"])
    check("cross_split.pooled_top1_at_ordinal_12",
          sum(entry["top1_at_ordinal_12"]
              for entry in cross["per_condition"].values()),
          exp_cross["pooled_top1_at_ordinal_12"])
    check("cross_split.frozen_top1_at_ordinal_12",
          cross["per_condition"]["zero-shot"]["top1_at_ordinal_12"],
          exp_cross["frozen_top1_at_ordinal_12"])
    check("cross_split.frozen_success_states",
          cross["per_condition"]["zero-shot"]["states_with_top1"],
          exp_cross["frozen_success_states"])
    # per-state localization is pooled over conditions in the expectation
    pooled_state = Counter()
    for condition in CONDITIONS:
        for state, (hits, _) in cross["per_condition"][condition]["per_state"].items():
            pooled_state[state] += hits
    for state, hits in exp_cross["per_state_top1"].items():
        check(f"cross_split.per_state_top1.{state}", pooled_state[state], hits)

    check("constant_ordinal.n1", compute["constant_ordinal"]["n1"]["table"],
          EXPECTED["constant_ordinal_n1"])
    check("constant_ordinal.cross_split",
          compute["constant_ordinal"]["cross_split"]["table"],
          EXPECTED["constant_ordinal_cross_split"])
    return checks


def verdicts(compute, checks, guards):
    dispositions = {}
    details = {}
    if any(not guard["passed"] for guard in guards):
        for question in ("q1_bounded_breadth", "q2_cross_split_selection",
                         "q3_n2_typed_coverage"):
            dispositions[question] = STOP_TOKEN
            details[question] = {"reason": "a precision guard tripped",
                                 "guards": guards}
        return {"question_dispositions": dispositions, "detail": details}

    q1_checks = [check for check in checks
                 if check["path"].startswith("bounded.")]
    resolution = compute["resolution_cell"]["chunk_recomputation"]
    q1_ok = (all(check["match"] for check in q1_checks)
             and resolution["pig_removed_on_runtime_pig"])
    dispositions["q1_bounded_breadth"] = (
        "supported" if q1_ok else "not_supported_by_this_experiment")
    details["q1_bounded_breadth"] = {
        "checks": q1_checks,
        "resolution_recomputed": resolution,
        "rule": "all pre-declared bounded expectations verified plus the 002-a10 "
                "chunk recomputation confirming pig_removed on runtime:pig:0000",
    }

    zero = compute["cross_split"]["per_condition"]["zero-shot"]
    interval = zero["paired_diff"]["interval"]
    q2_state = ("supported" if (zero["top1_rate"] < zero["uniform_chance"]
                                and interval[1] < 0
                                and zero["states_with_top1"] >= 3)
                else "not_supported_by_this_experiment"
                if zero["top1_rate"] >= zero["uniform_chance"]
                else STOP_TOKEN)
    dispositions["q2_cross_split_selection"] = q2_state
    details["q2_cross_split_selection"] = {
        "zero_shot_top1_rate": zero["top1_rate"],
        "uniform_chance": zero["uniform_chance"],
        "paired_diff_interval": interval,
        "states_with_top1": zero["states_with_top1"],
        "margin": 0.0,
        "rule": "zero-shot top-1 rate strictly below the uniform reference with "
                "the member-clustered paired-difference interval entirely below "
                "0 and >= 3 states carrying a success",
    }

    q3_checks = [check for check in checks
                 if check["path"].startswith(("n2.", "n2n."))]
    q3_ok = all(check["match"] for check in q3_checks)
    dispositions["q3_n2_typed_coverage"] = (
        "supported" if q3_ok else "not_supported_by_this_experiment")
    details["q3_n2_typed_coverage"] = {
        "checks": q3_checks,
        "rule": "all pre-declared N2/N2n coverage expectations verified",
    }
    return {"question_dispositions": dispositions, "detail": details,
            "expectation_checks": checks}


def guard_report(cells, rows_by_pool, states):
    total = len(cells)
    per_condition = Counter(cell["condition"] for cell in cells)
    joined = sum(1 for cell in cells for item in cell["candidates"])
    guards = [
        {"guard": "cells_total", "expected": 1224, "actual": total,
         "passed": total == 1224},
        {"guard": "cells_per_condition", "expected": 612,
         "actual": dict(per_condition),
         "passed": all(count == 612 for count in per_condition.values())
         and len(per_condition) == 2},
        {"guard": "cross_split_states", "expected": 17, "actual": len(states),
         "passed": len(states) == 17},
        {"guard": "oracle_pools", "expected": 3, "actual": len(rows_by_pool),
         "passed": len(rows_by_pool) == 3},
        {"guard": "eval_candidates_joined", "actual": joined,
         "passed": joined > 0},
    ]
    return guards


# -------------------------------------------------------------------- pipeline

def load_all(notify=None):
    rows_by_pool = {}
    members_by_pool = {}
    for name, root in (("bounded", BOUNDED), ("n2", N2_POOL), ("n2n", N2N_POOL)):
        rows, members, _ = oracle_census(root)
        rows_by_pool[name] = rows
        members_by_pool[name] = members
    cells = load_eval_cells(rows_by_pool, notify=notify)
    return rows_by_pool, members_by_pool, cells


def compute_everything(rows_by_pool, members_by_pool, cells):
    compute = {"identity": IDENTITY, "schema": SCHEMA_COMPUTE, "version": 1}

    bounded = pool_rollup(rows_by_pool["bounded"], members_by_pool["bounded"])
    compute["bounded"] = {key: value for key, value in bounded.items()
                          if key != "by_member"}

    n2 = pool_rollup(rows_by_pool["n2"], members_by_pool["n2"])
    type2 = [member for member in n2["ceiling_members"]
             if members_by_pool["n2"][member]["generator_family"] == "type010102"]
    measurable2 = [member for member, rows in
                   ((m, r) for m, r in n2["by_member"].items())
                   if members_by_pool["n2"][member]["generator_family"] == "type010102"]
    compute["n2"] = {key: value for key, value in n2.items() if key != "by_member"}
    compute["n2"]["type010102_measurable"] = len(measurable2)
    compute["n2"]["type010102_successes"] = len(type2)

    n2n = pool_rollup(rows_by_pool["n2n"], members_by_pool["n2n"])
    compute["n2n"] = {key: value for key, value in n2n.items() if key != "by_member"}

    log("recomputing the 002-a10 resolution cell from its retained native chunks")
    compute["resolution_cell"] = recompute_resolution_cell()

    states = sorted({cell["state"] for cell in cells})
    guards = guard_report(cells, rows_by_pool, states)
    per_condition = cross_split_tables(cells)
    cross = {"states": states,
             "per_condition": {condition: condition_summary(rows, states)
                               for condition, rows in per_condition.items()}}
    compute["cross_split"] = cross
    compute["cross_split_rows"] = sum(len(rows) for rows in per_condition.values())

    n1_table, sealed_count = n1_constant_table()
    cross_rows = [row for rows in per_condition.values() for row in rows]
    state_branch_rows = defaultdict(list)
    for cell in cells:
        if cell["condition"] == "zero-shot" and cell["seed"] == SEEDS[0]:
            state_branch_rows[cell["state"]] = [
                {"ordinal": item["ordinal"], "success": item["success"]}
                for item in cell["candidates"] if item["admissible"]]
    cross_table = constant_ordinal_table(state_branch_rows.values())
    bounded_table = constant_ordinal_table(
        [[{"ordinal": ordinal, "success": success}
          for ordinal, success in rows]
         for rows in pool_rollup(rows_by_pool["bounded"],
                                 members_by_pool["bounded"])["by_member"].values()])
    compute["constant_ordinal"] = {
        "n1": {"table": n1_table, "sealed_ceiling_states": sealed_count},
        "cross_split": {"table": cross_table, "states": len(state_branch_rows)},
        "bounded_pool": {"table": bounded_table,
                         "members": compute["bounded"]["measurable_members"]},
    }

    checks = verify_expectations(compute)
    compute["verdict"] = verdicts(compute, checks, guards)
    compute["guards"] = guards
    compute["_cell_rows"] = cross_rows
    return compute


def measured(function, *arguments, **keyword):
    began = time.monotonic()
    value = function(*arguments, **keyword)
    return value, {"seconds": time.monotonic() - began,
                   "peak_rss_mib": resource.getrusage(
                       resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def cells_csv(rows):
    lines = ["condition,seed,state,system,candidates,successes,chosen_ordinal,"
             "top1_hit,top3_hit,chance1,chance3,diff1,diff3,auc"]
    for row in sorted(rows, key=lambda item: (item["condition"], item["state"],
                                              item["system"], item["seed"])):
        lines.append(",".join([
            row["condition"], str(row["seed"]), row["state"], row["system"],
            str(row["candidates"]), str(row["successes"]),
            str(row["chosen_ordinal"]),
            "1" if row["top1_hit"] else "0",
            "1" if row["top3_hit"] else "0",
            f"{row['chance1']:.6f}", f"{row['chance3']:.6f}",
            f"{row['diff1']:.6f}", f"{row['diff3']:.6f}",
            "" if row["auc"] is None else f"{row['auc']:.6f}"]))
    return "\n".join(lines) + "\n"


def constant_csv(compute):
    lines = ["pool,ordinal,successes,opportunities"]
    for pool, entry in compute["constant_ordinal"].items():
        for ordinal, (successes, opportunities) in sorted(
                entry["table"].items(), key=lambda kv: int(kv[0][1:])):
            lines.append(f"{pool},{ordinal},{successes},{opportunities}")
    return "\n".join(lines) + "\n"


def comparisons_csv(compute):
    lines = ["table,scope,metric,value,interval_low,interval_high,label"]
    for pool in ("bounded", "n2", "n2n"):
        entry = compute[pool]
        prevalence = entry["member_prevalence"]
        lines.append(",".join([
            "member_ceiling", pool,
            f"{entry['member_ceiling']}/{entry['measurable_members']}",
            f"{prevalence['mean']:.4f}",
            f"{prevalence['interval'][0]:.4f}",
            f"{prevalence['interval'][1]:.4f}", INTERVAL_LABEL]))
    for condition, entry in compute["cross_split"]["per_condition"].items():
        lines.append(",".join([
            "cross_split_top1", condition,
            f"{entry['top1_hits']}/{entry['cells']}",
            f"{entry['top1_rate']:.4f}", "", "", "frozen-arm selection validity"]))
        lines.append(",".join([
            "cross_split_uniform_chance", condition,
            f"{entry['uniform_chance']:.4f}", "", "", "",
            "inventory-matched uniform selection"]))
        diff = entry["paired_diff"]
        lines.append(",".join([
            "cross_split_paired_diff", condition,
            f"{diff['mean']:.4f}", f"{diff['interval'][0]:.4f}",
            f"{diff['interval'][1]:.4f}", INTERVAL_LABEL]))
    return "\n".join(lines) + "\n"


def findings_md(plan, compute, timings):
    verdict = compute["verdict"]["question_dispositions"]
    bounded = compute["bounded"]
    zero = compute["cross_split"]["per_condition"]["zero-shot"]
    few = compute["cross_split"]["per_condition"]["few-shot"]
    resolution = compute["resolution_cell"]
    lines = [
        "# Issue-92 WP3 cross-pool selection-validity audit - findings",
        "",
        f"Plan identity `{plan['identity']}`, schema `{plan['schema']}`, version "
        f"{plan['version']} (terminal), frozen {plan['frozen_at']}.",
        "",
        "## Dispositions",
        "",
        f"- Q1 bounded-transfer breadth: **{verdict['q1_bounded_breadth']}**",
        f"- Q2 cross-split selection validity: **{verdict['q2_cross_split_selection']}**",
        f"- Q3 N2/N2n typed coverage: **{verdict['q3_n2_typed_coverage']}**",
        "",
        "## Q1 bounded-transfer ceiling (member unit)",
        "",
        f"- measurable members {bounded['measurable_members']}; admissible cells "
        f"{bounded['admissible_cells']}; typed failures {bounded['failed_cells']}",
        f"- member-unit ceiling {bounded['member_ceiling']}/"
        f"{bounded['measurable_members']} = "
        f"{bounded['member_prevalence']['mean']:.4f} "
        f"[{bounded['member_prevalence']['interval'][0]:.4f}, "
        f"{bounded['member_prevalence']['interval'][1]:.4f}] (DESCRIPTIVE)",
        f"- successful cells {bounded['success_cells']} with ordinal histogram "
        f"{json.dumps(bounded['ordinal_histogram'], sort_keys=True)}",
        "",
        "### 002-a10 typed-failure resolution",
        "",
        f"- coverage status `{resolution['coverage']['status']}` "
        f"(operationally_valid {resolution['coverage']['operationally_valid']}); "
        f"segment summary failure `{resolution['result_summary_failure']}` with "
        f"event counts {json.dumps(resolution['result_summary_event_counts'], sort_keys=True)}",
        f"- chunk recomputation: pig_removed on runtime:pig:0000 = "
        f"{resolution['chunk_recomputation']['pig_removed_on_runtime_pig']} "
        f"({resolution['chunk_recomputation']['pig_removed_events']} events)",
        f"- exclusion rule: {resolution['exclusion_rule']}",
        f"- typed failures carrying an engine-truth success: "
        f"{bounded['typed_failures_with_success']} of {bounded['failed_cells']} - "
        "the bounded pool's typed failures are NOT outcome-blind",
        "",
        "## Q2 cross-split selection validity (conditions never pooled)",
        "",
        f"- cells {zero['cells']} per condition over "
        f"{len(compute['cross_split']['states'])} states; pooled top-1 "
        f"{zero['top1_hits'] + few['top1_hits']}/1224",
        f"- zero-shot (frozen arm): {zero['top1_hits']}/{zero['cells']} = "
        f"{zero['top1_rate']:.4f} vs uniform chance {zero['uniform_chance']:.4f}; "
        f"paired difference {zero['paired_diff']['mean']:.4f} "
        f"[{zero['paired_diff']['interval'][0]:.4f}, "
        f"{zero['paired_diff']['interval'][1]:.4f}] (member-clustered, DESCRIPTIVE)",
        f"- few-shot (adapted arm, not a frozen-system result): "
        f"{few['top1_hits']}/{few['cells']} = {few['top1_rate']:.4f}",
        f"- #87-analogue systems: {zero['analogue']['top1_hits']}/"
        f"{zero['analogue']['cells']} per condition",
        f"- frozen-arm successes at ordinal 12: {zero['top1_at_ordinal_12']} of "
        f"{zero['top1_hits']} on {zero['states_with_top1']} states",
        "",
        "## Q3 N2/N2n typed coverage",
        "",
        f"- N2: {compute['n2']['admissible_cells']}/208 admissible; ceiling "
        f"{compute['n2']['member_ceiling']}/{compute['n2']['measurable_members']} "
        f"[{compute['n2']['member_prevalence']['interval'][0]:.4f}, "
        f"{compute['n2']['member_prevalence']['interval'][1]:.4f}]; type010102 "
        f"half {compute['n2']['type010102_successes']}/"
        f"{compute['n2']['type010102_measurable']}",
        f"- N2n: {compute['n2n']['failed_cells']}/104 typed failures; ceiling "
        f"{compute['n2n']['member_ceiling']}/{compute['n2n']['measurable_members']}; "
        "no successful action exists in the pool, so no selection statement is "
        "estimable",
        "",
        "## Claim boundary",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Cost",
        "",
        f"- measured wall {sum(entry['seconds'] for entry in timings.values()):.3f}s "
        f"of {WALL_CAP_SECONDS}s cap; zero GPU, zero engine seconds",
    ]
    return "\n".join(lines) + "\n"


def build_report(plan, compute, timings):
    return {
        "identity": IDENTITY,
        "schema": SCHEMA_SUMMARY,
        "version": 1,
        "plan_identity": plan["identity"],
        "claim_boundary": CLAIM_BOUNDARY,
        "question_dispositions": compute["verdict"]["question_dispositions"],
        "bounded": compute["bounded"],
        "n2": compute["n2"],
        "n2n": compute["n2n"],
        "resolution_cell": compute["resolution_cell"],
        "cross_split": compute["cross_split"],
        "constant_ordinal": compute["constant_ordinal"],
        "guards": compute["guards"],
        "phases": timings,
        "wall_seconds_total": sum(entry["seconds"] for entry in timings.values()),
        "gpu_seconds_total": 0.0,
    }


def run_audit(output, write=True):
    output = Path(output)
    plan = load_terminal_plan(output)
    timings = {}
    began = time.monotonic()
    bundle = load_all(notify=lambda done: log(
        f"eval cell join {done}/1224; elapsed {time.monotonic() - began:.1f}s"))
    rows_by_pool, members_by_pool, cells = bundle
    timings["load_and_join"] = {"seconds": time.monotonic() - began,
                                "peak_rss_mib": resource.getrusage(
                                    resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    compute, timings["compute_statistics"] = measured(
        compute_everything, rows_by_pool, members_by_pool, cells)
    total = sum(entry["seconds"] for entry in timings.values())
    if total > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: measured wall {total:.3f}s exceeds the "
                         f"frozen cap {WALL_CAP_SECONDS}s")
    compute["phases"] = timings
    cell_rows = compute.pop("_cell_rows")
    log(f"dispositions "
        f"{json.dumps(compute['verdict']['question_dispositions'], sort_keys=True)}"
        f"; measured wall {total:.3f}s of {WALL_CAP_SECONDS}s cap")
    path = output / "compute.json"
    retained = read_json(path) if path.is_file() else None
    if retained is not None and (
            {k: v for k, v in retained.items() if k != "phases"} !=
            {k: v for k, v in compute.items() if k != "phases"}):
        raise ValueError("retained compute.json differs from the fresh "
                         "recomputation; the ticket is frozen and is never "
                         "silently overwritten")
    if retained is not None:
        log("retained compute.json reproduces exactly (deterministic resume)")
    if write:
        write_text(output / "cross_split_cells.csv", cells_csv(cell_rows))
        write_text(output / "constant_ordinal.csv", constant_csv(compute))
        write_json(path, compute)
        log(f"compute retained at {path}")
    compute["_cell_rows"] = cell_rows
    return compute


def publish(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    path = output / "compute.json"
    if not path.is_file():
        raise ValueError("compute.json missing; run --run before --publish")
    compute = read_json(path)
    timings = compute["phases"]
    report = build_report(plan, compute, timings)
    write_json(output / "summary.json", report)
    write_text(output / "comparisons.csv", comparisons_csv(compute))
    write_text(output / "findings.md", findings_md(plan, compute, timings))
    write_json(output / "receipts.json", {
        "schema": SCHEMA_RECEIPTS,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "phases": timings,
        "wall_seconds_total": sum(entry["seconds"] for entry in timings.values()),
        "gpu_seconds_total": 0.0,
        "derived_bytes": file_bytes(output),
        "derived_bytes_budget": DERIVED_BYTES_BUDGET,
    })
    log(f"published summary.json, comparisons.csv, findings.md, "
        f"cross_split_cells.csv, constant_ordinal.csv and receipts.json in {output}")
    return report


def validate(output):
    output = Path(output)
    compute = run_audit(output, write=False)
    cell_rows = compute.pop("_cell_rows")
    published_compute = read_json(output / "compute.json")
    retained = {k: v for k, v in published_compute.items() if k != "phases"}
    recomputed = {k: v for k, v in compute.items() if k != "phases"}
    if retained != recomputed:
        for key in sorted(set(retained) | set(recomputed)):
            if retained.get(key) != recomputed.get(key):
                raise ValueError(f"compute.json section {key!r} differs from the "
                                 "fresh recomputation")
    plan = load_terminal_plan(output)
    phases = published_compute["phases"]
    fresh = {
        "cross_split_cells.csv": cells_csv(cell_rows).encode(),
        "constant_ordinal.csv": constant_csv(compute).encode(),
        "comparisons.csv": comparisons_csv(compute).encode(),
        "summary.json": json_text(build_report(plan, compute, phases)).encode(),
        "findings.md": findings_md(plan, compute, phases).encode(),
    }
    for name, content in fresh.items():
        path = output / name
        if not path.is_file():
            raise ValueError(f"published artifact missing: {name}")
        if path.read_bytes() != content:
            raise ValueError(f"published {name} differs from the recomputation "
                             "from the frozen inputs")
    log("exact recomputation validation passed: every published table "
        "byte-compared; no engine evidence was read and no content hash was "
        "recomputed")
    return 0


def dry_run(output):
    """No-write, NO-SCORING structural inventory (before or after the freeze)."""
    output = Path(output)
    log("no-write dry run; structural inventory only; no mean, rate, difference, "
        "proportion, correlation, margin test or interval is computed at any point")
    for name, root in (("bounded", BOUNDED), ("n2", N2_POOL), ("n2n", N2N_POOL)):
        coverage = read_json(root / "coverage.json")
        statuses = Counter(entry["status"] for entry in coverage["branches"].values())
        results = sum(1 for _ in (root / "results").glob("*.json"))
        plan = read_json(root / "plan.json")
        log(f"{name}: members {len(plan['members'])}; coverage statuses "
            f"{json.dumps(dict(sorted(statuses.items())))}; result files {results}")
    fixed = EVAL_POOL / "fixed"
    counts = {}
    for condition in CONDITIONS:
        states = set()
        systems = set()
        files = 0
        for seed in SEEDS:
            seed_root = fixed / condition / f"seed-{seed}"
            for state_root in sorted(seed_root.iterdir()):
                states.add(state_root.name)
                for system_root in sorted(state_root.iterdir()):
                    systems.add(system_root.name)
                    files += sum(1 for _ in system_root.glob("candidate-*.json"))
        counts[condition] = (len(states), len(systems), files)
        log(f"eval {condition}: states {len(states)}, systems {len(systems)}, "
            f"candidate files {files}")
    resolution = BOUNDED / "results" / f"{RESOLUTION_CELL}.json"
    log(f"resolution cell result present: {resolution.is_file()}")
    plan_path = Path(output) / "plan.json"
    if plan_path.is_file():
        plan = load_plan(plan_path)
        log(f"frozen protocol present: version {plan['version']} role "
            f"{plan['role']} frozen {plan['frozen_at']}")
    else:
        log("frozen protocol absent: run --prepare to freeze the protocol before "
            "--run")
    log("dry run complete; nothing was written and no statistic was computed")
    return 0


def prepare(output):
    output = Path(output)
    plan_path = output / "plan.json"
    if plan_path.is_file():
        frozen = load_plan(plan_path)
        log(f"existing frozen protocol validated (version {frozen['version']}, "
            f"frozen {frozen['frozen_at']}); no statistic was computed")
        return frozen
    frozen = frozen_plan(utc_now())
    frozen["inputs"] = [
        {**entry, "sha256_at_freeze": (
            sha256_of(Path(entry["artifact"])) if Path(entry["artifact"]).is_file()
            else f"directory:{sum(1 for _ in Path(entry['artifact']).rglob('*.json'))} json files")}
        for entry in frozen["inputs"]]
    write_json(plan_path, frozen)
    log(f"frozen protocol published at {plan_path}: version {frozen['version']}, "
        f"{len(frozen['questions'])} questions; no statistic was computed")
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        if args.dry_run:
            return dry_run(args.output)
        if args.prepare:
            prepare(args.output)
            return 0
        if args.run:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            with GPULock():
                run_audit(args.output)
            return 0
        if args.publish:
            with GPULock():
                publish(args.output)
            return 0
        with GPULock():
            return validate(args.output)
    except (ValueError, OSError, KeyError, FileNotFoundError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
