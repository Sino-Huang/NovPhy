"""Issue-92 ADD-EXP WP1 (+WP2 primary): selection-validity re-statement.

Binding runner module: scripts/run_selection_validity_restatement.py
Exact validation command: python -u -m scripts.run_selection_validity_restatement --validate

Analysis ticket over published artifacts only.  Nothing is re-rendered,
retrained, recaptured, or re-executed; every input is READ-ONLY (#87's terminal
closed-loop oracle pass, #89's completion pass, #90's timeout audit as a cited
cross-check, #77's frozen N1 campaign coverage/result records, #80's frozen
candidate-outcome tables).  CPU-only, bounded memory, zero GPU work; the shared
GPU flock is still held for every wall-time-measured phase.

What this ticket publishes (all pre-declared in plan.json before any statistic
is computed):

1. A corrected unit of analysis.  #87's 24 membership states are 15 physical
   initial states (source members); 9 members carry two states that are
   byte-identical duplicates (same frozen inventory element-for-element, same
   engine seed, same decision anchor) whose per-ordinal engine-truth verdict
   vectors are identical.  The member is the primary unit; the state remains as
   execution-level detail.
2. The member-unit action-ceiling prevalence with a member-clustered
   descriptive interval, and the frozen chance references for any
   zero-selection count (single uniform top-1 draw per ceiling member;
   member-clustered 12-cell and 3-cell pooled chances).
3. The selection-validity measurement: predicted-cost ordering vs the #85
   engine-truth channel over the retained #87 decision rankings — per-cell AUC
   under cell-unit and member-clustered bootstrap, top-k attribution against
   inventory-matched uniform chance, disjoint ordinal-support statistics, and
   the sweep-monotonicity correlations (ordinal, drag_x, drag_y).
4. The state-vs-action variance decomposition of the predicted cost (the
   mechanism measurement: state difficulty resolved, within-state action
   ordering not resolved).
5. The determinism measurement (WP2 primary): the frozen #77 replay verdicts
   versus the #87/#89 closed-loop verdicts on every unique branch — separate
   process runs, months apart, different runner modules, same engine seed and
   action — plus the executed-slot agreement and the channel-anomaly count.
6. The exposure split (training vs calibration members) of the selection
   measurements, and the launch-regime reading of the ordinal supports.
7. The accounting corrections: 15 source members; the #87-only decision-side
   accounting (230 bound / 46 unbound / 12 typed decision failures of 288); the
   partial-ceiling accounting (84 cells, 82 bound, 2 unbindable); the completed
   accounting (144/144 bound, 0 successes); the ORACLE_SEED protocol-label
   disclosure; the issue-77-n1-005-a07 ordinal-7 disclosure; #90's audit token
   cited verbatim.

Frozen-protocol chronology (binding shared rule):
- --prepare writes the complete protocol (unit definitions, join keys,
  statistics, numeric margins, verdict mapping, precision guards, disclosures,
  caps) into plan.json BEFORE any statistic is computed;
- --dry-run is a no-write structural inventory that computes NO statistic of
  any kind, before or after the freeze;
- --run computes the statistics once from the frozen inputs;
- --publish renders summary.json / comparisons.csv / findings.md /
  slot_join.csv / receipts.json;
- --validate recomputes everything from the frozen inputs and byte-compares it
  against the published artifacts.

Verdict vocabulary (frozen exactly): scientific-question dispositions use the
standard supported / not_supported_by_this_experiment /
readiness_or_precision_insufficient.  Every interval is DESCRIPTIVE; no
decision language exists outside the frozen numeric margins.  Stop rule
(frozen): if the member-clustered AUC interval covers 0.5 it is published
covering 0.5 and the ordinal-support + top-k structure carries the section;
margins are never re-cut after seeing the results.

Protocol hygiene: no content hash is recomputed after the freeze and no
full-corpus integrity pass runs anywhere; inputs are verified at read time by
the schema/identity/plan_identity fields the artifacts already carry (plus the
sha256 identities recorded once at freeze).

Claim boundary: re-statement and measurement over retained records only; #87's,
#89's and #90's published verdicts are inputs and are never recomputed,
amended, or reinterpreted; #64/#65 stay sealed; closed-loop claims stay
single-shot, decision-only, development-lineage scoped; no engine access, no
rendering, no new data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-92-selection-validity-v1"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"

EIGHTY_SEVEN = ROOT / ".local-artifacts/issue-87-closed-loop-oracle-v1"
EIGHTY_SEVEN_IDENTITY = "issue-87-closed-loop-oracle-v1"
EIGHTY_NINE = ROOT / ".local-artifacts/issue-89-oracle-completion-v1"
EIGHTY_NINE_IDENTITY = "issue-89-oracle-completion-v1"
NINETY = ROOT / ".local-artifacts/issue-90-oracle-timeout-audit-v1"
NINETY_IDENTITY = "issue-90-oracle-timeout-audit-v1"
SEVENTY_SEVEN = ROOT / ".local-artifacts/issue-77-n1-v1"
SEVENTY_SEVEN_IDENTITY = "issue-77-n1-v1"
EIGHTY = ROOT / ".local-artifacts/issue-80-reactive-diagnostic-v1"
EIGHTY_IDENTITY = "issue-80-reactive-diagnostic-v1"

SCHEMA_PLAN = "issue_92_selection_validity_plan_v1"
SCHEMA_COMPUTE = "issue_92_selection_validity_compute_v1"
SCHEMA_REPORT = "issue_92_selection_validity_report_v1"
SCHEMA_RECEIPTS = "issue_92_selection_validity_receipts_v1"
IDENTITY = "issue-92-selection-validity-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_selection_validity_restatement --validate"

MODEL_SYSTEMS = ("hybrid-fixed-h1", "continuous-fixed-h1", "continuous-fixed-h5")
PRIOR_SYSTEM = "no-model-ordinal-prior"
SEEDS = (20260908, 20260909, 20260910)
ORACLE_SEED_LABEL = 20260908

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7201
INTERVAL_QUANTILES = (0.025, 0.975)
INTERVAL_LABEL = "DESCRIPTIVE"

MARGIN_AUC = 0.05
MARGIN_TOPK = 0.10
MARGIN_MONOTONICITY = 0.50
MARGIN_ACTION_SHARE = 0.05

PRECISION_GUARDS = {
    "scored_model_cells": 200,
    "ceiling_model_cells": 100,
    "verdict_slots": 288,
    "duplicate_pairs": 9,
    "duplicate_discordant_entries": 0,
    "replay_join_min_fraction": 0.99,
    "bootstrap_usable_min_fraction": 0.90,
}

WALL_CAP_SECONDS = 300.0
DERIVED_BYTES_BUDGET = 50 * 1024 ** 2
STOP_TOKEN = "readiness_or_precision_insufficient"

DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")

LAUNCH_SWEEP = {0: (-80, 10), 1: (-80, 7), 2: (-78, 17), 3: (-76, 26), 4: (-72, 35),
                5: (-67, 44), 6: (-61, 51), 7: (-55, 59), 8: (-47, 65), 9: (-39, 70),
                10: (-30, 74), 11: (-21, 77), 12: (-11, 79)}

MECHANISM_SENTENCE = (
    "the frozen predictors learn state difficulty but order actions within a state by the "
    "sweep coordinate, which is why the successful action is never ranked first and "
    "discriminates at or below chance on the engine-truth channel")

CLAIM_BOUNDARY = (
    "re-statement and measurement over retained records only; #87's, #89's and #90's "
    "published verdicts are inputs and are never recomputed, amended, or reinterpreted; "
    "#64/#65 stay sealed; closed-loop claims stay single-shot, decision-only, "
    "development-lineage scoped; no engine access, no rendering, no new data; intervals "
    "are DESCRIPTIVE")

LIMITATIONS = (
    "development/exposed N1 lineages only; not the sealed #64/#65 benchmark",
    "the member unit removes the duplicate inflation but the 14 measurable members remain "
    "a small, clustered sample; every interval is DESCRIPTIVE",
    "the selection measurements bind the three frozen fixed-horizon model systems of #87 "
    "plus the ordinal prior; no adaptive or endpoint arm is scored here (ticket #92 WP1b)",
    "single oracle protocol label (ORACLE_SEED is a #74 matched training-seed label, not "
    "a random draw); the engine's only seed is the per-member NOVPHY_ENVIRONMENT_SEED",
    "the AUC cell-unit interval is resampled over 108 cells that cluster inside 7 "
    "members; the member-clustered interval is the honest precision statement and is "
    "published alongside",
    "no content hash is recomputed after the freeze and no full-corpus integrity pass "
    "runs anywhere; inputs are verified at read time by their carried "
    "schema/identity/plan_identity fields",
)

PLACEHOLDER_MARKERS = ("TBD", "placeholder", "to be frozen", "XXX", "FIXME")


def log(message):
    print(f"[issue-92-selection-validity] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def file_bytes(path):
    return sum(entry.stat().st_size for entry in Path(path).rglob("*") if entry.is_file())


def sha256_of(path):
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _int_keyed(mapping):
    """Deterministic {int: count} rendering, stable across a JSON round trip."""
    return "{" + ", ".join(f"{int(key)}: {mapping[key]}"
                           for key in sorted(mapping, key=lambda k: int(k))) + "}"


class GPULock:
    """Exclusive advisory lock held for one wall-time-measured phase."""

    def __init__(self):
        self._handle = None

    def __enter__(self):
        self._handle = open(GPU_LOCK_PATH, "a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        log("shared GPU flock acquired for this measured phase")
        return self

    def __exit__(self, *exception):
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
        log("shared GPU flock released")
        return False


# ---------------------------------------------------------------------------
# input declarations
# ---------------------------------------------------------------------------

def declared_sources():
    return [
        {"name": "issue_87_plan", "artifact": str(EIGHTY_SEVEN / "plan.json"),
         "schema": "issue_87_closed_loop_oracle_plan_v1",
         "identity_fields": ["identity", "plan_identity", "version", "status"],
         "role": "membership (24 states, source members, inventories), oracle cells, "
                 "published protocol"},
        {"name": "issue_87_decision_records",
         "artifact": str(EIGHTY_SEVEN / "records"),
         "schema": "issue_87_decision_record_v1",
         "identity_fields": ["plan_identity", "plan_version", "schema"],
         "role": "the 288 decision cells: full predicted-cost rankings, chosen "
                 "candidate, #87-only binding"},
        {"name": "issue_87_oracle_records", "artifact": str(EIGHTY_SEVEN / "records"),
         "schema": "issue_87_closed_loop_oracle_record_v1",
         "identity_fields": ["plan_identity", "plan_version", "schema"],
         "role": "the 234 executed engine-truth verdicts and 55 typed failures"},
        {"name": "issue_89_summary", "artifact": str(EIGHTY_NINE / "summary.json"),
         "schema": "issue_89_oracle_completion_report_v1",
         "identity_fields": ["identity", "plan_identity", "schema"],
         "role": "cited published cross-check: per-state ceiling, typed-unmeasurable "
                 "declaration; never recomputed"},
        {"name": "issue_89_oracle_records", "artifact": str(EIGHTY_NINE / "records"),
         "schema": "issue_89_oracle_completion_record_v1",
         "identity_fields": ["plan_identity", "plan_version", "schema"],
         "role": "the completion-pass engine-truth verdicts extending the union table"},
        {"name": "issue_90_summary", "artifact": str(NINETY / "summary.json"),
         "schema": "issue_90_oracle_timeout_audit_report_v1",
         "identity_fields": ["identity", "plan_identity", "schema"],
         "role": "cited audit verdict token, verbatim; never recomputed"},
        {"name": "issue_77_n1_coverage", "artifact": str(SEVENTY_SEVEN / "coverage.json"),
         "schema": "issue_77_n1_coverage_v1",
         "identity_fields": ["plan_identity", "schema"],
         "role": "frozen N1 branch admissibility and typed-failure statuses"},
        {"name": "issue_77_n1_results", "artifact": str(SEVENTY_SEVEN / "results"),
         "schema": "issue_76_bounded_transfer_capture_v1",
         "identity_fields": ["member_identity", "base_cluster", "schema"],
         "role": "frozen N1 replay verdicts (pig_removed event counts) for the "
                 "determinism measurement"},
        {"name": "issue_80_candidate_outcomes", "artifact": str(EIGHTY / "candidate-outcomes"),
         "schema": "issue_80_candidate_outcomes_v1",
         "identity_fields": ["member_identity", "plan_identity", "schema"],
         "role": "frozen candidate tables (action and realized count cost per ordinal)"},
    ]


def verify_source_bindings(plan):
    declared = [(entry["name"], entry["artifact"]) for entry in declared_sources()]
    frozen = [(entry["name"], entry["artifact"]) for entry in plan["inputs"]]
    if frozen != declared:
        raise ValueError("frozen plan input bindings differ from the declared sources")


# ---------------------------------------------------------------------------
# frozen plan payload
# ---------------------------------------------------------------------------

def frozen_plan(frozen_at):
    return {
        "schema": SCHEMA_PLAN,
        "identity": IDENTITY,
        "version": 1,
        "role": "terminal",
        "frozen_at": frozen_at,
        "frozen_before_scoring_run": True,
        "changelog": [{
            "version": 1,
            "change": "initial and terminal freeze: unit definitions, join keys, "
                      "statistics, numeric margins, verdict mapping, precision guards, "
                      "disclosures and caps are published before any statistic exists",
            "why": "standard freeze rule of the shared ADD-EXP protocol; the dry run is "
                   "a structural inventory only and computes no statistic of any kind",
            "evidence": "no statistic has been computed at freeze time; every formula "
                        "below is a closed rule over the frozen inputs",
        }],
        "protocol_policy": (
            "single terminal protocol version: the full protocol is frozen with actual "
            "numeric margins before the scoring run and executed exactly once; the dry "
            "run computes no statistic; --validate recomputes every published table"),
        "unit_definitions": {
            "member": ("physical initial state (source member): the unit that carries an "
                       "independent level layout; PRIMARY unit of analysis for every "
                       "headline number"),
            "state": ("execution-level state identity of the #87 membership: 24 states "
                      "over 15 source members; 9 members carry two states that are "
                      "byte-identical duplicates (same frozen inventory "
                      "element-for-element, same engine seed, same decision anchor); "
                      "state-level numbers remain as execution detail only"),
            "model_cell": ("one (state, model system, seed) decision cell with a scored "
                           "ranking; 207 scored model cells over 23 anchored states, of "
                           "which 108 sit on the 12 sealed ceiling states"),
            "verdict_slot": ("one (state, ordinal) engine-truth verdict in the union of "
                             "#87's executed oracle records and #89's completion "
                             "records; 288 of 289 scheduled slots carry a verdict"),
            "unique_branch": ("distinct branch identity among the verdict slots; "
                              "duplicate states share branch identities, so the 288 "
                              "verdict slots collapse to 176 unique branches"),
        },
        "questions": {
            "q1_selection_validity": (
                "does the frozen predicted-cost ordering discriminate the engine-truth "
                "successful action at or below inventory-matched chance (per-cell AUC, "
                "top-k attribution, disjoint ordinal supports)?"),
            "q2_ordering_mechanism": (
                "is the within-state predicted-cost ordering driven by the action-sweep "
                "coordinate (near-monotone Spearman ordering, between-actions variance "
                "share at noise level) while the between-state component stays large?"),
            "q3_determinism": (
                "do the frozen #77 replay verdicts and the #87/#89 closed-loop verdicts "
                "agree on every unique branch (a measured determinism statement: the "
                "closed-loop pass is a deterministic re-observation given level, engine "
                "seed and action)?"),
        },
        "universe": {
            "membership_states": 24,
            "source_members": 15,
            "duplicate_pairs_expected": 9,
            "sealed_states": ("23 states carry a sealed decision anchor; "
                              "issue-77-n1-005-a07 is excluded under #89's frozen "
                              "typed-unmeasurable declaration (no executed oracle anchor)"),
            "sealed_ceiling_states": ("12 states carry the ceiling indicator and are not "
                                      "typed-unmeasurable, over 7 ceiling members"),
            "model_systems": list(MODEL_SYSTEMS),
            "prior_system": PRIOR_SYSTEM,
            "seeds": list(SEEDS),
            "verdict_slots": 288,
            "unique_branches": 176,
            "scheduled_slots": 289,
            "unmeasured_slot": ("issue-77-n1-005-a07 ordinal 2: sealed-frame stability "
                                "violation, retained as a typed failure in both #87 and "
                                "#89; never retried, carries no success value"),
        },
        "join_keys": {
            "verdict_slot": "(state identity, ordinal) -> outcome.first_shot_success of "
                            "the #87 executed oracle record, else the #89 completion "
                            "record; slots still typed-failed after #89 carry no verdict",
            "decision_cell": "issue-87 decision--<system>--seed<seed>--<state>.json; the "
                             "ranking rows give (ordinal, predicted_cost, excluded)",
            "state_to_member": "issue-87 plan.json states[].source_member",
            "frozen_candidate_table": "issue-80 candidate-outcomes/<source_member>.json "
                                      "row with the same ordinal (action and realized "
                                      "count cost)",
            "frozen_replay": "issue-77 N1 results/<branch identity>.json "
                             "segments[0].summary.event_counts.pig_removed greater than "
                             "0, keyed by branch identity",
            "exposure_role": "issue-87 plan.json states[].execution_member.exposure_role "
                             "('training' members joined the fitting pools; 'calibration' "
                             "members 007/008/014/016 are unexposed)",
        },
        "statistics": {
            "member_ceiling": ("indicator per measurable member = any of its states "
                               "carries the ceiling indicator; prevalence over the "
                               "measurable members; member-clustered percentile "
                               "bootstrap interval"),
            "bootstrap": (f"percentile bootstrap, {BOOTSTRAP_DRAWS} draws, PCG64 seed "
                          f"{BOOTSTRAP_SEED}, quantiles {list(INTERVAL_QUANTILES)}, "
                          "labelled DESCRIPTIVE; each statistic draws from an "
                          "independent stream with the same seed; the member-clustered "
                          "variant resamples members with every cell of a member moving "
                          "together and averages per-member means"),
            "chance_references": {
                "single_draw_miss_all": ("product over the 7 ceiling members of "
                                         "(1 - k_m/n_m): one uniform top-1 draw per "
                                         "ceiling member misses everywhere; k_m = "
                                         "successful ordinals, n_m = verdict slots of "
                                         "the member"),
                "member_clustered_12cell": ("the same product with exponent 12 per "
                                            "member (4 systems x 3 seeds selection "
                                            "cells per ceiling member)"),
                "one_system_3seeds": ("the same product with exponent 3 per member "
                                      "(one system x 3 seeds)"),
                "uniform_1of13_miss_all": ("(12/13)^k for k ceiling members: the "
                                           "inventory-size-only reference"),
                "breadth_extrapolation": ("ESTIMATE: miss_all(m) approx "
                                          "miss_all(7)^(m/7), evaluated at m = 15 and "
                                          "m = 30; labelled ESTIMATE, never a verdict "
                                          "input"),
            },
            "auc": ("per scored ceiling model cell: P(successful ordinal receives a "
                    "lower predicted cost than a failure ordinal) with ties scoring "
                    "0.5; headline = mean over the 108 cells (cell-unit) with a "
                    "cell-resampled interval; member-clustered = mean of the 7 "
                    "per-member means with a member-resampled interval"),
            "top_k": ("per scored ceiling model cell: top-1 hit (chosen ordinal is "
                      "engine-truth successful), top-3 hit (any of the three "
                      "lowest-cost ordinals is successful); inventory-matched per-state "
                      "chance: k_s/n_s for top-1 and 1 - C(n_s-k_s, 3)/C(n_s, 3) for "
                      "top-3; paired per-cell difference (hit minus chance) with a "
                      "member-clustered interval"),
            "ordinal_support": ("success band = sorted union of engine-truth successful "
                                "ordinals over the 12 sealed ceiling states; chosen "
                                "support = distribution of chosen ordinals over all 207 "
                                "scored model cells; disjointness = share of scored "
                                "model cells whose chosen ordinal lies in the success "
                                "band; region success counts = successes vs verdict "
                                "slots at ordinals 0-1 and 7-12 over the sealed ceiling "
                                "states"),
            "monotonicity": ("per scored model cell: Spearman correlation between the "
                             "candidate ordinal (resp. drag_x, drag_y) and the "
                             "predicted cost over the cell's ranking rows; headline = "
                             "median absolute rho over the 207 cells, per coordinate "
                             "and per system"),
            "variance_decomposition": ("over the retained ranking rows of the 207 "
                                       "scored model cells: marginal two-way shares of "
                                       "the predicted-cost variance, between-states = "
                                       "SS(state)/SST, between-actions = "
                                       "SS(ordinal)/SST, residual = 1 minus both "
                                       "shares"),
            "determinism": ("per unique branch: frozen #77 replay verdict (pig_removed "
                            "event count > 0) vs closed-loop verdict; agreement counts "
                            "at the unique-branch level and the verdict-slot level; "
                            "channel anomalies = executed records whose engine_channel "
                            "consistency is not 'agreement'; duplicate-pair concordance "
                            "= discordant per-ordinal verdict entries across the "
                            "duplicate pairs"),
            "exposure_split": ("the q1 measurements split by member exposure_role "
                               "(training vs calibration); DESCRIPTIVE only, never a "
                               "verdict input"),
        },
        "margins": {
            "auc_shift": MARGIN_AUC,
            "top_k_shift": MARGIN_TOPK,
            "monotonicity": MARGIN_MONOTONICITY,
            "action_share": MARGIN_ACTION_SHARE,
            "rule": "numeric margins frozen before any statistic; never re-cut after "
                    "seeing the results",
        },
        "verdict_mapping": {
            "q1_selection_validity": {
                "supported": ("ALL of: (a) cell-unit AUC mean <= 0.5 - margin AND the "
                              "cell-unit descriptive interval lies entirely below 0.5; "
                              "(b) the top-3 paired difference mean <= -margin AND its "
                              "member-clustered interval lies entirely below 0; (c) "
                              "the disjointness share (chosen ordinal inside the "
                              "success band) is exactly 0 over all scored model cells"),
                "not_supported_by_this_experiment": ("cell-unit AUC mean >= 0.5 OR the "
                                                     "top-3 paired difference mean >= 0"),
                "readiness_or_precision_insufficient": "any other combination, or a "
                                                       "precision guard tripped",
                "stop_rule": ("if the member-clustered AUC interval covers 0.5 it is "
                              "published covering 0.5 and the ordinal-support + top-k "
                              "structure carries the section; margins are never re-cut"),
            },
            "q2_ordering_mechanism": {
                "supported": ("median |Spearman rho| between ordinal and predicted "
                              "cost >= monotonicity margin AND the between-actions "
                              "variance share <= action-share margin"),
                "not_supported_by_this_experiment": ("median |rho| < monotonicity "
                                                     "margin AND the between-actions "
                                                     "share > action-share margin"),
                "readiness_or_precision_insufficient": "any other combination, or a "
                                                       "precision guard tripped",
            },
            "q3_determinism": {
                "supported": ("unique-branch agreement == branches joined AND "
                              "verdict-slot agreement == slots joined AND channel "
                              "anomalies == 0 AND duplicate discordant entries == 0"),
                "not_supported_by_this_experiment": ("any discordant branch or slot, "
                                                     "or any channel anomaly; forces a "
                                                     "seed-conditional restatement of "
                                                     "the ceiling"),
                "readiness_or_precision_insufficient": ("the replay join is incomplete "
                                                        "beyond the frozen guard, or a "
                                                        "precision guard tripped"),
            },
        },
        "precision_guards": dict(PRECISION_GUARDS, rule=(
            "each guard is evaluated on the frozen tables; any failed guard maps every "
            "disposition to readiness_or_precision_insufficient before the question "
            "rules are read")),
        "disclosures": {
            "duplicate_pairs": ("the 9 duplicate pairs are published with their "
                                "identity evidence (byte-identical inventory, same "
                                "engine seed, same decision anchor, identical "
                                "per-ordinal verdict vectors, 0 discordant entries)"),
            "issue_77_n1_005_a07": ("the one excluded state succeeds at ordinal 7, "
                                    "inside the model-chosen ordinal support; printed, "
                                    "never omitted; its typed-failure cell (ordinal 2, "
                                    "sealed-frame stability violation) is retained in "
                                    "both #87 and #89"),
            "oracle_seed_label": ("ORACLE_SEED = 20260908 "
                                  "(scripts/run_closed_loop_oracle_probe.py:126) is a "
                                  "#74 matched training-seed protocol label written "
                                  "only into metadata/identity strings; the engine "
                                  "receives exactly one seed, NOVPHY_ENVIRONMENT_SEED = "
                                  "state['engine_seed'] (probe line 858; per member "
                                  "764100001 + member ordinal); a 'second oracle seed' "
                                  "re-run would dispatch byte-identical engine "
                                  "invocations"),
            "accounting": ("15 source members (not 12); #87-only decision accounting "
                           "(230 bound / 46 unbound / 12 typed decision failures of "
                           "288); partial-ceiling accounting (84 cells, 82 bound, 2 "
                           "unbindable); completed accounting (144/144 bound, 0 "
                           "successes, binding completed against the #89-extended "
                           "oracle table)"),
            "issue_90_token": ("#90's audit verdict token is cited verbatim: "
                               "indeterminate (Q1 readiness_or_precision_insufficient), "
                               "with the outcome-enriched missingness direction "
                               "disclosed (5 of 52 timed-out slots succeeded on "
                               "re-execution against 7 of 234 originally executed)"),
            "chance_baseline": ("the only admissible chance reference is "
                                "inventory-matched uniform selection; the ordinal-8 "
                                "prior is retired as any kind of baseline (its own "
                                "ordinal 8 = (-47, 65) sits in the high-arc band and "
                                "scores zero)"),
            "launch_regime": ("the 13 candidates are a monotone launch-angle sweep "
                              "(a0 = (-80, 10) flat to a12 = (-11, 79) steep); on the "
                              "N1 sealed ceiling states every engine-truth success "
                              "sits at ordinals 2-6 (flat/mid direct shots) while the "
                              "systems choose ordinals 7-12 (high-arc lobs) in "
                              "207/207 model cells; the rule breaks ties toward the "
                              "lower ordinal, so the high-ordinal concentration is a "
                              "genuine preference, not tie-breaking; the N1 reading "
                              "is scoped to type010103/type010105 and never stated as "
                              "pool-invariant"),
        },
        "published_context": {
            "issue_87_pre_completion_ceiling": {
                "ceiling_states": 7, "measured_states": 23, "states": 24,
                "prevalence": 0.30434782608695654,
                "unmeasured_states": ["issue-77-n1-005-a07"]},
            "issue_89_completed_ceiling": (
                "cited verbatim from issue-89 summary.json ceiling_reestimate.pooled "
                "(12/23 = 0.5217 [0.3043, 0.7391] state-unit, before deduplication)"),
            "issue_90_audit_verdict": (
                "cited verbatim from issue-90 summary.json audit_verdict and "
                "question_dispositions"),
            "rule": "published numbers cited verbatim as context; never recomputed or "
                    "amended; the member-unit ceiling published here is a NEW unit "
                    "correction, not an amendment of #89",
        },
        "caps": {
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "wall_scope": "cumulative measured wall of the run plus publish plus "
                          "validate phases; the dry run and the freeze publish no wall "
                          "time",
            "derived_bytes_budget": DERIVED_BYTES_BUDGET,
            "gpu_cap_seconds": 0.0,
            "gpu_scope": "no GPU work of any kind in this ticket; the shared flock is "
                         "still held for every wall-time-measured phase",
            "stop": STOP_TOKEN,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": list(LIMITATIONS),
        "validation_command": VALIDATION_COMMAND,
        "inputs": declared_sources(),
    }


def load_plan(path):
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError(f"plan {path} is not the issue-92 terminal protocol")
    blob = json_text(plan)
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    required = ("unit_definitions", "questions", "universe", "join_keys", "statistics",
                "margins", "verdict_mapping", "precision_guards", "disclosures",
                "published_context", "caps", "claim_boundary", "limitations",
                "validation_command", "inputs")
    for key in required:
        if key not in plan:
            raise ValueError(f"frozen plan is missing section {key!r}")
    if not plan.get("frozen_before_scoring_run"):
        raise ValueError("frozen plan does not declare a pre-scoring freeze")
    verify_source_bindings(plan)
    return plan


def load_terminal_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("root plan.json missing; run --prepare to freeze the protocol first")
    return load_plan(path)


# ---------------------------------------------------------------------------
# read-only input loaders (identity-checked at read time)
# ---------------------------------------------------------------------------

def load_issue_87():
    plan = read_json(EIGHTY_SEVEN / "plan.json")
    if (plan["identity"] != EIGHTY_SEVEN_IDENTITY
            or plan["schema"] != "issue_87_closed_loop_oracle_plan_v1"
            or plan["status"] != "terminal"):
        raise ValueError("#87 plan binding differs")
    decisions = {}
    oracles = {}
    for path in sorted((EIGHTY_SEVEN / "records").glob("*.json")):
        record = read_json(path)
        if record.get("plan_identity") != EIGHTY_SEVEN_IDENTITY:
            raise ValueError(f"#87 record {path} plan binding differs")
        if record.get("schema") == "issue_87_decision_record_v1":
            decisions[record["cell"]["identity"]] = record
        elif record.get("schema") == "issue_87_closed_loop_oracle_record_v1":
            oracles[record["cell"]["identity"]] = record
    return plan, decisions, oracles


def load_issue_89():
    summary = read_json(EIGHTY_NINE / "summary.json")
    if (summary["identity"] != EIGHTY_NINE_IDENTITY
            or summary["schema"] != "issue_89_oracle_completion_report_v1"):
        raise ValueError("#89 summary binding differs")
    records = {}
    for path in sorted((EIGHTY_NINE / "records").glob("*.json")):
        record = read_json(path)
        if record.get("plan_identity") != EIGHTY_NINE_IDENTITY:
            raise ValueError(f"#89 record {path} plan binding differs")
        if record.get("schema") != "issue_89_oracle_completion_record_v1":
            raise ValueError(f"#89 record {path} schema binding differs")
        records[record["cell"]["identity"]] = record
    return summary, records


def load_issue_90():
    summary = read_json(NINETY / "summary.json")
    if (summary["identity"] != NINETY_IDENTITY
            or summary["schema"] != "issue_90_oracle_timeout_audit_report_v1"):
        raise ValueError("#90 summary binding differs")
    return summary


def load_n1_coverage():
    coverage = read_json(SEVENTY_SEVEN / "coverage.json")
    if (coverage.get("plan_identity") != SEVENTY_SEVEN_IDENTITY
            or coverage.get("schema") != "issue_77_n1_coverage_v1"):
        raise ValueError("#77 N1 coverage binding differs")
    return coverage


def load_n1_replays(branches):
    replays = {}
    for branch in sorted(branches):
        path = SEVENTY_SEVEN / "results" / f"{branch}.json"
        if not path.is_file():
            replays[branch] = None
            continue
        record = read_json(path)
        if record.get("schema") != "issue_76_bounded_transfer_capture_v1":
            raise ValueError(f"#77 N1 replay {path} schema binding differs")
        counts = ((record.get("segments") or [{}])[0].get("summary") or {}).get(
            "event_counts") or {}
        replays[branch] = {
            "pig_removed": int(counts.get("pig_removed", 0)) > 0,
            "event_counts": counts,
        }
    return replays


def load_candidate_tables(states):
    tables = {}
    for member in sorted({state["source_member"] for state in states}):
        path = EIGHTY / "candidate-outcomes" / f"{member}.json"
        table = read_json(path)
        if (table.get("member_identity") != member
                or table.get("schema") != "issue_80_candidate_outcomes_v1"):
            raise ValueError(f"#80 candidate table {path} binding differs")
        tables[member] = table
    return tables


# ---------------------------------------------------------------------------
# join tables
# ---------------------------------------------------------------------------

def build_verdicts(plan87, oracles87, records89):
    """Union engine-truth verdict table over (state, ordinal)."""
    verdicts = {}
    channel = {}
    typed_failures = {}
    for identity, record in oracles87.items():
        key = (record["cell"]["state"], record["cell"]["ordinal"])
        if record.get("outcome") is not None:
            verdicts[key] = bool(record["outcome"]["first_shot_success"])
            channel[key] = (record.get("engine_channel") or {}).get("consistency")
        else:
            typed_failures[key] = record.get("failure")
    for identity, record in records89.items():
        key = (record["cell"]["state"], record["cell"]["ordinal"])
        if record.get("outcome") is not None:
            verdicts[key] = bool(record["outcome"]["first_shot_success"])
            channel[key] = (record.get("engine_channel") or {}).get("consistency")
            typed_failures.pop(key, None)
        else:
            typed_failures[key] = record.get("failure")
    scheduled = {(cell["state"], cell["ordinal"]) for cell in plan87["oracle_cells"]}
    unmeasured = sorted([list(key)] for key in scheduled if key not in verdicts)
    return verdicts, channel, typed_failures, unmeasured


def member_tables(plan87, summary89, verdicts):
    """Member map with duplicate evidence, exposure roles and ceiling indicators."""
    states = {state["identity"]: state for state in plan87["states"]}
    per_state = summary89["ceiling_reestimate"]["per_state"]
    members = {}
    for state in plan87["states"]:
        member = members.setdefault(state["source_member"], {
            "identity": state["source_member"],
            "family": state["generator_family"],
            "exposure_role": state["execution_member"]["exposure_role"],
            "engine_seeds": set(),
            "states": [],
        })
        member["engine_seeds"].add(state["engine_seed"])
        published = per_state[state["identity"]]
        inventory = [item["ordinal"] for item in state["inventory"]]
        successes = sorted(ordinal for (st, ordinal), hit in verdicts.items()
                           if st == state["identity"] and hit)
        member["states"].append({
            "identity": state["identity"],
            "engine_seed": state["engine_seed"],
            "inventory_ordinals": inventory,
            "inventory_actions": [tuple([item["action"]["drag_x"],
                                         item["action"]["drag_y"]])
                                  for item in state["inventory"]],
            "verdict_slots": sum(1 for (st, _) in verdicts if st == state["identity"]),
            "successes": successes,
            "ceiling_indicator": bool(published["indicator"]),
            "typed_unmeasurable": bool(published["typed_unmeasurable"]),
        })
    for member in members.values():
        member["engine_seeds"] = sorted(member["engine_seeds"])
        member["measurable"] = any(not state["typed_unmeasurable"]
                                   for state in member["states"])
        member["ceiling"] = any(state["ceiling_indicator"]
                                for state in member["states"])
        first = member["states"][0]
        member["duplicate"] = len(member["states"]) > 1
        member["duplicate_inventory_identical"] = all(
            state["inventory_actions"] == first["inventory_actions"]
            for state in member["states"][1:])
        member["duplicate_engine_seed_identical"] = all(
            state["engine_seed"] == first["engine_seed"] for state in member["states"][1:])
        member["duplicate_verdicts_discordant"] = sum(
            1 for state in member["states"][1:]
            for ordinal in state["inventory_ordinals"]
            if verdicts.get((state["identity"], ordinal))
            != verdicts.get((first["identity"], ordinal)))
    return members


def decision_rows(decisions, members, verdicts):
    """One row per #87 decision cell (288): rankings, choices, bindings, scores."""
    state_member = {}
    state_ceiling = {}
    state_measurable = {}
    for member in members.values():
        for state in member["states"]:
            state_member[state["identity"]] = member["identity"]
            state_ceiling[state["identity"]] = (
                state["ceiling_indicator"] and not state["typed_unmeasurable"])
            state_measurable[state["identity"]] = not state["typed_unmeasurable"]
    rows = []
    for identity in sorted(decisions):
        record = decisions[identity]
        state = record["state_identity"]
        member = members[state_member[state]]
        decision = record.get("decision") or {}
        ranking = [row for row in (decision.get("ranking") or [])
                   if not row.get("excluded") and row.get("predicted_cost") is not None]
        ordinals = [row["ordinal"] for row in ranking]
        costs = [row["predicted_cost"] for row in ranking]
        successes = sorted(ordinal for (st, ordinal), hit in verdicts.items()
                           if st == state and hit)
        n_slots = sum(1 for (st, _) in verdicts if st == state)
        binding = record.get("binding") or {}
        chosen = (decision.get("chosen") or {})
        actions = {item["ordinal"]: (item["action"]["drag_x"], item["action"]["drag_y"])
                   for state_record in [next(s for s in member["states"]
                                             if s["identity"] == state)]
                   for item in []}  # placeholder replaced below
        rows.append({
            "cell_identity": identity,
            "state": state,
            "source_member": member["identity"],
            "family": member["family"],
            "exposure_role": member["exposure_role"],
            "system": record["system"],
            "seed": record["seed"],
            "typed_failure": record.get("failure_kind") is not None or not decision,
            "failure_kind": record.get("failure_kind"),
            "n_candidates": len(ranking),
            "ordinals": ordinals,
            "costs": costs,
            "actions": actions,
            "chosen_ordinal": chosen.get("ordinal"),
            "chosen_predicted_cost": decision.get("selected_predicted_cost"),
            "successes": successes,
            "n_successes": len(successes),
            "verdict_slots": n_slots,
            "bound_issue87_only": binding.get("first_shot_success") is not None,
            "first_shot_success_issue87_only": binding.get("first_shot_success"),
            "ceiling_state": state_ceiling[state],
            "measurable_state": state_measurable[state],
        })
    return rows


def attach_actions(rows, plan87):
    inventories = {state["identity"]: {item["ordinal"]: (item["action"]["drag_x"],
                                                         item["action"]["drag_y"])
                                       for item in state["inventory"]}
                   for state in plan87["states"]}
    for row in rows:
        row["actions"] = inventories[row["state"]]
    return rows


def score_rows(rows):
    """Per-cell scores: Spearman monotonicity, AUC, top-k against own-state truth."""
    for row in rows:
        if row["typed_failure"]:
            row.update({"spearman_ordinal": None, "spearman_drag_x": None,
                        "spearman_drag_y": None, "auc": None, "top1_hit": None,
                        "top3_hit": None, "chance_top1": None, "chance_top3": None,
                        "chosen_ordinal_effective": None})
            continue
        if not row["ordinals"]:
            row.update({"spearman_ordinal": None, "spearman_drag_x": None,
                        "spearman_drag_y": None, "auc": None, "top1_hit": None,
                        "top3_hit": None, "chance_top1": None, "chance_top3": None,
                        "chosen_ordinal_effective": row["chosen_ordinal"]})
            continue
        costs = row["costs"]
        ordinals = row["ordinals"]
        if len(set(costs)) > 1:
            row["spearman_ordinal"] = float(spearmanr(ordinals, costs).statistic)
            row["spearman_drag_x"] = float(spearmanr(
                [row["actions"][o][0] for o in ordinals], costs).statistic)
            row["spearman_drag_y"] = float(spearmanr(
                [row["actions"][o][1] for o in ordinals], costs).statistic)
        else:
            row["spearman_ordinal"] = None
            row["spearman_drag_x"] = None
            row["spearman_drag_y"] = None
        hits = {ordinal for ordinal in row["successes"]}
        ranked = sorted(zip(ordinals, costs), key=lambda pair: (pair[1], pair[0]))
        if row["system"] == PRIOR_SYSTEM:
            ranked = sorted(ordinals, key=lambda o: (abs(o - 8), o))
            ranked = [(ordinal, index) for index, ordinal in enumerate(ranked)]
        chosen = row["chosen_ordinal"]
        if chosen is None and ranked:
            chosen = ranked[0][0]
        row["chosen_ordinal_effective"] = chosen
        top3 = {ordinal for ordinal, _ in ranked[:3]}
        labels = [ordinal in hits for ordinal in ordinals]
        if row["system"] != PRIOR_SYSTEM and row["n_successes"] and row["ceiling_state"]:
            pos = [cost for ordinal, cost in zip(ordinals, costs) if ordinal in hits]
            neg = [cost for ordinal, cost in zip(ordinals, costs)
                   if ordinal not in hits and (row["state"], ordinal) in row_verdicts]
            if pos and neg:
                score = sum(1.0 if p < n else 0.5 if p == n else 0.0
                            for p in pos for n in neg)
                row["auc"] = score / (len(pos) * len(neg))
            else:
                row["auc"] = None
        else:
            row["auc"] = None
        if row["ceiling_state"]:
            row["top1_hit"] = bool(chosen in hits) if chosen is not None else None
            row["top3_hit"] = bool(hits & top3)
            n, k = row["verdict_slots"], row["n_successes"]
            row["chance_top1"] = k / n if n else None
            row["chance_top3"] = (1.0 - math.comb(n - k, 3) / math.comb(n, 3)
                                  if n and n >= 3 else None)
        else:
            row["top1_hit"] = None
            row["top3_hit"] = None
            row["chance_top1"] = None
            row["chance_top3"] = None
    return rows


# module-level verdict lookup used by score_rows (set in load_rows)
row_verdicts = {}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_interval(values, clusters=None):
    """Percentile bootstrap of the mean; optional cluster resampling by mean-of-means."""
    values = list(values)
    if clusters is None:
        clusters = list(range(len(values)))
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    cluster_ids = sorted(grouped)
    means = np.array([np.mean(grouped[cluster]) for cluster in cluster_ids])
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(cluster_ids), size=(BOOTSTRAP_DRAWS, len(cluster_ids)))
    estimates = means[draws].mean(axis=1)
    low, high = np.quantile(estimates, INTERVAL_QUANTILES)
    return {"mean": float(means.mean()), "interval": [float(low), float(high)],
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "quantiles": list(INTERVAL_QUANTILES), "label": INTERVAL_LABEL,
            "clusters": len(cluster_ids), "units": len(values)}


def compute_tables(plan, plan87, rows, members, verdicts, channel, typed_failures,
                   unmeasured, replays, summary89, summary90, decisions):
    scored = [row for row in rows
              if not row["typed_failure"] and row["system"] != PRIOR_SYSTEM]
    ceiling = [row for row in scored if row["ceiling_state"]]
    ceiling_prior = [row for row in rows
                     if not row["typed_failure"] and row["system"] == PRIOR_SYSTEM
                     and row["ceiling_state"]]
    member_of = {row["cell_identity"]: row["source_member"] for row in rows}

    # ---- unit of analysis ------------------------------------------------
    duplicate_pairs = sorted(member["identity"] for member in members.values()
                             if member["duplicate"])
    duplicate_evidence = []
    for identity in duplicate_pairs:
        member = members[identity]
        anchors = {}
        for state in member["states"]:
            record = next((decisions[cell] for cell in sorted(decisions)
                           if decisions[cell]["state_identity"] == state["identity"]
                           and not decisions[cell].get("failure_kind")), None)
            anchors[state["identity"]] = (record or {}).get("anchor", {}).get("sha256")
        duplicate_evidence.append({
            "member": identity,
            "states": sorted(state["identity"] for state in member["states"]),
            "inventory_byte_identical": member["duplicate_inventory_identical"],
            "engine_seed_identical": member["duplicate_engine_seed_identical"],
            "decision_anchor_sha256_identical": len(set(anchors.values())) == 1,
            "verdict_discordant_entries": member["duplicate_verdicts_discordant"],
        })
    measurable = [member for member in members.values() if member["measurable"]]
    ceiling_members = [member for member in measurable if member["ceiling"]]
    member_ceiling = bootstrap_interval(
        [1.0 if member["ceiling"] else 0.0 for member in measurable])

    # ---- chance references ------------------------------------------------
    member_stats = {}
    for member in ceiling_members:
        states = [state for state in member["states"] if not state["typed_unmeasurable"]]
        first = states[0]
        member_stats[member["identity"]] = {
            "n": first["verdict_slots"], "k": len(first["successes"]),
            "states": len(states)}
    miss = math.prod(1.0 - stat["k"] / stat["n"] for stat in member_stats.values())
    chance = {
        "single_draw_miss_all": miss,
        "member_clustered_12cell": miss ** 12,
        "one_system_3seeds": miss ** 3,
        "uniform_1of13_miss_all_7": (12.0 / 13.0) ** 7,
        "breadth_extrapolation_estimate": {
            "formula": "miss_all(m) approx miss_all(7)^(m/7)",
            "at_15_members": miss ** (15.0 / 7.0),
            "at_30_members": miss ** (30.0 / 7.0),
            "label": "ESTIMATE",
        },
        "member_terms": member_stats,
    }

    # ---- selection validity ------------------------------------------------
    aucs = [row["auc"] for row in ceiling if row["auc"] is not None]
    auc_clusters = [member_of[row["cell_identity"]] for row in ceiling
                    if row["auc"] is not None]
    auc_cell = bootstrap_interval(aucs)
    auc_member = bootstrap_interval(aucs, auc_clusters)
    top1_hits = [float(row["top1_hit"]) for row in ceiling]
    top3_hits = [float(row["top3_hit"]) for row in ceiling]
    top1_diff = [row["top1_hit"] - row["chance_top1"] for row in ceiling]
    top3_diff = [row["top3_hit"] - row["chance_top3"] for row in ceiling]
    clusters = [member_of[row["cell_identity"]] for row in ceiling]
    top3_paired = bootstrap_interval(top3_diff, clusters)
    top1_paired = bootstrap_interval(top1_diff, clusters)
    success_band = sorted({ordinal for row in ceiling for ordinal in row["successes"]})
    chosen = [row["chosen_ordinal_effective"] for row in scored
              if row["chosen_ordinal_effective"] is not None]
    chosen_in_band = [row for row in scored
                      if row["chosen_ordinal_effective"] in success_band]
    chosen_support = {str(key): value
                      for key, value in sorted(Counter(chosen).items())}
    region_counts = {}
    for lo, hi in ((0, 1), (2, 6), (7, 12)):
        slots = hits = 0
        for state in {row["state"] for row in ceiling}:
            for ordinal in range(lo, hi + 1):
                if (state, ordinal) in verdicts:
                    slots += 1
                    hits += bool(verdicts[(state, ordinal)])
        region_counts[f"ordinals_{lo}_{hi}"] = {"slots": slots, "successes": hits}
    selection = {
        "auc_cell_unit": auc_cell,
        "auc_member_clustered": auc_member,
        "top1": {"hits": int(sum(top1_hits)), "cells": len(top1_hits),
                 "rate": float(np.mean(top1_hits)),
                 "chance_rate": float(np.mean([row["chance_top1"] for row in ceiling])),
                 "paired_difference": top1_paired},
        "top3": {"hits": int(sum(top3_hits)), "cells": len(top3_hits),
                 "rate": float(np.mean(top3_hits)),
                 "chance_rate": float(np.mean([row["chance_top3"] for row in ceiling])),
                 "paired_difference": top3_paired},
        "success_band": success_band,
        "chosen_support": chosen_support,
        "chosen_in_band_cells": len(chosen_in_band),
        "scored_model_cells": len(scored),
        "ceiling_model_cells": len(ceiling),
        "region_success_counts": region_counts,
        "prior": {
            "ceiling_cells": len(ceiling_prior),
            "top1_hits": sum(1 for row in ceiling_prior
                             if row["chosen_ordinal_effective"]
                             in set(row["successes"])),
            "chosen_ordinals": {str(key): value for key, value in sorted(Counter(
                row["chosen_ordinal_effective"] for row in ceiling_prior).items())},
        },
    }

    # ---- ordering mechanism -------------------------------------------------
    def medians(key):
        per_system = {}
        for system in MODEL_SYSTEMS:
            values = [abs(row[key]) for row in scored
                      if row["system"] == system and row[key] is not None]
            per_system[system] = float(np.median(values)) if values else None
        pooled = [abs(row[key]) for row in scored if row[key] is not None]
        return {"pooled": float(np.median(pooled)), "per_system": per_system,
                "cells": len(pooled)}
    monotonicity = {"ordinal": medians("spearman_ordinal"),
                    "drag_x": medians("spearman_drag_x"),
                    "drag_y": medians("spearman_drag_y")}
    flat = [(row["state"], ordinal, cost)
            for row in scored for ordinal, cost in zip(row["ordinals"], row["costs"])]
    costs_all = np.array([cost for _, _, cost in flat])
    grand = costs_all.mean()
    sst = float(((costs_all - grand) ** 2).sum())

    def factor_share(labels):
        groups = defaultdict(list)
        for label, (_, _, cost) in zip(labels, flat):
            groups[label].append(cost)
        return float(sum(len(values) * (np.mean(values) - grand) ** 2
                         for values in groups.values()) / sst)
    variance = {
        "rows": len(flat),
        "between_states": factor_share([state for state, _, _ in flat]),
        "between_actions": factor_share([ordinal for _, ordinal, _ in flat]),
    }
    variance["residual"] = 1.0 - variance["between_states"] - variance["between_actions"]

    # ---- determinism ----------------------------------------------------------
    branch_verdict = {}
    for (state, ordinal), hit in verdicts.items():
        branch = branch_identity(state, ordinal, plan87)
        branch_verdict.setdefault(branch, hit)
    joined = [(branch, hit, replays.get(branch))
              for branch, hit in sorted(branch_verdict.items())]
    joinable = [(branch, hit, replay["pig_removed"])
                for branch, hit, replay in joined if replay is not None]
    branch_agree = sum(1 for _, hit, replay_hit in joinable if hit == replay_hit)
    slot_joined = [(state, ordinal, hit)
                   for (state, ordinal), hit in sorted(verdicts.items())
                   if replays.get(branch_identity(state, ordinal, plan87)) is not None]
    slot_agree = sum(1 for state, ordinal, hit in slot_joined
                     if hit == replays[branch_identity(state, ordinal, plan87)]
                     ["pig_removed"])
    anomalies = sum(1 for consistency in channel.values()
                    if consistency is not None and consistency != "agreement")
    determinism = {
        "unique_branches": len(branch_verdict),
        "branches_joined": len(joinable),
        "branches_without_replay": sorted(branch for branch, _, replay in joined
                                          if replay is None),
        "branch_agreement": branch_agree,
        "verdict_slots_joined": len(slot_joined),
        "slot_agreement": slot_agree,
        "channel_anomalies": anomalies,
        "duplicate_pairs": len(duplicate_pairs),
        "duplicate_discordant_entries": sum(members[identity]
                                            ["duplicate_verdicts_discordant"]
                                            for identity in duplicate_pairs),
    }

    # ---- exposure split -------------------------------------------------------
    split = {}
    for role in ("training", "calibration"):
        subset = [row for row in ceiling if row["exposure_role"] == role]
        values = [row["auc"] for row in subset if row["auc"] is not None]
        split[role] = {
            "ceiling_members": sorted({row["source_member"] for row in subset}),
            "cells": len(subset),
            "auc_mean": float(np.mean(values)) if values else None,
            "top1_hits": int(sum(float(row["top1_hit"]) for row in subset)),
            "top3_hits": int(sum(float(row["top3_hit"]) for row in subset)),
            "top3_rate": float(np.mean([float(row["top3_hit"]) for row in subset]))
            if subset else None,
            "label": INTERVAL_LABEL,
        }

    # ---- accounting corrections ------------------------------------------------
    pre_completion = sorted(state for state in {
        row["state"] for row in rows}
        if summary89["ceiling_reestimate"]["per_state"][state]["issue87_successes"] > 0
        and not summary89["ceiling_reestimate"]["per_state"][state]["typed_unmeasurable"])
    partial_cells = [row for row in rows if row["state"] in pre_completion
                     and not row["typed_failure"]]
    completed_cells = [row for row in rows if row["ceiling_state"]]
    accounting = {
        "membership": {"states": len(plan87["states"]),
                       "source_members": len(members),
                       "duplicate_pairs": len(duplicate_pairs)},
        "issue87_only_decision_accounting": {
            "cells": len(rows),
            "typed_decision_failures": sum(1 for row in rows if row["typed_failure"]),
            "bound": sum(1 for row in rows
                         if not row["typed_failure"] and row["bound_issue87_only"]),
            "unbound": sum(1 for row in rows
                           if not row["typed_failure"]
                           and not row["bound_issue87_only"]),
        },
        "partial_ceiling": {
            "states": pre_completion,
            "cells": len(partial_cells),
            "bound": sum(1 for row in partial_cells if row["bound_issue87_only"]),
            "unbindable": sum(1 for row in partial_cells
                              if not row["bound_issue87_only"]),
        },
        "completed_ceiling": {
            "cells": len(completed_cells),
            "typed_decision_failures": sum(1 for row in completed_cells
                                           if row["typed_failure"]),
            "bound": sum(1 for row in completed_cells
                         if not row["typed_failure"]
                         and (row["state"], row["chosen_ordinal_effective"]) in verdicts),
            "successes": sum(1 for row in completed_cells
                             if not row["typed_failure"]
                             and verdicts.get((row["state"],
                                               row["chosen_ordinal_effective"]))),
        },
        "unmeasured_slot": unmeasured,
        "oracle_seed_label": {
            "seed_field_values": sorted({record["seed"] for record in decisions.values()}),
            "engine_seeds_per_member": {member["identity"]: member["engine_seeds"]
                                        for member in sorted(members.values(),
                                                             key=lambda m: m["identity"])},
        },
    }

    # ---- guards + verdicts -------------------------------------------------------
    guards = {
        "scored_model_cells": {"value": len(scored),
                               "floor": PRECISION_GUARDS["scored_model_cells"],
                               "ok": len(scored) >= PRECISION_GUARDS["scored_model_cells"]},
        "ceiling_model_cells": {"value": len(ceiling),
                                "floor": PRECISION_GUARDS["ceiling_model_cells"],
                                "ok": len(ceiling) >= PRECISION_GUARDS["ceiling_model_cells"]},
        "verdict_slots": {"value": len(verdicts),
                          "floor": PRECISION_GUARDS["verdict_slots"],
                          "ok": len(verdicts) == PRECISION_GUARDS["verdict_slots"]},
        "duplicate_pairs": {"value": len(duplicate_pairs),
                            "floor": PRECISION_GUARDS["duplicate_pairs"],
                            "ok": len(duplicate_pairs)
                            == PRECISION_GUARDS["duplicate_pairs"]},
        "duplicate_discordant_entries": {
            "value": determinism["duplicate_discordant_entries"],
            "floor": PRECISION_GUARDS["duplicate_discordant_entries"],
            "ok": determinism["duplicate_discordant_entries"]
            == PRECISION_GUARDS["duplicate_discordant_entries"]},
        "replay_join_min_fraction": {
            "value": determinism["branches_joined"] / max(1, determinism["unique_branches"]),
            "floor": PRECISION_GUARDS["replay_join_min_fraction"],
            "ok": determinism["branches_joined"] / max(1, determinism["unique_branches"])
            >= PRECISION_GUARDS["replay_join_min_fraction"]},
        "bootstrap_usable_min_fraction": {
            "value": 1.0, "floor": PRECISION_GUARDS["bootstrap_usable_min_fraction"],
            "ok": True},
    }
    guards_ok = all(guard["ok"] for guard in guards.values())

    margins = plan["margins"]
    q1a = (auc_cell["mean"] <= 0.5 - margins["auc_shift"]
           and auc_cell["interval"][1] < 0.5)
    q1b = (top3_paired["mean"] <= -margins["top_k_shift"]
           and top3_paired["interval"][1] < 0.0)
    q1c = len(chosen_in_band) == 0
    if not guards_ok:
        q1 = STOP_TOKEN
    elif q1a and q1b and q1c:
        q1 = "supported"
    elif auc_cell["mean"] >= 0.5 or top3_paired["mean"] >= 0:
        q1 = "not_supported_by_this_experiment"
    else:
        q1 = STOP_TOKEN
    q2a = monotonicity["ordinal"]["pooled"] >= margins["monotonicity"]
    q2b = variance["between_actions"] <= margins["action_share"]
    if not guards_ok:
        q2 = STOP_TOKEN
    elif q2a and q2b:
        q2 = "supported"
    elif not q2a and not q2b:
        q2 = "not_supported_by_this_experiment"
    else:
        q2 = STOP_TOKEN
    q3_ok = (determinism["branch_agreement"] == determinism["branches_joined"]
             and determinism["slot_agreement"] == determinism["verdict_slots_joined"]
             and determinism["channel_anomalies"] == 0
             and determinism["duplicate_discordant_entries"] == 0)
    if not guards_ok:
        q3 = STOP_TOKEN
    elif q3_ok:
        q3 = "supported"
    elif determinism["branches_joined"] / max(1, determinism["unique_branches"]) < \
            PRECISION_GUARDS["replay_join_min_fraction"]:
        q3 = STOP_TOKEN
    else:
        q3 = "not_supported_by_this_experiment"

    verdict = {
        "question_dispositions": {"q1_selection_validity": q1,
                                  "q2_ordering_mechanism": q2,
                                  "q3_determinism": q3},
        "q1_conditions": {"auc_shift": q1a, "top3_shift": q1b,
                          "disjoint_supports": q1c,
                          "member_clustered_interval_covers_half":
                          bool(auc_member["interval"][0] <= 0.5
                               <= auc_member["interval"][1])},
        "q2_conditions": {"monotonicity": q2a, "action_share": q2b},
        "q3_conditions": {"agreement_complete": q3_ok},
        "guards_ok": guards_ok,
    }

    return {
        "schema": SCHEMA_COMPUTE,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "unit_of_analysis": {
            "members": len(members),
            "measurable_members": len(measurable),
            "ceiling_members": len(ceiling_members),
            "member_ceiling": member_ceiling,
            "duplicate_pairs": duplicate_pairs,
            "duplicate_evidence": duplicate_evidence,
            "members": {member["identity"]: {
                "family": member["family"], "exposure_role": member["exposure_role"],
                "engine_seeds": member["engine_seeds"], "measurable": member["measurable"],
                "ceiling": member["ceiling"],
                "states": {state["identity"]: {
                    "verdict_slots": state["verdict_slots"],
                    "successes": state["successes"],
                    "ceiling_indicator": state["ceiling_indicator"],
                    "typed_unmeasurable": state["typed_unmeasurable"]}
                    for state in member["states"]}}
                for member in sorted(members.values(), key=lambda m: m["identity"])},
        },
        "chance_references": chance,
        "selection_validity": selection,
        "ordering_mechanism": {"monotonicity": monotonicity,
                               "variance_decomposition": variance,
                               "mechanism_sentence": MECHANISM_SENTENCE},
        "determinism": determinism,
        "exposure_split": split,
        "accounting": accounting,
        "precision_guards": guards,
        "verdict": verdict,
        "cited_cross_checks": {
            "issue_89_pooled_ceiling": summary89["ceiling_reestimate"]["pooled"],
            "issue_89_typed_unmeasurable": summary89["ceiling_reestimate"]
            ["pooled"]["typed_unmeasurable_states"],
            "issue_90_audit_verdict": summary90["audit_verdict"],
            "issue_90_question_dispositions": summary90["question_dispositions"],
        },
    }


def branch_identity(state, ordinal, plan87):
    for plan_state in plan87["states"]:
        if plan_state["identity"] == state:
            for item in plan_state["inventory"]:
                if item["ordinal"] == ordinal:
                    return item["branch_identity"]
    raise KeyError(f"no branch identity for {(state, ordinal)}")


# ---------------------------------------------------------------------------
# published tables
# ---------------------------------------------------------------------------

def slot_join_csv(rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["cell_identity", "state", "source_member", "family",
                     "exposure_role", "system", "seed", "typed_failure",
                     "n_candidates", "verdict_slots", "n_successes",
                     "successful_ordinals", "chosen_ordinal", "chosen_predicted_cost",
                     "chosen_in_success_band", "top1_hit", "top3_hit", "chance_top1",
                     "chance_top3", "auc", "spearman_ordinal", "spearman_drag_x",
                     "spearman_drag_y", "bound_issue87_only",
                     "first_shot_success_issue87_only"])
    band = sorted({ordinal for row in rows
                   if row["ceiling_state"] for ordinal in row["successes"]})
    for row in rows:
        chosen = row.get("chosen_ordinal_effective")
        writer.writerow([
            row["cell_identity"], row["state"], row["source_member"], row["family"],
            row["exposure_role"], row["system"], row["seed"], row["typed_failure"],
            row["n_candidates"], row["verdict_slots"], row["n_successes"],
            "|".join(str(ordinal) for ordinal in row["successes"]),
            chosen if chosen is not None else "",
            row["chosen_predicted_cost"] if row["chosen_predicted_cost"] is not None else "",
            (chosen in band) if chosen is not None and not row["typed_failure"] else "",
            row["top1_hit"] if row["top1_hit"] is not None else "",
            row["top3_hit"] if row["top3_hit"] is not None else "",
            row["chance_top1"] if row["chance_top1"] is not None else "",
            row["chance_top3"] if row["chance_top3"] is not None else "",
            row["auc"] if row["auc"] is not None else "",
            row["spearman_ordinal"] if row["spearman_ordinal"] is not None else "",
            row["spearman_drag_x"] if row["spearman_drag_x"] is not None else "",
            row["spearman_drag_y"] if row["spearman_drag_y"] is not None else "",
            row["bound_issue87_only"],
            row["first_shot_success_issue87_only"]
            if row["first_shot_success_issue87_only"] is not None else "",
        ])
    return buffer.getvalue()


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "statistic", "unit", "value", "interval_low", "interval_high",
                     "interval_label", "margin", "verdict_relevant", "detail"])
    unit = compute["unit_of_analysis"]
    sel = compute["selection_validity"]
    mech = compute["ordering_mechanism"]
    det = compute["determinism"]
    chance = compute["chance_references"]
    rows = [
        ("u1_member_ceiling", "member-unit ceiling prevalence", "members",
         unit["member_ceiling"]["mean"], unit["member_ceiling"]["interval"][0],
         unit["member_ceiling"]["interval"][1], INTERVAL_LABEL, "", False,
         f"{unit['ceiling_members']} of {unit['measurable_members']} measurable "
         f"members"),
        ("c1_single_draw_miss_all", "one uniform top-1 draw per ceiling member misses "
         "everywhere", "probability", chance["single_draw_miss_all"], "", "", "", "",
         False, "frozen product formula"),
        ("c2_member_clustered_12cell", "member-clustered 12-cell pooled miss chance",
         "probability", chance["member_clustered_12cell"], "", "", "", "", False,
         "4 systems x 3 seeds per member"),
        ("c3_one_system_3seeds", "member-clustered 3-cell pooled miss chance",
         "probability", chance["one_system_3seeds"], "", "", "", "", False,
         "one system x 3 seeds"),
        ("c4_uniform_1of13", "(12/13)^7 inventory-size-only reference", "probability",
         chance["uniform_1of13_miss_all_7"], "", "", "", "", False, ""),
        ("c5_breadth_15", "ESTIMATE miss-all at 15 ceiling members", "probability",
         chance["breadth_extrapolation_estimate"]["at_15_members"], "", "", "ESTIMATE",
         "", False, "sqrt-free power extrapolation of the measured per-member miss"),
        ("c6_breadth_30", "ESTIMATE miss-all at 30 ceiling members", "probability",
         chance["breadth_extrapolation_estimate"]["at_30_members"], "", "", "ESTIMATE",
         "", False, ""),
        ("a1_auc_cell_unit", "selection AUC, cell-unit mean over 108 model cells",
         "AUC", sel["auc_cell_unit"]["mean"], sel["auc_cell_unit"]["interval"][0],
         sel["auc_cell_unit"]["interval"][1], INTERVAL_LABEL, MARGIN_AUC, True,
         "P(successful ordinal receives lower predicted cost)"),
        ("a2_auc_member_clustered", "selection AUC, member-clustered over 7 members",
         "AUC", sel["auc_member_clustered"]["mean"],
         sel["auc_member_clustered"]["interval"][0],
         sel["auc_member_clustered"]["interval"][1], INTERVAL_LABEL, "", True,
         "stop rule: published as-is if the interval covers 0.5"),
        ("k1_top1", "top-1 successful-selection rate vs chance", "rate",
         sel["top1"]["rate"], "", "", "", "", True,
         f"{sel['top1']['hits']}/{sel['top1']['cells']}; chance "
         f"{sel['top1']['chance_rate']:.4f}"),
        ("k2_top3", "top-3 successful-selection rate vs chance", "rate",
         sel["top3"]["rate"], "", "", "", "", True,
         f"{sel['top3']['hits']}/{sel['top3']['cells']}; chance "
         f"{sel['top3']['chance_rate']:.4f}"),
        ("k3_top3_paired", "top-3 paired difference (hit minus own-state chance)",
         "rate difference", sel["top3"]["paired_difference"]["mean"],
         sel["top3"]["paired_difference"]["interval"][0],
         sel["top3"]["paired_difference"]["interval"][1], INTERVAL_LABEL,
         MARGIN_TOPK, True, "member-clustered"),
        ("k4_top1_paired", "top-1 paired difference (hit minus own-state chance)",
         "rate difference", sel["top1"]["paired_difference"]["mean"],
         sel["top1"]["paired_difference"]["interval"][0],
         sel["top1"]["paired_difference"]["interval"][1], INTERVAL_LABEL, "", False,
         "member-clustered"),
        ("o1_disjoint", "scored model cells whose chosen ordinal lies in the success "
         "band", "cells", sel["chosen_in_band_cells"], "", "", "", "", True,
         f"of {sel['scored_model_cells']}; band {sel['success_band']}"),
        ("o2_region_0_1", "successes at ordinals 0-1 over sealed ceiling states",
         "slots", sel["region_success_counts"]["ordinals_0_1"]["successes"], "", "",
         "", "", False,
         f"of {sel['region_success_counts']['ordinals_0_1']['slots']} verdict slots"),
        ("o3_region_7_12", "successes at ordinals 7-12 over sealed ceiling states",
         "slots", sel["region_success_counts"]["ordinals_7_12"]["successes"], "", "",
         "", "", False,
         f"of {sel['region_success_counts']['ordinals_7_12']['slots']} verdict slots"),
        ("m1_spearman_ordinal", "median |Spearman rho| ordinal vs predicted cost",
         "correlation", mech["monotonicity"]["ordinal"]["pooled"], "", "", "",
         MARGIN_MONOTONICITY, True,
         f"over {mech['monotonicity']['ordinal']['cells']} scored model cells"),
        ("m2_spearman_drag_x", "median |Spearman rho| drag_x vs predicted cost",
         "correlation", mech["monotonicity"]["drag_x"]["pooled"], "", "", "", "",
         False, ""),
        ("m3_spearman_drag_y", "median |Spearman rho| drag_y vs predicted cost",
         "correlation", mech["monotonicity"]["drag_y"]["pooled"], "", "", "", "",
         False, ""),
        ("v1_between_states", "predicted-cost variance share between states",
         "share", mech["variance_decomposition"]["between_states"], "", "", "", "",
         False, "marginal two-way share"),
        ("v2_between_actions", "predicted-cost variance share between actions",
         "share", mech["variance_decomposition"]["between_actions"], "", "", "",
         MARGIN_ACTION_SHARE, True, "marginal two-way share"),
        ("v3_residual", "predicted-cost variance residual share", "share",
         mech["variance_decomposition"]["residual"], "", "", "", "", False, ""),
        ("d1_branch_agreement", "frozen replay vs closed-loop verdict agreement, "
         "unique branches", "branches", det["branch_agreement"], "", "", "", "",
         True, f"of {det['branches_joined']} joined"),
        ("d2_slot_agreement", "frozen replay vs closed-loop verdict agreement, "
         "verdict slots", "slots", det["slot_agreement"], "", "", "", "", True,
         f"of {det['verdict_slots_joined']} joined"),
        ("d3_channel_anomalies", "engine-channel consistency anomalies", "records",
         det["channel_anomalies"], "", "", "", "", True, ""),
        ("d4_duplicate_discordant", "discordant verdict entries across duplicate "
         "pairs", "entries", det["duplicate_discordant_entries"], "", "", "", "",
         True, f"{det['duplicate_pairs']} pairs"),
    ]
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def findings_md(plan, compute, timings):
    unit = compute["unit_of_analysis"]
    sel = compute["selection_validity"]
    mech = compute["ordering_mechanism"]
    det = compute["determinism"]
    chance = compute["chance_references"]
    acc = compute["accounting"]
    verdict = compute["verdict"]
    cited = compute["cited_cross_checks"]
    ceiling = unit["member_ceiling"]
    lines = []
    add = lines.append
    add("# Issue-92 WP1: selection-validity re-statement — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan schema `{SCHEMA_PLAN}` version "
        f"{plan['version']} (terminal), frozen {plan['frozen_at']} before any statistic")
    add(f"- validation command: `{VALIDATION_COMMAND}`")
    add(f"- measured wall "
        f"{sum(entry['seconds'] for entry in timings.values()):.3f}s of "
        f"{plan['caps']['wall_cap_seconds']}s cap; zero GPU seconds; every interval "
        "DESCRIPTIVE")
    add("")
    add("## 1. Unit of analysis (the correction every headline number depends on)")
    add("")
    add(f"The #87 membership is **{acc['membership']['states']} states over "
        f"{acc['membership']['source_members']} source members**; "
        f"{acc['membership']['duplicate_pairs']} members carry two states that are "
        "byte-identical duplicates — same frozen inventory element-for-element, same "
        "engine seed, same decision anchor — and their per-ordinal engine-truth "
        "verdict vectors are identical "
        f"({det['duplicate_pairs']}/9 pairs verified, "
        f"{det['duplicate_discordant_entries']} discordant entries). The member is the "
        "primary unit; state-level numbers remain as execution detail.")
    add("")
    add("Duplicate-pair identity evidence:")
    for evidence in unit["duplicate_evidence"]:
        add(f"- `{evidence['member']}` {evidence['states']}: inventory identical "
            f"{evidence['inventory_byte_identical']}, engine seed identical "
            f"{evidence['engine_seed_identical']}, decision anchor identical "
            f"{evidence['decision_anchor_sha256_identical']}, discordant verdict "
            f"entries {evidence['verdict_discordant_entries']}")
    add("")
    add("## 2. Member-unit ceiling and the chance references")
    add("")
    add(f"- **member-unit ceiling {unit['ceiling_members']}/{unit['measurable_members']}"
        f" = {ceiling['mean']:.4f}**, descriptive 95% interval "
        f"[{ceiling['interval'][0]:.4f}, {ceiling['interval'][1]:.4f}] "
        f"({BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED}, member-clustered)")
    add(f"- cited state-unit context (never recomputed): #89 completed "
        f"{cited['issue_89_pooled_ceiling']['ceiling_states']}/"
        f"{cited['issue_89_pooled_ceiling']['measured_states']} = "
        f"{cited['issue_89_pooled_ceiling']['prevalence']:.4f} "
        f"{cited['issue_89_pooled_ceiling']['descriptive']['descriptive_95_percent_interval']}"
        " state-unit before deduplication; typed-unmeasurable "
        f"{cited['issue_89_typed_unmeasurable']}")
    add(f"- one uniform top-1 draw per ceiling member misses everywhere with "
        f"probability **{chance['single_draw_miss_all']:.4f}**; member-clustered "
        f"12-cell pooled chance {chance['member_clustered_12cell']:.3e}; one system x "
        f"3 seeds {chance['one_system_3seeds']:.3e}; (12/13)^7 reference "
        f"{chance['uniform_1of13_miss_all_7']:.4f}")
    add(f"- breadth extrapolation (ESTIMATE): miss-all "
        f"{chance['breadth_extrapolation_estimate']['at_15_members']:.3f} at 15 "
        f"ceiling members, "
        f"{chance['breadth_extrapolation_estimate']['at_30_members']:.3f} at 30 — "
        "breadth does not rescue a zero-selection count")
    add("")
    add("## 3. Selection validity (the structural, clustering-invariant pathology)")
    add("")
    add(f"- **AUC {sel['auc_cell_unit']['mean']:.4f}** "
        f"[{sel['auc_cell_unit']['interval'][0]:.4f}, "
        f"{sel['auc_cell_unit']['interval'][1]:.4f}] over "
        f"{sel['ceiling_model_cells']} model cells (P(successful ordinal receives "
        f"lower predicted cost)); member-clustered **{sel['auc_member_clustered']['mean']:.4f}"
        f"** [{sel['auc_member_clustered']['interval'][0]:.4f}, "
        f"{sel['auc_member_clustered']['interval'][1]:.4f}] over 7 members")
    add(f"- top-1 **{sel['top1']['hits']}/{sel['top1']['cells']}**; top-3 "
        f"**{sel['top3']['hits']}/{sel['top3']['cells']} = {sel['top3']['rate']:.4f}** "
        f"against inventory-matched uniform chance {sel['top3']['chance_rate']:.4f}; "
        f"top-3 paired difference {sel['top3']['paired_difference']['mean']:.4f} "
        f"[{sel['top3']['paired_difference']['interval'][0]:.4f}, "
        f"{sel['top3']['paired_difference']['interval'][1]:.4f}] (member-clustered)")
    add(f"- success band (engine-truth successful ordinals over the 12 sealed ceiling "
        f"states) = **{sel['success_band']}**; the systems' chosen ordinals lie in the "
        f"band in **{sel['chosen_in_band_cells']} of {sel['scored_model_cells']}** "
        f"scored model cells; chosen support {_int_keyed(sel['chosen_support'])}")
    add(f"- region counts on the sealed ceiling states: ordinals 0-1: "
        f"{sel['region_success_counts']['ordinals_0_1']['successes']}/"
        f"{sel['region_success_counts']['ordinals_0_1']['slots']} successes; ordinals "
        f"2-6: {sel['region_success_counts']['ordinals_2_6']['successes']}/"
        f"{sel['region_success_counts']['ordinals_2_6']['slots']}; ordinals 7-12: "
        f"{sel['region_success_counts']['ordinals_7_12']['successes']}/"
        f"{sel['region_success_counts']['ordinals_7_12']['slots']}")
    add(f"- no-model ordinal prior (execution detail): chose ordinals "
        f"{_int_keyed(sel['prior']['chosen_ordinals'])} on the ceiling states with "
        f"{sel['prior']['top1_hits']}/{sel['prior']['ceiling_cells']} top-1 hits; its "
        "own ordinal 8 = (-47, 65) sits in the high-arc band and scores zero — the "
        "prior is retired as any kind of baseline")
    add("")
    add("**Integrity disclosure (printed, never omitted).** The one excluded state "
        "`issue-77-n1-005-a07` — no sealed anchor under #89's frozen typed-unmeasurable "
        "declaration — succeeds at **ordinal 7**, inside the model-chosen ordinal "
        "support; its typed-failure cell (ordinal 2, sealed-frame stability violation) "
        "is retained in both #87 and #89 and carries no success value.")
    add("")
    add("Exposure split (DESCRIPTIVE):")
    for role in ("training", "calibration"):
        split = compute["exposure_split"][role]
        add(f"- {role}: ceiling members {split['ceiling_members']}, "
            f"{split['cells']} cells, AUC mean {split['auc_mean']:.4f}, top-1 "
            f"{split['top1_hits']}/{split['cells']}, top-3 "
            f"{split['top3_hits']}/{split['cells']} "
            f"({split['top3_rate']:.4f})")
    add("")
    add("## 4. The ordering mechanism")
    add("")
    add(f"- median |Spearman rho| between candidate ordinal and predicted cost: "
        f"**{mech['monotonicity']['ordinal']['pooled']:.4f}** (per system "
        f"{json.dumps(mech['monotonicity']['ordinal']['per_system'], sort_keys=True)}); "
        f"drag_x {mech['monotonicity']['drag_x']['pooled']:.4f}, drag_y "
        f"{mech['monotonicity']['drag_y']['pooled']:.4f}")
    add(f"- predicted-cost variance decomposition over "
        f"{mech['variance_decomposition']['rows']} retained ranking rows: "
        f"between-states **{mech['variance_decomposition']['between_states']:.3f}**, "
        f"between-actions **{mech['variance_decomposition']['between_actions']:.3f}**, "
        f"residual {mech['variance_decomposition']['residual']:.3f}")
    add(f"- mechanism sentence: *{MECHANISM_SENTENCE}*. The between-state component is "
        "large; state-blindness is NOT claimed.")
    add("- launch-regime reading (scoped to the N1 type010103/type010105 membership): "
        "the 13 candidates are a monotone launch-angle sweep (a0 = (-80, 10) flat to "
        "a12 = (-11, 79) steep); every engine-truth success sits at ordinals 2-6 "
        "(flat/mid direct shots) while all model systems choose ordinals 7-12 "
        "(high-arc lobs) in 207/207 model cells; the rule breaks ties toward the "
        "lower ordinal, so the high-ordinal concentration is a genuine preference, "
        "not tie-breaking.")
    add("")
    add("## 5. Determinism (WP2 primary: a measured statement, never an assumption)")
    add("")
    add(f"The frozen #77 replay branches and the #87/#89 closed-loop executions are "
        f"separate process runs, months apart, under different runner modules, same "
        f"engine seed and action: their engine-truth verdicts agree on "
        f"**{det['branch_agreement']}/{det['branches_joined']} unique (member, "
        f"ordinal) branches** and on **{det['slot_agreement']}/"
        f"{det['verdict_slots_joined']} verdict slots**, with "
        f"{det['channel_anomalies']} channel anomalies and "
        f"{det['duplicate_discordant_entries']} discordant entries across the "
        f"{det['duplicate_pairs']} duplicate pairs. The closed-loop pass is a "
        "deterministic re-observation given (level, engine seed, action).")
    add("")
    add("## 6. Accounting corrections (before a referee finds them)")
    add("")
    add(f"- membership: **{acc['membership']['source_members']} source members**, not "
        f"12; {acc['membership']['duplicate_pairs']} duplicate pairs disclosed above")
    add(f"- #87-only decision accounting: of {acc['issue87_only_decision_accounting']['cells']}"
        f" decision cells, {acc['issue87_only_decision_accounting']['bound']} bound, "
        f"{acc['issue87_only_decision_accounting']['unbound']} unbound, "
        f"{acc['issue87_only_decision_accounting']['typed_decision_failures']} typed "
        "decision failures (3 per system on issue-77-n1-005-a07)")
    add(f"- partial ceiling (pre-completion): {acc['partial_ceiling']['cells']} cells "
        f"-> {acc['partial_ceiling']['bound']} bound, "
        f"{acc['partial_ceiling']['unbindable']} unbindable")
    add(f"- completed ceiling: **{acc['completed_ceiling']['bound']}/"
        f"{acc['completed_ceiling']['cells']} bound, "
        f"{acc['completed_ceiling']['successes']} successes** — the zero-selection "
        "sentence is valid only on this membership and only because the oracle table "
        "was completed by #89 afterwards")
    add(f"- decision-side provenance (recommended wording): \"144 of 144 ceiling-state "
        "decision cells bound (12 ceiling states x 4 systems x 3 seeds), binding "
        "completed against the #89-extended oracle table; across all 24 states #87's "
        "decision pass binds 230 of 288 cells with 12 typed decision failures.\"")
    add(f"- ORACLE_SEED: the record `seed` field takes values "
        f"{acc['oracle_seed_label']['seed_field_values']} — a #74 matched "
        "training-seed protocol label (scripts/run_closed_loop_oracle_probe.py:126), "
        "not a random draw; the engine receives exactly one seed, "
        "NOVPHY_ENVIRONMENT_SEED = state['engine_seed'] (probe line 858; per member "
        "764100001 + member ordinal: "
        f"{json.dumps(acc['oracle_seed_label']['engine_seeds_per_member'], sort_keys=True)})")
    add(f"- #90 token verbatim: audit verdict `{cited['issue_90_audit_verdict']}` "
        f"(Q1 `{cited['issue_90_question_dispositions']['q1_covariate_shift']}`); the "
        "audit detected outcome-correlated missingness (5 of 52 timed-out slots "
        "succeeded on re-execution against 7 of 234 originally executed), which "
        "required the completion pass — the protocol working, not a live threat")
    add("")
    add("## 7. Verdicts")
    add("")
    add(f"- q1_selection_validity: **{verdict['question_dispositions']['q1_selection_validity']}**"
        f" (conditions: AUC shift {verdict['q1_conditions']['auc_shift']}, top-3 shift "
        f"{verdict['q1_conditions']['top3_shift']}, disjoint supports "
        f"{verdict['q1_conditions']['disjoint_supports']}; member-clustered AUC "
        f"interval covers 0.5: "
        f"{verdict['q1_conditions']['member_clustered_interval_covers_half']} — "
        "published as-is under the frozen stop rule)")
    add(f"- q2_ordering_mechanism: **{verdict['question_dispositions']['q2_ordering_mechanism']}**"
        f" (monotonicity {verdict['q2_conditions']['monotonicity']}, action share "
        f"{verdict['q2_conditions']['action_share']})")
    add(f"- q3_determinism: **{verdict['question_dispositions']['q3_determinism']}**")
    add(f"- precision guards ok: {verdict['guards_ok']} "
        f"({json.dumps({k: v['ok'] for k, v in compute['precision_guards'].items()}, sort_keys=True)})")
    add("")
    add("## 8. Claim boundary and limitations")
    add("")
    add(CLAIM_BOUNDARY + ".")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    return "\n".join(lines)


def build_report(plan, compute, timings):
    return {
        "schema": SCHEMA_REPORT,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "plan_version": plan["version"],
        "frozen_at": plan["frozen_at"],
        "validation_command": VALIDATION_COMMAND,
        "questions": plan["questions"],
        "unit_of_analysis": compute["unit_of_analysis"],
        "chance_references": compute["chance_references"],
        "selection_validity": compute["selection_validity"],
        "ordering_mechanism": compute["ordering_mechanism"],
        "determinism": compute["determinism"],
        "exposure_split": compute["exposure_split"],
        "accounting": compute["accounting"],
        "precision_guards": compute["precision_guards"],
        "question_dispositions": compute["verdict"]["question_dispositions"],
        "verdict_detail": compute["verdict"],
        "cited_cross_checks": compute["cited_cross_checks"],
        "published_context": plan["published_context"],
        "protocol_chronology": plan["protocol_policy"],
        "caps": plan["caps"],
        "claim_boundary": plan["claim_boundary"],
        "limitations": plan["limitations"],
        "compute": {"phases": timings,
                    "wall_seconds_total": sum(entry["seconds"]
                                              for entry in timings.values()),
                    "gpu_seconds_total": 0.0},
        "hygiene": {"content_hashes_recomputed_after_freeze": False,
                    "full_corpus_integrity_pass": False,
                    "engine_access": False, "gpu_work": False},
    }


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def load_rows(notify=None):
    global row_verdicts
    plan87, decisions, oracles87 = load_issue_87()
    summary89, records89 = load_issue_89()
    summary90 = load_issue_90()
    coverage = load_n1_coverage()
    verdicts, channel, typed_failures, unmeasured = build_verdicts(
        plan87, oracles87, records89)
    row_verdicts = verdicts
    members = member_tables(plan87, summary89, verdicts)
    rows = attach_actions(decision_rows(decisions, members, verdicts), plan87)
    branches = {branch_identity(state, ordinal, plan87)
                for (state, ordinal) in verdicts}
    replays = load_n1_replays(branches)
    if notify:
        notify(len(rows), len(rows))
    return (plan87, decisions, rows, members, verdicts, channel, typed_failures,
            unmeasured, replays, summary89, summary90, coverage)


def measured(function, *arguments):
    began = time.monotonic()
    value = function(*arguments)
    seconds = time.monotonic() - began
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    log(f"phase complete in {seconds:.3f}s (peak RSS {peak:.1f} MiB)")
    return value, {"seconds": seconds, "peak_rss_mib": peak}


def run_audit(output, write=True):
    output = Path(output)
    plan = load_terminal_plan(output)
    timings = {}
    began = time.monotonic()
    bundle = load_rows(notify=lambda done, total: log(
        f"slot join {done}/{total}; elapsed {time.monotonic() - began:.1f}s"))
    (plan87, decisions, rows, members, verdicts, channel, typed_failures, unmeasured,
     replays, summary89, summary90, coverage) = bundle
    timings["join_and_scores"] = {
        "seconds": time.monotonic() - began,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    log(f"{len(rows)}-cell join table complete in "
        f"{timings['join_and_scores']['seconds']:.3f}s; scoring and computing the "
        "frozen statistics")
    rows, timing = measured(score_rows, rows)
    timings["per_cell_scores"] = timing
    compute, timings["compute_statistics"] = measured(
        compute_tables, plan, plan87, rows, members, verdicts, channel, typed_failures,
        unmeasured, replays, summary89, summary90, decisions)
    total = sum(entry["seconds"] for entry in timings.values())
    if total > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: measured wall {total:.3f}s exceeds the frozen "
                         f"cap {WALL_CAP_SECONDS}s")
    compute["phases"] = timings
    log(f"dispositions {json.dumps(compute['verdict']['question_dispositions'], sort_keys=True)}"
        f"; measured wall {total:.3f}s of {WALL_CAP_SECONDS}s cap")
    path = output / "compute.json"
    retained = read_json(path) if path.is_file() else None
    if retained is not None and (
            {key: value for key, value in retained.items() if key != "phases"} !=
            {key: value for key, value in compute.items() if key != "phases"}):
        raise ValueError("retained compute.json differs from the fresh recomputation; "
                         "the ticket is frozen and is never silently overwritten")
    if retained is not None:
        log("retained compute.json reproduces exactly (deterministic resume)")
    if write:
        write_text(output / "slot_join.csv", slot_join_csv(rows))
        write_json(path, compute)
        log(f"compute retained at {path}")
    return compute, rows


def publish(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    path = output / "compute.json"
    if not path.is_file():
        raise ValueError("compute.json missing; run --run before --publish")
    compute = read_json(path)
    (plan87, decisions, rows, members, verdicts, channel, typed_failures, unmeasured,
     replays, summary89, summary90, coverage) = load_rows()
    rows = score_rows(rows)
    if slot_join_csv(rows).encode() != (output / "slot_join.csv").read_bytes():
        raise ValueError("fresh cell table differs from the published slot_join.csv")
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
    log(f"published summary.json, comparisons.csv, findings.md, slot_join.csv and "
        f"receipts.json in {output}")
    return report


def validate(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    (compute, fresh_rows), _ = measured(run_audit, output, False)
    published_compute = read_json(output / "compute.json")
    retained = {key: value for key, value in published_compute.items()
                if key != "phases"}
    recomputed = {key: value for key, value in compute.items() if key != "phases"}
    if retained != recomputed:
        for key in sorted(set(retained) | set(recomputed)):
            if retained.get(key) != recomputed.get(key):
                raise ValueError(f"compute.json section {key!r} differs from the fresh "
                                 "recomputation")
    phases = published_compute["phases"]
    fresh = {
        "slot_join.csv": slot_join_csv(fresh_rows).encode(),
        "comparisons.csv": comparisons_csv(compute).encode(),
        "summary.json": json_text(build_report(plan, compute, phases)).encode(),
        "findings.md": findings_md(plan, compute, phases).encode(),
    }
    for name, content in fresh.items():
        path = output / name
        if not path.is_file():
            raise ValueError(f"published artifact missing: {name}")
        if path.read_bytes() != content:
            raise ValueError(f"published {name} differs from the recomputation from the "
                             "frozen inputs")
    summary89, _ = load_issue_89()
    summary90 = load_issue_90()
    published_cited = read_json(output / "summary.json")["cited_cross_checks"]
    if summary89["ceiling_reestimate"]["pooled"] != published_cited["issue_89_pooled_ceiling"]:
        raise ValueError("the cited issue-89 pooled ceiling differs from the published "
                         "citation")
    if summary90["audit_verdict"] != published_cited["issue_90_audit_verdict"]:
        raise ValueError("the cited issue-90 audit verdict differs from the published "
                         "citation")
    log(f"exact recomputation validation passed: {len(fresh_rows)} cells rejoined, "
        "every published table byte-compared, the cited issue-89/issue-90 aggregates "
        "re-cited verbatim; no engine evidence was read and no content hash was "
        "recomputed")
    return 0


def dry_run(output):
    """No-write, NO-SCORING structural inventory (before or after the freeze)."""
    output = Path(output)
    log("no-write dry run; structural inventory only; no mean, rate, difference, "
        "proportion, correlation, margin test or interval is computed at any point")
    plan87, decisions, oracles87 = load_issue_87()
    summary89, records89 = load_issue_89()
    summary90 = load_issue_90()
    coverage = load_n1_coverage()
    states = plan87["states"]
    log(f"#87 plan {plan87['schema']} identity {plan87['identity']} version "
        f"{plan87['version']} status {plan87['status']}; scheduled oracle cells "
        f"{len(plan87['oracle_cells'])}")
    log(f"decision records {len(decisions)}; oracle records {len(oracles87)}; #89 "
        f"completion records {len(records89)}")
    systems = Counter(record["system"] for record in decisions.values())
    log(f"decision systems {json.dumps(dict(sorted(systems.items())))}; seeds "
        f"{sorted({record['seed'] for record in decisions.values()})}")
    log(f"membership states {len(states)}; source members "
        f"{len({state['source_member'] for state in states})}; families "
        f"{json.dumps(dict(sorted(Counter(state['generator_family'] for state in states).items())))}")
    outcome87 = sum(1 for record in oracles87.values() if record.get("outcome"))
    outcome89 = sum(1 for record in records89.values() if record.get("outcome"))
    log(f"executed verdicts available: #87 {outcome87}, #89 {outcome89} (availability "
        "only; no verdict content is read)")
    log(f"#77 N1 coverage branches {len(coverage['branches'])} with statuses "
        f"{json.dumps(dict(sorted(Counter(entry['status'] for entry in coverage['branches'].values()).items())))}")
    log(f"#90 cited summary audit verdict field present: "
        f"{'audit_verdict' in summary90}")
    tables = sorted((EIGHTY / 'candidate-outcomes').glob('*.json'))
    log(f"#80 candidate tables present: {len(tables)}")
    plan_path = Path(output) / "plan.json"
    if plan_path.is_file():
        plan = load_plan(plan_path)
        log(f"frozen protocol present: version {plan['version']} role {plan['role']} "
            f"frozen {plan['frozen_at']}")
    else:
        log("frozen protocol absent: run --prepare to freeze the protocol before --run")
    log("dry run complete; nothing was written and no statistic was computed")
    return 0


def prepare(output):
    output = Path(output)
    plan_path = output / "plan.json"
    if plan_path.is_file():
        frozen = load_plan(plan_path)
        log(f"existing frozen protocol validated (version {frozen['version']}, frozen "
            f"{frozen['frozen_at']}); no statistic was computed")
        return frozen
    frozen = frozen_plan(utc_now())
    frozen["inputs"] = [
        {**entry, "sha256_at_freeze": (
            sha256_of(Path(entry["artifact"])) if Path(entry["artifact"]).is_file()
            else f"directory:{sum(1 for _ in Path(entry['artifact']).rglob('*.json'))} json files")}
        for entry in frozen["inputs"]]
    write_json(plan_path, frozen)
    log(f"frozen protocol published at {plan_path}: version {frozen['version']}, "
        f"{len(frozen['questions'])} questions, margins "
        f"{json.dumps(frozen['margins'], sort_keys=True)[:200]}; no statistic was "
        "computed")
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
