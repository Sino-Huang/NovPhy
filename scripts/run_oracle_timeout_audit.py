"""Issue-90 ADD-EXP: timeout-outcome correlation audit for the #87 oracle pass.

Binding runner module: scripts/run_oracle_timeout_audit.py
Exact validation command: python -u -m scripts.run_oracle_timeout_audit --validate

Analysis/audit ticket over published artifacts only.  Nothing is re-rendered,
retrained, recaptured, or re-executed; every input is READ-ONLY (#87's terminal
oracle pass, #80's frozen candidate-outcome tables, #77's N1 campaign
coverage/result records and diagnostic headroom block, #73's headroom registry,
#89's published completion report as a cited cross-check).  CPU-only, bounded
memory, zero GPU work.

Audited object: #87's 289 terminal oracle slots (234 executed + 55 typed
failures), stratified exactly as published: 52 connect/barrier/readiness
TIMEOUTS and 3 sealed-frame STABILITY violations.  The stability violations are
a separate typed stratum (different mechanism: the physics is not static at the
sealed decision step 30000) and are never pooled into the timeout contrast.

Frozen-protocol chronology (binding shared rule):
- --prepare writes the complete protocol (covariate derivations, join keys,
  statistics, numeric margins, verdict mapping, bias-direction rules, precision
  guards, caps) into plan.json BEFORE any audit statistic is computed;
- --dry-run is a no-write structural inventory that computes NO statistic of any
  kind, before or after the freeze;
- --run computes the audit statistics once from the frozen inputs;
- --publish renders summary.json / comparisons.csv / findings.md / slot_join.csv;
- --validate recomputes everything from the frozen inputs and byte-compares it
  against the published artifacts.

Verdict vocabulary (frozen exactly): outcome_blind / outcome_correlated /
indeterminate.  Scientific-question dispositions use the standard supported /
not_supported_by_this_experiment / readiness_or_precision_insufficient.  Every
interval is DESCRIPTIVE; no decision language exists outside the frozen numeric
margins.

Protocol hygiene: no content hash is recomputed after the freeze and no
full-corpus integrity pass runs anywhere; inputs are verified at read time by
the schema/identity/plan_identity fields the artifacts already carry (plus the
sha256 identities recorded once at freeze).  The exclusive flock on
/tmp/novphy-addexp-gpu.lock is held for every wall-time-measured phase
(--run/--publish/--validate) and released immediately after.

Claim boundary: descriptive audit of the #87 publication only.  #87's, #80's,
#77's, #73's and #89's published numbers are inputs and are never recomputed,
amended, or reinterpreted; #64/#65 stay sealed; no causal claim about the
timeout mechanism beyond the frozen covariate set; no new data, no rendering,
no engine access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import io
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-90-oracle-timeout-audit-v1"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"

EIGHTY_SEVEN = ROOT / ".local-artifacts/issue-87-closed-loop-oracle-v1"
EIGHTY_SEVEN_IDENTITY = "issue-87-closed-loop-oracle-v1"
EIGHTY = ROOT / ".local-artifacts/issue-80-reactive-diagnostic-v1"
EIGHTY_IDENTITY = "issue-80-reactive-diagnostic-v1"
SEVENTY_SEVEN = ROOT / ".local-artifacts/issue-77-n1-v1"
SEVENTY_SEVEN_IDENTITY = "issue-77-n1-v1"
SEVENTY_SEVEN_DIAGNOSTIC = ROOT / ".local-artifacts/issue-77-n1-diagnostic-v1"
SEVENTY_SEVEN_DIAGNOSTIC_IDENTITY = "issue-77-n1-diagnostic-v1"
SEVENTY_THREE = ROOT / ".local-artifacts/issue-73-headroom-v1"
SEVENTY_THREE_IDENTITY = "issue-73-headroom-plan-v1"
EIGHTY_NINE = ROOT / ".local-artifacts/issue-89-oracle-completion-v1"
EIGHTY_NINE_IDENTITY = "issue-89-oracle-completion-v1"

SCHEMA_PLAN = "issue_90_oracle_timeout_audit_plan_v1"
SCHEMA_COMPUTE = "issue_90_oracle_timeout_audit_compute_v1"
SCHEMA_REPORT = "issue_90_oracle_timeout_audit_report_v1"
IDENTITY = "issue-90-oracle-timeout-audit-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_oracle_timeout_audit --validate"

SCHEDULED_SLOTS = 289
STRATA = ("executed", "timeout", "stability")
REFERENCE_STRATUM = "executed"
TARGET_STRATUM = "timeout"
STABILITY_SIGNATURE = "sealed pre-decision state/RGB advanced during readiness or hold"

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 9001
BOOTSTRAP_UNIT = "state identity (every slot of a resampled state moves together)"
INTERVAL_QUANTILES = (0.025, 0.975)

MARGIN_NORMALIZED_COST = 0.10
MARGIN_INDICATOR_RATE = 0.10
MARGIN_FAMILY_SHARE = 0.15
MARGIN_ORDINAL = 1.00
RAW_COST_MARGIN_FRACTION = 0.25

PRECISION_GUARDS = {
    "min_timeout_slots": 20,
    "min_executed_slots": 100,
    "min_join_fraction": 0.95,
    "min_usable_draws_fraction": 0.90,
    "min_timeout_states": 8,
}

VERDICT_TOKENS = ("outcome_blind", "outcome_correlated", "indeterminate")
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")

WALL_CAP_SECONDS = 360.0
DERIVED_BYTES_BUDGET = 100 * 1024 ** 2
STOP_TOKEN = "readiness_or_precision_insufficient"

COVARIATE_ROLES = {
    "normalized_cost_position": "outcome_linked_lower_is_better_progress",
    "frozen_replay_count_cost": "outcome_linked_lower_is_better_progress",
    "frozen_replay_pig_removed": "outcome_linked_engine_truth_removal_event",
    "is_type010105": "outcome_linked_family_mechanics",
    "state_best_observed_pig_removed": "outcome_linked_engine_truth_removal_event",
    "ordinal": "inventory_position_not_outcome_linked_by_itself",
    "live_at_dispatch": "mechanism_inventory_only",
    "dispatch_burst_size": "mechanism_inventory_only",
    "dispatch_index": "mechanism_inventory_only",
    "minutes_since_first_dispatch": "mechanism_inventory_only",
    "wall_seconds": "mechanism_inventory_only",
    "engine_wall_seconds": "mechanism_inventory_only",
    "peak_cpu_rss_mib": "mechanism_inventory_only",
}

BIAS_DIRECTIONS = {
    ("frozen_replay_pig_removed", "timeout_higher"): (
        "the timeout stratum over-represents slots whose frozen replay removed a pig "
        "(same action, same decision state, same engine seed); under the frozen semantics "
        "those are the more successful slots, so #87's published ceiling is plausibly a "
        "downward-biased lower bound and the ceiling-state conditioning set of #87's Q2 is "
        "biased toward states in which engine-truth success is less likely"),
    ("frozen_replay_pig_removed", "timeout_lower"): (
        "the timeout stratum under-represents slots whose frozen replay removed a pig; the "
        "timed-out slots are then the less successful ones, so the published ceiling is not "
        "detectably depressed by the timeouts and #87's Q2 conditioning set is not "
        "detectably enriched in easier states by them"),
    ("state_best_observed_pig_removed", "timeout_higher"): (
        "the timeout stratum over-represents states carrying at least one frozen-replay "
        "pig-removal candidate (the best-observed pig-removal indicator); the unmeasured "
        "states are therefore disproportionately states with frozen removal evidence, which "
        "biases #87's published ceiling downward and its Q2 conditioning set toward states "
        "without such evidence"),
    ("state_best_observed_pig_removed", "timeout_lower"): (
        "the timeout stratum under-represents states carrying a frozen-replay pig-removal "
        "candidate; the unmeasured states are the ones without frozen removal evidence, so "
        "the published ceiling is not detectably depressed by the timeouts and the Q2 "
        "conditioning set is not detectably enriched in easier states by them"),
    ("normalized_cost_position", "timeout_higher"): (
        "timed-out slots sit higher inside their own state's frozen candidate cost span "
        "(less frozen progress, since the count cost is the task objective and lower is "
        "better); the hidden-success risk of #87's ceiling is correspondingly lower and the "
        "ceiling is a plausible lower bound on the states it covers"),
    ("normalized_cost_position", "timeout_lower"): (
        "timed-out slots sit lower inside their own state's frozen candidate cost span (more "
        "frozen progress); the slots #87 failed to measure are disproportionately the better "
        "ones, so the published ceiling is plausibly understated and the Q2 conditioning set "
        "is biased toward less successful states"),
    ("frozen_replay_count_cost", "timeout_higher"): (
        "the raw frozen end-of-window count cost is higher for timed-out slots (less frozen "
        "progress); the hidden-success risk of #87's ceiling is correspondingly lower and the "
        "ceiling is a plausible lower bound on the states it covers"),
    ("frozen_replay_count_cost", "timeout_lower"): (
        "the raw frozen end-of-window count cost is lower for timed-out slots (more frozen "
        "progress); the unmeasured slots are disproportionately the better ones, so the "
        "published ceiling is plausibly understated"),
    ("is_type010105", "timeout_higher"): (
        "the timeout stratum over-represents type010105 states, whose published ceiling "
        "prevalence (3 of 12, 0.25) is lower than type010103's (4 of 12, 0.363636); under "
        "the frozen carry-over assumption the measured pooled ceiling is then plausibly "
        "biased upward relative to the full 24-state inventory"),
    ("is_type010105", "timeout_lower"): (
        "the timeout stratum over-represents type010103 states, whose published ceiling "
        "prevalence (4 of 12, 0.363636) is higher than type010105's (3 of 12, 0.25); under "
        "the frozen carry-over assumption the measured pooled ceiling is then plausibly "
        "biased downward relative to the full 24-state inventory"),
}

FROZEN_SEMANTICS = (
    "frozen replay evidence: for every audited slot the #77 N1 campaign replayed the same "
    "action from the same source-member decision state (native fixed step 30000, engine seed "
    "764100001) BEFORE #87 executed anything, so the frozen pig-removal event, the frozen "
    "terminal kind and the frozen count cost are engine-truth pre-execution properties of the "
    "slot; the frozen count cost is the task objective (1000 * pig presence + block presence) "
    "at the frozen replay's observed window endpoint, where lower means more progress")

PLACEHOLDER_MARKERS = ("TBD", "placeholder", "to be frozen", "XXX", "FIXME")

CLAIM_BOUNDARY = (
    "descriptive audit of the published #87 oracle pass only; no re-execution, no rendering, "
    "no new data, zero GPU work; cannot reopen #87's or #89's published dispositions; no "
    "causal claim about the timeout mechanism beyond the frozen covariate set; intervals are "
    "DESCRIPTIVE")

LIMITATIONS = (
    "development/exposed N1 lineages only; not the sealed #64/#65 benchmark",
    "the frozen covariates are engine truth for the frozen #77 replay, not for the #87 "
    "execution that timed out; per-slot execution variance between the two runs is unobserved",
    "states sharing a source member share one physical initial state and candidate table, so "
    "the 24 state identities carry fewer independent initial states; the state-clustered "
    "bootstrap is descriptive only",
    "the cold-start reading is inferred from published dispatch timestamps and per-cell wall "
    "seconds at one-second resolution; engine boot timestamps are not published, so "
    "live_at_dispatch is a proxy",
    "unobserved-confounder residual risk: any covariate not in the frozen list (per-worker "
    "engine identity, host load outside the published timestamps, port-level collisions) is "
    "unmeasured, and a timeout mechanism that acts through such a covariate would not be "
    "detected here",
    "no content hash is recomputed after the freeze and no full-corpus integrity pass runs "
    "anywhere; inputs are verified at read time by their carried "
    "schema/identity/plan_identity fields",
)


def log(message):
    print(f"[issue-90-timeout-audit] {message}", flush=True)


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
         "role": "slot inventory, membership, oracle seed, published protocol"},
        {"name": "issue_87_ledger", "artifact": str(EIGHTY_SEVEN / "ledger.json"),
         "schema": "issue_87_closed_loop_oracle_ledger_v1",
         "identity_fields": ["identity", "plan_identity", "status"],
         "role": "per-cell execution status and stop"},
        {"name": "issue_87_records", "artifact": str(EIGHTY_SEVEN / "records"),
         "schema": "issue_87_closed_loop_oracle_record_v1",
         "identity_fields": ["plan_identity", "plan_version", "schema"],
         "role": "the 289 terminal slots: failure kind, wall seconds, engine channel"},
        {"name": "issue_87_markers", "artifact": str(EIGHTY_SEVEN / "markers"),
         "schema": "issue_87_oracle_dispatch_marker_v1",
         "identity_fields": ["identity", "phase"],
         "role": "per-slot dispatch timestamps (mechanism inventory)"},
        {"name": "issue_87_receipts", "artifact": str(EIGHTY_SEVEN / "receipts"),
         "schema": "issue_87_cell_receipt_v1",
         "identity_fields": ["identity", "phase"],
         "role": "per-slot peak RSS (mechanism inventory)"},
        {"name": "issue_80_candidate_outcomes", "artifact": str(EIGHTY / "candidate-outcomes"),
         "schema": "issue_80_candidate_outcomes_v1",
         "identity_fields": ["member_identity", "plan_identity", "schema"],
         "role": "frozen candidate table: count cost and action per source member"},
        {"name": "issue_77_n1_coverage", "artifact": str(SEVENTY_SEVEN / "coverage.json"),
         "schema": "issue_77_n1_coverage_v1",
         "identity_fields": ["plan_identity", "schema"],
         "role": "frozen N1 branch admissibility, candidate ordinal, terminal kind"},
        {"name": "issue_77_n1_results", "artifact": str(SEVENTY_SEVEN / "results"),
         "schema": "issue_76_bounded_transfer_capture_v1",
         "identity_fields": ["member_identity", "base_cluster", "schema"],
         "role": "frozen N1 replay summary: engine-truth pig-removal event count"},
        {"name": "issue_77_n1_diagnostic_summary",
         "artifact": str(SEVENTY_SEVEN_DIAGNOSTIC / "summary.json"),
         "schema": "issue_77_n1_diagnostic_report_v1",
         "identity_fields": ["identity", "plan_identity", "schema"],
         "role": "published N1 headroom block (precedent for the best-observed indicator)"},
        {"name": "issue_73_headroom_plan", "artifact": str(SEVENTY_THREE / "plan.json"),
         "schema": "issue_73_readonly_diagnostic_v1",
         "identity_fields": ["identity", "schema"],
         "role": "headroom registry identities (join attempt; typed unjoinable)"},
        {"name": "issue_89_completion_summary", "artifact": str(EIGHTY_NINE / "summary.json"),
         "schema": "issue_89_oracle_completion_report_v1",
         "identity_fields": ["identity", "plan_identity", "schema"],
         "role": "cited published cross-check only; never recomputed"},
    ]


def verify_source_bindings(plan):
    declared = [(entry["name"], entry["artifact"]) for entry in declared_sources()]
    frozen = [(entry["name"], entry["artifact"]) for entry in plan["inputs"]]
    if frozen != declared:
        raise ValueError("frozen plan input bindings differ from the declared sources")


# ---------------------------------------------------------------------------
# frozen plan payload
# ---------------------------------------------------------------------------

def raw_cost_margin(states, tables):
    """Frozen-formula raw-cost margin: a quarter of the median within-state span.

    Derived from the frozen #80 candidate table alone (no stratum, no contrast is
    touched), materialized into plan.json at freeze time and re-derived at
    validate time from the same frozen table.
    """
    spans = []
    for state in states:
        ordinals = {item["ordinal"] for item in state["inventory"]}
        costs = [row["realized_count_cost"]
                 for row in tables[state["source_member"]]["candidates"]
                 if row["ordinal"] in ordinals]
        spans.append(max(costs) - min(costs))
    return RAW_COST_MARGIN_FRACTION * float(np.median(sorted(spans)))


def frozen_plan(raw_margin, frozen_at):
    return {
        "schema": SCHEMA_PLAN,
        "identity": IDENTITY,
        "version": 1,
        "role": "terminal",
        "frozen_at": frozen_at,
        "frozen_before_scoring_run": True,
        "changelog": [{
            "version": 1,
            "change": "initial and terminal freeze: covariates, join keys, statistics, "
                      "numeric margins, verdict mapping, bias-direction rules, precision "
                      "guards and caps are published before any audit statistic exists",
            "why": "standard freeze rule of the shared ADD-EXP protocol; the dry run is a "
                   "structural inventory only and computes no statistic of any kind",
            "evidence": "no audit statistic has been computed at freeze time; the only "
                        "quantity derived here is the raw-cost margin formula evaluated on "
                        "the frozen candidate table, which involves no stratum and no "
                        "contrast",
        }],
        "protocol_policy": (
            "single terminal protocol version: the full protocol is frozen with actual "
            "numeric margins before the scoring run and executed exactly once; the dry run "
            "computes no statistic; --validate recomputes every published table"),
        "questions": {
            "q1_covariate_shift": (
                "do the 52 timed-out slots differ from the 234 executed slots on the frozen "
                "pre-execution covariates (frozen replay count cost distribution, state type, "
                "ordinal position, frozen-replay and best-observed pig-removal indicators), "
                "with descriptive state-clustered intervals over states"),
            "q2_audit_verdict": (
                "under the frozen mapping with pre-declared margins, is the timeout mechanism "
                "outcome_blind, outcome_correlated, or indeterminate, and what does the "
                "observed correlation, if any, imply for the direction of #87's ceiling bias "
                "(typed, descriptive)"),
        },
        "universe": {
            "artifact": str(EIGHTY_SEVEN),
            "scheduled_slots": SCHEDULED_SLOTS,
            "strata": list(STRATA),
            "stratum_rules": {
                "executed": "the issue-87 oracle record failure_kind is null",
                "timeout": "the issue-87 oracle record failure_kind is execution_failure and "
                           "the failure text starts with TimeoutError",
                "stability": "the issue-87 oracle record failure_kind is execution_failure and "
                             "the failure text equals 'ValueError: " + STABILITY_SIGNATURE + "'",
            },
            "stability_is_separate": (
                "the 3 sealed-frame stability violations are a separate typed stratum "
                "(different mechanism: the physics is not static at the sealed decision step "
                "30000) and are NEVER pooled into the timeout contrast; they are reported with "
                "their frozen covariates and typed as interval-ineligible"),
            "reference_stratum": REFERENCE_STRATUM,
            "target_stratum": TARGET_STRATUM,
        },
        "join_keys": {
            "slot_identity": "issue-87 oracle cell identity = state identity x candidate ordinal",
            "state_to_source_member": "issue-87 plan.json states[].source_member",
            "frozen_candidate_table": "issue-80 candidate-outcomes/<source_member>.json row with "
                                      "the same ordinal (branch identity, action and count cost "
                                      "must agree with the issue-87 cell)",
            "frozen_replay": "issue-77 N1 coverage branch keyed by the slot's branch identity, "
                             "plus results/<branch identity>.json",
            "headroom": "issue-73 headroom registry keyed by its own state identity (typed "
                        "unjoinable: the #73 registry holds issue-68 production-v2 calibration "
                        "identities, disjoint from the issue-77 N1 membership identities, so the "
                        "best-observed pig-removal indicator is computed from frozen #77 "
                        "evidence instead, as the frozen substitution records)",
            "dispatch": "issue-87 markers/<slot identity>.json dispatched_at_utc and "
                        "receipts/<slot identity>.json peak_cpu_rss_mib",
        },
        "covariates": {
            "normalized_cost_position": {
                "source": "issue-80 candidate-outcomes/<source_member>.json: "
                          "(cost - min)/(max - min) over the state's admissible inventory, 0 "
                          "when the span is zero, following the frozen #72 normalized-regret "
                          "convention",
                "role": COVARIATE_ROLES["normalized_cost_position"],
                "unit": "fraction of the state's own frozen candidate cost span"},
            "frozen_replay_count_cost": {
                "source": "issue-80 candidate-outcomes/<source_member>.json "
                          "realized_count_cost of the row with this ordinal",
                "role": COVARIATE_ROLES["frozen_replay_count_cost"],
                "unit": "task-objective units (1000 * pig presence + block presence) at the "
                        "frozen replay's observed window endpoint"},
            "frozen_replay_pig_removed": {
                "source": "issue-77 N1 results/<branch identity>.json "
                          "segments[0].summary.event_counts.pig_removed greater than 0",
                "role": COVARIATE_ROLES["frozen_replay_pig_removed"],
                "unit": "boolean engine-truth removal event in the frozen replay"},
            "is_type010105": {
                "source": "issue-87 plan.json states[].generator_family",
                "role": COVARIATE_ROLES["is_type010105"],
                "unit": "boolean family indicator (type010105 versus type010103)"},
            "state_best_observed_pig_removed": {
                "source": "true when any admissible slot of the same state carries "
                          "frozen_replay_pig_removed (the #73-style best-observed pig-removal "
                          "indicator, computed from frozen #77 evidence)",
                "role": COVARIATE_ROLES["state_best_observed_pig_removed"],
                "unit": "boolean state-level indicator, constant within a state"},
            "ordinal": {
                "source": "issue-87 plan.json oracle_cells[].ordinal",
                "role": COVARIATE_ROLES["ordinal"],
                "unit": "0-based candidate ordinal in the state's admissible inventory"},
            "live_at_dispatch": {
                "source": "derived: number of oracle slots whose [dispatched_at_utc, "
                          "dispatched_at_utc + wall_seconds] interval covers this slot's "
                          "dispatched_at_utc, over the 4 concurrent workers",
                "role": COVARIATE_ROLES["live_at_dispatch"], "unit": "worker count 1 to 4"},
            "dispatch_burst_size": {
                "source": "derived: number of oracle slots dispatched within 5 seconds of this "
                          "slot's dispatch timestamp, including itself",
                "role": COVARIATE_ROLES["dispatch_burst_size"], "unit": "slot count"},
            "dispatch_index": {
                "source": "derived: 0-based rank of dispatched_at_utc over the 289 slots, ties "
                          "broken by slot identity",
                "role": COVARIATE_ROLES["dispatch_index"], "unit": "dispatch rank"},
            "minutes_since_first_dispatch": {
                "source": "derived: (this dispatch timestamp - the first dispatch timestamp)/60",
                "role": COVARIATE_ROLES["minutes_since_first_dispatch"], "unit": "minutes"},
            "wall_seconds": {
                "source": "issue-87 oracle records[].wall_seconds",
                "role": COVARIATE_ROLES["wall_seconds"],
                "unit": "seconds; for a timeout slot this is the timeout bound, so the field is "
                        "mechanism inventory and never part of a covariate shift claim"},
            "engine_wall_seconds": {
                "source": "issue-87 oracle records[].execution.engine_wall_seconds (executed "
                          "slots only)",
                "role": COVARIATE_ROLES["engine_wall_seconds"], "unit": "seconds"},
            "peak_cpu_rss_mib": {
                "source": "issue-87 receipts/<slot identity>.json peak_cpu_rss_mib",
                "role": COVARIATE_ROLES["peak_cpu_rss_mib"], "unit": "MiB"},
        },
        "frozen_semantics": FROZEN_SEMANTICS,
        "statistics": {
            "contrast": "value(target stratum mean) minus value(reference stratum mean), where "
                        "the mean runs over the slots of the stratum and boolean covariates are "
                        "averaged as rates",
            "interval": f"state-clustered percentile bootstrap, {BOOTSTRAP_DRAWS} draws, PCG64 "
                        f"seed {BOOTSTRAP_SEED}, unit {BOOTSTRAP_UNIT}, quantiles "
                        f"{list(INTERVAL_QUANTILES)}, labelled DESCRIPTIVE; draws in which "
                        "either side is empty are dropped and counted",
            "raw_cost_margin_formula": f"{RAW_COST_MARGIN_FRACTION} times the median over the 24 "
                                       "states of (max minus min) of the frozen candidate count "
                                       "cost over the state's admissible inventory; materialized "
                                       "at freeze from the frozen #80 table alone",
            "raw_cost_margin": raw_margin,
            "association_context": (
                "within the executed stratum only: the published engine-truth success rate of "
                "slots whose frozen replay removed a pig minus the rate of the rest, with the "
                "same clustered interval; context for the semantics of the indicators and NEVER "
                "a verdict input"),
            "mechanism_inventory": (
                "per stratum: slot, state and source-member counts, dispatch index and minutes "
                "since first dispatch, live_at_dispatch distribution and all-workers-live share, "
                "dispatch burst size, per-cell wall seconds, engine wall seconds (executed "
                "slots), peak RSS, frozen terminal kind histogram, published success count; "
                "DESCRIPTIVE only and never a verdict input"),
        },
        "contrasts": [
            {"id": "d1_normalized_cost_position", "covariate": "normalized_cost_position",
             "kind": "mean", "margin": MARGIN_NORMALIZED_COST, "verdict_relevant": True},
            {"id": "d2_frozen_replay_count_cost", "covariate": "frozen_replay_count_cost",
             "kind": "mean", "margin": raw_margin, "verdict_relevant": True},
            {"id": "d3_frozen_replay_pig_removed", "covariate": "frozen_replay_pig_removed",
             "kind": "rate", "margin": MARGIN_INDICATOR_RATE, "verdict_relevant": True},
            {"id": "d4_type010105_share", "covariate": "is_type010105", "kind": "rate",
             "margin": MARGIN_FAMILY_SHARE, "verdict_relevant": True},
            {"id": "d5_state_best_observed_pig_removed",
             "covariate": "state_best_observed_pig_removed", "kind": "rate",
             "margin": MARGIN_INDICATOR_RATE, "verdict_relevant": True},
            {"id": "d6_ordinal_position", "covariate": "ordinal", "kind": "mean",
             "margin": MARGIN_ORDINAL, "verdict_relevant": False},
        ],
        "verdict_mapping": {
            "shift_rule": "a verdict-relevant contrast shows a shift iff the absolute "
                          "difference is at least its margin AND its descriptive interval "
                          "excludes 0",
            "borderline_rule": "a verdict-relevant contrast is borderline iff the absolute "
                               "difference is at least its margin AND its interval includes 0",
            "outcome_correlated": "no precision guard tripped AND at least one "
                                  "verdict-relevant contrast shows a shift",
            "outcome_blind": "no precision guard tripped AND no verdict-relevant contrast "
                             "shows a shift AND no verdict-relevant contrast is borderline",
            "indeterminate": "a precision guard tripped OR a verdict-relevant contrast is "
                             "borderline",
            "dispositions": {
                "q1_covariate_shift": {
                    "supported": "the audit verdict is outcome_correlated",
                    "not_supported_by_this_experiment": "the audit verdict is outcome_blind",
                    "readiness_or_precision_insufficient": "the audit verdict is indeterminate",
                },
                "q2_audit_verdict": "reported directly with the frozen audit verdict token",
            },
        },
        "precision_guards": dict(PRECISION_GUARDS, rule=(
            "each guard is evaluated on the frozen slot table; any failed guard maps the audit "
            "verdict to indeterminate before the shift rules are read")),
        "bias_direction_rules": {f"{key[0]}|{key[1]}": value
                                 for key, value in BIAS_DIRECTIONS.items()},
        "bias_direction_conventions": {
            "direction_tokens": ["timeout_higher", "timeout_lower", "no_shift"],
            "no_shift": "no frozen bias direction is stated for a contrast with no shift",
            "ordinal": "the ordinal contrast has no bias-direction rule: an inventory position "
                       "is not outcome-linked by itself",
            "label": "DESCRIPTIVE",
        },
        "published_context": {
            "issue_87_pooled_ceiling": {"ceiling_states": 7, "measured_states": 23, "states": 24,
                                        "prevalence": 0.30434782608695654,
                                        "unmeasured_states": ["issue-77-n1-005-a07"]},
            "issue_87_by_family": {
                "type010103": {"ceiling_states": 4, "states": 12,
                               "prevalence": 0.36363636363636365},
                "type010105": {"ceiling_states": 3, "states": 12, "prevalence": 0.25}},
            "rule": "published numbers cited verbatim as context; never recomputed or amended",
        },
        "cited_cross_check": {
            "source": str(EIGHTY_NINE),
            "rule": "the issue-89 completion pass re-executed the 55 typed-failure slots of #87 "
                    "under a separately frozen infrastructure protocol; its published per-slot "
                    "verdicts (record fields engine_channel.pig_removed and "
                    "outcome.first_shot_success, with the originating #87 failure text in "
                    "issue87_typed_failure) are cited here verbatim and tallied by #87 failure "
                    "stratum; the tally is never recomputed from engine evidence and is never a "
                    "verdict input",
            "attribution_limit": "a different execution infrastructure cannot be separated from "
                                 "slot selection by this row; it is a directional cross-check "
                                 "only",
            "published_aggregates": "issue-89 summary.json completion_pass.inventory and "
                                    "ceiling_reestimate.pooled are cited verbatim",
        },
        "caps": {
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "wall_scope": "cumulative measured wall of the audit phases (run plus publish plus "
                          "validate); the dry run and the freeze publish no wall time",
            "derived_bytes_budget": DERIVED_BYTES_BUDGET,
            "gpu_cap_seconds": 0.0,
            "gpu_scope": "no GPU work of any kind in this ticket; the shared flock is still held "
                         "for every wall-time-measured phase",
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
        raise ValueError(f"plan {path} is not the issue-90 terminal protocol")
    blob = json_text(plan)
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    required = ("covariates", "contrasts", "verdict_mapping", "precision_guards",
                "bias_direction_rules", "bias_direction_conventions", "statistics", "universe",
                "join_keys", "caps", "limitations", "inputs", "validation_command",
                "published_context", "cited_cross_check")
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
    ledger = read_json(EIGHTY_SEVEN / "ledger.json")
    if (ledger["identity"] != EIGHTY_SEVEN_IDENTITY
            or ledger["plan_identity"] != EIGHTY_SEVEN_IDENTITY):
        raise ValueError("#87 ledger binding differs")
    records = {}
    for path in sorted((EIGHTY_SEVEN / "records").glob("oracle--*.json")):
        record = read_json(path)
        if (record["schema"] != "issue_87_closed_loop_oracle_record_v1"
                or record["plan_identity"] != EIGHTY_SEVEN_IDENTITY
                or record["plan_version"] != plan["version"]):
            raise ValueError(f"#87 record binding differs: {path.name}")
        records[record["cell"]["identity"]] = record
    markers = {}
    for path in sorted((EIGHTY_SEVEN / "markers").glob("oracle--*.json")):
        marker = read_json(path)
        if marker["phase"] != "oracle" or marker["identity"] != path.stem:
            raise ValueError(f"#87 dispatch marker binding differs: {path.name}")
        markers[marker["identity"]] = marker["dispatched_at_utc"]
    receipts = {}
    for path in sorted((EIGHTY_SEVEN / "receipts").glob("oracle--*.json")):
        receipt = read_json(path)
        if receipt["phase"] != "oracle" or receipt["identity"] != path.stem:
            raise ValueError(f"#87 receipt binding differs: {path.name}")
        receipts[receipt["identity"]] = receipt
    return plan, ledger, records, markers, receipts


def load_candidate_tables(states):
    tables = {}
    for state in states:
        member = state["source_member"]
        if member in tables:
            continue
        table = read_json(EIGHTY / "candidate-outcomes" / f"{member}.json")
        if (table["schema"] != "issue_80_candidate_outcomes_v1"
                or table["member_identity"] != member
                or table["plan_identity"] != EIGHTY_IDENTITY):
            raise ValueError(f"#80 candidate table binding differs for {member}")
        tables[member] = table
    return tables


def load_n1_coverage():
    coverage = read_json(SEVENTY_SEVEN / "coverage.json")
    if (coverage["schema"] != "issue_77_n1_coverage_v1"
            or coverage["plan_identity"] != SEVENTY_SEVEN_IDENTITY):
        raise ValueError("#77 N1 coverage binding differs")
    return coverage["branches"]


def load_n1_replays(branches):
    """Frozen N1 replay summaries for the branch identities that matter."""
    replays = {}
    for branch in sorted(branches):
        path = SEVENTY_SEVEN / "results" / f"{branch}.json"
        if not path.is_file():
            replays[branch] = None
            continue
        record = read_json(path)
        if (record["schema"] != "issue_76_bounded_transfer_capture_v1"
                or record["member_identity"] != branch
                or record["base_cluster"] != branch.rsplit("-a", 1)[0]):
            raise ValueError(f"#77 N1 replay binding differs for {branch}")
        segments = record.get("segments") or []
        if len(segments) != 1:
            raise ValueError(f"#77 N1 replay segment count differs for {branch}")
        replays[branch] = segments[0]["summary"]
    return replays


def load_n1_diagnostic():
    summary = read_json(SEVENTY_SEVEN_DIAGNOSTIC / "summary.json")
    if (summary["schema"] != "issue_77_n1_diagnostic_report_v1"
            or summary["identity"] != "issue-77-n1-diagnostic-report-v1"
            or summary["plan_identity"] != SEVENTY_SEVEN_DIAGNOSTIC_IDENTITY):
        raise ValueError("#77 diagnostic summary binding differs")
    return summary


def load_seventy_three():
    plan = read_json(SEVENTY_THREE / "plan.json")
    if (plan["schema"] != "issue_73_readonly_diagnostic_v1"
            or plan["identity"] != SEVENTY_THREE_IDENTITY):
        raise ValueError("#73 headroom plan binding differs")
    return plan


def load_eighty_nine():
    summary = read_json(EIGHTY_NINE / "summary.json")
    if (summary["schema"] != "issue_89_oracle_completion_report_v1"
            or summary["identity"] != EIGHTY_NINE_IDENTITY
            or summary["plan_identity"] != EIGHTY_NINE_IDENTITY):
        raise ValueError("#89 completion summary binding differs")
    records = {}
    for path in sorted((EIGHTY_NINE / "records").glob("oracle--*.json")):
        record = read_json(path)
        if record["plan_identity"] != EIGHTY_NINE_IDENTITY:
            raise ValueError(f"#89 record binding differs: {path.name}")
        records[record["cell"]["identity"]] = record
    return summary, records


# ---------------------------------------------------------------------------
# slot table (the join)
# ---------------------------------------------------------------------------

def stratum_of(record):
    kind = record.get("failure_kind")
    text = record.get("failure") or ""
    if kind is None:
        return "executed"
    if kind == "execution_failure" and text.startswith("TimeoutError"):
        return "timeout"
    if kind == "execution_failure" and text == f"ValueError: {STABILITY_SIGNATURE}":
        return "stability"
    raise ValueError(f"unstratifiable issue-87 failure kind {kind!r} with text {text!r}")


def parse_stamp(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def frozen_evidence(states, replays):
    """Frozen pig-removal evidence per candidate slot and per state."""
    per_slot = {}
    best_observed = {}
    for state in states:
        flags = []
        for item in state["inventory"]:
            branch = item["branch_identity"]
            summary = replays.get(branch)
            events = (summary or {}).get("event_counts") or {}
            flag = bool(events.get("pig_removed", 0) > 0) if summary else None
            per_slot[branch] = flag
            flags.append(flag)
        best_observed[state["identity"]] = (
            None if any(flag is None for flag in flags) else any(flags))
    return per_slot, best_observed


def build_slot_table(plan87, ledger, records, markers, receipts, states, tables, coverage,
                     replays, headroom_ids, notify=None):
    """One row per terminal #87 oracle slot with typed per-source join status."""
    cells = {}
    for cell in plan87["oracle_cells"]:
        if cell["identity"] in cells:
            raise ValueError(f"duplicate oracle cell {cell['identity']}")
        cells[cell["identity"]] = cell
    state_by_identity = {state["identity"]: state for state in states}
    if len(records) != SCHEDULED_SLOTS or set(records) != set(cells):
        raise ValueError(f"#87 terminal record set differs from the scheduled inventory "
                         f"({len(records)} records, {len(cells)} scheduled)")
    for identity in cells:
        if identity not in ledger["cells"]:
            raise ValueError(f"slot {identity} is absent from the #87 ledger")

    per_state_cost = {}
    for state in states:
        rows = {row["ordinal"]: row for row in tables[state["source_member"]]["candidates"]}
        inventory = {item["ordinal"]: item for item in state["inventory"]}
        costs = [rows[ordinal]["realized_count_cost"] for ordinal in sorted(inventory)]
        low, high = min(costs), max(costs)
        per_state_cost[state["identity"]] = (rows, inventory, low, high, high > low)
    per_slot_removal, best_observed = frozen_evidence(states, replays)

    stamps = {identity: parse_stamp(markers[identity]) for identity in markers}
    if set(stamps) != set(cells):
        raise ValueError("dispatch marker set differs from the slot inventory")
    ordered = sorted(stamps, key=lambda identity: (stamps[identity], identity))
    index_of = {identity: position for position, identity in enumerate(ordered)}
    first_stamp = min(stamps.values())
    finishes = {identity: stamps[identity] + float(records[identity]["wall_seconds"])
                for identity in cells}

    rows = []
    for position, identity in enumerate(sorted(cells)):
        cell = cells[identity]
        record = records[identity]
        state = state_by_identity[cell["state"]]
        member_rows, inventory, low, high, informative = per_state_cost[cell["state"]]
        frozen_row = member_rows.get(cell["ordinal"])
        entry = coverage.get(cell["branch_identity"])
        replay = replays.get(cell["branch_identity"])
        burst = sum(1 for other in cells if abs(stamps[other] - stamps[identity]) <= 5.0)
        live = sum(1 for other in cells
                   if stamps[other] <= stamps[identity] < finishes[other])
        status = {
            "issue_80_candidate_table": "joined",
            "issue_77_n1_coverage": "joined",
            "issue_77_n1_replay": "joined",
            "issue_73_headroom_registry": "unjoinable:registry_disjoint_from_issue_77_membership",
            "issue_87_dispatch_marker": "joined",
            "issue_87_receipt": "joined" if identity in receipts else "missing:no_receipt",
        }
        if frozen_row is None:
            status["issue_80_candidate_table"] = "unjoinable:ordinal_absent_from_frozen_table"
        elif (frozen_row["branch_identity"] != cell["branch_identity"]
                or frozen_row["action"] != cell["action"]
                or abs(frozen_row["realized_count_cost"]
                       - cell["frozen_replay_count_cost"]) > 1e-9):
            status["issue_80_candidate_table"] = "unjoinable:frozen_row_disagrees_with_cell"
        if entry is None:
            status["issue_77_n1_coverage"] = "unjoinable:branch_absent_from_n1_coverage"
        elif entry["status"] != "admissible" or entry["candidate_ordinal"] != cell["ordinal"]:
            status["issue_77_n1_coverage"] = "unjoinable:branch_not_admissible_or_ordinal_differs"
        if replay is None:
            status["issue_77_n1_replay"] = "unjoinable:no_frozen_replay_record"
        if cell["state"] in headroom_ids:
            status["issue_73_headroom_registry"] = "joined"
        cost = None if frozen_row is None else frozen_row["realized_count_cost"]
        execution = record.get("execution") or {}
        channel = record.get("engine_channel") or {}
        rows.append({
            "slot_identity": identity,
            "state_identity": cell["state"],
            "source_member": state["source_member"],
            "family": state["generator_family"],
            "study_role": state["study_role"],
            "ordinal": cell["ordinal"],
            "branch_identity": cell["branch_identity"],
            "stratum": stratum_of(record),
            "issue_87_failure": record.get("failure"),
            "issue_87_wall_seconds": float(record["wall_seconds"]),
            "issue_87_engine_wall_seconds": (
                None if execution.get("engine_wall_seconds") is None
                else float(execution["engine_wall_seconds"])),
            "published_outcome_first_shot_success": (
                None if record.get("outcome") is None
                else bool(record["outcome"]["first_shot_success"])),
            "published_engine_channel_pig_removed": (
                None if not channel else bool(channel["pig_removed"])),
            "frozen_replay_count_cost": cost,
            "normalized_cost_position": (
                None if cost is None
                else (0.0 if not informative else (cost - low) / (high - low))),
            "frozen_cost_span_informative": (None if cost is None
                                             else bool(informative)),
            "frozen_replay_terminal_kind": (None if entry is None
                                            else entry["stop_kind"]),
            "frozen_replay_pig_removed": per_slot_removal.get(cell["branch_identity"]),
            "state_best_observed_pig_removed": best_observed.get(cell["state"]),
            "is_type010105": state["generator_family"] == "type010105",
            "dispatched_at_utc": markers[identity],
            "dispatch_index": index_of[identity],
            "minutes_since_first_dispatch": (stamps[identity] - first_stamp) / 60.0,
            "live_at_dispatch": live,
            "dispatch_burst_size": burst,
            "peak_cpu_rss_mib": (None if identity not in receipts
                                 else receipts[identity].get("peak_cpu_rss_mib")),
            "join_status": status,
            "join_complete": all(value == "joined" for key, value in status.items()
                                 if key != "issue_73_headroom_registry"),
        })
        if notify is not None and (position + 1) % 50 == 0:
            notify(position + 1, len(cells))
    if len(rows) != SCHEDULED_SLOTS:
        raise ValueError(f"slot table has {len(rows)} rows, expected {SCHEDULED_SLOTS}")
    return rows


def slot_inventory_joins(plan87, coverage, tables, replays, markers):
    """Structural join coverage counts (no covariate value is read)."""
    counts = Counter()
    for cell in plan87["oracle_cells"]:
        entry = coverage.get(cell["branch_identity"])
        counts["issue_77_n1_coverage_joined"] += int(entry is not None)
        counts["issue_77_n1_coverage_admissible"] += int(
            entry is not None and entry["status"] == "admissible"
            and entry["candidate_ordinal"] == cell["ordinal"])
        counts["issue_77_n1_replay_joined"] += int(replays.get(cell["branch_identity"]) is not None)
        counts["issue_87_dispatch_marker_joined"] += int(cell["identity"] in markers)
        counts["scheduled"] += 1
    for state in plan87["states"]:
        rows = {row["ordinal"] for row in tables[state["source_member"]]["candidates"]}
        for item in state["inventory"]:
            counts["issue_80_candidate_table_joined"] += int(item["ordinal"] in rows)
    counts["issue_80_candidate_table_rows_examined"] = sum(
        len(state["inventory"]) for state in plan87["states"])
    return counts


# ---------------------------------------------------------------------------
# frozen statistics
# ---------------------------------------------------------------------------

def clustered_contrast(rows, value_of, target_of, draws, seed):
    """State-clustered percentile bootstrap of mean(target) - mean(reference)."""
    states = sorted({row["state_identity"] for row in rows})
    index = {state: position for position, state in enumerate(states)}
    target_sum = np.zeros(len(states))
    target_count = np.zeros(len(states))
    reference_sum = np.zeros(len(states))
    reference_count = np.zeros(len(states))
    for row in rows:
        value = value_of(row)
        if value is None:
            continue
        side = target_of(row)
        if side is None:
            continue
        position = index[row["state_identity"]]
        if side:
            target_sum[position] += float(value)
            target_count[position] += 1.0
        else:
            reference_sum[position] += float(value)
            reference_count[position] += 1.0
    if target_count.sum() == 0 or reference_count.sum() == 0:
        raise ValueError("clustered contrast requires both sides to be non-empty")
    rng = np.random.Generator(np.random.PCG64(seed))
    picks = rng.integers(0, len(states), size=(draws, len(states)))
    drawn_target = target_count[picks].sum(axis=1)
    drawn_reference = reference_count[picks].sum(axis=1)
    usable = (drawn_target > 0) & (drawn_reference > 0)
    if not usable.any():
        raise ValueError("every bootstrap draw dropped a side; the contrast is unavailable")
    differences = (target_sum[picks].sum(axis=1) / np.where(usable, drawn_target, 1.0)
                   - reference_sum[picks].sum(axis=1) / np.where(usable, drawn_reference, 1.0))
    usable_differences = differences[usable]
    return {
        "value_target": float(target_sum.sum() / target_count.sum()),
        "value_reference": float(reference_sum.sum() / reference_count.sum()),
        "n_target": int(target_count.sum()),
        "n_reference": int(reference_count.sum()),
        "states_target": int((target_count > 0).sum()),
        "states_reference": int((reference_count > 0).sum()),
        "usable_draws": int(usable.sum()),
        "dropped_draws": int((~usable).sum()),
        "ci_low": float(np.quantile(usable_differences, INTERVAL_QUANTILES[0])),
        "ci_high": float(np.quantile(usable_differences, INTERVAL_QUANTILES[1])),
    }


def stratum_side(row):
    if row["stratum"] == TARGET_STRATUM:
        return True
    if row["stratum"] == REFERENCE_STRATUM:
        return False
    return None


def contrast_rows(plan, rows):
    results = []
    for spec in plan["contrasts"]:
        drew = clustered_contrast(rows, lambda row, key=spec["covariate"]: row[key],
                                  stratum_side, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
        difference = drew["value_target"] - drew["value_reference"]
        excludes_zero = drew["ci_low"] > 0.0 or drew["ci_high"] < 0.0
        exceeds = abs(difference) >= spec["margin"]
        if spec["verdict_relevant"]:
            shift = ("shift" if exceeds and excludes_zero
                     else ("borderline" if exceeds else "no_shift"))
        else:
            shift = "descriptive_shift" if excludes_zero else "descriptive_no_shift"
        results.append({
            "contrast_id": spec["id"],
            "covariate": spec["covariate"],
            "kind": spec["kind"],
            "role": "verdict_relevant" if spec["verdict_relevant"] else "descriptive_only",
            "unit": plan["covariates"][spec["covariate"]]["unit"],
            "margin": spec["margin"],
            "target_stratum": TARGET_STRATUM,
            "reference_stratum": REFERENCE_STRATUM,
            "value_target": drew["value_target"],
            "value_reference": drew["value_reference"],
            "difference": difference,
            "ci_low": drew["ci_low"],
            "ci_high": drew["ci_high"],
            "n_target": drew["n_target"],
            "n_reference": drew["n_reference"],
            "states_target": drew["states_target"],
            "states_reference": drew["states_reference"],
            "usable_draws": drew["usable_draws"],
            "dropped_draws": drew["dropped_draws"],
            "interval_excludes_zero": bool(excludes_zero),
            "exceeds_margin": bool(exceeds),
            "shift": shift,
            "label": "DESCRIPTIVE",
        })
    return results


def value_summary(rows, key):
    values = [float(row[key]) for row in rows if row[key] is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values)
    return {"n": int(array.size), "mean": float(array.mean()),
            "median": float(np.median(array)), "min": float(array.min()),
            "max": float(array.max())}


def mechanism_inventory(rows):
    inventory = {}
    for stratum in STRATA:
        subset = [row for row in rows if row["stratum"] == stratum]
        inventory[stratum] = {
            "slots": len(subset),
            "states": len({row["state_identity"] for row in subset}),
            "source_members": len({row["source_member"] for row in subset}),
            "published_successes": sum(1 for row in subset
                                       if row["published_outcome_first_shot_success"]),
            "wall_seconds": value_summary(subset, "issue_87_wall_seconds"),
            "engine_wall_seconds": value_summary(subset, "issue_87_engine_wall_seconds"),
            "peak_cpu_rss_mib": value_summary(subset, "peak_cpu_rss_mib"),
            "minutes_since_first_dispatch": value_summary(subset, "minutes_since_first_dispatch"),
            "dispatch_index": value_summary(subset, "dispatch_index"),
            "live_at_dispatch": value_summary(subset, "live_at_dispatch"),
            "dispatch_burst_size": value_summary(subset, "dispatch_burst_size"),
            "live_at_dispatch_histogram": {
                str(key): int(value) for key, value in
                sorted(Counter(row["live_at_dispatch"] for row in subset).items())},
            "frozen_terminal_kind_histogram": {
                str(key): int(value) for key, value in
                sorted(Counter(row["frozen_replay_terminal_kind"] for row in subset).items())},
        }
    annotated = [dict(row, live_four=1.0 if row["live_at_dispatch"] == 4 else 0.0)
                 for row in rows]
    live_four = clustered_contrast(annotated, lambda row: row["live_four"], stratum_side,
                                   BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    live_four["difference"] = live_four["value_target"] - live_four["value_reference"]
    inventory["all_workers_live_dispatch_share"] = {
        "definition": "share of slots dispatched while 4 oracle slots were live, derived from "
                      "the published dispatch timestamps and per-cell wall seconds",
        "target": {key: live_four[key] for key in ("value_target", "n_target", "states_target")},
        "reference": {key: live_four[key] for key in ("value_reference", "n_reference",
                                                      "states_reference")},
        "difference": live_four["difference"],
        "ci_low": live_four["ci_low"],
        "ci_high": live_four["ci_high"],
        "usable_draws": live_four["usable_draws"],
        "label": "DESCRIPTIVE mechanism inventory (never a verdict input)",
    }
    return inventory


def per_state_table(rows):
    table = {}
    for state in sorted({row["state_identity"] for row in rows}):
        subset = [row for row in rows if row["state_identity"] == state]
        table[state] = {
            "family": subset[0]["family"],
            "source_member": subset[0]["source_member"],
            "study_role": subset[0]["study_role"],
            "slots": len(subset),
            "executed": sum(1 for row in subset if row["stratum"] == "executed"),
            "timeouts": sum(1 for row in subset if row["stratum"] == "timeout"),
            "stability_violations": sum(1 for row in subset if row["stratum"] == "stability"),
            "published_successes": sum(1 for row in subset
                                       if row["published_outcome_first_shot_success"]),
            "frozen_cost_mean": value_summary(subset, "frozen_replay_count_cost")["mean"],
            "frozen_cost_median": value_summary(subset, "frozen_replay_count_cost")["median"],
            "frozen_pig_removal_indicator_slots": sum(
                1 for row in subset if row["frozen_replay_pig_removed"]),
            "best_observed_pig_removal": subset[0]["state_best_observed_pig_removed"],
            "frozen_terminal_kinds": {
                str(key): int(value) for key, value in
                sorted(Counter(row["frozen_replay_terminal_kind"] for row in subset).items())},
            "timeout_slots": [row["slot_identity"] for row in subset
                              if row["stratum"] == "timeout"],
            "stability_slots": [row["slot_identity"] for row in subset
                                if row["stratum"] == "stability"],
        }
    return table


def association_context(rows):
    executed = [dict(row, outcome=1.0 if row["published_outcome_first_shot_success"] else 0.0)
                for row in rows if row["stratum"] == REFERENCE_STRATUM]
    drew = clustered_contrast(executed, lambda row: row["outcome"],
                              lambda row: (None if row["frozen_replay_pig_removed"] is None
                                           else bool(row["frozen_replay_pig_removed"])),
                              BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    return {
        "definition": "within the executed stratum: published engine-truth success rate of "
                      "slots whose frozen replay removed a pig minus the rate of the rest, with "
                      "the same state-clustered interval",
        "role": "DESCRIPTIVE association context for the semantics of the frozen indicators; "
                "never a verdict input",
        "value_indicator_true": drew["value_target"],
        "value_indicator_false": drew["value_reference"],
        "difference": drew["value_target"] - drew["value_reference"],
        "ci_low": drew["ci_low"],
        "ci_high": drew["ci_high"],
        "n_indicator_true": drew["n_target"],
        "n_indicator_false": drew["n_reference"],
        "states_indicator_true": drew["states_target"],
        "states_indicator_false": drew["states_reference"],
        "usable_draws": drew["usable_draws"],
        "dropped_draws": drew["dropped_draws"],
        "label": "DESCRIPTIVE",
    }


def stability_stratum(rows):
    subset = sorted([row for row in rows if row["stratum"] == "stability"],
                    key=lambda row: row["slot_identity"])
    keys = ("slot_identity", "state_identity", "family", "source_member", "ordinal",
            "frozen_replay_count_cost", "normalized_cost_position",
            "frozen_replay_terminal_kind", "frozen_replay_pig_removed",
            "state_best_observed_pig_removed", "live_at_dispatch",
            "minutes_since_first_dispatch")
    return {
        "slots": [row["slot_identity"] for row in subset],
        "states": sorted({row["state_identity"] for row in subset}),
        "ordinals": sorted(row["ordinal"] for row in subset),
        "signature": STABILITY_SIGNATURE,
        "mechanism": "the physics is not static at the sealed decision step 30000; distinct "
                     "from the connect/barrier/readiness timeout mechanism",
        "interval_eligible": False,
        "typed_note": "reported as its own typed stratum with its frozen covariates; no interval "
                      "is computed for a single-state stratum of 3 slots, and it is never pooled "
                      "into the timeout contrast",
        "frozen_covariates": [{key: row[key] for key in keys} for row in subset],
    }


def citation_block():
    summary, records = load_eighty_nine()
    tallies = Counter()
    for record in records.values():
        origin = record.get("issue87_typed_failure") or "none"
        kind = ("timeout" if origin.startswith("TimeoutError")
                else ("stability" if origin.startswith("ValueError") else "other"))
        success = bool(record.get("outcome") and record["outcome"]["first_shot_success"])
        tallies[(kind, success)] += 1
    return {
        "source_identity": summary["identity"],
        "source_schema": summary["schema"],
        "cited_fields": ["completion_pass.inventory", "ceiling_reestimate.pooled",
                         "records/*.json outcome.first_shot_success",
                         "records/*.json issue87_typed_failure"],
        "published_inventory": summary["completion_pass"]["inventory"],
        "published_pooled_ceiling": summary["ceiling_reestimate"]["pooled"],
        "successes_by_origin_stratum": {
            stratum: {"slots": tallies[(stratum, True)] + tallies[(stratum, False)],
                      "successes": tallies[(stratum, True)]}
            for stratum in ("timeout", "stability", "other")},
        "label": "CITED PUBLISHED CROSS-CHECK (never recomputed from engine evidence; never a "
                 "verdict input)",
    }


def verdict_record(plan, rows, contrasts):
    guards = plan["precision_guards"]
    strata = Counter(row["stratum"] for row in rows)
    joined = sum(1 for row in rows if row["join_complete"])
    coverage = joined / len(rows)
    timeout_states = len({row["state_identity"] for row in rows
                          if row["stratum"] == TARGET_STRATUM})
    relevant = [row for row in contrasts if row["role"] == "verdict_relevant"]
    usable_fraction = min(row["usable_draws"] / max(row["usable_draws"] + row["dropped_draws"], 1)
                          for row in relevant)
    failures = []
    if strata[TARGET_STRATUM] < guards["min_timeout_slots"]:
        failures.append(f"timeout slots {strata[TARGET_STRATUM]} below "
                        f"{guards['min_timeout_slots']}")
    if strata[REFERENCE_STRATUM] < guards["min_executed_slots"]:
        failures.append(f"executed slots {strata[REFERENCE_STRATUM]} below "
                        f"{guards['min_executed_slots']}")
    if coverage < guards["min_join_fraction"]:
        failures.append(f"join coverage {coverage:.6f} below {guards['min_join_fraction']}")
    if usable_fraction < guards["min_usable_draws_fraction"]:
        failures.append(f"usable draws fraction {usable_fraction:.6f} below "
                        f"{guards['min_usable_draws_fraction']}")
    if timeout_states < guards["min_timeout_states"]:
        failures.append(f"timeout states {timeout_states} below {guards['min_timeout_states']}")
    if failures:
        audit_verdict = "indeterminate"
    elif any(row["shift"] == "shift" for row in relevant):
        audit_verdict = "outcome_correlated"
    elif any(row["shift"] == "borderline" for row in relevant):
        audit_verdict = "indeterminate"
    else:
        audit_verdict = "outcome_blind"
    if audit_verdict not in VERDICT_TOKENS:
        raise ValueError(f"audit verdict {audit_verdict!r} is outside the frozen vocabulary")
    disposition = {"outcome_correlated": "supported",
                   "outcome_blind": "not_supported_by_this_experiment",
                   "indeterminate": "readiness_or_precision_insufficient"}[audit_verdict]
    if disposition not in DISPOSITION_TOKENS:
        raise ValueError("disposition token outside the frozen vocabulary")
    return {
        "audit_verdict": audit_verdict,
        "q1_covariate_shift": disposition,
        "guard_failures": failures,
        "join_coverage": coverage,
        "usable_draws_fraction_min": usable_fraction,
        "timeout_states": timeout_states,
        "strata": {stratum: strata[stratum] for stratum in STRATA},
        "shifted_contrasts": [row["contrast_id"] for row in relevant if row["shift"] == "shift"],
        "borderline_contrasts": [row["contrast_id"] for row in relevant
                                 if row["shift"] == "borderline"],
    }


def bias_direction_rows(plan, contrasts):
    rows = []
    for row in contrasts:
        relevant = row["role"] == "verdict_relevant"
        if relevant:
            if row["shift"] not in ("shift", "borderline"):
                direction = "no_shift"
            else:
                direction = "timeout_higher" if row["difference"] > 0 else "timeout_lower"
            if direction == "no_shift":
                statement = ("no frozen bias direction is stated: this contrast shows no shift "
                             "under the frozen margin and interval rule, so the timeout stratum "
                             "is not distinguishable from the executed stratum on this "
                             "covariate")
            else:
                statement = plan["bias_direction_rules"].get(f"{row['covariate']}|{direction}")
                if statement is None:
                    raise ValueError(f"no frozen bias-direction rule for "
                                     f"{row['covariate']}|{direction}")
        else:
            direction = "no_shift"
            statement = ("no bias direction is defined for an inventory position; the "
                         "descriptive contrast is reported for completeness only")
        rows.append({"contrast_id": row["contrast_id"], "covariate": row["covariate"],
                     "role": row["role"], "direction": direction,
                     "difference": row["difference"], "margin": row["margin"],
                     "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                     "shift": row["shift"], "statement": statement, "label": "DESCRIPTIVE"})
    return rows


def compute_tables(plan, rows):
    contrasts = contrast_rows(plan, rows)
    diagnostic = load_n1_diagnostic()
    return {
        "schema": SCHEMA_COMPUTE,
        "identity": IDENTITY,
        "plan_identity": IDENTITY,
        "plan_version": plan["version"],
        "strata": {stratum: sum(1 for row in rows if row["stratum"] == stratum)
                   for stratum in STRATA},
        "join_status_counts": {
            source: dict(Counter(row["join_status"][source] for row in rows))
            for source in sorted(rows[0]["join_status"])},
        "contrasts": contrasts,
        "verdict": verdict_record(plan, rows, contrasts),
        "bias_direction": bias_direction_rows(plan, contrasts),
        "per_state": per_state_table(rows),
        "mechanism_inventory": mechanism_inventory(rows),
        "association_context": association_context(rows),
        "stability_stratum": stability_stratum(rows),
        "cited_cross_check": citation_block(),
        "headroom_precedent": {
            "source": str(SEVENTY_SEVEN_DIAGNOSTIC / "summary.json"),
            "rule": "the #77 diagnostic headroom block is the published precedent for the "
                    "best-observed pig-removal indicator; it covers the 4 N1 held-out "
                    "evaluation base states only, so it is cited and never used as the "
                    "slot-level covariate",
            "states": [{"state": entry["state"], "family": entry["family"],
                        "pig_removed_opportunity": entry["opportunities"]["pig_removed"]}
                       for entry in diagnostic["headroom"]],
        },
    }


# ---------------------------------------------------------------------------
# published artifact renderers (byte-deterministic)
# ---------------------------------------------------------------------------

def comparisons_csv(compute):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("row_type", "contrast_id", "covariate", "kind", "role", "unit",
                     "target_stratum", "reference_stratum", "n_target", "n_reference",
                     "states_target", "states_reference", "value_target", "value_reference",
                     "difference", "ci_low", "ci_high", "margin", "interval_excludes_zero",
                     "exceeds_margin", "shift", "label"))
    for row in compute["contrasts"]:
        writer.writerow((
            "contrast", row["contrast_id"], row["covariate"], row["kind"], row["role"],
            row["unit"], row["target_stratum"], row["reference_stratum"], row["n_target"],
            row["n_reference"], row["states_target"], row["states_reference"],
            repr(row["value_target"]), repr(row["value_reference"]), repr(row["difference"]),
            repr(row["ci_low"]), repr(row["ci_high"]), repr(row["margin"]),
            str(row["interval_excludes_zero"]).lower(), str(row["exceeds_margin"]).lower(),
            row["shift"], row["label"]))
    for row in compute["bias_direction"]:
        writer.writerow(("bias_direction", row["contrast_id"], row["covariate"], "", row["role"],
                         "", "", "", "", "", "", "", "", "", repr(row["difference"]),
                         repr(row["ci_low"]), repr(row["ci_high"]), repr(row["margin"]), "", "",
                         row["direction"], row["label"]))
    for stratum in STRATA:
        entry = compute["mechanism_inventory"][stratum]
        writer.writerow(("mechanism_stratum", "", "", "", "mechanism_inventory_only", "slots",
                         stratum, "", entry["slots"], "", entry["states"], "",
                         entry["source_members"], "", "", "", "", "", "", "",
                         "descriptive", "DESCRIPTIVE"))
        for key in ("wall_seconds", "engine_wall_seconds", "peak_cpu_rss_mib",
                    "minutes_since_first_dispatch", "dispatch_index", "live_at_dispatch",
                    "dispatch_burst_size"):
            measure = entry[key]
            writer.writerow(("mechanism_measure", "", "", "", "mechanism_inventory_only", key,
                             stratum, "", measure["n"], "", "", "",
                             "" if measure["mean"] is None else repr(measure["mean"]),
                             "" if measure["median"] is None else repr(measure["median"]),
                             repr(measure["min"]), repr(measure["max"]), "", "", "", "",
                             "descriptive", "DESCRIPTIVE"))
    share = compute["mechanism_inventory"]["all_workers_live_dispatch_share"]
    writer.writerow(("mechanism_measure", "", "", "", "mechanism_inventory_only",
                     "all_workers_live_dispatch_share", TARGET_STRATUM, REFERENCE_STRATUM,
                     share["target"]["n_target"], share["reference"]["n_reference"],
                     share["target"]["states_target"], share["reference"]["states_reference"],
                     repr(share["target"]["value_target"]),
                     repr(share["reference"]["value_reference"]),
                     repr(share["difference"]), repr(share["ci_low"]), repr(share["ci_high"]),
                     "", "", "", "descriptive", "DESCRIPTIVE"))
    context = compute["association_context"]
    writer.writerow(("association_context", "", "frozen_replay_pig_removed",
                     "rate", "descriptive_only", "published success rate",
                     "indicator_true", "indicator_false", context["n_indicator_true"],
                     context["n_indicator_false"], context["states_indicator_true"],
                     context["states_indicator_false"], repr(context["value_indicator_true"]),
                     repr(context["value_indicator_false"]), repr(context["difference"]),
                     repr(context["ci_low"]), repr(context["ci_high"]), "", "", "",
                     "descriptive", "DESCRIPTIVE"))
    for state, entry in sorted(compute["per_state"].items()):
        writer.writerow(("per_state", "", "", "", "", "slots", state, entry["family"],
                         entry["slots"], entry["executed"], "", "",
                         repr(entry["frozen_cost_mean"]), repr(entry["frozen_cost_median"]),
                         repr(entry["timeouts"]), repr(entry["stability_violations"]),
                         repr(entry["frozen_pig_removal_indicator_slots"]),
                         repr(entry["best_observed_pig_removal"]), "", "", "descriptive",
                         "DESCRIPTIVE"))
    for stratum, block in sorted(
            compute["cited_cross_check"]["successes_by_origin_stratum"].items()):
        writer.writerow(("cited_cross_check", "", "", "", "cited_published_never_recomputed",
                         "slots", stratum, "", block["slots"], block["successes"], "", "", "", "",
                         "", "", "", "", "", "", "descriptive", "DESCRIPTIVE"))
    for row in compute["stability_stratum"]["frozen_covariates"]:
        writer.writerow(("stability_stratum", "", "", "", "separate_typed_stratum", "slot",
                         row["slot_identity"], row["state_identity"],
                         repr(row["ordinal"]), "", "", "",
                         repr(row["frozen_replay_count_cost"]),
                         repr(row["normalized_cost_position"]),
                         repr(row["frozen_replay_pig_removed"]), "", "", "", "", "", "", "", ""))
    return stream.getvalue()


SLOT_CSV_FIELDS = (
    "slot_identity", "state_identity", "source_member", "family", "study_role", "ordinal",
    "branch_identity", "stratum", "issue_87_failure", "frozen_replay_count_cost",
    "normalized_cost_position", "frozen_cost_span_informative", "frozen_replay_terminal_kind",
    "frozen_replay_pig_removed", "state_best_observed_pig_removed", "is_type010105",
    "published_outcome_first_shot_success", "published_engine_channel_pig_removed",
    "issue_87_wall_seconds", "issue_87_engine_wall_seconds", "dispatched_at_utc",
    "dispatch_index", "minutes_since_first_dispatch", "live_at_dispatch", "dispatch_burst_size",
    "peak_cpu_rss_mib", "join_complete")

SLOT_CSV_SOURCES = ("issue_80_candidate_table", "issue_77_n1_coverage", "issue_77_n1_replay",
                    "issue_73_headroom_registry", "issue_87_dispatch_marker",
                    "issue_87_receipt")


def slot_join_csv(rows):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(SLOT_CSV_FIELDS) + [f"join_{source}" for source in SLOT_CSV_SOURCES])
    for row in rows:
        values = []
        for key in SLOT_CSV_FIELDS:
            value = row[key]
            values.append("" if value is None
                          else (repr(value) if isinstance(value, float) else str(value)))
        writer.writerow(values + [row["join_status"][source] for source in SLOT_CSV_SOURCES])
    return stream.getvalue()


def findings_md(plan, compute, timings):
    verdict = compute["verdict"]
    context = plan["published_context"]["issue_87_pooled_ceiling"]
    by_family = plan["published_context"]["issue_87_by_family"]
    lines = ["# Issue-90 timeout-outcome correlation audit for the #87 oracle pass (analysis-only)",
             "",
             f"Frozen protocol: plan.json version {plan['version']}, frozen "
             f"{plan['frozen_at']} before any audit statistic. Zero re-execution, zero "
             "rendering, zero GPU work; every input READ-ONLY. All intervals DESCRIPTIVE.",
             "",
             f"Audit verdict (frozen vocabulary): `{verdict['audit_verdict']}`",
             f"Q1 covariate-shift disposition: `{verdict['q1_covariate_shift']}`",
             ""]
    if verdict["guard_failures"]:
        lines.append(f"Precision guards tripped: {json.dumps(verdict['guard_failures'])}")
        lines.append("")
    lines += ["## Universe and strata",
              "",
              "| stratum | slots | states | role |",
              "| --- | --- | --- | --- |"]
    for stratum in STRATA:
        entry = compute["mechanism_inventory"][stratum]
        role = {"executed": "reference stratum of every contrast",
                "timeout": "target stratum of every contrast",
                "stability": "separate typed stratum, never pooled into the timeout contrast"}[stratum]
        lines.append(f"| {stratum} | {entry['slots']} | {entry['states']} | {role} |")
    lines += ["",
              f"Join coverage of the verdict-relevant sources: {verdict['join_coverage']:.6f} of "
              f"{SCHEDULED_SLOTS} slots; the timeout stratum spans {verdict['timeout_states']} "
              "states. Every slot carries a typed per-source join status in slot_join.csv.",
              "",
              "## Q1 - frozen covariate contrasts (timeout minus executed, descriptive)",
              "",
              "| contrast | covariate | target | reference | difference | descriptive 95% "
              "interval | margin | shift |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in compute["contrasts"]:
        lines.append(f"| {row['contrast_id']} | {row['covariate']} | {row['value_target']:.6f} "
                     f"| {row['value_reference']:.6f} | {row['difference']:+.6f} | "
                     f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['margin']:.6f} | "
                     f"{row['shift']} |")
    lines += ["",
              "`shift` means the absolute difference reaches the frozen margin and the "
              "descriptive interval excludes 0; `borderline` means it reaches the margin with "
              "the interval including 0. Only the five verdict-relevant contrasts enter the "
              "verdict; d6_ordinal_position is reported for completeness because an inventory "
              "position is not outcome-linked by itself.",
              "",
              "## Q2 - bias direction for #87's ceiling (typed, descriptive)",
              "",
              f"- Published #87 pooled ceiling: {context['ceiling_states']} of "
              f"{context['measured_states']} measured states ({context['prevalence']:.6f}); "
              f"{context['states']} scheduled states, unmeasured {context['unmeasured_states']}.",
              f"- Published #87 family ceilings: type010103 "
              f"{by_family['type010103']['ceiling_states']}/"
              f"{by_family['type010103']['states']}, type010105 "
              f"{by_family['type010105']['ceiling_states']}/"
              f"{by_family['type010105']['states']}.",
              ""]
    for row in compute["bias_direction"]:
        lines.append(f"- `{row['contrast_id']}` ({row['direction']}, difference "
                     f"{row['difference']:+.6f}, margin {row['margin']:.6f}, shift "
                     f"{row['shift']}): {row['statement']} (DESCRIPTIVE)")
    association = compute["association_context"]
    lines += ["",
              "## Frozen indicator semantics (association context)",
              "",
              f"Within the executed stratum, slots whose frozen replay removed a pig show a "
              f"published success rate of {association['value_indicator_true']:.6f} "
              f"(n={association['n_indicator_true']}, states="
              f"{association['states_indicator_true']}) against "
              f"{association['value_indicator_false']:.6f} "
              f"(n={association['n_indicator_false']}, states="
              f"{association['states_indicator_false']}): difference "
              f"{association['difference']:+.6f} with descriptive interval "
              f"[{association['ci_low']:.6f}, {association['ci_high']:.6f}]. This is context "
              "for the semantics of the frozen indicators and is never a verdict input.",
              "",
              "## Mechanism inventory (descriptive; never a verdict input)",
              ""]
    for stratum in STRATA:
        entry = compute["mechanism_inventory"][stratum]
        engine = ("not applicable" if entry["engine_wall_seconds"]["median"] is None
                  else format(entry["engine_wall_seconds"]["median"], ".1f"))
        rss = ("not applicable" if entry["peak_cpu_rss_mib"]["median"] is None
               else format(entry["peak_cpu_rss_mib"]["median"], ".1f"))
        lines.append(f"- {stratum}: {entry['slots']} slots over {entry['states']} "
                     f"{'state' if entry['states'] == 1 else 'states'} and "
                     f"{entry['source_members']} source "
                     f"{'member' if entry['source_members'] == 1 else 'members'}; wall seconds "
                     f"median {entry['wall_seconds']['median']:.1f} (max "
                     f"{entry['wall_seconds']['max']:.1f}); engine wall seconds median {engine}; "
                     f"peak RSS median {rss} MiB; live_at_dispatch "
                     f"{entry['live_at_dispatch_histogram']}; dispatch window "
                     f"{entry['minutes_since_first_dispatch']['min']:.1f} to "
                     f"{entry['minutes_since_first_dispatch']['max']:.1f} minutes after the "
                     f"first dispatch; published successes {entry['published_successes']}")
    share = compute["mechanism_inventory"]["all_workers_live_dispatch_share"]
    lines += ["",
              f"All-workers-live dispatch share (derived cold-start contention proxy): timeout "
              f"{share['target']['value_target']:.6f} versus executed "
              f"{share['reference']['value_reference']:.6f}, difference "
              f"{share['difference']:+.6f} with descriptive interval "
              f"[{share['ci_low']:.6f}, {share['ci_high']:.6f}] (DESCRIPTIVE mechanism "
              "inventory).",
              "",
              "## Stability stratum (typed, separate mechanism)",
              ""]
    stability = compute["stability_stratum"]
    lines.append(f"- slots {stability['slots']} in states {stability['states']} at ordinals "
                 f"{stability['ordinals']}; signature {stability['signature']!r}. Typed "
                 f"note: {stability['typed_note']}.")
    for row in stability["frozen_covariates"]:
        lines.append(f"  - {row['slot_identity']}: frozen cost "
                     f"{row['frozen_replay_count_cost']!r}, normalized position "
                     f"{row['normalized_cost_position']!r}, frozen terminal kind "
                     f"{row['frozen_replay_terminal_kind']!r}, frozen pig removal "
                     f"{row['frozen_replay_pig_removed']!r}, best-observed pig removal "
                     f"{row['state_best_observed_pig_removed']!r}")
    cited = compute["cited_cross_check"]
    lines += ["",
              "## Cited published cross-check",
              "",
              f"- source `{cited['source_identity']}` (`{cited['source_schema']}`); cited fields "
              f"{cited['cited_fields']}",
              f"- published inventory: {json.dumps(cited['published_inventory'], sort_keys=True)}",
              f"- cited successes by #87 failure stratum: "
              f"{json.dumps(cited['successes_by_origin_stratum'], sort_keys=True)}",
              f"- attribution limit: {plan['cited_cross_check']['attribution_limit']}",
              "",
              "## Frozen protocol",
              "",
              f"- Frozen semantics: {plan['frozen_semantics']}.",
              "- Join keys: " + "; ".join(f"{key}: {value}"
                                          for key, value in plan["join_keys"].items()) + ".",
              "- Statistics: " + plan["statistics"]["contrast"] + ". Interval: "
              + plan["statistics"]["interval"] + ".",
              "- Margins: " + "; ".join(f"{spec['id']}={spec['margin']!r}"
                                        for spec in plan["contrasts"]) + ".",
              "- Verdict mapping: " + "; ".join(
                  f"{key}: {plan['verdict_mapping'][key]}"
                  for key in ("shift_rule", "borderline_rule", "outcome_correlated",
                              "outcome_blind", "indeterminate")) + ".",
              "- Precision guards: "
              + json.dumps({key: value for key, value in plan["precision_guards"].items()
                            if key != "rule"}, sort_keys=True) + ".",
              "",
              "## Limitations and claim boundary",
              ""]
    for item in plan["limitations"]:
        lines.append(f"- {item}")
    lines += ["", f"- Claim boundary: {plan['claim_boundary']}.",
              "",
              "## Protocol chronology",
              ""]
    for entry in plan["changelog"]:
        lines.append(f"- version {entry['version']}, frozen {plan['frozen_at']} before its "
                     f"scoring run: {entry['change']}. Why: {entry['why']} Evidence: "
                     f"{entry['evidence']}")
    lines += ["", "## Compute accounting", ""]
    for name in sorted(timings):
        entry = timings[name]
        lines.append(f"- {name}: {entry['seconds']:.3f}s wall, peak RSS "
                     f"{entry['peak_rss_mib']:.1f} MiB")
    lines += [f"- Caps: {plan['caps']['wall_cap_seconds']}s measured wall, "
              f"{plan['caps']['derived_bytes_budget']} bytes derived artifacts, "
              f"{plan['caps']['gpu_cap_seconds']} GPU seconds.",
              "",
              "## Reproduction",
              "",
              f"`{plan['validation_command']}` recomputes every published table from the frozen "
              "inputs and byte-compares it against the published artifacts.",
              ""]
    return "\n".join(lines)


def build_report(plan, compute, timings):
    return {
        "schema": SCHEMA_REPORT,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "plan_version": plan["version"],
        "frozen_at": plan["frozen_at"],
        "validation_command": plan["validation_command"],
        "questions": plan["questions"],
        "audit_verdict": compute["verdict"]["audit_verdict"],
        "verdict_detail": compute["verdict"],
        "question_dispositions": {
            "q1_covariate_shift": compute["verdict"]["q1_covariate_shift"],
            "q2_audit_verdict": compute["verdict"]["audit_verdict"],
        },
        "strata": compute["strata"],
        "join_status_counts": compute["join_status_counts"],
        "covariates": plan["covariates"],
        "contrasts": compute["contrasts"],
        "bias_direction": compute["bias_direction"],
        "per_state": compute["per_state"],
        "mechanism_inventory": compute["mechanism_inventory"],
        "association_context": compute["association_context"],
        "stability_stratum": compute["stability_stratum"],
        "cited_cross_check": compute["cited_cross_check"],
        "headroom_precedent": compute["headroom_precedent"],
        "published_context": plan["published_context"],
        "precision_guards": plan["precision_guards"],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": BOOTSTRAP_UNIT,
                      "quantiles": list(INTERVAL_QUANTILES), "label": "DESCRIPTIVE"},
        "caps": plan["caps"],
        "claim_boundary": plan["claim_boundary"],
        "limitations": plan["limitations"],
        "protocol_chronology": plan["changelog"],
        "hygiene": {
            "dry_run": "the --dry-run mode computes NO statistic of any kind (structural "
                       "inventory only), before or after the freeze",
            "freeze": "the full protocol with numeric margins is published by --prepare before "
                      "--run exists",
            "flock": "the exclusive flock on /tmp/novphy-addexp-gpu.lock is held for --run, "
                     "--publish and --validate and released immediately after each",
            "content_hashes": "no content hash is recomputed after the freeze and no "
                              "full-corpus integrity pass runs; inputs are verified at read "
                              "time by their carried schema/identity/plan_identity fields",
            "gpu": "no GPU work of any kind in this ticket",
        },
        "compute": {"phases": timings,
                    "wall_seconds_total": sum(entry["seconds"] for entry in timings.values()),
                    "gpu_seconds_total": 0.0},
    }


# ---------------------------------------------------------------------------
# measured phases
# ---------------------------------------------------------------------------

def load_rows(notify=None):
    plan87, ledger, records, markers, receipts = load_issue_87()
    states = plan87["states"]
    tables = load_candidate_tables(states)
    coverage = load_n1_coverage()
    branches = {cell["branch_identity"] for cell in plan87["oracle_cells"]}
    branches.update(item["branch_identity"] for state in states for item in state["inventory"])
    replays = load_n1_replays(branches)
    headroom_ids = set(load_seventy_three()["state_ids"])
    return build_slot_table(plan87, ledger, records, markers, receipts, states, tables,
                            coverage, replays, headroom_ids, notify)


def progress(started, done, total):
    elapsed = time.monotonic() - started
    eta = elapsed / done * (total - done) if done else 0.0
    log(f"slot join {done}/{total}; elapsed {elapsed:.1f}s; eta {eta:.1f}s")


def measured(function, *arguments, notify=None):
    began = time.monotonic()
    if notify is None:
        value = function(*arguments)
    else:
        value = function(*arguments, notify=notify)
    seconds = time.monotonic() - began
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    log(f"phase complete in {seconds:.3f}s (peak RSS {peak:.1f} MiB)")
    return value, {"seconds": seconds, "peak_rss_mib": peak}


def run_audit(output, write=True):
    output = Path(output)
    plan = load_terminal_plan(output)
    timings = {}
    began = time.monotonic()
    rows = load_rows(notify=lambda done, total: progress(began, done, total))
    timings["join_and_covariates"] = {
        "seconds": time.monotonic() - began,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    log(f"{len(rows)}-slot join table complete in "
        f"{timings['join_and_covariates']['seconds']:.3f}s; computing the frozen statistics")
    compute, timings["compute_statistics"] = measured(compute_tables, plan, rows)
    total = sum(entry["seconds"] for entry in timings.values())
    if total > WALL_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: measured wall {total:.3f}s exceeds the frozen cap "
                         f"{WALL_CAP_SECONDS}s")
    compute["phases"] = timings
    log(f"audit verdict {compute['verdict']['audit_verdict']}; disposition "
        f"{compute['verdict']['q1_covariate_shift']}; measured wall {total:.3f}s of "
        f"{WALL_CAP_SECONDS}s cap")
    path = output / "compute.json"
    retained = read_json(path) if path.is_file() else None
    if retained is not None and (
            {key: value for key, value in retained.items() if key != "phases"} !=
            {key: value for key, value in compute.items() if key != "phases"}):
        raise ValueError("retained compute.json differs from the fresh recomputation; the "
                         "audit is frozen and is never silently overwritten")
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
    rows = load_rows()
    if slot_join_csv(rows).encode() != (output / "slot_join.csv").read_bytes():
        raise ValueError("fresh slot table differs from the published slot_join.csv")
    timings = compute["phases"]
    report = build_report(plan, compute, timings)
    write_json(output / "summary.json", report)
    write_text(output / "comparisons.csv", comparisons_csv(compute))
    write_text(output / "findings.md", findings_md(plan, compute, timings))
    write_json(output / "receipts.json", {
        "schema": "issue_90_oracle_timeout_audit_receipts_v1",
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "phases": timings,
        "wall_seconds_total": sum(entry["seconds"] for entry in timings.values()),
        "gpu_seconds_total": 0.0,
        "derived_bytes": file_bytes(output),
        "derived_bytes_budget": DERIVED_BYTES_BUDGET,
    })
    log(f"published summary.json, comparisons.csv, findings.md, slot_join.csv and receipts.json "
        f"in {output}")
    return report


def validate(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    states = read_json(EIGHTY_SEVEN / "plan.json")["states"]
    tables = load_candidate_tables(states)
    margin = raw_cost_margin(states, tables)
    if plan["statistics"]["raw_cost_margin"] != margin:
        raise ValueError("frozen raw-cost margin differs from the frozen formula recomputation")
    if plan["statistics"]["raw_cost_margin"] is None:
        raise ValueError("frozen plan lacks a materialized raw-cost margin")
    (compute, fresh_rows), _ = measured(run_audit, output, False)
    published_compute = read_json(output / "compute.json")
    retained = {key: value for key, value in published_compute.items() if key != "phases"}
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
            raise ValueError(f"published {name} differs from the recomputation from the frozen "
                             "inputs")
    cited = load_eighty_nine()[0]
    published_cited = read_json(output / "summary.json")["cited_cross_check"]
    if cited["completion_pass"]["inventory"] != published_cited["published_inventory"]:
        raise ValueError("the cited issue-89 inventory differs from the published citation")
    if cited["ceiling_reestimate"]["pooled"] != published_cited["published_pooled_ceiling"]:
        raise ValueError("the cited issue-89 pooled ceiling differs from the published citation")
    log(f"exact recomputation validation passed: {len(fresh_rows)} slots rejoined, every "
        f"published table byte-compared, the cited issue-89 aggregates re-cited verbatim; no "
        f"engine evidence was read and no content hash was recomputed")
    return 0


def dry_run(output):
    """No-write, NO-SCORING structural inventory (before or after the freeze)."""
    output = Path(output)
    log("no-write dry run; structural inventory only; no mean, rate, difference, proportion, "
        "correlation, margin test or interval is computed at any point")
    plan87, ledger, records, markers, receipts = load_issue_87()
    states = plan87["states"]
    tables = load_candidate_tables(states)
    coverage = load_n1_coverage()
    branches = {cell["branch_identity"] for cell in plan87["oracle_cells"]}
    branches.update(item["branch_identity"] for state in states for item in state["inventory"])
    replays = load_n1_replays(branches)
    headroom73 = load_seventy_three()
    counts = slot_inventory_joins(plan87, coverage, tables, replays, markers)
    log(f"#87 plan {plan87['schema']} identity {plan87['identity']} version {plan87['version']} "
        f"status {plan87['status']}; scheduled oracle cells {len(plan87['oracle_cells'])}")
    log(f"#87 ledger {ledger['schema']} status {ledger['status']} cells {len(ledger['cells'])}; "
        f"records {len(records)}; dispatch markers {len(markers)}; receipts {len(receipts)}")
    strata = Counter(stratum_of(record) for record in records.values())
    log("stratum sizes (structural slot counts only): "
        + json.dumps({stratum: strata[stratum] for stratum in STRATA}, sort_keys=True))
    log(f"membership states {len(states)}; families "
        f"{json.dumps(dict(sorted(Counter(state['generator_family'] for state in states).items())), sort_keys=True)}; "
        f"source members {len({state['source_member'] for state in states})}; inventory sizes "
        f"{json.dumps(dict(sorted(Counter(len(state['inventory']) for state in states).items())), sort_keys=True)}")
    log(f"#80 candidate tables loaded {len(tables)}; #77 N1 coverage branches {len(coverage)} "
        f"with statuses "
        f"{json.dumps(dict(sorted(Counter(entry['status'] for entry in coverage.values()).items())), sort_keys=True)}")
    exposed = sum(1 for summary in replays.values()
                  if summary and "pig_removed" in (summary.get("event_counts") or {}))
    log(f"#77 N1 replay records {sum(1 for summary in replays.values() if summary)}; replay "
        f"records exposing an engine-truth pig_removed event count {exposed} (availability "
        f"only; no distribution is computed here)")
    log("structural join coverage: " + json.dumps(counts, sort_keys=True))
    diagnostic = load_n1_diagnostic()
    log(f"#77 diagnostic headroom block covers {len(diagnostic['headroom'])} N1 base states "
        f"(published precedent for the best-observed indicator; cited, never joined)")
    log(f"#73 headroom registry states {len(headroom73['state_ids'])}; first identity "
        f"{headroom73['state_ids'][0]!r} against the N1 membership prefix "
        f"{states[0]['source_member']!r}: the registry is disjoint, so the #73 join is typed "
        f"unjoinable in the slot table and the best-observed pig-removal indicator is taken "
        f"from the frozen #77 N1 replay evidence instead, as the frozen substitution records")
    stamps = sorted(markers.values())
    log(f"dispatch window {stamps[0]} to {stamps[-1]} over {len(set(stamps))} distinct "
        f"one-second dispatch stamps (structure, not a statistic)")
    plan_path = Path(output) / "plan.json"
    if plan_path.is_file():
        plan = load_plan(plan_path)
        log(f"frozen protocol present: version {plan['version']} role {plan['role']} frozen "
            f"{plan['frozen_at']}")
    else:
        log("frozen protocol absent: run --prepare to freeze the protocol before --run")
    log("dry run complete; nothing was written and no statistic was computed")
    return 0


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


def prepare(output):
    output = Path(output)
    plan_path = output / "plan.json"
    plan87, _, _, _, _ = load_issue_87()
    states = plan87["states"]
    tables = load_candidate_tables(states)
    margin = raw_cost_margin(states, tables)
    if plan_path.is_file():
        frozen = load_plan(plan_path)
        if frozen["statistics"]["raw_cost_margin"] != margin:
            raise ValueError("retained frozen plan raw-cost margin differs from the frozen "
                             "formula recomputation")
        log(f"existing frozen protocol validated (version {frozen['version']}, frozen "
            f"{frozen['frozen_at']}); no statistic was computed")
        return frozen
    frozen = frozen_plan(margin, utc_now())
    frozen["inputs"] = [
        {**entry, "sha256_at_freeze": (
            sha256_of(Path(entry["artifact"])) if Path(entry["artifact"]).is_file()
            else f"directory:{sum(1 for _ in Path(entry['artifact']).rglob('*.json'))} json files")}
        for entry in frozen["inputs"]]
    write_json(plan_path, frozen)
    log(f"frozen protocol published at {plan_path}: version {frozen['version']}, "
        f"{len(frozen['contrasts'])} contrasts, raw-cost margin "
        f"{frozen['statistics']['raw_cost_margin']!r}, "
        f"{len(PRECISION_GUARDS)} precision guards; no audit statistic was computed")
    return frozen


if __name__ == "__main__":
    raise SystemExit(main())
