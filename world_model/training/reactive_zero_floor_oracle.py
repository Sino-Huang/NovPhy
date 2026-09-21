"""Issue-82 oracle-ceiling diagnostic: Phase-0 binding and estimand logic.

Pure, artifact-free functions for the zero-floor oracle probe over the #80
frozen 24-state membership. The runner (scripts/run_reactive_zero_floor_oracle.py)
performs all I/O; this module defines, before any outcome is seen:

- the Phase-0 success-predicate binding rule (a frozen numeric threshold on
  the #80-published realized t=600 end-of-window replay count cost, anchored
  on per-cell replay-side engine pig evidence from the issue-77 N1 campaign),
- the declared divergence of that predicate from the closed-loop first-shot
  outcome (replay is right-censored; gameplay is not; proxy, labelled),
- the per-state oracle indicator, pooled prevalence with a descriptive
  paired-bootstrap interval over the 24 states, per-state best-candidate
  selection, the system-vs-oracle gap tables, and the pre-declared mapping
  from those tables to the two typed dispositions.

Descriptive only: no beats/survives language, zero-shot/adapted never pooled,
and the #80 pilot record is never amended.
"""
from __future__ import annotations

import math

import numpy as np

IDENTITY = "issue-82-zero-floor-oracle-v1"
SCHEMA_PLAN = "issue_82_zero_floor_oracle_plan_v1"
SCHEMA_REPORT = "issue_82_zero_floor_oracle_report_v1"
SCHEMA_EVIDENCE = "issue_82_replay_evidence_v1"
SCHEMA_EVALUATION = "issue_82_zero_floor_evaluation_v1"

PRIOR_ORDINAL = 8
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7201
GPU_ALLOWANCE_SECONDS = 360  # 0.1 GPU-hours, the whole-ticket ceiling
WALL_CAP_SECONDS = 3600
ARTIFACT_BYTES_CAP = 0.5 * 2**30
THRESHOLD_SENSITIVITY_GRID = (25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0)
MODEL_SYSTEMS = ("hybrid-fixed-h1", "continuous-fixed-h1", "continuous-fixed-h5")
PRIOR_SYSTEM = "no-model-ordinal-prior"
SYSTEMS = (*MODEL_SYSTEMS, PRIOR_SYSTEM)

SUCCESS_THRESHOLD_RULE = (
    "stage 1 (candidate threshold): threshold = the maximum realized t=600 "
    "end-of-window replay count cost over the frozen admissible state-candidate "
    "cells whose issue-77 N1 replay native trace records pig removal "
    "(interaction_coverage pig_removed), i.e. the least inclusive monotone bound "
    "consistent with observed replay-side pig removals; if no frozen cell "
    "records pig removal, no candidate threshold exists")
AGREEMENT_FLOOR = 0.5
DEFENSIBILITY_GATE_RULE = (
    f"stage 2 (defensibility gate, pre-declared before any state-level outcome): "
    f"the stage-1 candidate threshold is a defensible binding only if the "
    f"predicate 'realized cost <= threshold' agrees with the replay-side engine "
    f"pig evidence on at least {AGREEMENT_FLOOR:.2f} of the frozen cells (the "
    f"predicate must agree with its own anchoring executions at least half the "
    f"time); a stage-1 threshold below the floor means the realized cost cannot "
    f"express the success event and NO defensible binding exists")
BLOCKED_BINDINGS_ARE_TERMINAL = (
    "publishing readiness_or_precision_insufficient for a blocked binding is a "
    "valid terminal outcome; the binding is never forced")

DECLARED_DIVERGENCE = (
    "The predicate is a REPLAY-LEVEL PROXY, not the closed-loop first-shot "
    "outcome: (1) the replay cost is read at the right-censored t=600 "
    "end-of-window offset and is not a settled cost, while gameplay outcomes "
    "are uncensored engine events; (2) the cost aggregates a parser-derived "
    "expected surviving-pig count and block count, not an engine death event; "
    "(3) the replay rows come from the issue-77 N1 campaign executions of the "
    "frozen actions, whereas closed-loop first-shot success was recorded by "
    "the #80 pilot on freshly materialized states; (4) the threshold is the "
    "least inclusive monotone bound consistent with observed replay-side pig "
    "removals, so its agreement with both replay engine evidence and the #80 "
    "closed-loop pilot outcomes is quantified and published alongside every "
    "oracle table")

PRIOR_FALLBACK_RULE = (
    "pre-declared for FUTURE pilots only (never amended into the #80 pilot "
    "record): if the ordinal-8 candidate is not admissible on a state, the "
    "no-model ordinal prior falls back to the admissible ordinal nearest 8 by "
    "absolute distance, ties broken by the lower ordinal")

PROTOCOL_NOTE = (
    "the #80 pilot retained 3 typed decision_failure: prior_candidate_absent "
    "cells (no-model-ordinal-prior on issue-77-n1-003-a00 across seeds "
    "20260908/20260909/20260910; ordinal 8 is a typed-dropped branch of member "
    "issue-77-n1-003). " + PRIOR_FALLBACK_RULE)

CLAIM_BOUNDARY = (
    "oracle-ceiling diagnostic on the frozen #80 membership from existing "
    "#77/#80 artifacts; no new gameplay, captures, or renders; no multi-shot, "
    "adaptation, few-shot, zero-shot, or complete-gameplay claim; the oracle "
    "ceiling bounds what ANY planner could show on this membership and does "
    "not evaluate a planner; descriptive intervals only; the #80 pilot record "
    "and every prior disposition (#15/#72/#74/#75/#77/#78/#79/#80) stay "
    "unchanged; #64/#65 stay sealed and unauthorized")

LIMITATIONS = (
    "the success predicate is a proxy with quantified, non-zero divergence from "
    "both replay engine evidence and closed-loop first-shot outcomes",
    "the replay cost is right-censored at t=600 and is not a settled cost",
    "states sharing a source member share one physical initial state and "
    "candidate table; the bootstrap over 24 state identities is descriptive only",
    "the oracle ceiling is bounded by the frozen 13-candidate inventory reduced "
    "by typed branch failures; no unlisted action is considered",
    "model-system gap rows reuse the frozen #77/#80 checkpoints; no planner is "
    "evaluated and no competence is implied")


# --------------------------------------------------------------- Phase-0 binding

def bind_threshold(evidence_rows):
    """Bind the frozen numeric threshold from per-cell #77 replay evidence.

    Two pre-declared stages, both reading per-cell INPUT evidence only (no
    oracle verdict, indicator, or prevalence enters this step):

    stage 1: candidate threshold K = max realized cost over pig-removed cells.
    stage 2: defensibility gate - the K-predicate must agree with the
             replay-side engine pig evidence on >= AGREEMENT_FLOOR of the
             frozen cells; below the floor the realized cost cannot express
             the success event and no defensible binding exists.
    """
    rows = list(evidence_rows)
    removed = [row for row in rows if row["pig_removed"]]
    binding = {
        "stage1_rule": SUCCESS_THRESHOLD_RULE,
        "stage2_rule": DEFENSIBILITY_GATE_RULE,
        "declared_divergence": DECLARED_DIVERGENCE,
        "cells": len(rows),
        "pig_removed_cells": len(removed),
        "right_censored_cells": sum(1 for row in rows if row["censored"]),
        "blocked_bindings_are_terminal": BLOCKED_BINDINGS_ARE_TERMINAL,
    }
    if not removed:
        binding.update({
            "status": "no_defensible_binding", "threshold": None,
            "stage1": {"candidate_threshold": None, "setter_branches": []},
            "stage2": {"agreement_floor": AGREEMENT_FLOOR,
                       "observed_agreement": None, "passed": False},
            "blocker": ("no frozen admissible cell records a replay-side pig "
                        "removal in its issue-77 native trace, so no monotone "
                        "numeric cost threshold can anchor a success predicate "
                        "against the closed-loop first-shot success semantics")})
        return binding
    candidate = max(row["realized_count_cost"] for row in removed)
    setters = sorted(row["branch_identity"] for row in removed
                     if row["realized_count_cost"] == candidate)
    agreement = replay_divergence_table(rows, {"status": "bound", "threshold": candidate})
    always_failure = (len(rows) - len(removed)) / len(rows)
    false_successes = agreement["predicate_success_and_no_pig_removed"]
    removed_costs = sorted(row["realized_count_cost"] for row in removed)
    binding["stage1"] = {"candidate_threshold": candidate, "setter_branches": setters,
                         "pig_removed_cost_min": removed_costs[0],
                         "pig_removed_cost_max": removed_costs[-1]}
    binding["stage2"] = {
        "agreement_floor": AGREEMENT_FLOOR,
        "observed_agreement": agreement["agreement_fraction"],
        "always_failure_agreement": always_failure,
        "false_success_cells": false_successes,
        "missed_removal_cells": agreement["predicate_failure_and_pig_removed"],
        "agreement_table": agreement,
        "passed": bool(agreement["agreement_fraction"] >= AGREEMENT_FLOOR),
    }
    if binding["stage2"]["passed"]:
        binding.update({"status": "bound", "threshold": candidate,
                        "threshold_setter_branches": setters})
        return binding
    binding.update({
        "status": "no_defensible_binding", "threshold": None,
        "blocker": (
            f"the frozen estimand basis cannot express the success event: the "
            f"stage-1 candidate threshold (realized cost <= {candidate:.6f}, the "
            f"least inclusive bound consistent with the "
            f"{len(removed)} replay-side pig removals observed among "
            f"{len(rows)} frozen cells) agrees with the replay engine evidence on "
            f"only {agreement['agreement_fraction']:.4f} of the cells - below the "
            f"pre-declared {AGREEMENT_FLOOR:.2f} defensibility floor - because it "
            f"marks {false_successes} no-pig-removal cells as successes (their "
            f"parser-derived pig component is ~0 although the pig survived); the "
            f"pig-removed costs [{removed_costs[0]:.3f}, {removed_costs[-1]:.3f}] "
            f"lie interior to the overall cost distribution, so no numeric "
            f"threshold rule on the realized t=600 replay cost can separate "
            f"replay-side pig removal from survival, and the predicate cannot "
            f"defensibly proxy the closed-loop first-shot success")})
    return binding


def predicate_verdict(cost, binding):
    """The frozen predicate: success iff realized cost <= threshold."""
    if binding["status"] != "bound":
        raise ValueError("success predicate is not bound; no verdict is defined")
    if not math.isfinite(cost):
        raise ValueError("realized replay cost must be finite")
    return bool(cost <= binding["threshold"])


# ------------------------------------------------------------- oracle estimands

def best_candidate_row(inventory_rows):
    """Predicate-free best candidate by realized cost (ties -> lower ordinal)."""
    best = min(inventory_rows,
               key=lambda row: (row["realized_count_cost"], row["ordinal"]))
    return {"ordinal": best["ordinal"], "branch_identity": best["branch_identity"],
            "realized_count_cost": best["realized_count_cost"]}


def oracle_state_row(inventory_rows, binding):
    """Oracle summary of one state from its admissible candidate rows.

    ``inventory_rows``: dicts with ``ordinal``, ``branch_identity``,
    ``realized_count_cost``.
    """
    verdicts = [(predicate_verdict(row["realized_count_cost"], binding), row)
                for row in inventory_rows]
    best = best_candidate_row(inventory_rows)
    return {
        "admissible_candidates": len(inventory_rows),
        "best_candidate": best,
        "oracle_indicator": int(any(verdict for verdict, _ in verdicts)),
        "success_ordinals": sorted(row["ordinal"] for verdict, row in verdicts if verdict),
        "best_candidate_tied_ordinals": sorted(
            row["ordinal"] for row in inventory_rows
            if row["realized_count_cost"] == best["realized_count_cost"]),
    }


def replay_divergence_table(evidence_rows, binding):
    """Agreement of the predicate with replay-side engine pig evidence."""
    table = {"cells": 0, "predicate_success_and_pig_removed": 0,
             "predicate_success_and_no_pig_removed": 0,
             "predicate_failure_and_pig_removed": 0,
             "predicate_failure_and_no_pig_removed": 0}
    for row in evidence_rows:
        verdict = predicate_verdict(row["realized_count_cost"], binding)
        key = ("predicate_success_and_pig_removed" if verdict and row["pig_removed"]
               else "predicate_success_and_no_pig_removed" if verdict
               else "predicate_failure_and_pig_removed" if row["pig_removed"]
               else "predicate_failure_and_no_pig_removed")
        table[key] += 1
        table["cells"] += 1
    agreeing = (table["predicate_success_and_pig_removed"]
                + table["predicate_failure_and_no_pig_removed"])
    table["agreement_fraction"] = agreeing / table["cells"] if table["cells"] else None
    return table


def pilot_divergence_table(pilot_cells, binding):
    """Agreement of the predicate with #80 closed-loop first-shot outcomes.

    ``pilot_cells``: dicts with ``success`` (bool closed-loop first-shot
    outcome), ``chosen_realized_count_cost`` (the executed candidate's frozen
    replay cost), and ``failure`` (typed failure text or None). Typed cells
    carry no success value and are reported, never counted.
    """
    table = {"valid_cells": 0, "typed_failure_cells": 0,
             "predicate_success_and_gameplay_success": 0,
             "predicate_success_and_gameplay_failure": 0,
             "predicate_failure_and_gameplay_success": 0,
             "predicate_failure_and_gameplay_failure": 0}
    for cell in pilot_cells:
        if cell["failure"] is not None or cell["success"] is None:
            table["typed_failure_cells"] += 1
            continue
        verdict = predicate_verdict(cell["chosen_realized_count_cost"], binding)
        key = ("predicate_success_and_gameplay_success" if verdict and cell["success"]
               else "predicate_success_and_gameplay_failure" if verdict
               else "predicate_failure_and_gameplay_success" if cell["success"]
               else "predicate_failure_and_gameplay_failure")
        table[key] += 1
        table["valid_cells"] += 1
    agreeing = (table["predicate_success_and_gameplay_success"]
                + table["predicate_failure_and_gameplay_failure"])
    table["agreement_fraction"] = (agreeing / table["valid_cells"]
                                   if table["valid_cells"] else None)
    return table


def threshold_sensitivity(evidence_rows, grid=THRESHOLD_SENSITIVITY_GRID):
    """Pre-declared descriptive sensitivity of prevalence to the threshold.

    Reads the frozen per-cell evidence only; the primary estimand stays bound
    to plan["binding"], never to this grid.
    """
    table = []
    for threshold in grid:
        binding = {"status": "bound", "threshold": float(threshold)}
        table.append({"threshold": float(threshold),
                      "cells_meeting_predicate": sum(
                          1 for row in evidence_rows
                          if predicate_verdict(row["realized_count_cost"], binding))})
    return table


# ------------------------------------------------------------------- uncertainty

def indicator_interval(indicators, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Descriptive paired-bootstrap interval for a mean of per-state values.

    Identical algorithm to the #72/#80 tooling: resample the per-state values
    with replacement, ``draws`` times, seeded; percentile interval. Descriptive
    only.
    """
    values = np.asarray(indicators, dtype=float)
    if values.size == 0:
        return {"mean": None, "descriptive_95_percent_interval": None,
                "resampled_units": 0}
    rng = np.random.default_rng(seed)
    resample = values[rng.integers(values.size, size=(draws, values.size))].mean(1)
    return {"mean": float(values.mean()),
            "descriptive_95_percent_interval": np.quantile(
                resample, [.025, .975]).tolist(),
            "resampled_units": int(values.size)}


# ------------------------------------------------------------------ dispositions

def decide_dispositions(binding, prevalence_mean, closed_loop_successes):
    """Pre-declared mapping from the tables to the two typed dispositions.

    ``closed_loop_successes``: {system: first-shot successes recorded by the
    #80 pilot} (all four frozen systems). Zero-shot/adapted conditions do not
    exist here; the pilot outcomes are the only closed-loop evidence and stay
    unamended.
    """
    rules = {
        "q1_oracle_ceiling_nonzero": (
            "supported iff at least one frozen state has oracle_indicator=1 "
            "(prevalence point estimate > 0); prevalence 0 or a blocked "
            "Phase-0 binding yields the stated alternative token"),
        "q1_reading": (
            "state_difficulty_floor if prevalence = 0; ranking_failure if "
            "prevalence > 0 and all four frozen systems recorded 0 closed-loop "
            "first-shot successes in the #80 pilot"),
        "q2_recommendation": (
            "prevalence_floor_state_selection when prevalence = 0; "
            "h1_ranking_calibration_probe when the ranking_failure reading "
            "holds; otherwise not typed by this experiment"),
    }
    if binding["status"] != "bound":
        return {"rules": rules,
                "q1_oracle_ceiling_nonzero": "readiness_or_precision_insufficient",
                "q1_reading": None, "q2_recommendation": None,
                "q2_disposition": "readiness_or_precision_insufficient",
                "blocker": binding["blocker"]}
    q1 = "supported" if prevalence_mean > 0 else "not_supported_by_this_experiment"
    systems_at_zero = all(count == 0 for count in closed_loop_successes.values())
    if prevalence_mean == 0:
        reading, recommendation, q2 = ("state_difficulty_floor",
                                       "prevalence_floor_state_selection", "supported")
    elif prevalence_mean > 0 and systems_at_zero:
        reading, recommendation, q2 = ("ranking_failure",
                                       "h1_ranking_calibration_probe", "supported")
    else:
        reading, recommendation, q2 = (None, None, "not_supported_by_this_experiment")
    return {"rules": rules, "q1_oracle_ceiling_nonzero": q1,
            "q1_reading": reading, "q2_recommendation": recommendation,
            "q2_disposition": q2, "blocker": None}


def prior_fallback_ordinal(ordinals, prior_ordinal=PRIOR_ORDINAL):
    """The pre-declared future-pilot fallback (protocol note; never #80)."""
    ordered = tuple(sorted(ordinals))
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("ordinal-prior inventory is empty or has duplicate ordinals")
    if prior_ordinal in ordered:
        return prior_ordinal
    return min(ordered, key=lambda ordinal: (abs(ordinal - prior_ordinal), ordinal))
