"""Issue-85 engine-side outcome channel and bounded reactive re-diagnostic.

Pure, artifact-free functions for the ADD-EXP engine-truth outcome channel.
The runner (scripts/run_engine_outcome_reactive_diagnostic.py) performs all
I/O; this module defines, before any outcome is seen:

- the ENGINE-SIDE outcome detection rule: first-shot success is read from the
  game's own state stream during rendered play (native macro events emitted by
  the game's death/destruction hooks plus the fixed-step entity lifecycle
  stream), never from parser-derived cost;
- the declared failure modes of that channel and the consistency rule between
  its event leg and its lifecycle leg;
- the bounded Phase-A verification set (replay-anchored positive/negative
  controls) and its pass rule;
- the pre-declared no-model ordinal prior with the #82-recorded fallback rule
  (nearest admissible ordinal to 8, ties to lower);
- the pilot gate (>= 10% pooled engine-truth first-shot success prevalence,
  minimum 24 valid executions), paired descriptive bootstrap contrasts, the
  Phase-C proxy-vs-engine cross-check against the #82 published candidate
  threshold, and the mapping from those tables to the typed dispositions.

Descriptive only: no beats/survives language, zero-shot/adapted never pooled,
and every prior disposition (#15/#72/#74/#75/#77/#78/#79/#80/#81/#82/#83/#84)
stays unchanged. This ticket never re-runs, amends, or reinterprets the #80 or
#82 published outcomes.
"""
from __future__ import annotations

import math

import numpy as np

IDENTITY = "issue-85-engine-outcome-reactive-v1"
SCHEMA_PLAN = "issue_85_engine_outcome_reactive_plan_v1"
SCHEMA_RECORD = "issue_85_engine_outcome_reactive_record_v1"
SCHEMA_LEDGER = "issue_85_engine_outcome_reactive_ledger_v1"
SCHEMA_CHANNEL = "issue_85_channel_verification_v1"
SCHEMA_GATE = "issue_85_pilot_gate_v1"
SCHEMA_REPORT = "issue_85_engine_outcome_reactive_report_v1"
SCHEMA_GALLERY = "issue_85_review_gallery_v1"

PRIOR_ORDINAL = 8
MODEL_SYSTEMS = ("hybrid-fixed-h1", "continuous-fixed-h1", "continuous-fixed-h5")
PRIOR_SYSTEM = "no-model-ordinal-prior"
SYSTEMS = (*MODEL_SYSTEMS, PRIOR_SYSTEM)

PREVALENCE_FLOOR = 0.10
PILOT_MINIMUM_VALID_EXECUTIONS = 24
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7201

# The #82 replay-cost proxy predicate, bound to the published candidate
# threshold (issue-82 summary.json stage1.candidate_threshold; the least
# inclusive monotone bound over the 8 observed replay-side pig removals).
# Phase C reuses the number exactly as published; it is never re-fit here.
PROXY_THRESHOLD = 81.68825840950012

# ---------------------------------------------------------------- the channel

PIG_EVENT_TYPES = ("pig_removed", "entity_death", "entity_destroyed")
PIG_PARTICIPANT_PREFIX = "runtime:pig:"
BLOCK_PARTICIPANT_PREFIX = "runtime:block:"

CHANNEL_DETECTION_RULE = (
    "engine-truth first-shot success is read from the game's own state stream "
    "of the executed shot segment (the engine's canonical native fixed-step "
    "capture written during rendered play; no parser-derived cost enters the "
    "verdict). Primary event leg: pig_removed = the segment records at least "
    "one native macro event of type 'pig_removed', 'entity_death' or "
    "'entity_destroyed' with at least one participant whose identity starts "
    "'runtime:pig:' (the game's own pig-removal/death/destruction hooks). "
    "structure_destroyed is the same rule over 'runtime:block:' participants. "
    "Lifecycle leg (confirmatory): at least one 'runtime:pig:' entity whose "
    "per-sample lifecycle in the fixed-step state stream is 'destroyed'. The "
    "published verdict is the event leg; the lifecycle leg is published "
    "alongside for agreement review against the retained frames")
CHANNEL_FAILURE_MODES = (
    "declared failure modes: (1) a death event with an unresolvable participant "
    "identity (empty participants) would hide which entity died - detected by "
    "the lifecycle leg and typed as a channel anomaly; (2) a pig killed at the "
    "very end of the bounded native window may fire its death event before the "
    "registry marks the entity destroyed - event leg true with lifecycle leg "
    "false is a declared timing mode, recorded, NOT an anomaly; (3) the state "
    "stream records a destroyed pig while the event leg is silent - the actual "
    "missing-event failure mode - typed as a channel anomaly on that cell; "
    "(4) window censoring before any pig death leaves both legs false: the pig "
    "survived within the bounded window, not an anomaly")


def pig_channel(events):
    """Primary event-leg verdict over the executed segment's macro events."""
    pig_events, block_events, empty_pig_suspects = [], [], 0
    for event in events:
        participants = list(event.get("participants") or [])
        joined = " ".join(participants)
        if event.get("event_type") in PIG_EVENT_TYPES:
            if any(str(p).startswith(PIG_PARTICIPANT_PREFIX) for p in participants):
                pig_events.append({"fixed_step": event.get("fixed_step"),
                                   "event_type": event["event_type"],
                                   "participants": participants})
            elif "pig" in joined:
                # A pig-adjacent death event whose pig identity did not resolve
                # to the runtime:pig: namespace; counted, never silently dropped.
                empty_pig_suspects += 1
        if event.get("event_type") in ("entity_death", "entity_destroyed") and any(
                str(p).startswith(BLOCK_PARTICIPANT_PREFIX) for p in participants):
            block_events.append(event["fixed_step"])
    return {"pig_removed": bool(pig_events),
            "pig_removed_events": pig_events,
            "pig_identity_failures": empty_pig_suspects,
            "structure_destroyed": bool(block_events),
            "structure_destroyed_events": len(block_events)}


def pig_lifecycle_destroyed(samples):
    """Lifecycle leg: 'runtime:pig:' entities the state stream saw destroyed."""
    destroyed = set()
    for sample in samples:
        for entity in sample.get("entities") or []:
            if (str(entity.get("entity_id", "")).startswith(PIG_PARTICIPANT_PREFIX)
                    and entity.get("lifecycle") == "destroyed"):
                destroyed.add(entity["entity_id"])
    return sorted(destroyed)


def channel_consistency(pig_removed, lifecycle_destroyed):
    """Frozen agreement rule between the two legs (see CHANNEL_FAILURE_MODES).

    Returns 'agreement' (both legs equal), 'death_at_window_edge' (event leg
    fired, destruction not yet observed - declared timing mode), or 'anomaly'
    (state stream saw a destroyed pig the event leg never reported).
    """
    if pig_removed and lifecycle_destroyed:
        return "agreement"
    if pig_removed and not lifecycle_destroyed:
        return "death_at_window_edge"
    if not pig_removed and lifecycle_destroyed:
        return "anomaly"
    return "agreement"


# ------------------------------------------------- no-model ordinal prior arm

PRIOR_FALLBACK_RULE = (
    "the no-model ordinal prior executes the frozen candidate ordinal 8 when "
    "admissible; when ordinal 8 is a typed-dropped branch of the state, it "
    "falls back to the admissible ordinal nearest 8 by absolute distance, "
    "ties broken by the lower ordinal (the #82-recorded fallback rule, "
    "pre-declared here before any outcome; the #80 pilot record is never "
    "amended)")


def prior_fallback_choice(ordinals, prior_ordinal: int = PRIOR_ORDINAL) -> int:
    """The prior arm's executed ordinal over one admissible inventory."""
    ordinals = tuple(ordinals)
    if not ordinals or len(set(ordinals)) != len(ordinals):
        raise ValueError("ordinal-prior inventory is empty or has duplicate ordinals")
    return min(ordinals, key=lambda ordinal: (abs(ordinal - prior_ordinal), ordinal))


# ------------------------------------------------ Phase-A verification set

VERIFICATION_SEED = 20260908
VERIFICATION_SPEC = (
    # (role, state identity, candidate ordinal, expected pig_removed). The two
    # positives replay-verify from the #82 evidence: member issue-77-n1-010
    # branch a02 and member issue-77-n1-016 branch a03 each recorded an
    # engine-side pig removal in their issue-77 N1 replay native traces. The
    # two negatives come from members whose full replay inventory recorded pig
    # survival. Expectations are frozen; no outcome-conditioned replacement.
    ("positive", "issue-77-n1-010-a06", 2, True),
    ("positive", "issue-77-n1-016-a12", 3, True),
    ("negative", "issue-77-n1-001-a00", 0, False),
    ("negative", "issue-77-n1-009-a00", 0, False),
)
VERIFICATION_PASS_RULE = (
    "every scheduled verification execution runs exactly once and is retained "
    "whatever the outcome (typed execution failures retained, never retried); "
    "the channel is verified iff (a) every control produced a valid executed "
    "segment with exactly one native launch, (b) every negative control's "
    "event-leg verdict is False, (c) at least one positive control's event-leg "
    "verdict is True (one positive removal demonstrates the closed-loop "
    "channel end to end; the second positive bounds execution variance), and "
    "(d) no control shows a channel anomaly. The recorded media review of the "
    "retained gallery (data/issue-85-engine-outcome-reactive/"
    "channel-verification-media-review.json) must additionally agree with "
    "every control's channel verdict - positives: the pig is gone in the "
    "final retained frame; negatives: the pig is visible at the end - before "
    "verified may be true. Otherwise the channel is unverified and the ticket "
    "publishes readiness_or_precision_insufficient naming the blocker - "
    "terminal, complete")
MEDIA_REVIEW_SCHEMA = "issue_85_channel_media_review_v1"
MEDIA_REVIEW_RULE = (
    "the owner reviews the retained decision frame, final canonical frame and "
    "shot WebM per control (vision inspection of the published gallery) and "
    "records pig_visible_in_final_frame per control; positives must show the "
    "pig gone, negatives the pig visible, agreeing with the channel verdict")
VERIFICATION_REVIEW_RULE = (
    "each control retains its decision frame, shot WebM and final canonical "
    "frame under data/ so the published verdict is reviewable against the "
    "retained frames: positives must show the pig disappearing, negatives the "
    "pig surviving at the end of the window")


def verification_set(states):
    """Resolve the frozen verification spec against the membership inventories."""
    by_identity = {state["identity"]: state for state in states}
    resolved = []
    for role, state_identity, ordinal, expectation in VERIFICATION_SPEC:
        state = by_identity.get(state_identity)
        if state is None:
            raise ValueError(f"verification state {state_identity} is not a membership state")
        item = next((item for item in state["inventory"]
                     if item["ordinal"] == ordinal), None)
        if item is None:
            raise ValueError(
                f"verification ordinal {ordinal} is not admissible on {state_identity}")
        resolved.append({"role": role, "state": state_identity,
                         "ordinal": ordinal, "branch_identity": item["branch_identity"],
                         "action": dict(item["action"]),
                         "engine_seed": state["engine_seed"],
                         "seed": VERIFICATION_SEED, "expectation": expectation})
    if sum(1 for entry in resolved if entry["role"] == "positive") < 2:
        raise ValueError("the frozen verification set requires two positive controls")
    if sum(1 for entry in resolved if entry["role"] == "negative") < 2:
        raise ValueError("the frozen verification set requires two negative controls")
    return resolved


def verification_verdict(entries):
    """Apply the frozen pass rule to executed verification cell records."""
    details = []
    for entry, record in entries:
        outcome = record.get("engine_channel") or {}
        executed = (record.get("failure") is None
                    and outcome.get("bird_launches") == 1)
        consistency = None if not executed else channel_consistency(
            bool(outcome.get("pig_removed")), bool(outcome.get("pig_lifecycle_destroyed")))
        matched = executed and bool(outcome.get("pig_removed")) == entry["expectation"]
        details.append({
            "identity": record.get("cell", {}).get("identity") or entry["state"],
            "role": entry["role"], "state": entry["state"],
            "ordinal": entry["ordinal"], "branch_identity": entry["branch_identity"],
            "expectation": entry["expectation"],
            "executed": executed, "failure": record.get("failure"),
            "pig_removed": None if not executed else bool(outcome.get("pig_removed")),
            "pig_lifecycle_destroyed": None if not executed else bool(
                outcome.get("pig_lifecycle_destroyed")),
            "consistency": consistency, "expectation_matched": matched})
    valid = [detail for detail in details if detail["executed"]]
    anomalies = [detail["identity"] for detail in valid if detail["consistency"] == "anomaly"]
    negatives_ok = all(
        any(detail["state"] == entry["state"] and detail["expectation_matched"]
            for detail in valid if detail["role"] == "negative")
        for entry, _ in entries if entry["role"] == "negative")
    positives_ok = any(detail["expectation_matched"] for detail in valid
                       if detail["role"] == "positive")
    failures = [detail["identity"] for detail in details if not detail["executed"]]
    verified = (not failures and not anomalies and negatives_ok and positives_ok)
    blocker = None
    if not verified:
        if failures:
            blocker = (f"verification executions did not produce valid channel "
                       f"verdicts (typed failures retained): {failures}")
        elif anomalies:
            blocker = f"channel anomaly on {anomalies}: the event leg missed a state-stream pig destruction"
        elif not positives_ok:
            blocker = ("no positive control produced an engine-truth pig removal in "
                       "closed loop, so the pig-removal leg is not positively verified")
        else:
            blocker = "a negative control recorded an engine-truth pig removal"
    return {"verified": verified, "cells": details, "anomalies": anomalies,
            "failures": failures, "positives_matched": positives_ok,
            "negatives_matched": negatives_ok, "blocker": blocker,
            "pass_rule": VERIFICATION_PASS_RULE, "review_rule": VERIFICATION_REVIEW_RULE}


def apply_media_review(verdict, media_review):
    """Merge the owner's recorded review of the retained gallery media.

    media_review follows MEDIA_REVIEW_SCHEMA: one entry per control with
    pig_visible_in_final_frame (and optional notes). Positives must show the
    pig gone in the final retained frame, negatives the pig visible; the
    verdict is only verified when the game-state rule AND the media review
    both agree. Without a recorded review the verdict stays unverified with
    an explicit pending blocker.
    """
    verdict = dict(verdict)
    if media_review is None:
        verdict["media_review_pending"] = True
        verdict["media_review"] = None
        verdict["verified"] = False
        verdict["blocker"] = ("media review of the retained gallery is not "
                              "recorded yet; the channel verdict is not "
                              "verified against the retained rendered media")
        return verdict
    observations = {entry.get("identity"): entry
                    for entry in media_review.get("cells") or []}
    reviewed, disagreements, missing = [], [], []
    for detail in verdict["cells"]:
        identity = detail["identity"]
        observation = dict(observations.get(identity) or {})
        visible = observation.get("pig_visible_in_final_frame")
        if not detail["executed"] or visible is None:
            if detail["executed"]:
                missing.append(identity)
            observation.update({"identity": identity, "agrees": None})
            reviewed.append(observation)
            continue
        agrees = bool(visible) == (not detail["expectation"])
        observation["agrees"] = agrees
        if not agrees:
            disagreements.append(identity)
        reviewed.append(observation)
    verdict["media_review_pending"] = False
    verdict["media_review"] = {"schema": MEDIA_REVIEW_SCHEMA,
                               "rule": MEDIA_REVIEW_RULE,
                               "reviewed_by": media_review.get("reviewed_by"),
                               "reviewed_at_utc": media_review.get("reviewed_at_utc"),
                               "cells": reviewed}
    verdict["media_disagreements"] = disagreements
    verdict["media_missing"] = missing
    if disagreements or missing:
        verdict["verified"] = False
        if disagreements:
            verdict["blocker"] = (f"the recorded media review disagrees with the "
                                  f"channel verdict on {disagreements}")
        else:
            verdict["blocker"] = (f"the recorded media review lacks an observation "
                                  f"for {missing}")
    return verdict


# ------------------------------------------------------------------ the gate

def evaluate_gate(rows):
    """Pooled engine-truth first-shot success prevalence over the pilot cells."""
    valid = success = 0
    typed = []
    for row in rows:
        if row.get("failure") is not None or row.get("success") is None:
            typed.append({"identity": row["identity"], "failure": row.get("failure")})
            continue
        valid += 1
        success += int(bool(row["success"]))
    prevalence = success / valid if valid else 0.0
    return {"valid_executions": valid, "typed_failures": typed,
            "successes": success, "prevalence": prevalence,
            "floor": PREVALENCE_FLOOR,
            "minimum_valid_executions": PILOT_MINIMUM_VALID_EXECUTIONS,
            "passed": (prevalence >= PREVALENCE_FLOOR
                       and valid >= PILOT_MINIMUM_VALID_EXECUTIONS),
            "disposition_if_failed": "readiness_or_precision_insufficient"}


# ------------------------------------------------------------- estimand math

def normalized_regret(cost, outcomes):
    """Frozen #80 normalized regret semantics over the member's outcome table."""
    if not outcomes["informative"]:
        return 0.0 if outcomes["all_tied"] else None
    return (cost - outcomes["lowest_cost"]) / (outcomes["highest_cost"] - outcomes["lowest_cost"])


def paired_interval(values, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED):
    """Descriptive paired bootstrap (10,000 draws, seed 7201); no decision use."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    resamples = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "descriptive_95_percent_interval": np.quantile(resamples, [.025, .975]).tolist()}


def summarize_contrasts(per_state, seeds, systems=SYSTEMS,
                        tested="hybrid-fixed-h1",
                        references=("continuous-fixed-h1", "continuous-fixed-h5",
                                    PRIOR_SYSTEM),
                        metrics=("success", "regret")):
    """Paired differences of the tested system against each reference.

    Units are the 24 state identities; per-seed differences are averaged
    before resampling. A state pair with any missing outcome on either side
    contributes no difference (reported as paired_states).
    """
    result = []
    for reference in references:
        for metric in metrics:
            paired = []
            states = sorted(per_state[str(seeds[0])][tested])
            for state in states:
                differences = []
                for seed in seeds:
                    tested_row = per_state[str(seed)][tested][state]
                    reference_row = per_state[str(seed)][reference][state]
                    if tested_row[metric] is None or reference_row[metric] is None:
                        differences = None
                        break
                    # positive_is_improvement for every published contrast:
                    # success is higher-is-better (tested - reference), regret
                    # is lower-is-better (reference - tested).
                    if metric == "success":
                        differences.append(tested_row[metric] - reference_row[metric])
                    else:
                        differences.append(reference_row[metric] - tested_row[metric])
                if differences:
                    paired.append(float(np.mean(differences)))
            result.append({
                "tested": tested, "reference": reference, "metric": metric,
                "positive_is_improvement": True,
                "descriptive": paired_interval(paired),
                "paired_states": len(paired),
                "scope": ("descriptive paired bootstrap over state identities; "
                          "seeds averaged first"),
                "label": "DESCRIPTIVE"})
    return result


def disposition_for_primary(primary_contrast):
    """Q1 mapping: supported iff the pre-declared primary contrast has a mean
    above 0 AND its descriptive 95% interval entirely above 0 (the practical
    margin is the interval-exclusion rule itself)."""
    descriptive = primary_contrast.get("descriptive")
    if descriptive is None:
        return "not_supported_by_this_experiment", "no paired outcomes"
    low = descriptive["descriptive_95_percent_interval"][0]
    if descriptive["mean"] > 0 and low > 0:
        return "supported", "mean > 0 and descriptive interval entirely above 0"
    return "not_supported_by_this_experiment", (
        "mean/interval condition not met; negative or null differences are a "
        "valid descriptive outcome")


def proxy_cross_check(rows, threshold: float = PROXY_THRESHOLD):
    """Phase C: the #82 replay-cost proxy predicate against engine truth.

    rows: pilot cell rows with engine-truth success and the executed realized
    end-of-window count cost. Cells missing either side (typed failures) are
    excluded and counted. The proxy is a published, right-censored replay
    threshold, never re-fit; DESCRIPTIVE agreement only.
    """
    counts = {"both_success": 0, "proxy_only": 0, "engine_only": 0, "both_failure": 0}
    per_system = {}
    per_state_differences = {}
    excluded = 0
    for row in rows:
        success = row.get("success")
        cost = row.get("realized_count_cost")
        if success is None or cost is None or not math.isfinite(cost):
            excluded += 1
            continue
        proxy = bool(cost <= threshold)
        key = ("both_success" if proxy and success else
               "proxy_only" if proxy else
               "engine_only" if success else "both_failure")
        system = row["system"]
        bucket = per_system.setdefault(system, {"both_success": 0, "proxy_only": 0,
                                                "engine_only": 0, "both_failure": 0,
                                                "cells": 0})
        counts[key] += 1
        bucket[key] += 1
        bucket["cells"] += 1
        # Paired per-state differences (proxy minus engine truth; positive =
        # the proxy over-calls success): every compared system cell enters its
        # (state, seed) bucket; systems are averaged first, then seeds.
        state = row.get("state")
        seed = row.get("seed")
        if state is not None and seed is not None:
            per_state_differences.setdefault(state, {}).setdefault(seed, []).append(
                int(proxy) - int(bool(success)))
    compared = sum(counts.values())
    agreement = ((counts["both_success"] + counts["both_failure"]) / compared
                 if compared else None)
    paired = [float(np.mean([np.mean(cells) for cells in seeds.values()]))
              for seeds in per_state_differences.values()]
    return {"rule": ("proxy success = executed realized end-of-window count "
                     "cost <= the #82 published candidate threshold "
                     f"{threshold!r}; engine truth = the engine-side channel "
                     "verdict; DESCRIPTIVE"),
            "proxy_threshold": threshold,
            "proxy_threshold_source": (
                "issue-82-zero-floor-oracle-v1 summary.json "
                "binding.stage1.candidate_threshold (the least inclusive "
                "bound over the 8 observed replay-side pig removals)"),
            "cells_compared": compared, "cells_excluded": excluded,
            "counts": counts, "agreement": agreement,
            "paired_state_differences": {
                "descriptive": paired_interval(paired),
                "paired_states": len(paired),
                "unit": "state",
                "label": "DESCRIPTIVE",
                "positive_means": "proxy over-calls success relative to engine truth",
                "scope": ("descriptive paired bootstrap over pilot state "
                          "identities; seeds averaged first; frozen settings "
                          f"({BOOTSTRAP_DRAWS} draws, seed {BOOTSTRAP_SEED})")},
            "per_system": per_system}


def channel_agreement_table(rows):
    """Closed-loop event-leg vs lifecycle-leg agreement over executed cells."""
    table = {"agreement": 0, "death_at_window_edge": 0, "anomaly": 0}
    for row in rows:
        consistency = row.get("consistency")
        if consistency is not None:
            table[consistency] += 1
    return table


def parameter_counts(models):
    """Active-parameter counts per arm from loaded frozen predictors."""
    return {name: int(sum(parameter.numel() for parameter in model.parameters()))
            for name, model in models.items()}
