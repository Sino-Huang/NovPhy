"""Frozen regime-statistic machinery for issue-86 cross-family synthesis.

This module holds the pure computation for the cross-family regime-modulation
synthesis of the horizon-decay boundary (issue #86).  The runner
(``scripts/run_cross_family_regime_synthesis.py``) freezes the plan; this
module evaluates the frozen formulas against the published family artifacts:

- CLEVRER:   ``.local-artifacts/issue-79-clevrer-boundary-v1`` scene shards
             (12 scenes x 40 windows of 61 annotation frames).
- Physion:   ``.local-artifacts/issue-84-third-family-v1`` scene shards
             (12 trials x 40 windows of 61 released frames at dt=0.01 s).
- NovPhy:    ``.local-artifacts/issue-77-n1-diagnostic-v1`` target records
             (4 held-out states x 13 accepted fixed-action branches; the
             branch is the evaluation window, observed frames 0..600 at
             50 native steps per observed frame).

Identical event-statistic formulas are evaluated over each family's frozen
event timelines and carrier records.  Families differ in native event
semantics and sampling cadence; those differences are carried as typed
per-family disclosure fields, never averaged away, and no unavailable field
is imputed (typed ``None`` plus a reason string instead).
"""
from __future__ import annotations

import numpy as np

DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")

# Shared 236-value carrier layout (issue-71 layout, unchanged across families:
# 2 global values plus 18 slots of 13 columns).
CARRIER_GLOBALS = 2
SLOTS = 18
SLOT_COLUMNS = 13
CARRIER_DIM = CARRIER_GLOBALS + SLOTS * SLOT_COLUMNS
PRESENCE_COLUMN = 0
MOTION_COLUMNS = (8, 9)

# Typed numeric floor for the velocity-decay denominator.
SPEED_EPS = 1e-12

STATISTIC_NAMES = ("contact_active_fraction", "post_settlement_fraction",
                   "collision_event_rate", "velocity_decay_ratio")

RULE_DIRECTIONS = ("increasing", "decreasing")
RULE_OUTCOMES = ("consistent", "inverted", "unresolved")


# --------------------------------------------------------------------------
# carrier helpers
# --------------------------------------------------------------------------

def slot_view(carrier):
    """(..., 236) carrier -> (..., 18, 13) per-slot view."""
    carrier = np.asarray(carrier, dtype=np.float64)
    if carrier.shape[-1] != CARRIER_DIM:
        raise ValueError(f"carrier has {carrier.shape[-1]} columns, expected {CARRIER_DIM}")
    return carrier[..., CARRIER_GLOBALS:].reshape(*carrier.shape[:-1], SLOTS, SLOT_COLUMNS)


def carrier_speed(carrier):
    """Mean motion-channel speed over present slots of one carrier (scale-free within a family)."""
    slots = slot_view(carrier)
    present = slots[..., PRESENCE_COLUMN] > 0
    if not present.any():
        return 0.0
    speed = np.sqrt((slots[..., MOTION_COLUMNS] ** 2).sum(axis=-1))
    return float(speed[present].mean())


# --------------------------------------------------------------------------
# frozen event-statistic formulas (identical shape for every family)
# --------------------------------------------------------------------------

def window_event_statistics(event_frames, window_start, window_length, settlement_frame):
    """contact-active fraction, post-settlement fraction, collision-event rate.

    ``event_frames``: per-record collision frame indices of the unit's frozen
    timeline (one entry per native record; family semantics disclosed in the
    plan).  ``window_length`` counts frame slots, inclusive of both ends.
    ``settlement_frame`` is the last collision frame of the unit's full frozen
    timeline, or -1 when the timeline carries no collision record (the unit is
    then typed ``no_collision_events`` and its whole window counts as
    post-settlement: nothing is imputed, the degenerate case is retained).
    """
    window_frames = range(window_start, window_start + window_length)
    in_window = [f for f in event_frames if window_start <= f < window_start + window_length]
    n = window_length
    contact_active_fraction = len(set(in_window)) / n
    if settlement_frame < 0:
        post_settlement_fraction = 1.0
    else:
        post_settlement_fraction = sum(1 for f in window_frames if f > settlement_frame) / n
    collision_event_rate = len(in_window) / n
    return {"contact_active_fraction": contact_active_fraction,
            "post_settlement_fraction": post_settlement_fraction,
            "collision_event_rate": collision_event_rate,
            "no_collision_events": settlement_frame < 0,
            "event_records_in_window": len(in_window),
            "distinct_event_frames_in_window": len(set(in_window))}


def window_velocity_profile(speeds):
    """Frozen velocity-decay profile from a window's ordered per-record speeds.

    ``speeds``: mean-per-record carrier motion speed over the window's frozen
    carrier records, in family carrier units (the decay ratio divides two
    speeds of the same window, so each family's carrier velocity scale
    cancels).  Returns the typed profile; when no record of the window carries
    motion the ratio is retained as ``None`` with reason ``no_motion_in_window``
    and the raw speeds are kept verbatim.
    """
    if len(speeds) == 0:
        raise ValueError("velocity profile requires at least one carrier record")
    first, last, peak = float(speeds[0]), float(speeds[-1]), max(float(s) for s in speeds)
    ratio = last / peak if peak > SPEED_EPS else None
    return {"speed_first": first, "speed_last": last, "speed_peak": peak,
            "velocity_decay_ratio": ratio,
            "typed_reason": None if ratio is not None else "no_motion_in_window"}


# --------------------------------------------------------------------------
# descriptive window bootstrap
# --------------------------------------------------------------------------

def descriptive_interval(values, draws, seed):
    """Descriptive percentile bootstrap interval of the mean over windows."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("descriptive_interval requires a non-empty sample")
    rng = np.random.Generator(np.random.PCG64(seed))
    picks = rng.integers(0, values.size, size=(draws, values.size))
    means = values[picks].mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(means, [0.025, 0.975]))
    return {"mean": float(values.mean()), "ci_low": lo, "ci_high": hi, "n": int(values.size)}


# --------------------------------------------------------------------------
# frozen monotone-modulation rules
# --------------------------------------------------------------------------

def rule_outcome(novphy, physion, clevrer, direction, margin):
    """Outcome of one frozen single-variable monotone rule.

    NovPhy is the only family with all three signature components present
    (presence count 3); Physion (S1 absent) and CLEVRER (S3 absent) both
    present two, so the rule constrains only the NovPhy-versus-pair ordering:
    under ``increasing`` the rule is consistent iff
    ``novphy > max(pair) + margin`` and inverted iff
    ``novphy < min(pair) - margin``; ``decreasing`` mirrors.  The pair tie is
    unconstrained by construction.
    """
    if direction not in RULE_DIRECTIONS:
        raise ValueError(f"unknown rule direction {direction!r}")
    others = (physion, clevrer)
    if direction == "increasing":
        consistent = novphy > max(others) + margin
        inverted = novphy < min(others) - margin
    else:
        consistent = novphy < min(others) - margin
        inverted = novphy > max(others) + margin
    if consistent:
        return "consistent"
    if inverted:
        return "inverted"
    return "unresolved"


def falsification_record(novphy, physion, clevrer, direction, margin):
    """Typed fourth-family prediction intervals of one frozen rule.

    A fourth family with presence count ``c4`` and rule value ``x4`` is
    consistent with the rule iff, against every existing family F: ``x4`` is
    on the count-correct side of ``x_F`` by more than the frozen margin when
    the counts differ, and unconstrained when the counts tie.  The returned
    intervals are the frozen consistency regions per ``c4``; a measured ``x4``
    outside its region falsifies the rule.
    """
    high, low = max(physion, clevrer), min(physion, clevrer)
    if direction == "increasing":
        # more components <=> higher value; count ties constrain nothing
        c4_at_least_3 = (high + margin, float("inf"))       # c4 = 3: above the count-2 pair
        c4_at_most_1 = (float("-inf"), min(novphy, low) - margin)   # below every family
        c4_exactly_2 = (float("-inf"), novphy - margin)     # c4 = 2: below NovPhy (count 3)
    else:
        # more components <=> lower value; count ties constrain nothing
        c4_at_least_3 = (float("-inf"), low - margin)       # c4 = 3: below the count-2 pair
        c4_at_most_1 = (max(novphy, high) + margin, float("inf"))   # above every family
        c4_exactly_2 = (novphy + margin, float("inf"))      # c4 = 2: above NovPhy (count 3)
    return {"count_3": c4_at_least_3, "count_2": c4_exactly_2, "count_le_1": c4_at_most_1}


def question_1_disposition(primary_outcome, primary_computable):
    """Frozen mapping from the primary rule outcome to a disposition token."""
    if not primary_computable:
        return "readiness_or_precision_insufficient"
    return "supported" if primary_outcome == "consistent" else "not_supported_by_this_experiment"


def question_2_disposition(rule_outcomes):
    """Frozen mapping from all single-variable rule outcomes to a disposition token."""
    computable = [outcome for outcome in rule_outcomes if outcome is not None]
    if not computable:
        return "readiness_or_precision_insufficient"
    return "supported" if "consistent" in computable else "not_supported_by_this_experiment"


# --------------------------------------------------------------------------
# frozen per-family window enumeration (the scheduled evidence cells)
# --------------------------------------------------------------------------

def _window_statistics_row(cell_id, event_frames, window_start, window_length,
                           settlement_frame, speeds, extra=None):
    row = {"cell_id": cell_id, "status": "computed",
           **window_event_statistics(event_frames, window_start, window_length, settlement_frame),
           **window_velocity_profile(speeds), "speed_records": len(speeds)}
    if extra:
        row.update(extra)
    return row


def clevrer_window_rows(shard):
    """40 evaluation windows of one CLEVRER scene shard (starts 0..39, 61 frames)."""
    windows = shard["windows"]["z"]
    if tuple(windows.shape) != (40, 61, CARRIER_DIM):
        raise ValueError(f"unexpected CLEVRER window tensor shape {tuple(windows.shape)}")
    events = [int(f) for f in shard["collision_frames"].tolist()]
    settlement = max(events) if events else -1
    rows = []
    for start in range(40):
        speeds = [carrier_speed(windows[start, frame]) for frame in range(61)]
        rows.append(_window_statistics_row(
            f"scene-{int(shard['scene_index'])}:window-{start:02d}",
            events, start, 61, settlement, speeds))
    return rows


def physion_window_rows(shard):
    """40 evaluation windows of one Physion shard (frozen evenly-spread starts)."""
    from world_model.training import third_family_boundary as family84
    windows = shard["windows"]["z"]
    frames = int(shard["clip"].shape[0])
    starts = family84.window_starts(frames)
    if len(starts) != windows.shape[0]:
        raise ValueError("stored Physion window count differs from the frozen start rule")
    events = [int(f) for f in shard["collision_frames"].tolist()]
    settlement = max(events) if events else -1
    rows = []
    for index, start in enumerate(starts):
        speeds = [carrier_speed(windows[index, frame]) for frame in range(61)]
        rows.append(_window_statistics_row(
            f"trial-{int(shard['scene_index'])}:window-{index:02d}",
            events, int(start), 61, settlement, speeds,
            {"window_start_frame": int(start)}))
    return rows


NOVPHY_NATIVE_STEPS_PER_FRAME = 50
NOVPHY_WINDOW_START = 0
NOVPHY_WINDOW_END = 600  # inclusive observed-frame endpoint (task window, 30000 native steps)


def novphy_window_rows(target_document):
    """One evaluation window per accepted #77 normal-mechanics branch.

    The branch IS the published evaluation window (observed frames 0..600,
    right-censored absorption beyond the recording exactly as #77 declared).
    Carrier speeds exist only at the frozen parsed offsets; the profile is
    computed over those records and the sparse cadence is disclosed per row.
    """
    state = str(target_document["state"]).rsplit("-", 1)[-1]
    rows = []
    for candidate in target_document["candidates"]:
        source = candidate["source"]
        cell_id = f"state-{state}:candidate-{source['ordinal']:02d}"
        if candidate.get("status") != "available":
            rows.append({"cell_id": cell_id, "status": "typed_terminal_failure",
                         "reason": f"candidate status {candidate.get('status')!r}"})
            continue
        carriers = candidate["carriers"]
        offsets = sorted((int(o) for o in carriers), key=int)
        if offsets[0] != NOVPHY_WINDOW_START or offsets[-1] != NOVPHY_WINDOW_END:
            rows.append({"cell_id": cell_id, "status": "typed_terminal_failure",
                         "reason": "branch window endpoints 0/600 missing from frozen carriers"})
            continue
        timing = candidate["timing"]
        events = [int(step) // NOVPHY_NATIVE_STEPS_PER_FRAME
                  for step in timing["events"]["collision"]]
        settlement = max(events) if events else -1
        speeds = [carrier_speed(carriers[str(offset)]) for offset in offsets]
        rows.append(_window_statistics_row(
            cell_id, events, NOVPHY_WINDOW_START, NOVPHY_WINDOW_END + 1, settlement, speeds,
            {"absorbed_endpoint": NOVPHY_WINDOW_END in set(candidate.get("absorbed_offsets", ())),
             "censored": bool(timing.get("censored")),
             "last_offset": int(timing.get("last_offset", -1))}))
    return rows


def family_intervals(rows, statistic, draws, seed):
    """Descriptive bootstrap interval of one statistic over one family's computed cells."""
    values = [row[statistic] for row in rows
              if row.get("status") == "computed" and row.get(statistic) is not None]
    typed = [row for row in rows if row.get("status") != "computed"]
    typed_no_motion = [row for row in rows
                       if row.get("status") == "computed" and row.get(statistic) is None]
    unavailable = len(typed) + len(typed_no_motion)
    if not values:
        return None, unavailable
    return descriptive_interval(values, draws, seed), unavailable
