"""Issue-86 ADD-EXP: cross-family regime-modulation synthesis of the horizon-decay boundary.

Binding runner module: scripts/run_cross_family_regime_synthesis.py.
Functional module: world_model/training/cross_family_regime_synthesis.py.
Exact validation command: python -u -m scripts.run_cross_family_regime_synthesis --validate

Analysis-scale ticket over published artifacts only (no new training, no new
captures, no checkpoint retraining, CPU-only computation).  Per the shared GPU
timing contract the exclusive flock on /tmp/novphy-addexp-gpu.lock is acquired
BEFORE the wall timer starts and held for the ENTIRE wall-time-measured phase
(--run/--publish/--validate) even though no GPU work exists here; --dry-run and
--prepare publish no wall time and never take the lock.

Inputs (READ-ONLY):
- .local-artifacts/issue-79-clevrer-boundary-v1   CLEVRER scene shards (12 x 40 windows)
- .local-artifacts/issue-84-third-family-v1       Physion scene shards   (12 x 40 windows)
- .local-artifacts/issue-77-n1-diagnostic-v1      NovPhy normal-mechanics branch targets
- .local-artifacts/issue-79-clevrer-boundary-v1/summary.json    published S1/S2/S3 outcomes
- .local-artifacts/issue-84-third-family-v1/summary.json        published S1/S2/S3 outcomes
- .local-artifacts/issue-77-n1-diagnostic-v1/findings.md        published NovPhy contrasts
- .local-artifacts/issue-83-clevrer-nonmonotonicity-v1/findings.md  published localization

Modes (mutually exclusive):
--dry-run    no-write inventory: input readability, scheduled cell counts, timing sample
--prepare    freeze plan.json (inputs bound to identifiers the artifacts already carry --
             schema/identity/plan_identity, never new content hashes; identical statistic
             formulas; typed event-semantics disclosures; component table from published
             dispositions; monotone rules + numeric margins; bootstrap contract;
             disposition mappings); no regime statistic is computed at freeze time
--run        execute every scheduled family-window cell (deterministic per-family resume
             checkpoints, foreground progress with ETA), then bootstrap + frozen rules;
             writes compute.json; GPU lock held for the whole measured phase
--publish    transform compute.json into summary.json, comparisons.csv, findings.md and the
             machine-readable regime-modulation table regime_modulation.csv; GPU lock held
--validate   recompute every published table from the frozen inputs (fresh cell computation
             and fresh frozen-seed bootstrap) and byte-compare against the published
             artifacts; GPU lock held

Prior dispositions (#77, #79, #83, #84, #81) are inputs and are never recomputed or
amended; the #64/#65 sealed benchmark stays untouched; zero-shot and adapted conditions
do not exist here (no model is run), and no cross-family effect-size pooling happens.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import time
from pathlib import Path

import torch

from world_model.training import cross_family_regime_synthesis as regime

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-86-cross-family-regime-v1"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"
DERIVED_BYTES_BUDGET = 1024 ** 3

SCHEMA = "issue_86_regime_synthesis_plan_v1"
IDENTITY = "issue-86-cross-family-regime-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_cross_family_regime_synthesis --validate"

FAMILY_ORDER = ("novphy_normal_mechanics", "physion_dominoes", "clevrer")

# ---------------------------------------------------------------------------
# frozen component-presence table (published dispositions; never recomputed)
# ---------------------------------------------------------------------------

COMPONENT_DEFINITIONS = (
    "S1 training-effect position_mse difference (reference minus tested, positive favors "
    "hybrid) is positive at h=1, endpoint 15, and exceeds 5% of the continuous-arm mean "
    "position_mse at h=1 endpoint 15",
    "S2 the h=15 training-effect difference is smaller than the h=1 difference at endpoint "
    "15 (separation shrinks or reverses with horizon)",
    "S3 continuous-arm recursive position_mse grows monotonically across endpoints "
    "15,30,60,120 at every horizon",
)

COMPONENT_TABLE = {
    "novphy_normal_mechanics": {
        "S1": "present", "S2": "present", "S3": "present", "presence_count": 3,
        "published_source": ("issue-77-n1-diagnostic-v1 findings: h=1 training effect +0.0887 "
                             "[+0.0155, +0.2118]; h=5/h=15 training-effect intervals include "
                             "zero (S2); claim nm-recursive-growth in the issue-81 registry "
                             "records the recursive growth analog (S3)")},
    "physion_dominoes": {
        "S1": "absent", "S2": "present", "S3": "present", "presence_count": 2,
        "published_source": ("issue-84-third-family-v1 summary.json: S1 difference "
                             "6.471482791110047e-05 not above the frozen relative-5% margin "
                             "2.4976621360919024e-04; S2 and S3 hold")},
    "clevrer": {
        "S1": "present", "S2": "present", "S3": "absent", "presence_count": 2,
        "published_source": ("issue-79-clevrer-boundary-v1 summary.json: S1 holds (difference "
                             "0.003663336865025081 above margin 0.0010093385783774365), S2 "
                             "holds, S3 fails (h=15 curve 0.0013 -> 0.0017 -> 0.0013 -> "
                             "0.0116 is non-monotone); issue-83 localizes the failure to "
                             "post-settlement segments (rho 0.777 [0.611, 0.975] below the "
                             "frozen 0.80 uniformity bound)")},
}

# ---------------------------------------------------------------------------
# frozen family contracts (window enumeration + typed event semantics)
# ---------------------------------------------------------------------------

CLEVRER_SCENES = tuple(range(10000, 10012))
PHYSION_TRIALS = tuple(range(12))
NOVPHY_STATES = ("007", "008", "014", "016")
NOVPHY_CANDIDATES_PER_STATE = 13

FAMILIES = {
    "novphy_normal_mechanics": {
        "root": ".local-artifacts/issue-77-n1-diagnostic-v1",
        "targets": tuple(f"targets/state-{state}.json" for state in NOVPHY_STATES),
        "source_identity": "issue-77-n1-diagnostic-v1",
        "membership_binding": (
            "the four target documents ARE the published #77 normal-mechanics held-out "
            "evaluation membership: definitions.membership of issue-77-n1-diagnostic-v1/"
            "summary.json reads 'held_out_evaluation lineages with >=1 coverage-admissible "
            "branch, sorted by identity', its inventory records target_states=4 and "
            "fixed_candidate_records=1872 (= 4 states x 13 coverage-admissible branches x "
            "36 system-seed records), and every published paired contrast contributes "
            "n=4 state rows; the branches analyzed here are exactly those held-out states "
            "(issue-77-n1-007/008, -014/-016) x their 13 accepted fixed-action candidate "
            "branches; no development or training branch is substituted"),
        "cell_unit": "accepted fixed-action branch of the frozen #77 diagnostic",
        "window_frames": ("observed frames 0..600 inclusive (601 frame slots; 1 observed "
                          "frame = 50 native fixed steps; right-censored branches reuse the "
                          "terminal-frame carrier past the recording, exactly as #77 "
                          "declared)"),
        "scheduled_cells": len(NOVPHY_STATES) * NOVPHY_CANDIDATES_PER_STATE,
        "event_indicator_semantics": ("engine collision events of the branch's native trace, "
                                      "stored relative to first_fixed_step and mapped to "
                                      "observed frames by integer division by 50; one record "
                                      "per discrete collision onset"),
        "event_record_units": "collision onsets",
        "velocity_records": ("carrier motion channels (slots columns 8,9) of the frozen #77 "
                             "parsed carriers at the frozen parsed offsets of the diagnostic "
                             "(including 0 and 600; sparse cadence), endpoint carriers "
                             "absorbed at the terminal frame for right-censored branches"),
        "frame_cadence": "50 native fixed steps per observed frame",
    },
    "physion_dominoes": {
        "root": ".local-artifacts/issue-84-third-family-v1",
        "shards": tuple(f"shards/scene-{index}.pt" for index in PHYSION_TRIALS),
        "source_identity": "physion-dominoes-annotation-feature-carrier-v1",
        "membership_binding": (
            "the twelve scene shards ARE the frozen #84 membership (scene-0..11 in stored "
            "order; schema issue_84_physion_scene_v1; the published #84 plan lists the same "
            "twelve dominoes trials with windows_per_trial 40), 40 windows each"),
        "cell_unit": "(trial, start) window of the frozen #84 shard",
        "window_frames": ("61 released frames [start, start+60]; 40 evenly spread starts "
                          "spanning [0, frames-61] per the frozen #84 start rule"),
        "scheduled_cells": len(PHYSION_TRIALS) * 40,
        "event_indicator_semantics": ("TDW per-frame object-object collision records of the "
                                      "released annotations: any recorded enter/stay/exit "
                                      "state labels that frame, so consecutive frames recur "
                                      "during persistent contact"),
        "event_record_units": ("per-frame per-pair contact records (enter/stay/exit); NOT "
                               "onset events"),
        "velocity_records": ("carrier motion channels (slots columns 8,9) of the frozen #84 "
                             "window tensors: 61 per-frame records per window"),
        "frame_cadence": "1 released frame = 0.01 s",
    },
    "clevrer": {
        "root": ".local-artifacts/issue-79-clevrer-boundary-v1",
        "shards": tuple(f"shards/scene-{index}.pt" for index in CLEVRER_SCENES),
        "source_identity": "clevrer-annotation-feature-carrier-v1",
        "membership_binding": (
            "the twelve scene shards ARE the frozen #79 membership (scenes 10000..10011, "
            "the first 12 collision-complete validation-split scenes of the published #79 "
            "membership rule; schema issue_79_clevrer_scene_v1), 40 windows each"),
        "cell_unit": "(scene, start) window of the frozen #79 shard",
        "window_frames": ("61 annotation frames [start, start+60]; starts 0..39 per the "
                          "frozen #79 window rule"),
        "scheduled_cells": len(CLEVRER_SCENES) * 40,
        "event_indicator_semantics": ("CLEVRER annotation collision onsets: one record per "
                                      "annotated collision-event frame"),
        "event_record_units": "collision onsets",
        "velocity_records": ("carrier motion channels (slots columns 8,9) of the frozen #79 "
                             "window tensors: 61 per-frame records per window"),
        "frame_cadence": "1 annotation frame (native CLEVRER annotation cadence)"},
}

EVENT_INDICATOR_DISCLOSURE = (
    "Typed incompatibility disclosure (frozen): the collision-event indicator is NOT the "
    "same native quantity in all three families. CLEVRER and NovPhy record discrete "
    "collision onsets; Physion records per-frame per-pair contact records including "
    "enter/stay/exit states, so Physion contact-active fractions and collision-event rates "
    "are contact-duration-like while the other two families provide onset lower bounds. "
    "Cross-family comparisons of those two statistics are rate-of-record comparisons under "
    "this disclosure and are never read as identical onset indicators. Window lengths are "
    "family frame units (601 observed NovPhy frames, 61 CLEVRER annotation frames, 61 "
    "Physion released frames) and are never equated across families; the velocity-decay "
    "ratio is scale-free within each window because it divides two speeds of the same "
    "window in the same family's carrier units, but NovPhy speeds exist only at the frozen "
    "parsed offsets."
)

# ---------------------------------------------------------------------------
# frozen monotone-modulation rules (directions grounded BEFORE any outcome)
# ---------------------------------------------------------------------------

RULES = (
    {"statistic": "post_settlement_fraction", "role": "primary",
     "direction": "decreasing", "margin": 0.10,
     "grounding": ("issue-83 localized the CLEVRER S3 failure to post-settlement segments "
                   "(rho 0.777 [0.611, 0.975] below the frozen 0.80 uniformity bound): "
                   "windows dominated by post-settlement frames mask recursive-error "
                   "growth, so the all-components family is predicted to hold the LOW "
                   "post-settlement extreme")},
    {"statistic": "contact_active_fraction", "role": "secondary",
     "direction": "increasing", "margin": 0.10,
     "grounding": ("dual of the post-settlement axis at window level: the all-components "
                   "family is predicted to hold the HIGH contact-active extreme")},
    {"statistic": "collision_event_rate", "role": "secondary",
     "direction": "increasing", "margin": 0.05,
     "grounding": ("issue-84's answered-positive regime question (contact activity "
                   "predicts where the micro symbolic-execution effect helps) motivates "
                   "the denser-contact direction; the typed indicator-semantics "
                   "disclosure applies to every cross-family reading")},
    {"statistic": "velocity_decay_ratio", "role": "secondary",
     "direction": "decreasing", "margin": 0.10,
     "grounding": ("settled windows decay harder from their within-window speed peak; "
                   "the same S3-masking mechanism as the primary axis predicts the "
                   "all-components family holds the low extreme")},
)

PRIMARY_RULE = "post_settlement_fraction"
BOOTSTRAP = {"draws": 10000, "level": 0.95, "unit": "window (NovPhy: branch); descriptive",
             "seeds": {"novphy_normal_mechanics": 86077, "physion_dominoes": 86841,
                       "clevrer": 86079},
             "aggregation": ("per-family descriptive percentile bootstrap of the mean over "
                             "that family's windows where the statistic is computable; "
                             "windows typed unavailable for a statistic (status "
                             "typed_terminal_failure, or velocity_decay_ratio None with "
                             "reason no_motion_in_window) are excluded from that "
                             "statistic's interval and counted as typed unavailable in "
                             "every published table; a family with zero computable windows "
                             "for a statistic leaves that rule uncomputable, which maps to "
                             "readiness_or_precision_insufficient under the frozen "
                             "disposition rules")}

DISPOSITION_RULES = {
    "question_1_regime_modulation": (
        "readiness_or_precision_insufficient - the ticket's question-1 asks whether the "
        "frozen regime statistics separate the three families in the same order as the "
        "component-presence table, whose three rows are DISTINCT patterns (NovPhy all "
        "present; Physion S1 absent; CLEVRER S3 absent). The frozen monotone rules "
        "operationalized only presence-count ordering (NovPhy extreme; the "
        "Physion/CLEVRER count-2 tie unconstrained) - a freeze-time SCOPE ERROR, not a "
        "measured answer - and never test which component fails; the primary statistic's "
        "settlement anchor is additionally not semantically identical across families "
        "(last collision onset vs last enter/stay/exit record). A component-resolved, "
        "onset-comparable rule was never operationalized at freeze; scoring one now would "
        "require a post-outcome re-freeze, which the shared rules prohibit"),
    "question_2_predictability_bound": (
        "readiness_or_precision_insufficient - the ticket's question-2 asks whether ANY "
        "frozen single-variable rule is consistent with the observed component-presence "
        "ordering (which includes which component fails) and what would falsify it. The "
        "frozen consistency semantics were count-only (the same freeze-time scope error), "
        "so the question as posed was never operationalized; a component-resolved "
        "consistency reading cannot be scored post-outcome without a prohibited re-freeze"),
    "question_1_regime_modulation_frozen_rule_exploratory": (
        "the frozen count-rule outcome, retained VERBATIM as an exploratory result that "
        "does not answer the ticket's question: supported iff the primary rule "
        "(post_settlement_fraction, decreasing, margin 0.10) is consistent under "
        "presence-count semantics; not_supported_by_this_experiment when it inverts or "
        "stays within the frozen margin"),
    "question_2_predictability_bound_frozen_rule_exploratory": (
        "the frozen count-rule outcome, retained VERBATIM as an exploratory result that "
        "does not answer the ticket's question: supported iff at least one frozen "
        "single-variable rule with computable family intervals is consistent under "
        "count-only semantics"),
}

CLAIM_BOUNDARY = (
    "cross-family synthesis of published diagnostics: no new training, no new captures, no "
    "new physics family; the #77/#79/#83/#84 dispositions are inputs and are never "
    "recomputed or amended; nothing reopens the #64/#65 sealed benchmark or the #76 "
    "advancement gate; three families support ordinal consistency at most - no causal or "
    "universal-law claim; intervals are DESCRIPTIVE window bootstraps; no cross-family "
    "effect-size pooling; no fourth family is measured here")

# Post-audit typed record (added 2026-09-22 after the parent acceptance audit and a
# read-only reviewer consultation; the frozen plan, every frozen disposition, and every
# computed table are untouched - this record adds additive typed context, it re-scores
# nothing and changes no frozen number, outcome, or disposition).
POST_AUDIT_PROTOCOL_RECORD = {
    "date": "2026-09-22",
    "protocol_deviation": (
        "the 2026-09-21 --dry-run scored 93 real cells (one CLEVRER shard, one Physion "
        "shard, one NovPhy target) for a wall-time ETA sample BEFORE --prepare froze the "
        "plan; freeze-before-any-outcome is therefore not claimable as written. The dry "
        "run also measured and emitted wall seconds WITHOUT the shared flock the USER "
        "requires for every wall-time-measured phase; the plan's lock-free declaration "
        "for the dry-run could not waive that user requirement, so the missing lock IS "
        "part of the deviation. The sample was discarded unused, never published, and no "
        "rule direction, margin, or membership choice derived from it - the rules were "
        "authored from the published #83 localization and #84 regime answer before the "
        "deviation occurred; all published numbers were computed under the frozen plan "
        "after --prepare. The scoring/timing behavior was removed from --dry-run "
        "prospectively (structural inventory remains); this record is the typed "
        "disclosure, not a refreeze"),
    "rule_expressiveness": (
        "the frozen monotone rules test presence-count ordering only (NovPhy extreme; the "
        "Physion count-2 / CLEVRER count-2 tie unconstrained) and never test which "
        "component fails, so the issue's question-1 ordering over three DISTINCT "
        "component-presence patterns was never operationalized at freeze; the gap was "
        "also not disclosed in the plan limitations or findings at freeze time. Recorded "
        "as a canonical ticket blocker (readiness_or_precision_insufficient); the raw "
        "count-rule tokens stand verbatim under exploratory_frozen_rule_dispositions and "
        "were not downgraded or amended"),
    "cross_family_comparability": (
        "the frozen event-indicator disclosure scoped itself to contact_active_fraction "
        "and collision_event_rate ('those two statistics'), so post_settlement_fraction - "
        "the PRIMARY rule's statistic - was left uncompensated even though its settlement "
        "anchor inherits the same semantics gap: the anchor is the last collision ONSET "
        "for CLEVRER/NovPhy but the last enter/stay/exit contact record for Physion, "
        "whose stay states extend the anchor later and bias Physion's "
        "post_settlement_fraction (0.045936) downward relative to onset semantics; an "
        "onset-comparable value could plausibly change the primary outcome. The primary "
        "rule's cross-family reading is therefore comparability-compromised; no "
        "onset-corrected re-scoring is performed (that would be a post-outcome re-freeze). "
        "Within-family descriptives are unaffected"),
}

LIMITATIONS = (
    EVENT_INDICATOR_DISCLOSURE,
    "NovPhy velocity records exist only at the frozen parsed offsets (sparse cadence); "
    "speed peaks there are lower bounds and the decay ratio is computed over available "
    "records with the cadence typed per row",
    "windows within a family share scenes/trials/states (clustered units); the bootstrap "
    "resamples windows and is descriptive only",
    "with three families only ordinal consistency is testable; the falsification record is "
    "a typed descriptive prediction for future tickets, not a measured result",
    "zero-shot and few-shot/adapted conditions do not arise (no model runs in this ticket); "
    "no condition pooling exists by construction",
)


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------

def log(message):
    print(f"[issue-86-regime-synthesis] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


DOCUMENT_BINDINGS = (
    # (repo-relative path, machine identifier verified at read time or None for
    #  prose artifacts that are existence-checked and cited verbatim)
    (".local-artifacts/issue-79-clevrer-boundary-v1/summary.json",
     "issue-79-clevrer-boundary-report-v1"),
    (".local-artifacts/issue-84-third-family-v1/summary.json",
     "issue-84-third-family-report-v1"),
    (".local-artifacts/issue-81-boundary-synthesis-v1/summary.json",
     "issue-81-boundary-synthesis-report-v1"),
    (".local-artifacts/issue-77-n1-diagnostic-v1/findings.md", None),
    (".local-artifacts/issue-83-clevrer-nonmonotonicity-v1/findings.md", None),
)


def declared_sources():
    """Binding of every frozen input to identifiers ALREADY carried by the artifact.

    No new content hashes are computed anywhere in this runner (shared ADD-EXP
    operator rule): a source is verified at read time by checking the schema/
    identity/plan_identity fields the published artifacts already carry; prose
    documents without a machine identifier are existence-checked and cited
    verbatim as published provenance.
    """
    pinned = []
    for family_name in FAMILY_ORDER:
        spec = FAMILIES[family_name]
        for relative in (*spec.get("targets", ()), *spec.get("shards", ())):
            pinned.append({"family": family_name, "artifact": relative,
                           "source_identity": spec["source_identity"],
                           "verified_at_read_time": True})
    for path, identity in DOCUMENT_BINDINGS:
        pinned.append({"family": "published_dispositions", "artifact": path,
                       "source_identity": identity or "published prose document",
                       "verified_at_read_time": identity is not None})
    return pinned


def verify_document_bindings(plan):
    """Read-time verification of the document bindings that carry machine identities."""
    for entry in plan["inputs"]:
        if not entry["verified_at_read_time"]:
            if not (ROOT / entry["artifact"]).exists():
                raise ValueError(f"bound prose document missing: {entry['artifact']}")
            continue
        if entry["family"] == "published_dispositions":
            document = read_json(ROOT / entry["artifact"])
            if document.get("identity") != entry["source_identity"]:
                raise ValueError(f"source identity differs in {entry['artifact']}")


def verify_source_bindings(plan):
    """Cheap structural check of the frozen plan against the module's declared bindings."""
    declared = [(entry["family"], entry["artifact"], entry["source_identity"])
                for entry in declared_sources()]
    frozen = [(entry["family"], entry["artifact"], entry["source_identity"])
              for entry in plan["inputs"]]
    if frozen != declared:
        raise ValueError("frozen plan input bindings differ from the declared source bindings")


class GPULock:
    """Exclusive advisory lock held for an entire wall-time-measured phase."""

    def __init__(self):
        self._handle = None

    def __enter__(self):
        self._handle = open(GPU_LOCK_PATH, "a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exception):
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
        return False


# ---------------------------------------------------------------------------
# cell execution
# ---------------------------------------------------------------------------

def _torch_shard(path, expected_identity):
    shard = torch.load(path, map_location="cpu", weights_only=True)
    schema = shard.get("schema", "")
    if shard.get("identity") != expected_identity:
        raise ValueError(f"source identity differs in {path}: {shard.get('identity')!r}")
    if "clevrer" in schema:
        rows = regime.clevrer_window_rows(shard)
    elif "physion" in schema:
        rows = regime.physion_window_rows(shard)
    else:
        raise ValueError(f"unknown shard schema {schema!r} in {path}")
    return rows


def scheduled_cell_ids():
    """The frozen scheduled-cell id set per family."""
    scheduled = {}
    for scene in CLEVRER_SCENES:
        scheduled.setdefault("clevrer", set()).update(
            f"scene-{scene}:window-{start:02d}" for start in range(40))
    for trial in PHYSION_TRIALS:
        scheduled.setdefault("physion_dominoes", set()).update(
            f"trial-{trial}:window-{index:02d}" for index in range(40))
    for state in NOVPHY_STATES:
        scheduled.setdefault("novphy_normal_mechanics", set()).update(
            f"state-{state}:candidate-{ordinal:02d}" for ordinal in range(NOVPHY_CANDIDATES_PER_STATE))
    return scheduled


def read_novphy_target(path, state):
    """Read one #77 target document, verifying its plan identity and state label."""
    document = read_json(path)
    if document.get("plan_identity") != FAMILIES["novphy_normal_mechanics"]["source_identity"]:
        raise ValueError(f"source identity differs in {path}")
    if not str(document.get("state", "")).endswith(state):
        raise ValueError(f"target state label differs in {path}")
    return document


def compute_family_cells(family_name, roots, notify=None):
    """Execute every scheduled cell of one family; typed rows cover the shortfall."""
    spec = FAMILIES[family_name]
    root = roots[family_name]
    began = time.monotonic()
    rows = []
    if family_name == "novphy_normal_mechanics":
        sources = [(root / relative, state) for relative, state in
                   zip(spec["targets"], NOVPHY_STATES, strict=True)]
        for index, (path, state) in enumerate(sources, start=1):
            document = read_novphy_target(path, state)
            rows.extend(regime.novphy_window_rows(document))
            if notify:
                notify(family_name, index * NOVPHY_CANDIDATES_PER_STATE,
                       spec["scheduled_cells"], began)
    else:
        sources = [(root / relative, relative) for relative in spec["shards"]]
        for index, (path, _) in enumerate(sources, start=1):
            rows.extend(_torch_shard(path, spec["source_identity"]))
            if notify:
                notify(family_name, index * 40, spec["scheduled_cells"], began)
    # every scheduled cell must carry an outcome row; missing ones are typed failures
    executed = {row["cell_id"] for row in rows}
    for cell_id in sorted(scheduled_cell_ids()[family_name] - executed):
        rows.append({"cell_id": cell_id, "status": "typed_terminal_failure",
                     "reason": "scheduled cell absent from the frozen input artifacts"})
    rows.sort(key=lambda row: row["cell_id"])
    return rows


def enumerate_family_cells(family_name, roots):
    """Untimed structural enumeration used by --dry-run/--prepare (counts only)."""
    spec = FAMILIES[family_name]
    root = roots[family_name]
    if family_name == "novphy_normal_mechanics":
        counts = []
        for relative, state in zip(spec["targets"], NOVPHY_STATES, strict=True):
            document = read_novphy_target(root / relative, state)
            counts.append(sum(1 for candidate in document["candidates"]
                              if candidate.get("status") == "available"))
        return {"states": len(counts), "accepted_candidates": sum(counts),
                "scheduled": spec["scheduled_cells"]}
    shape = None
    for relative in spec["shards"][:1]:
        shard = torch.load(root / relative, map_location="cpu", weights_only=True)
        shape = list(shard["windows"]["z"].shape)
    return {"shards": len(spec["shards"]), "window_tensor_shape": shape,
            "scheduled": spec["scheduled_cells"]}


# ---------------------------------------------------------------------------
# report assembly (pure transform of the executed cells)
# ---------------------------------------------------------------------------

def _json_regions(record):
    """Represent unbounded falsification-region edges as None (JSON-safe)."""
    if record is None:
        return None
    out = {}
    for case, (low, high) in record.items():
        out[case] = [None if low == float("-inf") else low,
                     None if high == float("inf") else high]
    return out


def build_report(plan, family_cells):
    """Assemble every published table from executed cells under the frozen plan."""
    draws = plan["bootstrap"]["draws"]
    seeds = plan["bootstrap"]["seeds"]
    statistics = {}
    for family_name in FAMILY_ORDER:
        rows = family_cells[family_name]
        statistics[family_name] = {}
        for statistic in regime.STATISTIC_NAMES:
            interval, unavailable = regime.family_intervals(
                rows, statistic, draws, seeds[family_name])
            statistics[family_name][statistic] = {
                "interval": interval,
                "typed_unavailable": unavailable,
                "n_computed": None if interval is None else interval["n"],
                "event_record_units": plan["families"][family_name]["event_record_units"]}
    means = {family_name: {statistic: (None
             if statistics[family_name][statistic]["interval"] is None
             else statistics[family_name][statistic]["interval"]["mean"])
             for statistic in regime.STATISTIC_NAMES}
             for family_name in FAMILY_ORDER}
    rules = []
    for rule in plan["rules"]:
        statistic = rule["statistic"]
        computable = all(means[family][statistic] is not None for family in FAMILY_ORDER)
        outcome = regime.rule_outcome(means["novphy_normal_mechanics"][statistic],
                                      means["physion_dominoes"][statistic],
                                      means["clevrer"][statistic],
                                      rule["direction"], rule["margin"]) if computable else None
        record = regime.falsification_record(means["novphy_normal_mechanics"][statistic],
                                             means["physion_dominoes"][statistic],
                                             means["clevrer"][statistic],
                                             rule["direction"],
                                             rule["margin"]) if computable else None
        rules.append({**rule, "family_means": {family: means[family][statistic]
                                              for family in FAMILY_ORDER},
                      "computable": computable, "outcome": outcome,
                      "fourth_family_consistency_regions": _json_regions(record)})
    primary = next(rule for rule in rules if rule["statistic"] == PRIMARY_RULE)
    q1_exploratory = regime.question_1_disposition(primary["outcome"], primary["computable"])
    q2 = regime.question_2_disposition([rule["outcome"] for rule in rules])
    cell_dispositions = {}
    for family_name in FAMILY_ORDER:
        rows = family_cells[family_name]
        computed = sum(1 for row in rows if row.get("status") == "computed")
        typed = sum(1 for row in rows if row.get("status") != "computed")
        if computed + typed != plan["families"][family_name]["scheduled_cells"]:
            raise ValueError(f"{family_name}: cell outcomes {computed + typed} differ from "
                             f"scheduled {plan['families'][family_name]['scheduled_cells']}")
        cell_dispositions[family_name] = {
            "scheduled": plan["families"][family_name]["scheduled_cells"],
            "computed": computed, "typed_terminal_failures": typed,
            "typed_rows": [{ "cell_id": row["cell_id"], "reason": row.get("reason")}
                           for row in rows if row.get("status") != "computed"]}
    return {"schema": "issue_86_regime_synthesis_report_v1",
            "identity": plan["identity"],
            "component_table": plan["component_table"],
            "statistics": statistics,
            "rules": rules,
            "dispositions": {
                "question_1_regime_modulation": "readiness_or_precision_insufficient",
                "question_2_predictability_bound": "readiness_or_precision_insufficient"},
            "exploratory_frozen_rule_dispositions": {
                "question_1_regime_modulation": q1_exploratory,
                "question_2_predictability_bound": q2,
                "scope_note": (
                    "frozen-rule outputs under the count-only presence semantics; they do "
                    "NOT answer the ticket's questions as posed - the count-only "
                    "operationalization was a freeze-time scope error (confirmed by the "
                    "parent acceptance audit and the read-only reviewer), so these tokens "
                    "are reported side by side, never as the canonical answers"),
                "disposition_rules": {
                    "question_1_regime_modulation":
                        DISPOSITION_RULES["question_1_regime_modulation_frozen_rule_exploratory"],
                    "question_2_predictability_bound":
                        DISPOSITION_RULES["question_2_predictability_bound_frozen_rule_exploratory"]}},
            "ticket_disposition": "readiness_or_precision_insufficient",
            "closure_eligible": False,
            "typed_blockers": [
                "rule_expressiveness: the frozen monotone rules operationalize only "
                "presence-count ordering and never test which component fails, so the "
                "ticket's three-distinct-pattern questions were never operationalized at "
                "freeze (a freeze-time scope error)",
                "cross_family_comparability: the primary statistic's settlement anchor is "
                "not semantically identical across families and the frozen disclosure "
                "scoped itself to the other two statistics, leaving the primary rule's "
                "cross-family reading uncompensated",
                "protocol_deviation: 93 real cells were scored pre-freeze by the dry-run "
                "timing sample, which also measured wall seconds without the shared flock "
                "the user requires for wall-time-measured phases"],
            "disposition_basis": (
                "the canonical dispositions reflect the ticket's actual questions, which "
                "cannot be validly answered under the existing freeze (typed blockers "
                "above); the frozen-rule outcomes are preserved verbatim under "
                "exploratory_frozen_rule_dispositions; no frozen number, rule, plan "
                "section, or computed table was changed"),
            "post_audit_protocol_record": POST_AUDIT_PROTOCOL_RECORD,
            "disposition_rules": DISPOSITION_RULES,
            "cell_dispositions": cell_dispositions,
            "claim_boundary": plan["claim_boundary"],
            "limitations": plan["limitations"]}


# ---------------------------------------------------------------------------
# published artifact renderers (byte-deterministic)
# ---------------------------------------------------------------------------

def _fixed(value, digits=6):
    return f"{value:.{digits}f}"


def comparisons_csv(report):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("family", "statistic", "scheduled_cells", "n_computed",
                     "typed_unavailable", "mean", "ci_low", "ci_high", "event_record_units"))
    for family_name in FAMILY_ORDER:
        cells = report["cell_dispositions"][family_name]
        for statistic in regime.STATISTIC_NAMES:
            entry = report["statistics"][family_name][statistic]
            interval = entry["interval"]
            writer.writerow((family_name, statistic, cells["scheduled"],
                             "" if interval is None else interval["n"],
                             entry["typed_unavailable"],
                             "" if interval is None else repr(interval["mean"]),
                             "" if interval is None else repr(interval["ci_low"]),
                             "" if interval is None else repr(interval["ci_high"]),
                             entry["event_record_units"]))
    return stream.getvalue()


def regime_modulation_csv(report):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("row_type", "family", "statistic", "direction", "margin", "value",
                     "ci_low", "ci_high", "rule_outcome", "disposition", "detail"))
    for family_name, components in report["component_table"].items():
        for component in ("S1", "S2", "S3"):
            writer.writerow(("component", family_name, component, "", "", "",
                             "", "", "", "",
                             f"{components[component]}; presence_count={components['presence_count']}"))
        writer.writerow(("presence_count", family_name, "", "", "",
                         str(components["presence_count"]), "", "", "", "",
                         components["published_source"]))
    for family_name in FAMILY_ORDER:
        for statistic in regime.STATISTIC_NAMES:
            entry = report["statistics"][family_name][statistic]
            interval = entry["interval"]
            writer.writerow(("family_statistic", family_name, statistic, "", "",
                             "" if interval is None else repr(interval["mean"]),
                             "" if interval is None else repr(interval["ci_low"]),
                             "" if interval is None else repr(interval["ci_high"]),
                             "", "", f"n={interval['n']}" if interval else "unavailable"))
    for rule in report["rules"]:
        writer.writerow(("rule", "", rule["statistic"], rule["direction"],
                         repr(rule["margin"]), "", "", "",
                         rule["outcome"] or "uncomputable",
                         report["dispositions"]["question_1_regime_modulation"]
                         if rule["statistic"] == PRIMARY_RULE else "",
                         f"role={rule['role']}; computable={rule['computable']}; "
                         f"novphy={rule['family_means']['novphy_normal_mechanics']!r}; "
                         f"physion={rule['family_means']['physion_dominoes']!r}; "
                         f"clevrer={rule['family_means']['clevrer']!r}"))
        regions = rule["fourth_family_consistency_regions"]
        if regions is None:
            writer.writerow(("falsification_region", "", rule["statistic"],
                             rule["direction"], repr(rule["margin"]),
                             "uncomputable", "", "", "", "",
                             "family intervals uncomputable for this rule; no "
                             "fourth-family region is defined"))
            continue
        for case in ("count_3", "count_2", "count_le_1"):
            low, high = regions[case]
            writer.writerow(("falsification_region", "", rule["statistic"],
                             rule["direction"], repr(rule["margin"]),
                             f"[{'' if low is None else repr(low)}, "
                             f"{'' if high is None else repr(high)}]", "", "", "", "",
                             f"fourth-family presence {case} (blank edge = unbounded); "
                             f"measured value outside the region falsifies the rule"))
    for key in ("question_1_regime_modulation", "question_2_predictability_bound"):
        writer.writerow(("disposition", "", key, "", "", "", "", "",
                         "", report["dispositions"][key],
                         report["disposition_rules"][key]))
    exploratory = report["exploratory_frozen_rule_dispositions"]
    for key in ("question_1_regime_modulation", "question_2_predictability_bound"):
        writer.writerow(("exploratory_frozen_rule_disposition", "", key, "", "", "", "",
                         "", exploratory[key],
                         exploratory["disposition_rules"][key]))
    writer.writerow(("exploratory_scope_note", "", "", "", "", "", "", "", "", "",
                     exploratory["scope_note"]))
    writer.writerow(("ticket_disposition", "", "", "", "", "", "", "", "",
                     report["ticket_disposition"], "canonical ticket-level disposition"))
    writer.writerow(("closure_eligible", "", "", "", "", "", "", "", "",
                     str(report["closure_eligible"]).lower(), "parent gates closure"))
    for blocker in report["typed_blockers"]:
        writer.writerow(("typed_blocker", "", "", "", "", "", "", "", "",
                         report["ticket_disposition"], blocker))
    for name in ("protocol_deviation", "rule_expressiveness", "cross_family_comparability"):
        writer.writerow(("post_audit_protocol_record", "", name, "", "", "", "", "",
                         "", "typed_record",
                         f"{report['post_audit_protocol_record']['date']}; "
                         f"{report['post_audit_protocol_record'][name]}"))
    return stream.getvalue()


def findings_md(report):
    lines = ["# Issue-86 cross-family regime-modulation synthesis - findings", ""]
    typed_total = sum(cells["typed_terminal_failures"]
                      for cells in report["cell_dispositions"].values())
    completion = "True" if typed_total == 0 else \
        f"True with {typed_total} typed terminal failure cells retained"
    lines.append(f"Diagnostics complete: {completion}. Claim boundary: "
                 f"{report['claim_boundary']}. "
                 "Window bootstrap intervals are DESCRIPTIVE; three families support "
                 "ordinal consistency at most; the event-indicator incompatibility below is "
                 "typed, never averaged away.")
    lines.append("")
    lines.append("## Frozen evidence cells (executed or typed)")
    lines.append("")
    lines.append("| Family | Cell unit | Scheduled | Computed | Typed terminal failures |")
    lines.append("| --- | --- | --- | --- | --- |")
    for family_name in FAMILY_ORDER:
        cells = report["cell_dispositions"][family_name]
        lines.append(f"| {family_name} | {FAMILIES[family_name]['cell_unit']} | "
                     f"{cells['scheduled']} | {cells['computed']} | "
                     f"{cells['typed_terminal_failures']} |")
    lines.append("")
    lines.append("## Per-family regime statistics (DESCRIPTIVE window bootstrap, 95%)")
    lines.append("")
    lines.append("| Family | Statistic | Mean | 95% interval | n | Typed unavailable | "
                 "Event-record units |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for family_name in FAMILY_ORDER:
        for statistic in regime.STATISTIC_NAMES:
            entry = report["statistics"][family_name][statistic]
            interval = entry["interval"]
            mean = "" if interval is None else _fixed(interval["mean"])
            band = "" if interval is None else \
                f"[{_fixed(interval['ci_low'])}, {_fixed(interval['ci_high'])}]"
            lines.append(f"| {family_name} | {statistic} | {mean} | {band} | "
                         f"{'' if interval is None else interval['n']} | "
                         f"{entry['typed_unavailable']} | {entry['event_record_units']} |")
    lines.append("")
    lines.append("## Frozen monotone-modulation rules (exploratory; protocol deviation "
                 "below)")
    lines.append("")
    lines.append("| Statistic | Role | Direction | Margin | NovPhy | Physion | CLEVRER | Outcome |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for rule in report["rules"]:
        def cell(family):
            value = rule["family_means"][family]
            return "" if value is None else _fixed(value)
        lines.append(f"| {rule['statistic']} | {rule['role']} | {rule['direction']} | "
                     f"{rule['margin']} | {cell('novphy_normal_mechanics')} | "
                     f"{cell('physion_dominoes')} | {cell('clevrer')} | "
                     f"{rule['outcome'] or 'uncomputable'} |")
    lines.append("")
    lines.append("## Dispositions")
    lines.append("")
    disp = report["dispositions"]
    primary = next(rule for rule in report["rules"] if rule["statistic"] == PRIMARY_RULE)
    consistent = [rule["statistic"] for rule in report["rules"] if rule["outcome"] == "consistent"]
    lines.append(f"- Q1 regime modulation: **{disp['question_1_regime_modulation']}** - "
                 f"{report['disposition_rules']['question_1_regime_modulation']}")
    lines.append(f"- Q2 post-hoc predictability bound: "
                 f"**{disp['question_2_predictability_bound']}** - "
                 f"{report['disposition_rules']['question_2_predictability_bound']}")
    lines.append(f"- Ticket disposition: **{report['ticket_disposition']}**; "
                 f"closure_eligible: {str(report['closure_eligible']).lower()}")
    lines.append("")
    exploratory = report["exploratory_frozen_rule_dispositions"]
    lines.append("## Exploratory frozen-rule dispositions (NOT answers to the ticket "
                 "questions)")
    lines.append("")
    lines.append(f"{exploratory['scope_note']}")
    lines.append("")
    lines.append(f"- Q1 frozen count-rule: **{exploratory['question_1_regime_modulation']}** - "
                 f"primary rule (post_settlement_fraction, decreasing, margin "
                 f"{primary['margin']}) outcome {primary['outcome'] or 'uncomputable'} "
                 f"under presence-count semantics; "
                 f"{exploratory['disposition_rules']['question_1_regime_modulation']}")
    lines.append(f"- Q2 frozen count-rule: **{exploratory['question_2_predictability_bound']}** - "
                 f"frozen single-variable rules consistent under count-only semantics: "
                 f"{consistent if consistent else 'none'}; "
                 f"{exploratory['disposition_rules']['question_2_predictability_bound']}")
    lines.append("")
    lines.append("## Post-audit protocol record (typed; plan and tables untouched)")
    lines.append("")
    record = report["post_audit_protocol_record"]
    lines.append(f"Recorded {record['date']} after the parent acceptance audit and a "
                 "read-only reviewer consultation. No table on this page was re-scored, "
                 "re-frozen, or amended; the record adds protocol context only.")
    lines.append("")
    for name in ("protocol_deviation", "rule_expressiveness", "cross_family_comparability"):
        lines.append(f"- {name}: {record[name]}")
    lines.append("")
    lines.append("## Fourth-family falsification record (typed, descriptive)")
    lines.append("")
    lines.append("Consistency regions for a future fourth family under each frozen rule; a "
                 "measured value outside its presence-count region falsifies that rule.")
    lines.append("")
    lines.append("| Rule | Direction | Margin | Fourth family presence count 3 | Count 2 | Count 1 or fewer |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for rule in report["rules"]:
        regions = rule["fourth_family_consistency_regions"]

        def band(case):
            if regions is None:
                return "uncomputable"
            low, high = regions[case]
            low_text = "-inf" if low is None else repr(low)
            high_text = "+inf" if high is None else repr(high)
            return f"({low_text}, {high_text})"
        lines.append(f"| {rule['statistic']} | {rule['direction']} | {rule['margin']} | "
                     f"{band('count_3')} | {band('count_2')} | {band('count_le_1')} |")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# plan freeze
# ---------------------------------------------------------------------------

def _validate_plan(plan):
    """Structural freeze validation: complete formulas, numeric margins, no placeholders."""
    for key in ("schema", "identity", "frozen_at", "inputs", "families", "component_table",
                "rules", "bootstrap", "disposition_rules", "claim_boundary", "limitations",
                "compute", "validation_command"):
        if key not in plan:
            raise ValueError(f"frozen plan missing section {key!r}")
    blob = json.dumps(plan)
    for marker in ("~", "e.g.", "TBD", "placeholder"):
        if marker in blob:
            raise ValueError(f"frozen plan contains placeholder marker {marker!r}")
    if plan["identity"] != IDENTITY or plan["schema"] != SCHEMA:
        raise ValueError("plan identity/schema differ from the frozen contract")
    if plan["validation_command"] != VALIDATION_COMMAND:
        raise ValueError("validation command is binding and differs")
    for family_name in FAMILY_ORDER:
        spec = plan["families"][family_name]
        for key in ("cell_unit", "window_frames", "scheduled_cells", "event_indicator_semantics",
                    "event_record_units", "velocity_records", "frame_cadence"):
            if not spec.get(key):
                raise ValueError(f"frozen family {family_name} missing {key!r}")
        if not isinstance(spec["scheduled_cells"], int) or spec["scheduled_cells"] <= 0:
            raise ValueError(f"frozen family {family_name} scheduled_cells not a positive count")
    if set(plan["component_table"]) != set(FAMILY_ORDER):
        raise ValueError("component table must cover exactly the three families")
    for family_name, components in plan["component_table"].items():
        if components["presence_count"] != sum(1 for c in ("S1", "S2", "S3")
                                               if components[c] == "present"):
            raise ValueError(f"component presence count differs for {family_name}")
    if {rule["statistic"] for rule in plan["rules"]} != set(regime.STATISTIC_NAMES):
        raise ValueError("frozen rules must cover exactly the four frozen statistics")
    for rule in plan["rules"]:
        if rule["direction"] not in regime.RULE_DIRECTIONS:
            raise ValueError(f"rule direction invalid for {rule['statistic']}")
        if not isinstance(rule["margin"], (int, float)) or rule["margin"] <= 0:
            raise ValueError(f"rule margin must be a positive number for {rule['statistic']}")
    primary = [rule["statistic"] for rule in plan["rules"] if rule["role"] == "primary"]
    if primary != [PRIMARY_RULE]:
        raise ValueError("exactly one primary rule is frozen and it is post_settlement_fraction")
    if plan["bootstrap"]["draws"] <= 0 or set(plan["bootstrap"]["seeds"]) != set(FAMILY_ORDER):
        raise ValueError("bootstrap contract incomplete")
    return plan


def make_plan():
    plan = {
        "schema": SCHEMA,
        "identity": IDENTITY,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_boundary": CLAIM_BOUNDARY,
        "component_definitions": list(COMPONENT_DEFINITIONS),
        "component_table": COMPONENT_TABLE,
        "families": {name: {key: spec[key] for key in
                            ("source_identity", "membership_binding", "cell_unit",
                             "window_frames", "scheduled_cells", "event_indicator_semantics",
                             "event_record_units", "velocity_records", "frame_cadence")}
                     for name, spec in FAMILIES.items()},
        "statistic_formulas": {
            "window_units": ("a window is the family's frozen evaluation frame span "
                             "(NovPhy branch: observed frames 0..600 inclusive, 601 slots; "
                             "CLEVRER/Physion: 61 frames [start, start+60]); every formula "
                             "is evaluated per window and aggregated per family by the "
                             "frozen window bootstrap"),
            "contact_active_fraction": ("distinct window frame slots carrying at least one "
                                        "collision record, divided by the window length"),
            "post_settlement_fraction": ("window frame slots strictly after the unit's "
                                         "settlement frame (the last collision record of the "
                                         "unit's FULL frozen timeline; -1 when the timeline "
                                         "has none, in which case the whole window counts as "
                                         "post-settlement and the row is typed "
                                         "no_collision_events), divided by the window length"),
            "collision_event_rate": ("collision records whose frame lies inside the window, "
                                     "divided by the window length; records keep each "
                                     "family's native multiplicity under the typed "
                                     "indicator disclosure"),
            "velocity_decay_ratio": ("mean carrier motion speed (present slots, columns 8,9) "
                                     "at the window's last carrier record divided by the "
                                     "maximum over the window's carrier records; typed None "
                                     "with reason no_motion_in_window when the peak does not "
                                     "exceed 1e-12; scale-free within a window"),
        },
        "event_indicator_disclosure": EVENT_INDICATOR_DISCLOSURE,
        "rules": [dict(rule) for rule in RULES],
        "bootstrap": BOOTSTRAP,
        "disposition_rules": DISPOSITION_RULES,
        "limitations": list(LIMITATIONS),
        "inputs": declared_sources(),
        "input_binding_rule": (
            "no new content hashes are computed anywhere in this ticket (shared ADD-EXP "
            "operator rule); shard and #77-target inputs are bound to identifiers the "
            "artifacts already carry (schema/identity/plan_identity) and are verified "
            "against those identifiers at read time in every phase; published summary "
            "documents are verified against their report identity at freeze and at every "
            "subsequent phase; prose findings documents carry no machine identifier and "
            "are existence-checked and cited verbatim as published provenance"),
        "compute": {"device": "cpu",
                    "gpu_lock": (f"exclusive flock on {GPU_LOCK_PATH} acquired BEFORE the "
                                 "wall timer and held for the ENTIRE --run/--publish/"
                                 "--validate phases (no GPU work exists; the lock is held "
                                 "because those phases publish wall times); --dry-run and "
                                 "--prepare publish no wall time and never take the lock"),
                    "derived_artifact_budget_bytes": DERIVED_BYTES_BUDGET,
                    "checkpoint_reload_gpu_hours": 0.0,
                    "resume": ("per-family cell checkpoints under results/ keyed by plan "
                               "identity and the declared source bindings; deterministic "
                               "recompute")},
        "validation_command": VALIDATION_COMMAND,
    }
    return _validate_plan(plan)


def load_plan(output):
    plan = read_json(Path(output) / "plan.json")
    plan = _validate_plan(plan)
    verify_source_bindings(plan)
    verify_document_bindings(plan)
    return plan


def prepare(output):
    path = Path(output) / "plan.json"
    if path.exists():
        plan = load_plan(output)
        log(f"plan already frozen at {plan['frozen_at']}; {len(plan['inputs'])} input "
            f"bindings verified against the declared source identifiers; no statistic "
            f"computed")
        return plan
    plan = make_plan()
    verify_document_bindings(plan)
    write_json(path, plan)
    log(f"plan frozen: {len(plan['inputs'])} source-bound inputs (existing artifact "
        f"identifiers, no new hashes); component table (novphy 3 present, physion S1 "
        f"absent, clevrer S3 absent); 4 monotone rules (primary post_settlement_fraction "
        f"decreasing margin 0.10); no regime statistic computed yet")
    return plan


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def dry_run():
    """No-write, NO-SCORING structural inventory.

    Post-audit fix (2026-09-22): the dry run computes NO window statistics of
    any kind - the earlier timing sample scored 93 real cells before the
    freeze, violating freeze-before-any-outcome; only structural enumeration
    (file readability, stored tensor shapes, candidate counts) remains.
    """
    roots = {name: ROOT / spec["root"] for name, spec in FAMILIES.items()}
    log("no-write dry-run; structural inventory only; no window statistic is "
        "computed at any point before the frozen plan exists")
    for family_name in FAMILY_ORDER:
        inventory = enumerate_family_cells(family_name, roots)
        log(f"{family_name}: {json.dumps(inventory, sort_keys=True)}")
    log(f"scheduled cells: clevrer={FAMILIES['clevrer']['scheduled_cells']} "
        f"physion={FAMILIES['physion_dominoes']['scheduled_cells']} "
        f"novphy={FAMILIES['novphy_normal_mechanics']['scheduled_cells']}; "
        f"bootstrap draws={BOOTSTRAP['draws']}")
    return 0


def _family_roots():
    return {name: ROOT / spec["root"] for name, spec in FAMILIES.items()}


def run(output):
    plan = load_plan(output)
    output = Path(output)
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    source_bindings = [[entry["family"], entry["artifact"], entry["source_identity"]]
                       for entry in plan["inputs"]]
    family_cells = {}

    def notify(family_name, done, total, began):
        elapsed = time.monotonic() - began
        eta = elapsed / done * (total - done)
        log(f"{family_name}: {done}/{total} cells; elapsed {elapsed:.1f}s; eta {eta:.1f}s")

    with GPULock():
        began = time.monotonic()
        log("gpu lock acquired; wall timer starts for the measured run phase")
        for family_name in FAMILY_ORDER:
            checkpoint = results / f"cells-{family_name}.json"
            if checkpoint.exists():
                cached = read_json(checkpoint)
                if cached.get("identity") == plan["identity"] and \
                        cached.get("source_bindings") == source_bindings:
                    family_cells[family_name] = cached["rows"]
                    log(f"{family_name}: resume from checkpoint "
                        f"({len(cached['rows'])} cell outcomes)")
                    continue
            family_cells[family_name] = compute_family_cells(
                family_name, _family_roots(), notify=notify)
            write_json(checkpoint, {"identity": plan["identity"],
                                    "source_bindings": source_bindings,
                                    "rows": family_cells[family_name]})
            computed = sum(1 for row in family_cells[family_name]
                           if row.get("status") == "computed")
            log(f"{family_name}: {computed} computed, "
                f"{len(family_cells[family_name]) - computed} typed terminal failures")
        report = build_report(plan, family_cells)
        elapsed = time.monotonic() - began
        log(f"cells executed; bootstrap draws={plan['bootstrap']['draws']} per family per "
            f"statistic; dispositions q1={report['dispositions']['question_1_regime_modulation']} "
            f"q2={report['dispositions']['question_2_predictability_bound']} "
            f"ticket={report['ticket_disposition']} closure_eligible="
            f"{str(report['closure_eligible']).lower()}; exploratory frozen-rule tokens: "
            f"q1={report['exploratory_frozen_rule_dispositions']['question_1_regime_modulation']} "
            f"q2={report['exploratory_frozen_rule_dispositions']['question_2_predictability_bound']}")
        compute = {"plan_identity": plan["identity"],
                   "source_bindings": source_bindings,
                   "tables": report,
                   "execution": {"wall_seconds": elapsed, "gpu_lock": plan["compute"]["gpu_lock"],
                                 "device": "cpu"}}
        write_json(output / "compute.json", compute)
        log(f"run phase complete in {elapsed:.2f}s (lock held throughout); compute.json "
            f"written as the deterministic resume/publication point")
    return compute


def publish(output):
    plan = load_plan(output)
    output = Path(output)
    compute_path = output / "compute.json"
    if not compute_path.exists():
        raise ValueError("compute.json missing; run --run before --publish")
    compute = read_json(compute_path)
    if compute.get("plan_identity") != plan["identity"] or \
            compute.get("source_bindings") != [[e["family"], e["artifact"], e["source_identity"]]
                                               for e in plan["inputs"]]:
        raise ValueError("compute.json was produced under a different frozen plan or inputs")
    with GPULock():
        began = time.monotonic()
        log("gpu lock acquired; wall timer starts for the measured publish phase")
        report = compute["tables"]
        write_json(output / "summary.json", report)
        write_text(output / "comparisons.csv", comparisons_csv(report))
        write_text(output / "regime_modulation.csv", regime_modulation_csv(report))
        write_text(output / "findings.md", findings_md(report))
        derived = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if derived > DERIVED_BYTES_BUDGET:
            raise ValueError(f"derived artifacts exceed the {DERIVED_BYTES_BUDGET}-byte budget")
        elapsed = time.monotonic() - began
        log(f"published summary.json comparisons.csv regime_modulation.csv findings.md; "
            f"derived bytes={derived}; publish phase {elapsed:.2f}s (lock held throughout)")
    return report


def validate(output):
    plan = load_plan(output)
    output = Path(output)
    with GPULock():
        began = time.monotonic()
        log("gpu lock acquired; wall timer starts for the measured validate phase")
        family_cells = {
            family_name: compute_family_cells(family_name, _family_roots())
            for family_name in FAMILY_ORDER}
        report = build_report(plan, family_cells)
        expected = {"summary.json": (json.dumps(report, indent=2, sort_keys=True,
                                                allow_nan=False) + "\n").encode(),
                    "comparisons.csv": comparisons_csv(report).encode(),
                    "regime_modulation.csv": regime_modulation_csv(report).encode(),
                    "findings.md": findings_md(report).encode()}
        for name, content in expected.items():
            path = output / name
            if not path.exists():
                raise ValueError(f"published artifact missing: {name}")
            if path.read_bytes() != content:
                raise ValueError(f"published {name} differs from the recomputation from "
                                 f"frozen inputs")
        compute = read_json(output / "compute.json")
        if compute.get("tables") != report:
            raise ValueError("compute.json tables differ from the fresh recomputation")
        elapsed = time.monotonic() - began
        cells = sum(spec["scheduled_cells"] for spec in FAMILIES.values())
        log(f"exact recomputation validation passed: every published table recomputed from "
            f"the frozen inputs (source-identifier-verified at read time) matches the "
            f"published bytes; {cells} scheduled cells re-executed; validate phase "
            f"{elapsed:.2f}s (lock held throughout)")
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
            return dry_run()
        if args.prepare:
            prepare(args.output)
            return 0
        if args.run:
            run(args.output)
        elif args.publish:
            publish(args.output)
        else:
            validate(args.output)
        return 0
    except (ValueError, OSError, KeyError, FileNotFoundError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
