"""Issue-92 ADD-EXP WP2b: cross-seed invariance probe (26 engine slots).

Binding runner module: scripts/run_cross_seed_invariance_probe.py
Exact validation command: python -u -m scripts.run_cross_seed_invariance_probe --validate

Secondary determinism package of ticket #92 (the primary, zero-engine
176/176 + 288/288 determinism measurement is published by #92 WP1).  Two
reviewers split; the reconciled form re-scopes the planned repeat-draw probe
as CROSS-SEED INVARIANCE: NOVPHY_ENVIRONMENT_SEED initializes the Unity RNG
(tasks/issue_76_canonical/CanonicalCaptureSeed.cs) — a canonical-capture
label, not an outcome draw (the engine receives exactly one seed per state,
state["engine_seed"] = 764100001 + member ordinal; ORACLE_SEED = 20260908 is
a #74 matched training-seed label written only into metadata).  A second
seed value therefore tests level-layout/perception invariance: the frozen
scenario authority (level, inventory, decision step 30000, action) is
unchanged, so a verdict flip under the second seed falsifies determinism and
forces a seed-conditional restatement of the ceiling; frames differing while
verdicts agree is the informative separation (selection channel noisy,
outcome channel invariant); 26/26 agreement closes the item as a measured
invariance.

Composition (fixed and pre-declared by ticket #92): 14 member anchors + 7
success ordinals + 5 controls = 26 slots, plus 2 development-tagged smoke
cells outside the 26.  Every slot is a (state, ordinal) whose first-seed
engine-truth verdict is retained in the #87/#89 union table; the first-seed
decision frame is retained in the owning pool's attempts.

Frozen-protocol chronology (binding shared rule):
- --dry-run is a no-write structural inventory that computes NO statistic;
- --smoke executes the 2 development smoke cells at the second seed under
  the full protocol and writes smoke.json (infrastructure assertions only:
  verdicts produced, frames captured, channel scans complete; verdict
  AGREEMENT is the experiment's result and is never a smoke assertion);
- --prepare freezes the complete protocol (slot inventory with concrete
  identities, second-seed constant, comparisons, stop rules, disposition
  mapping, guards, caps, smoke evidence) into plan.json BEFORE any
  experiment cell is executed;
- --run executes the 26 slots once, serialized (single isolated worker per
  cell, cold starts cannot collide), ledger-resumable, with the frozen
  verdict-flip stop rule checked after every cell;
- --rescore re-scores selection on both frames (retained first-seed anchor
  frame and the second-seed decision frame) per distinct state over the 12
  #87 model cells, decision-only GPU work, DESCRIPTIVE;
- --publish renders summary.json / comparisons.csv / findings.md /
  slot_join.csv / receipts.json;
- --validate re-derives every published table from the retained records and
  byte-compares (no engine, no fresh rollouts).

Disposition vocabulary (frozen exactly): supported /
not_supported_by_this_experiment / readiness_or_precision_insufficient.
Stop rules (frozen, both branches): any verdict flip — a success slot
failing or a control/anchor slot succeeding at the second seed — halts the
run, publishes the flip evidence, and maps to
not_supported_by_this_experiment with the seed-conditional restatement
obligation recorded.  Typed execution failures (connect/readiness/stability)
are retained, never silently retried: a slot without a verdict maps to
readiness_or_precision_insufficient unless a flip already settled the
question.

Claim boundary: re-execution of 26 frozen (state, ordinal) slots at one
frozen second seed value; single-shot, decision-only re-scoring,
development/exposed N1 lineages only; #64/#65 stay sealed; #87/#89/#90
published verdicts are inputs and are never recomputed, amended, or
reinterpreted; no retraining, no method changes, no complete-gameplay or
multi-shot claim; the capture-channel and selection comparisons are
DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-92-cross-seed-invariance-v1"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"

EIGHTY_SEVEN = ROOT / ".local-artifacts/issue-87-closed-loop-oracle-v1"
EIGHTY_NINE = ROOT / ".local-artifacts/issue-89-oracle-completion-v1"

SCHEMA_PLAN = "issue_92_cross_seed_invariance_plan_v1"
SCHEMA_COMPUTE = "issue_92_cross_seed_invariance_compute_v1"
SCHEMA_REPORT = "issue_92_cross_seed_invariance_report_v1"
SCHEMA_RECORD = "issue_92_cross_seed_invariance_slot_v1"
SCHEMA_RESCORE = "issue_92_cross_seed_invariance_rescore_v1"
SCHEMA_RECEIPTS = "issue_92_cross_seed_invariance_receipts_v1"
IDENTITY = "issue-92-cross-seed-invariance-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_cross_seed_invariance_probe --validate"

ORACLE_SEED_LABEL = 20260908  # protocol label only; never reaches the engine
SECOND_SEED_OFFSET = 1_000_000_000
SEEDS = (20260908, 20260909, 20260910)
SYSTEMS = ("continuous-fixed-h1", "continuous-fixed-h5",
           "hybrid-fixed-h1", "hybrid-fixed-h5")
DEVICE = "cuda"

# execution bounds inherited verbatim from #89's frozen v2 mitigation
DECISION_FIXED_STEP = 30000
NATIVE_STRIDE = 50
SHOT_SECONDS = 180
INFERENCE_HOLD_SECONDS = 2
ATTEMPT_SECONDS = 2400
AGENT_SOCKET_SECONDS = 600
AGENT_CONNECT_DEADLINE_SECONDS = 600
PHYSICS_SOCKET_SECONDS = 120
PREPARE_FOR_PLAY_SECONDS = 300
HISTORY_READY_SECONDS = 600
WORKER_RSS_MIB = 4096

WALL_CAP_SECONDS = 4 * 3600.0
GPU_CAP_SECONDS = 0.5 * 3600.0
ARTIFACT_BYTES_CAP = 8 * 1024 ** 3
MINIMUM_FREE_BYTES = 20 * 1024 ** 3
STOP_TOKEN = "readiness_or_precision_insufficient"
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")

CLAIM_BOUNDARY = (
    "re-execution of 26 frozen (state, ordinal) slots at one frozen second seed value "
    "with the scenario authority, decision step, inventory and actions unchanged; "
    "single-shot, decision-only re-scoring, development/exposed N1 lineages only; "
    "#64/#65 stay sealed; #87/#89/#90 published verdicts are inputs and are never "
    "recomputed, amended, or reinterpreted; no retraining, no method changes, no "
    "multi-shot or complete-gameplay claim; capture-channel and selection comparisons "
    "are DESCRIPTIVE")

LIMITATIONS = (
    "one second seed value: the probe measures invariance across one frozen "
    "perturbation of the Unity RNG, not a distribution over seeds",
    "NOVPHY_ENVIRONMENT_SEED initializes the Unity RNG only; whatever the episode "
    "does not randomize is unchanged by construction, so byte-identical frames are a "
    "possible and informative outcome",
    "the re-scoring covers the 12 #87 model cells on the composition's distinct "
    "states; it is a selection-channel reading, not a new selection-validity estimate",
    "the remaining scope caps (single-shot, decision-only, development lineages, "
    "sealed benchmark untouched) stay as disclosed limitations",
)

PLACEHOLDER_MARKERS = ("TBD", "placeholder", "to be frozen", "XXX", "FIXME")


def log(message):
    print(f"[issue-92-cross-seed] {message}", flush=True)


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
# slot membership (enumerated once at --prepare; the plan is the binding)
# ---------------------------------------------------------------------------

def load_inputs():
    from scripts import run_selection_validity_restatement as wp1
    plan87, decisions, oracles87 = wp1.load_issue_87()
    summary89, records89 = wp1.load_issue_89()
    verdicts, channel, typed_failures, unmeasured = wp1.build_verdicts(
        plan87, oracles87, records89)
    return plan87, decisions, summary89, verdicts, oracles87, records89


def second_seed(state):
    return state["engine_seed"] + SECOND_SEED_OFFSET


def anchor_of_state(state_identity, decisions):
    record = next(record for record in decisions.values()
                  if record["state_identity"] == state_identity
                  and not record.get("failure_kind"))
    return record["anchor"]


def owning_pool_record(state_identity, ordinal, oracles87, records89):
    """The record carrying the union verdict and the retained decision frame."""
    identity = f"oracle--{state_identity}--a{ordinal:02d}"
    if identity in records89:
        return EIGHTY_NINE, records89[identity]
    if identity in oracles87:
        return EIGHTY_SEVEN, oracles87[identity]
    raise ValueError(f"{STOP_TOKEN}: no retained oracle record for {identity}")


def enumerate_slots(plan87, decisions, summary89, verdicts, oracles87, records89):
    """The frozen composition: 14 member anchors + 7 success ordinals + 5 controls."""
    anchored = sorted({record["state_identity"] for record in decisions.values()
                       if not record.get("failure_kind")})
    state_by_identity = {state["identity"]: state for state in plan87["states"]}
    member_states = defaultdict(list)
    for identity in anchored:
        member_states[state_by_identity[identity]["source_member"]].append(identity)
    members = sorted(member_states)
    if len(members) != 14:
        raise ValueError(f"{STOP_TOKEN}: measurable members {len(members)} != 14")
    ceilings = sorted(state for state, entry in
                      summary89["ceiling_reestimate"]["per_state"].items()
                      if entry["indicator"] == 1 and not entry["typed_unmeasurable"])
    ceiling_members = sorted({state_by_identity[state]["source_member"]
                              for state in ceilings})
    non_ceiling = [member for member in members if member not in set(ceiling_members)]

    def slot(kind, state_identity, ordinal):
        pool, record = owning_pool_record(state_identity, ordinal, oracles87, records89)
        anchor = anchor_of_state(state_identity, decisions)
        return {
            "identity": f"seed2--{state_identity}--a{ordinal:02d}",
            "kind": kind,
            "state": state_identity,
            "member": state_by_identity[state_identity]["source_member"],
            "ordinal": ordinal,
            "branch_identity": next(
                item["branch_identity"]
                for item in state_by_identity[state_identity]["inventory"]
                if item["ordinal"] == ordinal),
            "engine_seed_first": state_by_identity[state_identity]["engine_seed"],
            "engine_seed_second": second_seed(state_by_identity[state_identity]),
            "retained_verdict": bool(verdicts[(state_identity, ordinal)]),
            "retained_frame": {
                "pool": str(pool),
                "path": str(pool / "attempts" / record["cell"]["identity"]
                            / "decision-frame.png"),
            },
            "state_anchor": {"cell": anchor["cell"], "sha256": anchor["sha256"]},
        }

    slots = []
    for member in members:  # 14 member anchors
        state_identity = sorted(member_states[member])[0]
        anchor = anchor_of_state(state_identity, decisions)
        slots.append(slot("anchor", state_identity, anchor["ordinal"]))
    for member in ceiling_members:  # 7 success ordinals
        successes = sorted((state, ordinal) for (state, ordinal), hit in verdicts.items()
                           if hit and state_by_identity[state]["source_member"] == member)
        state_identity, ordinal = successes[0]
        slots.append(slot("success", state_identity, ordinal))
    control_members = non_ceiling[:5]
    for member in control_members:  # 5 controls
        state_identity = sorted(member_states[member])[0]
        anchor = anchor_of_state(state_identity, decisions)
        candidates = sorted(ordinal for (state, ordinal), hit in verdicts.items()
                            if state == state_identity and not hit
                            and ordinal != anchor["ordinal"])
        if not candidates:
            raise ValueError(f"{STOP_TOKEN}: no control candidate on {state_identity}")
        slots.append(slot("control", state_identity, candidates[0]))
    if len(slots) != 26 or len({item["identity"] for item in slots}) != 26:
        raise ValueError(f"{STOP_TOKEN}: composition {len(slots)} != 26 or not unique")
    smoke = []
    success_identities = {item["identity"] for item in slots if item["kind"] == "success"}
    extra_successes = sorted(
        (state, ordinal) for (state, ordinal), hit in verdicts.items()
        if hit and f"seed2--{state}--a{ordinal:02d}" not in success_identities
        and state in set(anchored))
    if not extra_successes:
        raise ValueError(f"{STOP_TOKEN}: no duplicate-state success slot for smoke")
    state_identity, ordinal = extra_successes[0]
    smoke.append(slot("smoke_success", state_identity, ordinal)
                 | {"identity": f"smoke--{state_identity}--a{ordinal:02d}"})
    smoke_member = non_ceiling[5]
    state_identity = sorted(member_states[smoke_member])[0]
    anchor = anchor_of_state(state_identity, decisions)
    candidates = sorted(ordinal for (state, ordinal), hit in verdicts.items()
                        if state == state_identity and not hit
                        and ordinal != anchor["ordinal"])
    smoke.append(slot("smoke_control", state_identity, candidates[0])
                 | {"identity": f"smoke--{state_identity}--a{candidates[0]:02d}"})
    return slots, smoke


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def frozen_plan(frozen_at, slots, smoke, smoke_evidence):
    return {
        "schema": SCHEMA_PLAN,
        "identity": IDENTITY,
        "version": 1,
        "role": "terminal",
        "frozen_at": frozen_at,
        "frozen_before_scoring_run": True,
        "changelog": [{
            "version": 1,
            "change": "initial and terminal freeze: slot inventory, second-seed "
                      "constant, comparisons, stop rules, disposition mapping, guards, "
                      "caps and the smoke evidence are published before any experiment "
                      "cell is executed",
            "why": "standard freeze rule of the shared ADD-EXP protocol; the probe is "
                   "re-scoped from repeat-draw variance to cross-seed invariance per "
                   "ticket #92 (the env seed is a canonical-capture label, not an "
                   "outcome draw)",
            "evidence": "smoke.json infrastructure assertions passed and are embedded "
                        "below; no experiment verdict exists at freeze time",
        }],
        "protocol_policy": (
            "single terminal protocol version: the full protocol is frozen with actual "
            "numeric values before the scoring run and executed exactly once; the dry "
            "run computes no statistic; --validate re-derives every published table "
            "from the retained records"),
        "motivation": (
            "the primary determinism measurement (176/176 unique branches, 288/288 "
            "executed slots, zero channel anomalies) is published by #92 WP1 at zero "
            "engine cost; this probe adds the cross-seed axis: "
            "NOVPHY_ENVIRONMENT_SEED initializes the Unity RNG "
            "(CanonicalCaptureSeed.cs) and is the only engine-visible seed, so a "
            "second value perturbs the capture channel while the frozen scenario "
            "authority holds the level, inventory, decision step and action fixed"),
        "question": (
            "do the engine-truth verdicts of the 26 frozen slots survive one frozen "
            "second value of NOVPHY_ENVIRONMENT_SEED, and do the captured decision "
            "frames and the re-scored selections differ where the verdicts do not?"),
        "second_seed": {
            "rule": f"engine_seed_second = state.engine_seed + {SECOND_SEED_OFFSET}",
            "constant": SECOND_SEED_OFFSET,
            "first_seed_rule": "state.engine_seed = 764100001 + member ordinal (#87's "
                               "frozen binding, inherited verbatim)",
            "seed_label_disclosure": (
                "ORACLE_SEED = 20260908 is a #74 matched training-seed label written "
                "only into metadata/identity strings; the engine receives exactly one "
                "seed per execution (NOVPHY_ENVIRONMENT_SEED); re-running the slots at "
                "'seed 20260909' would dispatch byte-identical engine invocations"),
        },
        "universe": {
            "composition": "14 member anchors + 7 success ordinals + 5 controls = 26 "
                           "slots (fixed and pre-declared by ticket #92), plus 2 "
                           "development-tagged smoke cells outside the 26",
            "anchor_rule": ("one anchor slot per measurable member: the member's "
                            "lowest state identity's anchor cell from the retained #87 "
                            "decision records (lowest-ordinal executed oracle cell)"),
            "success_rule": ("one success slot per ceiling member: the member's "
                             "successful (state, ordinal) in the union verdict table "
                             "with the lowest state identity"),
            "control_rule": ("one control slot on each of the 5 lowest-identity "
                             "non-ceiling measurable members: the member's lowest "
                             "state identity's executed union-failure slot with the "
                             "lowest ordinal excluding the anchor ordinal"),
            "slots": slots,
            "smoke_cells": smoke,
        },
        "comparisons": {
            "verdict": ("per slot: the second-seed engine-truth first-shot-success "
                        "verdict (the #85-published detection rule, inherited "
                        "verbatim) vs the retained union-table verdict; a flip in "
                        "either direction fires the stop rule"),
            "capture": ("per slot: the second-seed decision-frame.png bytes vs the "
                        "owning pool's retained decision frame; byte equality "
                        "reported DESCRIPTIVELY; frames differing while verdicts "
                        "agree is the informative separation"),
            "selection": ("per distinct state in the composition: the 12 #87 model "
                          "cells (4 systems x 3 seeds) re-scored on the retained "
                          "first-seed anchor frame and on the second-seed decision "
                          "frame; chosen-ordinal agreement reported DESCRIPTIVELY"),
        },
        "stop_rules": {
            "verdict_flip": ("any slot whose second-seed verdict differs from its "
                             "retained verdict halts the run after the cell completes; "
                             "remaining cells are typed not_executed; disposition "
                             "not_supported_by_this_experiment with the "
                             "seed-conditional restatement obligation recorded"),
            "caps": "wall, GPU, artifact or free-storage caps halt with "
                    f"{STOP_TOKEN}",
        },
        "disposition_mapping": {
            "supported": "all 26 slots produced second-seed verdicts and all 26 agree "
                         "with the retained verdicts (measured invariance; the item "
                         "closes)",
            "not_supported_by_this_experiment": ("any verdict flip; determinism is "
                                                 "falsified and the ceiling becomes "
                                                 "seed-conditional (restatement "
                                                 "obligation)"),
            "readiness_or_precision_insufficient": ("caps exceeded, or any slot "
                                                    "without a second-seed verdict "
                                                    "and no flip observed"),
        },
        "precision_guards": {
            "slots_executed": 26,
            "verdicts_present": 26,
            "frame_comparisons_present": 26,
            "rescore_cells": "12 model cells x 2 frames x the composition's distinct "
                             "states",
            "rule": ("each guard is evaluated on the retained records; a failed guard "
                     f"maps the disposition to {STOP_TOKEN} before the agreement "
                     "rules are read"),
        },
        "execution_protocol": {
            "cold_start_mitigation": ("serialized execution: one isolated worker at a "
                                      "time (spawn context), so concurrent engine "
                                      "cold starts cannot collide; extended "
                                      "connect/readiness bounds inherited verbatim "
                                      "from #89's frozen v2 mitigation"),
            "bounds": {"agent_socket_seconds": AGENT_SOCKET_SECONDS,
                       "agent_connect_deadline_seconds": AGENT_CONNECT_DEADLINE_SECONDS,
                       "physics_socket_seconds": PHYSICS_SOCKET_SECONDS,
                       "prepare_for_play_seconds": PREPARE_FOR_PLAY_SECONDS,
                       "history_ready_seconds": HISTORY_READY_SECONDS,
                       "attempt_wall_seconds": ATTEMPT_SECONDS,
                       "worker_rss_mib": WORKER_RSS_MIB},
            "instruments": ("the #85-published channel scan inherited verbatim "
                            "(scripts.run_engine_outcome_reactive_diagnostic."
                            "channel_scan); decision fixed step 30000; native stride "
                            "50; shot window 180 s; inference hold 2 s; the "
                            "capture-segment member authority is presented with the "
                            "second seed substituted, so the native manifest's "
                            "recorded engine_seed binds the dispatched seed and is "
                            "published per slot as the dispatch proof; the scenario "
                            "authority is unchanged"),
            "typed_failures": ("connect/readiness/stability failures are retained as "
                               "typed execution failures, never silently retried; a "
                               "slot without a verdict maps to "
                               f"{STOP_TOKEN} unless a flip already settled the "
                               "question"),
        },
        "smoke": {
            "cells": [item["identity"] for item in smoke],
            "assertions": ("infrastructure only: both cells produce a channel-scan "
                           "verdict, a captured decision frame, and a retained-frame "
                           "comparison; verdict agreement is the experiment's result "
                           "and is never asserted at smoke"),
            "stop_rule": ("abort before the terminal freeze if an infrastructure "
                          "assertion fails; the smoke evidence is embedded in this "
                          "plan"),
            "evidence": smoke_evidence,
        },
        "caps": {
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "wall_scope": "cumulative measured wall of the run phase",
            "gpu_cap_seconds": GPU_CAP_SECONDS,
            "gpu_scope": "cumulative cuda-bracketed GPU seconds of the rescore phase",
            "artifact_bytes_cap": ARTIFACT_BYTES_CAP,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "stop": STOP_TOKEN,
        },
        "player_root": str(ROOT / ".local-artifacts/issue-77-n1-v1/player"),
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": list(LIMITATIONS),
        "validation_command": VALIDATION_COMMAND,
        "inputs": [
            {"name": "issue_87_plan", "artifact": str(EIGHTY_SEVEN / "plan.json"),
             "schema": "issue_87_closed_loop_oracle_plan_v1",
             "role": "membership, inventories, engine seeds, anchors"},
            {"name": "issue_87_records", "artifact": str(EIGHTY_SEVEN / "records"),
             "schema": "directory",
             "role": "the union verdict table (#87 side) and retained decision frames"},
            {"name": "issue_89_records", "artifact": str(EIGHTY_NINE / "records"),
             "schema": "directory",
             "role": "the union verdict table (#89 completion side) and retained "
                     "decision frames of the completed slots"},
            {"name": "issue_89_summary", "artifact": str(EIGHTY_NINE / "summary.json"),
             "schema": "issue_89_oracle_completion_report_v1",
             "role": "the sealed ceiling-state set, cited; never recomputed"},
            {"name": "issue_77_n1_player", "artifact": str(
                ROOT / ".local-artifacts/issue-77-n1-v1/player"),
             "schema": "directory", "role": "the frozen N1 campaign player"},
        ],
    }


def load_plan(path):
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError(f"plan {path} is not the issue-92 cross-seed protocol")
    blob = json_text(plan)
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    for key in ("motivation", "question", "second_seed", "universe", "comparisons",
                "stop_rules", "disposition_mapping", "precision_guards",
                "execution_protocol", "smoke", "caps", "player_root",
                "claim_boundary", "limitations", "validation_command", "inputs"):
        if key not in plan:
            raise ValueError(f"frozen plan is missing section {key!r}")
    if not plan.get("frozen_before_scoring_run"):
        raise ValueError("frozen plan does not declare a pre-scoring freeze")
    if len(plan["universe"]["slots"]) != 26:
        raise ValueError("frozen plan composition is not 26 slots")
    return plan


def load_terminal_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("root plan.json missing; run --smoke then --prepare first")
    return load_plan(path)


# ---------------------------------------------------------------------------
# cell execution (second seed; instruments inherited verbatim)
# ---------------------------------------------------------------------------

def _plain_record(plan, cell):
    kind = "smoke" if cell["identity"].startswith("smoke") else "invariance"
    return {"schema": SCHEMA_RECORD, "plan_identity": plan["identity"],
            "plan_version": plan["version"], "cell": cell, "cell_kind": kind,
            "state_identity": cell["state"],
            "seed": ORACLE_SEED_LABEL,
            "engine_seed_first": cell["engine_seed_first"],
            "engine_seed_second": cell["engine_seed_second"],
            "failure": None, "failure_kind": None,
            "decision_frame": None, "frame_comparison": None,
            "execution": None, "outcome": None, "engine_channel": None,
            "verdict_comparison": None,
            "fresh_scenario_lineage": False, "issue_64_authorized": False}


def execute_cell(payload):
    """One isolated, real-rendered second-seed execution of a frozen slot: the
    candidate's frozen action from the state's frozen initial condition with the
    scenario authority unchanged; engine-truth outcome and the decision frame only
    (no predictor scoring inside the attempt)."""
    from scripts import run_closed_loop_oracle_completion as c89
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    c89.torch.set_num_threads(2)
    output = Path(payload["output"])
    plan = read_json(Path(payload.get("plan_path") or (output / "plan.json")))
    cell = payload["cell"]
    identity = cell["identity"]
    state = payload["state"]
    attempt = output / "attempts" / identity
    attempt.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = _plain_record(plan, cell)
    bridge = physics = engine_process = display_process = None
    variables = ("DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
                 "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE",
                 "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
                 "NOVPHY_ENVIRONMENT_SEED", "NOVPHY_NATIVE_DECISION_STEP")
    environment = {key: os.environ.get(key) for key in variables}
    try:
        member = state["execution_member"]
        inventory = state["inventory"]
        item = next(item for item in inventory if item["ordinal"] == cell["ordinal"])
        if item["branch_identity"] != cell["branch_identity"]:
            raise ValueError("scheduled candidate differs from the frozen inventory")
        game = attempt / "runtime"
        c89.clone_player(Path(plan["player_root"]), game)
        c89.live_files.install_level(game, member)
        _, scenario = c89.live_files.materialize(member, member["template"],
                                                 attempt / "authority")
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("probe member differs from its frozen scenario authority")
        display, display_process = c89.start_display(attempt / "display.log")
        agent_port, game_port, physics_port = c89.reserve_ports(3)
        aligned = attempt / "aligned"
        os.environ.update(
            DISPLAY=display, XDG_DATA_HOME=str(attempt / "xdg"),
            NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port),
            NOVPHY_PHYSICS_CAPTURE_V2_STRIDE=str(NATIVE_STRIDE),
            NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
            NOVPHY_ENVIRONMENT_SEED=str(cell["engine_seed_second"]),
            NOVPHY_NATIVE_DECISION_STEP=str(DECISION_FIXED_STEP))
        record["ports"] = {"agent": agent_port, "game": game_port,
                           "physics": physics_port}
        engine_process = c89.start_engine(game, False, agent_port=agent_port,
                                          game_port=game_port,
                                          physics_port=physics_port)
        write_json(attempt / "runtime.json", record)
        bridge = c89.connect_with_retry(
            "127.0.0.1", agent_port, timeout=AGENT_SOCKET_SECONDS,
            deadline_seconds=AGENT_CONNECT_DEADLINE_SECONDS)
        physics = c89.ScienceBirdsBridge("127.0.0.1", physics_port,
                                         timeout=PHYSICS_SOCKET_SECONDS)
        bridge.configure(760001, c89.PlayingMode.TRAINING)
        bridge.set_speed(1)
        c89.prepare_for_play(bridge, timeout=PREPARE_FOR_PLAY_SECONDS, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("episode did not load its single assigned level")
        deadline = time.monotonic() + HISTORY_READY_SECONDS
        while not list(aligned.glob("decision-history-*/ready.json")):
            if time.monotonic() >= deadline:
                raise TimeoutError("native decision barrier readiness deadline; no retry")
            time.sleep(.1)
        _, rows = c89.read_history(aligned, DECISION_FIXED_STEP)
        policy = c89._ExposurePolicy()
        c89.expose_history(rows, attempt / "decision-1", scenario,
                           identity + ":decision", member["exposure_role"], policy)
        prepared = c89.prepare_action(bridge, item["action"])
        before = physics.get_observation_capture()
        time.sleep(INFERENCE_HOLD_SECONDS)
        after = physics.get_observation_capture()
        for name, frame in (("before", before), ("after", after)):
            write_json(attempt / f"paused-{name}.json", c89._plain_json(frame.metadata))
            (attempt / f"paused-{name}.png").write_bytes(frame.canonical_png)
        if (before.metadata["fixed_step"] != DECISION_FIXED_STEP
                or after.metadata["fixed_step"] != DECISION_FIXED_STEP
                or before.metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]
                or after.metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]
                or before.canonical_png != rows[-1]["canonical_png"]
                or after.canonical_png != rows[-1]["canonical_png"]):
            raise ValueError(c89.STABILITY_SIGNATURE)
        manifest = read_json(attempt / "decision-1" / c89.MANIFEST_NAME)
        frames = manifest["frame_records"]
        if frames[-1]["fixed_step"] != DECISION_FIXED_STEP:
            raise ValueError("decision trace lacks the sealed decision frame")
        ref = frames[-1]["agent_observation"]
        decision_frame_bytes = (attempt / "decision-1" / ref["relative_path"]).read_bytes()
        (attempt / "decision-frame.png").write_bytes(decision_frame_bytes)
        record["decision_frame"] = {
            "frame_identity": ref["identity"],
            "fixed_step": frames[-1]["fixed_step"],
            "fixed_time_seconds": frames[-1]["fixed_time_seconds"],
            "sha256": c89.bytes_identity(decision_frame_bytes),
            "observed_frames": policy.frames,
            "decision_compute": "none (frozen slot; re-scoring is a separate phase)",
        }
        retained_path = Path(cell["retained_frame"]["path"])
        if not retained_path.is_file():
            raise ValueError(f"retained decision frame missing: {retained_path}")
        retained_bytes = retained_path.read_bytes()
        record["frame_comparison"] = {
            "retained_frame_path": str(retained_path),
            "retained_sha256": c89.bytes_identity(retained_bytes),
            "second_seed_sha256": c89.bytes_identity(decision_frame_bytes),
            "byte_equal": retained_bytes == decision_frame_bytes,
        }
        segment_started = time.monotonic()
        # the segment authority declares the second seed: the native manifest's
        # recorded engine_seed is the positive proof of which seed was dispatched
        segment = c89.capture_segment(
            bridge, aligned, attempt / "shot-1",
            {**member, "engine_seed": cell["engine_seed_second"]},
            scenario, identity + ":shot-1", item["action"], SHOT_SECONDS)
        engine_seconds = time.monotonic() - segment_started
        initial_metadata = c89.live_files.read(
            Path(segment["native_root"]) / "frame_000001.json")
        initial_png = (Path(segment["native_root"]) / "frame_000001.png").read_bytes()
        if (segment["summary"]["first_fixed_step"] != DECISION_FIXED_STEP
                or initial_png != rows[-1]["canonical_png"]
                or initial_metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]):
            raise ValueError("executed shot pre-intervention state differs from its "
                             "decision frame")
        if initial_png != decision_frame_bytes:
            raise ValueError("sealed decision frame differs from the executed shot's "
                             "first frame")
        verdict = e85.channel_scan(segment["native_root"])
        if verdict["bird_launches"] != 1:
            raise ValueError("executed shot did not contain exactly one native launch")
        summary = segment["summary"]
        terminal = c89.terminal_evidence(segment)
        record["engine_channel"] = verdict
        record["execution"] = {
            "segment_identity": segment["identity"],
            "native_root": segment["native_root"],
            "observation_manifest": segment["observation_manifest"],
            "first_fixed_step": summary["first_fixed_step"],
            "last_fixed_step": summary["last_fixed_step"],
            "frame_count": summary["frame_count"],
            "censored": summary["censored"],
            "terminal_reason": None if terminal is None else terminal["reason"],
            "engine_wall_seconds": engine_seconds,
            "manifest_engine_seed": int(read_json(
                Path(segment["native_root"])
                / "native-manifest.json")["engine_seed"]),
        }
        record["outcome"] = {
            "first_shot_success": verdict["pig_removed"],
            "shots_to_success": 1 if verdict["pig_removed"] else None,
        }
        record["verdict_comparison"] = {
            "retained_verdict": bool(cell["retained_verdict"]),
            "second_seed_verdict": bool(verdict["pig_removed"]),
            "agrees": bool(verdict["pig_removed"]) == bool(cell["retained_verdict"]),
        }
    except Exception as error:  # typed terminal failure; never retried silently
        record["failure"] = f"{type(error).__name__}: {error}"
        if record["failure_kind"] is None:
            record["failure_kind"] = "execution_failure"
    finally:
        actions = [("connection.disconnect", connection.disconnect)
                   for connection in (bridge, physics) if connection is not None]
        actions.append(("stop_started_engine",
                        lambda: c89.stop_started_engine(engine_process)))
        if display_process is not None:
            actions.append(("display.terminate",
                            lambda: c89.terminate(display_process)))

        def restore_environment():
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        actions.append(("environment.restore", restore_environment))
        cleanup_failures = []
        for name, action in actions:
            try:
                action()
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    f"{name}: {type(cleanup_error).__name__}: {cleanup_error}")
        if cleanup_failures:
            record["cleanup_failures"] = cleanup_failures
    record["wall_seconds"] = time.monotonic() - started
    write_json(output / "records" / f"{identity}.json", record)
    print(f"issue-92 cross-seed cell {identity} complete={record['failure'] is None} "
          f"failure={record['failure']}", flush=True)
    return record


# ---------------------------------------------------------------------------
# serialized supervisor (one isolated worker at a time; stop rule per cell)
# ---------------------------------------------------------------------------

def ledger_path(output):
    return Path(output) / "ledger.json"


def load_ledger(output, plan, phase):
    path = ledger_path(output)
    if path.is_file():
        return read_json(path)
    return {"schema": "issue_92_cross_seed_invariance_ledger_v1", "identity": IDENTITY,
            "phase": phase, "status": "running", "cells": {},
            "wall_seconds_elapsed": 0.0, "gpu_seconds_elapsed": 0.0,
            "stop_reason": None}


def free_disk(output):
    usage = os.statvfs(Path(output).mkdir(parents=True, exist_ok=True) or output)
    return usage.f_bavail * usage.f_frsize


def supervise(output, plan, cells, phase, plan_path=None):
    """Serialized dispatch: one isolated worker at a time (cold starts cannot
    collide); the frozen verdict-flip stop rule is checked after every cell."""
    from scripts import run_closed_loop_oracle_completion as c89
    output = Path(output)
    ledger = load_ledger(output, plan, phase)
    ledger["phase"] = phase
    write_json(ledger_path(output), ledger)
    pending = [cell for cell in cells
               if not (output / "records" / f"{cell['identity']}.json").is_file()]
    if not pending:
        ledger["status"] = "complete"
        write_json(ledger_path(output), ledger)
        log(f"{phase}: all {len(cells)} scheduled cells already terminal")
        return ledger
    player = Path(plan["player_root"])
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")
    state_by_identity = {state["identity"]: state
                         for state in load_inputs()[0]["states"]}
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise RuntimeError("supervisor received SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index, cell in enumerate(pending):
            elapsed = ledger["wall_seconds_elapsed"] + time.monotonic() - started
            if ledger["stop_reason"] is None and elapsed >= WALL_CAP_SECONDS:
                ledger["stop_reason"] = "wall_cap_exceeded"
            if ledger["stop_reason"] is None and free_disk(output) < MINIMUM_FREE_BYTES:
                ledger["stop_reason"] = "minimum_free_storage"
            if ledger["stop_reason"] is None and file_bytes(output) > ARTIFACT_BYTES_CAP:
                ledger["stop_reason"] = "artifact_limit"
            if ledger["stop_reason"] is not None:
                log(f"cap stop: {ledger['stop_reason']}; remaining cells are retained "
                    "as typed not_executed (no replacement)")
                break
            identity = cell["identity"]
            payload = {"output": str(output), "cell": cell,
                       "state": state_by_identity[cell["state"]],
                       "plan_path": str(plan_path or (output / "plan.json"))}
            (output / "markers").mkdir(parents=True, exist_ok=True)
            write_json(output / "markers" / f"{identity}.json",
                       {"identity": identity, "phase": phase,
                        "dispatched_at_utc": utc_now()})
            process = c89.start_isolated_worker(context, execute_cell, (payload,))
            cell_started = time.monotonic()
            peak_rss = 0.0
            stop = None
            while process.is_alive():
                time.sleep(1)
                rss = c89.process_rss(process.pid)
                peak_rss = max(peak_rss, rss)
                if time.monotonic() - cell_started >= ATTEMPT_SECONDS:
                    stop = "attempt_wall_limit"
                elif peak_rss > WORKER_RSS_MIB:
                    stop = "worker_memory_limit"
                if stop is not None:
                    break
            wall = time.monotonic() - cell_started
            exitcode = process.exitcode
            c89.terminate_worker(process)
            record_file = output / "records" / f"{identity}.json"
            if not record_file.is_file():
                failure = stop or f"worker_exitcode={exitcode}"
                write_json(record_file, _plain_record(plan, cell)
                           | {"failure": failure, "failure_kind": "execution_failure"})
            record = read_json(record_file)
            ledger["cells"][identity] = {
                "status": "complete" if (stop is None and exitcode == 0
                                         and record["failure"] is None) else "failed",
                "stop": stop, "exit_code": exitcode, "wall_seconds": wall,
                "failure": record.get("failure"),
                "verdict_agrees": (record.get("verdict_comparison") or {}).get("agrees"),
            }
            ledger["wall_seconds_elapsed"] += wall
            write_json(output / "receipts" / f"{identity}.json", {
                "identity": identity, "phase": phase, "worker_exitcode": exitcode,
                "stop": stop, "wall_seconds": wall, "peak_cpu_rss_mib": peak_rss})
            write_json(ledger_path(output), ledger)
            done = len(ledger["cells"])
            log(f"{phase} {done}/{len(cells)} cell {identity} "
                f"verdict_agrees={ledger['cells'][identity]['verdict_agrees']} "
                f"failure={record.get('failure')} "
                f"elapsed={ledger['wall_seconds_elapsed'] / 60:.0f}m")
            comparison = record.get("verdict_comparison")
            if phase == "run" and comparison is not None and not comparison["agrees"]:
                ledger["stop_reason"] = "verdict_flip"
                ledger["flip_evidence"] = {
                    "cell": identity, "kind": cell["kind"],
                    "retained_verdict": comparison["retained_verdict"],
                    "second_seed_verdict": comparison["second_seed_verdict"]}
                write_json(ledger_path(output), ledger)
                log(f"STOP RULE: verdict flip on {identity}; remaining cells typed "
                    "not_executed; determinism is falsified")
                break
        ledger["status"] = ("complete" if ledger["stop_reason"] is None
                            else ledger["stop_reason"])
        write_json(ledger_path(output), ledger)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    return ledger


# ---------------------------------------------------------------------------
# smoke (development cells; infrastructure assertions only)
# ---------------------------------------------------------------------------

def run_smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log("smoke: 2 development cells at the second seed; infrastructure assertions "
        "only (agreement is the experiment's result, never asserted at smoke)")
    plan87, decisions, summary89, verdicts, oracles87, records89 = load_inputs()
    slots, smoke = enumerate_slots(plan87, decisions, summary89, verdicts,
                                   oracles87, records89)
    pseudo_plan = {"identity": IDENTITY, "version": 0,
                   "player_root": str(ROOT / ".local-artifacts/issue-77-n1-v1/player")}
    write_json(output / "smoke-plan.json", pseudo_plan)
    supervise(output, pseudo_plan, smoke, "smoke",
              plan_path=output / "smoke-plan.json")
    checks = []
    for cell in smoke:
        record = read_json(output / "records" / f"{cell['identity']}.json")
        checks.append({
            "cell": cell["identity"], "kind": cell["kind"],
            "verdict_produced": record.get("outcome") is not None,
            "frame_captured": record.get("decision_frame") is not None,
            "channel_scan_complete": record.get("engine_channel") is not None,
            "frame_comparison_recorded": record.get("frame_comparison") is not None,
            "failure": record.get("failure"),
            "engine_seed_second": cell["engine_seed_second"],
            "second_seed_verdict": (record.get("verdict_comparison") or {}).get(
                "second_seed_verdict"),
            "frame_byte_equal": (record.get("frame_comparison") or {}).get("byte_equal"),
            "wall_seconds": record.get("wall_seconds"),
            "ok": (record.get("outcome") is not None
                   and record.get("decision_frame") is not None
                   and record.get("engine_channel") is not None
                   and record.get("frame_comparison") is not None)})
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": "issue_92_cross_seed_invariance_smoke_v1",
                "identity": IDENTITY, "run_at": utc_now(), "checks": checks, "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; evidence at {output / 'smoke.json'}")
    if not ok:
        raise ValueError("smoke infrastructure assertion failed; abort before the "
                         "terminal freeze")
    return evidence


def prepare(output):
    output = Path(output)
    plan_path = output / "plan.json"
    if plan_path.is_file():
        frozen = load_plan(plan_path)
        log(f"existing frozen protocol validated (version {frozen['version']}, frozen "
            f"{frozen['frozen_at']}); no experiment cell was executed")
        return frozen
    smoke_path = output / "smoke.json"
    if not smoke_path.is_file():
        raise ValueError("smoke.json missing; run --smoke before --prepare")
    smoke_evidence = read_json(smoke_path)
    if not smoke_evidence.get("ok"):
        raise ValueError("smoke assertions failed; no terminal freeze")
    plan87, decisions, summary89, verdicts, oracles87, records89 = load_inputs()
    slots, smoke = enumerate_slots(plan87, decisions, summary89, verdicts,
                                   oracles87, records89)
    frozen = frozen_plan(utc_now(), slots, smoke, smoke_evidence)
    write_json(plan_path, frozen)
    log(f"frozen protocol published at {plan_path}: version {frozen['version']}, "
        f"{len(slots)} slots; no experiment cell was executed")
    return frozen


# ---------------------------------------------------------------------------
# selection re-scoring on both frames (DESCRIPTIVE)
# ---------------------------------------------------------------------------

def rescore(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    ledger = load_ledger(output, plan, "run")
    records = {cell["identity"]: read_json(output / "records" / f"{cell['identity']}.json")
               for cell in plan["universe"]["slots"]
               if (output / "records" / f"{cell['identity']}.json").is_file()}
    frames = {}
    for cell in plan["universe"]["slots"]:
        record = records.get(cell["identity"])
        if record is None or record.get("decision_frame") is None:
            continue
        state = cell["state"]
        if state not in frames:
            frames[state] = {"seed2": output / "attempts" / cell["identity"]
                             / "decision-frame.png"}
    plan87, decisions, summary89, verdicts, oracles87, records89 = load_inputs()
    from scripts import run_lookahead_control as wlc
    anchors = wlc.anchors_for([state for state in plan87["states"]
                               if state["identity"] in frames], decisions)
    adapter = wlc.load_adapter(DEVICE)
    objective = wlc.load_objective()
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    models = {}
    gpu_seconds = 0.0
    cells_done = 0
    for state_identity in sorted(frames):
        for tag, path in (("seed1", Path(anchors[state_identity]["frame_path"])),
                          ("seed2", frames[state_identity]["seed2"])):
            anchor = dict(anchors[state_identity])
            anchor["frame_path"] = path
            carrier = wlc.encode_carrier(adapter, anchor, DEVICE)
            state = next(item for item in plan87["states"]
                         if item["identity"] == state_identity)
            for system in SYSTEMS:
                family = system.split("-")[0]
                for seed in SEEDS:
                    identity = f"rescore--{state_identity}--{tag}--{system}--seed{seed}"
                    record_path = output / "rescore" / f"{identity}.json"
                    if record_path.is_file():
                        cells_done += 1
                        continue
                    key = (seed, family)
                    if key not in models:
                        models[key] = wlc.load_predictor(plan87, seed, family, DEVICE)
                    selector = e85.ReactiveSelector(system, seed, models[key],
                                                    objective, DEVICE)
                    began = time.monotonic()
                    with wlc.torch.no_grad():
                        decision = selector.choose(carrier, state["inventory"])
                    gpu_seconds += time.monotonic() - began
                    record = {
                        "schema": SCHEMA_RESCORE, "plan_identity": IDENTITY,
                        "plan_version": plan["version"],
                        "cell": {"identity": identity, "state": state_identity,
                                 "frame": tag, "system": system, "seed": seed},
                        "frame_sha256": anchors[state_identity]["sha256"]
                        if tag == "seed1" else records_sha(output, frames, state_identity),
                        "chosen_ordinal": decision["chosen"]["ordinal"],
                        "selected_predicted_cost": decision["chosen"]["predicted_cost"],
                        "issue_64_authorized": False,
                    }
                    write_json(record_path, record)
                    cells_done += 1
                    if cells_done % 50 == 0:
                        log(f"rescore {cells_done} cells written; gpu "
                            f"{gpu_seconds:.0f}s")
    ledger["gpu_seconds_elapsed"] += gpu_seconds
    write_json(ledger_path(output), ledger)
    if ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: gpu_cap_exceeded")
    log(f"rescore complete: {cells_done} cells, gpu {gpu_seconds:.0f}s")
    return gpu_seconds


def records_sha(output, frames, state_identity):
    from hashlib import sha256
    digest = sha256()
    digest.update(Path(frames[state_identity]["seed2"]).read_bytes())
    return f"sha256:{digest.hexdigest()}"


# ---------------------------------------------------------------------------
# published tables
# ---------------------------------------------------------------------------

def compute_tables(plan, output):
    output = Path(output)
    ledger = load_ledger(output, plan, "run")
    slots = plan["universe"]["slots"]
    records = {}
    for cell in slots:
        path = output / "records" / f"{cell['identity']}.json"
        if path.is_file():
            record = read_json(path)
            if record.get("schema") != SCHEMA_RECORD:
                raise ValueError(f"record {path} schema binding differs")
            records[cell["identity"]] = record
    per_slot = []
    flips = []
    verdicts_present = 0
    comparisons_present = 0
    frame_equal = 0
    for cell in slots:
        record = records.get(cell["identity"])
        row = {"identity": cell["identity"], "kind": cell["kind"],
               "state": cell["state"], "member": cell["member"],
               "ordinal": cell["ordinal"],
               "engine_seed_first": cell["engine_seed_first"],
               "engine_seed_second": cell["engine_seed_second"],
               "retained_verdict": cell["retained_verdict"]}
        if record is None:
            row.update({"status": "not_executed"})
        elif record.get("failure") is not None:
            row.update({"status": "typed_failure", "failure": record["failure"]})
        else:
            comparison = record["verdict_comparison"]
            verdicts_present += 1
            comparisons_present += int(record.get("frame_comparison") is not None)
            frame_equal += int(record["frame_comparison"]["byte_equal"])
            row.update({"status": "executed",
                        "second_seed_verdict": comparison["second_seed_verdict"],
                        "agrees": comparison["agrees"],
                        "frame_byte_equal": record["frame_comparison"]["byte_equal"],
                        "engine_wall_seconds": record["execution"]["engine_wall_seconds"]})
            if not comparison["agrees"]:
                flips.append({"identity": cell["identity"], "kind": cell["kind"],
                              "retained_verdict": comparison["retained_verdict"],
                              "second_seed_verdict": comparison["second_seed_verdict"]})
        per_slot.append(row)
    agreement = sum(1 for row in per_slot if row.get("agrees"))
    executed = sum(1 for row in per_slot if row["status"] == "executed")

    rescore_rows = []
    rescore_dir = output / "rescore"
    if rescore_dir.is_dir():
        by_cell = {}
        for path in sorted(rescore_dir.glob("rescore--*.json")):
            record = read_json(path)
            if record.get("schema") != SCHEMA_RESCORE:
                raise ValueError(f"rescore record {path} schema binding differs")
            cell = record["cell"]
            by_cell[(cell["state"], cell["frame"], cell["system"], cell["seed"])] = \
                record["chosen_ordinal"]
        states = sorted({key[0] for key in by_cell})
        for system in SYSTEMS:
            for seed in SEEDS:
                pairs = [(by_cell.get((state, "seed1", system, seed)),
                          by_cell.get((state, "seed2", system, seed)))
                         for state in states]
                scored = [(a, b) for a, b in pairs if a is not None and b is not None]
                rescore_rows.append({
                    "system": system, "seed": seed, "states": len(scored),
                    "chosen_agrees": sum(1 for a, b in scored if a == b),
                    "label": "DESCRIPTIVE"})
    guards = {
        "slots_executed": {"value": executed, "floor": 26, "ok": executed == 26},
        "verdicts_present": {"value": verdicts_present, "floor": 26,
                             "ok": verdicts_present == 26},
        "frame_comparisons_present": {"value": comparisons_present, "floor": 26,
                                      "ok": comparisons_present == 26},
    }
    guards_ok = all(guard["ok"] for guard in guards.values())
    if flips:
        disposition = "not_supported_by_this_experiment"
    elif not guards_ok or ledger.get("stop_reason") not in (None, "verdict_flip"):
        disposition = STOP_TOKEN
    else:
        disposition = "supported"
    return {
        "schema": SCHEMA_COMPUTE,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "per_slot": per_slot,
        "agreement": {"agree": agreement, "slots": 26},
        "flips": flips,
        "capture_channel": {
            "frame_byte_equal": frame_equal, "executed": executed,
            "reading": ("frames differing while verdicts agree is the informative "
                        "separation (selection channel noisy, outcome channel "
                        "invariant); byte-identical frames are the stronger "
                        "invariance outcome; DESCRIPTIVE")},
        "selection_rescore": {"rows": rescore_rows, "label": "DESCRIPTIVE"},
        "precision_guards": guards,
        "disposition": disposition,
        "disposition_conditions": {"flips": len(flips), "guards_ok": guards_ok,
                                   "stop_reason": ledger.get("stop_reason")},
        "restatement_obligation": (
            "a flipped verdict falsifies determinism and forces a seed-conditional "
            "restatement of the ceiling" if flips else None),
    }


def slot_join_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["identity", "kind", "state", "member", "ordinal",
                     "engine_seed_first", "engine_seed_second", "retained_verdict",
                     "second_seed_verdict", "agrees", "frame_byte_equal", "status",
                     "failure", "engine_wall_seconds"])
    for row in compute["per_slot"]:
        writer.writerow([row["identity"], row["kind"], row["state"], row["member"],
                         row["ordinal"], row["engine_seed_first"],
                         row["engine_seed_second"], row["retained_verdict"],
                         row.get("second_seed_verdict", ""),
                         row.get("agrees", ""), row.get("frame_byte_equal", ""),
                         row["status"], row.get("failure", ""),
                         f"{row['engine_wall_seconds']:.1f}"
                         if "engine_wall_seconds" in row else ""])
    return buffer.getvalue()


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["measurement", "value", "detail", "label"])
    writer.writerow(["verdict_agreement",
                     f"{compute['agreement']['agree']}/{compute['agreement']['slots']}",
                     "second-seed vs retained union-table verdicts", "DECISION"])
    writer.writerow(["flips", len(compute["flips"]),
                     json.dumps(compute["flips"], sort_keys=True), "DECISION"])
    writer.writerow(["frame_byte_equal",
                     f"{compute['capture_channel']['frame_byte_equal']}/"
                     f"{compute['capture_channel']['executed']}",
                     "second-seed decision frame vs retained frame", "DESCRIPTIVE"])
    for row in compute["selection_rescore"]["rows"]:
        writer.writerow([f"selection_agreement:{row['system']}:seed{row['seed']}",
                         f"{row['chosen_agrees']}/{row['states']}",
                         "chosen-ordinal agreement across frames", "DESCRIPTIVE"])
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    add("# Issue-92 WP2b: cross-seed invariance probe — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan schema `{SCHEMA_PLAN}` version "
        f"{plan['version']} (terminal), frozen {plan['frozen_at']} before any "
        "experiment cell")
    add(f"- validation command: `{VALIDATION_COMMAND}`")
    add(f"- second seed rule: engine_seed + {SECOND_SEED_OFFSET} "
        "(NOVPHY_ENVIRONMENT_SEED initializes the Unity RNG only; ORACLE_SEED "
        "20260908 is a protocol label, never an engine input)")
    add("")
    add("## 1. Verdict agreement (the decision)")
    add("")
    add(f"**{compute['agreement']['agree']} of {compute['agreement']['slots']}** "
        f"second-seed verdicts agree with the retained union-table verdicts; flips: "
        f"{len(compute['flips'])}.")
    add("")
    add("| kind | slot | retained | second seed | agrees | frame byte-equal |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in compute["per_slot"]:
        add(f"| {row['kind']} | {row['identity']} | {row['retained_verdict']} | "
            f"{row.get('second_seed_verdict', row['status'])} | "
            f"{row.get('agrees', '')} | {row.get('frame_byte_equal', '')} |")
    add("")
    add("## 2. Capture channel (DESCRIPTIVE)")
    add("")
    add(f"Decision frames byte-identical to the retained first-seed frames: "
        f"{compute['capture_channel']['frame_byte_equal']} of "
        f"{compute['capture_channel']['executed']} executed slots. "
        f"{compute['capture_channel']['reading']}.")
    add("")
    add("## 3. Selection re-scored on both frames (DESCRIPTIVE)")
    add("")
    add("| system | seed | states | chosen-ordinal agreement |")
    add("| --- | --- | --- | --- |")
    for row in compute["selection_rescore"]["rows"]:
        add(f"| {row['system']} | {row['seed']} | {row['states']} | "
            f"{row['chosen_agrees']}/{row['states']} |")
    add("")
    add("## 4. Disposition")
    add("")
    add(f"**{compute['disposition']}** (flips {compute['disposition_conditions']['flips']}; "
        f"guards ok {compute['disposition_conditions']['guards_ok']}; stop reason "
        f"{compute['disposition_conditions']['stop_reason']}).")
    if compute["restatement_obligation"]:
        add("")
        add(f"Restatement obligation: {compute['restatement_obligation']}.")
    add("")
    add("## 5. Claim boundary")
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
        "question": plan["question"],
        "motivation": plan["motivation"],
        "agreement": compute["agreement"],
        "flips": compute["flips"],
        "capture_channel": compute["capture_channel"],
        "selection_rescore": compute["selection_rescore"],
        "precision_guards": compute["precision_guards"],
        "disposition": compute["disposition"],
        "disposition_conditions": compute["disposition_conditions"],
        "restatement_obligation": compute["restatement_obligation"],
        "seed_disclosure": plan["second_seed"]["seed_label_disclosure"],
        "smoke_evidence": plan["smoke"]["evidence"],
        "caps": plan["caps"],
        "claim_boundary": plan["claim_boundary"],
        "limitations": plan["limitations"],
        "compute": {"phases": timings,
                    "wall_seconds_total": sum(entry["seconds"]
                                              for entry in timings.values())},
        "hygiene": {"content_hashes_recomputed_after_freeze": False,
                    "full_corpus_integrity_pass": False},
    }


def publish(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    ledger = load_ledger(output, plan, "run")
    started = time.monotonic()
    compute = compute_tables(plan, output)
    timing = {"seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    timings = {"tables": timing}
    compute["phases"] = timings
    report = build_report(plan, compute, timings)
    write_json(output / "compute.json", compute)
    write_json(output / "summary.json", report)
    write_text(output / "comparisons.csv", comparisons_csv(compute))
    write_text(output / "findings.md", findings_md(plan, compute))
    write_text(output / "slot_join.csv", slot_join_csv(compute))
    write_json(output / "receipts.json", {
        "schema": SCHEMA_RECEIPTS, "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "ledger": {"wall_seconds_elapsed": ledger["wall_seconds_elapsed"],
                   "gpu_seconds_elapsed": ledger["gpu_seconds_elapsed"],
                   "cells": len(ledger["cells"]),
                   "stop_reason": ledger.get("stop_reason")},
        "phases": timings,
        "derived_bytes": file_bytes(output),
        "artifact_bytes_cap": ARTIFACT_BYTES_CAP,
    })
    log(f"published summary.json, comparisons.csv, findings.md, slot_join.csv and "
        f"receipts.json in {output}")
    return report


def validate(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    began = time.monotonic()
    fresh = compute_tables(plan, output)
    fresh["phases"] = read_json(output / "compute.json")["phases"]
    published = read_json(output / "compute.json")
    if published != fresh:
        for key in sorted(set(published) | set(fresh)):
            if published.get(key) != fresh.get(key):
                raise ValueError(f"compute.json section {key!r} differs from the fresh "
                                 "recomputation")
    timings = published["phases"]
    regenerated = {
        "summary.json": json_text(build_report(plan, fresh, timings)).encode(),
        "comparisons.csv": comparisons_csv(fresh).encode(),
        "findings.md": findings_md(plan, fresh).encode(),
        "slot_join.csv": slot_join_csv(fresh).encode(),
    }
    for name, content in regenerated.items():
        path = output / name
        if not path.is_file():
            raise ValueError(f"published artifact missing: {name}")
        if path.read_bytes() != content:
            raise ValueError(f"published {name} differs from the recomputation from the "
                             "retained records")
    log(f"exact recomputation validation passed: 26 slots re-derived from the "
        f"retained records, every published table byte-compared; no engine, no fresh "
        f"rollout, no content hash recomputed ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    log("no-write dry run; structural inventory only; no statistic")
    plan87, decisions, summary89, verdicts, oracles87, records89 = load_inputs()
    slots, smoke = enumerate_slots(plan87, decisions, summary89, verdicts,
                                   oracles87, records89)
    kinds = Counter(cell["kind"] for cell in slots)
    log(f"composition: {len(slots)} slots {dict(sorted(kinds.items()))}; smoke cells: "
        f"{[cell['identity'] for cell in smoke]}")
    missing = [cell["identity"] for cell in slots + smoke
               if not Path(cell["retained_frame"]["path"]).is_file()]
    log(f"retained decision frames present for {len(slots + smoke) - len(missing)}/"
        f"{len(slots + smoke)} cells" + (f"; MISSING: {missing}" if missing else ""))
    player = ROOT / ".local-artifacts/issue-77-n1-v1/player"
    log(f"player complete: {(player / '9001-player.x86_64').is_file() and (player / 'game_playing_interface.jar').is_file()}")
    log(f"second seed rule: engine_seed + {SECOND_SEED_OFFSET} (e.g. "
        f"{slots[0]['engine_seed_first']} -> {slots[0]['engine_seed_second']})")
    plan_path = Path(output) / "plan.json"
    if plan_path.is_file():
        plan = load_plan(plan_path)
        log(f"frozen protocol present: version {plan['version']} frozen "
            f"{plan['frozen_at']}")
    else:
        log(f"frozen protocol absent; smoke evidence present: "
            f"{(Path(output) / 'smoke.json').is_file()}")
    log("dry run complete; nothing was written and no statistic was computed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke", "prepare", "run", "rescore", "publish",
                 "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        if args.dry_run:
            return dry_run(args.output)
        if args.smoke:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            with GPULock():
                run_smoke(args.output)
            return 0
        if args.prepare:
            prepare(args.output)
            return 0
        if args.run:
            plan = load_terminal_plan(args.output)
            with GPULock():
                supervise(Path(args.output), plan, plan["universe"]["slots"], "run")
            return 0
        if args.rescore:
            load_terminal_plan(args.output)
            with GPULock():
                rescore(args.output)
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
