"""Issue-85 ADD-EXP: engine-side closed-loop outcome channel and bounded
reactive re-diagnostic (unblocks #80/#82).

Bounded diagnostic (non-final, development data only) over the #80 frozen
24-state membership x 4 systems x 3 seeds, closed-loop in the real rendered
engine, with the ENGINE-SIDE first-shot success predicate:

- Phase A (blocking): the channel. Frozen detection rule over the game's own
  state stream (pig death/destruction macro events + the fixed-step entity
  lifecycle stream), verified before any diagnostic execution on a bounded
  rendered control set with retained frames/WebMs (mode --smoke-test). If the
  channel cannot be verified the ticket publishes
  readiness_or_precision_insufficient naming the blocker - terminal, complete.
- Phase B: the bounded re-diagnostic. Pilot gate (4 states x 4 systems x 3
  seeds = 48 executions; pooled ENGINE-TRUTH first-shot success prevalence
  >= 0.10, minimum 24 valid executions) -> full 288-cell matrix only if the
  gate clears; per-seed and pooled descriptive paired bootstrap over states.
  The no-model ordinal prior arm now carries the #82-recorded fallback rule
  (nearest admissible ordinal to 8, ties to lower), pre-declared in the
  freeze; the #80 pilot record is never amended.
- Phase C: on the pilot executions, the #82 replay-cost proxy predicate
  (realized end-of-window count cost <= 81.68825840950012, the published
  candidate threshold) is computed alongside engine truth and the
  agreement/divergence table is published - the instrumentation lesson.

Caps: 24 worker-hours wall across the pilot+run execution phases; <= 3
GPU-hours of active decision work inside that wall cap; exceeding either
stops with readiness_or_precision_insufficient. Every scheduled cell is
executed or retained as a typed terminal failure; no outcome-conditioned
exclusion, retry, replacement, or re-freeze. Zero fresh captures; zero
sealed/final lineage access; #64/#65 stay unauthorized; the #80/#82 published
artifacts are reused READ-ONLY and never re-run or amended.

Exact validation command: python -u -m scripts.run_engine_outcome_reactive_diagnostic --validate
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
from hashlib import sha256
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import torch

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_72_matched_grid as grid
from scripts import run_issue_77_n1_train as train77
from scripts.issue_76_censored_episode import capture_segment
from scripts.issue_76_display_start import start_display
from scripts.issue_76_episode_capture import prepare_action
from scripts.issue_76_live_episode import files as live_files
from scripts.issue_76_native_outcomes import terminal_evidence
from scripts.issue_76_shared_history_capture import expose_history, read_history
from scripts.issue_76_shared_player_storage import clone_player
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import MANIFEST_NAME, _plain_json
from scripts.run_issue_62_successor_cohort import (
    PILOT_AUDIT_FPS,
    PlayingMode,
    _encode_agent_frames_webm,
    _verify_webm_encoder,
    connect_with_retry,
    free_port,
    prepare_for_play,
    start_engine,
    stop_started_engine,
    terminate,
)
from src.webui.bridge import ScienceBirdsBridge
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.planning.task_objective import TaskObjective
from world_model.training import engine_outcome_reactive_diagnostic as engine
from world_model.training.lineage_scaling import (
    NO_MODEL_ORDINAL_PRIOR_ORDINAL,
    GameplayPlanningMode,
    fixed_mode_pair,
)
from world_model.training.matched_dynamics import (
    ContinuousDynamics,
    CNNHybridPredictor,
    DIM,
    active_capacity,
    pairs_for,
    work as pair_work,
)

ROOT = train77.ROOT
CAMPAIGN = train77.CAMPAIGN
DYNAMICS = train77.OUTPUT
EIGHTY = ROOT / ".local-artifacts/issue-80-reactive-diagnostic-v1"
EIGHTY_IDENTITY = "issue-80-reactive-diagnostic-v1"
EIGHTY2 = ROOT / ".local-artifacts/issue-82-zero-floor-oracle-v1"
ORACLE_IDENTITY = "issue-82-zero-floor-oracle-v1"
OUTPUT = ROOT / ".local-artifacts/issue-85-engine-outcome-reactive-v1"
DATA = ROOT / "data/issue-85-engine-outcome-reactive"
SCHEMA = engine.SCHEMA_RECORD
IDENTITY = engine.IDENTITY
SEEDS = train77.SEEDS
STATES_PER_FAMILY = 12
PILOT_STATES_PER_FAMILY = 2
FAMILIES = ("type010103", "type010105")
MODEL_SYSTEMS = engine.MODEL_SYSTEMS
PRIOR_SYSTEM = engine.PRIOR_SYSTEM
SYSTEMS = engine.SYSTEMS
SYSTEM_MODES = {
    "hybrid-fixed-h1": GameplayPlanningMode.HYBRID_FIXED,
    "continuous-fixed-h1": GameplayPlanningMode.CONTINUOUS_H1,
    "continuous-fixed-h5": GameplayPlanningMode.CONTINUOUS_H5,
}
SYSTEM_ARMS = {name: ("hybrid" if mode is GameplayPlanningMode.HYBRID_FIXED else "continuous",
                      fixed_mode_pair(mode))
               for name, mode in SYSTEM_MODES.items()}
PRIOR_ORDINAL = engine.PRIOR_ORDINAL
if PRIOR_ORDINAL != NO_MODEL_ORDINAL_PRIOR_ORDINAL:
    raise ValueError("prior ordinal disagrees between the frozen modules")
WALL_CAP_SECONDS = 24 * 3600
GPU_CAP_SECONDS = 3 * 3600
WORKERS = 4
WORKER_RSS_MIB = 4096
AGGREGATE_RSS_MIB = 32768
ATTEMPT_SECONDS = 1200
MINIMUM_FREE_BYTES = 256 * 2 ** 30
ARTIFACT_BYTES = 150 * 2 ** 30
DECISION_FIXED_STEP = 30000
SHOT_SECONDS = 180
HISTORY_READY_SECONDS = 90
INFERENCE_HOLD_SECONDS = 2
NATIVE_STRIDE = 50
SHOT_WINDOW_NATIVE = 30000
END_OFFSET = 600
DECISION_DEVICE = "cuda"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"
FILES = (
    "scripts/run_engine_outcome_reactive_diagnostic.py",
    "world_model/training/engine_outcome_reactive_diagnostic.py",
    "world_model/training/lineage_scaling.py",
)
PRIMARY_CONTRAST = ("regret", "continuous-fixed-h5")
SUCCESS_PRIMARY_CONTRAST = ("success", "continuous-fixed-h5")


def log(message):
    print(f"[issue-85-engine-outcome] {message}", flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def cell_identity(system, seed, state):
    return f"{system}--seed{seed}--{state}"


def gpu_lock():
    """Open and exclusively lock the shared-GPU contract file for this phase."""
    handle = open(GPU_LOCK_PATH, "a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def file_identity(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- membership

def membership_plan():
    """The #80 frozen 24-state membership plan, validated by identity."""
    path = EIGHTY / "plan.json"
    plan = read(path)
    if (plan.get("identity") != EIGHTY_IDENTITY
            or plan.get("frozen_before_outcome") is not True):
        raise ValueError("requires the frozen issue-80 reactive-diagnostic plan")
    return plan


def states_from_membership(eighty_plan):
    """The frozen states, verbatim, plus this ticket's prior-arm fallback choice."""
    states = deepcopy(eighty_plan["states"])
    if len(states) != 24:
        raise ValueError("the frozen #80 membership is not 24 states")
    for state in states:
        ordinals = [item["ordinal"] for item in state["inventory"]]
        state["prior_choice"] = engine.prior_fallback_choice(ordinals)
    families = {family: sum(1 for state in states
                            if state["generator_family"] == family)
                for family in FAMILIES}
    if families != {"type010103": STATES_PER_FAMILY, "type010105": STATES_PER_FAMILY}:
        raise ValueError(f"frozen membership family counts differ: {families}")
    return states


def recompute_pilot_states(states):
    """The #80 pilot rule: two lowest-identity states per family with distinct
    source members, so the pilot samples four distinct physical initial states."""
    pilot_states = []
    for family in FAMILIES:
        picked_members = set()
        for state in sorted((state for state in states
                             if state["generator_family"] == family),
                            key=lambda state: state["identity"]):
            if state["source_member"] in picked_members:
                continue
            pilot_states.append(state["identity"])
            picked_members.add(state["source_member"])
            if len(picked_members) >= PILOT_STATES_PER_FAMILY:
                break
    pilot_states.sort()
    return pilot_states


# ---------------------------------------------------------------- plan freeze

def definitions():
    return {
        "states_per_family": STATES_PER_FAMILY,
        "families": {"type010103": "rolling", "type010105": "sliding"},
        "membership_rule": (
            "the #80 frozen 24-state membership verbatim (12 type010103 + 12 "
            "type010105; identities, inventories, engine seeds and scenario "
            "authorities bound by the issue-80 plan file identity), re-executed "
            f"as gameplay; prior arm choice per state = {PRIOR_FALLBACK_RULE_SHORT}"),
        "seeds": list(SEEDS),
        "systems": {name: ({"kind": "model", "arm": arm,
                            "pair": {"delta": pair.delta, "abstraction": str(pair.abstraction)}}
                           if arm else {"kind": "no-model-ordinal-prior",
                                        "prior_ordinal": PRIOR_ORDINAL,
                                        "fallback_rule": engine.PRIOR_FALLBACK_RULE})
                    for name, (arm, pair) in {**SYSTEM_ARMS, PRIOR_SYSTEM: (None, None)}.items()},
        "decision": (
            "reactive, identical to #80: parse the sealed pre-decision frame "
            "with the frozen issue-70 adapter as a single-frame temporal "
            "context, roll every admissible candidate to the system's fixed "
            "pair, choose argmin predicted count cost with ties broken by "
            "lower ordinal; candidates with nonfinite predictions are "
            "excluded; the prior arm executes its pre-declared fallback "
            "ordinal and parses the frame for identical perception accounting; "
            "a typed decision failure is terminal and executes no shot"),
        "action_encoding": "drag/480, true 1000 ms hold, tap/1000, bias 1 (equals training)",
        "engine_channel": {
            "detection_rule": engine.CHANNEL_DETECTION_RULE,
            "failure_modes": engine.CHANNEL_FAILURE_MODES,
            "first_shot_success": (
                "the executed shot segment's engine-side channel verdict "
                "pig_removed (the game's own pig death/destruction events "
                "during rendered play); typed failures carry no success value"),
            "shots_to_success": (
                "single-shot protocol: 1 when the first shot succeeds, "
                "otherwise null; no multi-shot competence claim"),
            "no_parser_reliance": (
                "the verdict never reads parser-derived cost; the realized "
                "end-of-window count cost is recorded only as accounting "
                "context and for the Phase-C proxy cross-check"),
        },
        "verification_set": {
            "seed": engine.VERIFICATION_SEED,
            "pass_rule": engine.VERIFICATION_PASS_RULE,
            "review_rule": engine.VERIFICATION_REVIEW_RULE,
            "media_review": {"schema": engine.MEDIA_REVIEW_SCHEMA,
                             "rule": engine.MEDIA_REVIEW_RULE,
                             "file": "data/issue-85-engine-outcome-reactive/"
                                     "channel-verification-media-review.json"},
            "expectations": "frozen per control; no outcome-conditioned replacement",
        },
        "estimand": (
            "paired first-shot ENGINE-TRUTH outcome (pig_removed) differences "
            "and normalized ranking regret differences of the executed shot's "
            f"realized end-of-window count cost (observed offset {END_OFFSET}, "
            "terminal-absorbed; one frame = 50 native steps; right-censored, "
            "NOT a settled cost) against the state's frozen candidate outcome "
            "table recorded by the issue-77 N1 campaign; executed costs may "
            "fall outside [0,1] under execution variance and are recorded "
            "unclipped"),
        "pilot_gate": {
            "cells": PILOT_STATES_PER_FAMILY * 2 * len(SYSTEMS) * len(SEEDS),
            "states": PILOT_STATES_PER_FAMILY * 2,
            "state_rule": ("the two lowest-identity selected states of each "
                           "family with distinct source members (four distinct "
                           "physical initial states); identical to the #80 "
                           "pilot shape"),
            "prevalence_floor": engine.PREVALENCE_FLOOR,
            "minimum_valid_executions": engine.PILOT_MINIMUM_VALID_EXECUTIONS,
            "metric": ("pooled ENGINE-TRUTH first-shot success prevalence over "
                       "pilot cells with a valid executed segment; typed "
                       "failures excluded from numerator and denominator and "
                       "reported"),
            "if_cleared": "the full 288-cell matrix runs",
            "if_failed": ("publish readiness_or_precision_insufficient BEFORE "
                          "the full run - terminal, complete"),
        },
        "caps": {"wall_cap_seconds": WALL_CAP_SECONDS, "gpu_cap_seconds": GPU_CAP_SECONDS,
                 "wall_scope": "cumulative elapsed wall of the pilot+run execution phases",
                 "gpu_scope": ("cumulative synchronized decision wall (perception "
                               "plus ranking) on cuda across all executions")},
        "compute_accounting": (
            "per state wall time AND active-parameter/step counts AND candidate "
            "counts for every arm; published per-arm frozen predictor parameter "
            "counts, transition calls and linear MACs; inference is NOT "
            "equalized across arms and is recorded as such"),
        "phase_c_cross_check": (
            "on the pilot executions, the #82 replay-cost proxy predicate "
            "(realized end-of-window count cost <= the published candidate "
            f"threshold {engine.PROXY_THRESHOLD!r}) is evaluated against engine "
            "truth and the agreement/divergence table is published "
            "(DESCRIPTIVE; the threshold is never re-fit)"),
        "uncertainty": {"draws": engine.BOOTSTRAP_DRAWS, "seed": engine.BOOTSTRAP_SEED,
                        "unit": "state",
                        "note": "seed differences averaged before resampling; "
                                "descriptive only; this diagnostic cannot "
                                "support a confirmatory claim"},
        "disposition_rule": {
            "supported": ("Q1: the pre-declared primary contrast "
                          f"({SUCCESS_PRIMARY_CONTRAST[0]} hybrid-fixed-h1 vs "
                          f"{SUCCESS_PRIMARY_CONTRAST[1]}) has pooled mean > 0 "
                          "AND descriptive 95% interval entirely above 0 (the "
                          "practical margin is the interval-exclusion rule "
                          "itself); Q2: the pilot proxy-vs-engine agreement "
                          "table is complete over >= 24 valid pilot cells "
                          "(a measurement question; its value may be any)"),
            "not_supported_by_this_experiment": (
                "Q1: anything else after complete execution (negative or null "
                "results are valid outcomes and MUST be reported); Q2: never "
                "(a measurement, not a hypothesis)"),
            "readiness_or_precision_insufficient": (
                "channel verification failed, pilot gate failed, a cap stop, "
                "or incomplete execution inventory"),
        },
        "claim_boundary": (
            "bounded non-final diagnostic on development lineages; no "
            "multi-shot, adaptation, zero-shot, or complete-gameplay claim; "
            "cannot reopen #64/#65/#72/#75/#15; the #80/#82 published typed "
            "outcomes are never re-run, amended, or reinterpreted; this is a "
            "NEW experiment with a NEW predicate; descriptive intervals only"),
        "wall_context": {
            "issue_74_fixed_mode_seconds_per_state": {"continuous_h1": 1.79,
                                                      "continuous_h5": 0.37,
                                                      "continuous_h15": 0.13},
            "note": ("no adaptive system is present in this ticket; the #74 "
                     "hybrid_adaptive wall context carries no penalty "
                     "narrative")},
    }


PRIOR_FALLBACK_RULE_SHORT = (
    "ordinal 8 when admissible, else the admissible ordinal nearest 8 with "
    "ties to the lower ordinal")


def oracle_binding(retained=None):
    """Bind the #82 published proxy threshold to its exact published bytes.

    The #82 binding stayed 'no_defensible_binding' (the proxy failed its own
    defensibility gate); the Phase-C cross-check reuses the published CANDIDATE
    threshold exactly as recorded, never re-fit and never adopted as a
    defensible success predicate. Recomputes reuse the retained binding
    (no new hashing pass); freeze time hashes the published summary once.
    """
    path = EIGHTY2 / "summary.json"
    if retained is not None:
        if retained["candidate_threshold"] != engine.PROXY_THRESHOLD:
            raise ValueError("retained issue-82 threshold differs from the frozen proxy")
        if retained["identity"] != ORACLE_IDENTITY:
            raise ValueError("retained issue-82 binding identity differs")
        return retained
    summary = read(path)
    binding = summary.get("binding") or {}
    stage1 = binding.get("stage1") or {}
    threshold = stage1.get("candidate_threshold")
    if threshold != engine.PROXY_THRESHOLD:
        raise ValueError(f"issue-82 published candidate threshold {threshold!r} "
                         f"differs from the frozen proxy {engine.PROXY_THRESHOLD!r}")
    return {"identity": ORACLE_IDENTITY, "summary_sha256": file_identity(path),
            "candidate_threshold": threshold,
            "pig_removed_cells": stage1.get("pig_removed_cost_max_cells")
            if stage1.get("pig_removed_cost_max_cells") is not None
            else binding.get("pig_removed_cells"),
            "binding_status": binding.get("status"),
            "role": ("descriptive Phase-C cross-check threshold; the #82 "
                     "binding itself stays no_defensible_binding and is "
                     "never reopened")}


def make_plan(retained=None):
    """Build the plan. Freeze time (retained=None) hashes every bound input
    once. Post-freeze recompute (check_plan_bound/validate) passes the
    retained plan and reuses its archival identities - no new hashing pass;
    every scientific field is still recomputed and compared."""
    eighty_plan = membership_plan()
    states = states_from_membership(eighty_plan)
    pilot_states = recompute_pilot_states(states)
    if pilot_states != eighty_plan["pilot_states"]:
        raise ValueError("recomputed pilot states differ from the frozen #80 pilot states")
    player = Path(eighty_plan["player_root"])
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")
    verification = engine.verification_set(states)
    if retained is not None:
        eighty_sha = retained["eighty_binding"]["sha256"]
        if retained["eighty_binding"]["identity"] != EIGHTY_IDENTITY:
            raise ValueError("retained issue-80 binding identity differs")
    else:
        eighty_sha = file_identity(EIGHTY / "plan.json")
    plan = {
        "schema": engine.SCHEMA_PLAN, "identity": IDENTITY,
        "frozen_before_outcome": True,
        "definitions": definitions(),
        "engine_channel": plan_channel_bindings(
            None if retained is None else retained["engine_channel"]),
        "dynamics_checkpoints": dynamics_checkpoints(
            None if retained is None else retained["dynamics_checkpoints"]),
        "eighty_binding": {"identity": EIGHTY_IDENTITY, "sha256": eighty_sha,
                           "pilot_states": pilot_states,
                           "candidate_outcomes": str(EIGHTY / "candidate-outcomes")},
        "oracle_binding": oracle_binding(
            None if retained is None else retained["oracle_binding"]),
        "player_root": str(player),
        "states": states,
        "pilot_states": pilot_states,
        "verification_set": verification,
        "execution": {"workers": WORKERS, "device": DECISION_DEVICE,
                      "rendered": True, "nographics": False,
                      "isolated_process": True, "separate_ports_and_workdirs": True,
                      "worker_rss_mib": WORKER_RSS_MIB,
                      "aggregate_rss_mib": AGGREGATE_RSS_MIB,
                      "attempt_seconds": ATTEMPT_SECONDS,
                      "minimum_free_bytes": MINIMUM_FREE_BYTES,
                      "artifact_bytes": ARTIFACT_BYTES,
                      "decision_fixed_step": DECISION_FIXED_STEP,
                      "shot_seconds": SHOT_SECONDS,
                      "history_ready_seconds": HISTORY_READY_SECONDS,
                      "inference_hold_seconds": INFERENCE_HOLD_SECONDS,
                      "native_stride": NATIVE_STRIDE,
                      "shot_window_native": SHOT_WINDOW_NATIVE,
                      "end_offset": END_OFFSET,
                      "gpu_lock": GPU_LOCK_PATH,
                      "technical_retries": 0},
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_text": {path: (ROOT / path).read_text() for path in FILES},
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }
    expected_cells = len(states) * len(SYSTEMS) * len(SEEDS)
    if expected_cells != 288:
        raise ValueError(f"frozen matrix is {expected_cells} cells, expected 288")
    if len(pilot_states) != 2 * PILOT_STATES_PER_FAMILY:
        raise ValueError("pilot membership must be 4 states")
    if len(verification) != 4:
        raise ValueError("the frozen channel verification set is 4 controls")
    return plan


def plan_channel_bindings(retained=None):
    """The player-side channel binding: the frozen player must expose the
    engine hooks the detection rule reads (ABPig.Die -> RecordPigRemovedCallback).

    Hook presence is re-verified against the live source text on every
    recompute (cheap string checks, no hashing); the sha256 identities are
    archival and are reused from the retained plan on recompute.
    """
    runtime = ROOT / "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicalSnapshotRuntime.cs"
    pig = ROOT / "tasks/task_template_designer/Assets/Scripts/GameWorld/Characters/ABPig.cs"
    runtime_text = runtime.read_text()
    pig_text = pig.read_text()
    for needle in ("RecordPigRemovedCallback", '"pig_removed"', "entity_destroyed"):
        if needle not in runtime_text:
            raise ValueError(f"engine channel hook missing from {runtime.name}: {needle}")
    if "RecordPigRemovedCallback" not in pig_text:
        raise ValueError(f"engine pig hook missing from {pig.name}")
    if retained is not None:
        if (retained["runtime_source"] != str(runtime.relative_to(ROOT))
                or retained["pig_source"] != str(pig.relative_to(ROOT))):
            raise ValueError("retained engine channel source paths differ")
        return retained
    return {"runtime_source": str(runtime.relative_to(ROOT)),
            "pig_source": str(pig.relative_to(ROOT)),
            "runtime_sha256": file_identity(runtime),
            "pig_sha256": file_identity(pig),
            "note": ("the executed player binary is the frozen issue-77 N1 "
                     "campaign player; the Phase-A control executions verify "
                     "the live channel end to end regardless of source text")}


def load_plan():
    plan = read(OUTPUT / "plan.json")
    if plan.get("identity") != IDENTITY:
        raise ValueError("retained issue-85 plan identity differs")
    return plan


def check_plan_bound(plan):
    """Validate the frozen plan against the live frozen inputs it binds.

    Recompute reuses the plan's retained archival identities (no new hashing
    pass) and re-derives every scientific field. After execution has started,
    a difference confined to "source_text" is the documented
    post-execution reporting/protocol-fix provision (parent-directed): the
    executed source snapshot is retained inside the plan itself, and live
    reporting/validation code may be corrected without touching any frozen
    scientific definition. Any other difference is a hard freeze violation.
    """
    current = make_plan(retained=plan)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        differing = sorted(key for key in set(plan) | set(current)
                           if plan.get(key) != current.get(key))
        non_source = [key for key in differing if key != "source_text"]
        if non_source:
            raise ValueError(f"frozen issue-85 plan fields changed: {non_source}; "
                             "preserve and version")
        execution_started = any((OUTPUT / "records").glob("*.json")) \
            if (OUTPUT / "records").is_dir() else False
        if not execution_started:
            raise ValueError("frozen issue-85 plan source changed before "
                             "execution; re-freeze with --prepare to version it")
        log("post-execution reporting/protocol fix detected in source_text; "
            "frozen scientific content unchanged; the executed source snapshot "
            "remains retained inside the frozen plan")
    return plan


# ------------------------------------------------------- frozen outcome tables

def outcome_path(member_identity):
    return EIGHTY / "candidate-outcomes" / f"{member_identity}.json"


def check_member_outcomes(record, state):
    """Verify one stored #80 candidate outcome table binds to the inventory."""
    if (record.get("schema") != "issue_80_candidate_outcomes_v1"
            or record.get("member_identity") != state["source_member"]
            or len(record["candidates"]) != len(state["inventory"])):
        raise ValueError("candidate outcome table binding differs")
    for stored, item in zip(record["candidates"], state["inventory"], strict=True):
        if (stored["branch_identity"] != item["branch_identity"]
                or stored["ordinal"] != item["ordinal"] or stored["action"] != item["action"]
                or not math.isfinite(stored["realized_count_cost"])):
            raise ValueError("candidate outcome row differs from frozen inventory")
    costs = [candidate["realized_count_cost"] for candidate in record["candidates"]]
    if record["lowest_cost"] != min(costs) or record["highest_cost"] != max(costs):
        raise ValueError("candidate outcome range differs")


def require_prepared(plan):
    missing = [state["source_member"] for state in plan["states"]
               if not outcome_path(state["source_member"]).is_file()]
    if missing:
        raise ValueError(f"missing frozen #80 candidate outcome tables: {sorted(set(missing))[:3]}")
    for state in plan["states"]:
        check_member_outcomes(read(outcome_path(state["source_member"])), state)


# ---------------------------------------------------------------- channel scan

def channel_scan(native_root):
    """The frozen engine-side outcome channel over one executed segment."""
    trace = NativeSegmentTrace(native_root)
    events, samples = [], []
    launches = 0
    pig_contact = False
    for chunk in trace.trace.chunks():
        for event in chunk["events"]:
            name = event["event_type"]
            participants = " ".join(event.get("participants", []))
            if name == "bird_launched":
                launches += 1
            if name == "collision" and "pig" in participants:
                pig_contact = True
            events.append(event)
        samples.extend(chunk["fixed_step_samples"])
    verdict = engine.pig_channel(events)
    verdict["pig_lifecycle_destroyed"] = engine.pig_lifecycle_destroyed(samples)
    verdict["pig_contact"] = pig_contact
    verdict["bird_launches"] = launches
    verdict["consistency"] = engine.channel_consistency(
        verdict["pig_removed"], bool(verdict["pig_lifecycle_destroyed"]))
    return verdict


# --------------------------------------------------------------- decision phase

def action_tensor(action, device):
    return torch.tensor([[action["drag_x"] / 480., action["drag_y"] / 480.,
                          action["release_time_ms"] / 1000., action["tap_time_ms"] / 1000., 1.]],
                        device=device)


class ReactiveSelector:
    """System-side reactive ranking over one frozen candidate inventory."""

    def __init__(self, system, seed, model, objective, device):
        self.system = system
        self.seed = seed
        self.model = model
        self.objective = objective
        self.device = device
        self.arm = None if system == PRIOR_SYSTEM else SYSTEM_ARMS[system][0]
        self.pair = None if system == PRIOR_SYSTEM else SYSTEM_ARMS[system][1]

    def choose(self, carrier, inventory):
        if self.system == PRIOR_SYSTEM:
            ordinals = [item["ordinal"] for item in inventory]
            ordinal = engine.prior_fallback_choice(ordinals)
            chosen = next(item for item in inventory if item["ordinal"] == ordinal)
            return {"chosen": chosen, "failure": None,
                    "prior_fallback_used": PRIOR_ORDINAL not in ordinals,
                    "ranking": [{"ordinal": item["ordinal"], "predicted_cost": None,
                                 "excluded": "no-model prior"} for item in inventory],
                    "transition_calls": 0, "linear_macs": 0,
                    "candidate_count": len(inventory)}
        grid.synchronize(self.device)
        began = time.monotonic()
        macs = pair_work(self.model, self.pair)
        z = carrier[None].to(self.device)
        rows = []
        for item in inventory:
            value = z
            failed = False
            for _ in range(self.pair.delta):
                value = self.model.carrier(value, action_tensor(item["action"], self.device), self.pair)
                if not bool(torch.isfinite(value).all()):
                    failed = True
                    break
            rows.append({"ordinal": item["ordinal"],
                         "predicted_cost": None if failed else float(self.objective(value[0])),
                         "excluded": failed})
        grid.synchronize(self.device)
        wall = time.monotonic() - began
        finite = [row for row in rows if row["predicted_cost"] is not None]
        if not finite:
            return {"chosen": None, "failure": "all_candidate_predictions_failed",
                    "prior_fallback_used": False,
                    "ranking": rows, "transition_calls": len(inventory) * self.pair.delta,
                    "linear_macs": macs * len(inventory) * self.pair.delta,
                    "candidate_count": len(inventory), "wall_seconds": wall}
        best = min(finite, key=lambda row: (row["predicted_cost"], row["ordinal"]))
        chosen = next(item for item in inventory if item["ordinal"] == best["ordinal"])
        return {"chosen": chosen, "failure": None, "prior_fallback_used": False,
                "ranking": rows,
                "transition_calls": len(inventory) * self.pair.delta,
                "linear_macs": macs * len(inventory) * self.pair.delta,
                "candidate_count": len(inventory), "wall_seconds": wall,
                "selected_predicted_cost": best["predicted_cost"]}


def load_frozen_predictor(plan, seed, arm, device):
    binding_entry = plan["dynamics_checkpoints"][str(seed)][arm]
    path = Path(binding_entry["path"])
    if not path.is_file():
        raise ValueError(f"missing frozen predictor {path}")
    # The sha256 identity in the binding is archival freeze metadata (retained
    # in the frozen plan); no runtime hashing pass per cell. The binding is
    # re-verified from the predictor payload itself.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_binding = read(DYNAMICS / "plan.json")
    stored = payload.get("binding")
    if (not isinstance(stored, dict) or stored.get("plan_identity") != expected_binding["identity"]
            or stored.get("seed") != seed or stored.get("arm") != arm
            or stored.get("component") != "predictor"):
        raise ValueError(f"predictor binding differs for seed={seed} arm={arm}")
    model = (ContinuousDynamics(expected_binding["capacity"]["continuous_width"])
             if arm == "continuous" else CNNHybridPredictor())
    model.load_state_dict(payload["model"], strict=True)
    steps = expected_binding["training"]["steps"]
    expected_counts = {str(p.identity): steps // len(pairs_for(model)) for p in pairs_for(model)}
    if (payload["step"] != steps or payload["pair_counts"] != expected_counts):
        raise ValueError(f"incomplete predictor budget for seed={seed} arm={arm}")
    return model.to(torch.device(device)).eval()


def dynamics_checkpoints(retained=None):
    """The frozen issue-77 N1 predictor bindings.

    Freeze time (retained=None) each predictor file is hashed once and the
    identity is recorded. Any later recompute (check_plan_bound/validate)
    reuses the identities retained in the frozen plan - no new hashing pass;
    the retained bindings are the archival truth.
    """
    dynamics = read(DYNAMICS / "plan.json")
    if dynamics["identity"] != "issue-77-n1-dynamics-v1":
        raise ValueError("requires the frozen issue-77 N1 dynamics checkpoints")
    bindings = {}
    for seed in SEEDS:
        entry = {}
        for arm in ("continuous", "hybrid"):
            path = DYNAMICS / f"seed-{seed}" / arm / "predictor.pt"
            if not path.is_file():
                raise ValueError(f"missing frozen predictor {path}")
            if retained is not None:
                stored = retained[str(seed)][arm]
                if stored["path"] != str(path):
                    raise ValueError(f"retained predictor path differs for seed={seed} arm={arm}")
                entry[arm] = {"path": stored["path"], "identity": stored["identity"]}
            else:
                entry[arm] = {"path": str(path), "identity": file_identity(path)}
        bindings[str(seed)] = entry
    return bindings


# ------------------------------------------------------------- cell execution

def _plain_record(plan, cell, state):
    return {"schema": SCHEMA, "plan_identity": plan["identity"], "cell": cell,
            "state_identity": state["identity"], "system": cell["system"],
            "seed": cell["seed"], "failure": None, "failure_kind": None,
            "decision": None, "execution": None, "outcome": None,
            "engine_channel": None, "fresh_scenario_lineage": False,
            "issue_64_authorized": False}


def parse_offset_carrier(adapter, observation_root, needed_offsets=(END_OFFSET,)):
    """#77 target semantics: offset 600 carrier with terminal absorption."""
    with torch.no_grad():
        manifest = read(Path(observation_root) / MANIFEST_NAME)
        frames = manifest["frame_records"]
        if (frames[0]["fixed_step"] != DECISION_FIXED_STEP
                or any(f["fixed_step"] != DECISION_FIXED_STEP + NATIVE_STRIDE * i
                       for i, f in enumerate(frames[:-1]))
                or not (DECISION_FIXED_STEP + NATIVE_STRIDE * (len(frames) - 2)
                        < frames[-1]["fixed_step"]
                        <= DECISION_FIXED_STEP + NATIVE_STRIDE * (len(frames) - 1))):
            raise ValueError("candidate observation cadence differs")
        last_index = len(frames) - 1

        def observation(index):
            ref = frames[index]["agent_observation"]
            return AgentObservation(ref["identity"], frames[index]["fixed_step"],
                                    frames[index]["fixed_time_seconds"],
                                    (Path(observation_root) / ref["relative_path"]).read_bytes(),
                                    "agent")

        carriers, parsed, calls = {}, {}, 0
        for offset in needed_offsets:
            position = min(offset, last_index)
            chosen = sorted({max(0, position - 1), position})
            missing = [i for i in chosen if i not in parsed]
            if missing:
                values = adapter.parse_batch(tuple(observation(i) for i in chosen))
                parsed.update(zip(chosen, values, strict=True))
                calls += len(chosen)
            context = TemporalObservationContext(
                None if position == 0 else observation(max(0, position - 1)),
                observation(position))
            carriers[str(offset)] = adapter.build_from_parsed(
                context, parsed[position],
                None if position == 0 else parsed[max(0, position - 1)]).tensor
    return carriers, calls


def execute_cell(payload):
    """One isolated, real-rendered execution: a diagnostic cell (reactive
    decision + engine-truth outcome) or a Phase-A channel control (frozen
    action, channel verdict only)."""
    torch.set_num_threads(2)
    output = Path(payload["output"])
    plan = read(output / "plan.json")
    cell = payload["cell"]
    identity = cell["identity"]
    state = payload["state"]
    verification = payload.get("verification")
    record_path = output / "records" / f"{identity}.json"
    attempt = output / "attempts" / identity
    attempt.mkdir(parents=True)
    device = payload["device"]
    started = time.monotonic()
    record = _plain_record(plan, cell, state)
    if verification is not None:
        record["verification"] = {key: verification[key] for key in
                                  ("role", "state", "ordinal", "branch_identity",
                                   "expectation")}
    bridge = physics = engine_process = display_process = None
    variables = ("DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
                 "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
                 "NOVPHY_ENVIRONMENT_SEED", "NOVPHY_NATIVE_DECISION_STEP")
    environment = {key: os.environ.get(key) for key in variables}
    try:
        member = state["execution_member"]
        inventory = state["inventory"]
        adapter = model = objective = None
        if verification is None:
            # Frozen setup (adapter, predictor, objective) happens before the
            # engine starts so the measured decision wall is perception plus
            # ranking only. The prior arm parses the sealed frame with the
            # same frozen adapter for identical perception accounting.
            adapter = old.repair.load_repaired_adapter(old.repair.ROOT, device)
            objective = TaskObjective.from_vocabulary(
                read(DYNAMICS / "plan.json")["contract"]["vocabulary"])
            if cell["system"] != PRIOR_SYSTEM:
                model = load_frozen_predictor(plan, cell["seed"],
                                              SYSTEM_ARMS[cell["system"]][0], device)
        game = attempt / "runtime"
        clone_player(Path(plan["player_root"]), game)
        live_files.install_level(game, member)
        _, scenario = live_files.materialize(member, member["template"], attempt / "authority")
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("reactive member differs from its frozen scenario authority")
        display, display_process = start_display(attempt / "display.log")
        ports = set()
        while len(ports) < 3:
            ports.add(free_port())
        agent_port, game_port, physics_port = sorted(ports)
        aligned = attempt / "aligned"
        os.environ.update(
            DISPLAY=display, XDG_DATA_HOME=str(attempt / "xdg"),
            NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port),
            NOVPHY_PHYSICS_CAPTURE_V2_STRIDE=str(NATIVE_STRIDE),
            NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
            NOVPHY_ENVIRONMENT_SEED=str(state["engine_seed"]),
            NOVPHY_NATIVE_DECISION_STEP=str(DECISION_FIXED_STEP))
        record["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port}
        engine_process = start_engine(game, False, agent_port=agent_port,
                                      game_port=game_port, physics_port=physics_port)
        write(attempt / "runtime.json", record)
        bridge = connect_with_retry("127.0.0.1", agent_port, timeout=180, deadline_seconds=60)
        physics = ScienceBirdsBridge("127.0.0.1", physics_port, timeout=30)
        bridge.configure(760001, PlayingMode.TRAINING)
        bridge.set_speed(1)
        prepare_for_play(bridge, timeout=60, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("episode did not load its single assigned level")
        deadline = time.monotonic() + HISTORY_READY_SECONDS
        while not list(aligned.glob("decision-history-*/ready.json")):
            if time.monotonic() >= deadline:
                raise TimeoutError("native decision barrier readiness deadline; no retry")
            time.sleep(.1)
        _, rows = read_history(aligned, DECISION_FIXED_STEP)

        class Policy:
            def __init__(self):
                self.frames = 0

            def observe(self, png, timestamp):
                self.frames += 1

            def executed(self, action, timestamp):
                self.frames += 0

            def choose(self):
                raise RuntimeError("the choice is made upstream of the policy")

            def evidence(self):
                return {"observed_frames": self.frames,
                        "future_frames_used": False, "engine_outcomes_used": False}

        policy = Policy()
        shot_identity = identity + ":decision"
        expose_history(rows, attempt / "decision-1", scenario, shot_identity,
                       member["exposure_role"], policy)
        prepared = prepare_action(bridge, inventory[0]["action"])
        before = physics.get_observation_capture()
        time.sleep(INFERENCE_HOLD_SECONDS)
        after = physics.get_observation_capture()
        for name, frame in (("before", before), ("after", after)):
            write(attempt / f"paused-{name}.json", _plain_json(frame.metadata))
            (attempt / f"paused-{name}.png").write_bytes(frame.canonical_png)
        if (before.metadata["fixed_step"] != DECISION_FIXED_STEP
                or after.metadata["fixed_step"] != DECISION_FIXED_STEP
                or before.metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]
                or after.metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]
                or before.canonical_png != rows[-1]["canonical_png"]
                or after.canonical_png != rows[-1]["canonical_png"]):
            raise ValueError("native decision state/RGB advanced during readiness or hold")

        manifest = read(attempt / "decision-1" / MANIFEST_NAME)
        frames = manifest["frame_records"]
        if frames[-1]["fixed_step"] != DECISION_FIXED_STEP:
            raise ValueError("decision trace lacks the sealed decision frame")
        ref = frames[-1]["agent_observation"]
        decision_frame_bytes = (attempt / "decision-1" / ref["relative_path"]).read_bytes()
        (attempt / "decision-frame.png").write_bytes(decision_frame_bytes)

        if verification is not None:
            chosen = next(item for item in inventory
                          if item["ordinal"] == verification["ordinal"])
            decision = {"kind": "verification_action", "chosen": chosen,
                        "ordinal": chosen["ordinal"],
                        "branch_identity": chosen["branch_identity"],
                        "failure": None, "prior_fallback_used": False,
                        "ranking": [], "transition_calls": 0, "linear_macs": 0,
                        "candidate_count": len(inventory), "wall_seconds": 0.0,
                        "perception_seconds": 0.0, "gpu_seconds": 0.0,
                        "actuator_readiness": prepared.evidence,
                        "inference_not_equalized": False}
        else:
            perception_started = time.monotonic()
            observation = AgentObservation(ref["identity"], frames[-1]["fixed_step"],
                                           frames[-1]["fixed_time_seconds"],
                                           decision_frame_bytes, "agent")
            with torch.no_grad():
                parsed = adapter.parse_batch((observation,))[0]
                carrier = adapter.build_from_parsed(
                    TemporalObservationContext(None, observation), parsed, None).tensor
                grid.synchronize(device)
                perception_seconds = time.monotonic() - perception_started
                selector = ReactiveSelector(cell["system"], cell["seed"], model, objective, device)
                decision = selector.choose(carrier, inventory)
            decision["perception_seconds"] = perception_seconds
        decision["actuator_readiness"] = prepared.evidence
        decision["inference_not_equalized"] = verification is None
        decision["gpu_seconds"] = (decision.get("perception_seconds", 0.0)
                                   + decision.get("wall_seconds", 0.0)
                                   if device.startswith("cuda") else 0.0)
        record["decision"] = decision
        if decision["failure"] is not None:
            # Typed terminal decision failure: retained, no shot executed.
            record["failure"] = f"decision_failure: {decision['failure']}"
            record["failure_kind"] = "decision_failure"
        else:
            chosen = decision["chosen"]
            record["chosen"] = {"ordinal": chosen["ordinal"],
                                "branch_identity": chosen["branch_identity"],
                                "action": deepcopy(chosen["action"])}
            segment_started = time.monotonic()
            segment = capture_segment(bridge, aligned, attempt / "shot-1", member, scenario,
                                      identity + ":shot-1", chosen["action"], SHOT_SECONDS)
            engine_seconds = time.monotonic() - segment_started
            initial_metadata = live_files.read(Path(segment["native_root"]) / "frame_000001.json")
            initial_png = (Path(segment["native_root"]) / "frame_000001.png").read_bytes()
            if (segment["summary"]["first_fixed_step"] != DECISION_FIXED_STEP
                    or initial_png != rows[-1]["canonical_png"]
                    or initial_metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]):
                raise ValueError("executed shot pre-intervention state differs from its decision frame")
            verdict = channel_scan(segment["native_root"])
            if verdict["bird_launches"] != 1:
                raise ValueError("executed shot did not contain exactly one native launch")
            summary = segment["summary"]
            terminal = terminal_evidence(segment)
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
            }
            if verification is None:
                carriers, calls = parse_offset_carrier(
                    adapter, attempt / "shot-1" / "observation-trace")
                realized_cost = float(objective(carriers[str(END_OFFSET)]))
                outcomes = payload["member_outcomes"]
                frozen_cost = next(c["realized_count_cost"] for c in outcomes["candidates"]
                                   if c["ordinal"] == chosen["ordinal"])
                record["execution"]["realized_end_of_window_count_cost"] = realized_cost
                record["execution"]["end_of_window_parser_calls"] = calls
                record["outcome"] = {
                    "first_shot_success": verdict["pig_removed"],
                    "shots_to_success": 1 if verdict["pig_removed"] else None,
                    "realized_count_cost": realized_cost,
                    "regret": engine.normalized_regret(realized_cost, outcomes),
                    "informative": outcomes["informative"],
                    "all_tied": outcomes["all_tied"],
                    "chosen_frozen_cost": frozen_cost,
                    "chosen_frozen_regret": engine.normalized_regret(frozen_cost, outcomes),
                }
            else:
                record["outcome"] = {
                    "first_shot_success": verdict["pig_removed"],
                    "shots_to_success": 1 if verdict["pig_removed"] else None,
                    "expectation_matched": verdict["pig_removed"] == verification["expectation"],
                }
    except Exception as error:  # typed terminal failure; never retried silently
        record["failure"] = f"{type(error).__name__}: {error}"
        if record["failure_kind"] is None:
            record["failure_kind"] = "execution_failure"
    finally:
        actions = [("connection.disconnect", connection.disconnect)
                   for connection in (bridge, physics) if connection is not None]
        actions.append(("stop_started_engine", lambda: stop_started_engine(engine_process)))
        if display_process is not None:
            actions.append(("display.terminate", lambda: terminate(display_process)))

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
                cleanup_failures.append(f"{name}: {type(cleanup_error).__name__}: {cleanup_error}")
        if cleanup_failures:
            record["cleanup_failures"] = cleanup_failures
    record["wall_seconds"] = time.monotonic() - started
    write(record_path, record)
    print(f"issue-85 cell {identity} complete={record['failure'] is None} "
          f"failure={record['failure']}", flush=True)
    return record


# ---------------------------------------------------------------- ledger + supervisor

def ledger_path():
    return OUTPUT / "ledger.json"


def fresh_ledger(plan):
    return {"schema": engine.SCHEMA_LEDGER, "identity": IDENTITY,
            "plan_identity": plan["identity"], "status": "running",
            "wall_seconds_elapsed": 0.0, "gpu_seconds_elapsed": 0.0,
            "cells": {}, "phase": None}


def load_ledger(plan, phase=None):
    path = ledger_path()
    if not path.is_file():
        return fresh_ledger(plan)
    ledger = read(path)
    if (ledger.get("schema") != engine.SCHEMA_LEDGER
            or ledger.get("plan_identity") != plan["identity"]):
        raise ValueError("retained issue-85 ledger identity differs from the frozen plan")
    if (phase == "run" and ledger.get("phase") == "pilot"
            and ledger.get("status") == "complete"):
        # Phase promotion: the pilot finished cleanly; the run continues the
        # same cap accounting without relaunching any terminal pilot cell.
        ledger["phase"] = "run"
        ledger["status"] = "running"
        return ledger
    if ledger["status"] == "interrupted":
        # A crashed/killed phase is resumable: terminal_cells() reconciles
        # partial persistence outcome-blind and re-queues unrecorded cells.
        ledger["status"] = "running"
        ledger["phase"] = phase
        return ledger
    if ledger["status"] != "running":
        raise ValueError(f"issue-85 execution ledger is terminal ({ledger['status']}); "
                         "audit it; never relaunch a terminal campaign")
    return ledger


def terminal_cells(plan):
    """Cells with a retained record + receipt are terminal, whatever the outcome.

    Partial persistence from a supervisor crash is completed without touching
    the recorded outcome; an attempt tree with neither file never produced an
    outcome and is removed so the cell re-dispatches - crash recovery, not
    outcome-conditioned replacement.
    """
    terminal, recovered = {}, []
    for system in SYSTEMS:
        for seed in SEEDS:
            for state in plan["states"]:
                identity = cell_identity(system, seed, state["identity"])
                record = OUTPUT / "records" / f"{identity}.json"
                receipt = OUTPUT / "receipts" / f"{identity}.json"
                attempt = OUTPUT / "attempts" / identity
                if record.is_file() and receipt.is_file():
                    terminal[identity] = True
                    continue
                if record.is_file() and not receipt.is_file():
                    write(receipt, {"identity": identity, "phase": None,
                                    "worker_exitcode": 0, "stop": None,
                                    "wall_seconds": None,
                                    "peak_cpu_rss_mib": None,
                                    "receipt_recovered_from_record": True})
                    terminal[identity] = True
                    continue
                if receipt.is_file() and not record.is_file():
                    cell = split_cell_identity(plan, identity)
                    write(record, _plain_record(plan, cell, state)
                          | {"failure": "worker_terminated_before_record",
                             "failure_kind": "execution_failure",
                             "wall_seconds": None})
                    terminal[identity] = True
                    continue
                if attempt.exists():
                    shutil.rmtree(attempt)
                    recovered.append(identity)
                terminal[identity] = False
    return terminal, recovered


def free_disk(output):
    return shutil.disk_usage(output).free


def artifact_bytes(output):
    total = 0
    for root, _dirs, names in os.walk(output):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                return None
    return total


def supervise(args, plan, phase):
    """Run the phase's scheduled cells with isolated bounded workers and caps."""
    from scripts.process_lifecycle import start_isolated_worker
    from scripts.run_issue_76_compatibility import process_rss, terminate_worker
    ledger = load_ledger(plan, phase)
    terminal, recovered = terminal_cells(plan)
    if recovered:
        log(f"crash recovery: {len(recovered)} dispatched-but-unrecorded cells "
            f"re-queued (no outcome was recorded): {sorted(recovered)[:4]}")
    if phase == "pilot":
        cells = [cell_identity(system, seed, state["identity"])
                 for system in SYSTEMS for seed in SEEDS
                 for state in plan["states"] if state["identity"] in plan["pilot_states"]]
    else:
        cells = [cell_identity(system, seed, state["identity"])
                 for system in SYSTEMS for seed in SEEDS for state in plan["states"]]
    pending = [identity for identity in cells if not terminal[identity]]
    ledger["phase"] = phase
    write(ledger_path(), ledger)
    if not pending:
        ledger["status"] = "complete"
        write(ledger_path(), ledger)
        log(f"{phase}: all {len(cells)} scheduled cells already terminal")
        return ledger
    adapter_probe = old.repair.load_repaired_adapter(old.repair.ROOT, args.device)
    del adapter_probe  # fail fast on a broken frozen parser before any spawn
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    live = []
    stop_reason = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise RuntimeError("supervisor received SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        index = 0
        while index < len(pending) or live:
            elapsed = ledger["wall_seconds_elapsed"] + time.monotonic() - started
            if stop_reason is None and elapsed >= WALL_CAP_SECONDS:
                stop_reason = "wall_cap_exceeded"
            if stop_reason is None and ledger["gpu_seconds_elapsed"] >= GPU_CAP_SECONDS:
                stop_reason = "gpu_cap_exceeded"
            if stop_reason is None and free_disk(OUTPUT) < MINIMUM_FREE_BYTES:
                stop_reason = "minimum_free_storage"
            if stop_reason is None:
                size = artifact_bytes(OUTPUT)
                if size is not None and size > ARTIFACT_BYTES:
                    stop_reason = "artifact_limit"
            if stop_reason is not None:
                log(f"cap stop: {stop_reason}; remaining cells are retained as "
                    "typed not_executed failures (no replacement)")
                break
            while index < len(pending) and len(live) < WORKERS:
                identity = pending[index]
                cell = split_cell_identity(plan, identity)
                state = next(s for s in plan["states"] if s["identity"] == cell["state"])
                payload = {
                    "output": str(OUTPUT), "device": args.device,
                    "cell": cell, "state": state,
                    "member_outcomes": read(outcome_path(state["source_member"])),
                }
                (OUTPUT / "markers").mkdir(parents=True, exist_ok=True)
                write(OUTPUT / "markers" / f"{identity}.json",
                      {"identity": identity, "phase": phase,
                       "dispatched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                process = start_isolated_worker(context, execute_cell, (payload,))
                live.append({"identity": identity, "process": process,
                             "started": time.monotonic(), "peak_rss": 0.0, "stop": None})
                index += 1
            if not live:
                break
            time.sleep(1)
            now = time.monotonic()
            aggregate = 0.0
            for worker in live:
                if worker["process"].is_alive():
                    rss = process_rss(worker["process"].pid)
                    worker["peak_rss"] = max(worker["peak_rss"], rss)
                    aggregate += rss
                    if now - worker["started"] >= ATTEMPT_SECONDS:
                        worker["stop"] = "attempt_wall_limit"
                    elif worker["peak_rss"] > WORKER_RSS_MIB:
                        worker["stop"] = "worker_memory_limit"
            if stop_reason is None and aggregate > AGGREGATE_RSS_MIB:
                newest = max((w for w in live if w["process"].is_alive() and w["stop"] is None),
                             key=lambda w: w["started"], default=None)
                if newest is not None:
                    newest["stop"] = "aggregate_memory_limit"
            for worker in [w for w in live if w["stop"] is not None or not w["process"].is_alive()]:
                wall = time.monotonic() - worker["started"]
                exitcode = worker["process"].exitcode
                terminate_worker(worker["process"])
                live.remove(worker)
                record_file = OUTPUT / "records" / f"{worker['identity']}.json"
                gpu_seconds = 0.0
                if not record_file.is_file():
                    # Typed terminal failure: the worker finished (stop or nonzero
                    # exit) without persisting a record, so no outcome exists.
                    failure = worker["stop"] or f"worker_exitcode={exitcode}"
                    cell = split_cell_identity(plan, worker["identity"])
                    state = next(s for s in plan["states"] if s["identity"] == cell["state"])
                    write(record_file, _plain_record(plan, cell, state)
                          | {"failure": failure, "failure_kind": "execution_failure"})
                else:
                    stored = read(record_file)
                    gpu_seconds = float((stored.get("decision") or {}).get("gpu_seconds", 0.0))
                ledger["cells"][worker["identity"]] = {
                    "status": "complete" if (worker["stop"] is None and exitcode == 0) else "failed",
                    "stop": worker["stop"], "exit_code": exitcode,
                    "wall_seconds": wall, "gpu_seconds": gpu_seconds,
                    "failure": read(record_file).get("failure"),
                }
                ledger["gpu_seconds_elapsed"] += gpu_seconds
                write(OUTPUT / "receipts" / f"{worker['identity']}.json", {
                    "identity": worker["identity"], "phase": phase,
                    "worker_exitcode": exitcode, "stop": worker["stop"],
                    "wall_seconds": wall, "peak_cpu_rss_mib": worker["peak_rss"]})
                ledger["wall_seconds_elapsed"] = (
                    ledger["wall_seconds_elapsed"] + time.monotonic() - started)
                started = time.monotonic()
                write(ledger_path(), ledger)
                done = len(ledger["cells"])
                log(f"[{done}/{len(cells)}] {worker['identity']} "
                    f"stop={worker['stop']} wall={wall:.0f}s")
        elapsed = ledger["wall_seconds_elapsed"] + time.monotonic() - started
        ledger["wall_seconds_elapsed"] = elapsed
        if stop_reason is not None:
            # Cap/storage stop: every unlaunched cell is retained as a typed
            # terminal failure with the stop reason. No replacement, no retry.
            for identity in pending:
                record_file = OUTPUT / "records" / f"{identity}.json"
                if identity in ledger["cells"] or record_file.is_file():
                    continue
                cell = split_cell_identity(plan, identity)
                state = next(s for s in plan["states"] if s["identity"] == cell["state"])
                write(record_file, _plain_record(plan, cell, state)
                      | {"failure": f"not_executed: {stop_reason}",
                         "failure_kind": "not_executed", "wall_seconds": 0.0})
                write(OUTPUT / "receipts" / f"{identity}.json", {
                    "identity": identity, "phase": phase, "worker_exitcode": None,
                    "stop": stop_reason, "wall_seconds": 0.0, "peak_cpu_rss_mib": 0.0})
                ledger["cells"][identity] = {"status": "failed", "stop": stop_reason,
                                             "exit_code": None, "wall_seconds": 0.0,
                                             "gpu_seconds": 0.0,
                                             "failure": f"not_executed: {stop_reason}"}
        if stop_reason is None and len(ledger["cells"]) >= len(cells):
            ledger["status"] = "complete"
        else:
            ledger["status"] = stop_reason or "interrupted"
        write(ledger_path(), ledger)
    except BaseException:
        ledger["wall_seconds_elapsed"] = (
            ledger["wall_seconds_elapsed"] + time.monotonic() - started)
        ledger["status"] = "interrupted"
        write(ledger_path(), ledger)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    log(f"{phase} finished status={ledger['status']} "
        f"wall={ledger['wall_seconds_elapsed']:.0f}s gpu={ledger['gpu_seconds_elapsed']:.0f}s")
    return ledger


def split_cell_identity(plan, identity):
    system, seed_part, state = identity.split("--", 2)
    return {"identity": identity, "system": system,
            "seed": int(seed_part.removeprefix("seed")), "state": state}


# ------------------------------------------------------------ Phase A controls

def channel_wall_accounting(control_records):
    """Durable Phase-A wall accounting, derived from retained receipts
    (per-control record walls + archived failed-harness walls + record
    mtimes). The resume-mode timer is never used as an execution wall."""
    walls, starts, ends = 0.0, [], []
    for identity, record in control_records:
        wall = record.get("wall_seconds") or 0.0
        walls += wall
        record_file = OUTPUT / "records" / f"{identity}.json"
        if record_file.is_file():
            mtime = record_file.stat().st_mtime
            starts.append(mtime - wall)
            ends.append(mtime)

    def utc(ts):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    harness_files = sorted(OUTPUT.glob("failed-harness/*/verification--*.json"))
    harness = None
    if harness_files:
        harness = {
            "cells": len(harness_files),
            "wall_seconds": sum(json.loads(f.read_text()).get("wall_seconds") or 0.0
                                for f in harness_files),
            "archive_dir": harness_files[0].parent.relative_to(OUTPUT).as_posix(),
            "note": ("broken-harness run (KeyError before any shot); see "
                     "verification-harness-recovery.json"),
        }
    return {
        "wall_seconds": walls,
        "wall_seconds_scope": ("real executed Phase-A control wall: sum of the "
                               "four control records' wall_seconds from the "
                               "executing run"),
        "failed_harness": harness,
        "executed_at_utc": utc(min(starts)) if starts else None,
        "executed_at_receipt": ("derived from the retained control records: "
                                "earliest record mtime minus its wall_seconds; "
                                "run end = latest record mtime"),
        "controls_execution_run_end_utc": utc(max(ends)) if ends else None,
    }


def verification_records(plan):
    """Terminal per-control records keyed by control identity, with resume."""
    records = {}
    pending = []
    for entry in plan["verification_set"]:
        identity = f"verification--{entry['role']}--{entry['state']}"
        record_file = OUTPUT / "records" / f"{identity}.json"
        if record_file.is_file():
            records[entry["state"]] = read(record_file)
        else:
            pending.append((identity, entry))
    return records, pending


def smoke(args):
    """Phase A: the bounded real rendered channel verification set.

    Four controls (2 replay-anchored positives, 2 negatives), each executed
    once, real rendered, isolated ports/workdirs, retained frames/WebMs under
    data/. Published here, before any diagnostic execution. Resumable: already
    executed controls are never re-run; missing controls are executed.
    """
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    # Shared-GPU contract: the lock is acquired BEFORE the wall timer starts
    # and held until the timed verdict is written; lock wait is never counted.
    lock = gpu_lock()
    began = time.monotonic()
    try:
        records, pending = verification_records(plan)
        for identity, entry in pending:
            state = next(s for s in plan["states"] if s["identity"] == entry["state"])
            payload = {
                "output": str(OUTPUT), "device": "cpu",
                "cell": {"identity": identity, "system": "channel-verification",
                         "seed": entry["seed"], "state": entry["state"]},
                "state": state,
                "verification": entry,
            }
            log(f"verification {identity} role={entry['role']} "
                f"ordinal={entry['ordinal']} expectation={entry['expectation']}")
            record = execute_cell(payload)
            records[entry["state"]] = record
        entries = [(entry, records[entry["state"]])
                   for entry in plan["verification_set"] if entry["state"] in records]
        verdict = engine.verification_verdict(entries)
        verdict.update({"schema": engine.SCHEMA_CHANNEL, "identity": IDENTITY,
                        "plan_identity": plan["identity"],
                        "detection_rule": engine.CHANNEL_DETECTION_RULE,
                        "failure_modes": engine.CHANNEL_FAILURE_MODES,
                        "executed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "wall_seconds": time.monotonic() - began})
        media_review_file = DATA / "channel-verification-media-review.json"
        verdict = engine.apply_media_review(
            verdict, read(media_review_file) if media_review_file.is_file() else None)
        verdict.update(channel_wall_accounting(
            [(f"verification--{entry['role']}--{entry['state']}", records[entry["state"]])
             for entry in plan["verification_set"] if entry["state"] in records]))
        verdict["mode_wall_seconds"] = time.monotonic() - began
        verdict["mode_wall_seconds_note"] = ("wall of this --smoke-test invocation "
                                             "including any resume pass; never used "
                                             "as an execution wall")
        record_file = OUTPUT / "channel-verification.json"
        if record_file.is_file():
            superseded = record_file.with_name(
                "channel-verification.json.superseded-"
                + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
            record_file.replace(superseded)
            log(f"prior channel-verification record versioned as {superseded.name}")
        write(record_file, verdict)
        manifest = publish_verification_gallery(plan, verdict)
        for detail in verdict["cells"]:
            log(f"verification {detail['identity']} executed={detail['executed']} "
                f"pig_removed={detail['pig_removed']} matched={detail['expectation_matched']} "
                f"consistency={detail['consistency']}")
        verified = verdict["verified"]
        pending_media = bool(verdict.get("media_review_pending"))
        blocker = verdict["blocker"]
        mode_wall = time.monotonic() - began
        cells_n = len(verdict["cells"])
        gallery_n = len(manifest["cells"])
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    if not verified:
        if pending_media:
            log("channel game-state verdict holds but the media review is not "
                "recorded yet; review the retained gallery and write "
                "channel-verification-media-review.json, then re-run --smoke-test")
            return 0
        log(f"channel verification FAILED: {blocker}")
        return 3
    log(f"channel VERIFIED (game state AND retained media): {cells_n} controls, "
        f"{gallery_n} gallery entries, wall={mode_wall:.0f}s; "
        "the Phase-A record is published before any diagnostic execution")
    return 0


def publish_verification_gallery(plan, verdict):
    """Retain decision frame + WebM + final frame per control under data/."""
    _verify_webm_encoder()
    root = DATA / "channel-verification"
    manifest = {"schema": "issue_85_channel_gallery_v1", "identity": IDENTITY,
                "plan_identity": plan["identity"], "playback_fps": PILOT_AUDIT_FPS,
                "cells": []}
    for detail in verdict["cells"]:
        identity = detail["identity"]
        cell_root = root / identity
        cell_root.mkdir(parents=True, exist_ok=True)
        entry = {"identity": identity, **{key: detail[key] for key in
                                          ("role", "state", "ordinal", "expectation",
                                           "pig_removed", "expectation_matched",
                                           "consistency", "failure")}}
        attempt = OUTPUT / "attempts" / identity
        decision_source = attempt / "decision-frame.png"
        if decision_source.is_file():
            shutil.copy2(decision_source, cell_root / "decision.png")
            entry["decision_frame"] = str((cell_root / "decision.png").relative_to(DATA))
        observation_root = attempt / "shot-1" / "observation-trace"
        if observation_root.is_dir():
            trace_manifest = read(observation_root / MANIFEST_NAME)
            frames = [observation_root / frame["agent_observation"]["relative_path"]
                      for frame in trace_manifest["frame_records"]]
            video = cell_root / "shot.webm"
            if not video.exists():
                try:
                    _encode_agent_frames_webm(frames, video)
                except Exception as error:
                    entry["video_error"] = f"{type(error).__name__}: {error}"
            if video.exists():
                entry["video"] = str(video.relative_to(DATA))
                entry["video_frames"] = len(frames)
            final_png = cell_root / "final.png"
            if frames:
                final_png.write_bytes(frames[-1].read_bytes())
                entry["final_frame"] = str(final_png.relative_to(DATA))
        record = read(OUTPUT / "records" / f"{identity}.json")
        entry["failure"] = record.get("failure")
        entry["pig_removed_events"] = (record.get("engine_channel") or {}).get(
            "pig_removed_events")
        entry["pig_lifecycle_destroyed"] = (record.get("engine_channel") or {}).get(
            "pig_lifecycle_destroyed")
        manifest["cells"].append(entry)
        executed = record.get("failure") is None and record.get("engine_channel")
        if executed and not (entry.get("decision_frame") and entry.get("video")
                             and entry.get("final_frame")):
            raise ValueError(
                f"executed control {identity} cannot be published without its "
                f"decision frame, shot WebM and final frame: "
                f"{entry.get('video_error') or 'media missing'}")
    write(DATA / "channel-verification-manifest.json", manifest)
    sections = []
    for entry in manifest["cells"]:
        media = ""
        if entry.get("decision_frame"):
            media += f'<img src="{entry["decision_frame"]}" width="320"/> '
        if entry.get("final_frame"):
            media += f'<img src="{entry["final_frame"]}" width="320"/> '
        if entry.get("video"):
            media += f'<video src="{entry["video"]}" controls width="320"></video>'
        sections.append(
            f"<tr><td>{entry['identity']}</td><td>{entry['role']}</td>"
            f"<td>{entry['expectation']}</td><td>{entry['pig_removed']}</td>"
            f"<td>{entry['consistency']}</td><td>{entry['failure'] or ''}</td>"
            f"<td>{media}</td></tr>")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Issue 85 engine-channel verification</title></head><body>"
            "<h1>Issue 85 Phase-A engine-channel verification</h1>"
            "<p>Per control: decision frame, final canonical frame and shot WebM; "
            "positives must show the pig disappearing, negatives the pig "
            "surviving. " + engine.VERIFICATION_REVIEW_RULE + "</p>"
            "<table border='1'><tr><th>control</th><th>role</th><th>expected</th>"
            "<th>engine pig_removed</th><th>consistency</th><th>failure</th>"
            "<th>media</th></tr>" + "".join(sections) + "</table></body></html>")
    (DATA / "channel-verification.html").write_text(html, encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- pilot gate

def gate_cells(plan):
    return [cell_identity(system, seed, state["identity"])
            for system in SYSTEMS for seed in SEEDS
            for state in plan["states"] if state["identity"] in plan["pilot_states"]]


def evaluate_gate(plan):
    cells = gate_cells(plan)
    rows = []
    for identity in cells:
        record_file = OUTPUT / "records" / f"{identity}.json"
        receipt_file = OUTPUT / "receipts" / f"{identity}.json"
        if not record_file.is_file() or not receipt_file.is_file():
            raise ValueError(f"gate requires a terminal record for {identity}")
        record = read(record_file)
        outcome = record.get("outcome")
        rows.append({"identity": identity,
                     "failure": record.get("failure"),
                     "success": None if outcome is None else outcome.get("first_shot_success")})
    gate = engine.evaluate_gate(rows)
    gate.update({"schema": engine.SCHEMA_GATE, "plan_identity": plan["identity"],
                 "cells": len(cells), "metric": "engine-truth pig_removed first-shot success"})
    return gate


def pilot(args):
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    verification_file = OUTPUT / "channel-verification.json"
    if not verification_file.is_file():
        raise ValueError("the diagnostic requires the published Phase-A channel "
                         "verification record; run --smoke-test first")
    verdict = read(verification_file)
    if verdict["plan_identity"] != plan["identity"]:
        raise ValueError("channel verification belongs to a different plan")
    if not verdict["verified"]:
        log("channel verification failed: the diagnostic stays barred; publish "
            "readiness_or_precision_insufficient naming the blocker")
        return 2
    lock = gpu_lock()
    try:
        supervise(args, plan, "pilot")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    gate = evaluate_gate(plan)
    write(OUTPUT / "pilot-gate.json", gate)
    log(f"pilot gate: prevalence={gate['prevalence']:.4f} ({gate['successes']}/"
        f"{gate['valid_executions']}) floor={gate['floor']} passed={gate['passed']}")
    if not gate["passed"]:
        log("gate FAILED: the full run is barred; publish readiness_or_precision_insufficient")
        return 2
    return 0


def run(args):
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    gate_file = OUTPUT / "pilot-gate.json"
    if not gate_file.is_file():
        raise ValueError("the full run requires the recorded pilot-gate verdict; run --pilot first")
    stored = read(gate_file)
    gate = evaluate_gate(plan)
    if gate != stored:
        raise ValueError("stored pilot-gate verdict differs from its recomputation")
    if not gate["passed"]:
        log("pilot gate failed: the full run stays barred; disposition is "
            "readiness_or_precision_insufficient")
        return 2
    lock = gpu_lock()
    try:
        supervise(args, plan, "run")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return 0


# ---------------------------------------------------------------- publication

def read_record(identity):
    record_file = OUTPUT / "records" / f"{identity}.json"
    return read(record_file) if record_file.is_file() else None


def parameter_counts():
    """Total and executed-leaf parameter counts; shared tables counted in full."""
    counts = {"total": {}, "active": {PRIOR_SYSTEM: {str(seed): 0 for seed in SEEDS}}}
    plan = read(OUTPUT / "plan.json")
    carrier, action = torch.zeros(1, DIM), torch.zeros(1, 5)
    for seed in SEEDS:
        for arm in ("continuous", "hybrid"):
            model = load_frozen_predictor(plan, seed, arm, "cpu")
            total = int(sum(parameter.numel() for parameter in model.parameters()))
            counts["total"].setdefault(arm, {})[str(seed)] = total
            for system, (system_arm, pair) in SYSTEM_ARMS.items():
                if system_arm == arm:
                    counts["active"].setdefault(system, {})[str(seed)] = int(
                        active_capacity(model, carrier, action, pair))
    return counts


def cell_rows(plan, records, identity):
    record = records[identity]
    outcome = record.get("outcome")
    decision = record.get("decision") or {}
    execution = record.get("execution") or {}
    channel = record.get("engine_channel") or {}
    return {
        "failure": record.get("failure"),
        "failure_kind": record.get("failure_kind"),
        "success": None if outcome is None else outcome["first_shot_success"],
        "regret": None if outcome is None else outcome["regret"],
        "realized_count_cost": (None if outcome is None
                                else outcome.get("realized_count_cost")),
        "candidate_count": decision.get("candidate_count"),
        "transition_calls": decision.get("transition_calls"),
        "linear_macs": decision.get("linear_macs"),
        "prior_fallback_used": decision.get("prior_fallback_used"),
        "decision_wall_seconds": ((decision.get("perception_seconds") or 0.0)
                                  + (decision.get("wall_seconds") or 0.0)),
        "engine_wall_seconds": execution.get("engine_wall_seconds"),
        "wall_seconds": record.get("wall_seconds"),
        "channel_consistency": channel.get("consistency"),
    }


def phase_c_rows(plan, records):
    """Engine truth + proxy inputs over the pilot executions."""
    rows = []
    for identity in gate_cells(plan):
        record = records[identity]
        outcome = record.get("outcome")
        execution = record.get("execution") or {}
        rows.append({
            "identity": identity,
            "system": record["cell"]["system"], "seed": record["cell"]["seed"],
            "state": record["state_identity"],
            "success": None if outcome is None else outcome.get("first_shot_success"),
            "realized_count_cost": execution.get("realized_end_of_window_count_cost"),
            "failure": record.get("failure"),
        })
    return rows


def publication(plan):
    ledger = read(ledger_path())
    complete = (ledger.get("status") == "complete" and ledger.get("phase") == "run")
    channel_file = OUTPUT / "channel-verification.json"
    channel = read(channel_file) if channel_file.is_file() else None
    gate_file = OUTPUT / "pilot-gate.json"
    gate_verdict = read(gate_file) if gate_file.is_file() else None
    common = {
        "schema": engine.SCHEMA_REPORT, "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "diagnostics_complete": complete,
        "execution_ledger_status": ledger["status"],
        "wall_seconds_elapsed": ledger["wall_seconds_elapsed"],
        "gpu_seconds_elapsed": ledger["gpu_seconds_elapsed"],
        "caps": {"wall_cap_seconds": WALL_CAP_SECONDS, "gpu_cap_seconds": GPU_CAP_SECONDS},
        "engine_channel": plan["definitions"]["engine_channel"],
        "channel_verification": {
            "verified": None if channel is None else channel["verified"],
            "blocker": None if channel is None else channel["blocker"],
            "cells": None if channel is None else channel["cells"],
            "pass_rule": engine.VERIFICATION_PASS_RULE,
        },
        "pilot_gate": gate_verdict,
        "definitions": plan["definitions"],
        "claim_boundary": plan["definitions"]["claim_boundary"],
        "limitations": [
            "development/exposed N1 lineages only; zero fresh captures; not the sealed #64/#65 benchmark",
            "the engine-truth verdict covers the bounded 12 s native shot window per executed shot",
            "regret references the t=600 end-of-window replay cost: right-censored, not a settled cost",
            "states sharing a source member share one physical initial state and candidate table; "
            "the paired bootstrap over 24 state identities is descriptive only",
            "inference is NOT equalized across arms; per-arm decision compute is reported",
            "no adaptive system is present; the #74 hybrid_adaptive wall context carries no penalty narrative",
        ],
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }

    def blocked(dispositions, explanation):
        common["ticket_disposition"] = "readiness_or_precision_insufficient"
        common["question_dispositions"] = dispositions
        common["stopExplanation"] = explanation
        return common

    if channel is None:
        return blocked(
            {"q1_engine_truth_reactive_differences": "readiness_or_precision_insufficient",
             "q2_proxy_engine_agreement": "readiness_or_precision_insufficient"},
            "the Phase-A engine-channel verification record does not exist")
    if not channel["verified"]:
        return blocked(
            {"q1_engine_truth_reactive_differences": "readiness_or_precision_insufficient",
             "q2_proxy_engine_agreement": "readiness_or_precision_insufficient"},
            f"the engine channel could not be verified: {channel['blocker']}")
    if gate_verdict is None:
        return blocked(
            {"q1_engine_truth_reactive_differences": "readiness_or_precision_insufficient",
             "q2_proxy_engine_agreement": "readiness_or_precision_insufficient"},
            "the pilot gate verdict does not exist")
    parameters = parameter_counts()
    if not gate_verdict["passed"]:
        records = {identity: read_record(identity) for identity in gate_cells(plan)
                   if read_record(identity) is not None}
        proxy = engine.proxy_cross_check(phase_c_rows(plan, records))
        common["phase_c_proxy_cross_check"] = proxy
        # The gate bars Q1 scoring and the full run, but the executed pilot's
        # compute accounting is still published: per-state/arm wall time,
        # active-parameter/step counts and candidate counts for every arm.
        per_cell = {}
        rows_by_seed = {str(seed): {name: [] for name in SYSTEMS} for seed in SEEDS}
        for seed in SEEDS:
            for state in plan["states"]:
                if state["identity"] not in plan["pilot_states"]:
                    continue
                for system in SYSTEMS:
                    identity = cell_identity(system, seed, state["identity"])
                    row = cell_rows(plan, records, identity)
                    rows_by_seed[str(seed)][system].append(row)
                    per_cell[identity] = {
                        "wall_seconds": row["wall_seconds"],
                        "decision_wall_seconds": row["decision_wall_seconds"],
                        "engine_wall_seconds": row["engine_wall_seconds"],
                        "transition_calls": row["transition_calls"],
                        "linear_macs": row["linear_macs"],
                        "candidate_count": row["candidate_count"],
                        "active_predictor_parameters": (
                            parameters["active"][system][str(seed)]
                            if row["transition_calls"] else 0),
                        "failure_kind": row["failure_kind"],
                    }
        common["pilot_compute_accounting"] = {
            "scope": ("the 48 executed-or-typed-failure pilot cells; Q1 scoring "
                      "and the full matrix stay barred by the failed gate"),
            "per_seed": {seed: {name: aggregate_rows(rows)
                                for name, rows in systems.items()}
                         for seed, systems in rows_by_seed.items()},
            "pooled": {name: aggregate_rows(
                [row for systems in rows_by_seed.values() for row in systems[name]])
                for name in SYSTEMS},
            "parameter_counts": parameters["total"],
            "active_parameter_counts": parameters["active"],
            "per_cell": per_cell,
        }
        q2 = ("supported" if proxy["cells_compared"] >= engine.PILOT_MINIMUM_VALID_EXECUTIONS
              else "readiness_or_precision_insufficient")
        return blocked(
            {"q1_engine_truth_reactive_differences": "readiness_or_precision_insufficient",
             "q2_proxy_engine_agreement": q2},
            "the mandatory engine-truth prevalence pilot gate failed before the full run")

    records = {}
    missing = []
    for system in SYSTEMS:
        for seed in SEEDS:
            for state in plan["states"]:
                identity = cell_identity(system, seed, state["identity"])
                record = read_record(identity)
                if record is None:
                    missing.append(identity)
                else:
                    records[identity] = record
    if missing:
        raise ValueError(f"publication requires a terminal record for every cell; "
                         f"{len(missing)} missing, e.g. {missing[:3]}")
    inventory = {"scheduled_cells": len(records),
                 "executed": sum(1 for r in records.values() if r.get("execution")),
                 "decision_failures": sum(1 for r in records.values()
                                          if r.get("failure_kind") == "decision_failure"),
                 "execution_failures": sum(1 for r in records.values()
                                           if r.get("failure_kind") == "execution_failure"),
                 "not_executed": sum(1 for r in records.values()
                                     if r.get("failure_kind") == "not_executed")}
    common["inventory"] = inventory
    proxy = engine.proxy_cross_check(phase_c_rows(plan, records))
    common["phase_c_proxy_cross_check"] = proxy
    if not complete:
        q2 = ("supported" if proxy["cells_compared"] >= engine.PILOT_MINIMUM_VALID_EXECUTIONS
              else "readiness_or_precision_insufficient")
        return blocked(
            {"q1_engine_truth_reactive_differences": "readiness_or_precision_insufficient",
             "q2_proxy_engine_agreement": q2},
            f"execution inventory incomplete or a cap stop occurred (ledger: {ledger['status']})")

    per_seed, per_state = {}, {}
    for seed in SEEDS:
        rows = {name: {} for name in SYSTEMS}
        for state in plan["states"]:
            for system in SYSTEMS:
                rows[system][state["identity"]] = cell_rows(
                    plan, records, cell_identity(system, seed, state["identity"]))
        per_state[str(seed)] = rows
        per_seed[str(seed)] = {name: aggregate_rows(list(rows[name].values()))
                               for name in SYSTEMS}
    pooled = {name: aggregate_rows(
        [row for seed in SEEDS for row in per_state[str(seed)][name].values()])
        for name in SYSTEMS}
    contrasts = engine.summarize_contrasts(per_state, SEEDS)
    primary = next(contrast for contrast in contrasts
                   if (contrast["metric"], contrast["reference"]) == SUCCESS_PRIMARY_CONTRAST)
    q1, q1_reason = engine.disposition_for_primary(primary)
    agreement = engine.channel_agreement_table(
        [cell_rows(plan, records, cell_identity(system, seed, state["identity"]))
         for system in SYSTEMS for seed in SEEDS for state in plan["states"]])
    q2 = ("supported" if proxy["cells_compared"] >= engine.PILOT_MINIMUM_VALID_EXECUTIONS
          else "readiness_or_precision_insufficient")
    return {**common,
            "per_seed": per_seed, "pooled": pooled,
            "prior_arm": {"fallback_rule": engine.PRIOR_FALLBACK_RULE,
                          "fallback_states": sorted(
                              state["identity"] for state in plan["states"]
                              if state["prior_choice"] != PRIOR_ORDINAL),
                          "choices": {state["identity"]: state["prior_choice"]
                                      for state in plan["states"]}},
            "contrasts": contrasts,
            "primary_contrast": {"metric": SUCCESS_PRIMARY_CONTRAST[0],
                                 "reference": SUCCESS_PRIMARY_CONTRAST[1],
                                 "rule": plan["definitions"]["disposition_rule"]["supported"]},
            "closed_loop_channel_agreement": agreement,
            "parameter_counts": parameters["total"],
            "active_parameter_counts": parameters["active"],
            "question_dispositions": {
                "q1_engine_truth_reactive_differences": q1,
                "q1_reason": q1_reason,
                "q2_proxy_engine_agreement": q2,
            },
            "ticket_disposition": q1}


def aggregate_rows(rows):
    valid = [row for row in rows if row["failure"] is None]
    regrets = [row["regret"] for row in valid if row["regret"] is not None]
    successes = [row["success"] for row in valid if row["success"] is not None]
    costs = [row["realized_count_cost"] for row in valid if row["realized_count_cost"] is not None]
    return {
        "states": len(rows), "executed": len(valid),
        "decision_failures": sum(1 for row in rows if row["failure_kind"] == "decision_failure"),
        "execution_failures": sum(1 for row in rows if row["failure_kind"] == "execution_failure"),
        "not_executed": sum(1 for row in rows if row["failure_kind"] == "not_executed"),
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "success_fraction": float(np.mean(successes)) if successes else None,
        "successes": int(sum(successes)),
        "mean_realized_count_cost": float(np.mean(costs)) if costs else None,
        "candidate_counts": sorted({row["candidate_count"] for row in valid
                                    if row["candidate_count"] is not None}),
        "transition_calls": sum(row["transition_calls"] or 0 for row in valid),
        "linear_macs": sum(row["linear_macs"] or 0 for row in valid),
        "mean_decision_wall_seconds": float(np.mean(
            [row["decision_wall_seconds"] for row in valid])) if valid else None,
        "mean_engine_wall_seconds": float(np.mean(
            [row["engine_wall_seconds"] for row in valid
             if row["engine_wall_seconds"] is not None])) if valid else None,
        "mean_wall_seconds": float(np.mean(
            [row["wall_seconds"] for row in valid
             if row["wall_seconds"] is not None])) if valid else None,
    }


def compact_report(result):
    return {key: value for key, value in result.items() if key not in ("per_state",)}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("seed", "system", "executed", "mean_regret", "success_fraction",
                     "successes", "mean_realized_count_cost", "decision_failures",
                     "execution_failures", "candidate_counts", "transition_calls",
                     "linear_macs", "mean_decision_wall_seconds",
                     "mean_engine_wall_seconds"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, row in systems.items():
            writer.writerow((seed, name, row["executed"], row["mean_regret"],
                             row["success_fraction"], row["successes"],
                             row["mean_realized_count_cost"],
                             row["decision_failures"], row["execution_failures"],
                             " ".join(str(c) for c in row["candidate_counts"]),
                             row["transition_calls"], row["linear_macs"],
                             row["mean_decision_wall_seconds"],
                             row["mean_engine_wall_seconds"]))
    writer.writerow(())
    writer.writerow(("contrast_tested", "reference", "metric", "mean",
                     "descriptive_95_low", "descriptive_95_high", "paired_states",
                     "label"))
    for contrast in result.get("contrasts", []):
        interval = (contrast.get("descriptive") or {}).get("descriptive_95_percent_interval")
        writer.writerow((contrast["tested"], contrast["reference"], contrast["metric"],
                         None if not contrast.get("descriptive") else contrast["descriptive"]["mean"],
                         None if not interval else interval[0],
                         None if not interval else interval[1],
                         contrast["paired_states"], "DESCRIPTIVE"))
    writer.writerow(())
    proxy = result.get("phase_c_proxy_cross_check")
    writer.writerow(("phase_c_proxy_vs_engine", "cells_compared", "both_success",
                     "proxy_only", "engine_only", "both_failure", "agreement",
                     "label"))
    if proxy is not None:
        counts = proxy["counts"]
        writer.writerow(("realized_cost<=82_published_threshold", proxy["cells_compared"],
                         counts["both_success"], counts["proxy_only"],
                         counts["engine_only"], counts["both_failure"],
                         proxy["agreement"], "DESCRIPTIVE"))
        for system, bucket in sorted((proxy.get("per_system") or {}).items()):
            writer.writerow((f"realized_cost<=82_published_threshold[{system}]",
                             bucket["cells"], bucket["both_success"], bucket["proxy_only"],
                             bucket["engine_only"], bucket["both_failure"],
                             (bucket["both_success"] + bucket["both_failure"]) / bucket["cells"]
                             if bucket["cells"] else None, "DESCRIPTIVE"))
        interval = (proxy.get("paired_state_differences") or {}).get("descriptive")
        if interval is not None:
            low, high = interval["descriptive_95_percent_interval"]
            writer.writerow(("phase_c_paired_state_differences(proxy-engine)",
                             proxy["paired_state_differences"]["paired_states"],
                             interval["mean"], low, high,
                             proxy["paired_state_differences"]["paired_states"], "",
                             "DESCRIPTIVE"))
    accounting = result.get("pilot_compute_accounting")
    if accounting is not None:
        writer.writerow(())
        writer.writerow(("pilot_compute_accounting", "seed", "system", "executed",
                         "successes", "candidate_counts", "transition_calls",
                         "linear_macs", "mean_decision_wall_seconds",
                         "mean_engine_wall_seconds", "mean_wall_seconds"))
        for seed, systems in accounting["per_seed"].items():
            for name, row in systems.items():
                writer.writerow(("pilot_compute_accounting", seed, name, row["executed"],
                                 row["successes"],
                                 " ".join(str(c) for c in row["candidate_counts"]),
                                 row["transition_calls"], row["linear_macs"],
                                 row["mean_decision_wall_seconds"],
                                 row["mean_engine_wall_seconds"], row["mean_wall_seconds"]))
        for arm, counts in sorted((accounting.get("parameter_counts") or {}).items()):
            writer.writerow(("pilot_frozen_predictor_parameters", arm,
                             " ".join(f"{seed}:{count}" for seed, count in sorted(counts.items()))))
        for system, counts in sorted(accounting["active_parameter_counts"].items()):
            writer.writerow(("pilot_active_predictor_parameters", system,
                             " ".join(f"{seed}:{count}" for seed, count in sorted(counts.items()))))
    return stream.getvalue()


def fmt(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def findings_md(result):
    lines = ["# Issue-85 engine-side outcome channel and bounded reactive "
             "re-diagnostic - findings", ""]
    lines.append(f"Diagnostics complete: {result['diagnostics_complete']}. "
                 f"Ledger status: {result['execution_ledger_status']}.")
    lines.append("")
    lines.append(f"Claim boundary: {result['claim_boundary']}")
    lines.append("")
    lines.append("## Phase A - the engine-side outcome channel")
    lines.append("")
    lines.append("Detection rule: " + result["definitions"]["engine_channel"]["detection_rule"])
    lines.append("")
    lines.append("Failure modes: " + result["definitions"]["engine_channel"]["failure_modes"])
    lines.append("")
    channel = result["channel_verification"]
    lines.append(f"Channel verified: {channel['verified']}.")
    if channel["blocker"]:
        lines.append(f"Blocker: {channel['blocker']}.")
    lines.append("")
    for detail in channel.get("cells") or []:
        lines.append(f"- {detail['identity']} role={detail['role']} "
                     f"expected pig_removed={detail['expectation']} -> "
                     f"engine pig_removed={detail['pig_removed']} "
                     f"consistency={detail['consistency']} "
                     f"failure={detail['failure'] or 'none'}")
    if channel.get("cells"):
        lines.append("")
    lines.append("## Phase C - #82 proxy predicate vs engine truth on the pilot (DESCRIPTIVE)")
    lines.append("")
    proxy = result.get("phase_c_proxy_cross_check")
    if proxy is None:
        lines.append("Not computed (the pilot never executed).")
        lines.append("")
    else:
        counts = proxy["counts"]
        interval = (proxy.get("paired_state_differences") or {}).get("descriptive")
        lines.append(f"Rule: {proxy['rule']}. Compared {proxy['cells_compared']} valid pilot "
                     f"cells ({proxy['cells_excluded']} typed-failure cells excluded); "
                     f"agreement {fmt(proxy['agreement'])} "
                     f"(both-success {counts['both_success']}, proxy-only "
                     f"{counts['proxy_only']}, engine-only {counts['engine_only']}, "
                     f"both-failure {counts['both_failure']}).")
        lines.append("")
        if interval is not None:
            low, high = interval["descriptive_95_percent_interval"]
            lines.append(f"Paired proxy-minus-engine differences over pilot states "
                         f"(seeds averaged first; DESCRIPTIVE): mean "
                         f"{interval['mean']:+.4f}, descriptive 95% interval "
                         f"[{low:+.4f}, {high:+.4f}] over {proxy['paired_state_differences']['paired_states']} "
                         "state identities (positive = the proxy over-calls success).")
            lines.append("")
    gate = result.get("pilot_gate")
    lines.append("## Engine-truth prevalence pilot gate")
    lines.append("")
    if gate is not None:
        lines.append(f"Prevalence {gate['prevalence']:.4f} ({gate['successes']}/"
                     f"{gate['valid_executions']} valid executions; "
                     f"{len(gate['typed_failures'])} typed failures); "
                     f"floor {gate['floor']:.2f}; passed={gate['passed']}.")
    else:
        lines.append("No gate verdict (blocked before the pilot).")
    lines.append("")
    if not result.get("per_seed"):
        accounting = result.get("pilot_compute_accounting")
        if accounting is not None:
            lines.append("## Pilot compute accounting (executed cells; the gate barred "
                         "Q1 scoring and the full run)")
            lines.append("")
            lines.append("| System | Seed | Executed | Successes | Candidate counts | "
                         "Transition calls | Linear MACs | Decision wall s | Engine wall s | Wall s |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for seed, systems in accounting["per_seed"].items():
                for name, row in systems.items():
                    lines.append(
                        f"| {name} | {seed} | {row['executed']} | {row['successes']} | "
                        f"{' '.join(str(c) for c in row['candidate_counts'])} | "
                        f"{row['transition_calls']} | {row['linear_macs']} | "
                        f"{fmt(row['mean_decision_wall_seconds'], 3)} | "
                        f"{fmt(row['mean_engine_wall_seconds'], 1)} | "
                        f"{fmt(row['mean_wall_seconds'], 1)} |")
            lines.append("")
            for arm, counts in sorted((accounting.get("parameter_counts") or {}).items()):
                lines.append(f"- {arm}: frozen predictor parameters "
                             f"{sorted(set(counts.values()))} across seeds {sorted(counts)}")
            for system, counts in sorted(accounting["active_parameter_counts"].items()):
                lines.append(f"- {system}: active predictor parameters per transition "
                             f"{sorted(set(counts.values()))}; shared tables counted in full")
            if accounting.get("parameter_counts"):
                lines.append("")
            lines.append("Per-cell wall/step/candidate rows are published in "
                         "summary.json pilot_compute_accounting.per_cell. These are "
                         "accounting tables, not Q1 scoring; Q1 stays barred by the gate.")
            lines.append("")
        lines.append("Full-matrix execution incomplete; Q1 paired outcomes are not scored.")
        lines.append("")
        lines.append(f"Ticket disposition: {result.get('ticket_disposition')} "
                     "(tokens: supported / not_supported_by_this_experiment / "
                     "readiness_or_precision_insufficient).")
        lines.append("")
        for name, token in (result.get("question_dispositions") or {}).items():
            if name.endswith("_reason"):
                continue
            lines.append(f"- {name}: {token}")
        lines.append("")
        for item in result["limitations"]:
            lines.append(f"- {item}")
        return "\n".join(lines) + "\n"
    lines.append("## First-shot engine-truth outcomes (per seed; DESCRIPTIVE)")
    lines.append("")
    lines.append("| System | Seed | Executed | Mean regret | Success | Successes | "
                 "Mean cost | Decision wall s | Engine wall s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for seed, systems in result["per_seed"].items():
        for name, row in systems.items():
            lines.append(
                f"| {name} | {seed} | {row['executed']} | "
                f"{fmt(row['mean_regret'])} | "
                f"{fmt(row['success_fraction'], 3)} | {row['successes']} | "
                f"{fmt(row['mean_realized_count_cost'], 2)} | "
                f"{fmt(row['mean_decision_wall_seconds'], 3)} | "
                f"{fmt(row['mean_engine_wall_seconds'], 1)} |")
    lines.append("")
    lines.append("## Pooled over seeds (DESCRIPTIVE)")
    lines.append("")
    prior_arm = result.get("prior_arm") or {}
    if prior_arm:
        lines.append(f"Prior-arm fallback ({PRIOR_FALLBACK_RULE_SHORT}) applied on "
                     f"{len(prior_arm['fallback_states'])} of 24 membership states.")
        lines.append("")
    lines.append("| System | Executed | Mean regret | Success | Successes | "
                 "Transition calls | Linear MACs | Candidate counts |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, row in result["pooled"].items():
        lines.append(
            f"| {name} | {row['executed']} | "
            f"{fmt(row['mean_regret'])} | "
            f"{fmt(row['success_fraction'], 3)} | {row['successes']} | "
            f"{row['transition_calls']} | {row['linear_macs']} | "
            f"{' '.join(str(c) for c in row['candidate_counts'])} |")
    lines.append("")
    lines.append("## Paired contrasts, hybrid-fixed-h1 vs reference (positive favors "
                 "hybrid; DESCRIPTIVE)")
    lines.append("")
    lines.append("| Reference | Metric | Mean difference | Descriptive 95% interval | Paired states |")
    lines.append("| --- | --- | --- | --- | --- |")
    for contrast in result["contrasts"]:
        descriptive = contrast.get("descriptive")
        if descriptive is None:
            lines.append(f"| {contrast['reference']} | {contrast['metric']} | n/a | n/a | "
                         f"{contrast['paired_states']} |")
            continue
        interval = descriptive["descriptive_95_percent_interval"]
        lines.append(f"| {contrast['reference']} | {contrast['metric']} | "
                     f"{descriptive['mean']:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] | "
                     f"{contrast['paired_states']} |")
    lines.append("")
    agreement = result.get("closed_loop_channel_agreement")
    if agreement is not None:
        lines.append("## Closed-loop channel agreement, event leg vs lifecycle leg "
                     "(all executed diagnostic cells)")
        lines.append("")
        lines.append(f"agreement {agreement['agreement']}, "
                     f"death_at_window_edge {agreement['death_at_window_edge']} "
                     f"(declared timing mode), anomaly {agreement['anomaly']}.")
        lines.append("")
    parameters = result.get("parameter_counts")
    if parameters:
        lines.append("## Compute accounting")
        lines.append("")
        for arm, counts in sorted(parameters.items()):
            lines.append(f"- {arm}: frozen predictor parameters "
                         f"{sorted(set(counts.values()))} across seeds {sorted(counts)}")
        lines.append("")
    lines.append("## Disposition")
    lines.append("")
    for name, token in (result.get("question_dispositions") or {}).items():
        if name.endswith("_reason"):
            lines.append(f"  - reason: {token}")
        else:
            lines.append(f"- {name}: {token}")
    lines.append("")
    lines.append(f"Ticket disposition: {result.get('ticket_disposition')} "
                 "(tokens: supported / not_supported_by_this_experiment / "
                 "readiness_or_precision_insufficient; the pre-declared practical "
                 "effect margin is the interval-exclusion rule on the primary "
                 "success contrast).")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- review gallery

def publish_gallery(plan):
    """Copy one decision frame + one WebM per cell under data/, incl. failures."""
    _verify_webm_encoder()
    root = DATA / "cells"
    manifest = {"schema": engine.SCHEMA_GALLERY, "identity": IDENTITY,
                "plan_identity": plan["identity"], "playback_fps": PILOT_AUDIT_FPS,
                "canonical_observations_included": False, "cells": []}
    for system in SYSTEMS:
        for seed in SEEDS:
            for state in plan["states"]:
                identity = cell_identity(system, seed, state["identity"])
                record_file = OUTPUT / "records" / f"{identity}.json"
                entry = {"identity": identity, "system": system, "seed": seed,
                         "state": state["identity"],
                         "status": "missing" if not record_file.is_file() else (
                             "typed_failure" if read(record_file).get("failure") else "executed")}
                cell_root = root / identity
                cell_root.mkdir(parents=True, exist_ok=True)
                record = read(record_file) if record_file.is_file() else {}
                executed = record.get("failure") is None and bool(record.get("execution"))
                decision_ref = None
                if record.get("decision") is not None:
                    source = OUTPUT / "attempts" / identity / "decision-frame.png"
                    if source.is_file():
                        shutil.copy2(source, cell_root / "decision.png")
                        decision_ref = str((cell_root / "decision.png").relative_to(DATA))
                frames = []
                if record.get("execution"):
                    observation_root = OUTPUT / "attempts" / identity / "shot-1" / "observation-trace"
                    if observation_root.is_dir():
                        trace_manifest = read(observation_root / MANIFEST_NAME)
                        frames = [observation_root / frame["agent_observation"]["relative_path"]
                                  for frame in trace_manifest["frame_records"]]
                if frames:
                    video = cell_root / "shot.webm"
                    if not video.exists():
                        try:
                            _encode_agent_frames_webm(frames, video)
                        except Exception as error:
                            entry["video_error"] = f"{type(error).__name__}: {error}"
                    if video.exists():
                        entry["video"] = str(video.relative_to(DATA))
                        entry["video_frames"] = len(frames)
                if decision_ref:
                    entry["decision_frame"] = decision_ref
                if executed and not (entry.get("decision_frame") and entry.get("video")):
                    raise ValueError(
                        f"executed cell {identity} cannot be published without its "
                        f"decision frame and shot video: "
                        f"{entry.get('video_error') or 'media missing'}")
                entry["first_shot_success"] = None if not record.get("outcome") else bool(
                    record["outcome"]["first_shot_success"])
                entry["failure"] = record.get("failure")
                manifest["cells"].append(entry)
    write(DATA / "manifest.json", manifest)
    sections = []
    for entry in manifest["cells"]:
        media = ""
        if entry.get("decision_frame"):
            media += f'<img src="{entry["decision_frame"]}" width="320"/> '
        if entry.get("video"):
            media += f'<video src="{entry["video"]}" controls width="320"></video>'
        status = entry["status"]
        success = entry["first_shot_success"]
        sections.append(
            f"<tr><td>{entry['identity']}</td><td>{status}</td>"
            f"<td>{'n/a' if success is None else success}</td>"
            f"<td>{entry['failure'] or ''}</td><td>{media}</td></tr>")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Issue 85 engine-outcome reactive diagnostic review</title></head><body>"
            "<h1>Issue 85 bounded engine-truth reactive re-diagnostic</h1>"
            "<p>One decision frame and one 50 fps WebM per scheduled cell, including "
            "typed failures; the Phase-A channel controls are published alongside at "
            "channel-verification.html. Agent observations only; canonical frames "
            "excluded.</p>"
            "<table border='1'><tr><th>cell</th><th>status</th><th>engine-truth "
            "first-shot success</th><th>failure</th><th>media</th></tr>" + "".join(sections) +
            "</table></body></html>")
    (DATA / "index.html").write_text(html, encoding="utf-8")
    return manifest


def check_gallery(plan, manifest):
    """Recompute the expected gallery inventory and verify the stored one."""
    if (manifest.get("schema") != engine.SCHEMA_GALLERY
            or manifest.get("plan_identity") != plan["identity"]
            or len(manifest["cells"]) != 24 * len(SYSTEMS) * len(SEEDS)):
        raise ValueError("review gallery manifest binding differs")
    expected_identities = {cell_identity(system, seed, state["identity"])
                           for system in SYSTEMS for seed in SEEDS
                           for state in plan["states"]}
    if {entry["identity"] for entry in manifest["cells"]} != expected_identities:
        raise ValueError("review gallery cell inventory differs")
    for entry in manifest["cells"]:
        for key in ("decision_frame", "video"):
            if entry.get(key) and not (DATA / entry[key]).is_file():
                raise ValueError(f"gallery media missing for {entry['identity']}")
        record_file = OUTPUT / "records" / f"{entry['identity']}.json"
        if not record_file.is_file():
            # Post-publish evidence drift: a manifest entry must claim
            # "missing" exactly when its record is gone.
            if entry.get("status") != "missing":
                raise ValueError(f"gallery status differs for {entry['identity']}")
            continue
        record = read(record_file)
        executed = record.get("failure") is None and record.get("execution")
        if executed and (not entry.get("decision_frame") or not entry.get("video")):
            raise ValueError(f"executed cell {entry['identity']} lacks gallery media")
        if entry.get("failure") != record.get("failure"):
            raise ValueError(f"gallery failure text differs for {entry['identity']}")
        recomputed_status = ("missing" if not record_file.is_file() else
                             ("typed_failure" if record.get("failure") else "executed"))
        if entry.get("status") != recomputed_status:
            raise ValueError(f"gallery status differs for {entry['identity']}")
        recomputed_success = (None if not record.get("outcome")
                              else bool(record["outcome"]["first_shot_success"]))
        if entry.get("first_shot_success") != recomputed_success:
            raise ValueError(f"gallery first_shot_success differs for {entry['identity']}")
    index = DATA / "index.html"
    if not index.is_file():
        raise ValueError("review gallery index.html is missing")
    index_text = index.read_text(encoding="utf-8")
    for entry in manifest["cells"]:
        if entry["identity"] not in index_text:
            raise ValueError(f"review gallery index.html omits {entry['identity']}")


def channel_projection(verdict):
    """Timestamp-free projection of the channel record that must recompute."""
    if verdict is None:
        return None
    return {key: verdict[key] for key in
            ("verified", "cells", "anomalies", "failures", "positives_matched",
             "negatives_matched", "blocker", "pass_rule", "review_rule",
             "detection_rule", "failure_modes", "wall_seconds",
             "wall_seconds_scope", "failed_harness", "executed_at_utc",
             "executed_at_receipt", "controls_execution_run_end_utc")}


# ---------------------------------------------------------------------- modes

def dry_run(args):
    plan = make_plan()
    states = plan["states"]
    fallback = [state["identity"] for state in states if state["prior_choice"] != PRIOR_ORDINAL]
    controls = [f"{c['role']}:{c['state']}#a{c['ordinal']:02d}->expect={c['expectation']}"
                for c in plan["verification_set"]]
    log(f"no-write dry-run: states={len(states)} "
        f"per_family={ {family: sum(1 for s in states if s['generator_family'] == family) for family in FAMILIES} } "
        f"systems={len(SYSTEMS)} seeds={len(SEEDS)} cells={len(states) * len(SYSTEMS) * len(SEEDS)} "
        f"pilot_cells={len(gate_cells(plan))} workers={WORKERS} "
        f"wall_cap_s={WALL_CAP_SECONDS} gpu_cap_s={GPU_CAP_SECONDS} "
        f"verification_controls={controls} "
        f"prior_fallback_states={fallback}")
    return 0


def prepare(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_file = OUTPUT / "plan.json"
    if plan_file.exists():
        try:
            plan = check_plan_bound(read(plan_file))
        except ValueError as error:
            records_dir = OUTPUT / "records"
            ledger = read(OUTPUT / "ledger.json") if (OUTPUT / "ledger.json").is_file() else {}
            execution_started = (bool(ledger.get("cells"))
                                 or (records_dir.exists() and any(records_dir.iterdir())))
            if execution_started:
                raise ValueError(
                    "the frozen issue-85 plan changed after execution started; "
                    "audit and version the campaign instead of re-freezing") from error
            superseded = plan_file.with_name(
                "plan.json.superseded-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
            plan_file.replace(superseded)
            log(f"stale pre-execution freeze versioned as {superseded.name}; re-freezing")
            plan = make_plan()
            write(plan_file, plan)
    else:
        plan = make_plan()
        write(plan_file, plan)
    require_prepared(plan)
    log("issue-85 protocol frozen (engine-channel rule, verification set, "
        "membership, gate, caps, fallback rule, proxy threshold); the frozen #80 "
        "candidate outcome tables bind; no gameplay executed")
    return 0


def publish(args):
    plan = load_plan()
    check_plan_bound(plan)
    result = publication(plan)
    write(OUTPUT / "summary.json", compact_report(result))
    (OUTPUT / "comparisons.csv").write_text(comparisons_csv(result))
    (OUTPUT / "findings.md").write_text(findings_md(result))
    manifest = publish_gallery(plan)
    log(f"published diagnostics_complete={result['diagnostics_complete']} "
        f"ticket_disposition={result.get('ticket_disposition')} "
        f"question_dispositions={result.get('question_dispositions')} "
        f"gallery_cells={len(manifest['cells'])}")
    return 0


def validate(args):
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    result = publication(plan)
    problems = []
    if compact_report(result) != read(OUTPUT / "summary.json"):
        problems.append("summary.json")
    if comparisons_csv(result).encode("utf-8") != (OUTPUT / "comparisons.csv").read_bytes():
        problems.append("comparisons.csv")
    if findings_md(result) != (OUTPUT / "findings.md").read_text(encoding="utf-8"):
        problems.append("findings.md")
    gate = read(OUTPUT / "pilot-gate.json") if (OUTPUT / "pilot-gate.json").is_file() else None
    if gate is not None and gate != evaluate_gate(plan):
        problems.append("pilot-gate.json")
    channel_file = OUTPUT / "channel-verification.json"
    if not channel_file.is_file():
        problems.append("channel-verification.json: missing")
    else:
        stored = read(channel_file)
        control_records = []
        for entry in plan["verification_set"]:
            identity = f"verification--{entry['role']}--{entry['state']}"
            record = read(OUTPUT / "records" / f"{identity}.json")
            control_records.append((identity, record))
            # Recompute the published per-cell channel values from the
            # retained native event stream; never trust the saved verdict.
            native_root = (record.get("execution") or {}).get("native_root")
            if native_root is None:
                if record.get("failure") is None:
                    problems.append(f"{identity}: executed control lacks its native root")
                continue
            rescanned = channel_scan(native_root)
            saved_channel = record.get("engine_channel") or {}
            for key in ("pig_removed", "structure_destroyed", "bird_launches",
                        "consistency", "pig_lifecycle_destroyed"):
                if rescanned.get(key) != saved_channel.get(key):
                    problems.append(f"{identity}: engine_channel[{key}] differs from "
                                    f"the native rescan ({saved_channel.get(key)} vs {rescanned.get(key)})")
        entries = [(entry, read(OUTPUT / "records" /
                                f"verification--{entry['role']}--{entry['state']}.json"))
                   for entry in plan["verification_set"]]
        recomputed = {**engine.verification_verdict(entries),
                      "detection_rule": engine.CHANNEL_DETECTION_RULE,
                      "failure_modes": engine.CHANNEL_FAILURE_MODES}
        media_review_file = DATA / "channel-verification-media-review.json"
        if not media_review_file.is_file():
            problems.append("channel-verification-media-review.json: missing")
        else:
            media_review = read(media_review_file)
            recomputed = engine.apply_media_review(recomputed, media_review)
        recomputed.update(channel_wall_accounting(control_records))
        if channel_projection(recomputed) != channel_projection(stored):
            problems.append("channel-verification.json")
        stored_gallery = DATA / "channel-verification-manifest.json"
        if not stored_gallery.is_file():
            problems.append("channel-verification-manifest.json: missing")
        else:
            manifest = read(stored_gallery)
            if (manifest.get("schema") != "issue_85_channel_gallery_v1"
                    or manifest.get("plan_identity") != plan["identity"]
                    or {entry["identity"] for entry in manifest["cells"]}
                    != {f"verification--{entry['role']}--{entry['state']}"
                        for entry in plan["verification_set"]}):
                problems.append("channel-verification-manifest.json: inventory")
            for entry in manifest["cells"]:
                for key in ("decision_frame", "video", "final_frame"):
                    if entry.get(key) and not (DATA / entry[key]).is_file():
                        problems.append(f"channel gallery media missing for {entry['identity']}")
                        break
                record = read(OUTPUT / "records" / f"{entry['identity']}.json")
                if (record.get("failure") is None and record.get("engine_channel")
                        and not (entry.get("decision_frame") and entry.get("video")
                                 and entry.get("final_frame"))):
                    problems.append(f"executed control {entry['identity']} lacks channel gallery media")
                saved_channel = record.get("engine_channel") or {}
                for key in ("pig_removed", "consistency"):
                    if entry.get(key) != saved_channel.get(key):
                        problems.append(f"channel gallery {key} differs from the record "
                                        f"for {entry['identity']}")
                recomputed_matched = (record.get("failure") is None
                                      and saved_channel.get("bird_launches") == 1
                                      and saved_channel.get("pig_removed") == entry.get("expectation"))
                if entry.get("expectation_matched") != recomputed_matched:
                    problems.append(f"channel gallery expectation_matched differs for {entry['identity']}")
                if entry.get("failure") != record.get("failure"):
                    problems.append(f"channel gallery failure text differs for {entry['identity']}")
    if (DATA / "manifest.json").is_file():
        try:
            check_gallery(plan, read(DATA / "manifest.json"))
        except ValueError as error:
            problems.append(f"gallery: {error}")
    else:
        problems.append("gallery manifest.json: missing")
    if problems:
        raise ValueError(f"published issue-85 artifacts differ from bound source evidence: {problems}")
    log("exact saved-evidence validation passed: every published table recomputed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "pilot", "run", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DECISION_DEVICE)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            return dry_run(args)
        if args.smoke_test:
            return smoke(args)
        if args.prepare:
            return prepare(args)
        if args.pilot:
            return pilot(args)
        if args.run:
            return run(args)
        if args.publish:
            return publish(args)
        return validate(args)
    except (ValueError, OSError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
