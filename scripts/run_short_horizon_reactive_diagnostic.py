"""Issue-80 bounded non-final short-horizon reactive-control diagnostic.

Bounded diagnostic (non-final, development data only): paired first-shot
outcome/regret differences of reactive shot selection between hybrid-fixed-h1,
continuous-fixed-h1, continuous-fixed-h5, and the frozen no-model ordinal
prior, over 24 frozen N1 admissible-branch initial states x 3 seeds, executed
closed-loop in the real rendered engine under isolated processes.

Frozen-before-outcome discipline (declared before any execution):

- Membership: 24 states - 12 per N1 generator family (type010103 rolling,
  type010105 sliding) - drawn from the N1 admissible-branch initial states
  (issue-77-n1-v1 coverage), evenly spaced in sorted state-identity order.
  A state's identity is the admissible branch identity; its initial state is
  that branch's source-member decision state at native fixed step 30000.
- Systems: hybrid-fixed-h1 (hybrid-arm checkpoint at fixed pair (1,
  continuous)), continuous-fixed-h1, continuous-fixed-h5, and the frozen
  no-model ordinal prior (always candidate ordinal 8 when admissible).
- Reactive decision: the sealed pre-decision frame is parsed with the frozen
  issue-70 adapter exactly as #77 built per-state ranking contexts
  (single-frame temporal context), the system's fixed pair rolls the 13
  admissible candidates forward, and the argmin predicted count cost
  (ties broken by lower ordinal) is the executed shot.
- Estimand: normalized ranking regret of the EXECUTED shot's realized
  end-of-window count cost (offset 600, terminal-absorbed, right-censored,
  NOT a settled cost) against the state's frozen candidate outcome table
  recorded by the #77 N1 campaign; first-shot success = pig_removed engine
  evidence of the executed segment; paired descriptive bootstrap intervals
  over the 24 states (seeds averaged before resampling).
- Nonzero-prevalence precondition: a 48-execution pilot (4 states x 4
  systems x 3 seeds) must show pooled first-shot success prevalence >= 0.10
  or the ticket stops with readiness_or_precision_insufficient BEFORE the
  full run (#77 recorded Top1 = 0.000 on all 36 system-seed rows).
- Caps: 24 worker-hours wall across the pilot+run execution phases; <= 2
  GPU-hours of active decision work inside that wall cap. Exceeding either
  stops the ticket with readiness_or_precision_insufficient.
- Zero fresh captures: every execution re-runs a frozen N1 level instance
  with one action of its frozen 13-candidate inventory; no new scenario
  lineage, no sealed/final lineage, and #64/#65 stay unauthorized.
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
import tempfile
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
from world_model.model import Abstraction, PredictionPair
from world_model.planning.task_objective import TaskObjective
from world_model.training.lineage_scaling import (
    NO_MODEL_ORDINAL_PRIOR_ORDINAL,
    GameplayPlanningMode,
    fixed_mode_pair,
    ordinal_prior_candidate,
)
from world_model.training.matched_dynamics import (
    ContinuousDynamics,
    CNNHybridPredictor,
    pairs_for,
    work as pair_work,
)

ROOT = train77.ROOT
CAMPAIGN = train77.CAMPAIGN
DYNAMICS = train77.OUTPUT
OUTPUT = ROOT / ".local-artifacts/issue-80-reactive-diagnostic-v1"
DATA = ROOT / "data/issue-80-reactive-diagnostic"
SCHEMA = "issue_80_reactive_diagnostic_v1"
IDENTITY = "issue-80-reactive-diagnostic-v1"
SEEDS = train77.SEEDS
STATES_PER_FAMILY = 12
PILOT_STATES_PER_FAMILY = 2
FAMILIES = ("type010103", "type010105")
MODEL_SYSTEMS = ("hybrid-fixed-h1", "continuous-fixed-h1", "continuous-fixed-h5")
PRIOR_SYSTEM = "no-model-ordinal-prior"
SYSTEMS = (*MODEL_SYSTEMS, PRIOR_SYSTEM)
SYSTEM_MODES = {
    "hybrid-fixed-h1": GameplayPlanningMode.HYBRID_FIXED,
    "continuous-fixed-h1": GameplayPlanningMode.CONTINUOUS_H1,
    "continuous-fixed-h5": GameplayPlanningMode.CONTINUOUS_H5,
}
SYSTEM_ARMS = {name: ("hybrid" if mode is GameplayPlanningMode.HYBRID_FIXED else "continuous",
                      fixed_mode_pair(mode))
               for name, mode in SYSTEM_MODES.items()}
PRIOR_ORDINAL = NO_MODEL_ORDINAL_PRIOR_ORDINAL
WALL_CAP_SECONDS = 24 * 3600
GPU_CAP_SECONDS = 2 * 3600
PREVALENCE_FLOOR = 0.10
PILOT_MINIMUM_VALID_EXECUTIONS = 24
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7201
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
    "scripts/run_short_horizon_reactive_diagnostic.py",
    "world_model/training/lineage_scaling.py",
)


def log(message):
    print(f"[issue-80-reactive] {message}", flush=True)


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


# ---------------------------------------------------------------- membership

def family_of_member(campaign_plan, member_identity):
    for member in campaign_plan["members"]:
        if member["identity"] == member_identity:
            return member["generator_family"]
    raise ValueError(f"unknown N1 member {member_identity}")


def select_states(campaign_plan, coverage):
    """12 states per family, evenly spaced in sorted admissible-branch order."""
    admissible = {family: [] for family in FAMILIES}
    for branch in campaign_plan["branches"]:
        status = coverage["branches"][branch["identity"]]["status"]
        if status != "admissible":
            continue
        family = family_of_member(campaign_plan, branch["source_member_identity"])
        admissible[family].append(branch["identity"])
    states = []
    for family in FAMILIES:
        pool = sorted(admissible[family])
        if len(pool) < STATES_PER_FAMILY:
            raise ValueError(f"family {family} has {len(pool)} admissible branches; "
                             f"the frozen 12-state membership is unsatisfiable")
        chosen = [pool[round(index * (len(pool) - 1) / (STATES_PER_FAMILY - 1))]
                  for index in range(STATES_PER_FAMILY)]
        if len(set(chosen)) != STATES_PER_FAMILY:
            raise ValueError(f"even spacing collapsed for family {family}")
        states.extend(chosen)
    return states


def state_payload(campaign_plan, coverage, state_identity):
    member_identity = state_identity.rsplit("-a", 1)[0]
    member = next(m for m in campaign_plan["members"] if m["identity"] == member_identity)
    inventory = []
    for branch in sorted((b for b in campaign_plan["branches"]
                          if b["source_member_identity"] == member_identity),
                         key=lambda b: b["candidate_ordinal"]):
        status = coverage["branches"][branch["identity"]]["status"]
        if status != "admissible":
            continue
        inventory.append({"ordinal": branch["candidate_ordinal"],
                          "branch_identity": branch["identity"],
                          "action": deepcopy(branch["action"])})
    ordinals = [item["ordinal"] for item in inventory]
    return {
        "identity": state_identity,
        "generator_family": member["generator_family"],
        "source_member": member_identity,
        "study_role": member["study_role"],
        "engine_seed": member["engine_seed"],
        "inventory": inventory,
        "prior_ordinal": PRIOR_ORDINAL,
        "prior_available": ordinal_prior_candidate(ordinals) is not None,
        "typed_dropped_branches": sorted(
            b["identity"] for b in campaign_plan["branches"]
            if b["source_member_identity"] == member_identity
            and coverage["branches"][b["identity"]]["status"] != "admissible"),
        "execution_member": {key: member[key] for key in (
            "identity", "novelty_level", "generator_family", "generation_seed",
            "engine_seed", "exposure_role", "base_cluster", "generated_slots",
            "template", "scenario", "xml")},
    }


# ---------------------------------------------------------------- plan freeze

def definitions():
    return {
        "states_per_family": STATES_PER_FAMILY,
        "families": {"type010103": "rolling", "type010105": "sliding"},
        "membership_rule": (
            "N1 (issue-77-n1-v1) coverage-admissible branch identities, sorted and "
            "evenly spaced as pool[round(i*(len(pool)-1)/11)] for i in 0..11 per family; "
            "state identity = branch identity; initial state = source-member decision "
            f"state at native fixed step {DECISION_FIXED_STEP}"),
        "seeds": list(SEEDS),
        "systems": {name: ({"kind": "model", "arm": arm,
                            "pair": {"delta": pair.delta, "abstraction": str(pair.abstraction)}}
                           if arm else {"kind": "no-model-ordinal-prior",
                                        "prior_ordinal": PRIOR_ORDINAL})
                    for name, (arm, pair) in {**SYSTEM_ARMS, PRIOR_SYSTEM: (None, None)}.items()},
        "decision": (
            "reactive: parse the sealed pre-decision frame with the frozen issue-70 "
            "adapter as a single-frame temporal context (identical to the #77 per-state "
            "ranking context), roll every admissible candidate to the system's fixed "
            "pair, choose argmin predicted count cost with ties broken by lower "
            "ordinal; candidates with nonfinite predictions are excluded; a typed "
            "decision failure is terminal and executes no shot"),
        "action_encoding": "drag/480, true 1000 ms hold, tap/1000, bias 1 (equals training)",
        "estimand": (
            "normalized ranking regret of the executed shot's realized end-of-window "
            f"count cost (observed offset {END_OFFSET}, terminal-absorbed; one frame = "
            f"{NATIVE_STRIDE} native steps; right-censored, NOT a settled cost) against "
            "the state's frozen candidate outcome table recorded by the issue-77 N1 "
            "campaign; executed costs may fall outside [0,1] under execution variance "
            "and are recorded unclipped"),
        "first_shot_success": (
            "the executed shot segment's engine events record at least one pig death "
            "or destruction (interaction_coverage pig_removed); typed failures carry "
            "no success value"),
        "shots_to_success": (
            "single-shot protocol: 1 when the first shot succeeds, otherwise null; "
            "no multi-shot competence claim"),
        "pilot_gate": {
            "cells": PILOT_STATES_PER_FAMILY * 2 * len(SYSTEMS) * len(SEEDS),
            "states": PILOT_STATES_PER_FAMILY * 2,
            "state_rule": ("the two lowest-identity selected states of each family "
                           "with distinct source members (four distinct physical "
                           "initial states)"),
            "prevalence_floor": PREVALENCE_FLOOR,
            "minimum_valid_executions": PILOT_MINIMUM_VALID_EXECUTIONS,
            "metric": ("pooled first-shot success prevalence over pilot cells with a "
                       "valid executed segment; typed failures excluded from numerator "
                       "and denominator and reported"),
        },
        "caps": {"wall_cap_seconds": WALL_CAP_SECONDS, "gpu_cap_seconds": GPU_CAP_SECONDS,
                 "wall_scope": "cumulative elapsed wall of the pilot+run execution phases",
                 "gpu_scope": ("cumulative synchronized decision wall (perception plus "
                               "ranking) on cuda across all executions")},
        "compute_accounting": (
            "per state wall time AND active-parameter/step counts AND candidate counts "
            "for every arm; candidate count alone is never compute matching; inference "
            "is NOT equalized across arms and is recorded as such "
            "(#74 training_compute_matched=false)"),
        "uncertainty": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "state",
                        "note": "seed differences averaged before resampling; descriptive "
                                "only; this diagnostic cannot support a confirmatory claim"},
        "comparator_disclosure": (
            "continuous_h5 is the #74 selected comparator with selection_optimism=true "
            "(selection on development evidence); disclosed in every contrast"),
        "wall_context": {
            "issue_74_fixed_mode_seconds_per_state": {"continuous_h1": 1.79,
                                                      "continuous_h5": 0.37,
                                                      "continuous_h15": 0.13},
            "hybrid_adaptive_1_01_seconds_is_context_only": (
                "this ticket contains no adaptive system; at h=1 the continuous "
                "baseline is the MORE expensive arm (4.8x continuous_h5); no "
                "adaptive-penalty narrative may be attached to these results")},
        "disposition_rule": {
            "supported": ("pooled mean paired difference > 0 AND descriptive 95% "
                          "interval entirely above 0 (pre-declared practical margin: "
                          "the interval-exclusion rule itself)"),
            "not_supported_by_this_experiment": "anything else after complete execution",
            "readiness_or_precision_insufficient": (
                "pilot gate failed, or a cap stop, or incomplete execution inventory"),
        },
        "claim_boundary": (
            "bounded non-final diagnostic on development lineages; no multi-shot, "
            "adaptation, zero-shot, or complete-gameplay claim; cannot reopen "
            "#64/#65/#72/#75/#15; descriptive intervals only"),
    }


def checkpoint_bindings():
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
            entry[arm] = {"path": str(path), "identity": file_identity(path)}
        bindings[str(seed)] = entry
    return bindings


def file_identity(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def make_plan():
    coverage = read(CAMPAIGN / "coverage.json")
    if coverage["schema"] != "issue_77_n1_coverage_v1" or coverage["coverage_complete"] is not True:
        raise ValueError("requires completed issue-77 N1 campaign coverage")
    campaign_plan = read(CAMPAIGN / "plan.json")
    identities = select_states(campaign_plan, coverage)
    states = [state_payload(campaign_plan, coverage, identity) for identity in identities]
    # The four pilot states come from four DISTINCT source members (two per
    # family) so the 48-execution prevalence pilot samples four physical
    # initial states, not two branches of one member.
    pilot_states = []
    for family in FAMILIES:
        picked_members = set()
        for identity in identities_and_families(campaign_plan, identities, family=family):
            member = identity.rsplit("-a", 1)[0]
            if member in picked_members:
                continue
            pilot_states.append(identity)
            picked_members.add(member)
            if len(picked_members) >= PILOT_STATES_PER_FAMILY:
                break
    pilot_states.sort()
    player = CAMPAIGN / "player"
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")
    plan = {
        "schema": SCHEMA, "identity": IDENTITY,
        "frozen_before_outcome": True,
        "definitions": definitions(),
        "n1_campaign": {"root": str(CAMPAIGN), "identity": campaign_plan["identity"],
                        "coverage_status_counts": coverage["status_counts"]},
        "dynamics": {"root": str(DYNAMICS), "identity": "issue-77-n1-dynamics-v1",
                     "checkpoints": checkpoint_bindings()},
        "player_root": str(player),
        "states": states,
        "pilot_states": pilot_states,
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
    return plan


def identities_and_families(campaign_plan, identities, family=None):
    return [identity for identity in sorted(identities)
            if family is None or family_of_member(campaign_plan, identity.rsplit("-a", 1)[0]) == family]


def load_plan():
    plan = read(OUTPUT / "plan.json")
    if plan.get("identity") != IDENTITY:
        raise ValueError("retained issue-80 plan identity differs")
    return plan


def check_plan_bound(plan):
    """Validate the frozen plan against the live frozen inputs it binds."""
    current = make_plan()
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-80 plan source/membership changed; preserve and version")
    return plan


# ---------------------------------------------------------------- candidate outcomes

def outcome_path(member_identity):
    return OUTPUT / "candidate-outcomes" / f"{member_identity}.json"


def parse_offset_carrier(adapter, observation_root, needed_offsets=(END_OFFSET,)):
    """#77 target semantics: offset 600 carrier with terminal absorption."""
    with torch.no_grad():
        return _parse_offset_carrier(adapter, observation_root, needed_offsets)


def _parse_offset_carrier(adapter, observation_root, needed_offsets):
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


def member_outcomes(args, plan, member_identity, adapter):
    """Frozen per-candidate realized end-of-window count costs from the campaign."""
    path = outcome_path(member_identity)
    if path.exists():
        return read(path)
    state = next(s for s in plan["states"] if s["source_member"] == member_identity)
    vocabulary = read(DYNAMICS / "plan.json")["contract"]["vocabulary"]
    objective = TaskObjective.from_vocabulary(vocabulary)
    candidates = []
    for item in state["inventory"]:
        branch = item["branch_identity"]
        result = read(CAMPAIGN / "results" / f"{branch}.json")
        segment = result["segments"][0]
        if (result["member_identity"] != branch or result["complete"] is not True
                or segment["summary"]["first_fixed_step"] != DECISION_FIXED_STEP):
            raise ValueError(f"campaign candidate source differs for {branch}")
        executed = segment["action"]
        declared = item["action"]
        if (list(executed["drag_release"]) != [declared["drag_x"], declared["drag_y"]]
                or executed["release_time"] != declared["release_time_ms"]
                or executed["tap_time"] != declared["tap_time_ms"]):
            raise ValueError(f"campaign executed action differs from inventory for {branch}")
        carriers, calls = parse_offset_carrier(adapter, CAMPAIGN / "attempts" / branch / "shot-1" / "observation-trace")
        cost = float(objective(carriers[str(END_OFFSET)]))
        candidates.append({"branch_identity": branch, "ordinal": item["ordinal"],
                           "action": deepcopy(declared),
                           "realized_count_cost": cost,
                           "censored": segment["summary"]["censored"],
                           "parser_calls": calls})
    costs = [c["realized_count_cost"] for c in candidates]
    record = {"plan_identity": plan["identity"], "member_identity": member_identity,
              "schema": "issue_80_candidate_outcomes_v1",
              "candidates": candidates,
              "lowest_cost": min(costs), "highest_cost": max(costs),
              "informative": max(costs) > min(costs),
              "all_tied": max(costs) - min(costs) <= 1e-8}
    check_member_outcomes(record, state)
    write(path, record)
    return record


def normalized_regret(cost, outcomes):
    if not outcomes["informative"]:
        return 0.0 if outcomes["all_tied"] else None
    return (cost - outcomes["lowest_cost"]) / (outcomes["highest_cost"] - outcomes["lowest_cost"])


def check_member_outcomes(record, state):
    if (record["schema"] != "issue_80_candidate_outcomes_v1"
            or record["member_identity"] != state["source_member"]
            or len(record["candidates"]) != len(state["inventory"])):
        raise ValueError("candidate outcome table binding differs")
    for stored, item in zip(record["candidates"], state["inventory"], strict=True):
        if (stored["branch_identity"] != item["branch_identity"]
                or stored["ordinal"] != item["ordinal"] or stored["action"] != item["action"]
                or not math.isfinite(stored["realized_count_cost"])):
            raise ValueError("candidate outcome row differs from frozen inventory")
    costs = [c["realized_count_cost"] for c in record["candidates"]]
    if record["lowest_cost"] != min(costs) or record["highest_cost"] != max(costs):
        raise ValueError("candidate outcome range differs")


# ---------------------------------------------------------------- decision + execution

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
            ordinal = ordinal_prior_candidate([item["ordinal"] for item in inventory])
            if ordinal is None:
                return {"chosen": None, "failure": "prior_candidate_absent",
                        "ranking": [], "transition_calls": 0, "linear_macs": 0,
                        "candidate_count": len(inventory)}
            chosen = next(item for item in inventory if item["ordinal"] == ordinal)
            return {"chosen": chosen, "failure": None,
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
                    "ranking": rows, "transition_calls": len(inventory) * self.pair.delta,
                    "linear_macs": macs * len(inventory) * self.pair.delta,
                    "candidate_count": len(inventory), "wall_seconds": wall}
        best = min(finite, key=lambda row: (row["predicted_cost"], row["ordinal"]))
        chosen = next(item for item in inventory if item["ordinal"] == best["ordinal"])
        return {"chosen": chosen, "failure": None, "ranking": rows,
                "transition_calls": len(inventory) * self.pair.delta,
                "linear_macs": macs * len(inventory) * self.pair.delta,
                "candidate_count": len(inventory), "wall_seconds": wall,
                "selected_predicted_cost": best["predicted_cost"]}


def load_frozen_predictor(plan, seed, arm, device):
    binding_entry = plan["dynamics"]["checkpoints"][str(seed)][arm]
    path = Path(binding_entry["path"])
    if file_identity(path) != binding_entry["identity"]:
        raise ValueError(f"frozen predictor bytes changed: {path}")
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


def segment_coverage(native_root):
    trace = NativeSegmentTrace(native_root)
    pig_removed = pig_contact = block_contact = False
    launches = 0
    for chunk in trace.trace.chunks():
        for event in chunk["events"]:
            name = event["event_type"]
            participants = " ".join(event.get("participants", []))
            if name == "bird_launched":
                launches += 1
            if name in ("entity_death", "entity_destroyed") and "pig" in participants:
                pig_removed = True
            if name == "collision" and "pig" in participants:
                pig_contact = True
            if name == "collision" and "block" in participants:
                block_contact = True
    return {"pig_removed": pig_removed, "pig_contact": pig_contact,
            "block_contact": block_contact, "bird_launches": launches}


def execute_cell(payload):
    """One isolated, real-rendered, reactive first-shot execution (spawn worker)."""
    torch.set_num_threads(2)
    output = Path(payload["output"])
    plan = read(output / "plan.json")
    cell = payload["cell"]
    identity = cell["identity"]
    state = payload["state"]
    record_path = output / "records" / f"{identity}.json"
    attempt = output / "attempts" / identity
    attempt.mkdir(parents=True)
    device = payload["device"]
    limits = payload["limits"]
    started = time.monotonic()
    record = {"schema": SCHEMA, "plan_identity": plan["identity"], "cell": cell,
              "state_identity": state["identity"], "system": cell["system"],
              "seed": cell["seed"], "failure": None, "failure_kind": None,
              "decision": None, "execution": None, "outcome": None,
              "fresh_scenario_lineage": False, "issue_64_authorized": False}
    bridge = physics = engine = display_process = None
    variables = ("DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
                 "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
                 "NOVPHY_ENVIRONMENT_SEED", "NOVPHY_NATIVE_DECISION_STEP")
    environment = {key: os.environ.get(key) for key in variables}
    try:
        member = state["execution_member"]
        inventory = state["inventory"]
        # Frozen setup (adapter, predictor, objective) happens before the engine
        # starts so the measured decision wall is perception plus ranking only.
        adapter = old.repair.load_repaired_adapter(old.repair.ROOT, device)
        vocabulary = read(DYNAMICS / "plan.json")["contract"]["vocabulary"]
        objective = TaskObjective.from_vocabulary(vocabulary)
        model = None
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
        engine = start_engine(game, False, agent_port=agent_port,
                              game_port=game_port, physics_port=physics_port)
        write(attempt / "runtime.json", record)
        bridge = connect_with_retry("127.0.0.1", agent_port, timeout=180, deadline_seconds=60)
        physics = ScienceBirdsBridge("127.0.0.1", physics_port, timeout=30)
        bridge.configure(760001, PlayingMode.TRAINING)
        bridge.set_speed(1)
        prepare_for_play(bridge, timeout=60, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("reactive episode did not load its single assigned level")
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
                raise RuntimeError("reactive choice is made by the selector, not the policy")

            def evidence(self):
                return {"observed_frames": self.frames,
                        "future_frames_used": False, "engine_outcomes_used": False}

        policy = Policy()
        shot_identity = identity + ":decision"
        expose_history(rows, attempt / "decision-1", scenario, shot_identity,
                       member["exposure_role"], policy)
        inventory = state["inventory"]
        anchor = next((item["action"] for item in inventory if item["ordinal"] == PRIOR_ORDINAL),
                      inventory[0]["action"])
        prepared = prepare_action(bridge, anchor)
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

        # Reactive decision: parse the sealed decision frame exactly as #77 built
        # per-state ranking contexts (single-frame temporal context).
        perception_started = time.monotonic()
        manifest = read(attempt / "decision-1" / MANIFEST_NAME)
        frames = manifest["frame_records"]
        if frames[-1]["fixed_step"] != DECISION_FIXED_STEP:
            raise ValueError("decision trace lacks the sealed decision frame")
        ref = frames[-1]["agent_observation"]
        observation = AgentObservation(ref["identity"], frames[-1]["fixed_step"],
                                       frames[-1]["fixed_time_seconds"],
                                       (attempt / "decision-1" / ref["relative_path"]).read_bytes(),
                                       "agent")
        with torch.no_grad():
            parsed = adapter.parse_batch((observation,))[0]
            carrier = adapter.build_from_parsed(
                TemporalObservationContext(None, observation), parsed, None).tensor
            grid.synchronize(device)
            perception_seconds = time.monotonic() - perception_started
            (attempt / "decision-frame.png").write_bytes(
                (attempt / "decision-1" / ref["relative_path"]).read_bytes())

            selector = ReactiveSelector(cell["system"], cell["seed"], model, objective, device)
            decision = selector.choose(carrier, inventory)
        decision["perception_seconds"] = perception_seconds
        decision["gpu_seconds"] = (perception_seconds + decision.get("wall_seconds", 0.0)
                                   if device.startswith("cuda") else 0.0)
        decision["actuator_readiness"] = prepared.evidence
        decision["inference_not_equalized"] = True
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
            coverage = segment_coverage(segment["native_root"])
            if coverage["bird_launches"] != 1:
                raise ValueError("executed shot did not contain exactly one native launch")
            summary = segment["summary"]
            carriers, calls = parse_offset_carrier(adapter, attempt / "shot-1" / "observation-trace")
            realized_cost = float(objective(carriers[str(END_OFFSET)]))
            terminal = terminal_evidence(segment)
            record["execution"] = {
                "segment_identity": segment["identity"],
                "native_root": segment["native_root"],
                "observation_manifest": segment["observation_manifest"],
                "first_fixed_step": summary["first_fixed_step"],
                "last_fixed_step": summary["last_fixed_step"],
                "frame_count": summary["frame_count"],
                "censored": summary["censored"],
                "terminal_reason": None if terminal is None else terminal["reason"],
                "interaction_coverage": {k: v for k, v in coverage.items() if k != "bird_launches"},
                "realized_end_of_window_count_cost": realized_cost,
                "end_of_window_parser_calls": calls,
                "engine_wall_seconds": engine_seconds,
            }
            outcomes = payload["member_outcomes"]
            frozen_cost = next(c["realized_count_cost"] for c in outcomes["candidates"]
                               if c["ordinal"] == chosen["ordinal"])
            record["outcome"] = {
                "first_shot_success": coverage["pig_removed"],
                "shots_to_success": 1 if coverage["pig_removed"] else None,
                "realized_count_cost": realized_cost,
                "regret": normalized_regret(realized_cost, outcomes),
                "informative": outcomes["informative"],
                "all_tied": outcomes["all_tied"],
                "chosen_frozen_cost": frozen_cost,
                "chosen_frozen_regret": normalized_regret(frozen_cost, outcomes),
            }
    except Exception as error:  # typed terminal failure; never retried silently
        record["failure"] = f"{type(error).__name__}: {error}"
        if record["failure_kind"] is None:
            record["failure_kind"] = "execution_failure"
    finally:
        actions = [("connection.disconnect", connection.disconnect)
                   for connection in (bridge, physics) if connection is not None]
        actions.append(("stop_started_engine", lambda: stop_started_engine(engine)))
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
    print(f"issue-80 cell {identity} complete={record['failure'] is None} "
          f"failure={record['failure']}", flush=True)
    return record


# ---------------------------------------------------------------- ledger + supervisor

def ledger_path():
    return OUTPUT / "ledger.json"


def fresh_ledger(plan):
    return {"schema": "issue_80_ledger_v1", "identity": IDENTITY,
            "plan_identity": plan["identity"], "status": "running",
            "wall_seconds_elapsed": 0.0, "gpu_seconds_elapsed": 0.0,
            "cells": {}, "phase": None}


def load_ledger(plan, phase=None):
    path = ledger_path()
    if not path.is_file():
        return fresh_ledger(plan)
    ledger = read(path)
    if (ledger.get("schema") != "issue_80_ledger_v1"
            or ledger.get("plan_identity") != plan["identity"]):
        raise ValueError("retained issue-80 ledger identity differs from the frozen plan")
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
        raise ValueError(f"issue-80 execution ledger is terminal ({ledger['status']}); "
                         "audit it; never relaunch a terminal campaign")
    return ledger


def terminal_cells(plan):
    """Cells with a retained record + receipt are terminal, whatever the outcome.

    Partial persistence from a supervisor crash is completed without touching
    the recorded outcome: a record without a receipt gains the missing receipt
    (the record is the outcome; the receipt is supervisor bookkeeping), and a
    receipt without a record gains a typed failure record (the worker was
    terminated before any outcome was persisted). An attempt tree with neither
    file never produced an outcome and is removed so the cell re-dispatches —
    crash recovery, not outcome-conditioned replacement.
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
                    write(record, {
                        "schema": SCHEMA, "plan_identity": plan["identity"],
                        "cell": cell, "state_identity": cell["state"],
                        "system": cell["system"], "seed": cell["seed"],
                        "failure": "worker_terminated_before_record",
                        "failure_kind": "execution_failure", "decision": None,
                        "execution": None, "outcome": None,
                        "fresh_scenario_lineage": False, "issue_64_authorized": False,
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
    start_isolated_worker = _worker_tools()
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
            while (stop_reason is None and index < len(pending)
                   and len(live) < WORKERS):
                identity = pending[index]
                cell = split_cell_identity(plan, identity)
                state = next(s for s in plan["states"] if s["identity"] == cell["state"])
                payload = {
                    "output": str(OUTPUT), "device": args.device,
                    "limits": plan["execution"], "cell": cell, "state": state,
                    "member_outcomes": read(outcome_path(state["source_member"])),
                }
                (OUTPUT / "markers").mkdir(parents=True, exist_ok=True)
                write(OUTPUT / "markers" / f"{identity}.json",
                      {"identity": identity, "phase": phase,
                       "dispatched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                process = start_isolated_worker(context, _execute_cell, (payload,))
                live.append({"identity": identity, "process": process,
                             "started": time.monotonic(), "peak_rss": 0.0, "stop": None})
                index += 1
            if not live:
                break
            time.sleep(1)
            process_rss, terminate_worker = _supervisor_tools()
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
                    write(record_file, {
                        "schema": SCHEMA, "plan_identity": plan["identity"],
                        "cell": split_cell_identity(plan, worker["identity"]),
                        "failure": failure, "failure_kind": "execution_failure",
                        "decision": None, "execution": None, "outcome": None,
                        "fresh_scenario_lineage": False, "issue_64_authorized": False})
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
                write(record_file, {
                    "schema": SCHEMA, "plan_identity": plan["identity"], "cell": cell,
                    "state_identity": cell["state"], "system": cell["system"],
                    "seed": cell["seed"], "failure": f"not_executed: {stop_reason}",
                    "failure_kind": "not_executed", "decision": None,
                    "execution": None, "outcome": None,
                    "fresh_scenario_lineage": False, "issue_64_authorized": False,
                    "wall_seconds": 0.0})
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


def _worker_tools():
    from scripts.process_lifecycle import start_isolated_worker
    return start_isolated_worker


def _supervisor_tools():
    from scripts.run_issue_76_compatibility import process_rss, terminate_worker
    return process_rss, terminate_worker


# ---------------------------------------------------------------- pilot gate

def gate_cells(plan):
    return [cell_identity(system, seed, state["identity"])
            for system in SYSTEMS for seed in SEEDS
            for state in plan["states"] if state["identity"] in plan["pilot_states"]]


def evaluate_gate(plan):
    cells = gate_cells(plan)
    valid = success = 0
    typed = []
    for identity in cells:
        record_file = OUTPUT / "records" / f"{identity}.json"
        receipt_file = OUTPUT / "receipts" / f"{identity}.json"
        if not record_file.is_file() or not receipt_file.is_file():
            raise ValueError(f"gate requires a terminal record for {identity}")
        record = read(record_file)
        outcome = record.get("outcome")
        if record.get("failure") is not None or outcome is None:
            typed.append({"identity": identity, "failure": record.get("failure")})
            continue
        valid += 1
        success += int(bool(outcome["first_shot_success"]))
    prevalence = success / valid if valid else 0.0
    passed = (prevalence >= PREVALENCE_FLOOR
              and valid >= PILOT_MINIMUM_VALID_EXECUTIONS)
    return {"schema": "issue_80_pilot_gate_v1", "plan_identity": plan["identity"],
            "cells": len(cells), "valid_executions": valid,
            "typed_failures": typed, "successes": success,
            "prevalence": prevalence, "floor": PREVALENCE_FLOOR,
            "minimum_valid_executions": PILOT_MINIMUM_VALID_EXECUTIONS,
            "passed": passed,
            "disposition_if_failed": "readiness_or_precision_insufficient"}


# ---------------------------------------------------------------- publication

def publication(plan):
    ledger = read(ledger_path())
    complete = (ledger.get("status") == "complete" and ledger.get("phase") == "run")
    gate_verdict = read(OUTPUT / "pilot-gate.json") if (OUTPUT / "pilot-gate.json").is_file() else None
    gate_failed = gate_verdict is not None and gate_verdict["passed"] is False
    common = {
        "schema": "issue_80_reactive_report_v1", "identity": "issue-80-reactive-report-v1",
        "plan_identity": plan["identity"],
        "diagnostics_complete": complete,
        "execution_ledger_status": ledger["status"],
        "wall_seconds_elapsed": ledger["wall_seconds_elapsed"],
        "gpu_seconds_elapsed": ledger["gpu_seconds_elapsed"],
        "caps": {"wall_cap_seconds": WALL_CAP_SECONDS, "gpu_cap_seconds": GPU_CAP_SECONDS},
        "pilot_gate": gate_verdict,
        "definitions": plan["definitions"],
        "claim_boundary": plan["definitions"]["claim_boundary"],
        "comparator_disclosure": plan["definitions"]["comparator_disclosure"],
        "limitations": [
            "development/exposed N1 lineages only; zero fresh captures; not the sealed #64/#65 benchmark",
            "regret references the t=600 end-of-window replay cost: right-censored, not a settled cost",
            "states sharing a source member share one physical initial state and candidate table; "
            "the paired bootstrap over 24 state identities is descriptive only",
            "inference is NOT equalized across arms; per-arm decision compute is reported",
            "no adaptive system is present; #74 hybrid_adaptive wall context carries no penalty narrative",
        ],
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }
    if gate_failed:
        common["ticket_disposition"] = "readiness_or_precision_insufficient"
        common["stopExplanation"] = (
            "the mandatory nonzero-prevalence pilot gate failed before the full run")
        return common
    records = {}
    missing = []
    for system in SYSTEMS:
        for seed in SEEDS:
            for state in plan["states"]:
                identity = cell_identity(system, seed, state["identity"])
                record_file = OUTPUT / "records" / f"{identity}.json"
                if record_file.is_file():
                    records[identity] = read(record_file)
                else:
                    missing.append(identity)
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
    if not complete:
        common["ticket_disposition"] = "readiness_or_precision_insufficient"
        return common
    per_seed, per_state = {}, {}
    for seed in SEEDS:
        rows = {name: {} for name in SYSTEMS}
        for state in plan["states"]:
            for system in SYSTEMS:
                record = records[cell_identity(system, seed, state["identity"])]
                outcome = record.get("outcome")
                decision = record.get("decision") or {}
                execution = record.get("execution") or {}
                rows[system][state["identity"]] = {
                    "failure": record.get("failure"),
                    "failure_kind": record.get("failure_kind"),
                    "success": None if outcome is None else bool(outcome["first_shot_success"]),
                    "regret": None if outcome is None else outcome["regret"],
                    "realized_count_cost": None if outcome is None else outcome["realized_count_cost"],
                    "candidate_count": decision.get("candidate_count"),
                    "transition_calls": decision.get("transition_calls"),
                    "linear_macs": decision.get("linear_macs"),
                    "decision_wall_seconds": (
                        (decision.get("perception_seconds") or 0.0)
                        + (decision.get("wall_seconds") or 0.0)),
                    "engine_wall_seconds": execution.get("engine_wall_seconds"),
                    "wall_seconds": record.get("wall_seconds"),
                }
        per_state[str(seed)] = rows
        per_seed[str(seed)] = {name: aggregate_rows(list(rows[name].values()))
                               for name in SYSTEMS}
    pooled = {name: aggregate_rows(
        [row for seed in SEEDS for row in per_state[str(seed)][name].values()])
        for name in SYSTEMS}
    contrasts = summarize_contrasts(plan, per_state)
    hybrid_regret = next(c for c in contrasts if c["metric"] == "regret"
                         and c["reference"] == "continuous-fixed-h5")
    descriptive = hybrid_regret.get("descriptive")
    supported = (descriptive is not None and descriptive["mean"] > 0
                 and descriptive["descriptive_95_percent_interval"][0] > 0)
    disposition = "supported" if supported else "not_supported_by_this_experiment"
    return {**common,
            "per_seed": per_seed, "pooled": pooled,
            "contrasts": contrasts,
            "ticket_disposition": disposition}


def aggregate_rows(rows):
    valid = [row for row in rows if row["failure"] is None]
    regrets = [row["regret"] for row in valid if row["regret"] is not None]
    successes = [row["success"] for row in valid if row["success"] is not None]
    costs = [row["realized_count_cost"] for row in valid if row["realized_count_cost"] is not None]
    return {
        "states": len(rows), "executed": len(valid),
        "decision_failures": sum(1 for row in rows if row["failure_kind"] == "decision_failure"),
        "execution_failures": sum(1 for row in rows if row["failure_kind"] == "execution_failure"),
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "success_fraction": float(np.mean(successes)) if successes else None,
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


def summarize_contrasts(plan, per_state):
    result = []
    for reference in ("continuous-fixed-h1", "continuous-fixed-h5", "no-model-ordinal-prior"):
        for metric, key in (("regret", "regret"), ("success", "success")):
            paired = []
            for state in sorted(row_identity for row_identity in per_state[str(SEEDS[0])]["hybrid-fixed-h1"]):
                differences = []
                for seed in SEEDS:
                    tested = per_state[str(seed)]["hybrid-fixed-h1"][state]
                    reference_row = per_state[str(seed)][reference][state]
                    if tested[metric] is None or reference_row[metric] is None:
                        differences = None
                        break
                    differences.append(reference_row[metric] - tested[metric])
                if differences:
                    paired.append(float(np.mean(differences)))
            result.append({
                "tested": "hybrid-fixed-h1", "reference": reference, "metric": metric,
                "positive_is_improvement": True,
                "descriptive": paired_interval(paired) if paired else None,
                "paired_states": len(paired),
                "scope": "descriptive paired bootstrap over state identities; seeds averaged first",
            })
    return result


def compact_report(result):
    return {key: value for key, value in result.items() if key not in ("per_state",)}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("seed", "system", "executed", "mean_regret", "success_fraction",
                     "mean_realized_count_cost", "decision_failures", "execution_failures",
                     "candidate_counts", "transition_calls", "linear_macs",
                     "mean_decision_wall_seconds", "mean_engine_wall_seconds"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, row in systems.items():
            writer.writerow((seed, name, row["executed"], row["mean_regret"],
                             row["success_fraction"], row["mean_realized_count_cost"],
                             row["decision_failures"], row["execution_failures"],
                             " ".join(str(c) for c in row["candidate_counts"]),
                             row["transition_calls"], row["linear_macs"],
                             row["mean_decision_wall_seconds"], row["mean_engine_wall_seconds"]))
    writer.writerow(())
    writer.writerow(("contrast_tested", "reference", "metric", "mean",
                     "descriptive_95_low", "descriptive_95_high", "paired_states"))
    for contrast in result.get("contrasts", []):
        interval = (contrast.get("descriptive") or {}).get("descriptive_95_percent_interval")
        writer.writerow((contrast["tested"], contrast["reference"], contrast["metric"],
                         None if not contrast.get("descriptive") else contrast["descriptive"]["mean"],
                         None if not interval else interval[0],
                         None if not interval else interval[1],
                         contrast["paired_states"]))
    return stream.getvalue()


def fmt(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def paired_interval(values):
    """Descriptive paired bootstrap; identical algorithm to the #72 tooling
    (10,000 draws, seed 7201, seed differences averaged before resampling),
    implemented here so the analysis lives inside the frozen source text."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "descriptive_95_percent_interval": np.quantile(draws, [.025, .975]).tolist()}


def findings_md(result):
    lines = ["# Issue-80 short-horizon reactive-control diagnostic - findings", ""]
    lines.append(f"Diagnostics complete: {result['diagnostics_complete']}. "
                 f"Ledger status: {result['execution_ledger_status']}.")
    lines.append("")
    lines.append(f"Claim boundary: {result['claim_boundary']}")
    lines.append("")
    lines.append(f"Comparator disclosure: {result['comparator_disclosure']}")
    lines.append("")
    lines.append("Declared estimand: " + result["definitions"]["estimand"])
    lines.append("")
    gate = result.get("pilot_gate")
    if gate is not None:
        lines.append("## Nonzero-prevalence pilot gate")
        lines.append("")
        lines.append(f"Prevalence {gate['prevalence']:.4f} ({gate['successes']}/{gate['valid_executions']} "
                     f"valid executions; {len(gate['typed_failures'])} typed failures); "
                     f"floor {gate['floor']:.2f}; passed={gate['passed']}.")
        lines.append("")
    if not result.get("per_seed"):
        lines.append("Execution incomplete; no outcome numbers reported.")
        lines.append("")
        gate = result.get("pilot_gate")
        if gate is not None:
            lines.append(f"Pilot gate: prevalence {gate['prevalence']:.4f} "
                         f"({gate['successes']}/{gate['valid_executions']} valid executions; "
                         f"{len(gate['typed_failures'])} typed failures); floor "
                         f"{gate['floor']:.2f}; passed={gate['passed']}.")
            lines.append("")
        lines.append(f"Ticket disposition: {result.get('ticket_disposition')} "
                     "(tokens: supported / not_supported_by_this_experiment / "
                     "readiness_or_precision_insufficient).")
        lines.append("")
        for item in result["limitations"]:
            lines.append(f"- {item}")
        return "\n".join(lines) + "\n"
    lines.append("## First-shot outcomes (per seed; descriptive)")
    lines.append("")
    lines.append("| System | Seed | Executed | Mean regret | Success | Mean cost | "
                 "Decision wall s | Engine wall s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for seed, systems in result["per_seed"].items():
        for name, row in systems.items():
            lines.append(
                f"| {name} | {seed} | {row['executed']} | "
                f"{fmt(row['mean_regret'])} | "
                f"{fmt(row['success_fraction'], 3)} | "
                f"{fmt(row['mean_realized_count_cost'], 2)} | "
                f"{fmt(row['mean_decision_wall_seconds'], 3)} | "
                f"{fmt(row['mean_engine_wall_seconds'], 1)} |")
    lines.append("")
    lines.append("## Pooled over seeds (descriptive)")
    lines.append("")
    lines.append("| System | Executed | Mean regret | Success | Transition calls | Linear MACs | Candidate counts |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, row in result["pooled"].items():
        lines.append(
            f"| {name} | {row['executed']} | "
            f"{fmt(row['mean_regret'])} | "
            f"{fmt(row['success_fraction'], 3)} | "
            f"{row['transition_calls']} | {row['linear_macs']} | "
            f"{' '.join(str(c) for c in row['candidate_counts'])} |")
    lines.append("")
    lines.append("## Paired contrasts, hybrid-fixed-h1 vs reference (positive favors hybrid; DESCRIPTIVE)")
    lines.append("")
    lines.append("| Reference | Metric | Mean difference | Descriptive 95% interval | Paired states |")
    lines.append("| --- | --- | --- | --- | --- |")
    for contrast in result["contrasts"]:
        descriptive = contrast.get("descriptive")
        if descriptive is None:
            lines.append(f"| {contrast['reference']} | {contrast['metric']} | n/a | n/a | {contrast['paired_states']} |")
            continue
        interval = descriptive["descriptive_95_percent_interval"]
        lines.append(f"| {contrast['reference']} | {contrast['metric']} | "
                     f"{descriptive['mean']:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] | "
                     f"{contrast['paired_states']} |")
    lines.append("")
    lines.append("## Wall-time context from #74 (CONTEXT ONLY)")
    lines.append("")
    context = result["definitions"]["wall_context"]
    fixed = context["issue_74_fixed_mode_seconds_per_state"]
    lines.append(f"continuous_h1 {fixed['continuous_h1']:.2f} s, continuous_h5 {fixed['continuous_h5']:.2f} s, "
                 f"continuous_h15 {fixed['continuous_h15']:.2f} s per state (issue-74 measured context). "
                 + context["hybrid_adaptive_1_01_seconds_is_context_only"])
    lines.append("")
    lines.append("## Disposition")
    lines.append("")
    lines.append(f"Ticket disposition: {result.get('ticket_disposition')} "
                 "(tokens: supported / not_supported_by_this_experiment / "
                 "readiness_or_precision_insufficient; the pre-declared practical-effect "
                 "margin is the interval-exclusion rule on the pooled hybrid-vs-"
                 "continuous-fixed-h5 regret contrast).")
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
    manifest = {"schema": "issue_80_review_gallery_v1", "identity": IDENTITY,
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
                segment = (record.get("execution") or {})
                frames = []
                if segment:
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
            "<title>Issue 80 reactive diagnostic review</title></head><body>"
            "<h1>Issue 80 bounded reactive-control diagnostic</h1>"
            "<p>One decision frame and one 50 fps WebM per scheduled cell, including "
            "typed failures. Agent observations only; canonical frames excluded.</p>"
            "<table border='1'><tr><th>cell</th><th>status</th><th>first-shot success</th>"
            "<th>failure</th><th>media</th></tr>" + "".join(sections) +
            "</table></body></html>")
    (DATA / "index.html").write_text(html, encoding="utf-8")
    return manifest


def check_gallery(plan, manifest):
    """Recompute the expected gallery inventory and verify the stored one."""
    if (manifest.get("schema") != "issue_80_review_gallery_v1"
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
            continue
        record = read(record_file)
        executed = record.get("failure") is None and record.get("execution")
        if executed and (not entry.get("decision_frame") or not entry.get("video")):
            raise ValueError(f"executed cell {entry['identity']} lacks gallery media")
        if entry.get("failure") != record.get("failure"):
            raise ValueError(f"gallery failure text differs for {entry['identity']}")
    index = DATA / "index.html"
    if not index.is_file():
        raise ValueError("review gallery index.html is missing")
    index_text = index.read_text(encoding="utf-8")
    for entry in manifest["cells"]:
        if entry["identity"] not in index_text:
            raise ValueError(f"review gallery index.html omits {entry['identity']}")


# ---------------------------------------------------------------- modes

def dry_run(args):
    plan = make_plan()
    states = plan["states"]
    log(f"no-write dry-run: states={len(states)} "
        f"per_family={ {family: sum(1 for s in states if s['generator_family'] == family) for family in FAMILIES} } "
        f"systems={len(SYSTEMS)} seeds={len(SEEDS)} cells={len(states) * len(SYSTEMS) * len(SEEDS)} "
        f"pilot_cells={len(gate_cells(plan))} workers={WORKERS} "
        f"wall_cap_s={WALL_CAP_SECONDS} gpu_cap_s={GPU_CAP_SECONDS} "
        f"prior_available_states={sum(1 for s in states if s['prior_available'])}/{len(states)}")
    return 0


def smoke(args):
    """Bounded real rendered smoke: 1 state x 4 systems x 1 seed, production-dir free."""
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    began = time.monotonic()
    state = plan["states"][0]
    outcomes = read(outcome_path(state["source_member"]))
    lock = gpu_lock()
    try:
        with tempfile.TemporaryDirectory(prefix="novphy-issue80-smoke-") as directory:
            smoke_output = Path(directory)
            smoke_plan = deepcopy(plan)
            smoke_plan["identity"] = IDENTITY + ":smoke"
            smoke_plan["states"] = [state]
            write(smoke_output / "plan.json", smoke_plan)
            write(smoke_output / "candidate-outcomes" / f"{state['source_member']}.json",
                  outcomes)
            records = []
            for system in SYSTEMS:
                payload = {"output": str(smoke_output), "device": args.device,
                           "limits": plan["execution"],
                           "cell": {"identity": cell_identity(system, SEEDS[0], state["identity"]),
                                    "system": system, "seed": SEEDS[0], "state": state["identity"]},
                           "state": state,
                           "member_outcomes": outcomes}
                record = execute_cell(payload)
                records.append({"cell": payload["cell"]["identity"],
                                "failure": record["failure"],
                                "failure_kind": record.get("failure_kind"),
                                "success": None if not record.get("outcome") else record["outcome"]["first_shot_success"],
                                "wall_seconds": record["wall_seconds"]})
                log(f"smoke {payload['cell']['identity']} failure={record['failure']}")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    result = {"schema": "issue_80_smoke_v1", "production_evidence": False,
              "records": records, "wall_seconds": time.monotonic() - began}
    write(OUTPUT / "smoke.json", result)
    log(f"real rendered smoke complete cells={len(records)} wall={result['wall_seconds']:.1f}s; "
        "no production execution")
    return 0


def prepare(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_file = OUTPUT / "plan.json"
    if plan_file.exists():
        try:
            check_plan_bound(read(plan_file))
        except ValueError as error:
            records_dir = OUTPUT / "records"
            execution_started = ((OUTPUT / "ledger.json").exists()
                                 or (records_dir.exists() and any(records_dir.iterdir())))
            if execution_started:
                raise ValueError(
                    "the frozen issue-80 plan changed after execution started; "
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
    lock = gpu_lock()
    try:
        adapter = old.repair.load_repaired_adapter(old.repair.ROOT, args.device)
        members = sorted({state["source_member"] for state in plan["states"]})
        for n, member in enumerate(members, 1):
            record = member_outcomes(args, plan, member, adapter)
            log(f"candidate-outcomes member={n}/{len(members)} "
                f"candidates={len(record['candidates'])} "
                f"informative={record['informative']}")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    log("issue-80 protocol frozen and candidate outcome tables validated; no gameplay executed")
    return 0


def pilot(args):
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
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


def require_prepared(plan):
    members = {state["source_member"] for state in plan["states"]}
    missing = [member for member in members if not outcome_path(member).exists()]
    if missing:
        raise ValueError(f"--prepare must freeze candidate outcome tables first; missing {missing[:3]}")


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
        f"gallery_cells={len(manifest['cells'])}")
    return 0


def validate(args):
    plan = load_plan()
    check_plan_bound(plan)
    result = publication(plan)
    problems = []
    if compact_report(result) != read(OUTPUT / "summary.json"):
        problems.append("summary.json")
    if comparisons_csv(result).encode("utf-8") != (OUTPUT / "comparisons.csv").read_bytes():
        problems.append("comparisons.csv")
    if findings_md(result) != (OUTPUT / "findings.md").read_text():
        problems.append("findings.md")
    gate = read(OUTPUT / "pilot-gate.json") if (OUTPUT / "pilot-gate.json").is_file() else None
    if gate is not None and gate != evaluate_gate(plan):
        problems.append("pilot-gate.json")
    if (DATA / "manifest.json").is_file():
        try:
            check_gallery(plan, read(DATA / "manifest.json"))
        except ValueError as error:
            problems.append(f"gallery: {error}")
    if problems:
        raise ValueError(f"published issue-80 artifacts differ from bound source evidence: {problems}")
    log("exact saved-evidence validation passed")
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
