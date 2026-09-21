"""Issue-82 ADD-EXP: oracle-ceiling diagnostic of the closed-loop first-shot
zero floor (post-#80 probe).

Analysis-scale ticket on existing #77/#80 artifacts only: no new gameplay, no
new captures, no fresh renders, zero sealed/final lineage access. #80's
mandatory prevalence gate failed at 0.0000 (0/45 first-shot successes, floor
0.10), which leaves a 2-way ambiguity this ticket resolves descriptively:

(a) state-difficulty floor - on the frozen 24-state membership no admissible
    candidate of the frozen 13-candidate inventory produces a first-shot
    success at replay level, so no planner could clear the floor; or
(b) ranking failure - some candidates do succeed at replay level but all four
    frozen systems rank them after the first shot.

Phase 0 (blocking, before any state-level outcome): bind a replay-level
success predicate computable for EVERY candidate in each state's frozen
admissible set from the #80-published realized t=600 end-of-window replay
cost tables (the frozen estimand basis; right-censored, NOT a settled cost)
against a frozen numeric threshold rule, and record its declared divergence
from the closed-loop first-shot outcome (proxy, labelled). Binding is
two-stage and pre-declared: stage 1 anchors the least-inclusive candidate
threshold on per-cell replay-side engine pig evidence from the issue-77 N1
campaign; stage 2 gates defensibility (the K-predicate must agree with its
own anchoring evidence on at least half the frozen cells). If no defensible
binding exists the ticket publishes readiness_or_precision_insufficient
naming the exact blocker - that is the deliverable, never forced.

Modes (mutually exclusive): --dry-run (no-write), --smoke-test (bounded real
scoring of one member, production tables untouched), --prepare (freeze the
Phase-0 binding + plan), --run-evaluation (GPU gap scoring under the shared
flock), --publish (summary.json + comparisons.csv + findings.md, the #77
layout), --validate (recompute every published table, exit 0).

Compute: <= 0.1 GPU-hours for the optional gap scoring (flock
/tmp/novphy-addexp-gpu.lock), <= 0.5 GiB derived artifacts; exceeding either
stops with readiness_or_precision_insufficient. Validation command:
python -u -m scripts.run_reactive_zero_floor_oracle --validate
"""
from __future__ import annotations

import argparse
import csv
import fcntl
from hashlib import sha256
import io
import json
import math
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_72_matched_grid as grid
from scripts import run_issue_77_n1_train as train77
from scripts.native_segment_trace import NativeSegmentTrace
from scripts.observation_trace import MANIFEST_NAME
from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
from world_model.planning.task_objective import TaskObjective
from world_model.training import reactive_zero_floor_oracle as oracle
from world_model.training.lineage_scaling import (
    NO_MODEL_ORDINAL_PRIOR_ORDINAL,
    GameplayPlanningMode,
    fixed_mode_pair,
    ordinal_prior_candidate,
)
from world_model.training.matched_dynamics import (
    CNNHybridPredictor,
    ContinuousDynamics,
    pairs_for,
    work as pair_work,
)

ROOT = train77.ROOT
CAMPAIGN = train77.CAMPAIGN
DYNAMICS = train77.OUTPUT
EIGHTY = ROOT / ".local-artifacts/issue-80-reactive-diagnostic-v1"
EIGHTY_PLAN_IDENTITY = "issue-80-reactive-diagnostic-v1"
OUTPUT = ROOT / ".local-artifacts/issue-82-zero-floor-oracle-v1"
SCHEMA = oracle.SCHEMA_PLAN
IDENTITY = oracle.IDENTITY
EVALUATION_SCHEMA = oracle.SCHEMA_EVALUATION
EVIDENCE_SCHEMA = oracle.SCHEMA_EVIDENCE
SEEDS = train77.SEEDS
STATES_PER_FAMILY = 12
FAMILIES = {"type010103": "rolling", "type010105": "sliding"}
MODEL_SYSTEMS = oracle.MODEL_SYSTEMS
PRIOR_SYSTEM = oracle.PRIOR_SYSTEM
SYSTEMS = oracle.SYSTEMS
SYSTEM_MODES = {
    "hybrid-fixed-h1": GameplayPlanningMode.HYBRID_FIXED,
    "continuous-fixed-h1": GameplayPlanningMode.CONTINUOUS_H1,
    "continuous-fixed-h5": GameplayPlanningMode.CONTINUOUS_H5,
}
SYSTEM_ARMS = {name: ("hybrid" if mode is GameplayPlanningMode.HYBRID_FIXED else "continuous",
                      fixed_mode_pair(mode))
               for name, mode in SYSTEM_MODES.items()}
PRIOR_ORDINAL = NO_MODEL_ORDINAL_PRIOR_ORDINAL
if PRIOR_ORDINAL != oracle.PRIOR_ORDINAL:
    raise ValueError("prior ordinal disagrees between the frozen modules")
DECISION_FIXED_STEP = 30000
END_OFFSET = 600
NATIVE_STRIDE = 50
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"
DECISION_DEVICE = "cuda"
GPU_ALLOWANCE_SECONDS = oracle.GPU_ALLOWANCE_SECONDS
WALL_CAP_SECONDS = oracle.WALL_CAP_SECONDS
ARTIFACT_BYTES_CAP = oracle.ARTIFACT_BYTES_CAP
BOOTSTRAP_DRAWS = oracle.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = oracle.BOOTSTRAP_SEED
EXPECTED_CELLS = 289  # state x admissible-candidate slots across the 24 frozen states
EXPECTED_PILOT_VALID, EXPECTED_PILOT_TYPED = 45, 3
SCAN_WORKERS = 8
FILES = (
    "scripts/run_reactive_zero_floor_oracle.py",
    "world_model/training/reactive_zero_floor_oracle.py",
)


def log(message):
    print(f"[issue-82-zero-floor-oracle] {message}", flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def file_identity(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def gpu_lock():
    """Exclusive shared-GPU contract lock for every wall-time-measured GPU phase."""
    handle = open(GPU_LOCK_PATH, "a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


# ------------------------------------------------------- read-only frozen inputs

def load_eighty_plan():
    plan = read(EIGHTY / "plan.json")
    if (plan.get("identity") != EIGHTY_PLAN_IDENTITY
            or plan.get("schema") != "issue_80_reactive_diagnostic_v1"
            or plan.get("frozen_before_outcome") is not True
            or len(plan["states"]) != 24):
        raise ValueError("requires the frozen issue-80 24-state reactive plan")
    return plan


def load_outcomes(members):
    outcomes = {}
    eighty_states = load_eighty_plan()["states"]
    for member in members:
        record = read(EIGHTY / "candidate-outcomes" / f"{member}.json")
        if record["schema"] != "issue_80_candidate_outcomes_v1":
            raise ValueError(f"unexpected candidate-outcome schema for {member}")
        stored = {row["ordinal"]: row for row in record["candidates"]}
        state = next(s for s in eighty_states if s["source_member"] == member)
        if len(stored) != len(state["inventory"]):
            raise ValueError(f"inventory/table size differs for {state['identity']}")
        for item in state["inventory"]:
            row = stored.get(item["ordinal"])
            if (row is None or row["branch_identity"] != item["branch_identity"]
                    or row["action"] != item["action"]
                    or not math.isfinite(row["realized_count_cost"])):
                raise ValueError(f"candidate table row differs for {item['branch_identity']}")
        outcomes[member] = record
    return outcomes


def load_gate():
    gate = read(EIGHTY / "pilot-gate.json")
    if (gate.get("schema") != "issue_80_pilot_gate_v1"
            or gate.get("plan_identity") != EIGHTY_PLAN_IDENTITY
            or gate.get("passed") is not False or gate.get("prevalence") != 0.0
            or gate.get("valid_executions") != EXPECTED_PILOT_VALID
            or gate.get("successes") != 0
            or len(gate.get("typed_failures") or []) != EXPECTED_PILOT_TYPED):
        raise ValueError("requires the recorded failed issue-80 pilot gate "
                         f"({EXPECTED_PILOT_VALID} valid executions, "
                         f"{EXPECTED_PILOT_TYPED} typed failures, prevalence 0)")
    return gate


def pilot_successes_by_system():
    """Closed-loop first-shot successes per system from the #80 pilot records.

    Read-only import; the published #80 pilot record is never amended.
    """
    eighty = load_eighty_plan()
    successes = {name: 0 for name in SYSTEMS}
    valid = typed = 0
    for system in SYSTEMS:
        for seed in SEEDS:
            for state in eighty["pilot_states"]:
                identity = f"{system}--seed{seed}--{state}"
                record_file = EIGHTY / "records" / f"{identity}.json"
                if not record_file.is_file():
                    raise ValueError(f"missing #80 pilot record {identity}")
                record = read(record_file)
                outcome = record.get("outcome")
                if record.get("failure") is not None or outcome is None:
                    typed += 1
                    continue
                valid += 1
                successes[system] += int(bool(outcome["first_shot_success"]))
    if (valid, typed) != (EXPECTED_PILOT_VALID, EXPECTED_PILOT_TYPED):
        raise ValueError(f"#80 pilot record inventory differs: {valid} valid, {typed} typed")
    return successes


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
    eighty = load_eighty_plan()
    if eighty["dynamics"]["checkpoints"] != bindings:
        raise ValueError("frozen predictor bindings differ from the #80 record")
    return bindings, dynamics


# ----------------------------------------------------- replay-side pig evidence

def _scan_branch(payload):
    branch, member, ordinal, realized_count_cost = payload
    result = read(CAMPAIGN / "results" / f"{branch}.json")
    segment = result["segments"][0]
    trace = NativeSegmentTrace(segment["native_root"])
    pig_removed = pig_contact = block_contact = False
    death_participants = []
    launches = 0
    for chunk in trace.trace.chunks():
        for event in chunk["events"]:
            name = event["event_type"]
            participants = " ".join(event.get("participants", []))
            if name == "bird_launched":
                launches += 1
            if name in ("entity_death", "entity_destroyed"):
                death_participants.append(participants)
                if "pig" in participants:
                    pig_removed = True
            if name == "collision" and "pig" in participants:
                pig_contact = True
            if name == "collision" and "block" in participants:
                block_contact = True
    entry = {
        "branch_identity": branch, "member_identity": member, "ordinal": ordinal,
        "realized_count_cost": realized_count_cost,
        "capture_id": segment["capture_id"],
        "native_root": segment["native_root"],
        "observation_manifest": segment["observation_manifest"],
        "first_fixed_step": segment["summary"]["first_fixed_step"],
        "last_fixed_step": segment["summary"]["last_fixed_step"],
        "censored": trace.censored,
        "terminal_observed": trace.summary["terminal_observed"],
        "segment_failure": segment["summary"].get("failure"),
        "event_counts": segment["summary"]["event_counts"],
        "interaction_coverage": {"pig_removed": pig_removed, "pig_contact": pig_contact,
                                 "block_contact": block_contact, "bird_launches": launches},
        "death_participants": death_participants,
    }
    return member, branch, entry


def evidence_path(member):
    return OUTPUT / "replay-evidence" / f"{member}.json"


def check_member_evidence(record, member):
    """Verify one evidence file binds to the frozen issue-77 campaign results."""
    if record.get("schema") != EVIDENCE_SCHEMA or record.get("member_identity") != member:
        raise ValueError(f"replay evidence identity differs for {member}")
    branches = record.get("branches")
    if not isinstance(branches, dict) or not branches:
        raise ValueError(f"replay evidence has no branches for {member}")
    for branch, entry in branches.items():
        result = read(CAMPAIGN / "results" / f"{branch}.json")
        segment = result["segments"][0]
        binding = {
            "branch_identity": branch, "member_identity": member,
            "ordinal": entry["ordinal"],
            "realized_count_cost": entry["realized_count_cost"],
            "capture_id": segment["capture_id"],
            "native_root": segment["native_root"],
            "observation_manifest": segment["observation_manifest"],
            "first_fixed_step": segment["summary"]["first_fixed_step"],
            "last_fixed_step": segment["summary"]["last_fixed_step"],
            "segment_failure": segment["summary"].get("failure"),
            "event_counts": segment["summary"]["event_counts"],
        }
        if any(entry.get(key) != value for key, value in binding.items()):
            raise ValueError(f"replay evidence binding differs for {branch}")
        if entry["interaction_coverage"]["bird_launches"] != 1:
            raise ValueError(f"replay evidence launch count differs for {branch}")


def scan_evidence(members, outcomes, workers=SCAN_WORKERS):
    """Ground per-cell replay pig evidence from native traces (CPU scan).

    Cached member files are reused only after their frozen-campaign bindings
    verify; anything missing or stale is re-scanned (crash-safe resume at
    member granularity, branch-granular inside one run).
    """
    payloads = []
    for member in members:
        path = evidence_path(member)
        if path.is_file():
            try:
                record = read(path)
                check_member_evidence(record, member)
                log(f"evidence cached+verified member={member} "
                    f"branches={len(record['branches'])}")
                continue
            except ValueError as error:
                log(f"evidence cache rejected member={member}: {error}; re-scanning")
        for candidate in outcomes[member]["candidates"]:
            payloads.append((candidate["branch_identity"], member,
                             candidate["ordinal"], candidate["realized_count_cost"]))
    if not payloads:
        return
    began = time.monotonic()
    log(f"scanning replay pig evidence for {len(payloads)} branches "
        f"with {workers} workers")
    context = multiprocessing.get_context("spawn")
    collected = {member: {} for member in members}
    with context.Pool(workers) as pool:
        for n, (member, branch, entry) in enumerate(
                pool.imap_unordered(_scan_branch, payloads), 1):
            collected[member][branch] = entry
            elapsed = time.monotonic() - began
            log(f"[{n}/{len(payloads)}] {branch} "
                f"pig_removed={entry['interaction_coverage']['pig_removed']} "
                f"cost={entry['realized_count_cost']:.2f} "
                f"elapsed={elapsed:.0f}s eta={elapsed / n * (len(payloads) - n):.0f}s")
    for member in members:
        if not collected[member]:
            continue
        record = {"schema": EVIDENCE_SCHEMA, "member_identity": member,
                  "branches": dict(sorted(collected[member].items()))}
        check_member_evidence(record, member)
        write(evidence_path(member), record)


def evidence_rows(states, members, outcomes):
    """Frozen per-cell evidence rows joined with their state slots."""
    by_member = {}
    for state in states:
        by_member.setdefault(state["source_member"], []).append(state["identity"])
    rows = []
    for member in members:
        record = read(evidence_path(member))
        check_member_evidence(record, member)
        for candidate in outcomes[member]["candidates"]:
            entry = record["branches"][candidate["branch_identity"]]
            rows.append({
                "member_identity": member,
                "state_identities": sorted(by_member[member]),
                "ordinal": candidate["ordinal"],
                "branch_identity": candidate["branch_identity"],
                "realized_count_cost": candidate["realized_count_cost"],
                "censored": entry["censored"],
                "pig_removed": entry["interaction_coverage"]["pig_removed"],
                "pig_contact": entry["interaction_coverage"]["pig_contact"],
                "segment_failure": entry["segment_failure"],
            })
    return rows


# ------------------------------------------------------------------ plan freeze

def definitions(binding, gate, pilot_successes):
    predicate = {
        "stage1_rule": oracle.SUCCESS_THRESHOLD_RULE,
        "stage2_rule": oracle.DEFENSIBILITY_GATE_RULE,
        "status": binding["status"],
        "proxy_label": ("replay-level proxy for the closed-loop first-shot "
                        "success (engine pig removal); NOT the closed-loop "
                        "outcome itself"),
        "declared_divergence": oracle.DECLARED_DIVERGENCE,
        "blocked_bindings_are_terminal": oracle.BLOCKED_BINDINGS_ARE_TERMINAL,
    }
    if binding["status"] == "bound":
        predicate["threshold"] = binding["threshold"]
    else:
        predicate["blocker"] = binding["blocker"]
        predicate["stage1_candidate_threshold"] = binding["stage1"]["candidate_threshold"]
        predicate["stage2_gate"] = binding["stage2"]
    return {
        "membership_rule": (
            "imported VERBATIM from the frozen issue-80 reactive plan: 24 states, "
            "12 per N1 generator family (type010103 rolling, type010105 sliding), "
            "evenly spaced coverage-admissible branch identities "
            "pool[round(i*(len(pool)-1)/11)] for i in 0..11; state identity = "
            "branch identity; initial state = source-member decision state at "
            f"native fixed step {DECISION_FIXED_STEP}"),
        "states_per_family": STATES_PER_FAMILY,
        "families": dict(FAMILIES),
        "seeds": list(SEEDS),
        "candidate_inventory": (
            "the frozen 13-candidate set (reference a00 plus angular a01-a12) with "
            "per-state counts reduced by typed branch failures exactly as recorded "
            "in the #77 headroom tables; imported from the #80 frozen state "
            "inventories and their published candidate outcome tables"),
        "systems": {
            "hybrid-fixed-h1": {"kind": "model", "arm": "hybrid",
                                "pair": {"delta": 1, "abstraction": "continuous"}},
            "continuous-fixed-h1": {"kind": "model", "arm": "continuous",
                                    "pair": {"delta": 1, "abstraction": "continuous"}},
            "continuous-fixed-h5": {"kind": "model", "arm": "continuous",
                                    "pair": {"delta": 5, "abstraction": "continuous"}},
            "no-model-ordinal-prior": {"kind": "no-model-ordinal-prior",
                                       "prior_ordinal": PRIOR_ORDINAL},
        },
        "decision_rule": (
            "the #80 frozen reactive decision rule applied replay-side: parse the "
            "sealed pre-decision frame with the frozen issue-70 adapter as a "
            "single-frame temporal context (identical to the #77 per-state ranking "
            "context), roll every admissible candidate to the system's fixed pair, "
            "choose argmin predicted count cost with ties broken by lower ordinal; "
            "candidates with nonfinite predictions are typed-excluded; the prior "
            "chooses ordinal 8 or records typed prior_candidate_absent"),
        "estimand_basis": (
            "the #80-published per-candidate realized t=600 end-of-window replay "
            "count cost tables (issue_80_candidate_outcomes_v1, derived from the "
            "issue-77 N1 campaign); observed offset "
            f"{END_OFFSET} (one frame = {NATIVE_STRIDE} native steps), "
            "right-censored, NOT a settled cost"),
        "success_predicate": predicate,
        "estimands": {
            "per_state_oracle_indicator": (
                "1 iff at least one admissible candidate of the state meets the "
                "frozen predicate (defined only under a bound predicate)"),
            "pooled_oracle_prevalence": (
                "mean per-state indicator over the 24 frozen states with a "
                f"descriptive paired-bootstrap interval ({BOOTSTRAP_DRAWS} draws, "
                f"seed {BOOTSTRAP_SEED}, resampling states)"),
            "per_state_best_candidate": (
                "argmin realized replay cost over the admissible candidates, ties "
                "broken by lower ordinal; predicate-free and published even under "
                "a blocked binding"),
            "system_vs_oracle_gap": (
                "per frozen system and seed: the state's chosen candidate under "
                "the #80 decision rule, its realized cost, and (under a bound "
                "predicate) its predicate verdict, against the oracle best "
                "candidate; paired indicator gap (oracle - system) with a "
                "descriptive bootstrap interval"),
        },
        "uncertainty": {
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "state",
            "note": ("seed differences averaged before resampling; descriptive "
                     "only; this diagnostic cannot support a confirmatory claim"),
        },
        "threshold_sensitivity_grid": list(oracle.THRESHOLD_SENSITIVITY_GRID),
        "threshold_sensitivity_scope": (
            "descriptive only; published even under a blocked binding to expose "
            "the cost's inability to track the success event; the primary "
            "estimand stays bound to the frozen Phase-0 binding, never to this "
            "grid"),
        "protocol_note": {
            "prior_candidate_absent_cells": 3,
            "note": oracle.PROTOCOL_NOTE,
            "prior_fallback_rule": oracle.PRIOR_FALLBACK_RULE,
            "amendment": ("no amendment to the published #80 pilot record or its "
                          "gate; the fallback binds future pilots only"),
        },
        "pilot_record": {
            "cells": gate["cells"], "valid_executions": gate["valid_executions"],
            "typed_failures": len(gate["typed_failures"]),
            "successes": gate["successes"], "prevalence": gate["prevalence"],
            "floor": gate["floor"], "passed": gate["passed"],
            "closed_loop_first_shot_successes_by_system": pilot_successes,
            "scope": ("read from the #80 pilot records read-only; the published "
                      "#80 pilot record and its gate are never amended here"),
        },
        "disposition_rules": {
            "q1_oracle_ceiling_nonzero": (
                "under a BOUND predicate: supported iff at least one frozen state "
                "has oracle_indicator=1 (prevalence point estimate > 0); "
                "prevalence 0 yields not_supported_by_this_experiment; a BLOCKED "
                "predicate yields readiness_or_precision_insufficient for both "
                "questions because the oracle ceiling is not measurable"),
            "q1_reading": (
                "state_difficulty_floor if prevalence = 0; ranking_failure if "
                "prevalence > 0 and all four frozen systems recorded 0 closed-loop "
                "first-shot successes in the #80 pilot; undefined when blocked"),
            "q2_recommendation": (
                "prevalence_floor_state_selection when prevalence = 0; "
                "h1_ranking_calibration_probe when the ranking_failure reading "
                "holds; undefined when blocked"),
            "tokens": ("supported / not_supported_by_this_experiment / "
                       "readiness_or_precision_insufficient"),
        },
        "caps": {
            "gpu_allowance_seconds": GPU_ALLOWANCE_SECONDS,
            "gpu_scope": ("cumulative synchronized wall of the optional gap "
                          "scoring (perception plus candidate rollouts) on cuda"),
            "gpu_lock": GPU_LOCK_PATH,
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "artifact_bytes_cap": ARTIFACT_BYTES_CAP,
            "stop_rule": ("exceeding any cap stops the ticket with "
                          "readiness_or_precision_insufficient"),
        },
        "claim_boundary": oracle.CLAIM_BOUNDARY,
    }


def state_cells(states, outcomes):
    """The scheduled state x candidate cells (each executed or typed-dispositioned)."""
    cells = []
    for state in states:
        by_ordinal = {row["ordinal"]: row for row in outcomes[state["source_member"]]["candidates"]}
        for item in state["inventory"]:
            row = by_ordinal[item["ordinal"]]
            cells.append({"state": state["identity"],
                          "generator_family": state["generator_family"],
                          "source_member": state["source_member"],
                          "ordinal": item["ordinal"],
                          "branch_identity": item["branch_identity"],
                          "realized_count_cost": row["realized_count_cost"]})
    return cells


def make_plan():
    eighty = load_eighty_plan()
    gate = load_gate()
    pilot_successes = pilot_successes_by_system()
    states = eighty["states"]
    members = sorted({state["source_member"] for state in states})
    outcomes = load_outcomes(members)
    scan_evidence(members, outcomes)
    rows = evidence_rows(states, members, outcomes)
    binding = oracle.bind_threshold(rows)
    checkpoints, dynamics = checkpoint_bindings()
    cells = state_cells(states, outcomes)
    if len(cells) != EXPECTED_CELLS:
        raise ValueError(f"frozen cell inventory is {len(cells)} cells, expected {EXPECTED_CELLS}")
    absent = sorted(state["identity"] for state in states if not state["prior_available"])
    if absent != ["issue-77-n1-003-a00", "issue-77-n1-003-a10"]:
        raise ValueError(f"prior-absent states differ from the frozen record: {absent}")
    return {
        "schema": SCHEMA, "identity": IDENTITY, "frozen_before_outcome": True,
        "definitions": definitions(binding, gate, pilot_successes),
        "source_issue_80": {
            "identity": EIGHTY_PLAN_IDENTITY,
            "plan_path": str(EIGHTY / "plan.json"),
            "source_revision": eighty["source_revision"],
            "pilot_gate_path": str(EIGHTY / "pilot-gate.json"),
            "candidate_outcomes_root": str(EIGHTY / "candidate-outcomes"),
            "records_root": str(EIGHTY / "records"),
        },
        "issue_77_n1_campaign": {"root": str(CAMPAIGN), "identity": "issue-77-n1-v1"},
        "dynamics": {"root": str(DYNAMICS), "identity": "issue-77-n1-dynamics-v1",
                     "vocabulary_entries": len(dynamics["contract"]["vocabulary"]),
                     "checkpoints": checkpoints},
        "membership": {
            "states": [{"identity": state["identity"],
                        "generator_family": state["generator_family"],
                        "source_member": state["source_member"],
                        "study_role": state["study_role"],
                        "engine_seed": state["engine_seed"],
                        "prior_available": state["prior_available"],
                        "prior_ordinal": state["prior_ordinal"],
                        "inventory": state["inventory"],
                        "typed_dropped_branches": state["typed_dropped_branches"]}
                       for state in states],
            "pilot_states": eighty["pilot_states"],
            "members": members,
        },
        "binding": binding,
        "evidence_rows": rows,
        "cells": cells,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_text": {path: (ROOT / path).read_text() for path in FILES},
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }


def load_plan():
    plan = read(OUTPUT / "plan.json")
    if plan.get("identity") != IDENTITY:
        raise ValueError("retained issue-82 plan identity differs")
    return plan


def check_plan_bound(plan):
    """The frozen plan must still equal its recomputation from frozen inputs."""
    current = make_plan()
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-82 plan source/evidence changed; preserve and version")
    return plan


# ----------------------------------------------------------- optional GPU scoring

def load_frozen_predictor(plan, seed, arm, device):
    binding_entry = plan["dynamics"]["checkpoints"][str(seed)][arm]
    path = Path(binding_entry["path"])
    if file_identity(path) != binding_entry["identity"]:
        raise ValueError(f"frozen predictor bytes changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    dynamics = read(DYNAMICS / "plan.json")
    stored = payload.get("binding")
    if (not isinstance(stored, dict) or stored.get("plan_identity") != dynamics["identity"]
            or stored.get("seed") != seed or stored.get("arm") != arm
            or stored.get("component") != "predictor"):
        raise ValueError(f"predictor binding differs for seed={seed} arm={arm}")
    model = (ContinuousDynamics(dynamics["capacity"]["continuous_width"])
             if arm == "continuous" else CNNHybridPredictor())
    model.load_state_dict(payload["model"], strict=True)
    steps = dynamics["training"]["steps"]
    expected_counts = {str(p.identity): steps // len(pairs_for(model))
                       for p in pairs_for(model)}
    if payload["step"] != steps or payload["pair_counts"] != expected_counts:
        raise ValueError(f"incomplete predictor budget for seed={seed} arm={arm}")
    return model.to(torch.device(device)).eval()


def action_tensor(action, device):
    return torch.tensor([[action["drag_x"] / 480., action["drag_y"] / 480.,
                          action["release_time_ms"] / 1000., action["tap_time_ms"] / 1000., 1.]],
                        device=device)


def carrier_path(member):
    return OUTPUT / "scoring" / "decision-carriers" / f"{member}.json"


def reference_branch_for(states, member):
    state = next(state for state in states if state["source_member"] == member)
    return min(state["inventory"], key=lambda item: item["ordinal"])["branch_identity"]


def decision_carrier(states, member, adapter, device):
    """One B=1 perception of the member's sealed pre-decision frame.

    Reference branch = lowest-ordinal admissible branch, exactly the #77
    per-state ranking context (shot-1 frame 0, single-frame temporal context).
    """
    branch = reference_branch_for(states, member)
    result = read(CAMPAIGN / "results" / f"{branch}.json")
    obsroot = CAMPAIGN / "attempts" / branch / "shot-1" / "observation-trace"
    manifest = read(obsroot / MANIFEST_NAME)
    if manifest["identity"] != result["segments"][0]["observation_manifest"]:
        raise ValueError(f"decision-frame manifest differs for {branch}")
    frames = manifest["frame_records"]
    if frames[0]["fixed_step"] != DECISION_FIXED_STEP:
        raise ValueError(f"decision frame differs from the sealed step for {branch}")
    ref = frames[0]["agent_observation"]
    png = (obsroot / ref["relative_path"]).read_bytes()
    observation = AgentObservation(ref["identity"], frames[0]["fixed_step"],
                                   frames[0]["fixed_time_seconds"], png, "agent")
    grid.synchronize(device)
    began = time.monotonic()
    with torch.no_grad():
        parsed = adapter.parse_batch((observation,))[0]
        carrier = adapter.build_from_parsed(
            TemporalObservationContext(None, observation), parsed, None).tensor
        grid.synchronize(device)
    return {"schema": "issue_82_decision_carrier_v1", "member_identity": member,
            "reference_branch": branch, "frame_identity": ref["identity"],
            "png_sha256": "sha256:" + sha256(png).hexdigest(),
            "fixed_step": frames[0]["fixed_step"],
            "carrier": carrier.tolist(), "parser_wall_seconds": time.monotonic() - began,
            "parser_calls": 1}


def verify_carrier_record(states, member, record):
    branch = reference_branch_for(states, member)
    result = read(CAMPAIGN / "results" / f"{branch}.json")
    obsroot = CAMPAIGN / "attempts" / branch / "shot-1" / "observation-trace"
    manifest = read(obsroot / MANIFEST_NAME)
    ref = manifest["frame_records"][0]["agent_observation"]
    png = (obsroot / ref["relative_path"]).read_bytes()
    expected = {"schema": "issue_82_decision_carrier_v1", "member_identity": member,
                "reference_branch": branch, "frame_identity": ref["identity"],
                "png_sha256": "sha256:" + sha256(png).hexdigest(),
                "fixed_step": DECISION_FIXED_STEP, "parser_calls": 1}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"decision-carrier binding differs for {member}")
    carrier = record.get("carrier")
    if (not isinstance(carrier, list) or len(carrier) != 236
            or not all(math.isfinite(x) for x in carrier)):
        raise ValueError(f"decision carrier is not a finite 236-vector for {member}")


def rollout_predictions(plan, states, member, system, seed, model, pair, objective,
                        device, carrier_record):
    """Roll every admissible candidate of the member to the system's fixed pair.

    The rollout is bound to the decision carrier it was rolled from (frame
    identity + png hash) so a cached prediction can never be reused against a
    different perception of the sealed pre-decision state.
    """
    state = next(state for state in states if state["source_member"] == member)
    inventory = {item["ordinal"]: item["action"] for item in state["inventory"]}
    carrier = carrier_record["carrier"]
    macs = pair_work(model, pair)
    rows = []
    grid.synchronize(device)
    began = time.monotonic()
    with torch.no_grad():
        z0 = torch.tensor(carrier, device=device)[None]
        for ordinal in sorted(inventory):
            value = z0
            steps = 0
            failed = False
            for _ in range(pair.delta):
                value = model.carrier(value, action_tensor(inventory[ordinal], device), pair)
                steps += 1
                if not bool(torch.isfinite(value).all()):
                    failed = True
                    break
            rows.append({"ordinal": ordinal,
                         "predicted_cost": None if failed else float(objective(value[0])),
                         "excluded": failed, "transition_calls": steps,
                         "linear_macs": macs * steps})
        grid.synchronize(device)
    return {"member_identity": member, "system": system, "seed": seed,
            "pair": {"delta": pair.delta, "abstraction": str(pair.abstraction)},
            "checkpoint_identity": plan["dynamics"]["checkpoints"][str(seed)]
                                   [SYSTEM_ARMS[system][0]]["identity"],
            "carrier_frame_identity": carrier_record["frame_identity"],
            "carrier_png_sha256": carrier_record["png_sha256"],
            "candidate_count": len(inventory),
            "rows": sorted(rows, key=lambda row: row["ordinal"]),
            "rollout_wall_seconds": time.monotonic() - began,
            "linear_macs": sum(row["linear_macs"] for row in rows),
            "transition_calls": sum(row["transition_calls"] for row in rows)}


def verify_prediction_record(plan, states, member, record, system=None, seed=None,
                             carrier_record=None):
    if record.get("member_identity") != member:
        raise ValueError(f"prediction member differs for {member}")
    system = system or record["system"]
    seed = seed if seed is not None else record["seed"]
    if record.get("system") != system or record.get("seed") != seed:
        raise ValueError(f"prediction identity differs for {member}")
    arm, pair = SYSTEM_ARMS[system]
    if record.get("pair") != {"delta": pair.delta, "abstraction": str(pair.abstraction)}:
        raise ValueError(f"prediction pair differs for {system} seed {seed}")
    if record.get("checkpoint_identity") != plan["dynamics"]["checkpoints"][str(seed)][arm]["identity"]:
        raise ValueError(f"prediction checkpoint differs for {system} seed {seed}")
    if carrier_record is not None:
        if (record.get("carrier_frame_identity") != carrier_record["frame_identity"]
                or record.get("carrier_png_sha256") != carrier_record["png_sha256"]):
            raise ValueError(f"prediction carrier differs for {system} seed {seed} {member}")
    state = next(state for state in states if state["source_member"] == member)
    ordinals = sorted(item["ordinal"] for item in state["inventory"])
    rows = record.get("rows") or []
    if [row["ordinal"] for row in rows] != ordinals:
        raise ValueError(f"prediction inventory differs for {system} seed {seed} {member}")
    for row in rows:
        steps = row["transition_calls"]
        cost = row.get("predicted_cost")
        if row["excluded"]:
            if cost is not None or not 1 <= steps <= pair.delta:
                raise ValueError(f"excluded prediction row differs for {member}")
        elif (not isinstance(cost, (int, float)) or isinstance(cost, bool)
              or not math.isfinite(cost) or steps != pair.delta):
            raise ValueError(f"prediction row differs for {member}")
    if record["candidate_count"] != len(ordinals):
        raise ValueError(f"prediction candidate count differs for {member}")


def prediction_path(system, seed):
    return OUTPUT / "scoring" / "predictions" / f"{system}--seed{seed}.json"


class Allowance:
    """Pre-declared cap accounting; exceeding stops with a typed stop reason.

    Accounting is per invocation: a crashed phase resumes from verified
    caches, so repeated resume runs can each use up to the allowance; the
    realistic total here is minutes against a 360 s GPU / 3600 s wall
    allowance, and the stopped evaluation is retained as the terminal record.
    """

    def __init__(self):
        self.gpu_seconds = 0.0
        self.wall_seconds = 0.0
        self.stop = None

    def spend(self, gpu_segment, wall_segment):
        self.gpu_seconds += gpu_segment
        self.wall_seconds += wall_segment
        if self.stop is None and self.gpu_seconds > GPU_ALLOWANCE_SECONDS:
            self.stop = "gpu_allowance_exceeded"
        if self.stop is None and self.wall_seconds > WALL_CAP_SECONDS:
            self.stop = "wall_cap_exceeded"
        return self.stop

    def check_artifact_cap(self, output):
        if self.stop is None and artifact_bytes(output) is not None:
            if artifact_bytes(output) > ARTIFACT_BYTES_CAP:
                self.stop = "artifact_limit_exceeded"
        return self.stop


def artifact_bytes(output):
    total = 0
    for root, _dirs, names in os.walk(output):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                return None
    return total


def score(plan, args):
    """Optional model scoring for the gap diagnostics (GPU phase, flock-held).

    Predicate-independent: the decision rule and predicted costs do not use
    the success predicate, so this phase runs (and stays resumable) under
    either binding status.
    """
    states = plan["membership"]["states"]
    members = plan["membership"]["members"]
    adapter = old.repair.load_repaired_adapter(old.repair.ROOT, args.device)
    vocabulary = read(DYNAMICS / "plan.json")["contract"]["vocabulary"]
    objective = TaskObjective.from_vocabulary(vocabulary)
    allowance = Allowance()

    for n, member in enumerate(members, 1):
        path = carrier_path(member)
        if path.is_file():
            try:
                record = read(path)
                verify_carrier_record(states, member, record)
                log(f"carrier cached+verified [{n}/{len(members)}] {member}")
                continue
            except ValueError as error:
                log(f"carrier cache rejected {member}: {error}; re-parsing")
        record = decision_carrier(states, member, adapter, args.device)
        write(path, record)
        allowance.spend(record["parser_wall_seconds"], record["parser_wall_seconds"])
        log(f"carrier parsed [{n}/{len(members)}] {member} "
            f"branch={record['reference_branch']} "
            f"gpu={record['parser_wall_seconds']:.3f}s")
        if allowance.stop:
            return allowance

    for system in MODEL_SYSTEMS:
        arm, pair = SYSTEM_ARMS[system]
        for seed in SEEDS:
            path = prediction_path(system, seed)
            if path.is_file():
                try:
                    record = read(path)
                    store_members = record.get("members") or {}
                    if (record.get("schema") == "issue_82_predictions_v1"
                            and record.get("system") == system
                            and record.get("seed") == seed
                            and len(store_members) == len(members)):
                        for member in members:
                            verify_prediction_record(plan, states, member,
                                                     store_members[member],
                                                     system, seed,
                                                     read(carrier_path(member)))
                        log(f"predictions cached+verified {system} seed {seed}")
                        continue
                except ValueError as error:
                    log(f"prediction cache rejected {system} seed {seed}: {error}; re-rolling")
            model = load_frozen_predictor(plan, seed, arm, args.device)
            load_began = time.monotonic()
            grid.synchronize(args.device)
            allowance.spend(0.0, time.monotonic() - load_began)
            per_member = {}
            for n, member in enumerate(members, 1):
                member_began = time.monotonic()
                carrier_record = read(carrier_path(member))
                entry = rollout_predictions(plan, states, member, system, seed,
                                            model, pair, objective, args.device,
                                            carrier_record)
                verify_prediction_record(plan, states, member, entry, system, seed,
                                         carrier_record)
                per_member[member] = entry
                allowance.spend(entry["rollout_wall_seconds"],
                                time.monotonic() - member_began)
                log(f"predictions [{n}/{len(members)}] {system} seed {seed} {member} "
                    f"gpu={entry['rollout_wall_seconds']:.3f}s")
                if allowance.stop:
                    return allowance
            del model
            store = {"schema": "issue_82_predictions_v1", "system": system, "seed": seed,
                     "members": dict(sorted(per_member.items())),
                     "linear_macs": sum(entry["linear_macs"] for entry in per_member.values()),
                     "transition_calls": sum(entry["transition_calls"]
                                             for entry in per_member.values())}
            write(path, store)
            log(f"predictions stored {system} seed {seed} "
                f"macs={store['linear_macs']}")
    return allowance


# ------------------------------------------------------------- evaluation build

def typed_choice_for_prior(state):
    ordinal = ordinal_prior_candidate([item["ordinal"] for item in state["inventory"]])
    if ordinal is None:
        return {"failure": "prior_candidate_absent", "chosen_ordinal": None,
                "chosen_predicted_cost": None, "candidate_count": len(state["inventory"])}
    return {"failure": None, "chosen_ordinal": ordinal,
            "chosen_predicted_cost": None, "candidate_count": len(state["inventory"])}


def load_prediction_stores():
    stores = {}
    for system in MODEL_SYSTEMS:
        for seed in SEEDS:
            path = prediction_path(system, seed)
            if not path.is_file():
                raise ValueError(f"missing prediction store {path}; run --run-evaluation")
            store = read(path)
            if store.get("system") != system or store.get("seed") != seed:
                raise ValueError(f"prediction store identity differs for {path}")
            stores[(system, seed)] = store
    return stores


def build_evaluation(plan, allowance):
    states = sorted(plan["membership"]["states"], key=lambda state: state["identity"])
    members = plan["membership"]["members"]
    outcomes = load_outcomes(members)
    binding = plan["binding"]
    bound = binding["status"] == "bound"
    if allowance.stop is not None:
        # A cap stop is terminal: no new scoring happened for the unfinished
        # keys, so the retained record carries empty per-state tables (the
        # verified caches remain for audit; nothing is outcome-replaced).
        stores = {}
        per_state = []
        parser_calls = 0
        transition_calls = 0
        linear_macs = 0
        candidate_slots = 0
    else:
        stores = load_prediction_stores()
        transition_calls = sum(stores[(system, seed)]["members"][member]["transition_calls"]
                               for system in MODEL_SYSTEMS for seed in SEEDS
                               for member in members)
        linear_macs = sum(stores[(system, seed)]["members"][member]["linear_macs"]
                          for system in MODEL_SYSTEMS for seed in SEEDS
                          for member in members)
        candidate_slots = (sum(len(state["inventory"]) for state in states)
                           * len(MODEL_SYSTEMS) * len(SEEDS))
        parsed_members = set()
        parser_calls = 0
        per_state = []
    pilot_cells = []
    for system in SYSTEMS:
        for seed in SEEDS:
            for pilot_state in plan["membership"]["pilot_states"]:
                record = read(EIGHTY / "records" / f"{system}--seed{seed}--{pilot_state}.json")
                outcome = record.get("outcome")
                pilot_cells.append({
                    "identity": record["cell"]["identity"], "system": system,
                    "seed": seed, "state": pilot_state,
                    "failure": record.get("failure"),
                    "success": None if outcome is None else bool(outcome["first_shot_success"]),
                    "chosen_realized_count_cost": (None if outcome is None
                                                   else outcome["chosen_frozen_cost"]),
                })
    if allowance.stop is not None:
        return {
            "schema": EVALUATION_SCHEMA, "identity": IDENTITY + "-evaluation",
            "plan_identity": plan["identity"],
            "predicate_bound": bound,
            "complete": False,
            "stop": allowance.stop,
            "compute": {
                "gpu_seconds_elapsed": allowance.gpu_seconds,
                "wall_seconds_elapsed": allowance.wall_seconds,
                "gpu_allowance_seconds": GPU_ALLOWANCE_SECONDS,
                "wall_cap_seconds": WALL_CAP_SECONDS,
                "decision_frames": parser_calls,
                "parser_calls": parser_calls,
                "predictor_loads": 0,
                "transition_calls": transition_calls,
                "linear_macs": linear_macs,
                "candidate_slots": candidate_slots,
                "candidate_slots_scope": ("state x candidate cells x the three model "
                                          "systems x all seeds (2601 slots)"),
            },
            "per_state": per_state,
            "pilot_cells": pilot_cells,
        }
    for state in states:
        member = state["source_member"]
        table = {row["ordinal"]: row for row in outcomes[member]["candidates"]}
        evidence = {row["branch_identity"]: row for row in plan["evidence_rows"]
                    if row["member_identity"] == member}
        carrier = read(carrier_path(member))
        if member not in parsed_members:
            parsed_members.add(member)
            parser_calls += carrier["parser_calls"]
        rows = []
        for item in state["inventory"]:
            cost_row = table[item["ordinal"]]
            proof = evidence[item["branch_identity"]]
            rows.append({
                "ordinal": item["ordinal"],
                "branch_identity": item["branch_identity"],
                "realized_count_cost": cost_row["realized_count_cost"],
                "pig_removed": proof["pig_removed"],
                "pig_contact": proof["pig_contact"],
                "censored": proof["censored"],
                **({"replay_success": oracle.predicate_verdict(
                    cost_row["realized_count_cost"], binding)} if bound else {}),
            })
        best = oracle.best_candidate_row(rows)
        oracle_row = {"admissible_candidates": len(rows), "best_candidate": best,
                      "best_candidate_tied_ordinals": sorted(
                          row["ordinal"] for row in rows
                          if row["realized_count_cost"] == best["realized_count_cost"])}
        if bound:
            full = oracle.oracle_state_row(rows, binding)
            oracle_row["oracle_indicator"] = full["oracle_indicator"]
            oracle_row["success_ordinals"] = full["success_ordinals"]
        choices = {}
        for system in SYSTEMS:
            for seed in SEEDS:
                if system == PRIOR_SYSTEM:
                    choice = typed_choice_for_prior(state)
                else:
                    entry = stores[(system, seed)]["members"][member]
                    finite = [row for row in entry["rows"] if not row["excluded"]]
                    if not finite:
                        choice = {"failure": "all_candidate_predictions_failed",
                                  "chosen_ordinal": None, "chosen_predicted_cost": None,
                                  "candidate_count": len(entry["rows"])}
                    else:
                        best_row = min(finite, key=lambda row: (row["predicted_cost"],
                                                                row["ordinal"]))
                        choice = {"failure": None, "chosen_ordinal": best_row["ordinal"],
                                  "chosen_predicted_cost": best_row["predicted_cost"],
                                  "candidate_count": len(entry["rows"])}
                if choice["failure"] is not None:
                    choices[f"{system}--seed{seed}"] = {
                        **choice, "chosen_realized_count_cost": None,
                        "chosen_replay_success": None, "gap_cost": None,
                        "indicator_gap": None}
                    continue
                chosen_cost = table[choice["chosen_ordinal"]]["realized_count_cost"]
                slot = {**choice, "chosen_realized_count_cost": chosen_cost,
                        "gap_cost": chosen_cost - best["realized_count_cost"]}
                if bound:
                    success = oracle.predicate_verdict(chosen_cost, binding)
                    slot["chosen_replay_success"] = success
                    slot["indicator_gap"] = oracle_row["oracle_indicator"] - int(success)
                else:
                    slot["chosen_replay_success"] = None
                    slot["indicator_gap"] = None
                choices[f"{system}--seed{seed}"] = slot
        per_state.append({"state": state["identity"],
                          "generator_family": state["generator_family"],
                          "source_member": member,
                          "study_role": state["study_role"],
                          "cells": rows,
                          "oracle": oracle_row,
                          "system_choices": dict(sorted(choices.items()))})
    return {
        "schema": EVALUATION_SCHEMA, "identity": IDENTITY + "-evaluation",
        "plan_identity": plan["identity"],
        "predicate_bound": bound,
        "complete": allowance.stop is None,
        "stop": allowance.stop,
        "compute": {
            "gpu_seconds_elapsed": allowance.gpu_seconds,
            "wall_seconds_elapsed": allowance.wall_seconds,
            "gpu_allowance_seconds": GPU_ALLOWANCE_SECONDS,
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "decision_frames": parser_calls,
            "parser_calls": parser_calls,
            "predictor_loads": len(MODEL_SYSTEMS) * len(SEEDS),
            "transition_calls": transition_calls,
            "linear_macs": linear_macs,
            "candidate_slots": candidate_slots,
            "candidate_slots_scope": ("state x candidate cells x the three model "
                                      "systems x all seeds (2601 slots)"),
        },
        "per_state": per_state,
        "pilot_cells": pilot_cells,
    }


def load_verified_evaluation(plan):
    evaluation = read(OUTPUT / "evaluation.json")
    if (evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("plan_identity") != plan["identity"]):
        raise ValueError("retained issue-82 evaluation identity differs")
    bound = plan["binding"]["status"] == "bound"
    if evaluation.get("predicate_bound") is not bound:
        raise ValueError("evaluation predicate status differs from the frozen plan")
    return evaluation


# ------------------------------------------------------------------- publication

def system_gap_tables(plan, evaluation):
    """Per-system/per-seed gap rows, paired contrasts, and state choice rows."""
    stores = load_prediction_stores()
    bound = plan["binding"]["status"] == "bound"
    per_state_rows = []
    system_gap = {}
    contrasts = []
    for system in SYSTEMS:
        per_seed = {}
        per_state_differences = {}
        for seed in SEEDS:
            scored = [row for row in evaluation["per_state"]
                      if row["system_choices"][f"{system}--seed{seed}"]["failure"] is None]
            typed = [row for row in evaluation["per_state"]
                     if row["system_choices"][f"{system}--seed{seed}"]["failure"] is not None]
            chosen_costs = [row["system_choices"][f"{system}--seed{seed}"]["chosen_realized_count_cost"]
                            for row in scored]
            gaps = [row["system_choices"][f"{system}--seed{seed}"]["gap_cost"] for row in scored]
            successes = [row["system_choices"][f"{system}--seed{seed}"]["chosen_replay_success"]
                         for row in scored]
            per_seed[str(seed)] = {
                "states_scored": len(scored),
                "typed_decision_failures": len(typed),
                "typed_failure_states": sorted(row["state"] for row in typed),
                "predicate_success_states": (None if not bound
                                             else sum(int(value) for value in successes)),
                "mean_chosen_realized_count_cost": (float(np.mean(chosen_costs))
                                                    if chosen_costs else None),
                "mean_oracle_best_realized_count_cost": (float(np.mean(
                    [row["oracle"]["best_candidate"]["realized_count_cost"]
                     for row in scored])) if scored else None),
                "mean_gap_cost": float(np.mean(gaps)) if gaps else None,
                "transition_calls": (0 if system == PRIOR_SYSTEM else
                                     stores[(system, seed)]["transition_calls"]),
                "linear_macs": (0 if system == PRIOR_SYSTEM else
                                stores[(system, seed)]["linear_macs"]),
            }
            for row in evaluation["per_state"]:
                choice = row["system_choices"][f"{system}--seed{seed}"]
                per_state_rows.append({
                    "state": row["state"], "system": system, "seed": seed,
                    "failure": choice["failure"],
                    "chosen_ordinal": choice["chosen_ordinal"],
                    "chosen_predicted_cost": choice["chosen_predicted_cost"],
                    "chosen_realized_count_cost": choice["chosen_realized_count_cost"],
                    "chosen_replay_success": choice["chosen_replay_success"],
                    "oracle_best_realized_count_cost":
                        row["oracle"]["best_candidate"]["realized_count_cost"],
                    "gap_cost": choice["gap_cost"],
                    "indicator_gap": choice["indicator_gap"],
                })
            if bound:
                for row in evaluation["per_state"]:
                    choice = row["system_choices"][f"{system}--seed{seed}"]
                    if choice["failure"] is not None:
                        continue
                    per_state_differences.setdefault(row["state"], []).append(
                        row["oracle"]["oracle_indicator"] - int(choice["chosen_replay_success"]))
        # seeds averaged before resampling, per the frozen uncertainty rule
        gap_differences = [float(np.mean(differences))
                           for _, differences in sorted(per_state_differences.items())]
        pooled_gap = oracle.indicator_interval(gap_differences) if bound else None
        system_gap[system] = {
            "per_seed": per_seed,
            "indicator_gap_oracle_minus_system": pooled_gap,
            "paired_states": len(gap_differences),
        }
        if bound:
            contrasts.append({
                "system": system,
                "metric": "oracle_minus_system_replay_success_indicator",
                "descriptive": pooled_gap,
                "paired_states": len(gap_differences),
                "scope": ("descriptive paired bootstrap over state identities; "
                          "seeds averaged first"),
            })
    return system_gap, contrasts, per_state_rows


def publication(plan):
    binding = plan["binding"]
    bound = binding["status"] == "bound"
    common = {
        "schema": oracle.SCHEMA_REPORT, "identity": IDENTITY + "-report",
        "plan_identity": plan["identity"],
        "frozen_before_outcome": plan["frozen_before_outcome"],
        "binding": binding,
        "definitions": plan["definitions"],
        "claim_boundary": plan["definitions"]["claim_boundary"],
        "source_issue_80": plan["source_issue_80"],
        "issue_77_n1_campaign": plan["issue_77_n1_campaign"],
        "membership_summary": {
            "states": len(plan["membership"]["states"]),
            "members": len(plan["membership"]["members"]),
            "pilot_states": plan["membership"]["pilot_states"],
            "cells": len(plan["cells"]),
        },
        "limitations": list(oracle.LIMITATIONS),
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }
    evaluation = load_verified_evaluation(plan)
    common["compute"] = evaluation["compute"]
    common["protocol_note"] = plan["definitions"]["protocol_note"]
    candidate_threshold = binding["stage1"]["candidate_threshold"]
    if candidate_threshold is None:
        # No-removals blocked variant: no candidate predicate exists at all.
        common["pilot_divergence"] = {
            "valid_cells": 0, "typed_failure_cells": 0, "agreement_fraction": None,
            "predicate_success_and_gameplay_success": 0,
            "predicate_success_and_gameplay_failure": 0,
            "predicate_failure_and_gameplay_success": 0,
            "predicate_failure_and_gameplay_failure": 0,
            "note": "no stage-1 candidate threshold exists, so the predicate "
                    "divergence is undefined"}
    else:
        common["pilot_divergence"] = oracle.pilot_divergence_table(
            evaluation["pilot_cells"],
            binding if bound else {"status": "bound", "threshold": candidate_threshold})
    common["per_state_best_candidates"] = [
        {"state": row["state"], "generator_family": row["generator_family"],
         "source_member": row["source_member"],
         "admissible_candidates": row["oracle"]["admissible_candidates"],
         "best_candidate": row["oracle"]["best_candidate"],
         "best_candidate_tied_ordinals": row["oracle"]["best_candidate_tied_ordinals"],
         **({"success_ordinals": row["oracle"]["success_ordinals"],
             "oracle_indicator": row["oracle"]["oracle_indicator"]} if bound else {})}
        for row in evaluation["per_state"]]
    common["threshold_sensitivity"] = oracle.threshold_sensitivity(plan["evidence_rows"])
    if not evaluation["complete"]:
        common.update({
            "system_cost_gap": None, "system_choice_rows": [],
            "dispositions": None,
            "ticket_disposition": "readiness_or_precision_insufficient",
            "stopExplanation": (f"evaluation stopped: {evaluation['stop']}; "
                                "compute allowance exceeded"),
            "evaluation_complete": False,
        })
        return common
    system_gap, contrasts, choice_rows = system_gap_tables(plan, evaluation)
    common["system_cost_gap"] = {
        "per_system": {name: {"per_seed": system_gap[name]["per_seed"]}
                       for name in SYSTEMS},
        "scope": ("predicate-free cost gaps: the system's chosen candidate's "
                  "realized replay cost against the best realized cost"),
    }
    common["system_choice_rows"] = choice_rows
    common["evaluation_complete"] = True
    if not bound:
        common.update({
            "ticket_disposition": "readiness_or_precision_insufficient",
            "stopExplanation": (
                "Phase 0 found no defensible replay-level success predicate "
                "binding, so the oracle ceiling is not measurable on this "
                "membership: " + binding["blocker"]),
            "dispositions": oracle.decide_dispositions(
                binding, None,
                plan["definitions"]["pilot_record"]["closed_loop_first_shot_successes_by_system"]),
        })
        return common
    indicators = [row["oracle"]["oracle_indicator"] for row in evaluation["per_state"]]
    prevalence = oracle.indicator_interval(indicators)
    replay_divergence = oracle.replay_divergence_table(plan["evidence_rows"], binding)
    dispositions = oracle.decide_dispositions(
        binding, prevalence["mean"],
        plan["definitions"]["pilot_record"]["closed_loop_first_shot_successes_by_system"])
    return {**common,
            "estimands": {
                "oracle_prevalence": {
                    "states": len(indicators),
                    "states_with_oracle_success": int(sum(indicators)),
                    **prevalence,
                    "scope": ("descriptive paired bootstrap over the 24 frozen "
                              "state identities"),
                },
                "system_gap": system_gap,
                "system_gap_contrasts": contrasts,
                "divergence": {
                    "replay_side": replay_divergence,
                    "closed_loop_side": common["pilot_divergence"],
                    "declared": plan["definitions"]["success_predicate"]["declared_divergence"],
                },
            },
            "dispositions": dispositions,
            "ticket_disposition": dispositions["q1_oracle_ceiling_nonzero"]}


def fmt(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def fmt_int(value):
    return "n/a" if value is None else str(int(value))


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("system", "seed", "states_scored", "typed_decision_failures",
                     "predicate_success_states", "mean_chosen_realized_count_cost",
                     "mean_oracle_best_realized_count_cost", "mean_gap_cost",
                     "transition_calls", "linear_macs"))
    gap = (result.get("system_cost_gap") or {}).get("per_system", {})
    for system in SYSTEMS:
        entry = gap.get(system)
        if entry is None:
            continue
        for seed in SEEDS:
            row = entry["per_seed"][str(seed)]
            writer.writerow((system, seed, row["states_scored"],
                             row["typed_decision_failures"],
                             fmt_int(row["predicate_success_states"]),
                             fmt(row["mean_chosen_realized_count_cost"], 3),
                             fmt(row["mean_oracle_best_realized_count_cost"], 3),
                             fmt(row["mean_gap_cost"], 3),
                             row["transition_calls"], row["linear_macs"]))
    writer.writerow(())
    writer.writerow(("state", "system", "seed", "failure", "chosen_ordinal",
                     "chosen_realized_count_cost", "chosen_replay_success",
                     "oracle_best_realized_count_cost", "gap_cost", "indicator_gap"))
    for row in result.get("system_choice_rows", []):
        writer.writerow((row["state"], row["system"], row["seed"], row["failure"] or "",
                         fmt_int(row["chosen_ordinal"]), fmt(row["chosen_realized_count_cost"], 3),
                         fmt_int(row["chosen_replay_success"]),
                         fmt(row["oracle_best_realized_count_cost"], 3),
                         fmt(row["gap_cost"], 3), fmt_int(row["indicator_gap"])))
    writer.writerow(())
    writer.writerow(("state", "family", "source_member", "admissible_candidates",
                     "best_ordinal", "best_branch_identity", "best_realized_count_cost",
                     "oracle_indicator"))
    for row in result.get("per_state_best_candidates", []):
        writer.writerow((row["state"], row["generator_family"], row["source_member"],
                         row["admissible_candidates"], row["best_candidate"]["ordinal"],
                         row["best_candidate"]["branch_identity"],
                         fmt(row["best_candidate"]["realized_count_cost"], 3),
                         fmt_int(row.get("oracle_indicator"))))
    writer.writerow(())
    writer.writerow(("contrast_system", "metric", "mean", "descriptive_95_low",
                     "descriptive_95_high", "paired_states"))
    for contrast in result.get("estimands", {}).get("system_gap_contrasts", []):
        descriptive = contrast["descriptive"]
        interval = descriptive["descriptive_95_percent_interval"]
        writer.writerow((contrast["system"], contrast["metric"], fmt(descriptive["mean"]),
                         fmt(interval[0]), fmt(interval[1]), contrast["paired_states"]))
    return stream.getvalue()


def findings_md(result):
    lines = ["# Issue-82 oracle-ceiling diagnostic of the zero floor - findings", ""]
    lines.append(f"Plan: {result['plan_identity']} (frozen before outcome: "
                 f"{result['frozen_before_outcome']}).")
    lines.append("")
    lines.append(f"Claim boundary: {result['claim_boundary']}")
    lines.append("")
    binding = result["binding"]
    bound = binding["status"] == "bound"
    lines.append("## Phase-0 success-predicate binding")
    lines.append("")
    lines.append(f"Status: {binding['status']}.")
    lines.append("")
    lines.append("Stage 1 rule: " + binding["stage1_rule"])
    lines.append("")
    lines.append("Stage 2 rule: " + binding["stage2_rule"])
    lines.append("")
    stage1 = binding["stage1"]
    if stage1["candidate_threshold"] is None:
        lines.append("Stage 1 outcome: no frozen cell records a replay-side pig removal, "
                     "so no candidate threshold exists; "
                     f"{binding['pig_removed_cells']}/{binding['cells']} frozen cells record "
                     f"replay-side pig removal; {binding['right_censored_cells']}/"
                     f"{binding['cells']} replay segments are right-censored at the window "
                     f"limit.")
    else:
        lines.append(f"Stage 1 outcome: candidate threshold = realized cost <= "
                     f"{stage1['candidate_threshold']:.6f}, set by "
                     f"{len(stage1['setter_branches'])} cell(s) "
                     f"({', '.join(stage1['setter_branches'])}); "
                     f"{binding['pig_removed_cells']}/{binding['cells']} frozen cells record "
                     f"replay-side pig removal (their costs span "
                     f"[{stage1['pig_removed_cost_min']:.3f}, "
                     f"{stage1['pig_removed_cost_max']:.3f}]); "
                     f"{binding['right_censored_cells']}/{binding['cells']} replay segments "
                     f"are right-censored at the window limit.")
    lines.append("")
    stage2 = binding["stage2"]
    if stage2["observed_agreement"] is None:
        lines.append("Stage 2 outcome: not evaluated (no candidate threshold).")
    else:
        lines.append(f"Stage 2 outcome: observed agreement of the candidate predicate with "
                     f"the replay engine evidence {stage2['observed_agreement']:.4f} "
                     f"({stage2['false_success_cells']} false-success cells, "
                     f"{stage2['missed_removal_cells']} missed-removal cells) vs the floor "
                     f"{stage2['agreement_floor']:.2f} and the always-failure baseline "
                     f"{stage2['always_failure_agreement']:.4f}; "
                     f"passed={stage2['passed']}.")
    lines.append("")
    lines.append("Declared divergence: " +
                 result["definitions"]["success_predicate"]["declared_divergence"])
    lines.append("")
    closed = result["pilot_divergence"]
    agreement = ("n/a" if closed["agreement_fraction"] is None
                 else f"{closed['agreement_fraction']:.4f}")
    lines.append(f"Closed-loop divergence of the stage-1 candidate predicate over the #80 "
                 f"pilot ({closed['valid_cells']} valid cells, {closed['typed_failure_cells']} "
                 f"typed): agreement {agreement} "
                 f"({closed['predicate_success_and_gameplay_success']} both-success, "
                 f"{closed['predicate_success_and_gameplay_failure']} predicate-only, "
                 f"{closed['predicate_failure_and_gameplay_success']} gameplay-only, "
                 f"{closed['predicate_failure_and_gameplay_failure']} both-failure).")
    lines.append("")
    if bound:
        lines.append(f"Predicate threshold: realized t=600 end-of-window replay cost <= "
                     f"{binding['threshold']:.6f}.")
        lines.append("")
    else:
        lines.append(f"Blocker: {binding['blocker']}")
        lines.append("")
    lines.append("## Protocol note (future pilots only)")
    lines.append("")
    lines.append(result["definitions"]["protocol_note"]["note"])
    lines.append("")
    compute = result["compute"]
    lines.append("## Compute accounting")
    lines.append("")
    lines.append(f"GPU {compute['gpu_seconds_elapsed']:.3f} s of the "
                 f"{compute['gpu_allowance_seconds']} s allowance (0.1 GPU-h); wall "
                 f"{compute['wall_seconds_elapsed']:.1f} s of {compute['wall_cap_seconds']} s; "
                 f"decision frames parsed {compute['decision_frames']}; parser calls "
                 f"{compute['parser_calls']}; predictor loads {compute['predictor_loads']}; "
                 f"transition calls {compute['transition_calls']}; linear MACs "
                 f"{compute['linear_macs']}; model scoring slots {compute['candidate_slots']} "
                 f"state x candidate cells across system-seeds.")
    lines.append("")
    lines.append("## Predicate-free descriptive tables")
    lines.append("")
    lines.append("### Per-state best candidate by realized replay cost (sorted by state)")
    lines.append("")
    lines.append("| State | Family | Member | Admissible | Best ordinal | Best branch | Best realized cost | Oracle indicator |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in result["per_state_best_candidates"]:
        best = row["best_candidate"]
        lines.append(f"| {row['state']} | {row['generator_family']} | {row['source_member']} | "
                     f"{row['admissible_candidates']} | {best['ordinal']} | "
                     f"{best['branch_identity']} | {best['realized_count_cost']:.3f} | "
                     f"{fmt(row.get('oracle_indicator'))} |")
    lines.append("")
    lines.append("### System-vs-oracle cost gap (per seed; DESCRIPTIVE)")
    lines.append("")
    cost_gap = result.get("system_cost_gap")
    if cost_gap is None:
        lines.append("Not available: the evaluation stopped before scoring "
                     f"({result.get('stopExplanation', 'unknown stop')}).")
        lines.append("")
    else:
        lines.append("| System | Seed | States scored | Typed failures | Predicate-success states | "
                     "Mean chosen cost | Mean oracle-best cost | Mean gap cost |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for system in SYSTEMS:
            entry = cost_gap["per_system"][system]
            for seed in SEEDS:
                row = entry["per_seed"][str(seed)]
                lines.append(f"| {system} | {seed} | {row['states_scored']} | "
                             f"{row['typed_decision_failures']} | "
                             f"{fmt(row['predicate_success_states'])} | "
                             f"{fmt(row['mean_chosen_realized_count_cost'], 3)} | "
                             f"{fmt(row['mean_oracle_best_realized_count_cost'], 3)} | "
                             f"{fmt(row['mean_gap_cost'], 3)} |")
        lines.append("")
    lines.append("### Threshold sensitivity (pre-declared grid; DESCRIPTIVE only)")
    lines.append("")
    lines.append(f"| Threshold | Cells meeting the predicate (of {binding['cells']}) |")
    lines.append("| --- | --- |")
    for row in result["threshold_sensitivity"]:
        lines.append(f"| {row['threshold']:.0f} | {row['cells_meeting_predicate']} |")
    lines.append("")
    dispositions = result.get("dispositions")
    if dispositions is None:
        lines.append("## Dispositions")
        lines.append("")
        stop_text = result.get("stopExplanation", "unknown stop")
        lines.append(f"Evaluation incomplete ({stop_text}): both questions "
                     f"disposition as {result['ticket_disposition']}; no tables were "
                     f"computed beyond the frozen evidence.")
        lines.append("")
    elif not bound:
        lines.append("## Dispositions")
        lines.append("")
        lines.append(f"- Q1 (oracle ceiling nonzero): "
                     f"{dispositions['q1_oracle_ceiling_nonzero']} - the oracle ceiling is "
                     f"not measurable without a defensible success predicate.")
        lines.append(f"- Q2 (motivated follow-up): "
                     f"{dispositions['q2_disposition']} - no branch can be typed from an "
                     f"unmeasurable ceiling; the predicate-free tables above are the "
                     f"descriptive input any follow-up must start from.")
        lines.append("")
        lines.append(f"Ticket disposition: {result['ticket_disposition']} "
                     "(tokens: supported / not_supported_by_this_experiment / "
                     "readiness_or_precision_insufficient). The blocked Phase-0 binding is "
                     "a valid terminal outcome; nothing was forced.")
        lines.append("")
    else:
        estimands = result["estimands"]
        prevalence = estimands["oracle_prevalence"]
        interval = prevalence["descriptive_95_percent_interval"]
        lines.append("## Q1: oracle ceiling over the frozen 24-state membership (DESCRIPTIVE)")
        lines.append("")
        lines.append(f"Oracle prevalence {prevalence['mean']:.4f} "
                     f"({prevalence['states_with_oracle_success']}/{prevalence['states']} "
                     f"states with at least one admissible candidate meeting the predicate); "
                     f"descriptive 95% interval [{interval[0]:.4f}, {interval[1]:.4f}] "
                     f"(paired bootstrap, "
                     f"{result['definitions']['uncertainty']['draws']} draws, seed "
                     f"{result['definitions']['uncertainty']['seed']}, resampling states).")
        lines.append("")
        lines.append("Paired oracle-minus-system indicator gaps (DESCRIPTIVE):")
        lines.append("")
        lines.append("| System | Mean gap | Descriptive 95% interval | Paired states |")
        lines.append("| --- | --- | --- | --- |")
        for contrast in estimands["system_gap_contrasts"]:
            descriptive = contrast["descriptive"]
            bounds = descriptive["descriptive_95_percent_interval"]
            lines.append(f"| {contrast['system']} | {descriptive['mean']:+.4f} | "
                         f"[{bounds[0]:+.4f}, {bounds[1]:+.4f}] | {contrast['paired_states']} |")
        lines.append("")
        dispositions = result["dispositions"]
        lines.append("## Dispositions")
        lines.append("")
        lines.append(f"- Q1 (oracle ceiling nonzero): {dispositions['q1_oracle_ceiling_nonzero']}; "
                     f"reading: {dispositions['q1_reading']}.")
        lines.append(f"- Q2 (motivated follow-up): {dispositions['q2_recommendation']} "
                     f"({dispositions['q2_disposition']}).")
        lines.append("")
        lines.append(f"Ticket disposition: {result['ticket_disposition']} "
                     "(tokens: supported / not_supported_by_this_experiment / "
                     "readiness_or_precision_insufficient).")
        lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ modes

def dry_run(args):
    eighty = load_eighty_plan()
    gate = load_gate()
    states = eighty["states"]
    members = sorted({state["source_member"] for state in states})
    outcomes = load_outcomes(members)
    cells = state_cells(states, outcomes)
    cached = [member for member in members if evidence_path(member).is_file()]
    plan_file = OUTPUT / "plan.json"
    binding = read(plan_file)["binding"] if plan_file.is_file() else None
    log(f"no-write dry-run: states={len(states)} members={len(members)} "
        f"cells={len(cells)} (expected {EXPECTED_CELLS}) systems={len(SYSTEMS)} "
        f"seeds={len(SEEDS)} model_scoring_slots={len(MODEL_SYSTEMS) * len(SEEDS) * len(cells)} "
        f"pilot_cells={gate['cells']} evidence_members_cached={len(cached)}/{len(members)} "
        f"plan_frozen={plan_file.is_file()} binding_status="
        f"{binding['status'] if binding else 'not-yet-frozen'} "
        f"gpu_allowance_s={GPU_ALLOWANCE_SECONDS} wall_cap_s={WALL_CAP_SECONDS}")
    if binding is not None:
        stage1 = binding["stage1"]
        if binding["status"] == "bound":
            log(f"frozen predicate: cost <= {binding['threshold']:.6f} "
                f"({binding['pig_removed_cells']}/{binding['cells']} cells pig-removed)")
        elif stage1["candidate_threshold"] is None:
            log("frozen binding BLOCKED: no replay-side pig removal exists in the "
                "frozen cells, so no candidate threshold exists")
        else:
            log(f"frozen binding BLOCKED: stage-1 candidate "
                f"{stage1['candidate_threshold']:.6f} failed the stage-2 gate "
                f"({binding['stage2']['observed_agreement']:.4f} < "
                f"{binding['stage2']['agreement_floor']}); publishing the blocker")
    return 0


def smoke(args):
    """Bounded real scoring of one member; production tables are untouched."""
    began = time.monotonic()
    plan_file = OUTPUT / "plan.json"
    if plan_file.is_file():
        plan = check_plan_bound(load_plan())
        binding = plan["binding"]
        members = [plan["membership"]["members"][0]]
        states = plan["membership"]["states"]
    else:
        log("plan not frozen yet: smoke derives this member's evidence fresh "
            "(smoke-local, never written to production)")
        eighty = load_eighty_plan()
        members = sorted({state["source_member"] for state in eighty["states"]})[:1]
        states = eighty["states"]
        binding = None
        plan = None
    member = members[0]
    state = next(s for s in states if s["source_member"] == member)
    member_outcomes = load_outcomes([member])[member]
    cost_by_branch = {row["branch_identity"]: row["realized_count_cost"]
                      for row in member_outcomes["candidates"]}
    # Fresh evidence re-derivation for the member's lowest and highest ordinal
    # branches (CPU native-trace scan; deliberately OUTSIDE the shared GPU
    # flock), compared against the frozen rows when a plan exists.
    fresh = {}
    for item in (state["inventory"][0], state["inventory"][-1]):
        _, _, entry = _scan_branch((item["branch_identity"], member,
                                    item["ordinal"],
                                    cost_by_branch[item["branch_identity"]]))
        fresh[item["branch_identity"]] = entry
    checks = {"carrier_236": None,
              "prediction_rows": None,
              "excluded_rows": None,
              "evidence_branches_rederived": sorted(fresh),
              "evidence_matches": None}
    if plan is not None:
        stored = {row["branch_identity"]: row for row in plan["evidence_rows"]
                  if row["member_identity"] == member}
        checks["evidence_matches"] = all(
            stored[branch]["pig_removed"] == entry["interaction_coverage"]["pig_removed"]
            and stored[branch]["censored"] == entry["censored"]
            for branch, entry in fresh.items())
        if not checks["evidence_matches"]:
            raise ValueError("smoke re-derivation differs from the frozen evidence")
    lock = gpu_lock()
    try:
        adapter = old.repair.load_repaired_adapter(old.repair.ROOT, args.device)
        vocabulary = read(DYNAMICS / "plan.json")["contract"]["vocabulary"]
        objective = TaskObjective.from_vocabulary(vocabulary)
        carrier = decision_carrier(states, member, adapter, args.device)
        if len(carrier["carrier"]) != 236:
            raise ValueError("smoke carrier is not a 236-vector")
        checks["carrier_236"] = True
        seed = SEEDS[0]
        system = MODEL_SYSTEMS[0]
        arm, pair = SYSTEM_ARMS[system]
        smoke_plan_dict = plan if plan is not None else {
            "dynamics": {"checkpoints": checkpoint_bindings()[0]}}
        model = load_frozen_predictor(smoke_plan_dict, seed, arm, args.device)
        predictions = rollout_predictions(smoke_plan_dict, states, member, system,
                                          seed, model, pair, objective, args.device,
                                          carrier)
        del model
        checks["prediction_rows"] = len(predictions["rows"])
        checks["excluded_rows"] = sum(1 for row in predictions["rows"] if row["excluded"])
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    result = {"schema": "issue_82_smoke_v1", "production_evidence": False,
              "member": member,
              "binding_status": binding["status"] if binding else "unfrozen-smoke-local",
              "checks": checks, "wall_seconds": time.monotonic() - began}
    write(OUTPUT / "smoke.json", result)
    log(f"smoke complete member={member} rows={checks['prediction_rows']} "
        f"wall={result['wall_seconds']:.1f}s; production artifacts untouched")
    return 0


def prepare(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_file = OUTPUT / "plan.json"
    if plan_file.exists():
        try:
            check_plan_bound(read(plan_file))
            log("frozen plan already bound and current; nothing to do")
            return 0
        except ValueError as error:
            if (OUTPUT / "evaluation.json").is_file():
                raise ValueError(
                    "the frozen issue-82 plan changed after evaluation started; "
                    "audit and version instead of re-freezing") from error
            superseded = plan_file.with_name(
                "plan.json.superseded-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
            plan_file.replace(superseded)
            log(f"stale pre-evaluation freeze versioned as {superseded.name}; re-freezing")
    plan = make_plan()
    write(plan_file, plan)
    binding = plan["binding"]
    if binding["status"] == "bound":
        log(f"issue-82 protocol frozen: predicate bound at cost <= "
            f"{binding['threshold']:.6f} "
            f"({binding['pig_removed_cells']}/{binding['cells']} cells pig-removed, "
            f"{binding['right_censored_cells']} right-censored); "
            f"cells={len(plan['cells'])}; no outcome computed")
    else:
        stage2 = binding["stage2"]
        log(f"issue-82 protocol frozen with a BLOCKED Phase-0 binding: "
            f"stage-1 candidate {binding['stage1']['candidate_threshold']:.6f} failed "
            f"the stage-2 gate (agreement {stage2['observed_agreement']:.4f} < "
            f"{stage2['agreement_floor']}); the blocker is the deliverable")
    return 0


def run_evaluation(args):
    plan = load_plan()
    check_plan_bound(plan)
    if (OUTPUT / "evaluation.json").is_file():
        evaluation = read(OUTPUT / "evaluation.json")
        if (evaluation.get("plan_identity") == plan["identity"]
                and evaluation.get("predicate_bound")
                is (plan["binding"]["status"] == "bound")
                and evaluation.get("complete") is True):
            log("evaluation already complete and bound to the frozen plan; "
                "delete evaluation.json to force a re-score")
            return 0
        log("a previous evaluation is incomplete; resuming from the verified "
            "caches (no outcome-conditioned replacement)")
    if plan["binding"]["status"] != "bound":
        log("Phase-0 binding is BLOCKED: the success-predicate estimands are not "
            "measurable; scoring runs anyway because the cost-gap tables are "
            "predicate-free and the dispositions stay readiness tokens")
    lock = gpu_lock()
    try:
        allowance = score(plan, args)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    allowance.check_artifact_cap(OUTPUT)
    evaluation = build_evaluation(plan, allowance)
    write(OUTPUT / "evaluation.json", evaluation)
    compute = evaluation["compute"]
    log(f"evaluation complete={evaluation['complete']} stop={evaluation['stop']} "
        f"predicate_bound={evaluation['predicate_bound']} "
        f"gpu={compute['gpu_seconds_elapsed']:.3f}s "
        f"wall={compute['wall_seconds_elapsed']:.1f}s "
        f"frames={compute['decision_frames']} "
        f"transition_calls={compute['transition_calls']} macs={compute['linear_macs']}")
    return 0 if evaluation["complete"] else 2


def publish(args):
    plan = load_plan()
    check_plan_bound(plan)
    result = publication(plan)
    write(OUTPUT / "summary.json", result)
    (OUTPUT / "comparisons.csv").write_text(comparisons_csv(result))
    (OUTPUT / "findings.md").write_text(findings_md(result))
    log(f"published evaluation_complete={result.get('evaluation_complete')} "
        f"ticket_disposition={result.get('ticket_disposition')}")
    return 0


def validate(args):
    plan = load_plan()
    check_plan_bound(plan)
    result = publication(plan)
    problems = []
    if result != read(OUTPUT / "summary.json"):
        problems.append("summary.json")
    if comparisons_csv(result).encode("utf-8") != (OUTPUT / "comparisons.csv").read_bytes():
        problems.append("comparisons.csv")
    if findings_md(result) != (OUTPUT / "findings.md").read_text():
        problems.append("findings.md")
    evaluation = read(OUTPUT / "evaluation.json")
    if (evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("plan_identity") != plan["identity"]
            or evaluation.get("predicate_bound")
            is not (plan["binding"]["status"] == "bound")
            or evaluation.get("complete") is not True):
        problems.append("evaluation.json")
    if problems:
        raise ValueError("published issue-82 artifacts differ from bound source "
                         f"evidence: {problems}")
    log("exact saved-evidence validation passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "run-evaluation", "publish",
                 "validate"):
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
        if args.run_evaluation:
            return run_evaluation(args)
        if args.publish:
            return publish(args)
        return validate(args)
    except (ValueError, OSError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
