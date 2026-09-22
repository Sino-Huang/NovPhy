"""Issue-89 ADD-EXP: closed-loop oracle-ceiling completion pass — the 55
typed-failure slots of #87 + unmeasured-state resolution (post-#87).

Bounded completion measurement (development/exposed lineages only; zero fresh
captures) under a NEW frozen protocol whose full protocol (slot inventory,
cold-start mitigation, unreachability branch, disposition mapping) is frozen
BEFORE its scoring run:

- Membership: exactly the 55 typed-failure oracle slot identities of #87's
  terminal ledger (51 connect/barrier timeout slots on measured states + the 4
  slots of issue-77-n1-005-a07), bound read-only by sha256 to #87's terminal
  plan/summary/ledger; same oracle seed 20260908, same inherited #85
  detection rule, same candidate actions — ONLY the execution infrastructure
  changes. #87's executed cells are NEVER re-executed here.
- Cold-start mitigation (protocol content, frozen): serialized/staggered
  engine cold starts (a new worker is dispatched no sooner than
  START_STAGGER_SECONDS after the previous dispatch, so concurrent engine
  cold starts cannot collide), per-worker port warm-up probes before the
  first RPC, and extended connect/readiness timeouts.
- Unreachability branch (frozen up front): if any issue-77-n1-005-a07 slot
  fails again with the sealed-frame stability signature, the state is
  declared typed-unmeasurable (stability evidence + media published) and the
  ceiling questions are dispositioned over the 23 measured states with the
  exclusion declared.
- Analysis: per-slot verdicts joined side-by-side to #87's published ledger
  (never overwriting); per-state oracle indicators updated; pooled prevalence
  with a descriptive interval over states; explicit comparison row to #87's
  published partial ceiling.
- Caps (global, frozen): <= 8 worker-hours wall, <= 0.5 GPU-hours active
  decision work (this ticket executes no decision pass; the cap is a guard).
  Exceeding stops with readiness_or_precision_insufficient.

Scientific questions (descriptive; disposition tokens only:
supported / not_supported_by_this_experiment / readiness_or_precision_insufficient):
- Q1 completion: engine-truth first-shot-success verdicts for the 55
  typed-failure oracle slots of #87's frozen inventory, each either executed
  with a channel verdict or covered by the frozen typed-unmeasurable
  declaration.
- Q2 ceiling re-estimate: per-state oracle indicators and pooled prevalence
  over all 24 states (or 23 measured states plus one typed-unmeasurable
  declaration), with a descriptive interval over states.

Hard constraints honored: #87's terminal artifacts are READ-ONLY inputs
(sha256-bound); development/exposed lineages ONLY; zero fresh captures;
#64/#65/#76 untouched; no system decision pass (the ranking cells were fully
measured in #87); no multi-shot, adaptation, or complete-gameplay claim;
negative/null results are reported.

Exact validation command:
  python -u -m scripts.run_closed_loop_oracle_completion --validate
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import numpy as np
import torch

from scripts import run_engine_outcome_reactive_diagnostic as e85
from scripts import run_issue_77_n1_train as train77
from scripts.issue_76_censored_episode import capture_segment
from scripts.issue_76_display_start import start_display
from scripts.issue_76_episode_capture import prepare_action
from scripts.issue_76_live_episode import files as live_files
from scripts.issue_76_native_outcomes import terminal_evidence
from scripts.issue_76_shared_history_capture import expose_history, read_history
from scripts.issue_76_shared_player_storage import clone_player
from scripts.observation_trace import MANIFEST_NAME, _plain_json
from scripts.process_lifecycle import start_isolated_worker
from scripts.run_issue_62_successor_cohort import (
    PILOT_AUDIT_FPS,
    PlayingMode,
    _encode_agent_frames_webm,
    _verify_webm_encoder,
    connect_with_retry,
    prepare_for_play,
    start_engine,
    stop_started_engine,
    terminate,
)
from scripts.run_issue_76_compatibility import process_rss, terminate_worker
from src.webui.bridge import ScienceBirdsBridge
from world_model.training import engine_outcome_reactive_diagnostic as engine

ROOT = train77.ROOT
EIGHTY_SEVEN = ROOT / ".local-artifacts/issue-87-closed-loop-oracle-v1"
EIGHTY_SEVEN_IDENTITY = "issue-87-closed-loop-oracle-v1"
OUTPUT = ROOT / ".local-artifacts/issue-89-oracle-completion-v1"
DATA = ROOT / "data/issue-89-oracle-completion"
SCHEMA_PLAN = "issue_89_oracle_completion_plan_v1"
SCHEMA_RECORD = "issue_89_oracle_completion_record_v1"
SCHEMA_LEDGER = "issue_89_oracle_completion_ledger_v1"
SCHEMA_REPORT = "issue_89_oracle_completion_report_v1"
SCHEMA_GALLERY = "issue_89_review_gallery_v1"
IDENTITY = "issue-89-oracle-completion-v1"

ORACLE_SEED = 20260908
FAMILIES = ("type010103", "type010105")
UNMEASURABLE_CANDIDATE_STATE = "issue-77-n1-005-a07"
STABILITY_SIGNATURE = "sealed pre-decision state/RGB advanced during readiness or hold"

DEV_VERSION = 1
TERMINAL_VERSION = 3
VERSION_STATUSES = {1: "development", 2: "development",
                    TERMINAL_VERSION: "terminal"}

WALL_CAP_SECONDS = 8 * 3600
GPU_CAP_SECONDS = 1800
WORKERS = 4
WORKER_RSS_MIB = 4096
AGGREGATE_RSS_MIB = 32768
ATTEMPT_SECONDS = 2400
MINIMUM_FREE_BYTES = 256 * 2 ** 30
ARTIFACT_BYTES = 150 * 2 ** 30
PORT_LOW, PORT_HIGH = 28700, 28999

# Frozen measurement instruments (byte-identical to #87: same anchor step,
# same shot window, same sealed-frame hold, same native stride).
DECISION_FIXED_STEP = 30000
SHOT_SECONDS = 180
INFERENCE_HOLD_SECONDS = 2
NATIVE_STRIDE = 50

# Frozen cold-start mitigation (protocol content; the ONLY element that may
# differ from #87). #87's 51 connect/barrier timeouts clustered in cold-start
# windows of the 4 concurrent engine workers; the terminal mitigation is
# serialized engine cold starts plus extended connect/readiness bounds.
# (plan-v1 additionally carried bare-TCP port warm-up probes; the plan-v1
# smoke showed the probes POISON the engine session - the jar registers an
# anonymous probe as an agent and assigns the game window to the dead probe
# client - so v2 removed them; v1 is retained verbatim with its changelog.)
START_STAGGER_SECONDS = 45
AGENT_SOCKET_SECONDS = 600
AGENT_CONNECT_DEADLINE_SECONDS = 600
PHYSICS_SOCKET_SECONDS = 120
PREPARE_FOR_PLAY_SECONDS = 300
HISTORY_READY_SECONDS = 600

BOOTSTRAP_DRAWS = engine.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = engine.BOOTSTRAP_SEED

# Bounded rendered smoke (development version; EXPLORATORY; frozen before the
# smoke runs): one former-timeout slot and one issue-77-n1-005-a07 slot. The
# expectations are STRUCTURAL harness validations, not outcome predictions:
# the former-timeout slot must now reach an engine-truth channel verdict
# under the frozen mitigation (this gates the terminal freeze); the 005-a07
# slot must either reach a channel verdict or fail with the frozen
# sealed-frame stability signature (both are protocol-covered paths).
SMOKE_SPEC = (
    ("issue-77-n1-001-a00", 0, "channel_verdict"),
    (UNMEASURABLE_CANDIDATE_STATE, 1, "channel_verdict_or_stability_signature"),
)

FILES = (
    "scripts/run_closed_loop_oracle_completion.py",
    "world_model/training/engine_outcome_reactive_diagnostic.py",
    "scripts/run_engine_outcome_reactive_diagnostic.py",
)

DETECTION_RULE_CITATION = (
    "inherited VERBATIM from the published issue-85 engine-truth outcome "
    "channel (world_model/training/engine_outcome_reactive_diagnostic.py: "
    "CHANNEL_DETECTION_RULE; pig_removed/entity_death/entity_destroyed native "
    "macro events with runtime:pig: participants + lifecycle confirmatory "
    "leg), exactly as inherited by the frozen issue-87 terminal plan; the "
    "channel was verified with media review in issue-85 Phase A "
    "(issue-85 channel-verification.json: verified=true) and is NOT amended "
    "by this ticket")


def log(message):
    print(f"[issue-89-oracle-completion] {message}", flush=True)


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


def bytes_identity(payload):
    return "sha256:" + sha256(payload).hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def smoke_identity(state, ordinal, version):
    return f"smoke-v{version}--{state}--a{ordinal:02d}"


def reserve_ports(count=3):
    """Reserve distinct ports inside the issue-89 range (28700-28999)."""
    sockets, ports = [], []
    for candidate in range(PORT_LOW, PORT_HIGH + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", candidate))
        except OSError:
            sock.close()
            continue
        sockets.append(sock)
        ports.append(candidate)
        if len(ports) == count:
            break
    if len(ports) < count:
        for sock in sockets:
            sock.close()
        raise ValueError(f"no free ports in [{PORT_LOW}, {PORT_HIGH}] for this cell")
    for sock in sockets:
        sock.close()
    return sorted(ports)




# ---------------------------------------------------------------- slot membership

def issue87_terminal_plan():
    """#87's frozen terminal plan, a READ-ONLY input bound by identity."""
    path = EIGHTY_SEVEN / "plan.json"
    plan = read(path)
    if (plan.get("identity") != EIGHTY_SEVEN_IDENTITY
            or plan.get("terminal") is not True
            or plan.get("version") != 3):
        raise ValueError("requires the frozen terminal issue-87 plan (version 3)")
    return plan


def issue87_ledger():
    """#87's terminal execution ledger, a READ-ONLY input bound by identity."""
    ledger = read(EIGHTY_SEVEN / "ledger.json")
    if (ledger.get("plan_identity") != EIGHTY_SEVEN_IDENTITY
            or ledger.get("status") != "complete"):
        raise ValueError("requires the complete terminal issue-87 ledger")
    return ledger


def issue87_summary():
    """#87's published summary, a READ-ONLY input bound by identity."""
    summary = read(EIGHTY_SEVEN / "summary.json")
    if summary.get("identity") != EIGHTY_SEVEN_IDENTITY:
        raise ValueError("requires the published issue-87 summary")
    return summary


def completion_cells(plan87, ledger87):
    """Exactly the 55 typed-failure oracle slots of #87's terminal ledger.

    Slot identities and candidate actions are taken VERBATIM from #87's
    frozen terminal plan; the membership rule (non-null failure in #87's
    complete ledger) is frozen and deterministic."""
    cells87 = {cell["identity"]: cell for cell in plan87["oracle_cells"]}
    failed_identities = sorted(
        identity for identity, entry in ledger87["cells"].items()
        if identity.startswith("oracle--") and entry.get("failure") is not None)
    if len(failed_identities) != 55:
        raise ValueError(f"issue-87 terminal ledger carries "
                         f"{len(failed_identities)} typed-failure oracle slots, not 55")
    cells = []
    for identity in failed_identities:
        cell87 = cells87.get(identity)
        if cell87 is None:
            raise ValueError(f"typed-failure slot {identity} is missing from "
                             "the frozen issue-87 terminal plan")
        cells.append({
            "identity": identity,
            "state": cell87["state"], "ordinal": cell87["ordinal"],
            "branch_identity": cell87["branch_identity"],
            "action": dict(cell87["action"]),
            "engine_seed": cell87["engine_seed"], "seed": ORACLE_SEED,
            "frozen_replay_count_cost": cell87["frozen_replay_count_cost"],
            "issue87_typed_failure": ledger87["cells"][identity]["failure"],
        })
    on_unmeasurable = [cell["identity"] for cell in cells
                       if cell["state"] == UNMEASURABLE_CANDIDATE_STATE]
    if len(on_unmeasurable) != 4 or len(cells) - len(on_unmeasurable) != 51:
        raise ValueError("the completion membership is not 51 measured-state "
                         f"timeout slots + 4 {UNMEASURABLE_CANDIDATE_STATE} slots")
    return cells


def smoke_cells(cells, version):
    resolved = []
    by_key = {(cell["state"], cell["ordinal"]): cell for cell in cells}
    for state_identity, ordinal, expectation in SMOKE_SPEC:
        cell = by_key.get((state_identity, ordinal))
        if cell is None:
            raise ValueError(f"smoke slot ({state_identity}, a{ordinal:02d}) is "
                             "not in the completion membership")
        resolved.append(dict(cell, identity=smoke_identity(state_identity, ordinal, version),
                             completion_identity=cell["identity"],
                             expectation=expectation,
                             expectation_source=("frozen structural harness "
                                                 "validation, not an outcome "
                                                 "prediction")))
    return resolved


def cold_start_mitigation_definition(version):
    """The frozen mitigation text, per protocol version. plan-v1 carried
    bare-TCP port warm-up probes; the plan-v1 smoke showed they poison the
    engine session (the jar registers an anonymous probe as an agent and
    assigns the game window to the dead probe client, starving the real
    bridge), so v2 removed them. v1's text is retained VERBATIM (with its
    literals inlined) so check_plan_bound recomputes v1 byte-identically."""
    if version == 1:
        return {
            "rule": (
                "protocol content frozen before the scoring run: (1) "
                "serialized/staggered engine cold starts - the supervisor "
                "dispatches a new worker no sooner than "
                "45s after the previous dispatch, so "
                "concurrent engine cold starts cannot collide; (2) per-worker "
                "port warm-up probes (300s) on the "
                "agent/game/physics ports plus a "
                "5s settle before the first RPC; (3) "
                "extended connect/readiness bounds: agent socket "
                "600s, agent connect deadline "
                "600s, physics socket "
                "120s, prepare-for-play "
                "300s, native decision barrier "
                "readiness 600s, attempt wall "
                "2400s"),
            "motivation": (
                "the issue-87 ledger shows 52 of 55 typed failures are "
                "connect/barrier timeouts clustered in cold-start windows of "
                "the 4 concurrent engine workers; the mitigation changes "
                "execution infrastructure only, never the measurement "
                "instruments"),
        }
    return {
        "rule": (
            "protocol content frozen before the scoring run: (1) "
            "serialized/staggered engine cold starts - the supervisor "
            f"dispatches a new worker no sooner than "
            f"{START_STAGGER_SECONDS}s after the previous dispatch, so "
            "concurrent engine cold starts cannot collide; (2) "
            "extended connect/readiness bounds: agent socket "
            f"{AGENT_SOCKET_SECONDS}s, agent connect deadline "
            f"{AGENT_CONNECT_DEADLINE_SECONDS}s, physics socket "
            f"{PHYSICS_SOCKET_SECONDS}s, prepare-for-play "
            f"{PREPARE_FOR_PLAY_SECONDS}s, native decision barrier "
            f"readiness {HISTORY_READY_SECONDS}s, attempt wall "
            f"{ATTEMPT_SECONDS}s"),
        "motivation": (
            "the issue-87 ledger shows 52 of 55 typed failures are "
            "connect/barrier timeouts clustered in cold-start windows of "
            "the 4 concurrent engine workers; the mitigation changes "
            "execution infrastructure only, never the measurement "
            "instruments. The plan-v1 bare-TCP port warm-up probes were "
            "REMOVED in v2 after the plan-v1 smoke showed they poison the "
            "engine session: the jar registers an anonymous probe as an "
            "agent and assigns the game window to the dead probe client, "
            "starving the real bridge (both v1 smoke cells timed out at the "
            "600s socket bound with '2 agent/s has been connected' in the "
            "engine log)"),
    }


def definitions(version):
    return {
        "protocol_policy": (
            "bounded completion ticket: the FULL protocol (slot inventory "
            "bound to the issue-87 terminal ledger, cold-start mitigation, "
            "unreachability branch, disposition mapping, caps) is frozen "
            "BEFORE its scoring run; plan-v1 (development) is frozen before "
            "the bounded rendered smoke; the terminal plan-v2 is frozen with "
            "actual numeric values BEFORE its scoring run and executed "
            "exactly once; development outcomes are labelled EXPLORATORY and "
            "may not be cited as answers"),
        "membership_rule": (
            "exactly the 55 typed-failure oracle slot identities of the "
            "issue-87 terminal ledger (51 connect/barrier timeout slots on "
            "measured states + the 4 slots of issue-77-n1-005-a07), with "
            "candidate actions, engine seeds and frozen replay count costs "
            "VERBATIM from the issue-87 frozen terminal plan; same oracle "
            "seed 20260908; the issue-87 artifacts are READ-ONLY inputs "
            "bound by sha256; issue-87's executed cells are NEVER "
            "re-executed; only the execution infrastructure changes"),
        "oracle_seed": ORACLE_SEED,
        "completion_pass": (
            "every scheduled completion slot is executed exactly once in the "
            "real rendered engine from the state's frozen initial condition "
            f"(source-member decision state at native fixed step "
            f"{DECISION_FIXED_STEP}), the candidate's frozen action is "
            "executed as the single first shot, and engine-truth first-shot "
            "success is read with the inherited #85 detection rule; typed "
            "failures are retained and carry no success value"),
        "cold_start_mitigation": cold_start_mitigation_definition(version),
        "unreachability_branch": {
            "rule": (
                f"frozen up front: if any {UNMEASURABLE_CANDIDATE_STATE} slot "
                f"fails again with the sealed-frame stability signature "
                f"('{STABILITY_SIGNATURE}'), the state is declared "
                "typed-unmeasurable (the level physics is not static at the "
                f"sealed decision step {DECISION_FIXED_STEP}, so verdicts "
                "cannot bind to the sealed pre-decision state); the "
                "stability evidence (paused before/after frames + records) "
                "and media are published, and the ceiling questions are "
                "dispositioned over the 23 measured states with the "
                "exclusion declared; the declaration triggers on the "
                "signature REGARDLESS of the other slots' verdicts"),
            "state": UNMEASURABLE_CANDIDATE_STATE,
            "signature": STABILITY_SIGNATURE,
        },
        "engine_channel": {
            "detection_rule": engine.CHANNEL_DETECTION_RULE,
            "failure_modes": engine.CHANNEL_FAILURE_MODES,
            "inheritance": DETECTION_RULE_CITATION,
            "first_shot_success": (
                "the executed shot segment's engine-side channel verdict "
                "pig_removed; typed failures carry no success value"),
            "shots_to_success": (
                "single-shot protocol: 1 when the first shot succeeds, "
                "otherwise null; no multi-shot competence claim"),
        },
        "ceiling_reestimate_rule": (
            "per-state oracle indicators are UPDATED by joining this pass's "
            "channel verdicts to the issue-87 published per-state ledger "
            "rows (side-by-side, never overwriting): a state is measured "
            "when at least one of its candidates has a channel verdict "
            "(issue-87 or this pass) and its indicator is 1 when at least "
            "one executed candidate succeeded; issue-77-n1-005-a07 is either "
            "measured by this pass or excluded under the frozen "
            "typed-unmeasurable declaration; pooled prevalence over the "
            "measured states with a descriptive paired-bootstrap interval; "
            "an explicit comparison row to issue-87's published partial "
            "ceiling is reported"),
        "questions": {
            "q1_slot_completion": (
                "engine-truth first-shot-success verdicts for the 55 "
                "typed-failure oracle slots of the issue-87 frozen "
                "inventory, each either executed with a channel verdict or "
                "covered by the frozen typed-unmeasurable declaration"),
            "q2_ceiling_reestimate": (
                "per-state oracle indicators and pooled prevalence over all "
                "24 states (or 23 measured states plus one "
                "typed-unmeasurable declaration), with a descriptive "
                "interval over states; does the nonzero-ceiling question "
                "now reach supported under the frozen disposition mapping"),
        },
        "disposition_rule": {
            "q1_slot_completion": {
                "supported": (
                    "the completion ledger is complete AND every scheduled "
                    "slot carries an engine-truth channel verdict OR is a "
                    f"{UNMEASURABLE_CANDIDATE_STATE} slot covered by the "
                    "frozen typed-unmeasurable declaration (zero residual "
                    "typed terminal failures outside the declaration)"),
                "not_supported_by_this_experiment": (
                    "the completion pass ran to completion (no cap stop) "
                    "but residual typed terminal failures remain outside "
                    "the frozen typed-unmeasurable declaration: the "
                    "completion objective was not achieved by this "
                    "experiment; the failures are reported"),
                "readiness_or_precision_insufficient": (
                    "a cap stop occurred or the completion inventory is "
                    "incomplete"),
            },
            "q2_ceiling_reestimate": {
                "supported": (
                    "every one of the 24 states is measured OR (23 states "
                    "measured AND issue-77-n1-005-a07 excluded under the "
                    "frozen typed-unmeasurable declaration) AND the pooled "
                    "prevalence over the measured states is > 0 AND the "
                    "descriptive 95% paired bootstrap interval over the "
                    "per-state indicators excludes 0 (lower bound > 0; the "
                    "interval-exclusion rule is the pre-declared practical "
                    "margin): the closed-loop oracle ceiling is measurably "
                    "nonzero"),
                "not_supported_by_this_experiment": (
                    "complete measurement (as above) with pooled prevalence "
                    "= 0: no admissible candidate succeeded anywhere, a "
                    "measured state-difficulty floor (a valid null, "
                    "reported)"),
                "readiness_or_precision_insufficient": (
                    "prevalence > 0 but the descriptive interval contains 0 "
                    "(precision), or a state is unmeasured without the "
                    "frozen declaration, or the inventory is incomplete, or "
                    "a cap stop occurred (readiness)"),
            },
        },
        "uncertainty": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                        "unit": "state",
                        "note": ("descriptive paired bootstrap over state "
                                 "identities; the oracle pass uses the single "
                                 "frozen seed 20260908; cannot support a "
                                 "confirmatory claim")},
        "caps": {"wall_cap_seconds": WALL_CAP_SECONDS,
                 "gpu_cap_seconds": GPU_CAP_SECONDS,
                 "wall_scope": ("cumulative elapsed wall across ALL protocol "
                                "versions and phases (smoke + completion)"),
                 "gpu_scope": ("cumulative synchronized decision wall on "
                               "cuda; this ticket executes NO decision pass, "
                               "so the cap is a guard only"),
                 "stop": "readiness_or_precision_insufficient"},
        "compute_accounting": (
            "per completion cell: execution wall and engine wall; this "
            "ticket executes no predictor scoring, so decision compute is "
            "zero by construction"),
        "claim_boundary": (
            "bounded completion measurement on development/exposed lineages; "
            "zero fresh captures; no multi-shot, adaptation, zero-shot, or "
            "complete-gameplay claim; cannot reopen "
            "#64/#65/#72/#75/#15/#76; the #80/#82/#85/#87 published outcomes "
            "are read-only inputs, never re-run, amended, or reinterpreted; "
            "this ticket APPENDS the missing slots, it does not re-run #87; "
            "no system decision pass; descriptive intervals only"),
    }


def changelog_for(version, evidence=None):
    if version == DEV_VERSION:
        return [{
            "version": 1,
            "change": "initial protocol draft",
            "what": ("completion pass over exactly the 55 typed-failure "
                     "oracle slots of the issue-87 terminal ledger (slot "
                     "identities, actions, seed 20260908 and the inherited "
                     "#85 detection rule verbatim; only execution "
                     "infrastructure changes); frozen cold-start mitigation "
                     f"(staggered engine starts {START_STAGGER_SECONDS}s, "
                     "port warm-up probes 300s, extended "
                     "connect/readiness bounds); frozen unreachability "
                     "branch for issue-77-n1-005-a07; frozen disposition "
                     "mapping; bounded 2-cell rendered smoke (1 former-"
                     "timeout slot + 1 issue-77-n1-005-a07 slot); global "
                     f"caps {WALL_CAP_SECONDS}s wall / {GPU_CAP_SECONDS}s "
                     f"GPU; ports {PORT_LOW}+; {WORKERS} isolated workers"),
            "why": ("open the ticket under the frozen-before-scoring "
                    "policy: nothing is yet observed, so v1 fixes the "
                    "inherited measurement instruments, the mitigation, the "
                    "unreachability branch and the bounded smoke that "
                    "validates the harness before any long run"),
            "evidence": "none yet (initial freeze, before the smoke)",
        }]
    if version == 2:
        smoke = evidence or {}
        return [{
            "version": 2,
            "change": "development amendment after the plan-v1 rendered smoke: "
                      "cold-start mitigation corrected (probe removal only)",
            "what": (
                "the bare-TCP port warm-up probes of plan-v1 are REMOVED: "
                "the v1 smoke showed they poison the engine session - the "
                "jar registers an anonymous probe connection as an agent and "
                "assigns the game window to the dead probe client, starving "
                "the real bridge (both v1 smoke cells typed-failed with "
                "TimeoutError at the 600s socket bound, wall ~622s each; "
                "engine logs show '2 agent/s has been connected'). The "
                "mitigation is now: serialized/staggered engine cold starts "
                f"({START_STAGGER_SECONDS}s dispatch spacing) plus the "
                "extended connect/readiness bounds, byte-identical to v1. "
                "NO other protocol element changed: the 55-slot completion "
                "membership, oracle seed 20260908, the inherited #85 "
                "detection rule, the unreachability branch, the disposition "
                "mapping, workers, port range, and caps are byte-identical. "
                "The v2 smoke re-validates the corrected harness on the same "
                "2 frozen smoke slots before the terminal freeze. v1 is "
                "retained verbatim with its outcomes labelled EXPLORATORY."),
            "why": ("a frozen mitigation element was shown to be "
                    "self-defeating by the v1 smoke; correcting it during "
                    "development (before any terminal freeze) preserves the "
                    "freeze-before-scoring guarantee for the terminal "
                    "version"),
            "evidence": smoke.get("v1_summary", "development/v1 records"),
        }]
    if version == TERMINAL_VERSION:
        smoke = evidence or {}
        return [{
            "version": 3,
            "change": "terminal freeze after the plan-v2 rendered smoke",
            "what": (
                "NO protocol element changed relative to plan-v2: the 55-slot "
                "completion membership, oracle seed 20260908, the inherited "
                "#85 detection rule, the corrected cold-start mitigation "
                "(staggered starts + extended bounds, no probes), the "
                "unreachability branch, the disposition mapping, workers, "
                "port range, and caps are byte-identical. What CHANGES with "
                "v3 is the status: development -> terminal, so this version "
                "is frozen with actual numeric values BEFORE its own scoring "
                "run, is executed exactly once, and admits no further "
                "outcome-conditioned modification. Reviewed smoke evidence "
                "embedded: "
                + smoke.get("summary", "smoke records retained under development/v2")),
            "why": ("the v2 smoke validated the corrected oracle harness end "
                    "to end (real rendered execution under the frozen "
                    "mitigation, channel verdict or protocol-covered "
                    "stability signature, gallery media retention), so no "
                    "instrument change is needed and the terminal version "
                    "can be frozen before any terminal outcome exists"),
            "evidence": smoke.get("records_reference", "development/v2 records"),
        }]
    raise ValueError(f"no changelog defined for version {version}")


def make_plan(version, retained=None, evidence=None):
    """Build one protocol version. Freeze time (retained=None) hashes every
    bound input once; post-freeze recompute passes the retained plan and
    reuses its archival identities while re-deriving every scientific field."""
    if version not in VERSION_STATUSES:
        raise ValueError(f"unknown protocol version {version}")
    plan87 = issue87_terminal_plan()
    ledger87 = issue87_ledger()
    summary87 = issue87_summary()
    cells = completion_cells(plan87, ledger87)
    smoke = smoke_cells(cells, version)
    states = deepcopy(plan87["states"])
    if len(states) != 24:
        raise ValueError("the issue-87 terminal plan does not carry 24 states")
    player = Path(plan87["player_root"])
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")
    if retained is not None:
        if retained["version"] != version:
            raise ValueError("retained protocol version differs")
        binding = retained["issue87_binding"]
        frozen_at = retained["frozen_at_utc"]
        smoke_evidence = retained.get("smoke_evidence")
    else:
        binding = {
            "identity": EIGHTY_SEVEN_IDENTITY,
            "plan_sha256": file_identity(EIGHTY_SEVEN / "plan.json"),
            "summary_sha256": file_identity(EIGHTY_SEVEN / "summary.json"),
            "ledger_sha256": file_identity(EIGHTY_SEVEN / "ledger.json"),
            "terminal_version": 3,
            "typed_failure_slots": 55,
            "rule": ("READ-ONLY inputs; the issue-87 published outcomes are "
                     "inputs, never amended, reinterpreted, or re-executed"),
        }
        frozen_at = utc_now()
        smoke_evidence = evidence
    plan = {
        "schema": SCHEMA_PLAN, "identity": IDENTITY,
        "version": version,
        "status": VERSION_STATUSES[version],
        "terminal": version == TERMINAL_VERSION,
        "frozen_before_scoring_run": True,
        "frozen_at_utc": frozen_at,
        "changelog": changelog_for(version, evidence=smoke_evidence),
        "definitions": definitions(version),
        "issue87_binding": binding,
        "issue87_published_partial_ceiling": {
            "measured_states": summary87["oracle_pass"]["pooled"]["measured_states"],
            "ceiling_states": summary87["oracle_pass"]["pooled"]["ceiling_states"],
            "prevalence": summary87["oracle_pass"]["pooled"]["prevalence"],
            "descriptive": summary87["oracle_pass"]["pooled"]["descriptive"],
            "unmeasured_states": summary87["oracle_pass"]["pooled"]["unmeasured_states"],
        },
        "player_root": str(player),
        "states": states,
        "completion_cells": cells,
        "smoke_cells": smoke,
        "execution": {"workers": WORKERS,
                      "rendered": True, "nographics": False,
                      "isolated_process": True, "separate_ports_and_workdirs": True,
                      "port_range": [PORT_LOW, PORT_HIGH],
                      "worker_rss_mib": WORKER_RSS_MIB,
                      "aggregate_rss_mib": AGGREGATE_RSS_MIB,
                      "attempt_seconds": ATTEMPT_SECONDS,
                      "minimum_free_bytes": MINIMUM_FREE_BYTES,
                      "artifact_bytes": ARTIFACT_BYTES,
                      "decision_fixed_step": DECISION_FIXED_STEP,
                      "shot_seconds": SHOT_SECONDS,
                      "inference_hold_seconds": INFERENCE_HOLD_SECONDS,
                      "native_stride": NATIVE_STRIDE,
                      "start_stagger_seconds": START_STAGGER_SECONDS,
                      **({"port_probe_seconds": 300, "warmup_settle_seconds": 5}
                         if version == 1 else {}),
                      "agent_socket_seconds": AGENT_SOCKET_SECONDS,
                      "agent_connect_deadline_seconds": AGENT_CONNECT_DEADLINE_SECONDS,
                      "physics_socket_seconds": PHYSICS_SOCKET_SECONDS,
                      "prepare_for_play_seconds": PREPARE_FOR_PLAY_SECONDS,
                      "history_ready_seconds": HISTORY_READY_SECONDS,
                      "gpu_lock": None,
                      "gpu_lock_note": ("no cuda decision work in this "
                                        "ticket; the shared-GPU flock is not "
                                        "needed and is not taken"),
                      "technical_retries": 0},
        "smoke_evidence": smoke_evidence,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_text": {path: (ROOT / path).read_text() for path in FILES},
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }
    expected = (len(states), len(cells), len(smoke))
    if expected != (24, 55, 2):
        raise ValueError(f"frozen protocol inventory differs: {expected}")
    return plan


def protocols_dir():
    return OUTPUT / "protocols"


def plan_path(version):
    return protocols_dir() / f"plan-v{version}.json"


def load_plan():
    plan = read(OUTPUT / "plan.json")
    if plan.get("identity") != IDENTITY or not plan.get("terminal"):
        raise ValueError("requires the frozen terminal issue-89 plan at plan.json")
    return plan


def check_plan_bound(plan):
    """Validate the frozen plan against the live frozen inputs it binds.

    Recompute reuses the plan's retained archival identities (no new hashing
    pass) and re-derives every scientific field. After execution has started,
    a difference confined to "source_text" is the documented post-execution
    reporting/protocol-fix provision (parent-directed): the executed source
    snapshot is retained inside the plan itself, and live reporting/
    validation code may be corrected without touching any frozen scientific
    definition. Any other difference is a hard freeze violation."""
    current = make_plan(plan["version"], retained=plan)
    current["source_revision"] = plan["source_revision"]
    current["frozen_at_utc"] = plan["frozen_at_utc"]
    if plan != current:
        differing = sorted(key for key in set(plan) | set(current)
                           if plan.get(key) != current.get(key))
        non_source = [key for key in differing if key != "source_text"]
        if non_source:
            raise ValueError(f"frozen issue-89 plan fields changed: {non_source}; "
                             "preserve and version")
        execution_started = any((OUTPUT / "records").glob("*.json")) \
            if (OUTPUT / "records").is_dir() else False
        if not execution_started:
            raise ValueError("frozen issue-89 plan source changed before "
                             "execution; re-freeze with --prepare to version it")
        log("post-execution reporting/protocol fix detected in source_text; "
            "frozen scientific content unchanged; the executed source snapshot "
            "remains retained inside the frozen plan")
    return plan


def require_prepared(plan):
    """The bound inputs must exist and the read-only binding must verify."""
    binding = plan["issue87_binding"]
    for name, key in (("plan.json", "plan_sha256"),
                      ("summary.json", "summary_sha256"),
                      ("ledger.json", "ledger_sha256")):
        path = EIGHTY_SEVEN / name
        if not path.is_file():
            raise ValueError(f"missing bound issue-87 input {path}")
        if file_identity(path) != binding[key]:
            raise ValueError(f"bound issue-87 input {name} differs from the "
                             "frozen sha256 (READ-ONLY violation)")
    player = Path(plan["player_root"])
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")


# ------------------------------------------------------------- cell execution

def _plain_record(plan, cell, state):
    kind = "smoke" if cell["identity"].startswith("smoke") else "completion"
    return {"schema": SCHEMA_RECORD, "plan_identity": plan["identity"],
            "plan_version": plan["version"], "cell": cell,
            "cell_kind": kind,
            "state_identity": state["identity"],
            "seed": cell.get("seed", ORACLE_SEED),
            "issue87_typed_failure": cell.get("issue87_typed_failure"),
            "failure": None, "failure_kind": None,
            "decision_frame": None, "execution": None, "outcome": None,
            "engine_channel": None, "fresh_scenario_lineage": False,
            "issue_64_authorized": False}


class _ExposurePolicy:
    """Frame-counting placeholder policy (the choice is upstream/frozen)."""

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


def execute_completion_cell(payload):
    """One isolated, real-rendered completion execution under the frozen
    cold-start mitigation: the candidate's frozen action from the state's
    frozen initial condition; engine-truth outcome only (no predictor
    scoring, no parser, no cuda decision work)."""
    torch.set_num_threads(2)
    output = Path(payload["output"])
    plan = read(Path(payload.get("plan_path") or (output / "plan.json")))
    cell = payload["cell"]
    identity = cell["identity"]
    state = payload["state"]
    attempt = output / "attempts" / identity
    attempt.mkdir(parents=True)
    started = time.monotonic()
    record = _plain_record(plan, cell, state)
    bridge = physics = engine_process = display_process = None
    variables = ("DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
                 "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
                 "NOVPHY_ENVIRONMENT_SEED", "NOVPHY_NATIVE_DECISION_STEP")
    environment = {key: os.environ.get(key) for key in variables}
    try:
        member = state["execution_member"]
        inventory = state["inventory"]
        item = next(item for item in inventory if item["ordinal"] == cell["ordinal"])
        if item["branch_identity"] != cell["branch_identity"]:
            raise ValueError("scheduled candidate differs from the frozen inventory")
        game = attempt / "runtime"
        clone_player(Path(plan["player_root"]), game)
        live_files.install_level(game, member)
        _, scenario = live_files.materialize(member, member["template"], attempt / "authority")
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("completion member differs from its frozen scenario authority")
        display, display_process = start_display(attempt / "display.log")
        agent_port, game_port, physics_port = reserve_ports(3)
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
        bridge = connect_with_retry("127.0.0.1", agent_port,
                                    timeout=AGENT_SOCKET_SECONDS,
                                    deadline_seconds=AGENT_CONNECT_DEADLINE_SECONDS)
        physics = ScienceBirdsBridge("127.0.0.1", physics_port,
                                     timeout=PHYSICS_SOCKET_SECONDS)
        bridge.configure(760001, PlayingMode.TRAINING)
        bridge.set_speed(1)
        prepare_for_play(bridge, timeout=PREPARE_FOR_PLAY_SECONDS, poll_delay=.5)
        if bridge.get_current_level() != 1:
            raise ValueError("episode did not load its single assigned level")
        deadline = time.monotonic() + HISTORY_READY_SECONDS
        while not list(aligned.glob("decision-history-*/ready.json")):
            if time.monotonic() >= deadline:
                raise TimeoutError("native decision barrier readiness deadline; no retry")
            time.sleep(.1)
        _, rows = read_history(aligned, DECISION_FIXED_STEP)
        policy = _ExposurePolicy()
        expose_history(rows, attempt / "decision-1", scenario,
                       identity + ":decision", member["exposure_role"], policy)
        prepared = prepare_action(bridge, item["action"])
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
            raise ValueError(STABILITY_SIGNATURE)
        manifest = read(attempt / "decision-1" / MANIFEST_NAME)
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
            "sha256": bytes_identity(decision_frame_bytes),
            "observed_frames": policy.frames,
            "decision_compute": "none (frozen candidate; completion pass uses no predictor scoring)",
        }
        segment_started = time.monotonic()
        segment = capture_segment(bridge, aligned, attempt / "shot-1", member, scenario,
                                  identity + ":shot-1", item["action"], SHOT_SECONDS)
        engine_seconds = time.monotonic() - segment_started
        initial_metadata = live_files.read(Path(segment["native_root"]) / "frame_000001.json")
        initial_png = (Path(segment["native_root"]) / "frame_000001.png").read_bytes()
        if (segment["summary"]["first_fixed_step"] != DECISION_FIXED_STEP
                or initial_png != rows[-1]["canonical_png"]
                or initial_metadata["fixed_time_seconds"] != rows[-1]["fixed_time_seconds"]):
            raise ValueError("executed shot pre-intervention state differs from its decision frame")
        if initial_png != decision_frame_bytes:
            raise ValueError("sealed decision frame differs from the executed shot's first frame")
        # The #85-published detection rule, inherited verbatim and cited.
        verdict = e85.channel_scan(segment["native_root"])
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
            "frozen_replay_count_cost": cell.get("frozen_replay_count_cost"),
        }
        record["outcome"] = {
            "first_shot_success": verdict["pig_removed"],
            "shots_to_success": 1 if verdict["pig_removed"] else None,
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
    write(output / "records" / f"{identity}.json", record)
    print(f"issue-89 cell {identity} complete={record['failure'] is None} "
          f"failure={record['failure']}", flush=True)
    return record


# ------------------------------------------------------------- ledger + supervisor

def ledger_path():
    return OUTPUT / "ledger.json"


def fresh_ledger(plan):
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY,
            "plan_identity": plan["identity"], "status": "running",
            "wall_seconds_elapsed": 0.0, "gpu_seconds_elapsed": 0.0,
            "cells": {}, "phase": None}


CAP_STOPS = ("wall_cap_exceeded", "gpu_cap_exceeded", "minimum_free_storage",
             "artifact_limit")


def load_ledger(plan, phase=None):
    path = ledger_path()
    if not path.is_file():
        return fresh_ledger(plan)
    ledger = read(path)
    if (ledger.get("schema") != SCHEMA_LEDGER
            or ledger.get("plan_identity") != plan["identity"]):
        raise ValueError("retained issue-89 ledger identity differs from the frozen plan")
    if ledger["status"] in CAP_STOPS:
        raise ValueError(f"issue-89 execution ledger is terminal ({ledger['status']}); "
                         "audit it; never relaunch a cap-stopped campaign")
    if phase is not None and ledger.get("phase") != phase:
        if ledger["status"] != "complete":
            raise ValueError(
                f"issue-89 phase {ledger['phase']} is not complete ({ledger['status']}); "
                f"resume it before entering {phase}")
        # Phase promotion: the prior phase finished cleanly; the next phase
        # continues the same global cap accounting without relaunching any
        # terminal cell.
        ledger["phase"] = phase
        ledger["status"] = "running"
        return ledger
    if ledger["status"] == "interrupted":
        # A crashed/killed phase is resumable: terminal_cells() reconciles
        # partial persistence outcome-blind and re-queues unrecorded cells.
        ledger["status"] = "running"
        if phase is not None:
            ledger["phase"] = phase
        return ledger
    if phase is not None:
        ledger["phase"] = phase
    return ledger


def terminal_cells(plan, cells, phase_records="records"):
    """Cells with a retained record + receipt are terminal, whatever the outcome.

    Partial persistence from a supervisor crash is completed without touching
    the recorded outcome; an attempt tree with neither file never produced an
    outcome and is removed so the cell re-dispatches - crash recovery, not
    outcome-conditioned replacement."""
    terminal, recovered = {}, []
    for cell in cells:
        identity = cell["identity"]
        record = OUTPUT / phase_records / f"{identity}.json"
        receipt = OUTPUT / "receipts" / f"{identity}.json"
        attempt = OUTPUT / "attempts" / identity
        if record.is_file() and receipt.is_file():
            terminal[identity] = True
            continue
        if record.is_file() and not receipt.is_file():
            write(receipt, {"identity": identity, "phase": None,
                            "worker_exitcode": 0, "stop": None,
                            "wall_seconds": None, "peak_cpu_rss_mib": None,
                            "receipt_recovered_from_record": True})
            terminal[identity] = True
            continue
        if receipt.is_file() and not record.is_file():
            write(record, _plain_record(plan, cell,
                                        next(s for s in plan["states"]
                                             if s["identity"] == cell["state"]))
                  | {"failure": "worker_terminated_before_record",
                     "failure_kind": "execution_failure", "wall_seconds": None})
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


def _state_for(plan, state_identity):
    return next(state for state in plan["states"] if state["identity"] == state_identity)


def supervise(args, plan, cells, phase, payloads=None):
    """Run the phase's scheduled cells with isolated bounded workers, caps,
    and the frozen staggered cold-start dispatch (a new worker is dispatched
    no sooner than START_STAGGER_SECONDS after the previous dispatch)."""
    ledger = load_ledger(plan, phase)
    terminal, recovered = terminal_cells(plan, cells)
    if recovered:
        log(f"crash recovery: {len(recovered)} dispatched-but-unrecorded cells "
            f"re-queued (no outcome was recorded): {sorted(recovered)[:4]}")
    pending = [cell["identity"] for cell in cells if not terminal[cell["identity"]]]
    by_identity = {cell["identity"]: cell for cell in cells}
    ledger["phase"] = phase
    write(ledger_path(), ledger)
    if not pending:
        ledger["status"] = "complete"
        write(ledger_path(), ledger)
        log(f"{phase}: all {len(cells)} scheduled cells already terminal")
        return ledger
    player = Path(plan["player_root"])
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")
    started = time.monotonic()
    last_dispatch = started - START_STAGGER_SECONDS
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
            while (index < len(pending) and len(live) < WORKERS
                   and time.monotonic() - last_dispatch >= START_STAGGER_SECONDS):
                identity = pending[index]
                cell = by_identity[identity]
                payload = {"output": str(OUTPUT), "cell": cell,
                           "state": _state_for(plan, cell["state"])}
                if payloads and identity in payloads:
                    payload.update(payloads[identity])
                (OUTPUT / "markers").mkdir(parents=True, exist_ok=True)
                write(OUTPUT / "markers" / f"{identity}.json",
                      {"identity": identity, "phase": phase,
                       "dispatched_at_utc": utc_now()})
                process = start_isolated_worker(context, execute_completion_cell, (payload,))
                live.append({"identity": identity, "process": process,
                             "started": time.monotonic(), "peak_rss": 0.0, "stop": None})
                last_dispatch = time.monotonic()
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
                if not record_file.is_file():
                    failure = worker["stop"] or f"worker_exitcode={exitcode}"
                    cell = by_identity[worker["identity"]]
                    write(record_file, _plain_record(plan, cell, _state_for(plan, cell["state"]))
                          | {"failure": failure, "failure_kind": "execution_failure"})
                ledger["cells"][worker["identity"]] = {
                    "status": "complete" if (worker["stop"] is None and exitcode == 0) else "failed",
                    "stop": worker["stop"], "exit_code": exitcode,
                    "wall_seconds": wall,
                    "failure": read(record_file).get("failure"),
                }
                ledger["wall_seconds_elapsed"] = (
                    ledger["wall_seconds_elapsed"] + time.monotonic() - started)
                started = time.monotonic()
                write(OUTPUT / "receipts" / f"{worker['identity']}.json", {
                    "identity": worker["identity"], "phase": phase,
                    "worker_exitcode": exitcode, "stop": worker["stop"],
                    "wall_seconds": wall, "peak_cpu_rss_mib": worker["peak_rss"]})
                write(ledger_path(), ledger)
                done = len(ledger["cells"])
                remaining = len(cells) - done
                finished_walls = [entry["wall_seconds"] for entry in ledger["cells"].values()
                                  if entry.get("wall_seconds")]
                mean_wall = (sum(finished_walls) / len(finished_walls)) if finished_walls else 0.0
                eta_seconds = remaining * mean_wall / WORKERS
                log(f"[{done}/{len(cells)}] {worker['identity']} "
                    f"stop={worker['stop']} wall={wall:.0f}s "
                    f"eta={eta_seconds / 60:.0f}m "
                    f"elapsed={(ledger['wall_seconds_elapsed']) / 60:.0f}m")
        elapsed = ledger["wall_seconds_elapsed"] + time.monotonic() - started
        ledger["wall_seconds_elapsed"] = elapsed
        if stop_reason is not None:
            # Cap/storage stop: every unlaunched cell is retained as a typed
            # terminal failure with the stop reason. No replacement, no retry.
            for identity in pending:
                record_file = OUTPUT / "records" / f"{identity}.json"
                if identity in ledger["cells"] or record_file.is_file():
                    continue
                cell = by_identity[identity]
                write(record_file, _plain_record(plan, cell, _state_for(plan, cell["state"]))
                      | {"failure": f"not_executed: {stop_reason}",
                         "failure_kind": "not_executed", "wall_seconds": 0.0})
                write(OUTPUT / "receipts" / f"{identity}.json", {
                    "identity": identity, "phase": phase, "worker_exitcode": None,
                    "stop": stop_reason, "wall_seconds": 0.0, "peak_cpu_rss_mib": 0.0})
                ledger["cells"][identity] = {"status": "failed", "stop": stop_reason,
                                             "exit_code": None, "wall_seconds": 0.0,
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
        f"wall={ledger['wall_seconds_elapsed']:.0f}s "
        f"gpu={ledger['gpu_seconds_elapsed']:.0f}s")
    return ledger


# ---------------------------------------------------------------- publication

def read_record(identity, sub="records"):
    record_file = OUTPUT / sub / f"{identity}.json"
    return read(record_file) if record_file.is_file() else None


def unreachability_declaration(plan):
    """The frozen unreachability branch, evaluated on the terminal records:
    any issue-77-n1-005-a07 slot failing with the sealed-frame stability
    signature declares the state typed-unmeasurable."""
    failures = []
    cells = [cell for cell in plan["completion_cells"]
             if cell["state"] == UNMEASURABLE_CANDIDATE_STATE]
    for cell in cells:
        record = read_record(cell["identity"])
        if record is None:
            continue
        failure = record.get("failure") or ""
        if STABILITY_SIGNATURE in failure:
            failures.append({"identity": cell["identity"],
                             "ordinal": cell["ordinal"], "failure": failure})
    if not failures:
        return None
    return {"state": UNMEASURABLE_CANDIDATE_STATE,
            "declaration": "typed_unmeasurable",
            "rule": plan["definitions"]["unreachability_branch"]["rule"],
            "signature": STABILITY_SIGNATURE,
            "stability_failures": failures,
            "media": f"data/issue-89-oracle-completion/cells/",
            "consequence": ("the ceiling questions are dispositioned over the "
                            "23 measured states with this exclusion declared")}


def completion_tables(plan):
    """Per-slot verdicts joined side-by-side to #87's published ledger rows."""
    per_slot = {}
    for cell in plan["completion_cells"]:
        record = read_record(cell["identity"])
        if record is None:
            raise ValueError(f"publication requires a terminal record for {cell['identity']}")
        outcome = record.get("outcome") or {}
        execution = record.get("execution") or {}
        channel = record.get("engine_channel") or {}
        per_slot[cell["identity"]] = {
            "identity": cell["identity"], "state": cell["state"],
            "ordinal": cell["ordinal"], "branch_identity": cell["branch_identity"],
            "issue87_typed_failure": cell["issue87_typed_failure"],
            "failure": record.get("failure"),
            "failure_kind": record.get("failure_kind"),
            "success": outcome.get("first_shot_success"),
            "consistency": channel.get("consistency"),
            "censored": execution.get("censored"),
            "engine_wall_seconds": execution.get("engine_wall_seconds"),
            "wall_seconds": record.get("wall_seconds"),
            "stability_signature": bool(record.get("failure")
                                        and STABILITY_SIGNATURE in record["failure"]),
        }
    declaration = unreachability_declaration(plan)
    declared = declaration is not None
    executed = sum(1 for row in per_slot.values() if row["failure"] is None)
    residual = sorted(identity for identity, row in per_slot.items()
                      if row["failure"] is not None
                      and not (declared and row["state"] == UNMEASURABLE_CANDIDATE_STATE))
    inventory = {
        "scheduled_cells": len(plan["completion_cells"]),
        "executed": executed,
        "typed_failures": len(per_slot) - executed,
        "successful_cells": sum(1 for row in per_slot.values() if row["success"] is True),
        "residual_typed_failures_outside_declaration": residual,
        "unreachability_declaration": declaration,
    }
    return {"per_slot": dict(sorted(per_slot.items())), "inventory": inventory}


def ceiling_tables(plan, completion):
    """Updated per-state oracle indicators: #87's published per-state rows
    (read-only) plus this pass's channel verdicts, side-by-side."""
    summary87 = issue87_summary()
    declaration = completion["inventory"]["unreachability_declaration"]
    per_slot = completion["per_slot"]
    per_state = {}
    for state_identity, row87 in sorted(summary87["oracle_per_state"].items()):
        slots = [row for row in per_slot.values() if row["state"] == state_identity]
        completion_executed = sum(1 for row in slots if row["failure"] is None)
        completion_successes = sum(1 for row in slots if row["success"] is True)
        executed = row87["executed"] + completion_executed
        successes = row87["successes"] + completion_successes
        declared = bool(declaration and state_identity == UNMEASURABLE_CANDIDATE_STATE)
        if declared:
            indicator = None  # typed-unmeasurable: excluded, declared
        elif executed > 0:
            indicator = 1 if successes > 0 else 0
        else:
            indicator = None  # unmeasured without declaration
        per_state[state_identity] = {
            "state": state_identity, "family": row87["family"],
            "source_member": row87["source_member"],
            "candidates": row87["candidates"],
            "issue87_executed": row87["executed"],
            "issue87_successes": row87["successes"],
            "issue87_successful_ordinals": row87["successful_ordinals"],
            "issue87_typed_failures": len(row87["typed_failures"]),
            "completion_scheduled": len(slots),
            "completion_executed": completion_executed,
            "completion_successes": completion_successes,
            "completion_successful_ordinals": sorted(row["ordinal"] for row in slots
                                                     if row["success"] is True),
            "completion_typed_failures": sum(1 for row in slots
                                             if row["failure"] is not None),
            "executed": executed, "successes": successes,
            "typed_unmeasurable": declared,
            "indicator": indicator,
        }
    measured = [row for row in per_state.values() if row["indicator"] is not None]
    unmeasured = sorted(row["state"] for row in per_state.values()
                        if row["indicator"] is None and not row["typed_unmeasurable"])
    indicators = [float(row["indicator"]) for row in measured]
    interval = engine.paired_interval(indicators)
    pooled = {
        "states": len(per_state),
        "measured_states": len(measured),
        "typed_unmeasurable_states": ([UNMEASURABLE_CANDIDATE_STATE]
                                      if declaration else []),
        "unmeasured_states": unmeasured,
        "ceiling_states": int(sum(indicators)),
        "prevalence": float(np.mean(indicators)) if indicators else None,
        "descriptive": interval,
        "label": "DESCRIPTIVE",
    }
    partial = plan["issue87_published_partial_ceiling"]
    comparison = {
        "issue87_published_partial_ceiling": partial,
        "completed_ceiling": {
            "measured_states": pooled["measured_states"],
            "ceiling_states": pooled["ceiling_states"],
            "prevalence": pooled["prevalence"],
            "descriptive": pooled["descriptive"],
        },
        "delta_ceiling_states": (pooled["ceiling_states"]
                                 - partial["ceiling_states"]),
        "delta_prevalence": (None if pooled["prevalence"] is None
                             else pooled["prevalence"] - partial["prevalence"]),
        "note": ("issue-87's partial ceiling covered 23 measured states with "
                 "issue-77-n1-005-a07 unmeasured; this completion pass adds "
                 "the 55 missing slots; the comparison is descriptive"),
    }
    return {"per_state": per_state, "pooled": pooled, "comparison": comparison}


def development_summary(plan):
    """The development-version smoke outcomes retained side by side; EXPLORATORY."""
    versions = []
    for version in range(1, TERMINAL_VERSION):
        path = plan_path(version)
        if not path.is_file():
            continue
        version_plan = read(path)
        if version_plan.get("status") != "development":
            continue
        cells = []
        prefix = OUTPUT / "development" / f"v{version}"
        for cell in version_plan["smoke_cells"]:
            record_file = prefix / "records" / f"{cell['identity']}.json"
            record = read(record_file) if record_file.is_file() else None
            outcome = (record or {}).get("outcome") or {}
            expectation = cell["expectation"]
            failure = (record or {}).get("failure")
            if expectation == "channel_verdict":
                matched = record is not None and failure is None
            else:
                matched = record is not None and (
                    failure is None or STABILITY_SIGNATURE in failure)
            cells.append({
                "identity": cell["identity"],
                "completion_identity": cell["completion_identity"],
                "state": cell["state"], "ordinal": cell["ordinal"],
                "expectation": expectation,
                "executed": record is not None and failure is None,
                "pig_removed": outcome.get("first_shot_success"),
                "expectation_matched": matched if record is not None else None,
                "failure": failure,
                "wall_seconds": (record or {}).get("wall_seconds"),
            })
        versions.append({
            "version": version, "status": "development", "label": "EXPLORATORY",
            "may_not_be_cited_as_answers": True,
            "plan_path": str(path.relative_to(ROOT)),
            "changelog": version_plan["changelog"],
            "smoke_cells": cells,
            "note": ("development-version outcomes, retained verbatim; the "
                     "ticket's dispositions come only from the terminal "
                     "version"),
        })
    return versions


def development_smoke_complete(version):
    path = plan_path(version)
    if not path.is_file():
        return False
    version_plan = read(path)
    for cell in version_plan["smoke_cells"]:
        record = OUTPUT / "development" / f"v{version}" / "records" / f"{cell['identity']}.json"
        receipt = OUTPUT / "development" / f"v{version}" / "receipts" / f"{cell['identity']}.json"
        if not (record.is_file() and receipt.is_file()):
            return False
    return True


def development_versions_complete():
    for version in range(1, TERMINAL_VERSION):
        path = plan_path(version)
        if not path.is_file():
            return False
        version_plan = read(path)
        if version_plan.get("status") != "development":
            continue
        for cell in version_plan["smoke_cells"]:
            record = OUTPUT / "development" / f"v{version}" / "records" / f"{cell['identity']}.json"
            receipt = OUTPUT / "development" / f"v{version}" / "receipts" / f"{cell['identity']}.json"
            if not (record.is_file() and receipt.is_file()):
                return False
    return True


def question_dispositions(plan, completion, ceiling, ledger_status):
    inventory = completion["inventory"]
    pooled = ceiling["pooled"]
    if ledger_status != "complete":
        q1 = "readiness_or_precision_insufficient"
        q1_reason = "a cap stop occurred or the completion inventory is incomplete"
    elif inventory["residual_typed_failures_outside_declaration"]:
        q1 = "not_supported_by_this_experiment"
        q1_reason = ("the completion pass ran to completion but residual typed "
                     "terminal failures remain outside the frozen declaration: "
                     f"{inventory['residual_typed_failures_outside_declaration'][:4]}")
    else:
        q1 = "supported"
        q1_reason = ("every scheduled slot carries an engine-truth channel "
                     "verdict or is covered by the frozen typed-unmeasurable "
                     "declaration")
    if ledger_status != "complete":
        q2 = "readiness_or_precision_insufficient"
        q2_reason = "a cap stop occurred or the completion inventory is incomplete"
    elif pooled["unmeasured_states"]:
        q2 = "readiness_or_precision_insufficient"
        q2_reason = (f"state(s) unmeasured without the frozen declaration: "
                     f"{pooled['unmeasured_states'][:4]}")
    else:
        prevalence = pooled["prevalence"]
        interval = (pooled["descriptive"] or {}).get("descriptive_95_percent_interval")
        declared = pooled["typed_unmeasurable_states"]
        scope = (f"over the {pooled['measured_states']} measured states "
                 + (f"with {declared} excluded under the frozen "
                    "typed-unmeasurable declaration" if declared else ""))
        if prevalence is None:
            q2, q2_reason = "readiness_or_precision_insufficient", "no measured states"
        elif prevalence > 0 and interval is not None and interval[0] > 0:
            q2 = "supported"
            q2_reason = (f"pooled prevalence {prevalence:.4f} > 0 and the "
                         f"descriptive 95% interval [{interval[0]:.4f}, "
                         f"{interval[1]:.4f}] excludes 0 {scope}")
        elif prevalence == 0:
            q2 = "not_supported_by_this_experiment"
            q2_reason = ("complete measurement with pooled prevalence 0: no "
                         "admissible candidate achieved engine-truth "
                         f"first-shot success anywhere {scope} (a measured "
                         "state-difficulty floor)")
        else:
            q2 = "readiness_or_precision_insufficient"
            bounds = (None if interval is None
                      else f"[{interval[0]:.4f}, {interval[1]:.4f}]")
            q2_reason = (f"prevalence {prevalence:.4f} > 0 but the descriptive "
                         f"95% interval {bounds} contains 0 {scope}")
    return {"q1_slot_completion": q1, "q1_reason": q1_reason,
            "q2_ceiling_reestimate": q2, "q2_reason": q2_reason}


def publication(plan):
    ledger = read(ledger_path())
    common = {
        "schema": SCHEMA_REPORT, "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "terminal_version": plan["version"],
        "terminal_frozen_at_utc": plan["frozen_at_utc"],
        "protocol_versions": protocol_chronology(),
        "changelog": plan["changelog"],
        "definitions": plan["definitions"],
        "claim_boundary": plan["definitions"]["claim_boundary"],
        "engine_channel": plan["definitions"]["engine_channel"],
        "issue87_binding": plan["issue87_binding"],
        "execution_ledger_status": ledger["status"],
        "wall_seconds_elapsed": ledger["wall_seconds_elapsed"],
        "gpu_seconds_elapsed": ledger["gpu_seconds_elapsed"],
        "caps": plan["definitions"]["caps"],
        "limitations": [
            "development/exposed N1 lineages only; zero fresh captures; not the sealed #64/#65 benchmark",
            "the engine-truth verdict covers the bounded native shot window per executed shot",
            "the oracle pass uses the single frozen seed 20260908; per-state indicators are one "
            "engine-truth draw per candidate slot",
            "states sharing a source member share one physical initial state and candidate table; "
            "the paired bootstrap over state identities is descriptive only",
            "this ticket APPENDS the 55 missing slots of #87; #87's published outcomes are "
            "read-only inputs, never amended or reinterpreted; no system decision pass here",
            "a typed-unmeasurable declaration for issue-77-n1-005-a07 excludes that state from "
            "the ceiling estimate with the exclusion declared",
        ],
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False,
    }

    def blocked(dispositions, explanation):
        common["ticket_disposition"] = "readiness_or_precision_insufficient"
        common["question_dispositions"] = dispositions
        common["stopExplanation"] = explanation
        common["development_versions"] = development_summary(plan)
        return common

    ready = all(read_record(cell["identity"]) is not None
                for cell in plan["completion_cells"])
    if not ready:
        return blocked(
            {"q1_slot_completion": "readiness_or_precision_insufficient",
             "q2_ceiling_reestimate": "readiness_or_precision_insufficient"},
            "the terminal completion pass has not completed its scheduled inventory")
    completion = completion_tables(plan)
    ceiling = ceiling_tables(plan, completion)
    common["completion_pass"] = {"inventory": completion["inventory"]}
    common["ceiling_reestimate"] = {"per_state": ceiling["per_state"],
                                    "pooled": ceiling["pooled"],
                                    "comparison": ceiling["comparison"]}
    if ledger["status"] != "complete":
        return blocked(
            {"q1_slot_completion": "readiness_or_precision_insufficient",
             "q2_ceiling_reestimate": "readiness_or_precision_insufficient"},
            f"execution inventory incomplete or a cap stop occurred (ledger: {ledger['status']})")
    dispositions = question_dispositions(plan, completion, ceiling, ledger["status"])
    common["question_dispositions"] = dispositions
    common["ticket_disposition"] = dispositions["q2_ceiling_reestimate"]
    common["development_versions"] = development_summary(plan)
    common["completion_per_slot"] = completion["per_slot"]
    return common


def protocol_chronology():
    entries = []
    for version in range(1, TERMINAL_VERSION + 1):
        path = plan_path(version)
        if not path.is_file():
            entries.append({"version": version, "retained": False})
            continue
        version_plan = read(path)
        entries.append({
            "version": version, "retained": True,
            "path": str(path.relative_to(ROOT)),
            "status": version_plan["status"],
            "frozen_at_utc": version_plan["frozen_at_utc"],
            "changelog": version_plan["changelog"],
        })
    return entries


def compact_report(result):
    return {key: value for key, value in result.items()
            if key not in ("completion_per_slot",)}


def fmt(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("completion_per_slot", "state", "ordinal",
                     "issue87_typed_failure", "completion_failure",
                     "first_shot_success", "stability_signature"))
    for identity, row in sorted((result.get("completion_per_slot") or {}).items()):
        writer.writerow((identity, row["state"], row["ordinal"],
                         row["issue87_typed_failure"], row["failure"] or "",
                         row["success"], row["stability_signature"]))
    writer.writerow(())
    writer.writerow(("updated_oracle_per_state", "family", "candidates",
                     "issue87_executed", "issue87_successes",
                     "completion_scheduled", "completion_executed",
                     "completion_successes", "executed", "successes",
                     "typed_unmeasurable", "ceiling_indicator"))
    for state, row in ((result.get("ceiling_reestimate") or {})
                       .get("per_state") or {}).items():
        writer.writerow((state, row["family"], row["candidates"],
                         row["issue87_executed"], row["issue87_successes"],
                         row["completion_scheduled"], row["completion_executed"],
                         row["completion_successes"], row["executed"],
                         row["successes"], row["typed_unmeasurable"],
                         row["indicator"]))
    writer.writerow(())
    pooled = (result.get("ceiling_reestimate") or {}).get("pooled") or {}
    interval = (pooled.get("descriptive") or {}).get("descriptive_95_percent_interval")
    writer.writerow(("pooled_ceiling", "measured_states", "ceiling_states",
                     "prevalence", "descriptive_95_low", "descriptive_95_high",
                     "label"))
    if pooled:
        writer.writerow(("completed", pooled["measured_states"],
                         pooled["ceiling_states"], pooled["prevalence"],
                         None if not interval else interval[0],
                         None if not interval else interval[1], "DESCRIPTIVE"))
    comparison = (result.get("ceiling_reestimate") or {}).get("comparison") or {}
    partial = comparison.get("issue87_published_partial_ceiling") or {}
    partial_interval = (partial.get("descriptive") or {}).get("descriptive_95_percent_interval")
    if partial:
        writer.writerow(("issue87_published_partial", partial.get("measured_states"),
                         partial.get("ceiling_states"), partial.get("prevalence"),
                         None if not partial_interval else partial_interval[0],
                         None if not partial_interval else partial_interval[1],
                         "DESCRIPTIVE"))
    writer.writerow(())
    writer.writerow(("protocol_version", "status", "frozen_at_utc", "change"))
    for entry in result.get("protocol_versions") or []:
        for change in entry.get("changelog") or []:
            writer.writerow((f"plan-v{entry['version']}", entry["status"],
                             entry["frozen_at_utc"], change["change"]))
    writer.writerow(())
    writer.writerow(("development_versions", "version", "label", "cell",
                     "executed", "pig_removed", "expectation_matched"))
    for version in result.get("development_versions") or []:
        for cell in version["smoke_cells"]:
            writer.writerow(("EXPLORATORY", version["version"], version["label"],
                             cell["identity"], cell["executed"], cell["pig_removed"],
                             cell["expectation_matched"]))
    return stream.getvalue()


def findings_md(result):
    lines = ["# Issue-89 closed-loop oracle-ceiling completion pass - findings", ""]
    lines.append(f"Terminal protocol version: v{result.get('terminal_version')} "
                 f"(frozen {result.get('terminal_frozen_at_utc')}, before its scoring run). "
                 f"Ledger status: {result['execution_ledger_status']}.")
    lines.append("")
    lines.append(f"Claim boundary: {result['claim_boundary']}")
    lines.append("")
    lines.append("## Q1 - completion of the 55 typed-failure slots of #87")
    lines.append("")
    inventory = (result.get("completion_pass") or {}).get("inventory") or {}
    if inventory:
        lines.append(f"Completion inventory: {inventory.get('scheduled_cells')} scheduled "
                     f"slots, {inventory.get('executed')} executed with an engine-truth "
                     f"channel verdict, {inventory.get('typed_failures')} typed failures, "
                     f"{inventory.get('successful_cells')} engine-truth-successful slots.")
        residual = inventory.get("residual_typed_failures_outside_declaration") or []
        lines.append(f"Residual typed failures outside the frozen declaration: "
                     f"{len(residual)}"
                     + (f" ({', '.join(residual[:4])})" if residual else "."))
        declaration = inventory.get("unreachability_declaration")
        if declaration:
            lines.append(f"Typed-unmeasurable declaration: {declaration['state']} "
                         f"({len(declaration['stability_failures'])} slot(s) with the "
                         "sealed-frame stability signature; stability evidence + media "
                         "published).")
    else:
        lines.append("Completion pass not scored.")
    lines.append("")
    lines.append("## Q2 - ceiling re-estimate over the 24 frozen states")
    lines.append("")
    pooled = (result.get("ceiling_reestimate") or {}).get("pooled") or {}
    if pooled:
        interval = (pooled.get("descriptive") or {}).get("descriptive_95_percent_interval")
        lines.append(f"Engine-truth ceiling states: {pooled['ceiling_states']}/"
                     f"{pooled['measured_states']} measured states "
                     f"(pooled prevalence {fmt(pooled['prevalence'])}); descriptive 95% "
                     f"interval [{fmt(None if not interval else interval[0])}, "
                     f"{fmt(None if not interval else interval[1])}] over states "
                     "(DESCRIPTIVE).")
        if pooled.get("typed_unmeasurable_states"):
            lines.append(f"Excluded under the frozen typed-unmeasurable declaration: "
                         f"{pooled['typed_unmeasurable_states']}.")
        comparison = (result.get("ceiling_reestimate") or {}).get("comparison") or {}
        partial = comparison.get("issue87_published_partial_ceiling") or {}
        partial_interval = (partial.get("descriptive") or {}).get(
            "descriptive_95_percent_interval")
        lines.append(f"Comparison to #87's published partial ceiling: "
                     f"{partial.get('ceiling_states')}/{partial.get('measured_states')} "
                     f"measured states, prevalence {fmt(partial.get('prevalence'))}, "
                     f"interval [{fmt(None if not partial_interval else partial_interval[0])}, "
                     f"{fmt(None if not partial_interval else partial_interval[1])}] "
                     f"-> completed {pooled['ceiling_states']}/{pooled['measured_states']}, "
                     f"prevalence {fmt(pooled['prevalence'])} "
                     f"(delta ceiling states {comparison.get('delta_ceiling_states')}, "
                     f"delta prevalence {fmt(comparison.get('delta_prevalence'))}).")
    else:
        lines.append("Ceiling re-estimate not scored.")
    lines.append("")
    lines.append("## Dispositions")
    lines.append("")
    for name, token in (result.get("question_dispositions") or {}).items():
        if name.endswith("_reason"):
            lines.append(f"  - reason: {token}")
        else:
            lines.append(f"- {name}: {token}")
    lines.append("")
    lines.append(f"Ticket disposition: {result.get('ticket_disposition')} "
                 "(tokens: supported / not_supported_by_this_experiment / "
                 "readiness_or_precision_insufficient).")
    if result.get("stopExplanation"):
        lines.append(f"Stop explanation: {result['stopExplanation']}")
    lines.append("")
    lines.append("## Protocol chronology")
    lines.append("")
    for entry in result.get("protocol_versions") or []:
        for change in entry.get("changelog") or []:
            lines.append(f"- plan-v{entry['version']} ({entry['status']}, frozen "
                         f"{entry['frozen_at_utc']}): {change['what']} Why: {change['why']} "
                         f"Evidence: {change['evidence']}")
    lines.append("")
    lines.append("## Development-version outcomes (EXPLORATORY; not citable as answers)")
    lines.append("")
    for version in result.get("development_versions") or []:
        for cell in version["smoke_cells"]:
            lines.append(f"- [EXPLORATORY] {cell['identity']}: executed={cell['executed']} "
                         f"pig_removed={cell['pig_removed']} "
                         f"expectation_matched={cell['expectation_matched']} "
                         f"failure={cell['failure'] or 'none'}")
    lines.append("")
    lines.append("## Compute accounting")
    lines.append("")
    lines.append(f"Global wall consumed {result['wall_seconds_elapsed']:.0f}s of "
                 f"{WALL_CAP_SECONDS}s cap; active decision GPU seconds "
                 f"{result['gpu_seconds_elapsed']:.0f} of {GPU_CAP_SECONDS}s cap "
                 "(no decision pass in this ticket; the GPU cap is a guard).")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- review gallery

def gallery_entry(identity, record, cell_root, attempt_root):
    entry = {"identity": identity}
    if record is None:
        entry["status"] = "missing"
        return entry
    entry["status"] = "typed_failure" if record.get("failure") else "executed"
    executed = record.get("failure") is None and bool(record.get("execution"))
    if (record.get("decision_frame") or {}).get("sha256"):
        source = attempt_root / "decision-frame.png"
        if source.is_file():
            shutil.copy2(source, cell_root / "decision.png")
            entry["decision_frame"] = str((cell_root / "decision.png").relative_to(DATA))
    failure = record.get("failure") or ""
    if STABILITY_SIGNATURE in failure:
        # Typed-unmeasurable stability evidence: publish the paused
        # before/after sealed-frame captures.
        for name in ("paused-before", "paused-after"):
            source = attempt_root / f"{name}.png"
            if source.is_file():
                shutil.copy2(source, cell_root / f"{name}.png")
                entry.setdefault("stability_evidence", []).append(
                    str((cell_root / f"{name}.png").relative_to(DATA)))
    frames = []
    if record.get("execution"):
        observation_root = attempt_root / "shot-1" / "observation-trace"
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
        final_png = cell_root / "final.png"
        if frames:
            final_png.write_bytes(frames[-1].read_bytes())
            entry["final_frame"] = str(final_png.relative_to(DATA))
    entry["first_shot_success"] = (None if not record.get("outcome")
                                   else bool(record["outcome"]["first_shot_success"]))
    entry["failure"] = record.get("failure")
    entry["state"] = record.get("state_identity")
    if executed and not (entry.get("decision_frame") and entry.get("video")
                         and entry.get("final_frame")):
        raise ValueError(
            f"executed cell {identity} cannot be published without its "
            f"decision frame, shot WebM and final frame: "
            f"{entry.get('video_error') or 'media missing'}")
    return entry


def publish_gallery(plan):
    """Copy one decision frame + one WebM per completion cell under data/,
    including failed/incomplete attempts and the stability evidence of any
    typed-unmeasurable slot; development smoke cells are published in a
    separate EXPLORATORY section."""
    _verify_webm_encoder()
    root = DATA / "cells"
    manifest = {"schema": SCHEMA_GALLERY, "identity": IDENTITY,
                "plan_identity": plan["identity"],
                "terminal_version": plan["version"],
                "playback_fps": PILOT_AUDIT_FPS,
                "canonical_observations_included": False, "cells": [],
                "development_cells": []}
    for cell in plan["completion_cells"]:
        identity = cell["identity"]
        record_file = OUTPUT / "records" / f"{identity}.json"
        cell_root = root / identity
        cell_root.mkdir(parents=True, exist_ok=True)
        record = read(record_file) if record_file.is_file() else None
        entry = gallery_entry(identity, record, cell_root, OUTPUT / "attempts" / identity)
        entry.update({"ordinal": cell["ordinal"],
                      "branch_identity": cell["branch_identity"],
                      "issue87_typed_failure": cell["issue87_typed_failure"]})
        manifest["cells"].append(entry)
    for version in range(1, TERMINAL_VERSION):
        prefix = OUTPUT / "development" / f"v{version}"
        version_plan = read(plan_path(version)) if plan_path(version).is_file() else None
        if version_plan is None or version_plan.get("status") != "development":
            continue
        for cell in version_plan["smoke_cells"]:
            identity = cell["identity"]
            record_file = prefix / "records" / f"{identity}.json"
            cell_root = DATA / "development" / f"v{version}" / identity
            cell_root.mkdir(parents=True, exist_ok=True)
            record = read(record_file) if record_file.is_file() else None
            if record is None:
                continue
            entry = gallery_entry(identity, record, cell_root,
                                  prefix / "attempts" / identity)
            entry.update({"ordinal": cell["ordinal"], "label": "EXPLORATORY",
                          "expectation": cell["expectation"]})
            manifest["development_cells"].append(entry)
    write(DATA / "manifest.json", manifest)
    sections = []
    for entry in manifest["cells"]:
        media = ""
        if entry.get("decision_frame"):
            media += f'<img src="{entry["decision_frame"]}" width="320"/> '
        if entry.get("final_frame"):
            media += f'<img src="{entry["final_frame"]}" width="320"/> '
        if entry.get("video"):
            media += f'<video src="{entry["video"]}" controls width="320"></video>'
        for evidence in entry.get("stability_evidence") or []:
            media += f'<img src="{evidence}" width="320"/> '
        success = entry["first_shot_success"]
        sections.append(
            f"<tr><td>{entry['identity']}</td><td>{entry['status']}</td>"
            f"<td>{'n/a' if success is None else success}</td>"
            f"<td>{entry['failure'] or ''}</td><td>{media}</td></tr>")
    dev_sections = []
    for entry in manifest["development_cells"]:
        media = ""
        if entry.get("decision_frame"):
            media += f'<img src="{entry["decision_frame"]}" width="320"/> '
        if entry.get("video"):
            media += f'<video src="{entry["video"]}" controls width="320"></video>'
        for evidence in entry.get("stability_evidence") or []:
            media += f'<img src="{evidence}" width="320"/> '
        success = entry["first_shot_success"]
        dev_sections.append(
            f"<tr><td>{entry['identity']}</td><td>EXPLORATORY</td>"
            f"<td>{'n/a' if success is None else success}</td>"
            f"<td>{entry['failure'] or ''}</td><td>{media}</td></tr>")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Issue 89 oracle-ceiling completion pass review</title></head><body>"
            "<h1>Issue 89 completion pass review (terminal protocol version)</h1>"
            "<p>One decision frame, final canonical frame and 50 fps WebM per "
            "scheduled completion slot, including typed failures; paused "
            "before/after frames are published as stability evidence for any "
            "typed-unmeasurable slot. The detection rule is the #85-published "
            "engine-truth channel, inherited verbatim. Agent observations "
            "only; canonical frames excluded.</p>"
            "<table border='1'><tr><th>cell</th><th>status</th>"
            "<th>engine-truth first-shot success</th><th>failure</th>"
            "<th>media</th></tr>" + "".join(sections) + "</table>"
            "<h2>Development-version outcomes (EXPLORATORY)</h2>"
            "<table border='1'><tr><th>cell</th><th>label</th>"
            "<th>engine-truth first-shot success</th><th>failure</th>"
            "<th>media</th></tr>" + "".join(dev_sections) + "</table></body></html>")
    (DATA / "index.html").write_text(html, encoding="utf-8")
    return manifest


def check_gallery(plan, manifest):
    """Recompute the expected gallery inventory and verify the stored one."""
    if (manifest.get("schema") != SCHEMA_GALLERY
            or manifest.get("plan_identity") != plan["identity"]
            or len(manifest["cells"]) != len(plan["completion_cells"])):
        raise ValueError("review gallery manifest binding differs")
    expected_identities = {cell["identity"] for cell in plan["completion_cells"]}
    if {entry["identity"] for entry in manifest["cells"]} != expected_identities:
        raise ValueError("review gallery cell inventory differs")
    for entry in manifest["cells"]:
        for key in ("decision_frame", "video", "final_frame"):
            if entry.get(key) and not (DATA / entry[key]).is_file():
                raise ValueError(f"gallery media missing for {entry['identity']}")
        for evidence in entry.get("stability_evidence") or []:
            if not (DATA / evidence).is_file():
                raise ValueError(f"gallery stability evidence missing for {entry['identity']}")
        record_file = OUTPUT / "records" / f"{entry['identity']}.json"
        if not record_file.is_file():
            if entry.get("status") != "missing":
                raise ValueError(f"gallery status differs for {entry['identity']}")
            continue
        record = read(record_file)
        executed = record.get("failure") is None and record.get("execution")
        if executed and not (entry.get("decision_frame") and entry.get("video")
                             and entry.get("final_frame")):
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
        failure = record.get("failure") or ""
        if STABILITY_SIGNATURE in failure and not entry.get("stability_evidence"):
            raise ValueError(f"stability-signature cell {entry['identity']} lacks "
                             "published stability evidence")
    index = DATA / "index.html"
    if not index.is_file():
        raise ValueError("review gallery index.html is missing")
    index_text = index.read_text(encoding="utf-8")
    for entry in manifest["cells"] + (manifest.get("development_cells") or []):
        if entry["identity"] not in index_text:
            raise ValueError(f"review gallery index.html omits {entry['identity']}")


# ---------------------------------------------------------------------- modes

def latest_development_version():
    """The highest frozen development version, or v1 when none exists."""
    version = DEV_VERSION
    for candidate in range(1, TERMINAL_VERSION):
        if plan_path(candidate).is_file():
            version = candidate
    return version


def dry_run(args):
    version = (TERMINAL_VERSION if development_versions_complete()
               else latest_development_version())
    plan = make_plan(version)
    cells = plan["completion_cells"]
    states = {cell["state"] for cell in cells}
    log(f"no-write dry-run: version=v{plan['version']} status={plan['status']} "
        f"completion_slots={len(cells)} states_touched={len(states)} "
        f"unmeasurable_candidate_slots="
        f"{sum(1 for c in cells if c['state'] == UNMEASURABLE_CANDIDATE_STATE)} "
        f"smoke_cells={[c['identity'] for c in plan['smoke_cells']]} "
        f"workers={WORKERS} port_range=[{PORT_LOW}, {PORT_HIGH}] "
        f"stagger_s={START_STAGGER_SECONDS} "
        f"wall_cap_s={WALL_CAP_SECONDS} gpu_cap_s={GPU_CAP_SECONDS} "
        f"eta_estimate_h={len(cells) * 165 / WORKERS / 3600:.1f} (at the #85 measured "
        f"130-200 s/cell midpoint, stagger amortized)")
    return 0


def freeze_version(version, evidence=None):
    plan = make_plan(version, evidence=evidence)
    target = plan_path(version)
    if target.exists():
        raise ValueError(f"{target} already exists; protocol versions are immutable")
    write(target, plan)
    if version == TERMINAL_VERSION:
        write(OUTPUT / "plan.json", plan)
    log(f"frozen plan-v{version} ({plan['status']}) at {target}")
    return plan


def smoke_evidence_from_records(version):
    """Mechanical summary of one development version's executed smoke,
    embedded into freeze changelogs (reviewed before freezing)."""
    plan87 = issue87_terminal_plan()
    cells = completion_cells(plan87, issue87_ledger())
    entries = []
    prefix = OUTPUT / "development" / f"v{version}"
    for cell in smoke_cells(cells, version):
        record_file = prefix / "records" / f"{cell['identity']}.json"
        record = read(record_file) if record_file.is_file() else None
        outcome = (record or {}).get("outcome") or {}
        failure = (record or {}).get("failure")
        if cell["expectation"] == "channel_verdict":
            matched = record is not None and failure is None
        else:
            matched = record is not None and (
                failure is None or STABILITY_SIGNATURE in failure)
        entries.append({
            "identity": cell["identity"],
            "completion_identity": cell["completion_identity"],
            "expectation": cell["expectation"],
            "executed": record is not None and failure is None,
            "pig_removed": outcome.get("first_shot_success"),
            "expectation_matched": matched if record is not None else None,
            "wall_seconds": (record or {}).get("wall_seconds"),
            "decision_frame_retained": bool((record or {}).get("decision_frame")),
            "failure": failure,
        })
    executed = sum(1 for entry in entries if entry["executed"])
    matched = sum(1 for entry in entries if entry["expectation_matched"])
    return {
        "completion_slot_count": 55,
        "summary": (f"{executed}/{len(entries)} smoke cells executed real-rendered "
                    "under the frozen mitigation, "
                    f"{matched}/{len(entries)} matched their frozen structural "
                    "expectations; channel verdicts and gallery media retained"),
        "records_reference": (f"development/v{version} records + "
                              f"data/issue-89-oracle-completion/development/v{version}"),
        "cells": entries,
    }


def prepare(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocols_dir().mkdir(parents=True, exist_ok=True)
    plan_file = OUTPUT / "plan.json"
    if not plan_path(DEV_VERSION).is_file():
        freeze_version(DEV_VERSION)
        require_prepared(read(plan_path(DEV_VERSION)))
        log("issue-89 development protocol frozen (55-slot completion inventory "
            "bound to the issue-87 ledger, mitigation, unreachability branch, "
            "caps, port range); no gameplay executed")
        return 0
    if not plan_path(2).is_file():
        if not development_smoke_complete(DEV_VERSION):
            raise ValueError("the plan-v2 development amendment requires the "
                             "executed plan-v1 rendered smoke; run "
                             "--smoke-test first")
        v1_evidence = smoke_evidence_from_records(DEV_VERSION)
        freeze_version(2, evidence={"v1_summary": v1_evidence["summary"]})
        require_prepared(read(plan_path(2)))
        log("issue-89 development plan-v2 frozen (probe-poisoning correction "
            "after the plan-v1 smoke); run --smoke-test to validate the "
            "corrected harness")
        return 0
    if not plan_path(TERMINAL_VERSION).is_file():
        if not development_versions_complete():
            raise ValueError("the terminal freeze requires the executed "
                             "development rendered smokes; run --smoke-test first")
        if plan_file.is_file():
            prior = read(plan_file)
            if prior.get("terminal") and any((OUTPUT / "records").glob("*.json")):
                raise ValueError("a terminal plan is already frozen and the "
                                 "scoring run has started; audit and version "
                                 "instead of re-freezing")
        latest = latest_development_version()
        evidence = smoke_evidence_from_records(latest)
        gate = next(entry for entry in evidence["cells"]
                    if entry["expectation"] == "channel_verdict")
        if not gate["executed"]:
            raise ValueError("the former-timeout smoke slot did not reach an "
                             "engine-truth channel verdict under the frozen "
                             "mitigation; the terminal freeze is barred")
        freeze_version(TERMINAL_VERSION, evidence=evidence)
        plan = load_plan()
        check_plan_bound(plan)
        require_prepared(plan)
        log("issue-89 TERMINAL protocol frozen with actual numeric values "
            "(55 completion slots, mitigation, unreachability branch, "
            "disposition mapping) BEFORE its scoring run")
        return 0
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    log("issue-89 terminal protocol already frozen; verified against its bound inputs")
    return 0


def smoke_records_dir(version):
    return OUTPUT / "development" / f"v{version}"


def smoke_test(args):
    """Bounded real RENDERED smoke of the mitigated completion harness under
    the latest frozen development plan (outcomes retained, EXPLORATORY)."""
    if not plan_path(DEV_VERSION).is_file():
        raise ValueError("run --prepare first: the smoke requires a frozen "
                         "development plan")
    version = latest_development_version()
    plan = read(plan_path(version))
    prefix = smoke_records_dir(version)
    (prefix / "records").mkdir(parents=True, exist_ok=True)
    (prefix / "receipts").mkdir(parents=True, exist_ok=True)
    ledger = load_ledger(plan, "smoke")
    ledger["phase"] = "smoke"
    write(ledger_path(), ledger)
    pending = []
    for cell in plan["smoke_cells"]:
        record_file = prefix / "records" / f"{cell['identity']}.json"
        receipt_file = prefix / "receipts" / f"{cell['identity']}.json"
        if record_file.is_file() and receipt_file.is_file():
            log(f"smoke cell {cell['identity']} already terminal")
            continue
        pending.append(cell)
    if not pending:
        log("smoke: all development cells already executed")
        return 0
    context = multiprocessing.get_context("spawn")
    for cell in pending:
        state = _state_for(plan, cell["state"])
        payload = {"output": str(prefix), "cell": cell, "state": state,
                   "plan_path": str(plan_path(version))}
        write(prefix / "markers" / f"{cell['identity']}.json",
              {"identity": cell["identity"], "phase": "smoke",
               "dispatched_at_utc": utc_now()})
        began = time.monotonic()
        process = start_isolated_worker(context, execute_completion_cell, (payload,))
        while process.is_alive() and time.monotonic() - began < ATTEMPT_SECONDS:
            time.sleep(1)
        stop = None
        if process.is_alive():
            stop = "attempt_wall_limit"
            terminate_worker(process)
        wall = time.monotonic() - began
        record_file = prefix / "records" / f"{cell['identity']}.json"
        if not record_file.is_file():
            write(record_file, _plain_record(plan, cell, state)
                  | {"failure": stop or f"worker_exitcode={process.exitcode}",
                     "failure_kind": "execution_failure"})
        if stop is None:
            terminate_worker(process)
        write(prefix / "receipts" / f"{cell['identity']}.json", {
            "identity": cell["identity"], "phase": "smoke",
            "worker_exitcode": process.exitcode, "stop": stop,
            "wall_seconds": wall, "peak_cpu_rss_mib": None})
        record = read(record_file)
        ledger["cells"][cell["identity"]] = {
            "status": "complete" if process.exitcode == 0 else "failed",
            "stop": None, "exit_code": process.exitcode, "wall_seconds": wall,
            "failure": record.get("failure")}
        ledger["wall_seconds_elapsed"] += wall
        write(ledger_path(), ledger)
        outcome = record.get("outcome") or {}
        log(f"smoke {cell['identity']} executed={record.get('failure') is None} "
            f"pig_removed={outcome.get('first_shot_success')} "
            f"failure={record.get('failure')} wall={wall:.0f}s")
    ledger["status"] = "complete" if development_versions_complete() else "interrupted"
    write(ledger_path(), ledger)
    log(f"smoke complete: review the retained records/frames under "
        f"development/v{version} before freezing the next protocol with --prepare")
    return 0


def run(args):
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    ledger = load_ledger(plan)
    if ledger["status"] not in ("running", "complete"):
        raise ValueError(f"issue-89 ledger is terminal ({ledger['status']}); audit it")
    supervise(args, plan, plan["completion_cells"], "completion")
    final = read(ledger_path())
    if final["status"] != "complete":
        log(f"completion pass ended with status {final['status']}; the "
            "disposition is readiness_or_precision_insufficient")
        return 2
    log("terminal scoring run complete: proceed to --publish")
    return 0


def publish(args):
    plan = load_plan()
    check_plan_bound(plan)
    result = publication(plan)
    write(OUTPUT / "summary.json", compact_report(result))
    (OUTPUT / "comparisons.csv").write_text(comparisons_csv(result))
    (OUTPUT / "findings.md").write_text(findings_md(result))
    manifest = publish_gallery(plan)
    log(f"published ticket_disposition={result.get('ticket_disposition')} "
        f"question_dispositions={result.get('question_dispositions')} "
        f"gallery_cells={len(manifest['cells'])} "
        f"development_cells={len(manifest['development_cells'])}")
    return 0


def recompute_completion_channel(plan):
    """Rescan every executed completion cell's native trace with the inherited
    detection rule and compare to the stored verdict."""
    problems = []
    for cell in plan["completion_cells"]:
        identity = cell["identity"]
        record_file = OUTPUT / "records" / f"{identity}.json"
        if not record_file.is_file():
            problems.append(f"{identity}: completion record missing")
            continue
        record = read(record_file)
        execution = record.get("execution") or {}
        native_root = execution.get("native_root")
        if native_root is None:
            if record.get("failure") is None:
                problems.append(f"{identity}: executed completion cell lacks its native root")
            continue
        rescanned = e85.channel_scan(native_root)
        saved = record.get("engine_channel") or {}
        for key in ("pig_removed", "structure_destroyed", "bird_launches",
                    "pig_lifecycle_destroyed"):
            if rescanned.get(key) != saved.get(key):
                problems.append(f"{identity}: engine_channel[{key}] differs from the "
                                f"native rescan ({saved.get(key)} vs {rescanned.get(key)})")
        consistency = engine.channel_consistency(bool(saved.get("pig_removed")),
                                                 bool(saved.get("pig_lifecycle_destroyed")))
        if consistency != saved.get("consistency"):
            problems.append(f"{identity}: channel consistency differs from the rescan")
        decision_png = OUTPUT / "attempts" / identity / "decision-frame.png"
        first_png = Path(native_root) / "frame_000001.png"
        if decision_png.is_file() and first_png.is_file():
            if decision_png.read_bytes() != first_png.read_bytes():
                problems.append(f"{identity}: sealed decision frame differs from the "
                                "shot's first frame")
    return problems


def validate(args):
    plan = load_plan()
    check_plan_bound(plan)
    require_prepared(plan)
    problems = []
    # Protocol chronology: every version retained, terminal frozen before its run.
    for version in range(1, plan["version"]):
        if not plan_path(version).is_file():
            problems.append(f"plan-v{version}.json missing (chronology must be retained)")
    dev = plan.get("smoke_evidence") or {}
    if not dev.get("cells"):
        problems.append("terminal changelog lacks the reviewed smoke evidence")
    # The completion membership must equal the #87 ledger's typed-failure set.
    ledger87 = issue87_ledger()
    failed87 = sorted(identity for identity, entry in ledger87["cells"].items()
                      if identity.startswith("oracle--") and entry.get("failure") is not None)
    if sorted(cell["identity"] for cell in plan["completion_cells"]) != failed87:
        problems.append("completion membership differs from the issue-87 typed-failure set")
    earliest = None
    for cell in plan["completion_cells"]:
        marker = OUTPUT / "markers" / f"{cell['identity']}.json"
        if marker.is_file():
            stamp = read(marker).get("dispatched_at_utc")
            if stamp and (earliest is None or stamp < earliest):
                earliest = stamp
    if earliest is not None and plan["frozen_at_utc"] > earliest:
        problems.append("terminal plan freeze timestamp postdates its scoring run dispatch")
    # Recompute every published table.
    result = publication(plan)
    if compact_report(result) != read(OUTPUT / "summary.json"):
        problems.append("summary.json")
    if comparisons_csv(result).encode("utf-8") != (OUTPUT / "comparisons.csv").read_bytes():
        problems.append("comparisons.csv")
    if findings_md(result) != (OUTPUT / "findings.md").read_text(encoding="utf-8"):
        problems.append("findings.md")
    problems.extend(recompute_completion_channel(plan))
    if (DATA / "manifest.json").is_file():
        try:
            check_gallery(plan, read(DATA / "manifest.json"))
        except ValueError as error:
            problems.append(f"gallery: {error}")
    else:
        problems.append("gallery manifest.json: missing")
    if problems:
        raise ValueError(f"published issue-89 artifacts differ from bound source evidence: "
                         f"{sorted(set(problems))[:12]} ({len(set(problems))} distinct)")
    log("exact saved-evidence validation passed: every published table recomputed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            return dry_run(args)
        if args.smoke_test:
            return smoke_test(args)
        if args.prepare:
            return prepare(args)
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
