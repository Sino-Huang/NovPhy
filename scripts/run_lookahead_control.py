"""Issue-92 ADD-EXP WP1b: lookahead control (decision-only endpoint arms).

Binding runner module: scripts/run_lookahead_control.py
Exact validation command: python -u -m scripts.run_lookahead_control --validate

BLOCKING package of ticket #92.  Zero engine seconds, zero rendering, zero
retraining: decision-only rollouts of the frozen issue-77 N1 dynamics
checkpoints from the sealed #87 decision anchors to two frozen endpoints, with
the same argmin-predicted-count-cost rule (ties to the lower ordinal), bound
to the #87+#89 union engine-truth verdict table by candidate identity.

Why this package exists (verified from retained records and code): the #87
decision rule rolls each candidate only pair.delta carrier steps (1 agent
frame for *-fixed-h1, 25 for *-fixed-h5), while the engine-truth outcomes it
is scored against resolve 97.8-400.9 agent frames after the same sealed
decision frame.  No arm in the #87 record ever rolls far enough to reach the
moment the outcome is decided, so "frozen model-based systems never select a
successful candidate" is confounded with a lookahead of ~0.3-7% of the
outcome horizon.  This ticket adds the endpoint arms: every candidate rolled
to the 225-step deployment endpoint and to the 600-step (601-frame)
observed-window endpoint, fixed and learned-adaptive horizons, continuous and
hybrid abstractions.

#87 is terminal/version 3 and admits no outcome-conditioned modification, so
this is a separately frozen ticket: own plan identity, own smoke gate, #87's
anchors inherited verbatim, #87's 0.5 margin reused (not a new tunable).

Frozen-protocol chronology (binding shared rule):
- --dry-run is a no-write structural inventory that computes NO statistic;
- --smoke runs the pre-freeze controls (one #87 replication control asserting
  the chosen ordinal and the full ranking byte-match the retained #87
  decision record; one continuous-adaptive and one hybrid-adaptive cell
  asserting anchor sha256 equality, recorded horizon traces, union-verdict
  resolution, and B=1-reference equivalence of the batched implementation at
  a frozen float tolerance) and writes smoke.json; STOP RULE: the terminal
  freeze aborts if any control differs;
- --prepare writes the complete protocol (arms, endpoints, decision rule,
  binding, margins, disposition mapping, guards, caps, smoke evidence) into
  plan.json BEFORE any experiment statistic is computed;
- --run executes the 16-arm ladder once (ledger-resumable; existing records
  are re-read and schema-verified, never silently recomputed);
- --publish renders summary.json / comparisons.csv / findings.md /
  slot_join.csv / receipts.json;
- --validate re-derives every published table from the retained records and
  the union verdict table and byte-compares (no fresh rollouts).

Verdict vocabulary (frozen exactly): supported /
not_supported_by_this_experiment / readiness_or_precision_insufficient.
Every interval is DESCRIPTIVE.  `nonfinite recursive carrier` at long
endpoints is a pre-declared typed failure: retained, never retried, inherits
no success value.

Claim boundary: the adaptive arm selects its own prediction horizon and
abstraction at each rollout step from a controller trained by dynamic
programming on the same frozen world-model corpus; it does not learn from
task outcomes, does not adapt within or across episodes, and chooses only
among the frozen candidate inventory.  The record therefore bounds
decision-time horizon adaptivity, not policy learning.  No method or model
changes; #64/#65 stay sealed; closed-loop claims stay single-shot,
decision-only, development-lineage scoped; no engine access.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-92-lookahead-control-v1"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"

EIGHTY_SEVEN = ROOT / ".local-artifacts/issue-87-closed-loop-oracle-v1"
EIGHTY_SEVEN_IDENTITY = "issue-87-closed-loop-oracle-v1"
EIGHTY_NINE = ROOT / ".local-artifacts/issue-89-oracle-completion-v1"
NINETY = ROOT / ".local-artifacts/issue-90-oracle-timeout-audit-v1"
DYNAMICS = ROOT / ".local-artifacts/issue-77-n1-dynamics-v1"
DYNAMICS_IDENTITY = "issue-77-n1-dynamics-v1"

SCHEMA_PLAN = "issue_92_lookahead_control_plan_v1"
SCHEMA_COMPUTE = "issue_92_lookahead_control_compute_v1"
SCHEMA_REPORT = "issue_92_lookahead_control_report_v1"
SCHEMA_RECORD = "issue_92_lookahead_control_decision_v1"
SCHEMA_RECEIPTS = "issue_92_lookahead_control_receipts_v1"
IDENTITY = "issue-92-lookahead-control-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_lookahead_control --validate"

SEEDS = (20260908, 20260909, 20260910)
FAMILIES = ("continuous", "hybrid")
FIXED_DELTAS = (1, 5, 15)
MODES = ("fixed-h1", "fixed-h5", "fixed-h15", "adaptive")
ENDPOINTS = (225, 600)
ENDPOINT_NAMES = {225: "e225", 600: "e600"}
DEVICE = "cuda"

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7201
INTERVAL_QUANTILES = (0.025, 0.975)
INTERVAL_LABEL = "DESCRIPTIVE"
MARGIN_RATE = 0.5  # #87's own margin, reused verbatim (not a new tunable)
NONFINITE_SHARE_GUARD = 0.10
COST_TOLERANCE = 1e-3

WALL_CAP_SECONDS = 3600.0
GPU_CAP_SECONDS = 3.0 * 3600.0
DERIVED_BYTES_BUDGET = 50 * 1024 ** 2
STOP_TOKEN = "readiness_or_precision_insufficient"
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")

SMOKE_STATE = "issue-77-n1-001-a00"
SMOKE_SEED = 20260908

CLAIM_BOUNDARY_ADAPTIVE = (
    "The adaptive arm selects its own prediction horizon and abstraction at each "
    "rollout step from a controller trained by dynamic programming on the same frozen "
    "world-model corpus; it does not learn from task outcomes, does not adapt within or "
    "across episodes, and chooses only among the frozen candidate inventory. The record "
    "therefore bounds decision-time horizon adaptivity, not policy learning.")

POWER_NOTE = (
    "each sealed ceiling state has one successful ordinal among 9-13, so a single arm's "
    "0/36 has p approximately (12/13)^36 = 0.056 under a uniform-selection null; the "
    "informative statistic is the joint ordinal-band evidence, never a single arm's "
    "count")

CLAIM_BOUNDARY = (
    "decision-only rollouts of the frozen issue-77 N1 dynamics checkpoints from the "
    "sealed #87 anchors; no engine access, no rendering, no retraining, no gameplay; "
    "#87/#89/#90 published verdicts are inputs and are never recomputed, amended, or "
    "reinterpreted; #64/#65 stay sealed; single-shot, decision-only, "
    "development-lineage scoped; the cohort-v2 CEMPlanner gameplay-planning path is "
    "not admissible on this budget (it proposes continuous actions outside the frozen "
    "inventory, so the candidate-identity binding fails, and it binds a different "
    "predictor lineage); intervals are DESCRIPTIVE")

LIMITATIONS = (
    "decision-only endpoint arms: the rollout never touches the engine, so the arms "
    "measure selection from imagined rollouts, not executed gameplay",
    "the 225-step endpoint reaches the outcome horizon on only a subset of the "
    "successful slots; the 600-step endpoint covers all of them (coverage published "
    "from the retained engine-channel event steps)",
    "the adaptive arm bounds decision-time horizon adaptivity, not policy learning "
    "(claim-boundary sentence carried verbatim)",
    "the union verdict table is single-shot per (state, ordinal); per-slot execution "
    "variance is unobserved beyond the measured determinism statement of #92 WP1",
    "no content hash is recomputed after the freeze and no full-corpus integrity pass "
    "runs anywhere; inputs are verified at read time by their carried "
    "schema/identity/plan_identity fields",
)

PLACEHOLDER_MARKERS = ("TBD", "placeholder", "to be frozen", "XXX", "FIXME")


def log(message):
    print(f"[issue-92-lookahead-control] {message}", flush=True)


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
         "identity_fields": ["identity", "version", "status"],
         "role": "membership, inventories, dynamics-checkpoint bindings, decision "
                 "cells; anchors inherited verbatim"},
        {"name": "issue_87_records", "artifact": str(EIGHTY_SEVEN / "records"),
         "schema": "issue_87_decision_record_v1",
         "identity_fields": ["plan_identity", "plan_version", "schema"],
         "role": "the union verdict table (#87 side) and the replication-control "
                 "reference record"},
        {"name": "issue_87_attempts", "artifact": str(EIGHTY_SEVEN / "attempts"),
         "schema": "directory",
         "identity_fields": [],
         "role": "the sealed decision-frame.png anchors (read at smoke/run; byte-"
                 "verified against the retained anchor sha256 at smoke)"},
        {"name": "issue_89_records", "artifact": str(EIGHTY_NINE / "records"),
         "schema": "issue_89_oracle_completion_record_v1",
         "identity_fields": ["plan_identity", "plan_version", "schema"],
         "role": "the union verdict table (#89 completion side)"},
        {"name": "issue_89_summary", "artifact": str(EIGHTY_NINE / "summary.json"),
         "schema": "issue_89_oracle_completion_report_v1",
         "identity_fields": ["identity", "plan_identity", "schema"],
         "role": "the sealed ceiling-state set and typed-unmeasurable declaration, "
                 "cited; never recomputed"},
        {"name": "issue_77_n1_dynamics", "artifact": str(DYNAMICS),
         "schema": "directory",
         "identity_fields": [],
         "role": "the frozen predictor.pt/controller.pt checkpoints (payload binding "
                 "verified at load: plan identity, seed, arm, component)"},
    ]


def verify_source_bindings(plan):
    declared = [(entry["name"], entry["artifact"]) for entry in declared_sources()]
    frozen = [(entry["name"], entry["artifact"]) for entry in plan["inputs"]]
    if frozen != declared:
        raise ValueError("frozen plan input bindings differ from the declared sources")


def arm_ladder():
    return [{"family": family, "mode": mode, "endpoint": endpoint,
             "identity": f"{family}-{mode}-{ENDPOINT_NAMES[endpoint]}"}
            for family in FAMILIES for mode in MODES for endpoint in ENDPOINTS]


def frozen_plan(frozen_at, smoke_evidence):
    return {
        "schema": SCHEMA_PLAN,
        "identity": IDENTITY,
        "version": 1,
        "role": "terminal",
        "frozen_at": frozen_at,
        "frozen_before_scoring_run": True,
        "changelog": [{
            "version": 1,
            "change": "initial and terminal freeze: arms, endpoints, decision rule, "
                      "binding, margins, disposition mapping, guards, caps and the "
                      "smoke evidence are published before any experiment statistic "
                      "exists",
            "why": "standard freeze rule of the shared ADD-EXP protocol; #87 is "
                   "terminal/version 3 and admits no outcome-conditioned modification, "
                   "so the lookahead control is a separately frozen ticket",
            "evidence": "smoke.json controls passed and are embedded below; no "
                        "selection statistic has been computed at freeze time",
        }],
        "protocol_policy": (
            "single terminal protocol version: the full protocol is frozen with actual "
            "numeric margins before the scoring run and executed exactly once; the dry "
            "run computes no statistic; --validate re-derives every published table "
            "from the retained records"),
        "motivation": (
            "the #87 decision rule rolls each candidate only pair.delta carrier steps "
            "(1 agent frame for *-fixed-h1, 25 for *-fixed-h5) while the engine-truth "
            "outcomes resolve 97.8-400.9 agent frames after the same sealed decision "
            "frame; no #87 arm ever rolls far enough to reach the moment the outcome "
            "is decided, so the zero-selection result is confounded with a lookahead "
            "of ~0.3-7% of the outcome horizon; this ticket adds decision-only "
            "endpoint arms at the frozen 225-step deployment endpoint and the "
            "600-step (601-frame) observed-window endpoint"),
        "question": (
            "when every candidate is rolled to the endpoint that reaches the outcome "
            "horizon, does any frozen arm — fixed or learned-adaptive, continuous or "
            "hybrid — select the engine-truth successful candidate on the sealed "
            "ceiling states?"),
        "universe": {
            "states": ("the 23 sealed anchor-bearing states of the #87 membership; "
                       "issue-77-n1-005-a07 is excluded under #89's frozen "
                       "typed-unmeasurable declaration (no executed oracle anchor)"),
            "seeds": list(SEEDS),
            "arms": arm_ladder(),
            "ceiling_cells_per_arm": 36,
            "cells_total": 16 * 23 * 3,
        },
        "decision_rule": {
            "rollout": ("every candidate of the state's frozen #87 inventory is rolled "
                        "from the sealed anchor carrier to the arm's endpoint in "
                        "carrier steps: fixed arms use PredictionPair(delta, "
                        "Abstraction.CONTINUOUS) for delta in {1, 5, 15} (the hybrid "
                        "checkpoint runs its continuous-mode pair, exactly as #87's "
                        "hybrid-fixed-h1); adaptive arms choose the pair per step from "
                        "the frozen MatchedController over the same checkpoint "
                        "(fixed_pair=None semantics of "
                        "world_model.training.matched_dynamics.rollout); the action "
                        "tensor is the #87 normalization (drag/480, release_ms/1000, "
                        "tap_ms/1000, 1.0); the predicted cost is the frozen "
                        "TaskObjective count cost of the endpoint carrier"),
            "selection": ("argmin predicted cost over finite candidates, ties to the "
                          "lower ordinal (the #87 rule); a candidate whose carrier "
                          "goes nonfinite is a pre-declared typed failure: retained, "
                          "never retried, inherits no success value; a cell whose "
                          "candidates all fail is typed "
                          "all_candidate_predictions_failed"),
            "implementation": ("batched per-(state, arm, seed) execution, verified at "
                               "smoke to reproduce the B=1 reference "
                               "(matched_dynamics.rollout / #87 ReactiveSelector) on "
                               "the smoke cells"),
            "endpoints": {
                "225": "the frozen 225-step deployment endpoint",
                "600": "the 600-step endpoint reaching the 601-frame observed window",
            },
        },
        "binding": {
            "verdict_table": ("the #87+#89 union engine-truth verdict table keyed by "
                              "(state, ordinal), built read-only from the retained "
                              "oracle records (identical construction to #92 WP1's "
                              "frozen join)"),
            "rule": ("the arm's closed-loop outcome on a ceiling cell is the "
                     "engine-truth verdict first_shot_success of its chosen candidate, "
                     "bound by candidate identity (the #87 binding rule)"),
            "ceiling_states": ("the 12 sealed ceiling states cited from #89's "
                               "published ceiling_reestimate.per_state (indicator 1, "
                               "not typed-unmeasurable); never recomputed"),
            "success_band": ("the N1 success band = sorted union of engine-truth "
                             "successful ordinals over the sealed ceiling states, "
                             "computed from the union verdict table (the WP1 "
                             "definition)"),
        },
        "statistics": {
            "pooled_rate": ("per arm: pooled successful-selection rate over the 12 "
                            "sealed ceiling states x 3 seeds = 36 cells"),
            "interval": (f"member-clustered percentile bootstrap over the 7 ceiling "
                         f"members, {BOOTSTRAP_DRAWS} draws, PCG64 seed "
                         f"{BOOTSTRAP_SEED}, quantiles {list(INTERVAL_QUANTILES)}, "
                         "labelled DESCRIPTIVE"),
            "ordinal_band_evidence": ("per arm: chosen-ordinal histogram over the 36 "
                                      "ceiling cells and the share inside the success "
                                      "band; published with the power note; the joint "
                                      "band evidence is the informative statistic, "
                                      "never a single arm's count"),
            "endpoint_coverage": ("per successful slot (13 over the sealed ceiling "
                                  "states and the excluded 005-a07): the outcome "
                                  "resolution offset in agent frames from the retained "
                                  "engine-channel pig-removal event steps "
                                  "((event fixed_step - 30000)/50); per endpoint, the "
                                  "share of slots whose resolution is reached"),
            "compute_accounting": ("per arm: transition calls, controller calls, "
                                   "linear MACs, GPU seconds, wall seconds; receipts "
                                   "per cell"),
        },
        "margin": {"rate": MARGIN_RATE,
                   "rule": "#87's own 0.5 margin reused verbatim; not a new tunable"},
        "disposition_mapping": {
            "supported": ("every endpoint arm's pooled successful-selection rate over "
                          "the 36 ceiling cells is < 0.5 AND its member-clustered "
                          "descriptive interval lies entirely below 0.5"),
            "not_supported_by_this_experiment": ("any endpoint arm attains a pooled "
                                                 "rate >= 0.5 (the spine is then "
                                                 "restated as 'short-horizon reactive "
                                                 "rankers fail; endpoint planning "
                                                 "recovers X' — a reframe, not a "
                                                 "retraction)"),
            "readiness_or_precision_insufficient": ("anchor or verdict resolution "
                                                    "failure on any ceiling cell, or "
                                                    "nonfinite-rollout typed failures "
                                                    "beyond the frozen share guard, or "
                                                    "a precision guard tripped, or any "
                                                    "arm below 0.5 whose interval "
                                                    "covers 0.5 with no arm at 0.5"),
        },
        "precision_guards": {
            "anchors_resolved": 23,
            "records_written": 16 * 23 * 3,
            "ceiling_cells_per_arm": 36,
            "ceiling_verdict_resolution": 1.0,
            "nonfinite_share_max": NONFINITE_SHARE_GUARD,
            "rule": ("each guard is evaluated on the retained records; any failed guard "
                     "maps the disposition to readiness_or_precision_insufficient "
                     "before the rate rules are read"),
        },
        "smoke_cells": {
            "replication_control": {
                "cell": f"continuous-fixed-h1--seed{SMOKE_SEED}--{SMOKE_STATE}",
                "assertion": ("fresh rollout via the exact #87 ReactiveSelector path "
                              "byte-matches the retained #87 decision record: chosen "
                              "ordinal and every ranking row (ordinal, predicted_cost, "
                              "excluded)")},
            "adaptive_cells": [{
                "cell": f"continuous-adaptive-{ENDPOINT_NAMES[225]}--seed{SMOKE_SEED}--{SMOKE_STATE}",
                "assertions": ("anchor sha256 equality (file hash == retained record "
                               "anchor), recorded horizon trace covering the endpoint, "
                               "chosen ordinal resolved in the union verdict table, "
                               "and B=1-reference equivalence of the batched "
                               "implementation (matched_dynamics.rollout): identical "
                               "exclusion sets, identical chosen ordinal, endpoint "
                               f"costs within {COST_TOLERANCE:.0e}"),
            }, {
                "cell": f"hybrid-adaptive-{ENDPOINT_NAMES[225]}--seed{SMOKE_SEED}--{SMOKE_STATE}",
                "assertions": "same as the continuous-adaptive cell",
            }],
            "stop_rule": ("abort before the terminal freeze if any control differs; the "
                          "smoke evidence is embedded in this plan"),
            "evidence": smoke_evidence,
        },
        "power_note": POWER_NOTE,
        "claim_boundary_adaptive": CLAIM_BOUNDARY_ADAPTIVE,
        "caps": {
            "wall_cap_seconds": WALL_CAP_SECONDS,
            "wall_scope": "cumulative measured wall of the run phase; the dry run, the "
                          "smoke and the freeze publish no experiment wall time",
            "gpu_cap_seconds": GPU_CAP_SECONDS,
            "gpu_scope": "cumulative cuda-bracketed GPU seconds of the run phase "
                         "(3-GPU-hour cap)",
            "derived_bytes_budget": DERIVED_BYTES_BUDGET,
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
        raise ValueError(f"plan {path} is not the issue-92 lookahead-control protocol")
    blob = json_text(plan)
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    required = ("motivation", "question", "universe", "decision_rule", "binding",
                "statistics", "margin", "disposition_mapping", "precision_guards",
                "smoke_cells", "power_note", "claim_boundary_adaptive", "caps",
                "claim_boundary", "limitations", "validation_command", "inputs")
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
        raise ValueError("root plan.json missing; run --prepare to freeze the protocol "
                         "first (and --smoke before that)")
    return load_plan(path)


# ---------------------------------------------------------------------------
# read-only inputs and the rollout machinery
# ---------------------------------------------------------------------------

def load_inputs():
    from scripts import run_selection_validity_restatement as wp1
    plan87, decisions, oracles87 = wp1.load_issue_87()
    summary89, records89 = wp1.load_issue_89()
    verdicts, channel, typed_failures, unmeasured = wp1.build_verdicts(
        plan87, oracles87, records89)
    return plan87, decisions, summary89, verdicts


def ceiling_states(summary89):
    per_state = summary89["ceiling_reestimate"]["per_state"]
    return sorted(state for state, entry in per_state.items()
                  if entry["indicator"] == 1 and not entry["typed_unmeasurable"])


def sealed_states(plan87, decisions):
    """The 23 anchor-bearing states, from the retained decision records."""
    anchored = sorted({record["state_identity"] for record in decisions.values()
                       if not record.get("failure_kind")})
    states = [state for state in plan87["states"] if state["identity"] in set(anchored)]
    return states


def anchors_for(states, decisions):
    """#87's anchors inherited verbatim from the retained decision records."""
    anchors = {}
    for state in states:
        record = next(record for record in decisions.values()
                      if record["state_identity"] == state["identity"]
                      and not record.get("failure_kind"))
        anchor = record["anchor"]
        frame = EIGHTY_SEVEN / "attempts" / anchor["cell"] / "decision-frame.png"
        anchors[state["identity"]] = {
            "cell": anchor["cell"], "sha256": anchor["sha256"],
            "observation_identity": anchor["observation_identity"],
            "fixed_step": anchor["fixed_step"],
            "fixed_time_seconds": anchor["fixed_time_seconds"],
            "frame_path": frame,
        }
    return anchors


def load_adapter(device):
    from scripts import run_issue_71_hybrid_readiness as old
    return old.repair.load_repaired_adapter(old.repair.ROOT, device)


def load_objective():
    from world_model.planning.task_objective import TaskObjective
    return TaskObjective.from_vocabulary(
        read_json(DYNAMICS / "plan.json")["contract"]["vocabulary"])


def load_predictor(plan87, seed, family, device):
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    return e85.load_frozen_predictor(plan87, seed, family, device)


def load_controller(seed, family, device):
    from world_model.training.matched_dynamics import MatchedController
    path = DYNAMICS / f"seed-{seed}" / family / "controller.pt"
    if not path.is_file():
        raise ValueError(f"missing frozen controller {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    stored = payload.get("binding")
    expected = read_json(DYNAMICS / "plan.json")
    if (not isinstance(stored, dict) or stored.get("plan_identity") != expected["identity"]
            or stored.get("seed") != seed or stored.get("arm") != family
            or stored.get("component") != "controller"):
        raise ValueError(f"controller binding differs for seed={seed} arm={family}")
    controller = MatchedController(family == "continuous").to(torch.device(device))
    controller.load_state_dict(payload["model"], strict=True)
    return controller.eval()


def encode_carrier(adapter, anchor, device):
    from world_model.data.deployment_temporal import (AgentObservation,
                                                      TemporalObservationContext)
    observation = AgentObservation(anchor["observation_identity"],
                                   anchor["fixed_step"], anchor["fixed_time_seconds"],
                                   Path(anchor["frame_path"]).read_bytes(), "agent")
    with torch.no_grad():
        parsed = adapter.parse_batch((observation,))[0]
        carrier = adapter.build_from_parsed(
            TemporalObservationContext(None, observation), parsed, None).tensor
    return carrier


def action_tensor(action, device):
    return torch.tensor([[action["drag_x"] / 480., action["drag_y"] / 480.,
                          action["release_time_ms"] / 1000.,
                          action["tap_time_ms"] / 1000., 1.]], device=device)


def rollout_candidates(model, controller, objective, carrier, inventory, arm, device):
    """Batched decision rollout to the arm's endpoint; per-candidate typed failures.

    Mirrors matched_dynamics.rollout semantics exactly (residual carrier steps,
    controller pair choice with delta>remaining masked inside the controller);
    verified at smoke to reproduce the B=1 reference on the smoke cells.
    """
    from world_model.model import Abstraction, PredictionPair
    from world_model.training.matched_dynamics import pairs_for
    steps = arm["endpoint"]
    z = carrier[None].to(device).expand(len(inventory), -1).clone()
    actions = torch.cat([action_tensor(item["action"], device) for item in inventory], 0)
    fixed = None
    if arm["mode"] != "adaptive":
        fixed = PredictionPair(int(arm["mode"].split("-h")[1]), Abstraction.CONTINUOUS)
        if fixed not in pairs_for(model) or steps % fixed.delta:
            raise ValueError("unsupported pair or incomplete fixed horizon")
    pairs = pairs_for(model)
    alive = torch.ones(len(inventory), dtype=torch.bool, device=z.device)
    positions = torch.zeros(len(inventory), dtype=torch.long, device=z.device)
    traces = [[] for _ in inventory]
    transition_calls = controller_calls = 0
    torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            if fixed is not None:
                chosen = torch.full((len(index),), pairs.index(fixed),
                                    dtype=torch.long, device=z.device)
            else:
                remaining = steps - positions[index]
                logits = controller(z[index], actions[index], remaining)
                chosen = logits.argmax(-1)
                controller_calls += len(index)
            z_next = z
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next = z_next.clone()
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                transition_calls += len(rows)
                for row in rows.tolist():
                    traces[row].append({"start_fixed_step": int(positions[row]),
                                        "horizon": pair.delta,
                                        "mode": str(pair.abstraction)})
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < steps)
    torch.cuda.synchronize()
    gpu_seconds = time.monotonic() - began
    rows = []
    for slot, item in enumerate(inventory):
        failed = not bool(torch.isfinite(z[slot]).all())
        cost = None if failed else float(objective(z[slot]))
        rows.append({"ordinal": item["ordinal"], "predicted_cost": cost,
                     "excluded": failed,
                     "trace": traces[slot] if arm["mode"] == "adaptive" else None})
    return rows, {"transition_calls": transition_calls,
                  "controller_calls": controller_calls,
                  "gpu_seconds": gpu_seconds}



def choose(ranking):
    finite = [row for row in ranking if row["predicted_cost"] is not None]
    if not finite:
        return None, "all_candidate_predictions_failed"
    best = min(finite, key=lambda row: (row["predicted_cost"], row["ordinal"]))
    return best["ordinal"], None


# ---------------------------------------------------------------------------
# smoke controls (pre-freeze)
# ---------------------------------------------------------------------------

def run_smoke(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log("smoke: pre-freeze controls; STOP RULE: the terminal freeze aborts if any "
        "control differs")
    from world_model.model import Abstraction, PredictionPair
    from world_model.training import matched_dynamics
    plan87, decisions, summary89, verdicts = load_inputs()
    state = next(s for s in plan87["states"] if s["identity"] == SMOKE_STATE)
    anchors = anchors_for([state], decisions)
    anchor = anchors[SMOKE_STATE]
    adapter = load_adapter(DEVICE)
    objective = load_objective()
    carrier = encode_carrier(adapter, anchor, DEVICE)
    checks = []

    # 1. replication control: exact #87 ReactiveSelector path, byte-match
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    model = load_predictor(plan87, SMOKE_SEED, "continuous", DEVICE)
    selector = e85.ReactiveSelector("continuous-fixed-h1", SMOKE_SEED, model,
                                    objective, DEVICE)
    with torch.no_grad():
        fresh = selector.choose(carrier, state["inventory"])
    retained = next(record for record in decisions.values()
                    if record["cell"]["identity"]
                    == f"decision--continuous-fixed-h1--seed{SMOKE_SEED}--{SMOKE_STATE}")
    retained_decision = retained["decision"]
    fresh_rows = [(row["ordinal"], row["predicted_cost"], row["excluded"])
                  for row in fresh["ranking"]]
    retained_rows = [(row["ordinal"], row["predicted_cost"], row["excluded"])
                     for row in retained_decision["ranking"]]
    replication_ok = (fresh["chosen"]["ordinal"] == retained_decision["chosen"]["ordinal"]
                      and fresh_rows == retained_rows)
    checks.append({"control": "replication_control",
                   "cell": f"continuous-fixed-h1--seed{SMOKE_SEED}--{SMOKE_STATE}",
                   "chosen_ordinal_fresh": fresh["chosen"]["ordinal"],
                   "chosen_ordinal_retained": retained_decision["chosen"]["ordinal"],
                   "ranking_byte_match": fresh_rows == retained_rows,
                   "ok": replication_ok})
    log(f"smoke replication control: chosen {fresh['chosen']['ordinal']} vs retained "
        f"{retained_decision['chosen']['ordinal']}; ranking byte-match "
        f"{fresh_rows == retained_rows}")

    # anchor sha256 equality (file hash == retained record anchor)
    file_sha = sha256_of(anchor["frame_path"])
    anchor_ok = file_sha == anchor["sha256"] == retained["carrier_sha256"]
    checks.append({"control": "anchor_sha256_equality", "file_sha256": file_sha,
                   "record_anchor_sha256": anchor["sha256"], "ok": anchor_ok})
    log(f"smoke anchor sha256 equality: {anchor_ok}")

    # 2./3. adaptive cells: continuous + hybrid, endpoint 225
    for family in FAMILIES:
        arm = {"family": family, "mode": "adaptive", "endpoint": 225,
               "identity": f"{family}-adaptive-e225"}
        controller = load_controller(SMOKE_SEED, family, DEVICE)
        fam_model = model if family == "continuous" else load_predictor(
            plan87, SMOKE_SEED, family, DEVICE)
        ranking, timing = rollout_candidates(fam_model, controller, objective, carrier,
                                             state["inventory"], arm, DEVICE)
        chosen, failure = choose(ranking)
        # horizon trace covers the endpoint
        trace_ok = all(row["trace"] is not None
                       and sum(step["horizon"] for step in row["trace"]) == 225
                       for row in ranking if not row["excluded"])
        # union verdict resolution
        verdict_ok = (SMOKE_STATE, chosen) in verdicts if chosen is not None else False
        # batched vs B=1 reference: same exclusions, same chosen ordinal, costs within
        # the frozen float tolerance (kernel-scheduling noise compounds over steps)
        reference_ok = True
        reference_costs = []
        max_abs_delta = 0.0
        z0 = carrier[None].to(DEVICE)
        for item, row in zip(state["inventory"], ranking):
            try:
                z_end, _trace = matched_dynamics.rollout(
                    fam_model, controller, z0, action_tensor(item["action"], DEVICE),
                    steps=225, fixed_pair=None)
                reference = float(objective(z_end[0]))
            except ValueError as error:
                if "nonfinite" not in str(error):
                    raise
                reference = None
            reference_costs.append(reference)
            if (reference is None) != row["excluded"]:
                reference_ok = False
            elif reference is not None:
                max_abs_delta = max(max_abs_delta, abs(reference - row["predicted_cost"]))
        reference_chosen, _reference_failure = choose([
            {"ordinal": row["ordinal"], "predicted_cost": cost}
            for row, cost in zip(ranking, reference_costs)])
        reference_ok = (reference_ok and max_abs_delta <= COST_TOLERANCE
                        and reference_chosen == chosen)
        cell_ok = trace_ok and verdict_ok and reference_ok and chosen is not None
        checks.append({
            "control": f"adaptive_cell_{family}",
            "cell": f"{family}-adaptive-e225--seed{SMOKE_SEED}--{SMOKE_STATE}",
            "chosen_ordinal": chosen, "failure": failure,
            "trace_covers_endpoint": trace_ok,
            "union_verdict_resolved": verdict_ok,
            "first_shot_success": verdicts.get((SMOKE_STATE, chosen)),
            "batched_equals_reference": reference_ok,
            "reference_chosen_ordinal": reference_chosen,
            "max_abs_cost_delta": max_abs_delta,
            "cost_tolerance": COST_TOLERANCE,
            "reference_costs": reference_costs,
            "batched_costs": [row["predicted_cost"] for row in ranking],
            "horizon_histogram": dict(sorted(Counter(
                step["horizon"] for row in ranking if row["trace"]
                for step in row["trace"]).items())),
            "gpu_seconds": timing["gpu_seconds"],
            "ok": cell_ok})
        log(f"smoke {family}-adaptive cell: chosen {chosen} (reference "
            f"{reference_chosen}); trace {trace_ok}; verdict {verdict_ok}; "
            f"max|delta| {max_abs_delta:.3e} <= {COST_TOLERANCE:.0e}: {reference_ok}")
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": "issue_92_lookahead_control_smoke_v1", "identity": IDENTITY,
                "run_at": utc_now(), "device": DEVICE, "state": SMOKE_STATE,
                "seed": SMOKE_SEED, "checks": checks, "ok": ok}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; evidence at {output / 'smoke.json'}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the terminal "
                         "freeze")
    return evidence


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def load_smoke(output):
    path = Path(output) / "smoke.json"
    if not path.is_file():
        raise ValueError("smoke.json missing; run --smoke before --prepare")
    evidence = read_json(path)
    if not evidence.get("ok"):
        raise ValueError("smoke controls failed; STOP RULE: no terminal freeze")
    return evidence


def prepare(output):
    output = Path(output)
    plan_path = output / "plan.json"
    if plan_path.is_file():
        frozen = load_plan(plan_path)
        log(f"existing frozen protocol validated (version {frozen['version']}, frozen "
            f"{frozen['frozen_at']}); no statistic was computed")
        return frozen
    smoke = load_smoke(output)
    frozen = frozen_plan(utc_now(), smoke)
    frozen["inputs"] = [
        {**entry, "sha256_at_freeze": (
            sha256_of(Path(entry["artifact"])) if Path(entry["artifact"]).is_file()
            else f"directory:{sum(1 for _ in Path(entry['artifact']).rglob('*.json'))} json files")}
        for entry in frozen["inputs"]]
    write_json(plan_path, frozen)
    log(f"frozen protocol published at {plan_path}: version {frozen['version']}, "
        f"{len(frozen['universe']['arms'])} arms, margin "
        f"{frozen['margin']['rate']}; no experiment statistic was computed")
    return frozen


def cell_identity(arm, seed, state):
    return f"decision--{arm['identity']}--seed{seed}--{state}"


def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": "issue_92_lookahead_control_ledger_v1", "identity": IDENTITY,
            "status": "running", "cells": {}, "wall_seconds_elapsed": 0.0,
            "gpu_seconds_elapsed": 0.0}


def run_ladder(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    ledger = load_ledger(output)
    began = time.monotonic()
    plan87, decisions, summary89, verdicts = load_inputs()
    states = sealed_states(plan87, decisions)
    if len(states) != 23:
        raise ValueError(f"{STOP_TOKEN}: sealed anchor-bearing states "
                         f"{len(states)} != 23")
    anchors = anchors_for(states, decisions)
    adapter = load_adapter(DEVICE)
    arms = [dict(arm) for arm in plan["universe"]["arms"]]
    models = {}
    controllers = {}

    def model_for(seed, family):
        key = (seed, family)
        if key not in models:
            models[key] = load_predictor(plan87, seed, family, DEVICE)
        return models[key]

    def controller_for(seed, family):
        key = (seed, family)
        if key not in controllers:
            controllers[key] = load_controller(seed, family, DEVICE)
        return controllers[key]
    objective = load_objective()

    total = len(arms) * len(states) * len(SEEDS)
    done = 0
    for state in states:
        if ledger["wall_seconds_elapsed"] + (time.monotonic() - began) >= WALL_CAP_SECONDS:
            raise ValueError(f"{STOP_TOKEN}: wall_cap_exceeded")
        carrier = encode_carrier(adapter, anchors[state["identity"]], DEVICE)
        for seed in SEEDS:
            for arm in arms:
                identity = cell_identity(arm, seed, state["identity"])
                record_path = output / "records" / f"{identity}.json"
                if record_path.is_file():
                    retained = read_json(record_path)
                    if (retained.get("schema") != SCHEMA_RECORD
                            or retained.get("plan_identity") != IDENTITY):
                        raise ValueError(f"retained record {record_path} schema binding "
                                         "differs")
                    done += 1
                    continue
                model = model_for(seed, arm["family"])
                controller = (controller_for(seed, arm["family"])
                              if arm["mode"] == "adaptive" else None)
                cell_started = time.monotonic()
                ranking, timing = rollout_candidates(
                    model, controller, objective, carrier, state["inventory"], arm,
                    DEVICE)
                chosen, failure = choose(ranking)
                resolved = ((state["identity"], chosen) in verdicts
                            if chosen is not None else False)
                record = {
                    "schema": SCHEMA_RECORD,
                    "plan_identity": IDENTITY,
                    "plan_version": plan["version"],
                    "cell": {"identity": identity, "arm": arm["identity"],
                             "family": arm["family"], "mode": arm["mode"],
                             "endpoint": arm["endpoint"], "seed": seed,
                             "state": state["identity"]},
                    "cell_kind": "decision",
                    "anchor": {"cell": anchors[state["identity"]]["cell"],
                               "sha256": anchors[state["identity"]]["sha256"],
                               "fixed_step": anchors[state["identity"]]["fixed_step"]},
                    "decision": {
                        "candidate_count": len(state["inventory"]),
                        "chosen_ordinal": chosen,
                        "failure": failure,
                        "selected_predicted_cost": next(
                            (row["predicted_cost"] for row in ranking
                             if row["ordinal"] == chosen), None),
                        "ranking": [{"ordinal": row["ordinal"],
                                     "predicted_cost": row["predicted_cost"],
                                     "excluded": row["excluded"]} for row in ranking],
                        "horizon_trace": {str(row["ordinal"]): row["trace"]
                                          for row in ranking
                                          if row["trace"] is not None} or None,
                        "transition_calls": timing["transition_calls"],
                        "controller_calls": timing["controller_calls"],
                        "gpu_seconds": timing["gpu_seconds"],
                    },
                    "binding": {
                        "rule": ("the arm's closed-loop outcome is the engine-truth "
                                 "verdict of its chosen candidate from the #87+#89 "
                                 "union verdict table, bound by candidate identity"),
                        "chosen_branch_identity": next(
                            (item["branch_identity"] for item in state["inventory"]
                             if item["ordinal"] == chosen), None),
                        "verdict_resolved": resolved,
                        "first_shot_success": verdicts.get((state["identity"], chosen)),
                    },
                    "wall_seconds": time.monotonic() - cell_started,
                    "issue_64_authorized": False,
                }
                write_json(record_path, record)
                ledger["cells"][identity] = {
                    "status": "terminal" if failure is None else "failed",
                    "failure": failure, "wall_seconds": record["wall_seconds"],
                    "gpu_seconds": timing["gpu_seconds"]}
                ledger["wall_seconds_elapsed"] += record["wall_seconds"]
                ledger["gpu_seconds_elapsed"] += timing["gpu_seconds"]
                done += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.monotonic() - began
                    eta = elapsed / done * (total - done) if done else 0.0
                    log(f"ladder {done}/{total}; elapsed {elapsed:.0f}s; eta {eta:.0f}s; "
                        f"gpu={ledger['gpu_seconds_elapsed']:.0f}s")
                    write_json(output / "ledger.json", ledger)
    ledger["status"] = "terminal"
    write_json(output / "ledger.json", ledger)
    if ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: gpu_cap_exceeded "
                         f"({ledger['gpu_seconds_elapsed']:.0f}s)")
    log(f"ladder complete: {done} cells, wall {time.monotonic() - began:.0f}s, gpu "
        f"{ledger['gpu_seconds_elapsed']:.0f}s")
    return ledger


# ---------------------------------------------------------------------------
# published tables
# ---------------------------------------------------------------------------

def load_records(output):
    records = {}
    for path in sorted((Path(output) / "records").glob("decision--*.json")):
        record = read_json(path)
        if record.get("schema") != SCHEMA_RECORD:
            raise ValueError(f"record {path} schema binding differs")
        records[record["cell"]["identity"]] = record
    return records


def bootstrap_interval(values, clusters):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    cluster_ids = sorted(grouped)
    means = np.array([np.mean(grouped[cluster]) for cluster in cluster_ids])
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(cluster_ids),
                               size=(BOOTSTRAP_DRAWS, len(cluster_ids)))
    estimates = means[draws].mean(axis=1)
    low, high = np.quantile(estimates, INTERVAL_QUANTILES)
    return {"mean": float(means.mean()), "interval": [float(low), float(high)],
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "quantiles": list(INTERVAL_QUANTILES), "label": INTERVAL_LABEL,
            "clusters": len(cluster_ids), "units": len(values)}


def compute_tables(plan, output):
    plan87, decisions, summary89, verdicts = load_inputs()
    ceilings = ceiling_states(summary89)
    state_member = {state["identity"]: state["source_member"]
                    for state in plan87["states"]}
    records = load_records(output)
    success_band = sorted({ordinal for state in ceilings
                           for (st, ordinal), hit in verdicts.items()
                           if st == state and hit})
    # endpoint coverage from the retained engine-channel event steps: for each
    # (state, ordinal) keep the successful execution (pig_removed > 0) from either
    # pool; the #89 completion re-executions supersede #87's timed-out records.
    resolution_offsets = []
    best = {}
    for directory in (EIGHTY_SEVEN / "records", EIGHTY_NINE / "records"):
        for path in sorted(directory.glob("oracle--*.json")):
            record = read_json(path)
            channel = record.get("engine_channel") or {}
            events = channel.get("pig_removed_events") or []
            if not channel.get("pig_removed") or not events:
                continue
            key = (record["cell"]["state"], record["cell"]["ordinal"])
            if key in best:
                continue
            best[key] = min(event["fixed_step"] for event in events)
    for (state, ordinal), step in sorted(best.items()):
        resolution_offsets.append({
            "state": state, "ordinal": ordinal, "fixed_step": step,
            "agent_frames": (step - 30000) / 50.0})
    resolution_offsets.sort(key=lambda row: (row["state"], row["ordinal"]))
    coverage = {str(endpoint): {
        "reached": sum(1 for row in resolution_offsets
                       if row["agent_frames"] <= endpoint),
        "slots": len(resolution_offsets)}
        for endpoint in ENDPOINTS}

    arms = {}
    nonfinite = total_candidates = 0
    for arm in plan["universe"]["arms"]:
        cells = [record for record in records.values()
                 if record["cell"]["arm"] == arm["identity"]]
        ceiling = [record for record in cells if record["cell"]["state"] in ceilings]
        hits = [bool(record["binding"]["first_shot_success"]) for record in ceiling]
        clusters = [state_member[record["cell"]["state"]] for record in ceiling]
        unresolved = sum(1 for record in ceiling
                         if not record["binding"]["verdict_resolved"])
        chosen = [record["decision"]["chosen_ordinal"] for record in ceiling
                  if record["decision"]["chosen_ordinal"] is not None]
        in_band = sum(1 for ordinal in chosen if ordinal in success_band)
        failed = sum(1 for record in cells
                     if record["decision"]["failure"] is not None)
        excluded = sum(1 for record in cells for row in record["decision"]["ranking"]
                       if row["excluded"])
        candidates = sum(record["decision"]["candidate_count"] for record in cells)
        nonfinite += excluded
        total_candidates += candidates
        interval = bootstrap_interval([float(hit) for hit in hits], clusters)
        arms[arm["identity"]] = {
            "cells": len(cells), "ceiling_cells": len(ceiling),
            "hits": int(sum(hits)), "rate": float(np.mean(hits)) if hits else None,
            "interval": interval, "verdict_unresolved": unresolved,
            "chosen_histogram": {str(key): value for key, value in
                                 sorted(Counter(chosen).items())},
            "chosen_in_band": in_band, "typed_cell_failures": failed,
            "typed_candidate_failures": excluded,
            "transition_calls": sum(record["decision"]["transition_calls"]
                                    for record in cells),
            "controller_calls": sum(record["decision"]["controller_calls"]
                                    for record in cells),
            "gpu_seconds": sum(record["decision"]["gpu_seconds"] for record in cells),
        }
    guards = {
        "anchors_resolved": {"value": len({record["cell"]["state"]
                                           for record in records.values()}),
                             "floor": 23,
                             "ok": len({record["cell"]["state"]
                                        for record in records.values()}) == 23},
        "records_written": {"value": len(records), "floor": 16 * 23 * 3,
                            "ok": len(records) == 16 * 23 * 3},
        "ceiling_cells_per_arm": {
            "value": min(arm["ceiling_cells"] for arm in arms.values()), "floor": 36,
            "ok": all(arm["ceiling_cells"] == 36 for arm in arms.values())},
        "ceiling_verdict_resolution": {
            "value": sum(arm["verdict_unresolved"] for arm in arms.values()), "floor": 0,
            "ok": all(arm["verdict_unresolved"] == 0 for arm in arms.values())},
        "nonfinite_share_max": {
            "value": nonfinite / max(1, total_candidates), "floor": NONFINITE_SHARE_GUARD,
            "ok": nonfinite / max(1, total_candidates) <= NONFINITE_SHARE_GUARD},
    }
    guards_ok = all(guard["ok"] for guard in guards.values())
    rates = {identity: arm["rate"] for identity, arm in arms.items()}
    uppers = {identity: arm["interval"]["interval"][1] for identity, arm in arms.items()}
    any_at_margin = any(rate is not None and rate >= MARGIN_RATE for rate in rates.values())
    all_below = all(rate is not None and rate < MARGIN_RATE
                    and upper < MARGIN_RATE
                    for rate, upper in zip(rates.values(), uppers.values()))
    if not guards_ok:
        disposition = STOP_TOKEN
    elif any_at_margin:
        disposition = "not_supported_by_this_experiment"
    elif all_below:
        disposition = "supported"
    else:
        disposition = STOP_TOKEN
    return {
        "schema": SCHEMA_COMPUTE,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "arms": arms,
        "success_band": success_band,
        "ceiling_states": ceilings,
        "resolution_offsets": resolution_offsets,
        "endpoint_coverage": coverage,
        "precision_guards": guards,
        "disposition": disposition,
        "disposition_conditions": {"any_arm_at_margin": any_at_margin,
                                   "all_arms_below_with_interval": all_below,
                                   "guards_ok": guards_ok},
        "power_note": POWER_NOTE,
        "claim_boundary_adaptive": CLAIM_BOUNDARY_ADAPTIVE,
    }


def slot_join_csv(compute, output):
    records = load_records(output)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["cell_identity", "arm", "family", "mode", "endpoint", "seed",
                     "state", "chosen_ordinal", "predicted_cost", "typed_failure",
                     "typed_candidate_failures", "in_success_band",
                     "first_shot_success", "transition_calls", "controller_calls",
                     "gpu_seconds"])
    band = set(compute["success_band"])
    for identity in sorted(records):
        record = records[identity]
        cell = record["cell"]
        decision = record["decision"]
        chosen = decision["chosen_ordinal"]
        writer.writerow([
            identity, cell["arm"], cell["family"], cell["mode"], cell["endpoint"],
            cell["seed"], cell["state"], chosen if chosen is not None else "",
            decision["selected_predicted_cost"]
            if decision["selected_predicted_cost"] is not None else "",
            decision["failure"] or "",
            sum(1 for row in decision["ranking"] if row["excluded"]),
            (chosen in band) if chosen is not None else "",
            record["binding"]["first_shot_success"]
            if record["binding"]["first_shot_success"] is not None else "",
            decision["transition_calls"], decision["controller_calls"],
            f"{decision['gpu_seconds']:.6f}"])
    return buffer.getvalue()


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["arm", "family", "mode", "endpoint", "ceiling_cells", "hits",
                     "rate", "interval_low", "interval_high", "interval_label",
                     "margin", "chosen_in_band", "chosen_histogram",
                     "typed_candidate_failures", "transition_calls",
                     "controller_calls", "gpu_seconds"])
    for identity in sorted(compute["arms"]):
        arm = compute["arms"][identity]
        family, mode, endpoint = identity.rsplit("-", 2)
        writer.writerow([
            identity, family, mode, endpoint, arm["ceiling_cells"], arm["hits"],
            f"{arm['rate']:.6f}", f"{arm['interval']['interval'][0]:.6f}",
            f"{arm['interval']['interval'][1]:.6f}", INTERVAL_LABEL, MARGIN_RATE,
            arm["chosen_in_band"],
            "{" + ", ".join(f"{int(key)}: {arm['chosen_histogram'][key]}"
                             for key in sorted(arm["chosen_histogram"], key=int)) + "}",
            arm["typed_candidate_failures"], arm["transition_calls"],
            arm["controller_calls"], f"{arm['gpu_seconds']:.3f}"])
    return buffer.getvalue()


def findings_md(plan, compute, timings):
    lines = []
    add = lines.append
    add("# Issue-92 WP1b: lookahead control (decision-only endpoint arms) — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan schema `{SCHEMA_PLAN}` version "
        f"{plan['version']} (terminal), frozen {plan['frozen_at']} before any "
        "experiment statistic")
    add(f"- validation command: `{VALIDATION_COMMAND}`")
    add(f"- zero engine seconds, zero rendering, zero retraining; GPU flock held for "
        f"every measured phase")
    add("")
    add("## 1. The confound this package closes")
    add("")
    add(plan["motivation"] + ".")
    add("")
    add("Endpoint coverage (measured from the retained engine-channel pig-removal "
        "event steps):")
    for endpoint in ENDPOINTS:
        coverage = compute["endpoint_coverage"][str(endpoint)]
        add(f"- endpoint {endpoint}: reaches the outcome resolution on "
            f"{coverage['reached']} of {coverage['slots']} successful slots")
    offsets = ", ".join(f"{row['state']}/a{row['ordinal']:02d} "
                        f"{row['agent_frames']:.1f}"
                        for row in compute["resolution_offsets"])
    add(f"- resolution offsets (agent frames after the sealed decision frame): {offsets}")
    add("")
    add("## 2. Smoke evidence (pre-freeze; stop rule held)")
    add("")
    smoke = plan["smoke_cells"]["evidence"]
    for check in smoke["checks"]:
        add(f"- {check['control']}: ok {check['ok']}")
    add("")
    add("## 3. The endpoint ladder over the 36 ceiling cells per arm")
    add("")
    add(f"Success band (N1, from the union verdict table): {compute['success_band']}; "
        f"{POWER_NOTE}.")
    add("")
    add("| arm | hits | rate | interval | chosen in band | chosen histogram | typed "
        "candidate failures | gpu s |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for identity in sorted(compute["arms"]):
        arm = compute["arms"][identity]
        histogram = "{" + ", ".join(
            f"{int(key)}: {arm['chosen_histogram'][key]}"
            for key in sorted(arm["chosen_histogram"], key=int)) + "}"
        add(f"| {identity} | {arm['hits']}/{arm['ceiling_cells']} | {arm['rate']:.4f} | "
            f"[{arm['interval']['interval'][0]:.4f}, "
            f"{arm['interval']['interval'][1]:.4f}] | {arm['chosen_in_band']} | "
            f"{histogram} | {arm['typed_candidate_failures']} | "
            f"{arm['gpu_seconds']:.1f} |")
    add("")
    add("## 4. Disposition")
    add("")
    add(f"**{compute['disposition']}** (any arm at the 0.5 margin: "
        f"{compute['disposition_conditions']['any_arm_at_margin']}; all arms below "
        f"0.5 with the interval below: "
        f"{compute['disposition_conditions']['all_arms_below_with_interval']}; guards "
        f"ok: {compute['disposition_conditions']['guards_ok']}).")
    add("")
    if compute["disposition"] == "not_supported_by_this_experiment":
        add("Restatement (pre-declared): \"short-horizon reactive rankers fail; "
            "endpoint planning recovers X\" — a reframe, not a retraction.")
        add("")
    add("## 5. Claim boundary (carried verbatim)")
    add("")
    add(CLAIM_BOUNDARY_ADAPTIVE)
    add("")
    add(CLAIM_BOUNDARY + ".")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    return "\n".join(lines)


def build_report(plan, compute, timings, output):
    return {
        "schema": SCHEMA_REPORT,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "plan_version": plan["version"],
        "frozen_at": plan["frozen_at"],
        "validation_command": VALIDATION_COMMAND,
        "question": plan["question"],
        "motivation": plan["motivation"],
        "arms": compute["arms"],
        "success_band": compute["success_band"],
        "ceiling_states": compute["ceiling_states"],
        "resolution_offsets": compute["resolution_offsets"],
        "endpoint_coverage": compute["endpoint_coverage"],
        "precision_guards": compute["precision_guards"],
        "disposition": compute["disposition"],
        "disposition_conditions": compute["disposition_conditions"],
        "power_note": POWER_NOTE,
        "claim_boundary_adaptive": CLAIM_BOUNDARY_ADAPTIVE,
        "smoke_evidence": plan["smoke_cells"]["evidence"],
        "caps": plan["caps"],
        "claim_boundary": plan["claim_boundary"],
        "limitations": plan["limitations"],
        "compute": {"phases": timings,
                    "wall_seconds_total": sum(entry["seconds"]
                                              for entry in timings.values()),
                    "gpu_seconds_total": sum(arm["gpu_seconds"]
                                             for arm in compute["arms"].values())},
        "hygiene": {"content_hashes_recomputed_after_freeze": False,
                    "full_corpus_integrity_pass": False,
                    "engine_access": False},
    }


def publish(output):
    output = Path(output)
    plan = load_terminal_plan(output)
    ledger = load_ledger(output)
    if ledger.get("status") != "terminal":
        raise ValueError("ledger is not terminal; run --run before --publish")
    started = time.monotonic()
    compute = compute_tables(plan, output)
    timing = {"seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    timings = {"tables": timing}
    compute["phases"] = timings
    report = build_report(plan, compute, timings, output)
    write_json(output / "compute.json", compute)
    write_json(output / "summary.json", report)
    write_text(output / "comparisons.csv", comparisons_csv(compute))
    write_text(output / "findings.md", findings_md(plan, compute, timings))
    write_text(output / "slot_join.csv", slot_join_csv(compute, output))
    write_json(output / "receipts.json", {
        "schema": SCHEMA_RECEIPTS,
        "identity": IDENTITY,
        "plan_identity": plan["identity"],
        "ledger": {"wall_seconds_elapsed": ledger["wall_seconds_elapsed"],
                   "gpu_seconds_elapsed": ledger["gpu_seconds_elapsed"],
                   "cells": len(ledger["cells"])},
        "phases": timings,
        "derived_bytes": file_bytes(output),
        "derived_bytes_budget": DERIVED_BYTES_BUDGET,
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
        "summary.json": json_text(build_report(plan, fresh, timings, output)).encode(),
        "comparisons.csv": comparisons_csv(fresh).encode(),
        "findings.md": findings_md(plan, fresh, timings).encode(),
        "slot_join.csv": slot_join_csv(fresh, output).encode(),
    }
    for name, content in regenerated.items():
        path = output / name
        if not path.is_file():
            raise ValueError(f"published artifact missing: {name}")
        if path.read_bytes() != content:
            raise ValueError(f"published {name} differs from the recomputation from the "
                             "retained records")
    log(f"exact recomputation validation passed: {len(fresh['arms'])} arms re-derived "
        "from the retained records, every published table byte-compared; no fresh "
        "rollout ran and no content hash was recomputed "
        f"({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    output = Path(output)
    log("no-write dry run; structural inventory only; no rollout, no statistic")
    plan87, decisions, summary89, verdicts = load_inputs()
    states = sealed_states(plan87, decisions)
    log(f"#87 plan version {plan87['version']} status {plan87['status']}; sealed "
        f"anchor-bearing states {len(states)} (expected 23)")
    ceilings = ceiling_states(summary89)
    log(f"sealed ceiling states {len(ceilings)} (cited from #89; never recomputed)")
    checkpoints = sorted(DYNAMICS.glob("seed-*/*/controller.pt"))
    predictors = sorted(DYNAMICS.glob("seed-*/*/predictor.pt"))
    log(f"dynamics checkpoints: {len(predictors)} predictors, {len(checkpoints)} "
        f"controllers (expected 6 each)")
    log(f"arm ladder: {len(arm_ladder())} arms x {len(states)} states x "
        f"{len(SEEDS)} seeds = {len(arm_ladder()) * len(states) * len(SEEDS)} cells")
    plan_path = Path(output) / "plan.json"
    if plan_path.is_file():
        plan = load_plan(plan_path)
        log(f"frozen protocol present: version {plan['version']} frozen "
            f"{plan['frozen_at']}")
    else:
        smoke_path = Path(output) / "smoke.json"
        log(f"frozen protocol absent; smoke evidence present: {smoke_path.is_file()}")
    log("dry run complete; nothing was written and no statistic was computed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke", "prepare", "run", "publish", "validate"):
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
            Path(args.output).mkdir(parents=True, exist_ok=True)
            with GPULock():
                run_ladder(args.output)
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
