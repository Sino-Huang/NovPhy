"""Issue-93 ADD-EXP: within-state engine-truth selection AUC on a second
candidate parameterization.

Binding runner module: scripts/run_second_parameterization_probe.py
Exact validation command: python -u -m scripts.run_second_parameterization_probe --validate

Question: is the below-chance within-state AUC of the three frozen N1 rankers
(#92: cell-unit 0.4367, member-clustered 0.4180 over 7 members) a property of
their action discrimination, or an artifact of the monotone 1-D launch-angle
sweep?  Two new candidate inventories are executed once each in the real
rendered engine over the 15 N1 source members and scored by the same frozen
rankers from the retained #87 decision anchors:
- Arm B (primary, claim C22): 4x4 Cartesian drag grid, drag_x in
  {-40,-80,-120,-160} x drag_y in {5,30,55,80}; WP0 found that the engine
  clamps every drag beyond 18.49 px to full launch speed, so in the engine the
  grid is 16 distinct launch angles at saturated power, non-monotone in the
  ordinal; the radius axis (40-179 px) exists on the ranker-input side only
  (owner decision after the WP0 stop comment: proceed, C22 reworded).
- Arm A (secondary, claim C23): offset angle sweep at radius 80 px, 11 angles
  8.5..78.5 deg, disjoint from the 13 N1 samples.

Modes (frozen-protocol chronology, binding shared rule):
- --verify    WP0 pre-freeze verification, analysis only, zero engine seconds:
              training_action_support.json, anchor_retention.json,
              engine_action_check.json, members.json.  No outcome statistic.
- --smoke     WP1: one member x 2 candidates per arm in the engine plus a
              scoring reproduction of retained #87 rankings; infrastructure
              assertions only; records under smoke/, never joined to scoring.
- --prepare   freezes plan.json (inventories, states, rankers, estimands,
              frozen disposition rule, reading matrix, caps, source text)
              BEFORE any scored engine slot.
- --run       WP2 oracle pass: every scheduled slot once, <=4 staggered
              isolated workers, ledger-resumable; cap stop -> typed not_executed.
- --completion  the single declared completion pass (only if typed failures
              exceed 25% of slots; same slots only; once).
- --score     decision-only ranker scoring (3 systems x 3 seeds x members x
              arms) from the retained anchors; GPU seconds.
- --publish   WP3 summary.json / findings.md / slot_join.csv /
              comparisons.csv / compute.json.
- --validate  recomputes every published table from the retained records and
              byte-compares (no engine, no rollout).

Claim boundary: within-state engine-truth selection AUC of the three frozen N1
rankers on two new candidate inventories over 15 N1 source members; no other
models, no adaptation, no fresh gameplay, no sealed benchmark (#64/#65 stay
sealed), no state-difficulty claim; #87/#89/#90/#92 dispositions are inputs and
are not re-opened.  Every interval is DESCRIPTIVE.
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
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-93-second-parameterization-v1"
DATA = ROOT / "data/issue-93-second-parameterization"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"

EIGHTY = ROOT / ".local-artifacts/issue-80-reactive-diagnostic-v1"
EIGHTY_SEVEN = ROOT / ".local-artifacts/issue-87-closed-loop-oracle-v1"
EIGHTY_NINE = ROOT / ".local-artifacts/issue-89-oracle-completion-v1"
NINETY_TWO = ROOT / ".local-artifacts/issue-92-selection-validity-v1"
DYNAMICS = ROOT / ".local-artifacts/issue-77-n1-dynamics-v1"
CAMPAIGN = ROOT / ".local-artifacts/issue-77-n1-v1"
ISSUE71 = ROOT / ".local-artifacts/issue-71-hybrid-readiness-v1"
PLAYER_SOURCE = ROOT / ".local-artifacts/issue-76-canonical-player-v1/project/Assets"

IDENTITY = "issue-93-second-parameterization-v1"
SCHEMA_SUPPORT = "issue_93_training_action_support_v1"
SCHEMA_ANCHORS = "issue_93_anchor_retention_v1"
SCHEMA_ENGINE = "issue_93_engine_action_check_v1"
SCHEMA_MEMBERS = "issue_93_members_v1"
SCHEMA_PLAN = "issue_93_second_parameterization_plan_v1"
SCHEMA_ORACLE = "issue_93_oracle_record_v1"
SCHEMA_DECISION = "issue_93_decision_record_v1"
SCHEMA_LEDGER = "issue_93_ledger_v1"
SCHEMA_SMOKE = "issue_93_smoke_v1"
SCHEMA_COMPUTE = "issue_93_compute_v1"
SCHEMA_REPORT = "issue_93_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_second_parameterization_probe --validate"

MODEL_SYSTEMS = ("continuous-fixed-h1", "continuous-fixed-h5", "hybrid-fixed-h1")
SEEDS = (20260908, 20260909, 20260910)
ORACLE_SEED_LABEL = 20260908  # #74 matched training-seed label; never an engine input
DEVICE = "cuda"
ARMS = ("grid", "offset")
ARM_CLAIMS = {"grid": "C22", "offset": "C23"}

# execution bounds: #87 caps/ports/attempt limit per ticket #93, #89 cold-start
# mitigation (staggered dispatch, extended connect/readiness bounds, no probes)
WORKERS = 4
START_STAGGER_SECONDS = 45
ATTEMPT_SECONDS = 1200
DECISION_FIXED_STEP = 30000
NATIVE_STRIDE = 50
SHOT_SECONDS = 180
INFERENCE_HOLD_SECONDS = 2
AGENT_SOCKET_SECONDS = 600
AGENT_CONNECT_DEADLINE_SECONDS = 600
PHYSICS_SOCKET_SECONDS = 120
PREPARE_FOR_PLAY_SECONDS = 300
HISTORY_READY_SECONDS = 600
WORKER_RSS_MIB = 4096
AGGREGATE_RSS_MIB = 32768
WALL_CAP_SECONDS = 24 * 3600.0
GPU_CAP_SECONDS = 3 * 3600.0
ARTIFACT_BYTES_CAP = 150 * 1024 ** 3
MINIMUM_FREE_BYTES = 256 * 1024 ** 3

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 7201
INTERVAL_QUANTILES = (0.025, 0.975)
RULE = {"ceiling_members_min": 4, "verdict_coverage_min": 0.75,
        "supported_max_auc": 0.50, "not_supported_min_auc": 0.60}
COMPLETION_TRIGGER = 0.25
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")
STOP_TOKEN = "readiness_or_precision_insufficient"
EXPECTED = {"members": 15, "grid_slots": 240, "offset_slots": 165, "slots": 405,
            "decision_cells": 15 * 2 * 3 * 3}
SMOKE_MEMBER = "issue-77-n1-001"
SMOKE_ORDINALS = {"grid": (0, 15), "offset": (0, 10)}

CLAIMS = {
    "C22": ("On a Cartesian drag-grid inventory (non-monotone in the candidate ordinal; "
            "realized in the engine as 16 distinct launch angles at saturated launch "
            "speed, its pull-radius axis present on the ranker-input side only), the "
            "frozen rankers' within-state engine-truth selection AUC is at or below "
            "chance."),
    "C23": ("On an offset launch-angle sweep at radius 80 px (11 angles disjoint from "
            "the 13 N1 samples), the frozen rankers' within-state engine-truth selection "
            "AUC is at or below chance."),
}
READING_MATRIX = [
    {"row": 1, "C22": "supported", "C23": "supported",
     "reading": ("The failure is not specific to the 1-D sweep's sample or its "
                 "parameterization within these inventories; the scope limit narrows.")},
    {"row": 2, "C22": "not_supported_by_this_experiment", "C23": "supported",
     "reading": ("Failure is parameterization- or support-specific: the rankers fail "
                 "along the angle sweep but discriminate on the grid. The 1-D scope limit "
                 "becomes a finding.")},
    {"row": 3, "C22": "supported", "C23": "not_supported_by_this_experiment",
     "reading": ("The N1 AUC depends on which angle samples are in the inventory; the "
                 "headline becomes inventory-specific.")},
    {"row": 4, "C22": "not_supported_by_this_experiment",
     "C23": "not_supported_by_this_experiment",
     "reading": ("The N1 below-chance AUC is specific to the original 13 samples; the "
                 "headline must be re-scoped hard.")},
    {"row": 5, "C22": "any readiness_or_precision_insufficient", "C23": "-",
     "reading": ("Report descriptively in the appendix only; body scope sentences "
                 "unchanged except a pointer.")},
]
ENGINE_READING = (
    "the engine projects any drag beyond _dragRadius (1 world unit = 18.49 px at the "
    "fixed camera) onto the drag circle, so every candidate of both arms launches at the "
    "saturated speed 10 in its drag direction (234/234 retained #87 launches at "
    "9.9992-9.9997); the grid's engine-side variation is launch angle only")

SOURCE_FILES = (
    "scripts/run_second_parameterization_probe.py",
    "scripts/run_closed_loop_oracle_completion.py",
    "scripts/run_engine_outcome_reactive_diagnostic.py",
    "scripts/run_lookahead_control.py",
    "scripts/run_selection_validity_restatement.py",
    "scripts/issue_76_censored_episode.py",
    "scripts/issue_76_episode_capture.py",
    "scripts/slingshot_readiness.py",
    "world_model/training/engine_outcome_reactive_diagnostic.py",
)
PLACEHOLDER_MARKERS = ("TBD", "placeholder", "to be frozen", "XXX", "FIXME")

TAP_MS = 0
RELEASE_MS = 1000
ACTION_BOUNDS = {"drag_x": (-160, -10), "drag_y": (-80, 80)}

GRID_X = (-40, -80, -120, -160)
GRID_Y = (5, 30, 55, 80)
OFFSET_ANGLES = tuple(8.5 + 7 * k for k in range(12))  # 85.5 excluded by bounds rule
SWEEP_RADIUS = 80


def log(message):
    print(f"[issue-93-second-parameterization] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path):
    return json.loads(Path(path).read_text())


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json_text(value))
    temporary.replace(path)


def sha256_of(path):
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class GPULock:
    """Exclusive advisory lock held for one wall-time-measured phase."""

    def __enter__(self):
        self._handle = open(GPU_LOCK_PATH, "a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        log("shared GPU flock acquired for this measured phase")
        return self

    def __exit__(self, *exception):
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        log("shared GPU flock released")
        return False


# ---------------------------------------------------------------------------
# candidate inventories (fixed by ticket #93 before any engine execution)
# ---------------------------------------------------------------------------

def launch_geometry(drag_x, drag_y):
    return {"radius_px": math.hypot(drag_x, drag_y),
            "angle_deg": math.degrees(math.atan2(drag_y, -drag_x))}


def in_bounds(drag_x, drag_y):
    return (ACTION_BOUNDS["drag_x"][0] <= drag_x <= ACTION_BOUNDS["drag_x"][1]
            and ACTION_BOUNDS["drag_y"][0] <= drag_y <= ACTION_BOUNDS["drag_y"][1])


def grid_inventory():
    """Arm B: 4x4 Cartesian drag grid, ordinal = 4*i + j."""
    items = []
    for i, drag_x in enumerate(GRID_X):
        for j, drag_y in enumerate(GRID_Y):
            items.append({"ordinal": 4 * i + j,
                          "action": {"drag_x": drag_x, "drag_y": drag_y,
                                     "tap_time_ms": TAP_MS, "release_time_ms": RELEASE_MS}})
    return items


def offset_inventory():
    """Arm A: offset angle sweep at radius 80 px; out-of-bounds angles excluded by rule."""
    items, excluded = [], []
    for angle in OFFSET_ANGLES:
        drag_x = round(-SWEEP_RADIUS * math.cos(math.radians(angle)))
        drag_y = round(SWEEP_RADIUS * math.sin(math.radians(angle)))
        if not in_bounds(drag_x, drag_y):
            excluded.append({"angle_deg": angle, "drag_x": drag_x, "drag_y": drag_y,
                             "rule": "outside ACTION_BOUNDS drag_x in [-160, -10]"})
            continue
        items.append({"ordinal": len(items), "angle_nominal_deg": angle,
                      "action": {"drag_x": drag_x, "drag_y": drag_y,
                                 "tap_time_ms": TAP_MS, "release_time_ms": RELEASE_MS}})
    return items, excluded


ARM_PREFIX = {"grid": "g", "offset": "o"}


def arm_inventories():
    grid = grid_inventory()
    offset, excluded = offset_inventory()
    if len(grid) != 16 or len(offset) != 11 or len(excluded) != 1:
        raise ValueError("candidate inventories differ from the ticket-#93 design")
    return {"grid": grid, "offset": offset}, excluded


# ---------------------------------------------------------------------------
# WP0.1 training action support of the frozen rankers
# ---------------------------------------------------------------------------

def decode_action(row):
    values = [float(value) for value in row]
    return (round(values[0] * 480.), round(values[1] * 480.),
            round(values[3] * 1000.), round(values[2] * 1000.))  # (x, y, tap, release)


def shard_actions(path):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=True)
    actions = {}
    for window, row in zip(value["windows"], value["tensors"]["action"], strict=True):
        actions.setdefault(int(window["shot"]), decode_action(row))
    return value["record"], actions


def support_summary(actions):
    geometry = [launch_geometry(x, y) for x, y, _, _ in actions]
    return {
        "shots": len(actions),
        "unique_actions": len(set(actions)),
        "drag_x_range": [min(a[0] for a in actions), max(a[0] for a in actions)],
        "drag_y_range": [min(a[1] for a in actions), max(a[1] for a in actions)],
        "tap_ms_range": [min(a[2] for a in actions), max(a[2] for a in actions)],
        "release_ms_values": sorted({a[3] for a in actions}),
        "radius_px_range": [min(g["radius_px"] for g in geometry),
                            max(g["radius_px"] for g in geometry)],
        "angle_deg_range": [min(g["angle_deg"] for g in geometry),
                            max(g["angle_deg"] for g in geometry)],
    }


def training_action_support(inventories):
    """Enumerate the executed action of every training shot the frozen
    predictors were fit on (the retained training tensors are the truth)."""
    dynamics = read_json(DYNAMICS / "plan.json")
    if dynamics["identity"] != "issue-77-n1-dynamics-v1":
        raise ValueError("dynamics plan binding differs")
    issue71_plan = read_json(ISSUE71 / "plan.json")
    fit71 = [index for index in range(1, 3001) if index % 5]
    pools = {"issue71_predictor_pool": [], "n1_predictor_pool": []}
    for count, index in enumerate(fit71, 1):
        record, actions = shard_actions(ISSUE71 / "shards" / f"lineage-{index:04d}.pt")
        if record != issue71_plan["records"][index - 1]:
            raise ValueError(f"#71 shard {index} record binding differs")
        pools["issue71_predictor_pool"].extend(actions.values())
        if count % 400 == 0:
            log(f"WP0.1 #71 predictor-pool shards read {count}/{len(fit71)}")
    campaign = read_json(CAMPAIGN / "plan.json")
    declared = {branch["identity"]: branch["action"] for branch in campaign["branches"]}
    n1_branches = []
    for identity in dynamics["n1_lineages"]["predictor"]:
        record = dynamics["n1_lineages"]["records"][identity]
        path = DYNAMICS / "shards" / f"lineage-n1-{record['ordinal']:03d}.pt"
        stored, actions = shard_actions(path)
        if stored != record:
            raise ValueError(f"N1 shard {identity} record binding differs")
        for shot, action in sorted(actions.items()):
            branch = record["branches"][shot]
            spec = declared[branch]
            if (action[0], action[1], action[3]) != (spec["drag_x"], spec["drag_y"],
                                                     spec["release_time_ms"]):
                raise ValueError(f"N1 training tensor differs from declared action {branch}")
            n1_branches.append(branch)
        pools["n1_predictor_pool"].extend(actions.values())
    n1_set = sorted(set(pools["n1_predictor_pool"]))
    set71 = set(pools["issue71_predictor_pool"])
    n1_summary = support_summary(pools["n1_predictor_pool"])
    s71 = support_summary(pools["issue71_predictor_pool"])
    points71 = sorted({(a[0], a[1]) for a in set71})
    per_candidate = {}
    for arm, items in inventories.items():
        rows = []
        for item in items:
            action = item["action"]
            key = (action["drag_x"], action["drag_y"], action["tap_time_ms"],
                   action["release_time_ms"])
            geometry = launch_geometry(action["drag_x"], action["drag_y"])
            rows.append({
                "ordinal": item["ordinal"],
                "drag_x": action["drag_x"], "drag_y": action["drag_y"],
                "radius_px": geometry["radius_px"], "angle_deg": geometry["angle_deg"],
                "exact_in_n1_training": key in set(n1_set),
                "exact_in_issue71_training": key in set71,
                "radius_within_n1_training_range": (
                    n1_summary["radius_px_range"][0] <= geometry["radius_px"]
                    <= n1_summary["radius_px_range"][1]),
                "release_1000_training_shots_at_this_radius": sum(
                    1 for a in n1_set if a[3] == RELEASE_MS
                    and abs(math.hypot(a[0], a[1]) - geometry["radius_px"]) < 1.0),
                "nearest_n1_drag_distance_px": min(
                    math.hypot(action["drag_x"] - a[0], action["drag_y"] - a[1])
                    for a in n1_set),
                "nearest_issue71_drag_distance_px": min(
                    math.hypot(action["drag_x"] - x, action["drag_y"] - y)
                    for x, y in points71),
            })
        per_candidate[arm] = rows
    return {
        "schema": SCHEMA_SUPPORT, "identity": IDENTITY, "computed_at": utc_now(),
        "ranker_component": ("predictor only: ReactiveSelector applies model.carrier "
                             "Delta times under a fixed mode pair; the controller and "
                             "the parser take no action input"),
        "predictor_fit_pool": {
            "rule": ("scripts/run_issue_77_n1_train.py fitting_pool: #71 lineages "
                     "1..3000 with index % 5 != 0 plus n1_lineages.predictor"),
            "issue71_lineages": len(fit71),
            "n1_lineages": list(dynamics["n1_lineages"]["predictor"]),
            "n1_branches": n1_branches,
        },
        "parser_pretraining": ("issue-70-parser-repair-v6 CNN slot parser: frames only, "
                               "no action input (parser-only)"),
        "sources": {"issue71_predictor_pool": s71, "n1_predictor_pool": n1_summary},
        "n1_training_actions": [list(a) for a in n1_set],
        "action_tensor": ("[drag_x/480, drag_y/480, release_ms/1000, tap_ms/1000, 1] "
                          "(scripts/run_engine_outcome_reactive_diagnostic.py:603-606); "
                          "decoded by exact inverse and rounding"),
        "per_candidate": per_candidate,
        "reading": {
            "release_1000_support": (
                "every training shot with release_time_ms = 1000 (the N1 action contract "
                "used by both arms) is an N1 radius-80 sweep action; the #71 corpus holds "
                f"release {s71['release_ms_values']} only"),
        },
    }


# ---------------------------------------------------------------------------
# WP0.2 decision-anchor retention; WP0.4 member-unit state list
# ---------------------------------------------------------------------------

def load_issue_87():
    from scripts import run_selection_validity_restatement as wp1
    return wp1.load_issue_87()


def member_states(plan87):
    """15 source members; canonical state = lowest state identity (the duplicate
    pairs are byte-identical: #92 findings.md section 1)."""
    by_member = defaultdict(list)
    for state in plan87["states"]:
        by_member[state["source_member"]].append(state)
    members = []
    for member in sorted(by_member):
        states = sorted(by_member[member], key=lambda state: state["identity"])
        canonical = states[0]
        for other in states[1:]:
            if (other["execution_member"] != canonical["execution_member"]
                    or other["engine_seed"] != canonical["engine_seed"]):
                raise ValueError(f"duplicate states of {member} are not identical")
        members.append({"member": member, "canonical_state": canonical["identity"],
                        "duplicate_states": [state["identity"] for state in states[1:]],
                        "engine_seed": canonical["engine_seed"],
                        "generator_family": canonical["generator_family"],
                        "exposure_role": canonical["execution_member"]["exposure_role"]})
    return members


def anchor_retention(plan87, decisions, members):
    rows = []
    for entry in members:
        states = [entry["canonical_state"], *entry["duplicate_states"]]
        anchored = [record for record in decisions.values()
                    if record["state_identity"] in states and not record.get("failure_kind")]
        if not anchored:
            failures = sorted({record.get("failure") for record in decisions.values()
                               if record["state_identity"] in states})
            rows.append({"member": entry["member"], "retained": False,
                         "decision_failures": failures})
            continue
        anchor_frames = {record["anchor"]["sha256"] for record in anchored}
        record = next(record for record in anchored
                      if record["state_identity"] == entry["canonical_state"])
        anchor = record["anchor"]
        path = EIGHTY_SEVEN / "attempts" / anchor["cell"] / "decision-frame.png"
        present = path.is_file()
        digest = sha256_of(path) if present else None
        rows.append({
            "member": entry["member"], "retained": present and digest == anchor["sha256"],
            "state": entry["canonical_state"], "anchor_cell": anchor["cell"],
            "observation_identity": anchor["observation_identity"],
            "fixed_step": anchor["fixed_step"],
            "fixed_time_seconds": anchor["fixed_time_seconds"],
            "path": str(path), "sha256_recorded": anchor["sha256"], "sha256_file": digest,
            "carrier_sha256_values": sorted({record["carrier_sha256"] for record in anchored
                                             if record["state_identity"]
                                             == entry["canonical_state"]}),
            "distinct_anchor_frame_sha256_across_member_states": len(anchor_frames),
        })
    return {"schema": SCHEMA_ANCHORS, "identity": IDENTITY, "computed_at": utc_now(),
            "rule": ("the #87 decision records bind the decision-anchor observation by "
                     "anchor.observation_identity/sha256; the frame is retained at "
                     "attempts/<anchor.cell>/decision-frame.png"),
            "members": rows,
            "missing": [row["member"] for row in rows if not row["retained"]]}


# ---------------------------------------------------------------------------
# WP0.3 executed-launch check (engine/API source + retained #87 launches)
# ---------------------------------------------------------------------------

def source_line(relative, pattern):
    path = PLAYER_SOURCE / relative
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line):
            return {"file": str(path.relative_to(ROOT)), "line": number,
                    "text": line.strip()}
    raise ValueError(f"source pattern {pattern!r} not found in {path}")


def first_launch(native_root):
    from scripts.canonical_native_trace import NativeTrace
    for chunk in NativeTrace(native_root).chunks():
        for event in chunk["events"]:
            if event["event_type"] == "bird_launched":
                return event
    return None


def engine_action_check(plan87, oracles87, inventories):
    drag_radius_prefab = source_line(
        "Resources/prefabs/gameworld/characters/birds/BirdRed.prefab", r"_dragRadius:")
    drag_radius = float(drag_radius_prefab["text"].split(":")[1])
    citations = {
        "clamp": source_line("Scripts/Assembly-CSharp/ABBird.cs",
                             r"Vector2.Distance\(dragPosition, vector\) > _dragRadius"),
        "clamp_projection": source_line("Scripts/Assembly-CSharp/ABBird.cs",
                                        r"normalized \* _dragRadius \+ vector"),
        "launch_impulse": source_line("Scripts/Assembly-CSharp/ABBird.cs",
                                      r"Vector2 force = -vector2 \* _launchForce"),
        "max_launch_speed": source_line("Scripts/Assembly-CSharp/ABConstants.cs",
                                        r"BIRD_MAX_LANUCH_SPEED"),
        "hud_drag": source_line("Scripts/Assembly-CSharp/HUD.cs",
                                r"selectedBird.DragBird\(vector\)"),
        "red_bird_drag_radius": drag_radius_prefab,
    }
    inventory = {state["identity"]: {item["ordinal"]: item["action"]
                                     for item in state["inventory"]}
                 for state in plan87["states"]}
    scale = set()
    launches = []
    executed = [record for record in oracles87.values() if record.get("execution")]
    for count, record in enumerate(sorted(executed, key=lambda r: r["cell"]["identity"]), 1):
        attempt = EIGHTY_SEVEN / "attempts" / record["cell"]["identity"]
        segment = read_json(attempt / "shot-1" / "segment.json")
        reference = segment["action"]["slingshot_reference"]
        scale.add(round(reference["pixelsPerWorldUnit"], 6))
        event = first_launch(record["execution"]["native_root"])
        declared = inventory[record["cell"]["state"]][record["cell"]["ordinal"]]
        vx, vy = event["payload"]["launch_velocity"]
        launches.append({
            "cell": record["cell"]["identity"],
            "drag": [declared["drag_x"], declared["drag_y"]],
            "executed_drag": list(segment["action"]["drag_release"]),
            "declared_angle_deg": launch_geometry(declared["drag_x"],
                                                  declared["drag_y"])["angle_deg"],
            "launch_speed": math.hypot(vx, vy),
            "launch_angle_deg": math.degrees(math.atan2(vy, vx)),
        })
        if count % 50 == 0:
            log(f"WP0.3 retained #87 launches read {count}/{len(executed)}")
    if len(scale) != 1:
        raise ValueError(f"slingshot scale differs across #87 segments: {sorted(scale)}")
    pixels_per_unit = scale.pop()
    clamp_px = drag_radius * pixels_per_unit
    speeds = [row["launch_speed"] for row in launches]
    candidates = {}
    for arm, items in inventories.items():
        rows = []
        for item in items:
            geometry = launch_geometry(item["action"]["drag_x"], item["action"]["drag_y"])
            clamped = geometry["radius_px"] > clamp_px
            rows.append({"ordinal": item["ordinal"], **geometry,
                         "clamped_to_drag_radius": clamped,
                         "engine_launch_speed_expected": 10.0 if clamped
                         else 10.0 * geometry["radius_px"] / clamp_px})
        collisions = [[a["ordinal"], b["ordinal"]]
                      for index, a in enumerate(rows) for b in rows[index + 1:]
                      if abs(a["angle_deg"] - b["angle_deg"]) < 1e-9
                      and abs(a["engine_launch_speed_expected"]
                              - b["engine_launch_speed_expected"]) < 1e-9]
        candidates[arm] = {"rows": rows, "identical_launch_pairs": collisions,
                           "clamped": sum(row["clamped_to_drag_radius"] for row in rows),
                           "removed_by_rule": sorted({pair[1] for pair in collisions})}
    return {
        "schema": SCHEMA_ENGINE, "identity": IDENTITY, "computed_at": utc_now(),
        "source_citations": citations,
        "mechanics": (
            "HUD.Drag -> ABBird.DragBird projects any drag farther than _dragRadius (1 "
            "world unit, BirdRed prefab) onto the drag circle; LaunchBird applies an "
            "impulse proportional to the clamped displacement, so every drag beyond the "
            "circle launches at the same (maximum) speed in the drag direction"),
        "pixels_per_world_unit": pixels_per_unit,
        "clamp_radius_px": clamp_px,
        "retained_issue87_launches": {
            "cells": len(launches), "speed_min": min(speeds), "speed_max": max(speeds),
            "executed_equals_declared": all(row["executed_drag"] == row["drag"]
                                            for row in launches),
            "max_abs_angle_offset_deg": max(abs(row["launch_angle_deg"]
                                                - row["declared_angle_deg"])
                                            for row in launches),
            "rows": launches,
        },
        "candidates": candidates,
        "removal_rule": ("a candidate the engine realizes as an identical launch "
                         "(angle and speed) to a lower-ordinal candidate is removed "
                         "before freeze"),
    }


# ---------------------------------------------------------------------------
# scheduled cells
# ---------------------------------------------------------------------------

def member_inventory(member, arm, inventories):
    prefix = ARM_PREFIX[arm]
    return [{"ordinal": item["ordinal"],
             "branch_identity": f"{member}-{prefix}{item['ordinal']:02d}",
             "action": dict(item["action"])} for item in inventories[arm]]


def oracle_identity(member, arm, ordinal, kind="oracle"):
    return f"{kind}--{member}--{ARM_PREFIX[arm]}{ordinal:02d}"


def decision_identity(system, seed, member, arm):
    return f"decision--{system}--seed{seed}--{member}--{arm}"


def make_cell(state, arm, item, kind="oracle"):
    return {"identity": oracle_identity(state["member"], arm, item["ordinal"], kind),
            "arm": arm, "member": state["member"], "state": state["canonical_state"],
            "ordinal": item["ordinal"], "branch_identity": item["branch_identity"],
            "action": dict(item["action"]), "engine_seed": state["engine_seed"],
            "retained_anchor_path": (state["anchor"] or {}).get("path")}


def scheduled_cells(plan):
    """Arm B (primary) for every member first, then Arm A; ordinal order within."""
    return [make_cell(state, arm, item)
            for arm in ARMS for state in plan["states"]
            for item in plan["inventories"][arm]["members"][state["member"]]]


def smoke_cells(states, inventories):
    state = next(state for state in states if state["member"] == SMOKE_MEMBER)
    cells = []
    for arm in ARMS:
        items = member_inventory(SMOKE_MEMBER, arm, inventories)
        cells.extend(make_cell(state, arm, items[ordinal], "smoke")
                     for ordinal in SMOKE_ORDINALS[arm])
    return cells


# ---------------------------------------------------------------------------
# cell execution: one isolated, real-rendered execution of a frozen candidate
# ---------------------------------------------------------------------------

def plain_record(plan_identity, plan_version, cell):
    return {"schema": SCHEMA_ORACLE, "plan_identity": plan_identity,
            "plan_version": plan_version, "cell": cell,
            "cell_kind": cell["identity"].split("--", 1)[0],
            "state_identity": cell["state"], "seed": ORACLE_SEED_LABEL,
            "engine_seed": cell["engine_seed"], "failure": None, "failure_kind": None,
            "decision_frame": None, "frame_vs_retained_anchor": None, "execution": None,
            "executed_action": None, "launch": None, "outcome": None,
            "engine_channel": None, "fresh_scenario_lineage": False,
            "issue_64_authorized": False}


def execute_cell(payload):
    """The candidate's frozen action from the member's frozen initial condition
    under the frozen scenario authority, decision anchor (native fixed step
    30000) and engine seed; engine-truth outcome, decision frame and executed
    launch only (no predictor scoring inside the attempt)."""
    from scripts import run_closed_loop_oracle_completion as c89
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    c89.torch.set_num_threads(2)
    output = Path(payload["output"])
    cell = payload["cell"]
    identity = cell["identity"]
    state = payload["state"]
    attempt = Path(payload["attempts"]) / identity
    attempt.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = plain_record(payload["plan_identity"], payload["plan_version"], cell)
    bridge = physics = engine_process = display_process = None
    variables = ("DISPLAY", "XDG_DATA_HOME", "NOVPHY_PHYSICS_CAPTURE_PORT",
                 "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE",
                 "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT",
                 "NOVPHY_ENVIRONMENT_SEED", "NOVPHY_NATIVE_DECISION_STEP")
    environment = {key: os.environ.get(key) for key in variables}
    try:
        member = state["execution_member"]
        if state["engine_seed"] != cell["engine_seed"] or member["engine_seed"] != cell["engine_seed"]:
            raise ValueError("scheduled engine seed differs from the frozen state")
        action = cell["action"]
        game = attempt / "runtime"
        c89.clone_player(Path(payload["player_root"]), game)
        c89.live_files.install_level(game, member)
        _, scenario = c89.live_files.materialize(member, member["template"],
                                                 attempt / "authority")
        if scenario.to_dict() != member["scenario"]:
            raise ValueError("member differs from its frozen scenario authority")
        display, display_process = c89.start_display(attempt / "display.log")
        agent_port, game_port, physics_port = c89.reserve_ports(3)
        aligned = attempt / "aligned"
        os.environ.update(
            DISPLAY=display, XDG_DATA_HOME=str(attempt / "xdg"),
            NOVPHY_PHYSICS_CAPTURE_PORT=str(physics_port),
            NOVPHY_PHYSICS_CAPTURE_V2_STRIDE=str(NATIVE_STRIDE),
            NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT=str(aligned),
            NOVPHY_ENVIRONMENT_SEED=str(cell["engine_seed"]),
            NOVPHY_NATIVE_DECISION_STEP=str(DECISION_FIXED_STEP))
        record["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port}
        engine_process = c89.start_engine(game, False, agent_port=agent_port,
                                          game_port=game_port, physics_port=physics_port)
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
        c89.prepare_action(bridge, action)
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
        frame_bytes = (attempt / "decision-1" / ref["relative_path"]).read_bytes()
        (attempt / "decision-frame.png").write_bytes(frame_bytes)
        record["decision_frame"] = {
            "frame_identity": ref["identity"], "fixed_step": frames[-1]["fixed_step"],
            "fixed_time_seconds": frames[-1]["fixed_time_seconds"],
            "sha256": c89.bytes_identity(frame_bytes), "observed_frames": policy.frames,
            "path": str(attempt / "decision-frame.png"),
            "decision_compute": "none (frozen candidate; scoring is a separate phase)"}
        if cell.get("retained_anchor_path"):
            retained = Path(cell["retained_anchor_path"]).read_bytes()
            record["frame_vs_retained_anchor"] = {
                "retained_anchor_path": cell["retained_anchor_path"],
                "byte_equal": retained == frame_bytes}
        segment_started = time.monotonic()
        segment = c89.capture_segment(bridge, aligned, attempt / "shot-1", member, scenario,
                                      identity + ":shot-1", action, SHOT_SECONDS)
        engine_seconds = time.monotonic() - segment_started
        executed = segment["action"]
        record["executed_action"] = {"drag_release": list(executed["drag_release"]),
                                     "release_time": executed["release_time"],
                                     "tap_time": executed["tap_time"],
                                     "socket_command": segment["socket_command"]}
        if (list(executed["drag_release"]) != [action["drag_x"], action["drag_y"]]
                or executed["release_time"] != action["release_time_ms"]
                or executed["tap_time"] != action["tap_time_ms"]):
            raise ValueError("executed action differs from the declared candidate action")
        initial_png = (Path(segment["native_root"]) / "frame_000001.png").read_bytes()
        if (segment["summary"]["first_fixed_step"] != DECISION_FIXED_STEP
                or initial_png != frame_bytes):
            raise ValueError("executed shot pre-intervention state differs from its "
                             "decision frame")
        verdict = e85.channel_scan(segment["native_root"])
        if verdict["bird_launches"] != 1:
            raise ValueError("executed shot did not contain exactly one native launch")
        event = first_launch(segment["native_root"])
        vx, vy = event["payload"]["launch_velocity"]
        record["launch"] = {"velocity": [vx, vy], "speed": math.hypot(vx, vy),
                            "angle_deg": math.degrees(math.atan2(vy, vx)),
                            "fixed_step": event["fixed_step"]}
        summary = segment["summary"]
        terminal = c89.terminal_evidence(segment)
        record["engine_channel"] = verdict
        record["execution"] = {
            "segment_identity": segment["identity"], "native_root": segment["native_root"],
            "observation_manifest": segment["observation_manifest"],
            "first_fixed_step": summary["first_fixed_step"],
            "last_fixed_step": summary["last_fixed_step"],
            "frame_count": segment["frame_count"], "censored": summary["censored"],
            "terminal_reason": None if terminal is None else terminal["reason"],
            "engine_wall_seconds": engine_seconds}
        record["outcome"] = {"first_shot_success": bool(verdict["pig_removed"])}
    except Exception as error:  # typed terminal failure; never retried silently
        record["failure"] = f"{type(error).__name__}: {error}"
        record["failure_kind"] = "execution_failure"
    finally:
        actions = [("connection.disconnect", connection.disconnect)
                   for connection in (bridge, physics) if connection is not None]
        actions.append(("stop_started_engine", lambda: c89.stop_started_engine(engine_process)))
        if display_process is not None:
            actions.append(("display.terminate", lambda: c89.terminate(display_process)))

        def restore_environment():
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        actions.append(("environment.restore", restore_environment))
        cleanup_failures = []
        for name, step in actions:
            try:
                step()
            except BaseException as cleanup_error:
                cleanup_failures.append(f"{name}: {type(cleanup_error).__name__}: {cleanup_error}")
        if cleanup_failures:
            record["cleanup_failures"] = cleanup_failures
        shutil.rmtree(attempt / "runtime", ignore_errors=True)
    record["wall_seconds"] = time.monotonic() - started
    write_json(output / payload["records"] / f"{identity}.json", record)
    print(f"issue-93 cell {identity} complete={record['failure'] is None} "
          f"success={(record['outcome'] or {}).get('first_shot_success')} "
          f"failure={record['failure']}", flush=True)
    return record


# ---------------------------------------------------------------------------
# supervisor: <=4 isolated workers, staggered cold starts, caps, resumable ledger
# ---------------------------------------------------------------------------

def free_disk(path):
    return shutil.disk_usage(path).free


def tree_bytes(path):
    path = Path(path)
    return sum(entry.stat().st_size for entry in path.rglob("*")
               if entry.is_file()) if path.exists() else 0


def supervise(output, attempts, plan_meta, cells, states, phase, records="records",
              ledger_name="ledger.json"):
    from scripts import run_closed_loop_oracle_completion as c89
    output, attempts = Path(output), Path(attempts)
    for sub in (records, "markers", "receipts"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    attempts.mkdir(parents=True, exist_ok=True)
    ledger_file = output / ledger_name
    ledger = read_json(ledger_file) if ledger_file.is_file() else {
        "schema": SCHEMA_LEDGER, "identity": IDENTITY, "phase": phase, "status": "running",
        "cells": {}, "wall_seconds_elapsed": 0.0, "stop_reason": None}
    pending = [cell for cell in cells if not (output / records / f"{cell['identity']}.json").is_file()]
    recovered = [cell["identity"] for cell in pending
                 if (output / "markers" / f"{cell['identity']}.json").is_file()]
    if recovered:
        log(f"crash recovery: {len(recovered)} dispatched-but-unrecorded cells re-queued "
            f"(no outcome was recorded): {recovered[:4]}")
        for identity in recovered:
            shutil.rmtree(attempts / identity, ignore_errors=True)
    if not pending:
        ledger["status"] = "complete"
        write_json(ledger_file, ledger)
        log(f"{phase}: all {len(cells)} scheduled cells already terminal")
        return ledger
    player = Path(plan_meta["player_root"])
    for name in ("9001-player.x86_64", "game_playing_interface.jar"):
        if not (player / name).is_file():
            raise ValueError(f"N1 campaign player is incomplete: missing {name}")
    by_state = {state["identity"]: state for state in states}
    base = ledger["wall_seconds_elapsed"]
    session = time.monotonic()
    last_dispatch = session - START_STAGGER_SECONDS
    context = multiprocessing.get_context("spawn")
    live, index, stop_reason, checked = [], 0, None, -1
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise RuntimeError("supervisor received SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    log(f"{phase}: {len(pending)}/{len(cells)} cells pending; workers={WORKERS} "
        f"stagger={START_STAGGER_SECONDS}s attempt_cap={ATTEMPT_SECONDS}s")
    try:
        while index < len(pending) or live:
            elapsed = base + time.monotonic() - session
            if stop_reason is None and elapsed >= WALL_CAP_SECONDS:
                stop_reason = "wall_cap_exceeded"
            if stop_reason is None and free_disk(attempts) < MINIMUM_FREE_BYTES:
                stop_reason = "minimum_free_storage"
            if stop_reason is None and index < len(pending) and index % 25 == 0 \
                    and index != checked:
                checked = index
                if tree_bytes(attempts) > ARTIFACT_BYTES_CAP:
                    stop_reason = "artifact_limit"
            if stop_reason is not None and index < len(pending):
                log(f"cap stop: {stop_reason}; undispatched cells are retained as typed "
                    "not_executed failures (no replacement)")
                for cell in pending[index:]:
                    write_json(output / records / f"{cell['identity']}.json",
                               plain_record(plan_meta["identity"], plan_meta["version"], cell)
                               | {"failure": f"not_executed: {stop_reason}",
                                  "failure_kind": "not_executed", "wall_seconds": 0.0})
                    ledger["cells"][cell["identity"]] = {
                        "status": "failed", "stop": stop_reason, "exit_code": None,
                        "wall_seconds": 0.0, "failure": f"not_executed: {stop_reason}"}
                index = len(pending)
            while (index < len(pending) and len(live) < WORKERS
                   and time.monotonic() - last_dispatch >= START_STAGGER_SECONDS):
                cell = pending[index]
                payload = {"output": str(output), "attempts": str(attempts),
                           "records": records, "cell": cell,
                           "state": by_state[cell["state"]],
                           "player_root": plan_meta["player_root"],
                           "plan_identity": plan_meta["identity"],
                           "plan_version": plan_meta["version"]}
                write_json(output / "markers" / f"{cell['identity']}.json",
                           {"identity": cell["identity"], "phase": phase,
                            "dispatched_at_utc": utc_now()})
                process = c89.start_isolated_worker(context, execute_cell, (payload,))
                live.append({"cell": cell, "process": process, "started": time.monotonic(),
                             "peak_rss": 0.0, "stop": None})
                last_dispatch = time.monotonic()
                index += 1
            if not live:
                time.sleep(1)
                continue
            time.sleep(1)
            now = time.monotonic()
            aggregate = 0.0
            for worker in live:
                if worker["process"].is_alive():
                    rss = c89.process_rss(worker["process"].pid)
                    worker["peak_rss"] = max(worker["peak_rss"], rss)
                    aggregate += rss
                    if now - worker["started"] >= ATTEMPT_SECONDS:
                        worker["stop"] = "attempt_wall_limit"
                    elif worker["peak_rss"] > WORKER_RSS_MIB:
                        worker["stop"] = "worker_memory_limit"
            if aggregate > AGGREGATE_RSS_MIB:
                newest = max((w for w in live if w["process"].is_alive() and w["stop"] is None),
                             key=lambda w: w["started"], default=None)
                if newest is not None:
                    newest["stop"] = "aggregate_memory_limit"
            for worker in [w for w in live if w["stop"] is not None or not w["process"].is_alive()]:
                wall = time.monotonic() - worker["started"]
                exitcode = worker["process"].exitcode
                c89.terminate_worker(worker["process"])
                live.remove(worker)
                cell = worker["cell"]
                record_file = output / records / f"{cell['identity']}.json"
                if not record_file.is_file():
                    write_json(record_file,
                               plain_record(plan_meta["identity"], plan_meta["version"], cell)
                               | {"failure": worker["stop"] or f"worker_exitcode={exitcode}",
                                  "failure_kind": "execution_failure", "wall_seconds": wall})
                record = read_json(record_file)
                ledger["cells"][cell["identity"]] = {
                    "status": "complete" if record["failure"] is None else "failed",
                    "stop": worker["stop"], "exit_code": exitcode, "wall_seconds": wall,
                    "failure": record["failure"]}
                ledger["wall_seconds_elapsed"] = base + time.monotonic() - session
                write_json(output / "receipts" / f"{cell['identity']}.json", {
                    "identity": cell["identity"], "phase": phase, "worker_exitcode": exitcode,
                    "stop": worker["stop"], "wall_seconds": wall,
                    "peak_cpu_rss_mib": worker["peak_rss"]})
                write_json(ledger_file, ledger)
                done = sum(1 for c in cells if c["identity"] in ledger["cells"])
                walls = [e["wall_seconds"] for e in ledger["cells"].values() if e["wall_seconds"]]
                eta = (len(cells) - done) * (sum(walls) / len(walls)) / WORKERS if walls else 0.0
                log(f"{phase} [{done}/{len(cells)}] {cell['identity']} "
                    f"success={(record.get('outcome') or {}).get('first_shot_success')} "
                    f"failure={record['failure']} wall={wall:.0f}s eta={eta / 3600:.2f}h "
                    f"elapsed={ledger['wall_seconds_elapsed'] / 3600:.2f}h")
        ledger["wall_seconds_elapsed"] = base + time.monotonic() - session
        ledger["stop_reason"] = stop_reason
        ledger["status"] = "complete" if stop_reason is None else stop_reason
        write_json(ledger_file, ledger)
    except BaseException:
        ledger["wall_seconds_elapsed"] = base + time.monotonic() - session
        ledger["status"] = "interrupted"
        write_json(ledger_file, ledger)
        for worker in live:
            c89.terminate_worker(worker["process"])
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    log(f"{phase} finished status={ledger['status']} "
        f"wall={ledger['wall_seconds_elapsed'] / 3600:.2f}h")
    return ledger


# ---------------------------------------------------------------------------
# decision-only scoring from the retained (or captured) anchors
# ---------------------------------------------------------------------------

def load_scoring_stack():
    from scripts import run_lookahead_control as wlc
    return wlc, wlc.load_adapter(DEVICE), wlc.load_objective()


def score_inventory(wlc, adapter, objective, models, plan_like, anchor, system, seed, inventory):
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    family = e85.SYSTEM_ARMS[system][0]
    key = (seed, family)
    if key not in models:
        models[key] = e85.load_frozen_predictor(plan_like, seed, family, DEVICE)
    began = time.monotonic()
    with wlc.torch.no_grad():
        carrier = wlc.encode_carrier(adapter, anchor, DEVICE)
        selector = e85.ReactiveSelector(system, seed, models[key], objective, DEVICE)
        decision = selector.choose(carrier, inventory)
    return decision, time.monotonic() - began


def scoring_anchor(plan, state, oracle):
    """Frozen anchor rule: the retained #87 anchor when WP0 found it retained;
    otherwise the decision frame of the member's lowest-ordinal Arm B slot with a
    sealed decision frame, valid only if every executed slot of the member (both
    arms) sealed a byte-identical decision frame."""
    if state["anchor"] is not None:
        return {**{key: state["anchor"][key] for key in
                   ("observation_identity", "fixed_step", "fixed_time_seconds")},
                "frame_path": state["anchor"]["path"], "sha256": state["anchor"]["sha256"],
                "source": "issue87_retained", "cell": state["anchor"]["cell"]}, None
    sealed = sorted(((cell["arm"], cell["ordinal"]), record)
                    for cell in scheduled_cells(plan) if cell["member"] == state["member"]
                    for record in [oracle.get(cell["identity"])]
                    if record is not None and record.get("decision_frame"))
    grid = [record for (arm, _), record in sealed if arm == "grid"]
    if not grid:
        return None, "no_sealed_decision_frame_in_arm_b"
    first = grid[0]["decision_frame"]
    if any(record["decision_frame"]["sha256"] != first["sha256"] for _, record in sealed):
        return None, "captured_decision_frames_disagree"
    return {"observation_identity": first["frame_identity"], "fixed_step": first["fixed_step"],
            "fixed_time_seconds": first["fixed_time_seconds"], "frame_path": first["path"],
            "sha256": first["sha256"], "source": "issue93_captured",
            "cell": grid[0]["cell"]["identity"], "frames_agreeing": len(sealed)}, None


def score(output):
    output = Path(output)
    plan = load_plan(output)
    oracle = load_oracle_records(output, plan)
    wlc, adapter, objective = load_scoring_stack()
    ledger_file = output / "ledger-score.json"
    ledger = read_json(ledger_file) if ledger_file.is_file() else {
        "schema": SCHEMA_LEDGER, "identity": IDENTITY, "phase": "score",
        "gpu_seconds_elapsed": 0.0, "cells": 0}
    models = {}
    gpu = 0.0
    written = 0
    for state in plan["states"]:
        anchor, failure = scoring_anchor(plan, state, oracle)
        for arm in ARMS:
            inventory = plan["inventories"][arm]["members"][state["member"]]
            for system in MODEL_SYSTEMS:
                for seed in SEEDS:
                    identity = decision_identity(system, seed, state["member"], arm)
                    path = output / "records" / f"{identity}.json"
                    if path.is_file():
                        continue
                    record = {"schema": SCHEMA_DECISION, "plan_identity": plan["identity"],
                              "plan_version": plan["version"],
                              "cell": {"identity": identity, "system": system, "seed": seed,
                                       "member": state["member"], "arm": arm,
                                       "state": state["canonical_state"]},
                              "anchor": anchor, "carrier_sha256": None, "decision": None,
                              "failure": None, "failure_kind": None, "gpu_seconds": 0.0,
                              "issue_64_authorized": False}
                    if anchor is None:
                        record["failure"] = f"decision_failure: {failure}"
                        record["failure_kind"] = "decision_failure"
                    else:
                        decision, seconds = score_inventory(
                            wlc, adapter, objective, models, plan, anchor, system, seed,
                            inventory)
                        gpu += seconds
                        record["gpu_seconds"] = seconds
                        record["carrier_sha256"] = anchor["sha256"]
                        record["decision"] = decision
                        if decision["failure"] is not None:
                            record["failure"] = f"decision_failure: {decision['failure']}"
                            record["failure_kind"] = "decision_failure"
                    write_json(path, record)
                    written += 1
    ledger["gpu_seconds_elapsed"] += gpu
    ledger["cells"] += written
    write_json(ledger_file, ledger)
    if ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: gpu_cap_exceeded")
    log(f"score: {written} decision records written, gpu {gpu:.1f}s "
        f"(total {ledger['gpu_seconds_elapsed']:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings():
    names = {
        "issue_80_plan": EIGHTY / "plan.json",
        "issue_87_plan": EIGHTY_SEVEN / "plan.json",
        "issue_89_summary": EIGHTY_NINE / "summary.json",
        "issue_92_summary": NINETY_TWO / "summary.json",
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "issue_77_campaign_plan": CAMPAIGN / "plan.json",
        "wp0_training_action_support": OUTPUT / "training_action_support.json",
        "wp0_anchor_retention": OUTPUT / "anchor_retention.json",
        "wp0_engine_action_check": OUTPUT / "engine_action_check.json",
        "wp0_members": OUTPUT / "members.json",
        "wp1_smoke": OUTPUT / "smoke" / "smoke.json",
    }
    return [{"name": name, "artifact": str(path), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_states():
    members = read_json(OUTPUT / "members.json")["members"]
    retention = {row["member"]: row for row in read_json(OUTPUT / "anchor_retention.json")["members"]}
    plan87 = read_json(EIGHTY_SEVEN / "plan.json")
    by_identity = {state["identity"]: state for state in plan87["states"]}
    states = []
    for entry in members:
        outcomes = EIGHTY / "candidate-outcomes" / f"{entry['member']}.json"
        row = retention[entry["member"]]
        states.append({
            **entry,
            "execution_member_identity": by_identity[entry["canonical_state"]]["execution_member"]["identity"],
            "issue80_candidate_outcomes": {"path": str(outcomes), "sha256": sha256_of(outcomes)},
            "anchor": ({key: row[key] for key in ("observation_identity", "fixed_step",
                                                   "fixed_time_seconds", "path")}
                       | {"cell": row["anchor_cell"], "sha256": row["sha256_recorded"]}
                       if row["retained"] else None),
            "anchor_rule": ("issue87_retained" if row["retained"] else
                            "issue93_captured (lowest-ordinal Arm B sealed frame; all executed "
                            "slots byte-agree) else typed-unmeasurable"),
        })
    return states


def frozen_plan(frozen_at):
    inventories, excluded = arm_inventories()
    states = frozen_states()
    engine = read_json(OUTPUT / "engine_action_check.json")
    support = read_json(OUTPUT / "training_action_support.json")
    smoke = read_json(OUTPUT / "smoke" / "smoke.json")
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal",
        "frozen_at": frozen_at, "frozen_before_scoring_run": True,
        "issue_64_authorized": False, "validation_command": VALIDATION_COMMAND,
        "runner": "scripts/run_second_parameterization_probe.py",
        "changelog": [{
            "version": 1,
            "change": ("initial and terminal freeze of both inventories, the 15-member "
                       "state list, rankers, estimands, frozen disposition rule, reading "
                       "matrix, caps, the declared completion pass and the smoke evidence, "
                       "before any scored engine slot"),
            "why": ("ticket #93; WP0 stop comment (issuecomment-5796923300) reported that the "
                    "engine clamps every grid pull to saturated launch speed; the owner "
                    "decided in session to proceed with the listed lattice and reword C22 "
                    "(the grid's engine-side variation is launch angle only; its radius axis "
                    "exists on the ranker-input side)"),
            "evidence": "engine_action_check.json; smoke/smoke.json",
        }],
        "owner_decisions": {
            "arms": "B + A (default)",
            "margins": "0.50 / 0.60, >=4 ceiling members, >=75% verdict coverage (default)",
            "wp4_breadth_extension": "not run (default)",
            "grid_lattice": "as listed (default); unchanged after the WP0 clamp finding",
            "c22_wording": "reworded after WP0 per owner decision (engine clamp)",
        },
        "inventories": {
            arm: {"design": inventories[arm],
                  "members": {state["member"]: member_inventory(state["member"], arm, inventories)
                              for state in states}}
            for arm in ARMS
        } | {"offset_excluded_by_bounds": excluded},
        "inventory_rules": {
            "grid": ("drag_x in (-40,-80,-120,-160) x drag_y in (5,30,55,80); ordinal = 4*i + "
                     "j; tap 0 ms, release 1000 ms; branch identity {member}-gNN"),
            "offset": ("angles 8.5 + 7k deg at radius 80 px, (round(-80 cos a), round(80 sin "
                       "a)); 85.5 deg excluded (drag_x = -6 outside bounds); tap 0 ms, release "
                       "1000 ms; branch identity {member}-oNN"),
            "engine_realization": ENGINE_READING,
            "identical_launch_removals": {arm: engine["candidates"][arm]["removed_by_rule"]
                                          for arm in ARMS},
            "training_support": {
                "grid": ("out of the release-1000 training support in radius on the "
                         f"ranker-input side (N1 radius {support['sources']['n1_predictor_pool']['radius_px_range']}); "
                         "0/16 exact training actions; the #71 pool covers the drag plane at "
                         "release 600 only"),
                "offset": "inside the N1 radius range; 0/11 exact training actions",
            },
        },
        "states": states,
        "unit_of_analysis": ("N1 source member (15); canonical state = lowest state identity; "
                             "the 9 duplicate pairs are byte-identical (#92 findings §1)"),
        "execution": {
            "decision_anchor": f"native fixed step {DECISION_FIXED_STEP}",
            "engine_seed": "764100001 + member ordinal (state.engine_seed, #87 binding)",
            "oracle_seed_label": ORACLE_SEED_LABEL, "workers": WORKERS,
            "start_stagger_seconds": START_STAGGER_SECONDS,
            "attempt_seconds": ATTEMPT_SECONDS, "port_range": [28700, 28999],
            "agent_socket_seconds": AGENT_SOCKET_SECONDS,
            "agent_connect_deadline_seconds": AGENT_CONNECT_DEADLINE_SECONDS,
            "physics_socket_seconds": PHYSICS_SOCKET_SECONDS,
            "prepare_for_play_seconds": PREPARE_FOR_PLAY_SECONDS,
            "history_ready_seconds": HISTORY_READY_SECONDS, "shot_seconds": SHOT_SECONDS,
            "port_probes": False,
            "detection_rule": ("#85 engine-truth rule inherited verbatim via "
                               "run_engine_outcome_reactive_diagnostic.channel_scan: "
                               "pig_removed (pig_removed / entity_death / entity_destroyed "
                               "events) with the lifecycle leg recorded"),
            "executed_action_check": "segment action drag_release/release/tap == declared",
            "slot_order": "Arm B for all members, then Arm A; ordinal order within member",
            "heavy_media": str(DATA / "attempts"),
        },
        "scheduled_slots": {"grid": 16 * len(states), "offset": 11 * len(states),
                            "total": 27 * len(states)},
        "caps": {"wall_seconds": WALL_CAP_SECONDS, "gpu_seconds": GPU_CAP_SECONDS,
                 "workers": WORKERS, "artifact_bytes": ARTIFACT_BYTES_CAP,
                 "minimum_free_bytes": MINIMUM_FREE_BYTES,
                 "stop_rule": ("cap hit -> remaining slots become typed failures "
                               "not_executed; publish as-is")},
        "completion_pass": {
            "completion_pass_allowed": True,
            "trigger": f"typed failures > {COMPLETION_TRIGGER:.0%} of scheduled slots after --run",
            "scope": "the same failed slots only, executed once; no other slot, no retry",
            "union": "a completion verdict replaces the failed original; failures stay typed",
        },
        "rankers": {
            "systems": list(MODEL_SYSTEMS), "seeds": list(SEEDS),
            "loader": "run_engine_outcome_reactive_diagnostic.load_frozen_predictor",
            "selector": ("ReactiveSelector.choose: Delta carrier applications per candidate, "
                         "argmin predicted cost, ties to the lower ordinal; full per-candidate "
                         "ranking recorded"),
            "carrier": ("run_lookahead_control.encode_carrier(load_adapter) of the member's "
                        "scoring anchor frame"),
            "no_model_prior": "not carried (defined on the N1 ordinal 8; no grid analogue)",
            "adaptation": "none: no retraining, fine-tuning or few-shot adaptation",
        },
        "dynamics_checkpoints": e85.dynamics_checkpoints(),
        "estimands": {
            "admissible_candidate": "a candidate whose slot ends with an engine-truth verdict",
            "measurable_member": "a member with a scoring anchor and >=1 admissible candidate in the arm",
            "ceiling": ("members with >=1 engine-truth success among the arm's admissible "
                        "candidates / measurable members; member bootstrap interval"),
            "primary_auc": ("per cell (system x seed x ceiling member) P(a successful admissible "
                            "candidate receives lower predicted cost than a failed one), ties "
                            "0.5; cell-unit mean and member-clustered mean-of-member-means, "
                            f"{BOOTSTRAP_DRAWS} bootstrap draws, seed {BOOTSTRAP_SEED}, quantiles "
                            f"{list(INTERVAL_QUANTILES)}; per system reported; never pooled "
                            "across arms or with N1"),
            "topk": ("top-1 = argmin predicted cost over admissible candidates (ties lower "
                     "ordinal); top-1/top-3 hits vs chance k/n and 1 - C(n-k,3)/C(n,3); paired "
                     "difference member-clustered; miss-all probability = prod over ceiling "
                     "members of (1 - k/n)"),
            "structure": ("Spearman rho of predicted cost vs drag_x, drag_y, pull radius, launch "
                          "angle and ordinal over the full scored inventory per cell; "
                          "chosen-ordinal and success-ordinal histograms; DESCRIPTIVE"),
            "mirror": ("post-hoc label: argmax predicted cost over admissible candidates (ties "
                       "lower ordinal) top-1 vs chance, paired difference member-clustered; "
                       "DESCRIPTIVE"),
        },
        "claims": CLAIMS,
        "disposition_rule": {
            **RULE,
            "statement": (
                "per claim on the member-clustered point estimate AUC_m over ceiling members: "
                "readiness_or_precision_insufficient if ceiling members < 4 or fewer than 75% "
                "of the arm's scheduled oracle slots end with an engine-truth verdict (typed "
                "failures count against); else supported if AUC_m <= 0.50; else "
                "not_supported_by_this_experiment if AUC_m >= 0.60; else (0.50 < AUC_m < "
                "0.60) readiness_or_precision_insufficient. The interval is printed beside "
                "every point estimate and is never the decision rule."),
            "tokens": list(DISPOSITION_TOKENS),
        },
        "reading_matrix": READING_MATRIX,
        "expected_structure": EXPECTED,
        "player_root": str(CAMPAIGN / "player"),
        "smoke": {"evidence": smoke, "records": str(OUTPUT / "smoke" / "records"),
                  "joined_to_scoring": False},
        "inputs": input_bindings(),
        "source_text": {name: (ROOT / name).read_text() for name in SOURCE_FILES},
        "claim_boundary": (
            "within-state engine-truth selection AUC of the three frozen N1 rankers on two "
            "new candidate inventories over 15 N1 source members; does not test other "
            "models, adaptation, fresh gameplay, sealed benchmarks (#64/#65 sealed) or state "
            "difficulty; does not re-open #87/#89/#90/#92 dispositions"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --verify, --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-93 protocol")
    if not plan.get("frozen_before_scoring_run"):
        raise ValueError("plan.json does not declare a pre-scoring freeze")
    inventories, _ = arm_inventories()
    if any(plan["inventories"][arm]["design"] != inventories[arm] for arm in ARMS):
        raise ValueError("runner inventories differ from the frozen plan")
    if len(plan["states"]) != EXPECTED["members"] or len(scheduled_cells(plan)) != EXPECTED["slots"]:
        raise ValueError("frozen plan structure differs from the expected counts")
    return plan


# ---------------------------------------------------------------------------
# retained records -> published tables
# ---------------------------------------------------------------------------

def load_oracle_records(output, plan):
    """Union oracle table: a completion-pass verdict replaces the failed original."""
    output = Path(output)
    records = {}
    for cell in scheduled_cells(plan):
        path = output / "records" / f"{cell['identity']}.json"
        if path.is_file():
            record = read_json(path)
            if record.get("schema") != SCHEMA_ORACLE or record.get("plan_identity") != IDENTITY:
                raise ValueError(f"oracle record {path} binding differs")
            completion = output / "records-completion" / f"{cell['identity']}.json"
            if record["outcome"] is None and completion.is_file():
                replacement = read_json(completion)
                replacement["completion_pass"] = True
                record = replacement if replacement["outcome"] is not None else record
            records[cell["identity"]] = record
    return records


def load_decisions(output, plan):
    decisions = {}
    for state in plan["states"]:
        for arm in ARMS:
            for system in MODEL_SYSTEMS:
                for seed in SEEDS:
                    identity = decision_identity(system, seed, state["member"], arm)
                    path = Path(output) / "records" / f"{identity}.json"
                    if path.is_file():
                        record = read_json(path)
                        if record.get("schema") != SCHEMA_DECISION:
                            raise ValueError(f"decision record {path} binding differs")
                        decisions[identity] = record
    return decisions


def bootstrap(values, clusters=None):
    import numpy as np
    values = list(values)
    if not values:
        return None
    clusters = list(range(len(values))) if clusters is None else list(clusters)
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    means = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(means), size=(BOOTSTRAP_DRAWS, len(means)))
    low, high = np.quantile(means[draws].mean(axis=1), INTERVAL_QUANTILES)
    return {"mean": float(means.mean()), "interval": [float(low), float(high)],
            "clusters": len(means), "units": len(values), "label": "DESCRIPTIVE"}


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def pairwise_auc(pos, neg):
    score = sum(1.0 if p < n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return score / (len(pos) * len(neg))


def decision_rows(plan, oracle, decisions):
    rows = []
    for state in plan["states"]:
        for arm in ARMS:
            inventory = {item["ordinal"]: item["action"]
                         for item in plan["inventories"][arm]["members"][state["member"]]}
            verdicts = {}
            for ordinal in inventory:
                record = oracle.get(oracle_identity(state["member"], arm, ordinal))
                if record is not None and record["outcome"] is not None:
                    verdicts[ordinal] = record["outcome"]["first_shot_success"]
            successes = sorted(o for o, hit in verdicts.items() if hit)
            for system in MODEL_SYSTEMS:
                for seed in SEEDS:
                    identity = decision_identity(system, seed, state["member"], arm)
                    record = decisions.get(identity)
                    row = {"cell_identity": identity, "arm": arm, "member": state["member"],
                           "state": state["canonical_state"],
                           "family": state["generator_family"],
                           "exposure_role": state["exposure_role"], "system": system,
                           "seed": seed, "verdict_slots": len(verdicts),
                           "successes": successes, "n_successes": len(successes),
                           "typed_failure": record is None or record["failure"] is not None,
                           "failure": None if record is None else record["failure"],
                           "anchor_source": None if record is None or record["anchor"] is None
                           else record["anchor"]["source"]}
                    if row["typed_failure"]:
                        row.update(ranking=[], selector_chosen=None, measurable=False)
                        rows.append(row)
                        continue
                    ranking = [(entry["ordinal"], entry["predicted_cost"])
                               for entry in record["decision"]["ranking"]
                               if entry["predicted_cost"] is not None]
                    admissible = sorted(((o, c) for o, c in ranking if o in verdicts),
                                        key=lambda pair: (pair[1], pair[0]))
                    row.update(ranking=ranking,
                               selector_chosen=record["decision"]["chosen"]["ordinal"],
                               selector_chosen_cost=record["decision"]["selected_predicted_cost"],
                               n_candidates=len(ranking), admissible=admissible,
                               measurable=bool(admissible), ceiling=bool(successes))
                    coords = {
                        "ordinal": [o for o, _ in ranking],
                        "drag_x": [inventory[o]["drag_x"] for o, _ in ranking],
                        "drag_y": [inventory[o]["drag_y"] for o, _ in ranking],
                        "radius": [launch_geometry(inventory[o]["drag_x"],
                                                   inventory[o]["drag_y"])["radius_px"]
                                   for o, _ in ranking],
                        "angle": [launch_geometry(inventory[o]["drag_x"],
                                                  inventory[o]["drag_y"])["angle_deg"]
                                  for o, _ in ranking]}
                    costs = [c for _, c in ranking]
                    row["spearman"] = {key: spearman(values, costs)
                                       for key, values in coords.items()}
                    if admissible:
                        top1 = admissible[0][0]
                        mirror = sorted(admissible, key=lambda pair: (-pair[1], pair[0]))[0][0]
                        n, k = len(admissible), sum(1 for o, _ in admissible if verdicts[o])
                        row.update(
                            chosen=top1, chosen_cost=admissible[0][1],
                            chosen_action=inventory[top1], mirror_chosen=mirror)
                        if successes:
                            row.update(
                                top1_hit=bool(verdicts[top1]),
                                top3_hit=any(verdicts[o] for o, _ in admissible[:3]),
                                mirror_hit=bool(verdicts[mirror]),
                                chance_top1=k / n,
                                chance_top3=(1.0 - math.comb(n - k, 3) / math.comb(n, 3)
                                             if n >= 3 else 1.0))
                            pos = [c for o, c in admissible if verdicts[o]]
                            neg = [c for o, c in admissible if not verdicts[o]]
                            row["auc"] = pairwise_auc(pos, neg) if pos and neg else None
                    rows.append(row)
    return rows


def disposition(ceiling_members, coverage, auc_m):
    if ceiling_members < RULE["ceiling_members_min"] or coverage < RULE["verdict_coverage_min"]:
        return STOP_TOKEN
    if auc_m is None:
        return STOP_TOKEN
    if auc_m <= RULE["supported_max_auc"]:
        return "supported"
    if auc_m >= RULE["not_supported_min_auc"]:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def arm_tables(plan, oracle, rows, arm):
    cells = [cell for cell in scheduled_cells(plan) if cell["arm"] == arm]
    slot_status = Counter()
    failures = Counter()
    for cell in cells:
        record = oracle.get(cell["identity"])
        if record is None:
            slot_status["not_executed"] += 1
        elif record["outcome"] is not None:
            slot_status["verdict"] += 1
        else:
            slot_status["typed_failure"] += 1
            failures[record["failure_kind"]] += 1
    coverage = slot_status["verdict"] / len(cells)
    arm_rows = [row for row in rows if row["arm"] == arm]
    members = [state["member"] for state in plan["states"]]
    measurable = sorted({row["member"] for row in arm_rows if row["measurable"]})
    ceiling = sorted({row["member"] for row in arm_rows if row["measurable"] and row["ceiling"]})
    per_member = {}
    for member in members:
        sample = next(row for row in arm_rows if row["member"] == member)
        per_member[member] = {"verdict_slots": sample["verdict_slots"],
                              "successes": sample["successes"],
                              "measurable": member in measurable,
                              "anchor_source": sample["anchor_source"]}
    ceiling_cells = [row for row in arm_rows if row["measurable"] and row["ceiling"]]
    auc_cells = [row for row in ceiling_cells if row.get("auc") is not None]

    def auc_block(selected):
        return {"cells": len(selected),
                "cell_unit": bootstrap([row["auc"] for row in selected]),
                "member_clustered": bootstrap([row["auc"] for row in selected],
                                              [row["member"] for row in selected])}

    def rate_block(selected, hit, chance):
        return {"cells": len(selected), "hits": sum(1 for row in selected if row[hit]),
                "rate": (sum(1 for row in selected if row[hit]) / len(selected)
                         if selected else None),
                "chance_mean": (sum(row[chance] for row in selected) / len(selected)
                                if selected else None),
                "paired_difference_member_clustered": bootstrap(
                    [float(row[hit]) - row[chance] for row in selected],
                    [row["member"] for row in selected])}

    auc = auc_block(auc_cells)
    auc["per_system"] = {system: auc_block([row for row in auc_cells if row["system"] == system])
                         for system in MODEL_SYSTEMS}
    member_chance = {}
    for member in ceiling:
        row = next(r for r in ceiling_cells if r["member"] == member)
        member_chance[member] = row["chance_top1"]
    miss_all = math.prod(1.0 - value for value in member_chance.values()) if ceiling else None
    scored = [row for row in arm_rows if row["measurable"]]
    structure = {}
    for key in ("ordinal", "drag_x", "drag_y", "radius", "angle"):
        values = [row["spearman"][key] for row in scored]
        structure[key] = {
            "median_rho": median(values),
            "median_abs_rho": median([abs(v) for v in values if v is not None]),
            "per_system_median_rho": {system: median([row["spearman"][key] for row in scored
                                                      if row["system"] == system])
                                      for system in MODEL_SYSTEMS}}
    chosen_hist = Counter(row["selector_chosen"] for row in scored)
    success_hist = Counter(o for member in measurable for o in per_member[member]["successes"])
    auc_m = auc["member_clustered"]["mean"] if auc["member_clustered"] else None
    token = disposition(len(ceiling), coverage, auc_m)
    return {
        "claim": ARM_CLAIMS[arm], "claim_text": CLAIMS[ARM_CLAIMS[arm]],
        "scheduled_slots": len(cells), "slot_status": dict(sorted(slot_status.items())),
        "typed_failures_by_kind": dict(sorted(failures.items())),
        "verdict_coverage": coverage,
        "members": per_member, "measurable_members": measurable, "ceiling_members": ceiling,
        "ceiling": {"value": len(ceiling) / len(measurable) if measurable else None,
                    "numerator": len(ceiling), "denominator": len(measurable),
                    "interval": bootstrap([float(m in ceiling) for m in measurable])},
        "auc": auc,
        "top1": rate_block(ceiling_cells, "top1_hit", "chance_top1"),
        "top3": rate_block(ceiling_cells, "top3_hit", "chance_top3"),
        "miss_all_probability": miss_all,
        "mirror_top1": rate_block(ceiling_cells, "mirror_hit", "chance_top1")
        | {"label": "post-hoc, DESCRIPTIVE"},
        "structure": structure,
        "chosen_ordinal_histogram": {str(k): chosen_hist[k] for k in sorted(chosen_hist)},
        "success_ordinal_histogram": {str(k): success_hist[k] for k in sorted(success_hist)},
        "success_band": sorted(success_hist),
        "disposition": token,
        "disposition_inputs": {"ceiling_members": len(ceiling), "verdict_coverage": coverage,
                               "auc_member_clustered": auc_m},
    }


def reading_row(c22, c23):
    if STOP_TOKEN in (c22, c23):
        return READING_MATRIX[4]
    return next(row for row in READING_MATRIX[:4] if row["C22"] == c22 and row["C23"] == c23)


def compute_accounting(output, plan, oracle):
    output = Path(output)
    walls = []
    engine_walls = []
    for cell in scheduled_cells(plan):
        receipt = output / "receipts" / f"{cell['identity']}.json"
        if receipt.is_file():
            walls.append(read_json(receipt)["wall_seconds"])
        record = oracle.get(cell["identity"])
        if record is not None and record.get("execution"):
            engine_walls.append(record["execution"]["engine_wall_seconds"])

    def distribution(values):
        if not values:
            return None
        ordered = sorted(values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        return {"count": len(ordered), "mean": sum(ordered) / len(ordered),
                "median": median(ordered), "p10": pick(0.10), "p90": pick(0.90),
                "max": ordered[-1]}

    ledgers = {name: read_json(output / name) for name in
               ("ledger.json", "ledger-completion.json", "ledger-score.json")
               if (output / name).is_file()}
    return {
        "oracle_wall_seconds": ledgers["ledger.json"]["wall_seconds_elapsed"],
        "oracle_stop_reason": ledgers["ledger.json"]["stop_reason"],
        "completion_wall_seconds": (ledgers["ledger-completion.json"]["wall_seconds_elapsed"]
                                    if "ledger-completion.json" in ledgers else None),
        "score_gpu_seconds": ledgers["ledger-score.json"]["gpu_seconds_elapsed"],
        "per_slot_worker_wall_seconds": distribution(walls),
        "per_slot_engine_wall_seconds": distribution(engine_walls),
        "launch_speed_range": [
            min(r["launch"]["speed"] for r in oracle.values() if r.get("launch")),
            max(r["launch"]["speed"] for r in oracle.values() if r.get("launch"))],
        "caps": {"wall_seconds": WALL_CAP_SECONDS, "gpu_seconds": GPU_CAP_SECONDS,
                 "workers": WORKERS},
    }


def compute_tables(output, plan):
    oracle = load_oracle_records(output, plan)
    decisions = load_decisions(output, plan)
    rows = decision_rows(plan, oracle, decisions)
    structure = {"members": len(plan["states"]), "slots": len(scheduled_cells(plan)),
                 "grid_slots": sum(1 for c in scheduled_cells(plan) if c["arm"] == "grid"),
                 "offset_slots": sum(1 for c in scheduled_cells(plan) if c["arm"] == "offset"),
                 "decision_cells": len(rows)}
    if structure != EXPECTED:
        raise ValueError(f"structural counts {structure} differ from EXPECTED {EXPECTED}")
    if len(decisions) != EXPECTED["decision_cells"]:
        raise ValueError(f"{len(decisions)} decision records, expected "
                         f"{EXPECTED['decision_cells']}; run --score")
    arms = {arm: arm_tables(plan, oracle, rows, arm) for arm in ARMS}
    row = reading_row(arms["grid"]["disposition"], arms["offset"]["disposition"])
    frames = [r["frame_vs_retained_anchor"]["byte_equal"] for r in oracle.values()
              if r.get("frame_vs_retained_anchor")]
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "plan_identity": plan["identity"],
        "structure": structure, "arms": arms,
        "dispositions": {"C22": arms["grid"]["disposition"],
                         "C23": arms["offset"]["disposition"]},
        "reading_matrix_row": row,
        "decision_frames_vs_retained_anchor": {"byte_equal": sum(frames),
                                               "compared": len(frames)},
        "compute": compute_accounting(output, plan, oracle),
        "rows": [{key: value for key, value in r.items() if key not in ("ranking", "admissible")}
                 for r in rows],
    }


def fmt(value, digits=4):
    return "n/a" if value is None else format(value, f".{digits}f")


def fmt_interval(block):
    if block is None:
        return "n/a"
    return f"{fmt(block['mean'])} [{fmt(block['interval'][0])}, {fmt(block['interval'][1])}]"


def slot_join_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["cell_identity", "state", "source_member", "family", "exposure_role",
                     "system", "seed", "typed_failure", "n_candidates", "verdict_slots",
                     "n_successes", "successful_ordinals", "chosen_ordinal",
                     "chosen_predicted_cost", "chosen_in_success_band", "top1_hit", "top3_hit",
                     "chance_top1", "chance_top3", "auc", "spearman_ordinal",
                     "spearman_drag_x", "spearman_drag_y", "bound_issue87_only",
                     "first_shot_success_issue87_only", "arm", "drag_x", "drag_y", "radius",
                     "angle_deg"])
    bands = {arm: set(compute["arms"][arm]["success_band"]) for arm in ARMS}

    def cell(value):
        return "" if value is None else value

    for row in compute["rows"]:
        chosen = row.get("chosen")
        action = row.get("chosen_action")
        geometry = launch_geometry(action["drag_x"], action["drag_y"]) if action else None
        spear = row.get("spearman") or {}
        writer.writerow([
            row["cell_identity"], row["state"], row["member"], row["family"],
            row["exposure_role"], row["system"], row["seed"], row["typed_failure"],
            cell(row.get("n_candidates")), row["verdict_slots"], row["n_successes"],
            "|".join(str(o) for o in row["successes"]), cell(chosen),
            cell(row.get("chosen_cost")),
            (chosen in bands[row["arm"]]) if chosen is not None else "",
            cell(row.get("top1_hit")), cell(row.get("top3_hit")),
            cell(row.get("chance_top1")), cell(row.get("chance_top3")), cell(row.get("auc")),
            cell(spear.get("ordinal")), cell(spear.get("drag_x")), cell(spear.get("drag_y")),
            chosen is not None, cell(row.get("top1_hit")) if chosen is not None else "",
            row["arm"], cell(action and action["drag_x"]), cell(action and action["drag_y"]),
            cell(geometry and geometry["radius_px"]), cell(geometry and geometry["angle_deg"])])
    return buffer.getvalue()


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "arm", "statistic", "unit", "value", "interval_low",
                     "interval_high", "interval_label", "detail"])

    def put(identity, arm, statistic, unit, value, block=None, detail=""):
        writer.writerow([identity, arm, statistic, unit, "" if value is None else value,
                         block["interval"][0] if block else "",
                         block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", detail])

    for arm in ARMS:
        table = compute["arms"][arm]
        put(f"{arm}_coverage", arm, "verdict coverage", "slots", table["verdict_coverage"],
            detail=json.dumps(table["slot_status"], sort_keys=True))
        put(f"{arm}_ceiling", arm, "member ceiling", "members", table["ceiling"]["value"],
            table["ceiling"]["interval"],
            f"{table['ceiling']['numerator']} of {table['ceiling']['denominator']}")
        for unit in ("cell_unit", "member_clustered"):
            block = table["auc"][unit]
            put(f"{arm}_auc_{unit}", arm, "within-state AUC", unit,
                block and block["mean"], block, f"{table['auc']['cells']} cells")
        for system in MODEL_SYSTEMS:
            block = table["auc"]["per_system"][system]["member_clustered"]
            put(f"{arm}_auc_{system}", arm, "within-state AUC per system", "member_clustered",
                block and block["mean"], block)
        for key in ("top1", "top3", "mirror_top1"):
            block = table[key]
            put(f"{arm}_{key}", arm, f"{key} hit rate", "cells", block["rate"],
                detail=f"{block['hits']}/{block['cells']}; chance {block['chance_mean']}")
            diff = block["paired_difference_member_clustered"]
            put(f"{arm}_{key}_minus_chance", arm, f"{key} minus chance", "member_clustered",
                diff and diff["mean"], diff)
        put(f"{arm}_miss_all", arm, "one uniform draw per ceiling member misses everywhere",
            "probability", table["miss_all_probability"])
        for key, block in table["structure"].items():
            put(f"{arm}_spearman_{key}", arm, f"median Spearman rho cost vs {key}", "cells",
                block["median_rho"], detail=json.dumps(block["per_system_median_rho"],
                                                       sort_keys=True))
        put(f"{arm}_disposition", arm, f"{table['claim']} disposition", "token",
            table["disposition"], detail=json.dumps(table["disposition_inputs"], sort_keys=True))
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    add("# Issue-93: within-state selection AUC on a second candidate parameterization — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan `{SCHEMA_PLAN}` v{plan['version']} (terminal), frozen "
        f"{plan['frozen_at']} before any scored engine slot")
    add(f"- validation command: `{VALIDATION_COMMAND}`")
    add(f"- engine realization: {ENGINE_READING}")
    add("- every interval is DESCRIPTIVE (10000 draws, seed 7201); the interval is never the "
        "decision rule")
    add("")
    add("## Dispositions (frozen rule)")
    add("")
    add(f"- **C22 (Arm B, drag grid): {compute['dispositions']['C22']}**")
    add(f"- **C23 (Arm A, offset sweep): {compute['dispositions']['C23']}**")
    row = compute["reading_matrix_row"]
    add(f"- reading-matrix row {row['row']}: \"{row['reading']}\"")
    add("")
    for arm, title in (("grid", "Arm B — Cartesian drag grid (C22, primary)"),
                       ("offset", "Arm A — offset angle sweep (C23)")):
        table = compute["arms"][arm]
        add(f"## {title}")
        add("")
        add(f"- claim: {table['claim_text']}")
        add(f"- slots: {table['scheduled_slots']} scheduled; {json.dumps(table['slot_status'], sort_keys=True)}; "
            f"verdict coverage **{fmt(table['verdict_coverage'])}**; typed failures "
            f"{json.dumps(table['typed_failures_by_kind'], sort_keys=True)}")
        add(f"- ceiling: **{table['ceiling']['numerator']}/{table['ceiling']['denominator']}** "
            f"measurable members = {fmt(table['ceiling']['value'])} "
            f"(interval {fmt_interval(table['ceiling']['interval'])}); ceiling members "
            f"{table['ceiling_members']}")
        add(f"- **within-state AUC**: cell-unit {fmt_interval(table['auc']['cell_unit'])} over "
            f"{table['auc']['cells']} cells; member-clustered "
            f"**{fmt_interval(table['auc']['member_clustered'])}** over "
            f"{(table['auc']['member_clustered'] or {}).get('clusters', 0)} members")
        for system in MODEL_SYSTEMS:
            block = table["auc"]["per_system"][system]
            add(f"  - {system}: member-clustered {fmt_interval(block['member_clustered'])} "
                f"({block['cells']} cells)")
        for key, label in (("top1", "top-1"), ("top3", "top-3"),
                           ("mirror_top1", "mirror (argmax-cost) top-1, post-hoc")):
            block = table[key]
            add(f"- {label}: {block['hits']}/{block['cells']} = {fmt(block['rate'])} vs chance "
                f"{fmt(block['chance_mean'])}; paired difference "
                f"{fmt_interval(block['paired_difference_member_clustered'])}")
        add(f"- miss-all probability (one uniform draw per ceiling member): "
            f"{fmt(table['miss_all_probability'])}")
        add("- structure (median Spearman rho of predicted cost, DESCRIPTIVE): " + "; ".join(
            f"{key} {fmt(block['median_rho'])} (|rho| {fmt(block['median_abs_rho'])})"
            for key, block in table["structure"].items()))
        add(f"- chosen-ordinal histogram (selector over full inventory): "
            f"{json.dumps(table['chosen_ordinal_histogram'])}")
        add(f"- success-ordinal histogram (members): "
            f"{json.dumps(table['success_ordinal_histogram'])}")
        add(f"- disposition inputs: {json.dumps(table['disposition_inputs'], sort_keys=True)} "
            f"-> **{table['disposition']}**")
        add("")
    frames = compute["decision_frames_vs_retained_anchor"]
    comp = compute["compute"]
    add("## Execution and compute")
    add("")
    add(f"- decision frames byte-identical to the member's retained #87 anchor: "
        f"{frames['byte_equal']}/{frames['compared']}")
    add(f"- executed launch speed range {fmt(comp['launch_speed_range'][0])}–"
        f"{fmt(comp['launch_speed_range'][1])} (saturated at 10 by the drag clamp)")
    add(f"- oracle wall {fmt(comp['oracle_wall_seconds'] / 3600, 2)} h (stop reason "
        f"{comp['oracle_stop_reason']}); completion pass wall "
        f"{comp['completion_wall_seconds']}; ranker scoring GPU "
        f"{fmt(comp['score_gpu_seconds'], 1)} s")
    add(f"- per-slot worker wall: {json.dumps(comp['per_slot_worker_wall_seconds'], sort_keys=True)}")
    add("")
    add("## Training action support (WP0)")
    add("")
    for arm in ARMS:
        add(f"- {arm}: {plan['inventory_rules']['training_support'][arm]}")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ". Arms are never pooled with each other or with N1. No "
        "state-difficulty claim is drawn from any row. `bound_issue87_only` / "
        "`first_shot_success_issue87_only` in slot_join.csv keep #92's column names and bind "
        "the chosen admissible candidate to this experiment's own oracle table.")
    add("")
    return "\n".join(lines)


def build_report(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "plan_identity": plan["identity"],
            "plan_version": plan["version"], "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "claims": CLAIMS,
            "dispositions": compute["dispositions"],
            "reading_matrix_row": compute["reading_matrix_row"],
            "arms": compute["arms"], "structure": compute["structure"],
            "decision_frames_vs_retained_anchor": compute["decision_frames_vs_retained_anchor"],
            "compute": compute["compute"], "engine_realization": ENGINE_READING,
            "training_support": plan["inventory_rules"]["training_support"],
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False,
            "hygiene": {"content_hashes_recomputed_after_freeze": False,
                        "full_corpus_integrity_pass": False}}


def rendered(plan, compute):
    return {"summary.json": json_text(build_report(plan, compute)),
            "findings.md": findings_md(plan, compute),
            "slot_join.csv": slot_join_csv(compute),
            "comparisons.csv": comparisons_csv(compute)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    log(f"published: C22 {compute['dispositions']['C22']}, C23 "
        f"{compute['dispositions']['C23']}; reading row {compute['reading_matrix_row']['row']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    fresh = compute_tables(output, plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"exact recomputation validation passed: {fresh['structure']} re-derived from the "
        f"retained records; compute.json, summary.json, findings.md, slot_join.csv and "
        f"comparisons.csv byte-compared; C22 {fresh['dispositions']['C22']}, C23 "
        f"{fresh['dispositions']['C23']} ({time.monotonic() - began:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# smoke (WP1): infrastructure assertions only, never joined to scoring
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output) / "smoke"
    inventories, _ = arm_inventories()
    members = read_json(OUTPUT / "members.json")["members"]
    retention = {row["member"]: row for row in read_json(OUTPUT / "anchor_retention.json")["members"]}
    states = [dict(entry, anchor=({"path": retention[entry["member"]]["path"]}
                                  if retention[entry["member"]]["retained"] else None))
              for entry in members]
    cells = smoke_cells(states, inventories)
    plan87, decisions, _ = load_issue_87()
    meta = {"identity": IDENTITY, "version": 0, "player_root": str(CAMPAIGN / "player")}
    supervise(output, DATA / "smoke" / "attempts", meta, cells, plan87["states"], "smoke")
    checks = []
    for cell in cells:
        record = read_json(output / "records" / f"{cell['identity']}.json")
        launch = record.get("launch") or {}
        checks.append({
            "cell": cell["identity"], "failure": record["failure"],
            "verdict_produced": record["outcome"] is not None,
            "frame_captured": record["decision_frame"] is not None,
            "frame_vs_retained_anchor": record["frame_vs_retained_anchor"],
            "executed_action_equals_declared": record["executed_action"] is not None
            and record["failure"] is None,
            "launch_speed": launch.get("speed"), "launch_angle_deg": launch.get("angle_deg"),
            "declared_angle_deg": launch_geometry(cell["action"]["drag_x"],
                                                  cell["action"]["drag_y"])["angle_deg"],
            "launch_saturated": launch.get("speed") is not None
            and abs(launch["speed"] - 10.0) < 0.01,
            "wall_seconds": record.get("wall_seconds"),
        })
    # scoring reproduction: the retained #87 rankings of one state from its anchor
    state = next(s for s in plan87["states"] if s["identity"] == f"{SMOKE_MEMBER}-a00")
    wlc, adapter, objective = load_scoring_stack()
    anchor_row = retention[SMOKE_MEMBER]
    anchor = {"observation_identity": anchor_row["observation_identity"],
              "fixed_step": anchor_row["fixed_step"],
              "fixed_time_seconds": anchor_row["fixed_time_seconds"],
              "frame_path": anchor_row["path"]}
    reproduction = []
    models = {}
    for system in MODEL_SYSTEMS:
        seed = SEEDS[0]
        retained = decisions[f"decision--{system}--seed{seed}--{state['identity']}"]["decision"]
        decision, _ = score_inventory(wlc, adapter, objective, models, plan87, anchor, system,
                                      seed, state["inventory"])
        stored = {row["ordinal"]: row["predicted_cost"] for row in retained["ranking"]}
        fresh = {row["ordinal"]: row["predicted_cost"] for row in decision["ranking"]}
        reproduction.append({
            "system": system, "seed": seed,
            "max_abs_cost_difference": max(abs(stored[o] - fresh[o]) for o in stored),
            "chosen_equal": retained["chosen"]["ordinal"] == decision["chosen"]["ordinal"]})
        grid_items = member_inventory(SMOKE_MEMBER, "grid", inventories)
        grid_decision, _ = score_inventory(wlc, adapter, objective, models, plan87, anchor,
                                           system, seed, grid_items)
        reproduction[-1]["grid_ranking_finite"] = sum(
            1 for row in grid_decision["ranking"] if row["predicted_cost"] is not None)
    ok = (all(c["verdict_produced"] and c["frame_captured"] and c["launch_saturated"]
              and c["executed_action_equals_declared"] for c in checks)
          and all(r["max_abs_cost_difference"] <= 1e-3 and r["chosen_equal"]
                  and r["grid_ranking_finite"] == 16 for r in reproduction))
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(),
                "cells": checks, "scoring_reproduction": reproduction, "ok": ok,
                "note": ("infrastructure assertions only; smoke verdicts are never joined to "
                         "scoring and are not asserted")}
    write_json(output / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'}: {json.dumps(checks)[:600]} "
        f"reproduction {reproduction}")
    if not ok:
        raise ValueError("smoke infrastructure assertion failed; no freeze")
    return 0


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        plan = load_plan(output)
        log(f"existing frozen plan validated (frozen {plan['frozen_at']})")
        return 0
    smoke_path = output / "smoke" / "smoke.json"
    if not smoke_path.is_file() or not read_json(smoke_path)["ok"]:
        raise ValueError("smoke evidence missing or failed; run --smoke first")
    if (output / "records").exists():
        raise ValueError("records exist before the freeze; refusing")
    plan = frozen_plan(utc_now())
    text = json_text(plan)
    for marker in PLACEHOLDER_MARKERS:
        if marker in json_text({k: v for k, v in plan.items() if k != "source_text"}):
            raise ValueError(f"frozen plan contains missing-value marker {marker!r}")
    path.write_text(text)
    load_plan(output)
    log(f"frozen plan published at {path}: {len(scheduled_cells(plan))} slots over "
        f"{len(plan['states'])} members; no scored engine slot has run")
    return 0


def run(output):
    output = Path(output)
    plan = load_plan(output)
    plan87 = read_json(EIGHTY_SEVEN / "plan.json")
    supervise(output, DATA / "attempts", plan, scheduled_cells(plan), plan87["states"], "run")
    return 0


def completion(output):
    output = Path(output)
    plan = load_plan(output)
    ledger = read_json(output / "ledger.json")
    if ledger["status"] == "interrupted" or ledger["status"] == "running":
        raise ValueError("the oracle pass is not terminal; resume --run first")
    cells = scheduled_cells(plan)
    failed = [cell for cell in cells
              if read_json(output / "records" / f"{cell['identity']}.json")["outcome"] is None]
    share = len(failed) / len(cells)
    if share <= COMPLETION_TRIGGER:
        log(f"completion pass not triggered: typed failures {len(failed)}/{len(cells)} = "
            f"{share:.3f} <= {COMPLETION_TRIGGER}")
        return 0
    if (output / "ledger-completion.json").is_file() and \
            read_json(output / "ledger-completion.json")["status"] != "interrupted":
        raise ValueError("the declared single completion pass has already run")
    plan87 = read_json(EIGHTY_SEVEN / "plan.json")
    supervise(output, DATA / "attempts-completion", plan, failed, plan87["states"],
              "completion", records="records-completion", ledger_name="ledger-completion.json")
    return 0


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def verify(output):
    output = Path(output)
    inventories, excluded = arm_inventories()
    plan87, decisions, oracles87 = load_issue_87()
    members = member_states(plan87)
    if len(members) != 15:
        raise ValueError(f"member-unit state list has {len(members)} members, not 15")
    write_json(output / "members.json", {
        "schema": SCHEMA_MEMBERS, "identity": IDENTITY, "computed_at": utc_now(),
        "source": str(EIGHTY_SEVEN / "plan.json"),
        "source_sha256": sha256_of(EIGHTY_SEVEN / "plan.json"),
        "membership_source": str(EIGHTY / "plan.json"),
        "membership_source_sha256": sha256_of(EIGHTY / "plan.json"),
        "members": members, "arm_a_excluded": excluded})
    retention = anchor_retention(plan87, decisions, members)
    write_json(output / "anchor_retention.json", retention)
    log(f"WP0.2 anchors retained for {15 - len(retention['missing'])}/15 members; "
        f"missing {retention['missing']}")
    engine = engine_action_check(plan87, oracles87, inventories)
    write_json(output / "engine_action_check.json", engine)
    log(f"WP0.3 clamp radius {engine['clamp_radius_px']:.2f} px; grid clamped "
        f"{engine['candidates']['grid']['clamped']}/16, offset clamped "
        f"{engine['candidates']['offset']['clamped']}/11; #87 launch speed "
        f"[{engine['retained_issue87_launches']['speed_min']:.4f}, "
        f"{engine['retained_issue87_launches']['speed_max']:.4f}]")
    support = training_action_support(inventories)
    write_json(output / "training_action_support.json", support)
    log(f"WP0.1 N1 training actions {len(support['n1_training_actions'])}; "
        f"#71 release values {support['sources']['issue71_predictor_pool']['release_ms_values']}")
    return 0


MODES = {"verify": (verify, False), "smoke": (smoke, True), "prepare": (prepare, False),
         "run": (run, True), "completion": (completion, True), "score": (score, True),
         "publish": (publish, True), "validate": (validate, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name))
    function, locked = MODES[mode]
    try:
        if locked:
            with GPULock():
                return function(args.output)
        return function(args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
