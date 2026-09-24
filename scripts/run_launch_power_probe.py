"""Issue-94 ADD-EXP: launch-power x angle inventory inside the 18.49 px drag clamp.

Binding runner module: scripts/run_launch_power_probe.py
Exact validation command: python -u -m scripts.run_launch_power_probe --validate

Question: every candidate inventory scored so far (N1 13-candidate sweep, #93
offset sweep, #93 drag grid) launches at the saturated speed 10 because the
engine projects any drag beyond _dragRadius (1 world unit = 18.49 px) onto the
drag circle.  Launch power varies in the engine only for drags shorter than
the clamp.  Arm P crosses 4 launch angles {10,30,50,70} deg with 4 sub-clamp
radii {5,9,13,17} px plus a saturated control column (80 px, the N1 radius),
executes every candidate once in the real rendered engine over the 15 N1
source members and scores the three frozen N1 rankers from the retained
decision anchors, with pre-registered claims for pooled top-1 (C24) and
within-state AUC (C25).

Modes (frozen-protocol chronology, binding shared rule):
- --prepare     WP0 pre-freeze verification (analysis only, zero engine seconds):
                action-bound authority, engine realization, training support,
                members, anchors; then freezes plan.json BEFORE any engine slot.
- --smoke       WP1: one member x 3 radii at one angle through the full pipeline;
                realized launch speeds must differ as WP0 predicts; records under
                smoke/, never joined to scoring.
- --run         WP2 oracle pass: every scheduled slot once, <=4 staggered isolated
                workers, ledger-resumable; cap stop -> typed not_executed.
- --completion  the single declared completion pass (only if typed failures exceed
                25% of slots; same slots only; once).
- --score       decision-only ranker scoring (3 systems x 3 seeds x 15 members)
                from the retained anchors; GPU seconds.
- --publish     WP3 summary.json / findings.md / slot_join.csv / comparisons.csv /
                compute.json.
- --validate    recomputes every published table from the retained records and
                byte-compares (no engine, no rollout).

Claim boundary: pooled top-1 and within-state engine-truth AUC of the three
frozen N1 rankers on one angle x launch-power inventory over 15 N1 source
members; no other models, no adaptation, no fresh gameplay, no sealed benchmark
(#64/#65 stay sealed), no state-difficulty claim; #87/#89/#90/#92/#93
dispositions are inputs and are not re-opened.  Every interval is DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import time

from scripts import run_second_parameterization_probe as p93

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-94-launch-power-v1"
DATA = ROOT / "data/issue-94-launch-power"
GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"

EIGHTY = p93.EIGHTY
EIGHTY_SEVEN = p93.EIGHTY_SEVEN
EIGHTY_NINE = p93.EIGHTY_NINE
NINETY_TWO = p93.NINETY_TWO
NINETY_THREE = p93.OUTPUT
DYNAMICS = p93.DYNAMICS
CAMPAIGN = p93.CAMPAIGN
ISSUE71 = p93.ISSUE71
PLAYER_SOURCE = p93.PLAYER_SOURCE

IDENTITY = "issue-94-launch-power-v1"
SCHEMA_SUPPORT = "issue_94_training_action_support_v1"
SCHEMA_ENGINE = "issue_94_engine_action_check_v1"
SCHEMA_MEMBERS = "issue_94_members_v1"
SCHEMA_PLAN = "issue_94_launch_power_plan_v1"
SCHEMA_ORACLE = "issue_94_oracle_record_v1"
SCHEMA_DECISION = "issue_94_decision_record_v1"
SCHEMA_LEDGER = "issue_94_ledger_v1"
SCHEMA_SMOKE = "issue_94_smoke_v1"
SCHEMA_COMPUTE = "issue_94_compute_v1"
SCHEMA_REPORT = "issue_94_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_launch_power_probe --validate"

MODEL_SYSTEMS = ("continuous-fixed-h1", "continuous-fixed-h5", "hybrid-fixed-h1")
SEEDS = (20260908, 20260909, 20260910)
ORACLE_SEED_LABEL = 20260908  # #74 matched training-seed label; never an engine input
DEVICE = "cuda"
ARM = "power"
PREFIX = "p"

# execution bounds: #87/#93 caps/ports/attempt limit, #89 cold-start mitigation
# (staggered dispatch, extended connect/readiness bounds, no port probes)
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
        "c24_margin": 0.0, "c25_supported_max_auc": 0.50,
        "c25_not_supported_min_auc": 0.60}
COMPLETION_TRIGGER = 0.25
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment",
                      "readiness_or_precision_insufficient")
STOP_TOKEN = "readiness_or_precision_insufficient"
MIN_SUBMAX_SPEEDS = 3

TAP_MS = 0
RELEASE_MS = 1000
ANGLES = (10.0, 30.0, 50.0, 70.0)
RADII = (5, 9, 13, 17, 80)
SATURATED_RADIUS = 80
SMOKE_MEMBER = "issue-77-n1-001"
SMOKE_ANGLE_INDEX = 1           # 30 deg
SMOKE_RADIUS_INDICES = (0, 2, 4)  # smallest (5 px), middle (13 px), saturated (80 px)
SMOKE_MIN_SPEED_GAP = 0.5

CLAIMS = {
    "C24": ("On an angle x launch-power inventory, the three frozen rankers' pooled top-1 "
            "is below inventory-matched chance."),
    "C25": ("On an angle x launch-power inventory, the frozen rankers' within-state "
            "engine-truth AUC is at or below chance."),
}
STATED_PREDICTIONS = {
    "source": ("ticket #94 owner-decision defaults (no owner comment before freeze); "
               "recorded before any engine slot and never edited after"),
    "C24": {"expected_token": "supported",
            "rationale": ("pooled top-1 was at or below inventory-matched chance in point "
                          "estimate on all three saturated-speed inventories (0/108, 7/81, "
                          "8/126), and every inner-radius candidate is out of the rankers' "
                          "training support, so nothing predicts a gain in selection.")},
    "C25": {"expected_token": "no prediction",
            "rationale": ("no directional prediction: member-clustered AUC moved with the "
                          "inventory (0.4180 / 0.5512 / 0.6178), so the prior evidence does "
                          "not fix its sign on a new engine-side coordinate.")},
}
READING_MATRIX = [
    {"row": 1, "C24": "supported", "C25": "supported",
     "reading": ("Selection and ordering both fail when an engine-side coordinate other than "
                 "angle varies; the one-dimensional engine-scope limit narrows.")},
    {"row": 2, "C24": "supported", "C25": "not_supported_by_this_experiment",
     "reading": ("Selection fails while ordering tracks an engine-relevant coordinate; the "
                 "ordering/selection dissociation is shown on an engine-effective second "
                 "coordinate (earns r6 I49's \"We show\", scoped to this inventory).")},
    {"row": 3, "C24": "not_supported_by_this_experiment", "C25": "any",
     "reading": ("Pooled top-1 is not below chance once launch power varies; the r6 headline "
                 "is re-scoped to saturated-speed inventories.")},
    {"row": 4, "C24": "any readiness_or_precision_insufficient", "C25": "any",
     "reading": ("Report descriptively in the appendix; body scope sentences unchanged "
                 "except a pointer.")},
]
READING_MATCH_RULE = ("rows are matched in table order and the first match applies: row 1 "
                      "(C24 supported, C25 supported), row 2 (C24 supported, C25 "
                      "not_supported), row 3 (C24 not_supported, any C25), row 4 (any "
                      "remaining combination, i.e. either claim insufficient)")

SOURCE_FILES = (
    "scripts/run_launch_power_probe.py",
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

read_json = p93.read_json
json_text = p93.json_text
write_json = p93.write_json
sha256_of = p93.sha256_of
spearman = p93.spearman
median = p93.median


def log(message):
    print(f"[issue-94-launch-power] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
# candidate inventory (fixed by ticket #94 before any engine execution)
# ---------------------------------------------------------------------------

def launch_geometry(drag_x, drag_y):
    return {"radius_px": math.hypot(drag_x, drag_y),
            "angle_deg": math.degrees(math.atan2(drag_y, -drag_x))}


def design_inventory():
    """Arm P: (round(-r cos a), round(r sin a)), ordinal = 5*i + j."""
    items = []
    for i, angle in enumerate(ANGLES):
        for j, radius in enumerate(RADII):
            drag_x = round(-radius * math.cos(math.radians(angle)))
            drag_y = round(radius * math.sin(math.radians(angle)))
            items.append({"ordinal": 5 * i + j, "angle_index": i, "radius_index": j,
                          "angle_nominal_deg": angle, "radius_nominal_px": radius,
                          "action": {"drag_x": drag_x, "drag_y": drag_y,
                                     "tap_time_ms": TAP_MS, "release_time_ms": RELEASE_MS}})
    return items


def realize(items, clamp_px):
    """Expected engine launch per candidate (ABBird.cs:188-190 clamp, :237 impulse)."""
    rows = []
    for item in items:
        geometry = launch_geometry(item["action"]["drag_x"], item["action"]["drag_y"])
        clamped = geometry["radius_px"] > clamp_px
        rows.append({"ordinal": item["ordinal"], "angle_index": item["angle_index"],
                     "radius_index": item["radius_index"],
                     "angle_nominal_deg": item["angle_nominal_deg"],
                     "radius_nominal_px": item["radius_nominal_px"],
                     "drag_x": item["action"]["drag_x"], "drag_y": item["action"]["drag_y"],
                     "realized_radius_px": geometry["radius_px"],
                     "realized_angle_deg": geometry["angle_deg"],
                     "clamped_to_drag_radius": clamped,
                     "expected_speed": 10.0 if clamped else 10.0 * geometry["radius_px"] / clamp_px})
    return rows


def identical_launch_removals(rows):
    removed, pairs = set(), []
    for index, a in enumerate(rows):
        for b in rows[index + 1:]:
            if (abs(a["realized_angle_deg"] - b["realized_angle_deg"]) < 1e-9
                    and abs(a["expected_speed"] - b["expected_speed"]) < 1e-9):
                pairs.append([a["ordinal"], b["ordinal"]])
                removed.add(b["ordinal"])
    return pairs, sorted(removed)


# ---------------------------------------------------------------------------
# WP0.1 action-bound authority (stop condition)
# ---------------------------------------------------------------------------

def repo_line(relative, pattern):
    import re
    path = ROOT / relative
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line):
            return {"file": relative, "line": number, "text": line.strip()}
    raise ValueError(f"pattern {pattern!r} not found in {relative}")


def jar_shoot_path():
    """javap of the shipped interface jar: Agent.shoot -> ShootAndTapSchema.shoot
    forwards (x, y, t_tap) into ProxyTapShootMessage with no comparison."""
    import subprocess
    import tempfile
    jar = CAMPAIGN / "player" / "game_playing_interface.jar"
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(["unzip", "-q", "-o", str(jar), "server/schema/ShootAndTapSchema.class",
                        "server/util/Shot.class"], cwd=scratch, check=True)
        schema = subprocess.run(["javap", "-c", "-p", "server/schema/ShootAndTapSchema.class"],
                                cwd=scratch, check=True, capture_output=True, text=True).stdout
    body = schema.split("public byte[] shoot(")[1].split("public org.json")[0]
    comparisons = [line.strip() for line in body.splitlines()
                   if any(op in line for op in ("if_icmp", "ifle", "ifge", "iflt", "ifgt"))]
    shot_getters = [line.strip() for line in body.splitlines() if "Shot.get" in line]
    return {"jar": str(jar.relative_to(ROOT)), "jar_sha256": sha256_of(jar),
            "method": "server.schema.ShootAndTapSchema.shoot(List<Shot>)",
            "shot_getters": shot_getters, "branch_instructions": comparisons,
            "reading": ("the only branches are the list loop bound and the engine reply code "
                        "(1 / -1); x, y and t_tap are forwarded to ProxyTapShootMessage "
                        "unchanged")}


def action_bound_authority():
    pipeline = {
        "successor_cohort_ACTION_BOUNDS": repo_line("world_model/data/successor_cohort.py",
                                                    r"^ACTION_BOUNDS: Final"),
        "successor_cohort_drag_x": repo_line("world_model/data/successor_cohort.py",
                                             r'"drag_x": \[-160, -10\]'),
        "successor_cohort_sampler": repo_line("world_model/data/successor_cohort.py",
                                              r"x_bounds = tuple\(ACTION_BOUNDS"),
        "gameplay_success_action_bounds": repo_line("world_model/planning/gameplay_success.py",
                                                    r'"action_bounds": \{'),
        "gameplay_success_validator": repo_line("world_model/planning/gameplay_success.py",
                                                r'value.get\("action_bounds"\) != \{'),
        "lineage_scaling_validator": repo_line("world_model/training/lineage_scaling.py",
                                               r"self.action_bounds.contains\(candidate"),
    }
    execution_path = {
        "selector_action_tensor": repo_line("scripts/run_engine_outcome_reactive_diagnostic.py",
                                            r"def action_tensor"),
        "command_anchor": repo_line("scripts/slingshot_readiness.py",
                                    r"game_x, game_y = start_x \+ dx, start_y - dy"),
        "bridge_transport": repo_line("src/webui/bridge.py",
                                      r'self._send\(code, "iiii"'),
        "engine_tapshoot_drag_read": p93.source_line(
            "Scripts/Assembly-CSharp/AIBirdsConnection.cs", r'float asFloat = data\[2\]\["x"\]'),
        "engine_tapshoot_simulated_drag": p93.source_line(
            "Scripts/Assembly-CSharp/AIBirdsConnection.cs",
            r"Vector2 vector3 = new Vector2\(asFloat, \(float\)Screen.height - asFloat2\)"),
        "engine_hud_drag": p93.source_line("Scripts/Assembly-CSharp/HUD.cs",
                                           r"selectedBird.DragBird\(vector\)"),
        "engine_drag_clamp": p93.source_line("Scripts/Assembly-CSharp/ABBird.cs",
                                             r"Vector2.Distance\(dragPosition, vector\) > _dragRadius"),
        "interface_jar": jar_shoot_path(),
    }
    return {
        "finding": "pipeline_only",
        "pipeline_bounds": pipeline,
        "execution_path": execution_path,
        "reading": (
            "drag_x in [-160, -10] (successor_cohort.py ACTION_BOUNDS) and [-160, -40] "
            "(gameplay_success.py action_bounds) are enforced only by pipeline samplers and "
            "plan/candidate validators (successor_cohort sampler, gameplay_success validator, "
            "lineage_scaling ActionRankingState). The executed path used here - "
            "ReactiveSelector.action_tensor (scoring), slingshot_readiness."
            "_anchored_action_and_command (command), ScienceBirdsBridge.shoot (socket), the "
            "interface jar ShootAndTapSchema.shoot, AIBirdsConnection.TapShoot, HUD.Drag and "
            "ABBird.DragBird (engine) - contains no drag bound; the engine's only limit is the "
            "_dragRadius clamp from above. The runner therefore uses its own frozen action "
            "contract (the 20 listed candidates, tap 0 ms, release 1000 ms)."),
        "bound_applied_by_rule": None,
    }


# ---------------------------------------------------------------------------
# WP0.2 engine realization; WP0.3 training support; members and anchors
# ---------------------------------------------------------------------------

def engine_action_check(plan87, oracles87, items):
    engine93 = read_json(NINETY_THREE / "engine_action_check.json")
    drag_radius_prefab = p93.source_line(
        "Resources/prefabs/gameworld/characters/birds/BirdRed.prefab", r"_dragRadius:")
    drag_radius = float(drag_radius_prefab["text"].split(":")[1])
    citations = {
        "clamp": p93.source_line("Scripts/Assembly-CSharp/ABBird.cs",
                                 r"Vector2.Distance\(dragPosition, vector\) > _dragRadius"),
        "clamp_projection": p93.source_line("Scripts/Assembly-CSharp/ABBird.cs",
                                            r"normalized \* _dragRadius \+ vector"),
        "launch_impulse": p93.source_line("Scripts/Assembly-CSharp/ABBird.cs",
                                          r"Vector2 force = -vector2 \* _launchForce"),
        "max_launch_speed": p93.source_line("Scripts/Assembly-CSharp/ABConstants.cs",
                                            r"BIRD_MAX_LANUCH_SPEED"),
        "hud_drag": p93.source_line("Scripts/Assembly-CSharp/HUD.cs",
                                    r"selectedBird.DragBird\(vector\)"),
        "red_bird_drag_radius": drag_radius_prefab,
    }
    launches = []
    executed = [record for record in oracles87.values() if record.get("execution")]
    for count, record in enumerate(sorted(executed, key=lambda r: r["cell"]["identity"]), 1):
        attempt = EIGHTY_SEVEN / "attempts" / record["cell"]["identity"]
        segment = read_json(attempt / "shot-1" / "segment.json")
        event = p93.first_launch(record["execution"]["native_root"])
        vx, vy = event["payload"]["launch_velocity"]
        drag = segment["action"]["drag_release"]
        launches.append({"cell": record["cell"]["identity"], "drag": list(drag),
                         "declared_angle_deg": launch_geometry(*drag)["angle_deg"],
                         "launch_speed": math.hypot(vx, vy),
                         "launch_angle_deg": math.degrees(math.atan2(vy, vx)),
                         "pixels_per_world_unit": segment["action"]["slingshot_reference"]["pixelsPerWorldUnit"]})
        if count % 50 == 0:
            log(f"WP0.2 retained #87 launches read {count}/{len(executed)}")
    scale = {round(row["pixels_per_world_unit"], 6) for row in launches}
    if len(scale) != 1 or scale != {round(engine93["pixels_per_world_unit"], 6)}:
        raise ValueError(f"slingshot scale differs from #93: {sorted(scale)}")
    pixels_per_unit = scale.pop()
    clamp_px = drag_radius * pixels_per_unit
    rows = realize(items, clamp_px)
    pairs, removed = identical_launch_removals(rows)
    offsets = [row["launch_angle_deg"] - row["declared_angle_deg"] for row in launches]
    lateral = [80.0 * math.sin(math.radians(abs(o))) for o in offsets]
    submax = sorted({round(row["expected_speed"], 9) for row in rows
                     if row["ordinal"] not in removed and not row["clamped_to_drag_radius"]})
    return {
        "schema": SCHEMA_ENGINE, "identity": IDENTITY, "computed_at": utc_now(),
        "source_citations": citations,
        "mechanics": (
            "HUD.Drag -> ABBird.DragBird projects any drag farther than _dragRadius (1 world "
            "unit, BirdRed prefab) onto the drag circle; LaunchBird applies an impulse "
            "proportional to the clamped displacement (speed 10 at the circle, "
            "BIRD_MAX_LANUCH_SPEED), so a drag of pixel radius r < clamp launches at expected "
            "speed 10 r / clamp in its drag direction and every drag beyond it at 10"),
        "pixels_per_world_unit": pixels_per_unit,
        "clamp_radius_px": clamp_px,
        "issue93_engine_action_check_sha256": sha256_of(NINETY_THREE / "engine_action_check.json"),
        "retained_issue87_launches": {
            "cells": len(launches),
            "speed_min": min(r["launch_speed"] for r in launches),
            "speed_max": max(r["launch_speed"] for r in launches),
            "angle_offset_deg_range": [min(offsets), max(offsets)],
            "implied_lateral_anchor_residual_px_max": max(lateral),
            "reading": ("the executed launch direction differs from the declared drag direction "
                        "by at most the listed offset at radius 80; the implied sub-pixel "
                        "anchor residual is a larger angular share at 5-17 px, so the "
                        "per-slot realized launch angle and speed are recorded from engine "
                        "telemetry and reported beside the expected values"),
        },
        "candidates": rows,
        "identical_launch_pairs": pairs,
        "removed_by_rule": removed,
        "removal_rule": ("a candidate that realizes an identical launch (angle and expected "
                         "speed) to a lower-ordinal candidate is removed before freeze"),
        "distinct_submaximal_speeds": len(submax),
        "stop_condition": {"minimum_distinct_submaximal_speeds": MIN_SUBMAX_SPEEDS,
                           "triggered": len(submax) < MIN_SUBMAX_SPEEDS},
        "action_bound_authority": action_bound_authority(),
    }


def training_action_support(items, engine):
    dynamics = read_json(DYNAMICS / "plan.json")
    issue71_plan = read_json(ISSUE71 / "plan.json")
    fit71 = [index for index in range(1, 3001) if index % 5]
    pool71 = []
    for count, index in enumerate(fit71, 1):
        record, actions = p93.shard_actions(ISSUE71 / "shards" / f"lineage-{index:04d}.pt")
        if record != issue71_plan["records"][index - 1]:
            raise ValueError(f"#71 shard {index} record binding differs")
        pool71.extend(actions.values())
        if count % 400 == 0:
            log(f"WP0.3 #71 predictor-pool shards read {count}/{len(fit71)}")
    pool_n1 = []
    for identity in dynamics["n1_lineages"]["predictor"]:
        record = dynamics["n1_lineages"]["records"][identity]
        stored, actions = p93.shard_actions(DYNAMICS / "shards" / f"lineage-n1-{record['ordinal']:03d}.pt")
        if stored != record:
            raise ValueError(f"N1 shard {identity} record binding differs")
        pool_n1.extend(actions.values())
    n1 = p93.support_summary(pool_n1)
    s71 = p93.support_summary(pool71)
    n1_set = sorted(set(pool_n1))
    set71 = set(pool71)
    rows = []
    for item, realized in zip(items, engine["candidates"], strict=True):
        action = item["action"]
        key = (action["drag_x"], action["drag_y"], action["tap_time_ms"], action["release_time_ms"])
        radius = realized["realized_radius_px"]
        rows.append({
            "ordinal": item["ordinal"], "drag_x": action["drag_x"], "drag_y": action["drag_y"],
            "radius_px": radius, "expected_speed": realized["expected_speed"],
            "exact_in_n1_training": key in set(n1_set), "exact_in_issue71_training": key in set71,
            "radius_within_n1_training_range": n1["radius_px_range"][0] <= radius <= n1["radius_px_range"][1],
            "radius_within_issue71_training_range": s71["radius_px_range"][0] <= radius <= s71["radius_px_range"][1],
            "nearest_n1_drag_distance_px": min(math.hypot(action["drag_x"] - a[0], action["drag_y"] - a[1])
                                               for a in n1_set),
        })
    inner = [row for row in rows if row["radius_px"] < engine["clamp_radius_px"]]
    return {
        "schema": SCHEMA_SUPPORT, "identity": IDENTITY, "computed_at": utc_now(),
        "ranker_component": ("predictor only: ReactiveSelector applies model.carrier Delta "
                             "times per candidate action; the parser takes no action input"),
        "sources": {"issue71_predictor_pool": s71, "n1_predictor_pool": n1},
        "n1_training_actions": [list(a) for a in n1_set],
        "issue93_training_action_support_sha256": sha256_of(NINETY_THREE / "training_action_support.json"),
        "per_candidate": rows,
        "reading": (
            f"the N1 predictor pool (the only release-1000 training actions) spans pull radius "
            f"{n1['radius_px_range'][0]:.2f}-{n1['radius_px_range'][1]:.2f} px; "
            f"{sum(1 for r in inner if not r['radius_within_n1_training_range'])}/{len(inner)} "
            "inner-radius candidates are outside it on the ranker input (out of training "
            "support); 0 candidates are exact training actions. Printed, not a reason to "
            "exclude."),
    }


def member_states():
    plan87, _, _ = p93.load_issue_87()
    return plan87, p93.member_states(plan87)


def frozen_anchor(member):
    """Retained anchors verified by #93: the #87 anchor where #93 found it retained;
    otherwise the #93 captured anchor its decision records scored from."""
    retention = {row["member"]: row for row in read_json(NINETY_THREE / "anchor_retention.json")["members"]}
    row = retention[member]
    if row["retained"]:
        anchor = {"observation_identity": row["observation_identity"], "fixed_step": row["fixed_step"],
                  "fixed_time_seconds": row["fixed_time_seconds"], "frame_path": row["path"],
                  "sha256": row["sha256_recorded"], "source": "issue87_retained",
                  "cell": row["anchor_cell"]}
    else:
        record = read_json(NINETY_THREE / "records" /
                           f"decision--{MODEL_SYSTEMS[0]}--seed{SEEDS[0]}--{member}--grid.json")
        source = record["anchor"]
        anchors = {read_json(path)["anchor"]["sha256"]
                   for path in (NINETY_THREE / "records").glob(f"decision--*--{member}--*.json")}
        if anchors != {source["sha256"]}:
            raise ValueError(f"#93 decision anchors of {member} disagree")
        anchor = {key: source[key] for key in ("observation_identity", "fixed_step",
                                               "fixed_time_seconds", "frame_path", "sha256",
                                               "cell")} | {"source": "issue93_captured"}
    if sha256_of(anchor["frame_path"]) != anchor["sha256"]:
        raise ValueError(f"anchor frame of {member} differs from its recorded sha256")
    return anchor


# ---------------------------------------------------------------------------
# scheduled cells
# ---------------------------------------------------------------------------

def member_inventory(member, items):
    return [{"ordinal": item["ordinal"], "branch_identity": f"{member}-{PREFIX}{item['ordinal']:02d}",
             "action": dict(item["action"])} for item in items]


def oracle_identity(member, ordinal, kind="oracle"):
    return f"{kind}--{member}--{PREFIX}{ordinal:02d}"


def decision_identity(system, seed, member):
    return f"decision--{system}--seed{seed}--{member}--{ARM}"


def make_cell(state, item, kind="oracle"):
    return {"identity": oracle_identity(state["member"], item["ordinal"], kind), "arm": ARM,
            "member": state["member"], "state": state["canonical_state"],
            "ordinal": item["ordinal"], "branch_identity": item["branch_identity"],
            "action": dict(item["action"]), "engine_seed": state["engine_seed"],
            "retained_anchor_path": state["anchor"]["frame_path"]}


def scheduled_cells(plan):
    """Member order, ordinal order within member."""
    return [make_cell(state, item) for state in plan["states"]
            for item in plan["inventory"]["members"][state["member"]]]


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
    launch telemetry only (no predictor scoring inside the attempt)."""
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
        _, scenario = c89.live_files.materialize(member, member["template"], attempt / "authority")
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
        bridge = c89.connect_with_retry("127.0.0.1", agent_port, timeout=AGENT_SOCKET_SECONDS,
                                        deadline_seconds=AGENT_CONNECT_DEADLINE_SECONDS)
        physics = c89.ScienceBirdsBridge("127.0.0.1", physics_port, timeout=PHYSICS_SOCKET_SECONDS)
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
            record["frame_vs_retained_anchor"] = {"retained_anchor_path": cell["retained_anchor_path"],
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
        if segment["summary"]["first_fixed_step"] != DECISION_FIXED_STEP or initial_png != frame_bytes:
            raise ValueError("executed shot pre-intervention state differs from its decision frame")
        verdict = e85.channel_scan(segment["native_root"])
        if verdict["bird_launches"] != 1:
            raise ValueError("executed shot did not contain exactly one native launch")
        event = p93.first_launch(segment["native_root"])
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
    print(f"issue-94 cell {identity} complete={record['failure'] is None} "
          f"success={(record['outcome'] or {}).get('first_shot_success')} "
          f"speed={(record['launch'] or {}).get('speed')} failure={record['failure']}", flush=True)
    return record


# ---------------------------------------------------------------------------
# supervisor: <=4 isolated workers, staggered cold starts, caps, resumable ledger
# ---------------------------------------------------------------------------

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
            if stop_reason is None and p93.free_disk(attempts) < MINIMUM_FREE_BYTES:
                stop_reason = "minimum_free_storage"
            if stop_reason is None and index < len(pending) and index % 25 == 0 and index != checked:
                checked = index
                if p93.tree_bytes(attempts) > ARTIFACT_BYTES_CAP:
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
                payload = {"output": str(output), "attempts": str(attempts), "records": records,
                           "cell": cell, "state": by_state[cell["state"]],
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
            time.sleep(1)
            if not live:
                continue
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
                remaining = len(cells) - done
                walls = [e["wall_seconds"] for e in ledger["cells"].values() if e["wall_seconds"]]
                per_slot = max(sum(walls) / len(walls) / WORKERS, START_STAGGER_SECONDS) if walls else 0.0
                log(f"{phase} [{done}/{len(cells)}] {cell['identity']} "
                    f"success={(record.get('outcome') or {}).get('first_shot_success')} "
                    f"speed={(record.get('launch') or {}).get('speed')} "
                    f"failure={record['failure']} wall={wall:.0f}s "
                    f"eta={remaining * per_slot / 3600:.2f}h "
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
    log(f"{phase} finished status={ledger['status']} wall={ledger['wall_seconds_elapsed'] / 3600:.2f}h")
    return ledger


# ---------------------------------------------------------------------------
# decision-only scoring from the retained anchors
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


def score(output):
    output = Path(output)
    plan = load_plan(output)
    wlc, adapter, objective = load_scoring_stack()
    ledger_file = output / "ledger-score.json"
    ledger = read_json(ledger_file) if ledger_file.is_file() else {
        "schema": SCHEMA_LEDGER, "identity": IDENTITY, "phase": "score",
        "gpu_seconds_elapsed": 0.0, "cells": 0}
    models = {}
    gpu = 0.0
    written = 0
    for state in plan["states"]:
        anchor = state["anchor"]
        failure = None
        if not Path(anchor["frame_path"]).is_file() or sha256_of(anchor["frame_path"]) != anchor["sha256"]:
            failure = "anchor_frame_missing_or_changed"
        inventory = plan["inventory"]["members"][state["member"]]
        for system in MODEL_SYSTEMS:
            for seed in SEEDS:
                identity = decision_identity(system, seed, state["member"])
                path = output / "records" / f"{identity}.json"
                if path.is_file():
                    continue
                record = {"schema": SCHEMA_DECISION, "plan_identity": plan["identity"],
                          "plan_version": plan["version"],
                          "cell": {"identity": identity, "system": system, "seed": seed,
                                   "member": state["member"], "arm": ARM,
                                   "state": state["canonical_state"]},
                          "anchor": anchor, "carrier_sha256": None, "decision": None,
                          "failure": None, "failure_kind": None, "gpu_seconds": 0.0,
                          "issue_64_authorized": False}
                if failure is not None:
                    record["failure"] = f"decision_failure: {failure}"
                    record["failure_kind"] = "decision_failure"
                else:
                    decision, seconds = score_inventory(wlc, adapter, objective, models, plan,
                                                        anchor, system, seed, inventory)
                    gpu += seconds
                    record["gpu_seconds"] = seconds
                    record["carrier_sha256"] = anchor["sha256"]
                    record["decision"] = decision
                    if decision["failure"] is not None:
                        record["failure"] = f"decision_failure: {decision['failure']}"
                        record["failure_kind"] = "decision_failure"
                write_json(path, record)
                written += 1
        log(f"score: member {state['member']} done ({written} records so far)")
    ledger["gpu_seconds_elapsed"] += gpu
    ledger["cells"] += written
    write_json(ledger_file, ledger)
    if ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: gpu_cap_exceeded")
    log(f"score: {written} decision records written, gpu {gpu:.1f}s "
        f"(total {ledger['gpu_seconds_elapsed']:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# WP0 + plan freeze
# ---------------------------------------------------------------------------

def input_bindings():
    names = {
        "issue_80_plan": EIGHTY / "plan.json",
        "issue_87_plan": EIGHTY_SEVEN / "plan.json",
        "issue_89_summary": EIGHTY_NINE / "summary.json",
        "issue_92_summary": NINETY_TWO / "summary.json",
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "issue_77_campaign_plan": CAMPAIGN / "plan.json",
        "issue_93_plan": NINETY_THREE / "plan.json",
        "issue_93_summary": NINETY_THREE / "summary.json",
        "issue_93_anchor_retention": NINETY_THREE / "anchor_retention.json",
        "issue_93_engine_action_check": NINETY_THREE / "engine_action_check.json",
        "issue_93_training_action_support": NINETY_THREE / "training_action_support.json",
        "issue_93_members": NINETY_THREE / "members.json",
        "wp0_engine_action_check": OUTPUT / "engine_action_check.json",
        "wp0_training_action_support": OUTPUT / "training_action_support.json",
        "wp0_members": OUTPUT / "members.json",
    }
    return [{"name": name, "artifact": str(path), "sha256": sha256_of(path)}
            for name, path in names.items()]


def admissible_design(engine):
    removed = set(engine["removed_by_rule"])
    return [item for item in design_inventory() if item["ordinal"] not in removed]


def frozen_states():
    members = read_json(OUTPUT / "members.json")["members"]
    plan87 = read_json(EIGHTY_SEVEN / "plan.json")
    by_identity = {state["identity"]: state for state in plan87["states"]}
    states = []
    for entry in members:
        outcomes = EIGHTY / "candidate-outcomes" / f"{entry['member']}.json"
        states.append({
            **entry,
            "execution_member_identity": by_identity[entry["canonical_state"]]["execution_member"]["identity"],
            "issue80_candidate_outcomes": {"path": str(outcomes), "sha256": sha256_of(outcomes)},
            "anchor": frozen_anchor(entry["member"]),
            "anchor_rule": ("retained anchors verified by #93: the #87 retained anchor where #93 "
                            "found it retained, else the #93 captured anchor its decision records "
                            "scored from (sha256-bound)"),
        })
    return states


def frozen_plan(frozen_at):
    engine = read_json(OUTPUT / "engine_action_check.json")
    support = read_json(OUTPUT / "training_action_support.json")
    design = admissible_design(engine)
    states = frozen_states()
    from scripts import run_engine_outcome_reactive_diagnostic as e85
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal",
        "frozen_at": frozen_at, "frozen_before_scoring_run": True,
        "issue_64_authorized": False, "validation_command": VALIDATION_COMMAND,
        "runner": "scripts/run_launch_power_probe.py",
        "changelog": [{"version": 1,
                       "change": ("initial and terminal freeze of the angle x launch-power "
                                  "inventory, 15-member state list, rankers, estimands, both "
                                  "frozen disposition rules, reading matrix, stated predictions, "
                                  "caps and the declared completion pass, before any engine slot"),
                       "why": "ticket #94 (owner-decision defaults; no owner comment before freeze)",
                       "evidence": "engine_action_check.json; training_action_support.json"}],
        "owner_decisions": {
            "inventory": "4 angles {10,30,50,70} deg x radii {5,9,13,17,80} px (default)",
            "margins": ("C24 cross-pool q2 criterion (margin 0.0); C25 0.50 / 0.60; readiness "
                        ">=4 ceiling members, >=75% verdict coverage (default)"),
            "stated_predictions": "C24 supported, C25 no prediction (default)",
            "completion_pass": "allowed once if typed failures > 25% (default)",
        },
        "inventory": {
            "design": design,
            "realization": [row for row in engine["candidates"]
                            if row["ordinal"] not in set(engine["removed_by_rule"])],
            "members": {state["member"]: member_inventory(state["member"], design) for state in states},
            "rule": ("angles a in (10,30,50,70) deg x radii r in (5,9,13,17,80) px, (drag_x, drag_y) "
                     "= (round(-r cos a), round(r sin a)); ordinal = 5*i + j; tap 0 ms, release "
                     "1000 ms; one shot per branch; branch identity {member}-pNN"),
            "identical_launch_removals": engine["removed_by_rule"],
            "action_bound_authority": engine["action_bound_authority"]["reading"],
            "training_support": support["reading"],
            "clamp_radius_px": engine["clamp_radius_px"],
        },
        "states": states,
        "unit_of_analysis": ("N1 source member (15); canonical state = lowest state identity "
                             "(#92 selection-validity / #93 members.json)"),
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
                               "run_engine_outcome_reactive_diagnostic.channel_scan: pig_removed "
                               "(pig_removed / entity_death / entity_destroyed events)"),
            "executed_action_check": "segment action drag_release/release/tap == declared",
            "launch_telemetry": "first native bird_launched event launch_velocity per slot",
            "slot_order": "member order, ordinal order within member",
            "heavy_media": str(DATA / "attempts"),
        },
        "scheduled_slots": len(design) * len(states),
        "caps": {"wall_seconds": WALL_CAP_SECONDS, "gpu_seconds": GPU_CAP_SECONDS,
                 "workers": WORKERS, "artifact_bytes": ARTIFACT_BYTES_CAP,
                 "minimum_free_bytes": MINIMUM_FREE_BYTES,
                 "stop_rule": "cap hit -> remaining slots become typed failures not_executed; publish as-is"},
        "completion_pass": {
            "completion_pass_allowed": True,
            "trigger": f"typed failures > {COMPLETION_TRIGGER:.0%} of scheduled slots after --run",
            "scope": "the same failed slots only, executed once; no other slot, no retry",
            "union": "a completion verdict replaces the failed original; failures stay typed",
        },
        "smoke": {"design": (f"{SMOKE_MEMBER} x ordinals "
                             f"{[5 * SMOKE_ANGLE_INDEX + j for j in SMOKE_RADIUS_INDICES]} "
                             "(30 deg at 5, 13, 80 px)"),
                  "pass": ("every smoke slot launches exactly once with the declared action and "
                           "a sealed decision frame; realized speeds strictly increase with "
                           f"radius, adjacent gaps >= {SMOKE_MIN_SPEED_GAP}, saturated slot "
                           "within 0.01 of 10"),
                  "records": str(OUTPUT / "smoke"), "joined_to_scoring": False},
        "rankers": {
            "systems": list(MODEL_SYSTEMS), "seeds": list(SEEDS),
            "loader": "run_engine_outcome_reactive_diagnostic.load_frozen_predictor",
            "selector": ("ReactiveSelector.choose: Delta carrier applications per candidate, "
                         "argmin predicted cost, ties to the lower ordinal; full per-candidate "
                         "ranking recorded"),
            "carrier": "run_lookahead_control.encode_carrier(load_adapter) of the member's anchor frame",
            "adaptation": "none: no retraining, fine-tuning or few-shot adaptation",
        },
        "dynamics_checkpoints": e85.dynamics_checkpoints(),
        "estimands": {
            "1_ceiling": ("members with >=1 engine-truth success among admissible candidates / "
                          "measurable members, member-clustered descriptive bootstrap interval; "
                          "engine truth follows the #85 detection rule"),
            "2_pooled_top1": ("top-1 = argmin predicted cost over admissible candidates (ties lower "
                              "ordinal), pooled over 3 systems x 3 seeds on ceiling members, vs "
                              "inventory-matched chance k_s/n_s; member-clustered paired "
                              f"difference (top-1 - chance), {BOOTSTRAP_DRAWS} draws, seed "
                              f"{BOOTSTRAP_SEED}, quantiles {list(INTERVAL_QUANTILES)}; per-system "
                              "top-1 also reported"),
            "3_auc": ("per cell (system x seed x ceiling member) P(a successful admissible "
                      "candidate receives lower predicted cost than a failed one), ties 0.5 "
                      "(cell_auc definition, run_cross_pool_selection_audit.py); cell-unit and "
                      "member-clustered AUC_m; per system reported; never pooled outside this "
                      "experiment"),
            "4_top3": "top-3 vs chance 1 - C(n-k,3)/C(n,3); miss-all = prod over ceiling members of (1 - k/n)",
            "5_structure": ("DESCRIPTIVE: Spearman rho of predicted cost vs realized launch speed "
                            "within each fixed angle and vs realized launch angle within each "
                            "fixed radius (engine telemetry of the member's slots); chosen-ordinal "
                            "and success-ordinal histograms; share of chosen candidates at the "
                            "saturated column"),
            "6_launch_speed": "realized launch speed per executed slot from engine telemetry, per radius column",
        },
        "claims": CLAIMS,
        "disposition_rules": {
            **RULE,
            "readiness": ("readiness_or_precision_insufficient for both claims if ceiling members "
                          "< 4 or fewer than 75% of scheduled oracle slots end with an "
                          "engine-truth verdict (typed failures count against)"),
            "C24": ("cross-pool q2 criterion (#92): supported if the pooled top-1 rate < pooled "
                    "chance AND the member-clustered paired-difference interval lies entirely "
                    "below 0; not_supported_by_this_experiment if the paired-difference point "
                    "estimate is >= 0; otherwise readiness_or_precision_insufficient"),
            "C25": ("on the member-clustered point estimate AUC_m: supported if AUC_m <= 0.50; "
                    "not_supported_by_this_experiment if AUC_m >= 0.60; otherwise "
                    "readiness_or_precision_insufficient; the interval is printed and is never "
                    "the decision rule"),
            "tokens": list(DISPOSITION_TOKENS),
        },
        "reading_matrix": READING_MATRIX,
        "reading_match_rule": READING_MATCH_RULE,
        "stated_predictions": STATED_PREDICTIONS,
        "expected_structure": {"members": len(states), "slots": len(design) * len(states),
                               "decision_cells": len(states) * len(MODEL_SYSTEMS) * len(SEEDS)},
        "player_root": str(CAMPAIGN / "player"),
        "inputs": input_bindings(),
        "source_text": {name: (ROOT / name).read_text() for name in SOURCE_FILES},
        "claim_boundary": (
            "pooled top-1 and within-state engine-truth AUC of the three frozen N1 rankers on "
            "one angle x launch-power inventory over 15 N1 source members; does not test other "
            "models, adaptation, fresh gameplay, sealed benchmarks (#64/#65 sealed) or state "
            "difficulty; never pooled with N1 or #93; does not re-open "
            "#87/#89/#90/#92/#93 dispositions"),
    }


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        plan = load_plan(output)
        log(f"existing frozen plan validated (frozen {plan['frozen_at']})")
        return 0
    if (output / "records").exists() or (output / "smoke").exists():
        raise ValueError("engine records exist before the freeze; refusing")
    items = design_inventory()
    plan87, members = member_states()
    if len(members) != 15:
        raise ValueError(f"member-unit state list has {len(members)} members, not 15")
    if [m["member"] for m in members] != [m["member"] for m in read_json(NINETY_THREE / "members.json")["members"]]:
        raise ValueError("member list differs from #93")
    write_json(output / "members.json", {
        "schema": SCHEMA_MEMBERS, "identity": IDENTITY, "computed_at": utc_now(),
        "source": str(EIGHTY_SEVEN / "plan.json"), "source_sha256": sha256_of(EIGHTY_SEVEN / "plan.json"),
        "issue93_members_sha256": sha256_of(NINETY_THREE / "members.json"), "members": members})
    _, _, oracles87 = p93.load_issue_87()
    engine = engine_action_check(plan87, oracles87, items)
    write_json(output / "engine_action_check.json", engine)
    log(f"WP0.1 action-bound authority: {engine['action_bound_authority']['finding']} -> the "
        "runner uses its own frozen action contract")
    log(f"WP0.2 clamp {engine['clamp_radius_px']:.6f} px; removals by rule "
        f"{engine['removed_by_rule']} (pairs {engine['identical_launch_pairs']}); distinct "
        f"sub-maximal speeds {engine['distinct_submaximal_speeds']}")
    for row in engine["candidates"]:
        log(f"  p{row['ordinal']:02d} a={row['angle_nominal_deg']:.0f} r={row['radius_nominal_px']} "
            f"drag=({row['drag_x']},{row['drag_y']}) realized angle {row['realized_angle_deg']:.2f} "
            f"radius {row['realized_radius_px']:.3f} expected speed {row['expected_speed']:.4f}")
    if engine["stop_condition"]["triggered"]:
        raise ValueError("STOP: fewer than 3 distinct sub-maximal speeds survive; comment on #94")
    support = training_action_support(items, engine)
    write_json(output / "training_action_support.json", support)
    log(f"WP0.3 {support['reading']}")
    plan = frozen_plan(utc_now())
    for marker in PLACEHOLDER_MARKERS:
        if marker in json_text({k: v for k, v in plan.items() if k != "source_text"}):
            raise ValueError(f"frozen plan contains missing-value marker {marker!r}")
    path.write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published at {path}: {len(scheduled_cells(plan))} slots over "
        f"{len(plan['states'])} members; no engine slot has run")
    return 0


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-94 protocol")
    if not plan.get("frozen_before_scoring_run"):
        raise ValueError("plan.json does not declare a pre-scoring freeze")
    removed = set(plan["inventory"]["identical_launch_removals"])
    if plan["inventory"]["design"] != [i for i in design_inventory() if i["ordinal"] not in removed]:
        raise ValueError("runner inventory differs from the frozen plan")
    expected = plan["expected_structure"]
    if len(plan["states"]) != expected["members"] or len(scheduled_cells(plan)) != expected["slots"]:
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
        for system in MODEL_SYSTEMS:
            for seed in SEEDS:
                identity = decision_identity(system, seed, state["member"])
                path = Path(output) / "records" / f"{identity}.json"
                if path.is_file():
                    record = read_json(path)
                    if record.get("schema") != SCHEMA_DECISION or record.get("plan_identity") != IDENTITY:
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


def cell_auc(pos, neg):
    """cell_auc definition: P(success has lower predicted cost than failure); ties 0.5."""
    score_sum = sum(1.0 if p < n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return score_sum / (len(pos) * len(neg))


def decision_rows(plan, oracle, decisions):
    design = {item["ordinal"]: item for item in plan["inventory"]["design"]}
    realization = {row["ordinal"]: row for row in plan["inventory"]["realization"]}
    rows = []
    for state in plan["states"]:
        verdicts, launches = {}, {}
        for ordinal in design:
            record = oracle.get(oracle_identity(state["member"], ordinal))
            if record is not None and record["outcome"] is not None:
                verdicts[ordinal] = record["outcome"]["first_shot_success"]
            if record is not None and record.get("launch"):
                launches[ordinal] = record["launch"]
        successes = sorted(o for o, hit in verdicts.items() if hit)
        for system in MODEL_SYSTEMS:
            for seed in SEEDS:
                identity = decision_identity(system, seed, state["member"])
                record = decisions.get(identity)
                row = {"cell_identity": identity, "member": state["member"],
                       "state": state["canonical_state"], "family": state["generator_family"],
                       "exposure_role": state["exposure_role"], "system": system, "seed": seed,
                       "verdict_slots": len(verdicts), "successes": successes,
                       "n_successes": len(successes),
                       "typed_failure": record is None or record["failure"] is not None,
                       "failure": None if record is None else record["failure"],
                       "anchor_source": None if record is None else record["anchor"]["source"]}
                if row["typed_failure"]:
                    row.update(ranking=[], selector_chosen=None, measurable=False)
                    rows.append(row)
                    continue
                ranking = [(entry["ordinal"], entry["predicted_cost"])
                           for entry in record["decision"]["ranking"]
                           if entry["predicted_cost"] is not None]
                cost = dict(ranking)
                admissible = sorted(((o, c) for o, c in ranking if o in verdicts),
                                    key=lambda pair: (pair[1], pair[0]))
                row.update(ranking=ranking, selector_chosen=record["decision"]["chosen"]["ordinal"],
                           n_candidates=len(ranking), admissible=admissible,
                           measurable=bool(admissible), ceiling=bool(successes))
                within_angle, within_radius = {}, {}
                for i in range(len(ANGLES)):
                    group = [o for o in cost if design[o]["angle_index"] == i and o in launches]
                    within_angle[str(i)] = spearman([launches[o]["speed"] for o in group],
                                                    [cost[o] for o in group])
                for j in range(len(RADII)):
                    group = [o for o in cost if design[o]["radius_index"] == j and o in launches]
                    within_radius[str(j)] = spearman([launches[o]["angle_deg"] for o in group],
                                                     [cost[o] for o in group])
                row["spearman_within_angle_vs_speed"] = within_angle
                row["spearman_within_radius_vs_angle"] = within_radius
                row["spearman_ordinal"] = spearman([o for o, _ in ranking], [c for _, c in ranking])
                row["spearman_drag_x"] = spearman([design[o]["action"]["drag_x"] for o, _ in ranking],
                                                  [c for _, c in ranking])
                row["spearman_drag_y"] = spearman([design[o]["action"]["drag_y"] for o, _ in ranking],
                                                  [c for _, c in ranking])
                row["spearman_expected_speed"] = spearman(
                    [realization[o]["expected_speed"] for o, _ in ranking], [c for _, c in ranking])
                if admissible:
                    top1 = admissible[0][0]
                    n, k = len(admissible), sum(1 for o, _ in admissible if verdicts[o])
                    row.update(chosen=top1, chosen_cost=admissible[0][1],
                               chosen_action=design[top1]["action"],
                               chosen_realization=realization[top1],
                               chosen_realized_speed=(launches.get(top1) or {}).get("speed"),
                               chosen_realized_angle=(launches.get(top1) or {}).get("angle_deg"))
                    if successes:
                        row.update(top1_hit=bool(verdicts[top1]),
                                   top3_hit=any(verdicts[o] for o, _ in admissible[:3]),
                                   chance_top1=k / n,
                                   chance_top3=(1.0 - math.comb(n - k, 3) / math.comb(n, 3)
                                                if n >= 3 else 1.0))
                        pos = [c for o, c in admissible if verdicts[o]]
                        neg = [c for o, c in admissible if not verdicts[o]]
                        row["auc"] = cell_auc(pos, neg) if pos and neg else None
                rows.append(row)
    return rows


def c24_disposition(ready, top1):
    if not ready or top1["cells"] == 0:
        return STOP_TOKEN
    diff = top1["paired_difference_member_clustered"]
    if top1["rate"] < top1["chance_mean"] - RULE["c24_margin"] and diff["interval"][1] < 0:
        return "supported"
    if diff["mean"] >= 0:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def c25_disposition(ready, auc_m):
    if not ready or auc_m is None:
        return STOP_TOKEN
    if auc_m <= RULE["c25_supported_max_auc"]:
        return "supported"
    if auc_m >= RULE["c25_not_supported_min_auc"]:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def reading_row(c24, c25):
    if c24 == "supported" and c25 == "supported":
        return READING_MATRIX[0]
    if c24 == "supported" and c25 == "not_supported_by_this_experiment":
        return READING_MATRIX[1]
    if c24 == "not_supported_by_this_experiment":
        return READING_MATRIX[2]
    return READING_MATRIX[3]


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered), "median": median(ordered),
            "min": ordered[0], "p10": pick(0.10), "p90": pick(0.90), "max": ordered[-1]}


def launch_table(plan, oracle):
    design = {item["ordinal"]: item for item in plan["inventory"]["design"]}
    realization = {row["ordinal"]: row for row in plan["inventory"]["realization"]}
    per_radius = {}
    for j, radius in enumerate(RADII):
        records = [r for r in oracle.values() if r.get("launch")
                   and design[r["cell"]["ordinal"]]["radius_index"] == j]
        speeds = [r["launch"]["speed"] for r in records]
        expected = [realization[r["cell"]["ordinal"]]["expected_speed"] for r in records]
        angle_error = [r["launch"]["angle_deg"] - realization[r["cell"]["ordinal"]]["realized_angle_deg"]
                       for r in records]
        per_radius[str(radius)] = {
            "slots": len(records), "realized_speed": distribution(speeds),
            "realized_minus_expected_speed": distribution([s - e for s, e in zip(speeds, expected)]),
            "realized_minus_declared_angle_deg": distribution(angle_error)}
    inner = [per_radius[str(r)]["realized_speed"] for r in RADII if r != SATURATED_RADIUS]
    disjoint = all(a is not None and b is not None and a["max"] < b["min"] for a, b in zip(inner, inner[1:]))
    return {"per_radius_px": per_radius,
            "inner_radii_distinct_submaximal": bool(disjoint and inner[-1]["max"] < 10.0 - 1e-3),
            "rule": ("inner radii distinct iff the realized-speed ranges of 5, 9, 13, 17 px are "
                     "strictly increasing and disjoint and all below 10")}


def tables(plan, oracle, rows):
    cells = scheduled_cells(plan)
    slot_status, failures = Counter(), Counter()
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
    members = [state["member"] for state in plan["states"]]
    measurable = sorted({row["member"] for row in rows if row["measurable"]})
    ceiling = sorted({row["member"] for row in rows if row["measurable"] and row["ceiling"]})
    per_member = {}
    for member in members:
        sample = next(row for row in rows if row["member"] == member)
        per_member[member] = {"verdict_slots": sample["verdict_slots"], "successes": sample["successes"],
                              "measurable": member in measurable, "anchor_source": sample["anchor_source"]}
    ceiling_cells = [row for row in rows if row["measurable"] and row["ceiling"]]
    auc_cells = [row for row in ceiling_cells if row.get("auc") is not None]

    def auc_block(selected):
        return {"cells": len(selected), "cell_unit": bootstrap([row["auc"] for row in selected]),
                "member_clustered": bootstrap([row["auc"] for row in selected],
                                              [row["member"] for row in selected])}

    def rate_block(selected, hit, chance):
        hits = sum(1 for row in selected if row[hit])
        return {"cells": len(selected), "hits": hits,
                "rate": hits / len(selected) if selected else None,
                "chance_mean": sum(row[chance] for row in selected) / len(selected) if selected else None,
                "paired_difference_member_clustered": bootstrap(
                    [float(row[hit]) - row[chance] for row in selected],
                    [row["member"] for row in selected])}

    auc = auc_block(auc_cells)
    auc["per_system"] = {s: auc_block([r for r in auc_cells if r["system"] == s]) for s in MODEL_SYSTEMS}
    top1 = rate_block(ceiling_cells, "top1_hit", "chance_top1")
    top1["per_system"] = {s: rate_block([r for r in ceiling_cells if r["system"] == s],
                                        "top1_hit", "chance_top1") for s in MODEL_SYSTEMS}
    member_chance = {m: next(r for r in ceiling_cells if r["member"] == m)["chance_top1"] for m in ceiling}
    miss_all = math.prod(1.0 - v for v in member_chance.values()) if ceiling else None
    scored = [row for row in rows if row["measurable"]]
    design = {item["ordinal"]: item for item in plan["inventory"]["design"]}
    saturated = {o for o, item in design.items() if item["radius_nominal_px"] == SATURATED_RADIUS}
    structure = {
        "within_angle_rho_vs_realized_speed": {
            f"{ANGLES[i]:.0f}deg": {"median_rho": median([r["spearman_within_angle_vs_speed"][str(i)] for r in scored]),
                                    "per_system_median_rho": {s: median([r["spearman_within_angle_vs_speed"][str(i)]
                                                                         for r in scored if r["system"] == s])
                                                              for s in MODEL_SYSTEMS}}
            for i in range(len(ANGLES))},
        "within_radius_rho_vs_realized_angle": {
            f"{RADII[j]}px": {"median_rho": median([r["spearman_within_radius_vs_angle"][str(j)] for r in scored]),
                              "per_system_median_rho": {s: median([r["spearman_within_radius_vs_angle"][str(j)]
                                                                   for r in scored if r["system"] == s])
                                                        for s in MODEL_SYSTEMS}}
            for j in range(len(RADII))},
        "pooled_within_angle_median_rho": median([v for r in scored for v in r["spearman_within_angle_vs_speed"].values()]),
        "pooled_within_radius_median_rho": median([v for r in scored for v in r["spearman_within_radius_vs_angle"].values()]),
        "full_inventory_rho_vs_expected_speed": median([r["spearman_expected_speed"] for r in scored]),
        "full_inventory_rho_vs_ordinal": median([r["spearman_ordinal"] for r in scored]),
    }
    chosen_hist = Counter(row["selector_chosen"] for row in scored)
    success_hist = Counter(o for member in measurable for o in per_member[member]["successes"])
    admissible_top1 = [row["chosen"] for row in scored]
    saturated_share = {
        "selector_full_inventory": (sum(1 for row in scored if row["selector_chosen"] in saturated) / len(scored)
                                    if scored else None),
        "admissible_top1": (sum(1 for o in admissible_top1 if o in saturated) / len(admissible_top1)
                            if admissible_top1 else None),
        "uniform_reference": len(saturated) / len(design), "cells": len(scored)}
    ready = len(ceiling) >= RULE["ceiling_members_min"] and coverage >= RULE["verdict_coverage_min"]
    auc_m = auc["member_clustered"]["mean"] if auc["member_clustered"] else None
    c24 = c24_disposition(ready, top1)
    c25 = c25_disposition(ready, auc_m)
    return {
        "scheduled_slots": len(cells), "slot_status": dict(sorted(slot_status.items())),
        "typed_failures_by_kind": dict(sorted(failures.items())), "verdict_coverage": coverage,
        "members": per_member, "measurable_members": measurable, "ceiling_members": ceiling,
        "ceiling": {"value": len(ceiling) / len(measurable) if measurable else None,
                    "numerator": len(ceiling), "denominator": len(measurable),
                    "interval": bootstrap([float(m in ceiling) for m in measurable])},
        "top1": top1, "auc": auc,
        "top3": rate_block(ceiling_cells, "top3_hit", "chance_top3"),
        "miss_all_probability": miss_all, "structure": structure,
        "chosen_ordinal_histogram": {str(k): chosen_hist[k] for k in sorted(chosen_hist)},
        "success_ordinal_histogram": {str(k): success_hist[k] for k in sorted(success_hist)},
        "saturated_column_chosen_share": saturated_share,
        "readiness": {"ready": ready, "ceiling_members": len(ceiling), "verdict_coverage": coverage},
        "dispositions": {"C24": c24, "C25": c25},
        "disposition_inputs": {
            "C24": {"top1_rate": top1["rate"], "chance_mean": top1["chance_mean"],
                    "paired_difference": top1["paired_difference_member_clustered"]},
            "C25": {"auc_member_clustered": auc_m}},
    }


def compute_accounting(output, plan, oracle):
    output = Path(output)
    walls, engine_walls = [], []
    for cell in scheduled_cells(plan):
        receipt = output / "receipts" / f"{cell['identity']}.json"
        if receipt.is_file():
            walls.append(read_json(receipt)["wall_seconds"])
        record = oracle.get(cell["identity"])
        if record is not None and record.get("execution"):
            engine_walls.append(record["execution"]["engine_wall_seconds"])
    ledgers = {name: read_json(output / name) for name in
               ("ledger.json", "ledger-completion.json", "ledger-score.json") if (output / name).is_file()}
    speeds = [r["launch"]["speed"] for r in oracle.values() if r.get("launch")]
    return {
        "oracle_wall_seconds": ledgers["ledger.json"]["wall_seconds_elapsed"],
        "oracle_stop_reason": ledgers["ledger.json"]["stop_reason"],
        "completion_wall_seconds": (ledgers["ledger-completion.json"]["wall_seconds_elapsed"]
                                    if "ledger-completion.json" in ledgers else None),
        "score_gpu_seconds": ledgers["ledger-score.json"]["gpu_seconds_elapsed"],
        "per_slot_worker_wall_seconds": distribution(walls),
        "per_slot_engine_wall_seconds": distribution(engine_walls),
        "launch_speed_range": [min(speeds), max(speeds)] if speeds else None,
        "launch_speed": launch_table(plan, oracle),
        "caps": {"wall_seconds": WALL_CAP_SECONDS, "gpu_seconds": GPU_CAP_SECONDS, "workers": WORKERS},
        "caps_respected": (ledgers["ledger.json"]["wall_seconds_elapsed"] <= WALL_CAP_SECONDS
                           and ledgers["ledger-score.json"]["gpu_seconds_elapsed"] <= GPU_CAP_SECONDS),
    }


def compute_tables(output, plan):
    oracle = load_oracle_records(output, plan)
    decisions = load_decisions(output, plan)
    rows = decision_rows(plan, oracle, decisions)
    structure = {"members": len(plan["states"]), "slots": len(scheduled_cells(plan)),
                 "decision_cells": len(rows)}
    if structure != plan["expected_structure"]:
        raise ValueError(f"structural counts {structure} differ from {plan['expected_structure']}")
    if len(decisions) != structure["decision_cells"]:
        raise ValueError(f"{len(decisions)} decision records, expected {structure['decision_cells']}; run --score")
    if len(oracle) != structure["slots"]:
        raise ValueError(f"{len(oracle)} oracle records, expected {structure['slots']}; run --run")
    table = tables(plan, oracle, rows)
    frames = [r["frame_vs_retained_anchor"]["byte_equal"] for r in oracle.values()
              if r.get("frame_vs_retained_anchor")]
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY, "plan_identity": plan["identity"],
        "structure": structure, "arm": table,
        "dispositions": table["dispositions"],
        "reading_matrix_row": reading_row(table["dispositions"]["C24"], table["dispositions"]["C25"]),
        "stated_predictions": plan["stated_predictions"],
        "decision_frames_vs_retained_anchor": {"byte_equal": sum(frames), "compared": len(frames)},
        "compute": compute_accounting(output, plan, oracle),
        "rows": [{k: v for k, v in r.items() if k not in ("ranking", "admissible")} for r in rows],
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
                     "angle_deg", "radius_px", "realized_angle_deg", "expected_speed",
                     "realized_speed"])
    band = {int(k) for k in compute["arm"]["success_ordinal_histogram"]}

    def cell(value):
        return "" if value is None else value

    for row in compute["rows"]:
        chosen = row.get("chosen")
        action = row.get("chosen_action")
        realized = row.get("chosen_realization")
        writer.writerow([
            row["cell_identity"], row["state"], row["member"], row["family"],
            row["exposure_role"], row["system"], row["seed"], row["typed_failure"],
            cell(row.get("n_candidates")), row["verdict_slots"], row["n_successes"],
            "|".join(str(o) for o in row["successes"]), cell(chosen), cell(row.get("chosen_cost")),
            (chosen in band) if chosen is not None else "",
            cell(row.get("top1_hit")), cell(row.get("top3_hit")),
            cell(row.get("chance_top1")), cell(row.get("chance_top3")), cell(row.get("auc")),
            cell(row.get("spearman_ordinal")), cell(row.get("spearman_drag_x")),
            cell(row.get("spearman_drag_y")),
            chosen is not None, cell(row.get("top1_hit")) if chosen is not None else "",
            ARM, cell(action and action["drag_x"]), cell(action and action["drag_y"]),
            cell(realized and realized["realized_radius_px"]),
            cell(realized and realized["realized_angle_deg"]),
            cell(realized and realized["radius_nominal_px"]),
            cell(row.get("chosen_realized_angle")),
            cell(realized and realized["expected_speed"]),
            cell(row.get("chosen_realized_speed"))])
    return buffer.getvalue()


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "arm", "statistic", "unit", "value", "interval_low",
                     "interval_high", "interval_label", "detail"])

    def put(identity, statistic, unit, value, block=None, detail=""):
        writer.writerow([identity, ARM, statistic, unit, "" if value is None else value,
                         block["interval"][0] if block else "", block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", detail])

    table = compute["arm"]
    put("coverage", "verdict coverage", "slots", table["verdict_coverage"],
        detail=json.dumps(table["slot_status"], sort_keys=True))
    put("ceiling", "member ceiling", "members", table["ceiling"]["value"], table["ceiling"]["interval"],
        f"{table['ceiling']['numerator']} of {table['ceiling']['denominator']}")
    for key, label in (("top1", "pooled top-1 hit rate"), ("top3", "top-3 hit rate")):
        block = table[key]
        put(key, label, "cells", block["rate"],
            detail=f"{block['hits']}/{block['cells']}; chance {block['chance_mean']}")
        diff = block["paired_difference_member_clustered"]
        put(f"{key}_minus_chance", f"{key} minus chance", "member_clustered", diff and diff["mean"], diff)
    for system in MODEL_SYSTEMS:
        block = table["top1"]["per_system"][system]
        diff = block["paired_difference_member_clustered"]
        put(f"top1_{system}", "top-1 per system minus chance", "member_clustered", diff and diff["mean"],
            diff, f"{block['hits']}/{block['cells']}; chance {block['chance_mean']}")
    for unit in ("cell_unit", "member_clustered"):
        block = table["auc"][unit]
        put(f"auc_{unit}", "within-state AUC", unit, block and block["mean"], block,
            f"{table['auc']['cells']} cells")
    for system in MODEL_SYSTEMS:
        block = table["auc"]["per_system"][system]["member_clustered"]
        put(f"auc_{system}", "within-state AUC per system", "member_clustered", block and block["mean"], block)
    put("miss_all", "one uniform draw per ceiling member misses everywhere", "probability",
        table["miss_all_probability"])
    for key, block in table["structure"]["within_angle_rho_vs_realized_speed"].items():
        put(f"rho_speed_{key}", f"median Spearman rho cost vs realized speed at {key}", "cells",
            block["median_rho"], detail=json.dumps(block["per_system_median_rho"], sort_keys=True))
    for key, block in table["structure"]["within_radius_rho_vs_realized_angle"].items():
        put(f"rho_angle_{key}", f"median Spearman rho cost vs realized angle at {key}", "cells",
            block["median_rho"], detail=json.dumps(block["per_system_median_rho"], sort_keys=True))
    share = table["saturated_column_chosen_share"]
    put("saturated_chosen_share", "share of selector choices at the saturated column", "cells",
        share["selector_full_inventory"], detail=json.dumps(share, sort_keys=True))
    for radius, block in compute["compute"]["launch_speed"]["per_radius_px"].items():
        speed = block["realized_speed"]
        put(f"launch_speed_{radius}px", f"realized launch speed at {radius} px", "slots",
            speed and speed["median"], detail=json.dumps(speed, sort_keys=True))
    for claim in ("C24", "C25"):
        put(f"{claim}_disposition", f"{claim} disposition", "token", compute["dispositions"][claim],
            detail=json.dumps(table["disposition_inputs"][claim], sort_keys=True))
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    table = compute["arm"]
    comp = compute["compute"]
    add("# Issue-94: launch-power x angle inventory inside the 18.49 px drag clamp — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan `{SCHEMA_PLAN}` v{plan['version']} (terminal), frozen "
        f"{plan['frozen_at']} before any engine slot")
    add(f"- validation command: `{VALIDATION_COMMAND}`")
    add(f"- action-bound authority (WP0): {plan['inventory']['action_bound_authority']}")
    add(f"- training support (WP0): {plan['inventory']['training_support']}")
    add("- every interval is DESCRIPTIVE (10000 draws, seed 7201, quantiles 0.025/0.975)")
    add("")
    add("## Dispositions (frozen rules) vs stated predictions")
    add("")
    add("| claim | outcome | stated prediction | rationale |")
    add("|---|---|---|---|")
    for claim in ("C24", "C25"):
        prediction = plan["stated_predictions"][claim]
        add(f"| {claim} | **{compute['dispositions'][claim]}** | {prediction['expected_token']} | "
            f"{prediction['rationale']} |")
    row = compute["reading_matrix_row"]
    add("")
    add(f"- reading-matrix row {row['row']} (C24 {row['C24']}, C25 {row['C25']}): \"{row['reading']}\"")
    add(f"- readiness: {json.dumps(table['readiness'], sort_keys=True)}")
    add("")
    add("## Estimands")
    add("")
    add(f"- slots: {table['scheduled_slots']} scheduled; {json.dumps(table['slot_status'], sort_keys=True)}; "
        f"verdict coverage **{fmt(table['verdict_coverage'])}**; typed failures "
        f"{json.dumps(table['typed_failures_by_kind'], sort_keys=True)}")
    add(f"- 1. ceiling: **{table['ceiling']['numerator']}/{table['ceiling']['denominator']}** measurable "
        f"members = {fmt(table['ceiling']['value'])} (interval {fmt_interval(table['ceiling']['interval'])}); "
        f"ceiling members {table['ceiling_members']}")
    top1 = table["top1"]
    add(f"- 2. **pooled top-1**: {top1['hits']}/{top1['cells']} = {fmt(top1['rate'])} vs chance "
        f"{fmt(top1['chance_mean'])}; member-clustered paired difference "
        f"**{fmt_interval(top1['paired_difference_member_clustered'])}**")
    for system in MODEL_SYSTEMS:
        block = top1["per_system"][system]
        add(f"  - {system}: {block['hits']}/{block['cells']} = {fmt(block['rate'])} vs chance "
            f"{fmt(block['chance_mean'])}; paired difference "
            f"{fmt_interval(block['paired_difference_member_clustered'])}")
    add(f"- 3. **within-state AUC**: cell-unit {fmt_interval(table['auc']['cell_unit'])} over "
        f"{table['auc']['cells']} cells; member-clustered AUC_m "
        f"**{fmt_interval(table['auc']['member_clustered'])}** over "
        f"{(table['auc']['member_clustered'] or {}).get('clusters', 0)} members")
    for system in MODEL_SYSTEMS:
        block = table["auc"]["per_system"][system]
        add(f"  - {system}: member-clustered {fmt_interval(block['member_clustered'])} ({block['cells']} cells)")
    top3 = table["top3"]
    add(f"- 4. top-3: {top3['hits']}/{top3['cells']} = {fmt(top3['rate'])} vs chance "
        f"{fmt(top3['chance_mean'])}; paired difference "
        f"{fmt_interval(top3['paired_difference_member_clustered'])}; miss-all probability "
        f"{fmt(table['miss_all_probability'])}")
    structure = table["structure"]
    add("- 5. engine-side structure (DESCRIPTIVE, median Spearman rho of predicted cost):")
    add("  - within fixed angle vs realized launch speed: " + "; ".join(
        f"{k} {fmt(v['median_rho'])}" for k, v in structure["within_angle_rho_vs_realized_speed"].items())
        + f" (pooled {fmt(structure['pooled_within_angle_median_rho'])})")
    add("  - within fixed radius vs realized launch angle: " + "; ".join(
        f"{k} {fmt(v['median_rho'])}" for k, v in structure["within_radius_rho_vs_realized_angle"].items())
        + f" (pooled {fmt(structure['pooled_within_radius_median_rho'])})")
    add(f"  - full inventory vs expected speed {fmt(structure['full_inventory_rho_vs_expected_speed'])}; "
        f"vs ordinal {fmt(structure['full_inventory_rho_vs_ordinal'])}")
    add(f"  - chosen-ordinal histogram (selector over full inventory): {json.dumps(table['chosen_ordinal_histogram'])}")
    add(f"  - success-ordinal histogram (members): {json.dumps(table['success_ordinal_histogram'])}")
    add(f"  - saturated-column chosen share: {json.dumps(table['saturated_column_chosen_share'], sort_keys=True)}")
    speeds = comp["launch_speed"]
    add(f"- 6. realized launch speed per radius column (engine telemetry); inner radii distinct "
        f"sub-maximal: **{speeds['inner_radii_distinct_submaximal']}**")
    for radius, block in speeds["per_radius_px"].items():
        s = block["realized_speed"]
        d = block["realized_minus_expected_speed"]
        a = block["realized_minus_declared_angle_deg"]
        add(f"  - {radius} px: {block['slots']} slots, speed {fmt(s and s['min'])}–{fmt(s and s['max'])} "
            f"(median {fmt(s and s['median'])}); realized − expected median {fmt(d and d['median'])}; "
            f"angle offset {fmt(a and a['min'], 2)}…{fmt(a and a['max'], 2)} deg")
    add("")
    add("## Execution and compute")
    add("")
    frames = compute["decision_frames_vs_retained_anchor"]
    add(f"- decision frames byte-identical to the member's retained anchor: "
        f"{frames['byte_equal']}/{frames['compared']}")
    add(f"- oracle wall {fmt(comp['oracle_wall_seconds'] / 3600, 2)} h (stop reason "
        f"{comp['oracle_stop_reason']}); completion pass wall {comp['completion_wall_seconds']}; "
        f"ranker scoring GPU {fmt(comp['score_gpu_seconds'], 1)} s; caps respected {comp['caps_respected']}")
    add(f"- per-slot worker wall: {json.dumps(comp['per_slot_worker_wall_seconds'], sort_keys=True)}")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ". No state-difficulty claim is drawn from any row. "
        "`bound_issue87_only` / `first_shot_success_issue87_only` in slot_join.csv keep #92's "
        "column names and bind the chosen admissible candidate to this experiment's own oracle "
        "table.")
    add("")
    return "\n".join(lines)


def build_report(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "plan_identity": plan["identity"],
            "plan_version": plan["version"], "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "claims": CLAIMS,
            "dispositions": compute["dispositions"],
            "stated_predictions": plan["stated_predictions"],
            "reading_matrix_row": compute["reading_matrix_row"],
            "arm": compute["arm"], "structure": compute["structure"],
            "decision_frames_vs_retained_anchor": compute["decision_frames_vs_retained_anchor"],
            "compute": compute["compute"],
            "action_bound_authority": plan["inventory"]["action_bound_authority"],
            "training_support": plan["inventory"]["training_support"],
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False}


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
    log(f"published: C24 {compute['dispositions']['C24']}, C25 {compute['dispositions']['C25']}; "
        f"reading row {compute['reading_matrix_row']['row']}")
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
        f"comparisons.csv byte-compared; C24 {fresh['dispositions']['C24']}, C25 "
        f"{fresh['dispositions']['C25']}, reading row {fresh['reading_matrix_row']['row']} "
        f"({time.monotonic() - began:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# smoke (WP1), run, completion
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    plan = load_plan(output)
    state = next(s for s in plan["states"] if s["member"] == SMOKE_MEMBER)
    items = {item["ordinal"]: item for item in plan["inventory"]["members"][SMOKE_MEMBER]}
    ordinals = [5 * SMOKE_ANGLE_INDEX + j for j in SMOKE_RADIUS_INDICES]
    cells = [make_cell(state, items[o], "smoke") for o in ordinals]
    plan87 = read_json(EIGHTY_SEVEN / "plan.json")
    meta = {"identity": IDENTITY, "version": 0, "player_root": plan["player_root"]}
    supervise(output / "smoke", DATA / "smoke" / "attempts", meta, cells, plan87["states"], "smoke")
    realization = {row["ordinal"]: row for row in plan["inventory"]["realization"]}
    checks = []
    for cell in cells:
        record = read_json(output / "smoke" / "records" / f"{cell['identity']}.json")
        launch = record.get("launch") or {}
        expected = realization[cell["ordinal"]]
        checks.append({
            "cell": cell["identity"], "ordinal": cell["ordinal"],
            "radius_nominal_px": expected["radius_nominal_px"], "failure": record["failure"],
            "launched_once": record.get("launch") is not None,
            "frame_captured": record["decision_frame"] is not None,
            "frame_vs_retained_anchor": record["frame_vs_retained_anchor"],
            "executed_action_equals_declared": record["executed_action"] is not None and record["failure"] is None,
            "expected_speed": expected["expected_speed"], "realized_speed": launch.get("speed"),
            "expected_angle_deg": expected["realized_angle_deg"], "realized_angle_deg": launch.get("angle_deg"),
            "wall_seconds": record.get("wall_seconds")})
    speeds = [c["realized_speed"] for c in checks]
    infra = all(c["launched_once"] and c["frame_captured"] and c["executed_action_equals_declared"]
                for c in checks)
    distinct = (infra and all(b - a >= SMOKE_MIN_SPEED_GAP for a, b in zip(speeds, speeds[1:]))
                and abs(speeds[-1] - 10.0) < 0.01)
    # scoring infrastructure: every candidate receives a finite cost (costs not recorded)
    wlc, adapter, objective = load_scoring_stack()
    finite = []
    models = {}
    for system in MODEL_SYSTEMS:
        decision, _ = score_inventory(wlc, adapter, objective, models, plan, state["anchor"],
                                      system, SEEDS[0], plan["inventory"]["members"][SMOKE_MEMBER])
        finite.append({"system": system, "seed": SEEDS[0],
                       "finite_costs": sum(1 for r in decision["ranking"] if r["predicted_cost"] is not None),
                       "candidates": len(decision["ranking"])})
    scoring_ok = all(f["finite_costs"] == f["candidates"] == len(items) for f in finite)
    ok = infra and distinct and scoring_ok
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(),
                "plan_frozen_at": plan["frozen_at"], "cells": checks,
                "speeds_distinct_as_predicted": distinct, "infrastructure_ok": infra,
                "scoring_finite": finite, "ok": ok,
                "note": ("infrastructure assertions only; smoke verdicts and costs are never "
                         "joined to scoring")}
    write_json(output / "smoke" / "smoke.json", evidence)
    log(f"smoke {'PASSED' if ok else 'FAILED'}: " + "; ".join(
        f"r={c['radius_nominal_px']} expected {fmt(c['expected_speed'])} realized "
        f"{fmt(c['realized_speed'])} angle {fmt(c['realized_angle_deg'], 2)} (expected "
        f"{fmt(c['expected_angle_deg'], 2)}) failure={c['failure']}" for c in checks)
        + f"; scoring {finite}")
    if not ok:
        raise ValueError("STOP: smoke failed (see smoke/smoke.json); comment on #94")
    return 0


def run(output):
    output = Path(output)
    plan = load_plan(output)
    smoke_path = output / "smoke" / "smoke.json"
    if not smoke_path.is_file() or not read_json(smoke_path)["ok"]:
        raise ValueError("smoke evidence missing or failed; run --smoke first")
    plan87 = read_json(EIGHTY_SEVEN / "plan.json")
    supervise(output, DATA / "attempts", plan, scheduled_cells(plan), plan87["states"], "run")
    return 0


def completion(output):
    output = Path(output)
    plan = load_plan(output)
    ledger = read_json(output / "ledger.json")
    if ledger["status"] in ("interrupted", "running"):
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


MODES = {"prepare": (prepare, False), "smoke": (smoke, True), "run": (run, True),
         "completion": (completion, True), "score": (score, True),
         "publish": (publish, False), "validate": (validate, False)}


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
