"""Issue-83 ADD-EXP: CLEVRER h15 recursive-error non-monotonicity decomposition.

Post-#79 probe (analysis-scale). #79's Q1 verdict failed only on component S3:
at h=15 the recursive position-MSE curve is non-monotone over the frozen
endpoint grid while h=1 and h=5 grow monotonically. Two interpretations
survive: (i) regime property - post-settlement stillness dips mid-window error
while long-horizon integration blows up late; (ii) estimand/endpoint artifact.
This probe decomposes the frozen #79 membership by contact-activity phase and
tests endpoint/mask sensitivity, with no new training and no amendment of any
#79 verdict.

Trace cells (all scheduled; mutually exclusive modes below):
- window cells: 480 frozen #79 training windows (12 scenes x starts 0..39) x
  3 seeds x 2 arms x 3 horizons = 8640 recursive per-frame traces bounded by
  the 61-frame window timeline (visited relative frames k*h, k*h <= 60);
- grid cells: 48 #79 eval units (12 scenes x contexts 0..3) x 3 seeds x 2 arms
  x 3 horizons = 864 recursive per-frame traces over the S3-critical span
  (visited relative frames k*h, k*h <= 120), scoring the finer endpoint grid
  {15,30,60,90,120} and the #79 anchor points {15,30,60,120}.

Segmentation is data-frozen before any error is computed from the scenes'
frozen collision timelines; the post-settlement vs pre/active error-ratio
contrast carries a frozen uniformity band; sensitivity variants are new
descriptive estimands. The hybrid arm runs its continuous-abstraction pair
(the issue-79 hybrid_continuous systems); the micro symbolic-execution pair
stays with #79's published Q2 and is out of scope here.

Modes (mutually exclusive): --dry-run (no writes), --smoke-test (bounded real
pass with the frozen seed-20260908 checkpoints, CPU), --prepare (freeze the
plan from the #79 artifacts; no GPU), --run-evaluation (GPU lock held for the
whole phase), --publish, --validate.
Exact validation command: python -u -m scripts.run_clevrer_nonmonotonicity_probe --validate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import torch

from scripts import run_clevrer_boundary_replication as source79
from world_model.model import Abstraction, PredictionPair
from world_model.training import clevrer_boundary as clevrer
from world_model.training import clevrer_nonmonotonicity as probe
from world_model.training.cnn_hybrid import DIM, linear_macs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-83-clevrer-nonmonotonicity-v1"
SOURCE_DIR = ROOT / ".local-artifacts/issue-79-clevrer-boundary-v1"
SOURCE_SCHEMA = "issue_79_clevrer_boundary_v1"
SOURCE_IDENTITY = "issue-79-clevrer-boundary-v1"
SOURCE_REVISION = "ac2a4e1f875c26c6e7c092dd3616db42e14dcd04"
VALIDATION_ZIP = ROOT / "data/clevrer/annotations/annotation_validation.zip"
ZIP_FACTS = source79.ZIP_FACTS["validation"]
RAW_MEMBER = "annotation_10000-11000/annotation_{}.json"
SEEDS = source79.SEEDS
ARMS = ("continuous", "hybrid")
HORIZONS = clevrer.HORIZONS
WINDOW_STARTS = tuple(clevrer.WINDOW_STARTS)
GRID_CONTEXTS = tuple(clevrer.EVAL_CONTEXTS)
ARM_PAIR_NOTE = ("arm hybrid executes the hybrid checkpoint through its continuous-abstraction pair "
                 "(issue-79 hybrid_continuous systems); the micro symbolic-execution pair is out of scope")
WINDOW_KIND, GRID_KIND = "window", "grid"
TRACE_LAST = {WINDOW_KIND: probe.WINDOW_TRACE_LAST, GRID_KIND: probe.GRID_TRACE_LAST}
SCHEMA = "issue_83_clevrer_nonmonotonicity_v1"
IDENTITY = "issue-83-clevrer-nonmonotonicity-v1"
FILES = ("scripts/run_clevrer_nonmonotonicity_probe.py", "world_model/training/clevrer_nonmonotonicity.py")
GPU_LOCK = Path("/tmp/novphy-addexp-gpu.lock")
GPU_HOURS_ALLOWANCE = 0.5
ARTIFACTS_CAP_GIB = 1
STOP = "stop-rule.json"
DISPOSITION_TOKENS = ("supported", "not_supported_by_this_experiment", "readiness_or_precision_insufficient")
_ANNOTATION_CACHE = {}


def log(message):
    print(f"[issue-83] {message}", flush=True)


read = source79.read
write = source79.write
atomic_torch = source79.atomic_torch
memory = source79.memory
gpu_lock = source79.gpu_lock
load_frozen_predictor = source79.load_predictor


# ---------------------------------------------------------------- schedule


def scheduled_units(plan, kind):
    """Frozen (scene_index, start-or-context) roster for one trace kind."""
    scenes = [record["scene_index"] for record in plan["membership"]["scenes"]]
    starts = WINDOW_STARTS if kind == WINDOW_KIND else GRID_CONTEXTS
    return [(scene, start) for scene in scenes for start in starts]


def cell_name(kind, scene_index, start):
    return f"scene-{scene_index}-" + (f"w{start}" if kind == WINDOW_KIND else f"t{start}")


def cell_path(args, kind, seed, arm, horizon, scene_index, start):
    return (args.output / "traces" / kind / f"seed-{seed}" / f"{arm}_h{horizon}" /
            (cell_name(kind, scene_index, start) + ".pt"))


def scheduled_cells(plan):
    """Every scheduled (kind, seed, arm, horizon, scene, start) cell, frozen order."""
    return [(kind, seed, arm, horizon, scene, start)
            for kind in (WINDOW_KIND, GRID_KIND)
            for seed in plan["design"]["seeds"]
            for arm in plan["design"]["arms"]
            for horizon in plan["design"]["horizons"]
            for scene, start in scheduled_units(plan, kind)]


def pair_for(arm, horizon):
    return PredictionPair(horizon, Abstraction.CONTINUOUS)


def probe_binding(plan, seed, arm, horizon, kind):
    return {"plan_identity": plan["identity"], "source_plan_identity": plan["source"]["identity"],
            "source_plan_revision": plan["source"]["revision"],
            "representation_identity": plan["representation"]["identity"],
            "velocity_scale": plan["representation"]["velocity_scale"],
            "speed_tier_edges": plan["representation"]["speed_tier_edges"],
            "membership_scenes": [record["scene_index"] for record in plan["membership"]["scenes"]],
            "training": {"steps": plan["source"]["training_steps"], "seeds": plan["design"]["seeds"]},
            "seed": seed, "arm": arm, "horizon": horizon, "unit_kind": kind}


def check_binding(record, expected):
    if record["binding"] != expected:
        raise ValueError("trace record source/representation/binding differs from the frozen plan")


# ---------------------------------------------------------------- plan


def check_trace_grids():
    """Frozen divisibility and clip-bound rule for every (kind, endpoint, horizon)."""
    for horizon in HORIZONS:
        for kind, last in TRACE_LAST.items():
            if last % horizon:
                raise ValueError(f"trace span {last} of kind {kind} not divisible by horizon {horizon}")
        for grid in (probe.GRID_ENDPOINTS, probe.BROAD_ENDPOINTS):
            for endpoint in grid:
                if endpoint % horizon:
                    raise ValueError(f"endpoint {endpoint} not divisible by horizon {horizon}")
                if endpoint > probe.GRID_TRACE_LAST:
                    raise ValueError(f"endpoint {endpoint} outside the {probe.GRID_TRACE_LAST}-frame trace span")
    for context in GRID_CONTEXTS:
        if context + probe.GRID_TRACE_LAST > clevrer.FRAMES_PER_CLIP - 1:
            raise ValueError(f"context {context} + trace span {probe.GRID_TRACE_LAST} leaves the clip")
    for start in WINDOW_STARTS:
        if start + probe.WINDOW_TRACE_LAST > clevrer.FRAMES_PER_CLIP - 1:
            raise ValueError(f"window start {start} + trace span {probe.WINDOW_TRACE_LAST} leaves the clip")


def source_facts():
    """Read-only identity checks against the published #79 plan (inputs, never rewritten)."""
    plan = read(SOURCE_DIR / "plan.json")
    if plan["schema"] != SOURCE_SCHEMA or plan["identity"] != SOURCE_IDENTITY \
            or plan["source_revision"] != SOURCE_REVISION:
        raise ValueError("issue-79 frozen plan identity/revision differs from the bound source release")
    return plan


def source_shard(scene_index):
    shard = torch.load(SOURCE_DIR / "shards" / f"scene-{scene_index}.pt",
                       map_location="cpu", weights_only=True)
    if shard["schema"] != "issue_79_clevrer_scene_v1" or shard["identity"] != clevrer.IDENTITY:
        raise ValueError(f"scene shard {scene_index} schema/identity differs from the issue-79 release")
    return shard


def annotation_payload(scene_index):
    """One validation annotation dict, read once from the verified release archive."""
    if scene_index not in _ANNOTATION_CACHE:
        with zipfile.ZipFile(VALIDATION_ZIP) as archive:
            _ANNOTATION_CACHE[scene_index] = json.loads(archive.read(RAW_MEMBER.format(scene_index)))
    return _ANNOTATION_CACHE[scene_index]


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_zip():
    if not VALIDATION_ZIP.exists():
        raise ValueError(f"missing validation annotation archive at {VALIDATION_ZIP}")
    if VALIDATION_ZIP.stat().st_size != ZIP_FACTS["bytes"] or digest(VALIDATION_ZIP) != ZIP_FACTS["sha256"]:
        raise ValueError("validation annotation archive does not match the verified release digest")


def raw_targets(scene_index, velocity_scale):
    """Raw annotation position/velocity targets (no visibility mask) for one scene."""
    scene = clevrer.parse_scene(annotation_payload(scene_index))
    if scene.scene_index != scene_index:
        raise ValueError(f"annotation member {scene_index} carries scene_index {scene.scene_index}")
    count = len(scene.objects)
    if count > clevrer.SLOTS:
        raise ValueError(f"scene {scene_index}: {count} objects exceed the slot contract")
    raw_position = torch.zeros((clevrer.FRAMES_PER_CLIP, clevrer.SLOTS, 2))
    raw_velocity = torch.zeros((clevrer.FRAMES_PER_CLIP, clevrer.SLOTS, 2))
    declared = torch.zeros((clevrer.FRAMES_PER_CLIP, clevrer.SLOTS), dtype=torch.bool)
    raw_position[:, :count, 0] = scene.location[:, :count, 0] / clevrer.POS_SCALE
    raw_position[:, :count, 1] = scene.location[:, :count, 1] / clevrer.POS_SCALE
    raw_velocity[:, :count, 0] = scene.velocity[:, :count, 0] / velocity_scale
    raw_velocity[:, :count, 1] = scene.velocity[:, :count, 1] / velocity_scale
    declared[:, :count] = True
    return {"scene_index": scene_index, "objects": count, "declared": declared,
            "raw_position": raw_position, "raw_velocity": raw_velocity,
            "collision_frames": sorted(int(f) for f in scene.collision_frames.tolist())}


def make_plan(args):
    """Frozen Phase-0 plan: real numbers only; validated before any error is computed."""
    check_trace_grids()
    source = source_facts()
    verify_zip()
    membership = []
    bounds_by_scene = {}
    for record in source["membership"]["scenes"]:
        scene_index = record["scene_index"]
        shard = source_shard(scene_index)
        frames = sorted(int(f) for f in shard["collision_frames"].tolist())
        if len(frames) != record["collisions"]:
            raise ValueError(f"scene {scene_index}: shard collision count differs from the issue-79 plan")
        if shard["velocity_scale"] != source["representation"]["velocity_scale"] \
                or shard["speed_tier_edges"] != source["representation"]["speed_tier_edges"]:
            raise ValueError(f"scene shard {scene_index} representation facts differ from the issue-79 plan")
        if tuple(shard["clip"].shape) != (clevrer.FRAMES_PER_CLIP, DIM) \
                or tuple(shard["windows"]["z"].shape) != (clevrer.WINDOWS_PER_SCENE, clevrer.WINDOW_LENGTH, DIM):
            raise ValueError(f"scene shard {scene_index} clip/window geometry differs from the frozen contract")
        raw = raw_targets(scene_index, source["representation"]["velocity_scale"])
        if raw["collision_frames"] != frames:
            raise ValueError(f"scene {scene_index}: annotation collisions differ from the issue-79 shard")
        bounds = probe.cascade_bounds(shard["collision_frames"])
        bounds_by_scene[scene_index] = bounds
        membership.append({"scene_index": scene_index, "objects": record["objects"],
                           "collisions": record["collisions"], "collision_frames": frames,
                           "cascade_bounds": list(bounds)})
    grid_units = scheduled_units({"membership": {"scenes": membership}}, GRID_KIND)
    window_units = scheduled_units({"membership": {"scenes": membership}}, WINDOW_KIND)
    grid_census = [probe.segment_census(bounds_by_scene, grid_units, endpoint)
                   for endpoint in probe.GRID_ENDPOINTS]
    window_census = probe.segment_census(bounds_by_scene, window_units, probe.WINDOW_TRACE_LAST)
    window_cells = len(window_units) * len(SEEDS) * len(ARMS) * len(HORIZONS)
    grid_cells = len(grid_units) * len(SEEDS) * len(ARMS) * len(HORIZONS)
    return {
        "schema": SCHEMA, "identity": IDENTITY,
        "category": "ADD-EXP CLEVRER h15 recursive-error non-monotonicity decomposition (issue #83)",
        "source": {"identity": SOURCE_IDENTITY, "schema": SOURCE_SCHEMA, "revision": SOURCE_REVISION,
                   "artifacts": str(SOURCE_DIR), "training_steps": source["design"]["training"]["steps"],
                   "checkpoints": {f"{seed}/{arm}": str(SOURCE_DIR / f"seed-{seed}" / arm / "predictor.pt")
                                   for seed in SEEDS for arm in ARMS},
                   "rollout_records": "1296 published issue-79 eval cells, 0 typed failures (read-only inputs)",
                   "note": ("frozen checkpoints are re-loaded, never retrained; #79 verdicts are never recomputed "
                            "or amended")},
        "data": {"validation_zip": {"path": str(VALIDATION_ZIP.resolve()), **ZIP_FACTS},
                 "raw_member_pattern": RAW_MEMBER,
                 "raw_targets_use": ("all_frames sensitivity variant only; the masked variant uses the frozen "
                                     "issue-79 carriers")},
        "representation": {"identity": clevrer.IDENTITY,
                           "velocity_scale": source["representation"]["velocity_scale"],
                           "speed_tier_edges": source["representation"]["speed_tier_edges"],
                           "availability_mask": source["representation"]["availability_mask"],
                           "masked_variant": "frozen inside_camera_view mask, identical to issue-79 field_errors",
                           "all_frames_variant": ("no visibility mask: mean squared error over the declared objects' "
                                                  "position/velocity columns against raw annotation values in "
                                                  "carrier units, fixed denominator declared_objects x 2 per frame")},
        "membership": {"scenes": membership, "windows_per_scene": clevrer.WINDOWS_PER_SCENE,
                       "window_starts": list(WINDOW_STARTS), "grid_contexts": list(GRID_CONTEXTS),
                       "window_units": len(window_units), "grid_units": len(grid_units)},
        "segmentation": {
            "rule": ("frame-level labels from each scene's frozen collision timeline, frozen before any error is "
                     "computed: pre_first_collision before the scene's first collision event; active_cascade from "
                     "the first through the last event (objects in transit between contact events remain inside "
                     "the active cascade); post_settlement after the scene's last event. Window and context "
                     "boundaries only select which frames are scored; labels depend on scene timelines alone."),
            "cascade_bounds": {str(scene): list(bounds) for scene, bounds in bounds_by_scene.items()},
            "grid_census": grid_census,
            "window_census_at_relative_60": window_census,
            "census_note": ("strata sizes are outcome-free data facts published at freeze; the frozen precision "
                            "minimum is 8 units per stratum")},
        "design": {
            "arms": {"continuous": "ContinuousDynamics at the frozen issue-79 width 456, continuous-abstraction pair",
                     "hybrid": "CNNHybridPredictor checkpoint, continuous-abstraction pair"},
            "arm_pair_note": ARM_PAIR_NOTE,
            "seeds": list(SEEDS), "horizons": list(HORIZONS),
            "trace_kinds": {WINDOW_KIND: {"units": len(window_units), "timeline_frames": clevrer.WINDOW_LENGTH,
                                          "last_relative_frame": probe.WINDOW_TRACE_LAST,
                                          "visited_frames_rule": "relative frames k*h for k*h <= 60"},
                            GRID_KIND: {"units": len(grid_units), "last_relative_frame": probe.GRID_TRACE_LAST,
                                        "visited_frames_rule": "relative frames k*h for k*h <= 120"}},
            "cells": {"window": window_cells, "grid": grid_cells, "total": window_cells + grid_cells},
            "rollout": ("recursive fixed-pair rollout from the true carrier at the window start / eval context; one "
                        "shared initial carrier per cell, predictions feed the next transition, no truth resets; "
                        "per-visited-frame errors against the true carrier at the scored frame"),
            "estimands": {
                "E1": ("segment-conditional recursive error growth curves: mean masked position_mse and masked "
                       "velocity_mse per visited relative frame restricted to cells whose scored frame carries the "
                       "segment label, per h in {1,5,15}, per arm, per seed, plus the mean over seeds; unit counts "
                       "reported per point"),
                "E2": ("localization contrast: post_settlement vs pre/active (union) mean masked position_mse at "
                       "each scored endpoint; the statistic is the ratio post/pre_active of mean-over-seeds "
                       "per-unit errors, with a paired bootstrap interval over the joint unit roster (10000 draws, "
                       "rng 7201); reported for grid endpoints {15,30,60,90,120} and window relative endpoints "
                       "{15,30,60}"),
                "E3": ("sensitivity tables: recursive position_mse curves over the finer endpoint grid "
                       "{15,30,60,90,120} and over the issue-79 anchor points {15,30,60,120}, each under the "
                       "frozen inside_camera_view mask and the all_frames variant, per arm and horizon, with the "
                       "strict-increase monotonicity predicate across consecutive endpoints"),
                "dropped": ("no new training, no planning/ranking estimand, no perception claim; the micro "
                            "symbolic-execution pair is not an estimand here (issue-79 Q2 covers it)"),
                "anchor_note": ("anchor-grid, frozen-mask, continuous-arm curves recomputed here form a new "
                                "descriptive consistency table against the published issue-79 numbers; the #79 "
                                "verdict itself is never recomputed or amended"),
            },
            "metrics": ["position_mse_masked", "velocity_mse_masked", "carrier_mse",
                        "position_mse_all_frames", "velocity_mse_all_frames"],
            "grids": {"fine": list(probe.GRID_ENDPOINTS), "anchor": list(probe.BROAD_ENDPOINTS)},
            "masks": ["inside_camera_view", "all_frames"],
        },
        "decision_rules": {
            "tokens": list(DISPOSITION_TOKENS),
            "q1_localization": {
                "question": ("is the h=15 error dip at e=60 and blow-up at e=120 concentrated in post-settlement "
                             "segments, or uniform across pre-cascade / active-cascade / post-settlement segments"),
                "primary_statistic": ("grid rho(e=60, h=15, continuous arm): post_settlement vs pre/active mean "
                                      "masked position_mse ratio at scored frame context+60, mean over seeds per unit"),
                "practical_margin": ("uniformity band [0.80, 1.25] on rho, frozen before outcome computation; "
                                     "rho < 0.80 means the dip is concentrated in post-settlement segments"),
                "disposition_rule": ("supported iff rho < 0.80 with at least 8 units per stratum and a reliable "
                                     "interval; not_supported_by_this_experiment when rho >= 0.80 (uniform band or "
                                     "anti-concentration, sign reported); readiness_or_precision_insufficient when a "
                                     "stratum is below 8 units, the interval is unreliable, or any cell is "
                                     "missing or typed-failed"),
                "structural_note": ("at e=120 every unit is post-settlement (frozen grid census), so the e=120 "
                                    "blow-up is by construction measured on post-settlement frames; the decision "
                                    "endpoint is e=60 and the blow-up span is covered by the E1 segment curves and "
                                    "the e=90 contrast, reported descriptively"),
            },
            "q2_artifact_sensitivity": {
                "question": ("does the S3 non-monotone verdict change under the finer endpoint grid "
                             "{15,30,60,90,120} or under the all_frames availability-mask variant"),
                "primary_variants": ("continuous arm, h=15: {anchor, fine} grids x {inside_camera_view, all_frames} "
                                     "masks (4 curves), mean over seeds over units"),
                "monotonicity_predicate": ("strict increase across consecutive endpoints of the variant curve "
                                           "(issue-79 S3 convention); non-monotone = predicate false"),
                "disposition_rule": ("supported iff at least one primary variant curve is monotone (the non-monotone "
                                     "verdict changes: estimand/endpoint artifact supported); "
                                     "not_supported_by_this_experiment iff all four stay non-monotone; "
                                     "readiness_or_precision_insufficient when a predicate is undecidable or any "
                                     "cell is missing or typed-failed"),
            },
            "incomplete_rule": ("any missing or typed-failure cell -> both questions "
                                "readiness_or_precision_insufficient"),
        },
        "intervals": ("paired bootstrap over paired units, 10000 resamples, rng 7201; DESCRIPTIVE only, no "
                      "inferential superiority claim; zero-shot and adapted conditions do not exist here"),
        "compute": {
            "allowance_gpu_hours": GPU_HOURS_ALLOWANCE,
            "measured_phase": "--run-evaluation only (re-rollouts); prepare/publish/validate are CPU phases",
            "gpu_lock": f"exclusive fcntl.flock on {GPU_LOCK} held for the whole --run-evaluation phase",
            "stop_rule": ("if measured phase wall reaches 0.5 GPU-hours, stop executing groups, retain every "
                          "remaining scheduled cell as a typed failure compute_allowance_exceeded, and dispose "
                          "readiness_or_precision_insufficient"),
            "derived_artifacts_cap_gib": ARTIFACTS_CAP_GIB,
            "accounting": ["per-group rollout wall seconds", "recursive transition step counts per arm and horizon",
                           "declared linear MACs per transition (not FLOPs)", "active parameter counts per arm",
                           "peak CUDA and CPU RSS memory"],
            "memory_bound": ("batched bounded-memory rollouts: one batch of 480 (window) or 48 (grid) carriers of "
                             "236 values per step; targets precomputed on CPU"),
        },
        "claim_boundary": ("decomposition and sensitivity of the frozen issue-79 rollouts only; no re-freeze, "
                           "amendment, or reinterpretation of the issue-79 published dispositions; both families "
                           "(NovPhy / CLEVRER) stay reported separately; no new physics family, no planning/ranking "
                           "estimand, no perception claim, no leaderboard/VQA claim; descriptive intervals only; "
                           "either localization answer (concentrated or uniform) is a valid outcome"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_text": {path: (ROOT / path).read_text() for path in FILES},
    }


def load_plan(args):
    plan = read(args.output / "plan.json")
    if plan["schema"] != SCHEMA or plan["identity"] != IDENTITY:
        raise ValueError("frozen issue-83 plan identity differs")
    current = make_plan(args)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-83 source/settings differ; retain old output and explicitly version any change")
    return plan


def prepare(args):
    """Phase 0: verify inputs, freeze segmentation/census/estimands; no GPU, no errors computed."""
    if (args.output / "plan.json").exists():
        load_plan(args)
        log("frozen plan already present and validated; no re-freeze")
        return
    plan = make_plan(args)
    derived = {str(record["scene_index"]): raw_targets(record["scene_index"],
                                                       plan["representation"]["velocity_scale"])
               for record in plan["membership"]["scenes"]}
    atomic_torch(args.output / "scene-raw-targets.pt", {
        "schema": "issue_83_scene_raw_targets_v1", "identity": IDENTITY,
        "velocity_scale": plan["representation"]["velocity_scale"], "scenes": derived})
    write(args.output / "plan.json", plan)
    log(f"Phase-0 freeze complete: scenes={[m['scene_index'] for m in plan['membership']['scenes']]} "
        f"cells={plan['design']['cells']['total']} grid_census="
        f"{[(c['endpoint'], c['pre_first_collision'], c['active_cascade'], c['post_settlement']) for c in plan['segmentation']['grid_census']]}")


def raw_targets_loaded(args, plan):
    store = torch.load(args.output / "scene-raw-targets.pt", map_location="cpu", weights_only=True)
    if store["schema"] != "issue_83_scene_raw_targets_v1" or store["identity"] != plan["identity"] \
            or store["velocity_scale"] != plan["representation"]["velocity_scale"]:
        raise ValueError("raw-target store identity differs from the frozen plan")
    return store["scenes"]


# ---------------------------------------------------------------- traces


_CLIPS = {}


def scene_clip(scene_index):
    """Cached frozen #79 clip carrier for one scene (bounded: 12 x 128 x DIM floats)."""
    if scene_index not in _CLIPS:
        _CLIPS[scene_index] = source_shard(scene_index)["clip"]
    return _CLIPS[scene_index]


def build_batch(args, plan, kind, horizon, units, raw_scenes):
    """Initial carriers (N, DIM), true targets (N, K, DIM) and raw targets for one cell group."""
    initials, targets = [], []
    raw_position, raw_velocity, declared = [], [], []
    steps = TRACE_LAST[kind] // horizon
    for scene_index, start in units:
        clip = scene_clip(scene_index)
        initials.append(clip[start])
        targets.append(torch.stack([clip[start + step * horizon] for step in range(1, steps + 1)]))
        raw = raw_scenes[str(scene_index)]
        frames = [start + step * horizon for step in range(1, steps + 1)]
        raw_position.append(raw["raw_position"][frames])
        raw_velocity.append(raw["raw_velocity"][frames])
        declared.append(raw["declared"][0])
    return {"initial": torch.stack(initials), "targets": torch.stack(targets),
            "raw_position": torch.stack(raw_position), "raw_velocity": torch.stack(raw_velocity),
            "declared": torch.stack(declared)}


def typed_failure(args, plan, kind, seed, arm, horizon, scene_index, start, failure):
    atomic_torch(cell_path(args, kind, seed, arm, horizon, scene_index, start),
                 {"binding": probe_binding(plan, seed, arm, horizon, kind),
                  "unit": {"kind": kind, "scene_index": scene_index, "start": start},
                  "seed": seed, "arm": arm, "horizon": horizon, "failure": failure})


def run_group(args, plan, kind, seed, arm, horizon, model, raw_scenes, active_parameters, macs):
    """One (kind, seed, arm, horizon) group of cells in a single bounded batch."""
    units = scheduled_units(plan, kind)
    pair = pair_for(arm, horizon)
    batch = build_batch(args, plan, kind, horizon, units, raw_scenes)
    action = torch.zeros((len(units), 5))
    began = time.monotonic()
    trace = probe.trace_rollout(model, pair, batch["initial"].to(args.device),
                                batch["targets"].to(args.device), action.to(args.device),
                                batch["raw_position"].to(args.device), batch["raw_velocity"].to(args.device),
                                batch["declared"].to(args.device))
    wall = time.monotonic() - began
    steps = TRACE_LAST[kind] // horizon
    matrices = {key: torch.stack([row[key] for row in trace["rows"]])
                for key in ("position_mse", "velocity_mse", "carrier_mse", "position_all_frames",
                            "velocity_all_frames")}
    finite = torch.stack([row["finite"] for row in trace["rows"]]).all(0)
    bounds = {int(scene): tuple(value) for scene, value in plan["segmentation"]["cascade_bounds"].items()}
    failures = 0
    for index, (scene_index, start) in enumerate(units):
        failure = None
        if not bool(finite[index]):
            failure = "nonfinite_recursive_carrier"
        elif bool(matrices["position_mse"][:, index].isnan().all()):
            failure = "no_available_values_at_any_scored_frame"
        relative = [(step + 1) * horizon for step in range(steps)]
        record = {"binding": probe_binding(plan, seed, arm, horizon, kind),
                  "unit": {"kind": kind, "scene_index": scene_index, "start": start},
                  "seed": seed, "arm": arm, "horizon": horizon, "pair": list(pair.identity),
                  "active_parameters": active_parameters[arm],
                  "linear_macs_per_step": macs[(arm, horizon)], "transition_steps": steps,
                  "frames": {"relative": relative,
                             "absolute": [start + value for value in relative],
                             "segment": probe.segment_timeline(bounds[scene_index], start, relative),
                             "position_mse_masked": matrices["position_mse"][:, index].clone(),
                             "velocity_mse_masked": matrices["velocity_mse"][:, index].clone(),
                             "carrier_mse": matrices["carrier_mse"][:, index].clone(),
                             "position_mse_all_frames": matrices["position_all_frames"][:, index].clone(),
                             "velocity_mse_all_frames": matrices["velocity_all_frames"][:, index].clone()},
                  "group_wall_seconds": wall / len(units), "failure": failure}
        failures += 1 if failure else 0
        atomic_torch(cell_path(args, kind, seed, arm, horizon, scene_index, start), record)
    return wall, failures


def group_complete(args, kind, seed, arm, horizon, plan):
    return all(cell_path(args, kind, seed, arm, horizon, scene, start).exists()
               for scene, start in scheduled_units(plan, kind))


def run_evaluation(args, plan):
    """Every scheduled cell under one exclusive GPU lock, with the frozen stop rule."""
    if args.device != "cuda":
        raise ValueError("--run-evaluation measures wall time on the shared GPU; pass --device cuda")
    with gpu_lock():
        torch.cuda.reset_peak_memory_stats(args.device)
        return _run_evaluation_locked(args, plan)


def measured_gpu_wall(args, plan):
    """Wall seconds already spent on trace groups across resumes (the stop-rule budget)."""
    total = 0.
    for record in load_cells(args, plan).values():
        total += float(record.get("group_wall_seconds") or 0.)
    return total


def _run_evaluation_locked(args, plan):
    raw_scenes = raw_targets_loaded(args, plan)
    total = plan["design"]["cells"]["total"]
    done, failures = 0, 0
    source_plan = source_facts()
    phase_started = time.monotonic()
    spent_before_phase = measured_gpu_wall(args, plan)
    budget = plan["compute"]["allowance_gpu_hours"] * 3600
    stopped = False
    for seed in plan["design"]["seeds"]:
        models = {arm: load_frozen_predictor(argparse.Namespace(output=SOURCE_DIR, device=args.device),
                                             source_plan, seed, arm)[0] for arm in ARMS}
        active_parameters = {arm: int(source79.active_capacity_counts(models[arm], pair_for(arm, 1), args.device))
                             for arm in ARMS}
        macs = {(arm, horizon): linear_macs(models[arm], pair_for(arm, horizon))
                for arm in ARMS for horizon in HORIZONS}
        for kind in (WINDOW_KIND, GRID_KIND):
            for arm in ARMS:
                for horizon in HORIZONS:
                    done += len(scheduled_units(plan, kind))
                    spent = spent_before_phase + time.monotonic() - phase_started
                    if not stopped and spent >= budget:
                        stopped = True
                        write(args.output / STOP, {"schema": "issue_83_stop_rule_v1",
                                                   "reason": "compute_allowance_exceeded",
                                                   "measured_phase_wall_seconds": spent,
                                                   "allowance_gpu_hours": GPU_HOURS_ALLOWANCE})
                        log("stop rule reached: remaining cells retained as typed failures")
                    if stopped:
                        for scene, start in scheduled_units(plan, kind):
                            if not cell_path(args, kind, seed, arm, horizon, scene, start).exists():
                                typed_failure(args, plan, kind, seed, arm, horizon, scene, start,
                                              "compute_allowance_exceeded")
                                failures += 1
                        continue
                    if group_complete(args, kind, seed, arm, horizon, plan):
                        log(f"group {kind} seed={seed} arm={arm} h={horizon} complete; reused")
                        continue
                    wall, group_failures = run_group(args, plan, kind, seed, arm, horizon,
                                                     models[arm], raw_scenes, active_parameters, macs)
                    failures += group_failures
                    elapsed = time.monotonic() - phase_started
                    log(f"cells={done}/{total} group={kind} seed={seed} arm={arm} h={horizon} "
                        f"wall={wall:.2f}s elapsed={elapsed:.1f}s "
                        f"eta={elapsed / max(done, 1) * (total - done):.1f}s typed_failures={failures} "
                        f"memory={memory(args.device)}")
    log(f"evaluation phase complete wall={time.monotonic() - phase_started:.1f}s typed_failures={failures} "
        f"memory={memory(args.device)}")


# ---------------------------------------------------------------- publication


def load_cells(args, plan):
    """Load every scheduled cell record; typed failures stay in place (never rescued)."""
    records = {}
    for kind, seed, arm, horizon, scene_index, start in scheduled_cells(plan):
        path = cell_path(args, kind, seed, arm, horizon, scene_index, start)
        if not path.exists():
            continue
        record = torch.load(path, map_location="cpu", weights_only=True)
        check_binding(record, probe_binding(plan, seed, arm, horizon, kind))
        if record["unit"] != {"kind": kind, "scene_index": scene_index, "start": start}:
            raise ValueError("trace record identity differs from its path")
        records[(kind, seed, arm, horizon, scene_index, start)] = record
    return records


def frame_value(record, endpoint, mask):
    """Position MSE at one relative endpoint of a trace record under one mask variant."""
    if record["failure"]:
        return None
    values = record["frames"]["position_mse_masked"] if mask == "inside_camera_view" \
        else record["frames"]["position_mse_all_frames"]
    for index, relative in enumerate(record["frames"]["relative"]):
        if int(relative) == endpoint:
            value = float(values[index])
            return None if np.isnan(value) else value
    return None


def unit_value(records, kind, seed, arm, horizon, scene_index, start, endpoint, mask):
    record = records.get((kind, seed, arm, horizon, scene_index, start))
    return frame_value(record, endpoint, mask) if record else None


def _curve(records, plan, kind, seed, arm, horizon):
    """One (seed, arm, horizon) segment-conditional curve over visited frames."""
    collected = {segment: {} for segment in probe.SEGMENTS}
    for scene_index, start in scheduled_units(plan, kind):
        record = records.get((kind, seed, arm, horizon, scene_index, start))
        if record is None or record["failure"]:
            continue
        frames = record["frames"]
        for index, relative in enumerate(frames["relative"]):
            position = float(frames["position_mse_masked"][index])
            if np.isnan(position):
                continue
            velocity = float(frames["velocity_mse_masked"][index])
            bucket = collected[frames["segment"][index]].setdefault(int(relative), [])
            bucket.append((position, None if np.isnan(velocity) else velocity))
    return {segment: {str(relative): {"mean": float(np.mean([pair[0] for pair in values])),
                                      "velocity_mean": (float(np.mean([pair[1] for pair in values
                                                                       if pair[1] is not None]))
                                                        if any(pair[1] is not None for pair in values) else None),
                                      "units": len(values)}
                      for relative, values in sorted(bucket.items())}
            for segment, bucket in collected.items() if bucket}


def segment_curves(records, plan, kind):
    """E1: segment-conditional growth curves per arm, horizon, seed, plus the seed mean."""
    result = {}
    for arm in ARMS:
        for horizon in HORIZONS:
            per_seed = {str(seed): _curve(records, plan, kind, seed, arm, horizon) for seed in SEEDS}
            seed_mean = {}
            for segment in probe.SEGMENTS:
                relatives = sorted({int(relative) for curve in per_seed.values()
                                    for relative in curve.get(segment, {})})
                points = {}
                for relative in relatives:
                    contributing = [curve[segment][str(relative)] for curve in per_seed.values()
                                    if str(relative) in curve.get(segment, {})]
                    velocities = [point["velocity_mean"] for point in contributing
                                  if point["velocity_mean"] is not None]
                    points[str(relative)] = {
                        "mean": float(np.mean([point["mean"] for point in contributing])),
                        "velocity_mean": float(np.mean(velocities)) if velocities else None,
                        "units": min(point["units"] for point in contributing)}
                if points:
                    seed_mean[segment] = points
            result[f"{arm}_h{horizon}"] = {**per_seed, "seed_mean": seed_mean}
    return result


def ratio_table(records, plan, kind, endpoints):
    """E2: post-settlement vs pre/active ratio per arm, horizon, endpoint (descriptive)."""
    bounds = {int(scene): tuple(value) for scene, value in plan["segmentation"]["cascade_bounds"].items()}
    table = {}
    for arm in ARMS:
        for horizon in HORIZONS:
            for endpoint in endpoints:
                values, labels = [], []
                for scene_index, start in scheduled_units(plan, kind):
                    across = [unit_value(records, kind, seed, arm, horizon, scene_index, start, endpoint,
                                         "inside_camera_view") for seed in SEEDS]
                    if any(value is None for value in across):
                        continue
                    values.append(float(np.mean(across)))
                    labels.append(probe.segment_of_frame(bounds[scene_index], start + endpoint) == "post_settlement")
                entry = probe.ratio_statistic(values, labels)
                entry.update({"endpoint": endpoint, "arm": arm, "horizon": horizon, "unit_kind": kind,
                              "scored_units": len(values)})
                table[f"{arm}_h{horizon}_e{endpoint}"] = entry
    return table


def sensitivity_table(records, plan):
    """E3: endpoint-grid x mask sensitivity curves with the monotonicity predicate."""
    table = {}
    for arm in ARMS:
        for horizon in HORIZONS:
            for grid_name, endpoints in (("anchor", probe.BROAD_ENDPOINTS), ("fine", probe.GRID_ENDPOINTS)):
                for mask in ("inside_camera_view", "all_frames"):
                    curve = {}
                    for endpoint in endpoints:
                        across = []
                        for scene_index, start in scheduled_units(plan, GRID_KIND):
                            values = [unit_value(records, GRID_KIND, seed, arm, horizon, scene_index, start,
                                                 endpoint, mask) for seed in SEEDS]
                            if any(value is None for value in values):
                                continue
                            across.append(float(np.mean(values)))
                        curve[str(endpoint)] = float(np.mean(across)) if across else None
                    entry = probe.monotonicity(curve)
                    entry.update({"curve": curve, "grid": grid_name, "mask": mask, "arm": arm, "horizon": horizon})
                    table[f"{arm}_h{horizon}_{grid_name}_{mask}"] = entry
    return table


def anchor_consistency(sensitivity):
    """Descriptive consistency of the recomputed anchor curves against the published #79 numbers."""
    published = read(SOURCE_DIR / "summary.json")
    curves = published["dispositions"]["question_1_detail"]["components"]["S3"]["curves"]
    result = {}
    for horizon in HORIZONS:
        mine = sensitivity[f"continuous_h{horizon}_anchor_inside_camera_view"]["curve"]
        theirs = curves[str(horizon)]
        result[f"h{horizon}"] = {
            str(endpoint): {"issue83_probe": mine[str(endpoint)], "issue79_published": theirs[str(endpoint)],
                            "relative_difference": (abs(mine[str(endpoint)] - theirs[str(endpoint)])
                                                    / theirs[str(endpoint)]
                                                    if mine[str(endpoint)] is not None else None)}
            for endpoint in probe.BROAD_ENDPOINTS}
    return result


def compute_section(args, plan, records):
    groups = {}
    for (kind, seed, arm, horizon, _, _), record in records.items():
        key = f"{kind}/{seed}/{arm}_h{horizon}"
        entry = groups.setdefault(key, {"cells": 0, "wall_seconds": 0.,
                                        "transition_steps": record.get("transition_steps"), "failure_cells": 0})
        entry["cells"] += 1
        entry["wall_seconds"] += float(record.get("group_wall_seconds") or 0.)
        entry["failure_cells"] += 1 if record["failure"] else 0
    active, macs = {}, {}
    for (_, _, arm, horizon, _, _), record in records.items():
        if record.get("active_parameters") is not None:
            active[f"{arm}_h{horizon}"] = record["active_parameters"]
        if record.get("linear_macs_per_step") is not None:
            macs[f"{arm}_h{horizon}"] = record["linear_macs_per_step"]
    stable = [args.output / "plan.json", args.output / "scene-raw-targets.pt", args.output / STOP]
    stable += sorted((args.output / "traces").rglob("*.pt")) if (args.output / "traces").exists() else []
    derived_bytes = int(sum(path.stat().st_size for path in stable if path.exists()))
    return {"groups": groups, "active_parameters_per_system": active, "linear_macs_per_step": macs,
            "group_wall_seconds_total": float(sum(entry["wall_seconds"] for entry in groups.values())),
            "allowance_gpu_hours": GPU_HOURS_ALLOWANCE,
            "derived_artifacts_bytes": derived_bytes,
            "derived_artifacts_cap_gib": ARTIFACTS_CAP_GIB,
            "note": ("candidate counts do not exist on CLEVRER (no planning estimand); units, transition steps, "
                     "wall and active parameters are reported per arm; inference is not equalized; per-cell wall "
                     "seconds are equal shares of the measured group wall")}


def decide(ratios, sensitivity, inventory):
    """Frozen decision rules; descriptive tokens only."""
    insufficient = "readiness_or_precision_insufficient"
    detail = {}
    if inventory["missing_cells"] or inventory["typed_failure_cells"]:
        return ({key: insufficient for key in ("question_1_localization", "question_2_artifact_sensitivity")},
                {"reason": "missing or typed-failure cells present", "inventory": inventory})
    entry = ratios[GRID_KIND]["continuous_h15_e60"]
    if entry["post_units"] < probe.MIN_STRATUM_UNITS or entry["pre_active_units"] < probe.MIN_STRATUM_UNITS \
            or not entry["interval_reliable"]:
        q1 = insufficient
        detail["question_1_localization"] = {"reason": "stratum below the frozen minimum or unreliable interval",
                                             "entry": entry}
    elif entry["ratio"] < probe.RATIO_UNIFORM_LOWER:
        q1 = "supported"
        detail["question_1_localization"] = {"answer": "dip concentrated in post_settlement segments",
                                             "entry": entry}
    else:
        q1 = "not_supported_by_this_experiment"
        detail["question_1_localization"] = {
            "answer": ("uniform within the frozen band" if entry["ratio"] <= probe.RATIO_UNIFORM_UPPER
                       else "anti-concentrated: post-settlement error exceeds pre/active at e=60"),
            "entry": entry}
    variants = [sensitivity[f"continuous_h15_{grid}_{mask}"]
                for grid in ("anchor", "fine") for mask in ("inside_camera_view", "all_frames")]
    if any(variant["monotone_increasing"] is None for variant in variants):
        q2 = insufficient
        detail["question_2_artifact_sensitivity"] = {"reason": "an undecidable variant curve"}
    elif any(variant["monotone_increasing"] for variant in variants):
        q2 = "supported"
        detail["question_2_artifact_sensitivity"] = {
            "answer": "a primary variant turns monotone: endpoint/mask artifact supported"}
    else:
        q2 = "not_supported_by_this_experiment"
        detail["question_2_artifact_sensitivity"] = {"answer": "all primary variants stay non-monotone"}
    return {"question_1_localization": q1, "question_2_artifact_sensitivity": q2}, detail


def publication(args, plan):
    records = load_cells(args, plan)
    scheduled = len(scheduled_cells(plan))
    failures = Counter(record["failure"] for record in records.values() if record["failure"])
    inventory = {"scheduled_cells": scheduled, "executed_cells": len(records),
                 "typed_failure_cells": int(sum(failures.values())),
                 "missing_cells": scheduled - len(records), "typed_failures": dict(failures)}
    common = {"schema": "issue_83_clevrer_nonmonotonicity_report_v1",
              "identity": "issue-83-clevrer-nonmonotonicity-report-v1",
              "plan_identity": plan["identity"], "definitions": plan["design"],
              "membership": plan["membership"], "segmentation": plan["segmentation"],
              "representation": plan["representation"], "source": plan["source"],
              "compute": compute_section(args, plan, records), "inventory": inventory,
              "claim_boundary": plan["claim_boundary"],
              "limitations": [
                  "decomposition of the frozen issue-79 rollouts; no re-freeze or amendment of any issue-79 verdict",
                  "engine-annotation features in, no visual perception on either arm; shared frozen parser contract",
                  "descriptive paired bootstrap over (scene, start-or-context) units; no inferential claim",
                  "the hybrid arm runs its continuous-abstraction pair; the micro pair stays with issue-79 Q2",
                  "segment labels derive from collision-event timelines, not from measured object speeds"],
              "stop_rule": read(args.output / STOP)["reason"] if (args.output / STOP).exists() else None,
              "archived_release": False, "fresh_evaluation_opened": False,
              "final_evaluation_opened": False, "issue_64_authorized": False}
    if inventory["missing_cells"] or inventory["typed_failure_cells"]:
        dispositions, detail = decide({}, {}, inventory)
        return {**common, "_records": records, "diagnostics_complete": False,
                "dispositions": dispositions, "disposition_detail": detail}
    curves = {kind: segment_curves(records, plan, kind) for kind in (WINDOW_KIND, GRID_KIND)}
    ratios = {kind: ratio_table(records, plan, kind,
                                probe.GRID_ENDPOINTS if kind == GRID_KIND else (15, 30, 60))
              for kind in (WINDOW_KIND, GRID_KIND)}
    sensitivity = sensitivity_table(records, plan)
    dispositions, detail = decide(ratios, sensitivity, inventory)
    return {**common, "_records": records, "diagnostics_complete": True, "curves": curves, "ratios": ratios,
            "sensitivity": sensitivity, "anchor_consistency": anchor_consistency(sensitivity),
            "dispositions": dispositions, "disposition_detail": detail}


def compact_report(result):
    return {key: value for key, value in result.items() if key != "_records"}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("section", "arm", "horizon", "seed_or_mean", "segment", "relative_frame",
                     "position_mse_masked_mean", "velocity_mse_masked_mean", "units"))
    for kind, curves in result.get("curves", {}).items():
        for system, seeds in curves.items():
            arm, horizon = system.rsplit("_h", 1)
            for seed_or_mean, segments in seeds.items():
                for segment, points in segments.items():
                    for relative, point in points.items():
                        writer.writerow((f"E1_{kind}", arm, horizon, seed_or_mean, segment, relative,
                                         point["mean"], point["velocity_mean"], point["units"]))
    writer.writerow(())
    writer.writerow(("section", "unit_kind", "arm", "horizon", "endpoint_frames", "post_mean",
                     "pre_active_mean", "ratio", "interval_low", "interval_high",
                     "post_units", "pre_active_units", "uniform_band"))
    for kind, table in result.get("ratios", {}).items():
        for entry in table.values():
            low = high = None
            if entry.get("descriptive_95_percent_interval"):
                low, high = entry["descriptive_95_percent_interval"]
            writer.writerow((f"E2_{kind}", entry["unit_kind"], entry["arm"], entry["horizon"], entry["endpoint"],
                             entry["post_mean"], entry["pre_active_mean"], entry["ratio"], low, high,
                             entry["post_units"], entry["pre_active_units"],
                             "/".join(str(band) for band in entry["uniform_band"])))
    writer.writerow(())
    writer.writerow(("section", "arm", "horizon", "grid", "mask", "endpoint_frames", "position_mse_mean",
                     "monotone_increasing", "first_violation"))
    for entry in result.get("sensitivity", {}).values():
        for endpoint in entry["endpoints"]:
            writer.writerow(("E3", entry["arm"], entry["horizon"], entry["grid"], entry["mask"], endpoint,
                             entry["curve"].get(str(endpoint)), entry["monotone_increasing"],
                             None if not entry["first_violation"]
                             else "-".join(str(bound) for bound in entry["first_violation"])))
    writer.writerow(())
    writer.writerow(("section", "seed", "arm", "horizon", "scene_index", "context", "endpoint_frames",
                     "position_mse_masked", "position_mse_all_frames", "segment"))
    for (kind, seed, arm, horizon, scene_index, start), record in result["_records"].items():
        if kind != GRID_KIND or record["failure"]:
            continue
        frames = record["frames"]
        for endpoint in probe.GRID_ENDPOINTS:
            index = frames["relative"].index(endpoint)
            masked = float(frames["position_mse_masked"][index])
            all_frames = float(frames["position_mse_all_frames"][index])
            writer.writerow(("E3_evidence", seed, arm, horizon, scene_index, start, endpoint,
                             None if np.isnan(masked) else masked,
                             None if np.isnan(all_frames) else all_frames, frames["segment"][index]))
    writer.writerow(())
    writer.writerow(("section", "seed", "arm", "horizon", "scene_index", "window_start", "relative_frame",
                     "absolute_frame", "segment", "position_mse_masked", "velocity_mse_masked", "carrier_mse",
                     "position_mse_all_frames", "velocity_mse_all_frames"))
    for (kind, seed, arm, horizon, scene_index, start), record in result["_records"].items():
        if kind != WINDOW_KIND or record["failure"]:
            continue
        frames = record["frames"]
        for index, relative in enumerate(frames["relative"]):
            row = [seed, arm, horizon, scene_index, start, int(relative), frames["absolute"][index],
                   frames["segment"][index]]
            for values in (frames["position_mse_masked"], frames["velocity_mse_masked"], frames["carrier_mse"],
                           frames["position_mse_all_frames"], frames["velocity_mse_all_frames"]):
                value = float(values[index])
                row.append(None if np.isnan(value) else value)
            writer.writerow(tuple(["E1_evidence"] + row))
    return stream.getvalue()


def findings_md(result):
    lines = ["# Issue-83 CLEVRER h15 non-monotonicity decomposition - findings", ""]
    lines.append(f"Diagnostics complete: {result['diagnostics_complete']}. Claim boundary: {result['claim_boundary']}")
    lines.append("")
    lines.append("Endpoint semantics: window-relative or context-relative CLEVRER annotation frames; recursive "
                 "fixed-pair rollouts from the frozen initial carriers, no truth resets; the issue-79 published "
                 "verdicts are inputs and are never recomputed or amended.")
    lines.append("")
    if not result.get("curves"):
        lines.append("Diagnostics incomplete; no outcome numbers reported.")
        lines.append(f"- inventory: {result['inventory']}")
        if result.get("stop_rule"):
            lines.append(f"- stop rule: {result['stop_rule']}")
        lines.append(f"- dispositions: {result['dispositions']}")
        return "\n".join(lines) + "\n"
    lines.append("## Membership and frozen segment census")
    lines.append("")
    lines.append("| Scene | Objects | Collision frames | Cascade bounds (first, last) |")
    lines.append("| --- | --- | --- | --- |")
    for scene in result["membership"]["scenes"]:
        lines.append(f"| {scene['scene_index']} | {scene['objects']} | {scene['collision_frames']} "
                     f"| {tuple(scene['cascade_bounds'])} |")
    lines.append("")
    lines.append("| Grid endpoint e | pre_first_collision | active_cascade | post_settlement |")
    lines.append("| --- | --- | --- | --- |")
    for census in result["segmentation"]["grid_census"]:
        lines.append(f"| {census['endpoint']} | {census['pre_first_collision']} | {census['active_cascade']} "
                     f"| {census['post_settlement']} |")
    lines.append("")
    lines.append("## E1: segment-conditional recursive position MSE, h=15, seed mean (units per point; DESCRIPTIVE)")
    lines.append("")
    for kind, label in ((GRID_KIND, "grid traces (contexts 0..3, frames up to context+120)"),
                        (WINDOW_KIND, "window traces (starts 0..39, frames up to start+60)")):
        span = 120 if kind == GRID_KIND else 60
        visited = [endpoint for endpoint in (15, 30, 45, 60, 75, 90, 105, 120) if endpoint <= span]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Arm | Segment | " + " | ".join(f"e={e}" for e in visited) + " |")
        lines.append("| --- | --- | " + " | ".join(["---"] * len(visited)) + " |")
        for system, seeds in result["curves"][kind].items():
            if not system.endswith("h15"):
                continue
            for segment in probe.SEGMENTS:
                points = seeds["seed_mean"].get(segment, {})
                cells = []
                for endpoint in visited:
                    point = points.get(str(endpoint))
                    cells.append("n/a" if point is None else f"{point['mean']:.5f} ({point['units']})")
                lines.append(f"| {system.rsplit('_h', 1)[0]} | {segment} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## E2: post-settlement vs pre/active ratio of masked position MSE (DESCRIPTIVE)")
    lines.append("")
    lines.append("Ratio rho = post_settlement mean / pre+active mean over units at the scored frame; the frozen "
                 "uniformity band is [0.80, 1.25]; rho < 0.80 at the decision endpoint means the dip is "
                 "concentrated in post-settlement segments.")
    lines.append("")
    lines.append("| Kind | Arm | h | e | rho | 95% interval | post units | pre/active units |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for kind in (GRID_KIND, WINDOW_KIND):
        for entry in result["ratios"][kind].values():
            low = high = None
            if entry.get("descriptive_95_percent_interval"):
                low, high = entry["descriptive_95_percent_interval"]
            interval = "n/a" if low is None else f"[{low:.3f}, {high:.3f}]"
            ratio = "n/a" if entry["ratio"] is None else f"{entry['ratio']:.3f}"
            lines.append(f"| {kind} | {entry['arm']} | {entry['horizon']} | {entry['endpoint']} | {ratio} | "
                         f"{interval} | {entry['post_units']} | {entry['pre_active_units']} |")
    lines.append("")
    lines.append("## E3: endpoint-grid and mask sensitivity of the recursive position MSE (DESCRIPTIVE)")
    lines.append("")
    lines.append("| Arm | h | Grid | Mask | " + " | ".join(f"e={e}" for e in probe.GRID_ENDPOINTS)
                 + " | monotone increasing | first violation |")
    lines.append("| --- | --- | --- | --- | " + " | ".join(["---"] * len(probe.GRID_ENDPOINTS)) + " | --- | --- |")
    for entry in result["sensitivity"].values():
        cells = ["n/a" if entry["curve"].get(str(endpoint)) is None
                 else f"{entry['curve'][str(endpoint)]:.5f}" for endpoint in probe.GRID_ENDPOINTS]
        violation = "-" if not entry["first_violation"] \
            else f"{entry['first_violation'][0]}->{entry['first_violation'][1]}"
        lines.append(f"| {entry['arm']} | {entry['horizon']} | {entry['grid']} | {entry['mask']} | "
                     + " | ".join(cells) + f" | {entry['monotone_increasing']} | {violation} |")
    lines.append("")
    lines.append("## Anchor consistency against the published issue-79 curves (provenance only)")
    lines.append("")
    lines.append("| h | e | issue-83 probe | issue-79 published | relative difference |")
    lines.append("| --- | --- | --- | --- | --- |")
    for horizon, entries in result["anchor_consistency"].items():
        for endpoint, entry in entries.items():
            relative = entry["relative_difference"]
            probe_value = entry["issue83_probe"]
            lines.append(f"| {horizon[1:]} | {endpoint} | "
                         + (f"{probe_value:.6f}" if probe_value is not None else "n/a") + " | "
                         f"{entry['issue79_published']:.6f} | "
                         + (f"{relative:.4f}" if relative is not None else "n/a") + " |")
    lines.append("")
    lines.append("## Dispositions")
    lines.append("")
    for key, label in (("question_1_localization", "Q1 localization (regime property vs uniform)"),
                       ("question_2_artifact_sensitivity", "Q2 artifact sensitivity (grid/mask)")):
        detail = result["disposition_detail"].get(key, {})
        suffix = f" - {detail['answer']}" if detail.get("answer") else \
            (f" ({detail['reason']})" if detail.get("reason") else "")
        lines.append(f"- {label}: **{result['dispositions'][key]}**{suffix}")
    lines.append("")
    lines.append("Both families are reported separately; a CLEVRER non-replication does not falsify the NovPhy "
                 "boundary, and vice versa. Either localization answer is a valid outcome.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def publish(args, plan):
    result = publication(args, plan)
    write(args.output / "summary.json", compact_report(result))
    (args.output / "comparisons.csv").write_text(comparisons_csv(result))
    (args.output / "findings.md").write_text(findings_md(result))
    log(f"published diagnostics_complete={result['diagnostics_complete']} "
        f"dispositions={result['dispositions']}")


# ---------------------------------------------------------------- smoke / dry-run


def smoke(args):
    """Bounded real pass with the frozen seed-20260908 checkpoints, CPU, production_evidence false."""
    began = time.monotonic()
    plan = load_plan(args)
    raw_scenes = raw_targets_loaded(args, plan)
    source_plan = source_facts()
    models = {arm: load_frozen_predictor(argparse.Namespace(output=SOURCE_DIR, device="cpu"),
                                         source_plan, SEEDS[0], arm)[0] for arm in ARMS}
    cells = []
    for kind, units in ((WINDOW_KIND, [(10000, 0), (10000, 25)]), (GRID_KIND, [(10000, 0)])):
        for arm in ARMS:
            for horizon in HORIZONS:
                pair = pair_for(arm, horizon)
                batch = build_batch(args, plan, kind, horizon, units, raw_scenes)
                action = torch.zeros((len(units), 5))
                trace = probe.trace_rollout(models[arm], pair, batch["initial"], batch["targets"], action,
                                            batch["raw_position"], batch["raw_velocity"], batch["declared"])
                for index, (scene_index, start) in enumerate(units):
                    last = trace["rows"][-1]
                    cells.append({"kind": kind, "scene_index": scene_index, "start": start, "arm": arm,
                                  "horizon": horizon,
                                  "last_position_mse_masked": float(last["position_mse"][index]),
                                  "last_position_mse_all_frames": float(last["position_all_frames"][index]),
                                  "finite": all(bool(row["finite"][index]) for row in trace["rows"])})
    result = {"schema": "issue_83_clevrer_smoke_v1", "production_evidence": False,
              "seed": SEEDS[0], "device": "cpu", "cells": cells,
              "wall_seconds": time.monotonic() - began, "memory": memory("cpu")}
    write(args.output / "smoke.json", result)
    if read(args.output / "smoke.json")["cells"] != cells:
        raise ValueError("smoke evidence JSON round trip differs")
    log(f"real smoke complete cells={len(cells)} wall={result['wall_seconds']:.1f}s; "
        "frozen checkpoints only, no production evidence")


def dry_run(args):
    """No-write readiness report; exits 0 with or without the frozen inputs."""
    checks = {
        "issue79_plan": (SOURCE_DIR / "plan.json").exists(),
        "issue79_shards": all((SOURCE_DIR / "shards" / f"scene-{index}.pt").exists()
                              for index in range(10000, 10012)),
        "issue79_checkpoints": all((SOURCE_DIR / f"seed-{seed}" / arm / "predictor.pt").exists()
                                   for seed in SEEDS for arm in ARMS),
        "validation_zip": VALIDATION_ZIP.exists(),
        "frozen_plan": (args.output / "plan.json").exists(),
        "raw_targets": (args.output / "scene-raw-targets.pt").exists(),
    }
    log(f"no-write dry-run: modes=prepare,smoke-test,run-evaluation,publish,validate; "
        f"inputs={ {key: bool(value) for key, value in checks.items()} }; output={args.output}")
    if checks["issue79_plan"]:
        window_cells = len(WINDOW_STARTS) * 12 * len(SEEDS) * len(ARMS) * len(HORIZONS)
        grid_cells = len(GRID_CONTEXTS) * 12 * len(SEEDS) * len(ARMS) * len(HORIZONS)
        log(f"dry-run planned cells: window={window_cells} grid={grid_cells} total={window_cells + grid_cells}; "
            f"allowance={GPU_HOURS_ALLOWANCE} GPU-h on the --run-evaluation phase; lock={GPU_LOCK}")
    log("dry-run: --prepare freezes segmentation, census, margins and sensitivity grids before any error is computed")
    return 0


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "run-evaluation", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            return dry_run(args)
        if args.smoke_test:
            smoke(args)
            return 0
        if args.prepare:
            prepare(args)
            return 0
        plan = load_plan(args)
        if args.run_evaluation:
            run_evaluation(args, plan)
        elif args.publish:
            publish(args, plan)
        else:
            result = publication(args, plan)
            if compact_report(result) != read(args.output / "summary.json") \
                    or comparisons_csv(result).encode("utf-8") != (args.output / "comparisons.csv").read_bytes() \
                    or findings_md(result) != (args.output / "findings.md").read_text():
                raise ValueError("published issue-83 tables differ from bound source evidence")
            log("exact saved-evidence validation passed")
        return 0
    except (ValueError, OSError, KeyError, probe.ClevrerNonmonotonicityError,
            clevrer.ClevrerBoundaryError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
