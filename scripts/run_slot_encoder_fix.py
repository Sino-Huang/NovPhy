"""Issue-99 step 3: fix the dominant stage (perception) and score held-out ranking.

Binding runner module: scripts/run_slot_encoder_fix.py
Exact validation command: python -u -m scripts.run_slot_encoder_fix --validate

Gate A of the frozen #99 attribution (issue-99-decision-chain-attribution-v1,
grid, all members, tie-free cost, request-mean) named perception as the dominant
loss (a - ap = 0.2925 [0.0922, 0.4891]) and fixed the step-3 remedy as "improve
the slot encoder"; the frozen issue-70 parser reads the pig as absent in 859/859
decision frames. This runner replaces the slot encoder and asks whether the
predicted cost then orders held-out candidate launches above the pre-declared
target (grid AUC >= 0.65 with lower bound > 0.55).

Arms (identical fit data, windows, recipe, seeds; only the slot encoder differs):
- E   new spatial slot parser (world_model/training/spatial_slot_parser.py,
      320x240 input, slot queries attending over a 40x30 feature map), trained
      on fit-lineage frames against engine-projected targets (the issue-70
      repair.targets builder), then hybrid and continuous predictors retrained
      from scratch on its carriers.
- R0  the frozen issue-70 parser; hybrid and continuous predictors retrained on
      its carriers with the same data and recipe (isolates the encoder).
- current: the frozen #77 N1 checkpoints, read from the #99 attribution records
      (stage c), reference only.

Fit data: every #77 N1 admissible branch of the #77 predictor and controller
lineages plus every verdict-bearing #93/#94/#87/#89 inventory execution of those
lineages. Held-out lineages (issue-77-n1-007/008/014/016) never enter any
training step; every statistic here is on held-out members only.

Modes: --dry-run --smoke --prepare --run --publish --validate (frozen-protocol
chronology; the smoke trains nothing that is kept and joins no verdict).
Every interval is DESCRIPTIVE (member-clustered bootstrap, 10000 draws, PCG64 7201).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_launch_power_probe as p94
from scripts import run_lookahead_control as wlc
from scripts import run_tau_ad_within_checkpoint as t96

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-99-slot-encoder-fix-v1"
ATTRIBUTION = m99.OUTPUT
DYNAMICS = t96.DYNAMICS
CAMPAIGN = ROOT / ".local-artifacts/issue-77-n1-v1"

IDENTITY = "issue-99-slot-encoder-fix-v1"
SCHEMA_PLAN = "issue_99_slot_encoder_fix_plan_v1"
SCHEMA_SMOKE = "issue_99_slot_encoder_fix_smoke_v1"
SCHEMA_SHOT = "issue_99_slot_encoder_fix_shot_v1"
SCHEMA_CARRIERS = "issue_99_slot_encoder_fix_carriers_v1"
SCHEMA_ENCODER = "issue_99_spatial_slot_parser_checkpoint_v1"
SCHEMA_PREDICTOR = "issue_99_slot_encoder_fix_predictor_v1"
SCHEMA_EVAL = "issue_99_slot_encoder_fix_eval_record_v1"
SCHEMA_PERCEPTION = "issue_99_slot_encoder_fix_perception_record_v1"
SCHEMA_LEDGER = "issue_99_slot_encoder_fix_ledger_v1"
SCHEMA_COMPUTE = "issue_99_slot_encoder_fix_compute_v1"
SCHEMA_REPORT = "issue_99_slot_encoder_fix_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_slot_encoder_fix --validate"
RUNNER = "scripts/run_slot_encoder_fix.py"
MODEL_MODULE = "world_model/training/spatial_slot_parser.py"

DEVICE = "cuda"
SEEDS = t96.SEEDS
ENDPOINT = t96.ENDPOINT
INVENTORIES = t96.INVENTORIES
PRIMARY = t96.PRIMARY
HELD_OUT = m99.HELD_OUT
ARMS = ("E", "R0")
FAMILIES = ("hybrid", "continuous")
REQUESTS = (*t96.FIXED_ARMS, *t96.CONTINUOUS_FIXED_ARMS)
COSTS = m99.COSTS
PRIMARY_COST = m99.PRIMARY_COST

ENCODER = {"image_height": 240, "image_width": 320, "hidden_dim": 128, "epochs": 12,
           "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 9900001,
           "attention_layers": 2, "attention_heads": 4}
IMAGE_STRIDE = 5
MAX_POSITION = 285
WINDOW_STARTS = tuple(range(0, ENDPOINT + 1, 15))
WINDOW_LENGTH = 60
PARSE_CHUNK = 32
PARSE_THREADS = 6
SHOT_WORKERS = 6

TARGET_AUC = m99.TARGET_AUC
TARGET_LOWER = m99.TARGET_LOWER
DELTA = 0.02
NONFINITE_SHARE_MAX = 0.10
REPLICATION_TOLERANCE = 1e-3
GPU_CAP_SECONDS = 8 * 3600.0
WALL_CAP_SECONDS = 16 * 3600.0
STOP_TOKEN = t96.STOP_TOKEN
SMOKE_STEPS = 20
PLACEHOLDER_MARKERS = m99.PLACEHOLDER_MARKERS

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
bootstrap = t96.bootstrap
cell_auc = t96.cell_auc
fmt = t96.fmt
fmt_interval = t96.fmt_interval


def log(message):
    print(f"[issue-99-slot-encoder-fix] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def training_recipe():
    frozen = read_json(DYNAMICS / "plan.json")["training"]
    return {key: frozen[key] for key in ("steps", "batch_size", "learning_rate", "weight_decay",
                                         "grad_clip")}


def vocabulary():
    return read_json(DYNAMICS / "plan.json")["contract"]["vocabulary"]


# ---------------------------------------------------------------------------
# membership (no statistic)
# ---------------------------------------------------------------------------

def fit_members():
    predictor, controller = m99.fit_lineages()
    members = sorted(set(predictor) | set(controller))
    if set(members) & set(HELD_OUT):
        raise ValueError("held-out member inside the fit lineages")
    return members


def training_shots(sources, universe):
    """Frozen order: #77 N1 admissible branches of fit lineages, then inventory shots."""
    shots = []
    records = read_json(DYNAMICS / "plan.json")["n1_lineages"]["records"]
    for member in fit_members():
        for branch in records[member]["branches"]:
            result = read_json(CAMPAIGN / "results" / f"{branch}.json")
            if result["member_identity"] != branch or result["complete"] is not True:
                raise ValueError(f"#77 N1 branch {branch} is not a complete admissible capture")
            shots.append({"key": f"n1--{branch}", "member": member, "source": "issue_77_n1",
                          "attempt": str(CAMPAIGN / "attempts" / branch)})
    fitted = set(fit_members())
    for inventory, member, ordinal in m99.candidates(universe):
        if member not in fitted:
            continue
        record = m99.trace_record(sources, inventory, member, universe[inventory][member], ordinal)
        attempt = Path(record["execution"]["native_root"]).parent.parent
        shots.append({"key": f"inv--{m99.candidate_key(inventory, member, ordinal)}", "member": member,
                      "source": f"inventory_{inventory}", "attempt": str(attempt)})
    return shots


def held_out_shots(universe):
    return [(inventory, member, ordinal) for inventory, member, ordinal in m99.candidates(universe)
            if member in HELD_OUT]


# ---------------------------------------------------------------------------
# shot data: engine-projected labels and encoder training images
# ---------------------------------------------------------------------------

def shot_frames(attempt):
    shot_root = Path(attempt) / "shot-1"
    observation_root = shot_root / "observation-trace"
    frames = read_json(observation_root / "observation_trace_manifest.json")["frame_records"]
    if (len(frames) < 2 or frames[0]["fixed_step"] != m99.DECISION_FIXED_STEP
            or any(f["fixed_step"] != m99.DECISION_FIXED_STEP + m99.NATIVE_STRIDE * i
                   for i, f in enumerate(frames[:-1]))):
        raise ValueError(f"observation cadence differs for {attempt}")
    segment = read_json(shot_root / "segment.json") if (shot_root / "segment.json").is_file() else None
    return shot_root, observation_root, frames, segment


def shot_segment(shot):
    if shot["source"] == "issue_77_n1":
        branch = shot["key"].removeprefix("n1--")
        return read_json(CAMPAIGN / "results" / f"{branch}.json")["segments"][0]
    return read_json(Path(shot["attempt"]) / "shot-1" / "segment.json")


def shot_labels(segment, frames, positions, names):
    """repair.targets per observed position (the #77 build_n1_shot label semantics:
    cohort-v2 micro relations over the observed samples, macro labels from native
    stability events and the preceding native step's support set)."""
    from scripts.canonical_native_trace import NativeTrace
    from scripts.cohort_v2_micro_relations import derive_capture_micro_relations
    from scripts.run_issue_70_parser_repair import targets
    from scripts.run_issue_77_n1_train import _capture_shim
    trace = NativeTrace(segment["native_root"])
    steps = [frames[p]["fixed_step"] for p in range(len(frames))]
    if [f["fixed_step"] for f in trace.manifest["frame_records"]] != steps:
        raise ValueError("native physics and observations do not share exact endpoints")
    frame_steps = set(steps)
    samples, macro = [], {}
    steady, previous_supports = None, None
    for chunk in trace.chunks():
        stability = {event["fixed_step"]: event["event_type"] == "stable_entered"
                     for event in chunk["events"]
                     if event["event_type"] in ("stable_entered", "stable_exited")}
        for sample in chunk["fixed_step_samples"]:
            step = sample["fixed_step"]
            if step in stability:
                steady = stability[step]
            supports = {(s["supporter_entity_id"], s["supported_entity_id"]) for s in sample["supports"]}
            if step in frame_steps:
                steady_label = {"value": None if steady is None else bool(steady),
                                "availability": "available" if steady is not None
                                else "unavailable_incomplete_debounce_window"}
                available = steady is not None and previous_supports is not None
                unstable_label = {"value": (steady is False and previous_supports is not None
                                            and supports != previous_supports) if available else None,
                                  "availability": "available" if available
                                  else ("unavailable_steady_state" if steady is None
                                        else "unavailable_no_predecessor")}
                macro[step] = {"fixed_step": step, "predicates": {"steady-state": steady_label,
                                                                  "structure-unstable": unstable_label}}
                samples.append(sample)
            previous_supports = supports
    micro = derive_capture_micro_relations(
        _capture_shim(segment, samples, []), source_reference=f"issue-99:{segment['capture_id']}",
        source_capture_bundle_identity=f"issue-99:{segment['identity']}")["labels"]
    slots = set(names)
    rows = []
    for p in positions:
        sample = samples[p]
        filtered = {**sample, "entities": [e for e in sample["entities"] if e["scenario_object_id"] in slots]}
        rows.append(targets(filtered, frames[p]["capture_metadata"], micro[p], macro[sample["fixed_step"]],
                            names))
    return {k: torch.stack([row[k] for row in rows]) for k in rows[0]}


def image_tensor(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((ENCODER["image_width"], ENCODER["image_height"]),
                                            Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)


def action_of(segment):
    action = segment["action"]
    x, y = action["drag_release"]
    return torch.tensor((x / 480., y / 480., action.get("release_time", 1000) / 1000.,
                         action.get("tap_time", 0) / 1000., 1.), dtype=torch.float32)


def build_shot(shot, names):
    shot_root, observation_root, frames, _ = shot_frames(shot["attempt"])
    segment = shot_segment(shot)
    end = min(len(frames) - 1, MAX_POSITION)
    positions = list(range(end + 1))
    labels = shot_labels(segment, frames, positions, names)
    image_positions = list(range(0, end + 1, IMAGE_STRIDE))
    with ThreadPoolExecutor(PARSE_THREADS) as pool:
        images = list(pool.map(lambda p: image_tensor(
            observation_root / frames[p]["agent_observation"]["relative_path"]), image_positions))
    return {"schema": SCHEMA_SHOT, "plan_identity": IDENTITY, "key": shot["key"], "member": shot["member"],
            "source": shot["source"], "attempt": shot["attempt"], "end": end,
            "frame_count": len(frames), "action": action_of(segment), "labels": labels,
            "image_positions": image_positions, "images": torch.stack(images)}


def shot_path(output, key):
    return Path(output) / "shots" / f"{key}.pt"


def build_and_save(output, shot, names):
    """Process-pool worker: one shot's outcome-free labels and images."""
    torch.set_num_threads(1)
    atomic_torch(shot_path(output, shot["key"]), build_shot(shot, names))
    return shot["key"]


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------

def encoder_config():
    from world_model.training.cohort_v2_visual_parser import CohortV2VisualParserConfig
    return CohortV2VisualParserConfig(seed=ENCODER["seed"], image_height=ENCODER["image_height"],
                                      image_width=ENCODER["image_width"], hidden_dim=ENCODER["hidden_dim"],
                                      epochs=ENCODER["epochs"], batch_size=ENCODER["batch_size"],
                                      learning_rate=ENCODER["learning_rate"],
                                      weight_decay=ENCODER["weight_decay"], device=DEVICE)


def new_encoder():
    from world_model.training.spatial_slot_parser import SpatialSlotParser
    return SpatialSlotParser(encoder_config(), tuple(vocabulary()), layers=ENCODER["attention_layers"],
                             heads=ENCODER["attention_heads"])


def wrap(model):
    from scripts import run_issue_70_parser_repair as repair
    plan70 = repair.load_plan(repair.ROOT)
    return repair.RepairedAdapter(model.to(DEVICE).eval(), parser_checkpoint_identity=f"{IDENTITY}:E",
                                  temperatures=plan70["temperatures"], thresholds=plan70["thresholds"],
                                  object_kind_temperature=1.0, **repair.carrier_dimensions(plan70))


def load_adapter(output, arm):
    if arm == "R0":
        return wlc.load_adapter(DEVICE)
    payload = torch.load(Path(output) / "encoder" / "encoder.pt", map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_ENCODER or payload.get("plan_identity") != IDENTITY:
        raise ValueError("encoder checkpoint binding differs")
    model = new_encoder()
    model.load_state_dict(payload["model"], strict=True)
    return wrap(model)


def encoder_batches(paths, batch_size, generator):
    """Shot-shuffled frame batches (the repair.batches streaming pattern)."""
    pending, count = [], 0
    for index in torch.randperm(len(paths), generator=generator).tolist():
        shot = torch.load(paths[index], weights_only=True)
        labels = {k: v[shot["image_positions"]] for k, v in shot["labels"].items()}
        pending.append({"images": shot["images"], **labels})
        count += len(shot["images"])
        if count >= batch_size * 8:
            merged = {k: torch.cat([t[k] for t in pending]) for k in pending[0]}
            order = torch.randperm(count, generator=generator)
            for start in range(0, count - batch_size + 1, batch_size):
                yield {k: v[order[start:start + batch_size]] for k, v in merged.items()}
            pending, count = [], 0
    if pending:
        merged = {k: torch.cat([t[k] for t in pending]) for k in pending[0]}
        yield merged


def presence_balance(paths, names):
    from scripts.run_issue_70_parser_repair import class_balance_weights
    total, positive = 0, torch.zeros(len(names))
    for path in paths:
        shot = torch.load(path, weights_only=True)
        presence = shot["labels"]["presence"][shot["image_positions"]]
        total += len(presence)
        positive += presence.sum(0)
    return total, positive, class_balance_weights(total, positive)


def train_encoder(output, paths, steps_limit=None, target=None):
    """repair.parser_loss (presence x4, balanced pig/block presence x4, counts, centers,
    kinds, relations, macros) with the issue-70 optimizer; last epoch kept."""
    from scripts.run_issue_70_parser_repair import parser_loss
    names = vocabulary()
    total, positive, weights = presence_balance(paths, names)
    torch.manual_seed(ENCODER["seed"])
    model = new_encoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ENCODER["learning_rate"],
                                  weight_decay=ENCODER["weight_decay"])
    target = Path(target or Path(output) / "encoder" / "encoder.pt")
    progress = target.with_name("encoder-progress.pt")
    first, reports, gpu = 0, [], 0.0
    if progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first, reports, gpu = saved["epoch"], saved["reports"], saved["gpu_seconds"]
    epochs = 1 if steps_limit else ENCODER["epochs"]
    steps = 0
    for epoch in range(first, epochs):
        model.train()
        generator = torch.Generator().manual_seed(ENCODER["seed"] + epoch)
        began, seen, loss_sum = time.monotonic(), 0, 0.0
        for batch in encoder_batches(paths, ENCODER["batch_size"], generator):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = parser_loss(model, batch, names, weights.to(DEVICE))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite encoder loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            seen += len(batch["images"])
            loss_sum += float(loss.detach()) * len(batch["images"])
            steps += 1
            if steps_limit and steps >= steps_limit:
                break
        torch.cuda.synchronize()
        wall = time.monotonic() - began
        gpu += wall
        reports.append({"epoch": epoch + 1, "frames": seen, "mean_loss": loss_sum / max(seen, 1),
                        "wall_seconds": wall})
        log(f"encoder epoch {epoch + 1}/{epochs}: frames {seen}, loss {loss_sum / max(seen, 1):.5f}, "
            f"{wall:.0f}s")
        if not steps_limit:
            atomic_torch(progress, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "epoch": epoch + 1, "reports": reports, "gpu_seconds": gpu})
    payload = {"schema": SCHEMA_ENCODER, "plan_identity": IDENTITY,
               "architecture": model.architecture_identity, "config": ENCODER,
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "reports": reports, "gpu_seconds": gpu, "training_frames": total,
               "presence_positive": positive.tolist(), "presence_weights": weights.tolist(),
               "parameters": sum(p.numel() for p in model.parameters()),
               "complete": not steps_limit}
    atomic_torch(target, payload)
    return payload


# ---------------------------------------------------------------------------
# carriers and windows
# ---------------------------------------------------------------------------

def parse_positions(adapter, observation_root, frames, positions):
    from world_model.data.deployment_temporal import AgentObservation

    def observation(p):
        ref = frames[p]["agent_observation"]
        return AgentObservation(ref["identity"], frames[p]["fixed_step"], frames[p]["fixed_time_seconds"],
                                (observation_root / ref["relative_path"]).read_bytes(), "agent")

    chunks = [positions[i:i + PARSE_CHUNK] for i in range(0, len(positions), PARSE_CHUNK)]
    with torch.no_grad(), ThreadPoolExecutor(PARSE_THREADS) as pool:
        values = list(pool.map(lambda chunk: adapter.parse_batch(tuple(observation(p) for p in chunk)), chunks))
    return dict(zip(positions, [v for chunk in values for v in chunk], strict=True))


def carriers_for(adapter, attempt, positions):
    """Parsed carriers at positions (prior position - 1; terminal absorption)."""
    from world_model.data.deployment_temporal import AgentObservation
    _, observation_root, frames, _ = shot_frames(attempt)
    last = len(frames) - 1
    needed = sorted({q for p in positions for q in (max(0, min(p, last) - 1), min(p, last))})
    parsed = parse_positions(adapter, observation_root, frames, needed)

    def light(p):
        ref = frames[p]["agent_observation"]
        return AgentObservation(ref["identity"], frames[p]["fixed_step"], frames[p]["fixed_time_seconds"],
                                b"", "agent")

    out = []
    for p in positions:
        q = min(p, last)
        out.append(m99.build(adapter, None if q == 0 else light(q - 1), light(q), parsed[q],
                             None if q == 0 else parsed[q - 1]))
    return torch.stack(out)


def carrier_path(output, arm, key):
    return Path(output) / "carriers" / arm / f"{key}.pt"


def windows(shot, carriers):
    """Windows of WINDOW_LENGTH transitions at WINDOW_STARTS inside the shot's positions."""
    end = shot["end"]
    rows = []
    for start in WINDOW_STARTS:
        if start >= end:
            continue
        length = min(WINDOW_LENGTH, end - start)
        index = [min(start + i, start + length) for i in range(WINDOW_LENGTH + 1)]
        rows.append({"z": carriers[index], "action": shot["action"], "length": torch.tensor(length),
                     "relations": shot["labels"]["relations"][index].bool(),
                     "relations_mask": shot["labels"]["relation_mask"][index].bool(),
                     "macros": shot["labels"]["macros"][index].bool(),
                     "macros_mask": shot["labels"]["macro_mask"][index].bool()})
    return rows


def window_data(output, arm, shots):
    rows = []
    for shot in shots:
        data = torch.load(shot_path(output, shot["key"]), weights_only=True)
        carriers = torch.load(carrier_path(output, arm, shot["key"]), weights_only=True)["carriers"]
        rows.extend(windows(data, carriers))
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

def predictor_path(output, arm, seed, family):
    return Path(output) / "models" / arm / f"seed-{seed}" / family / "predictor.pt"


def new_predictor(family):
    from world_model.training.cnn_hybrid import CNNHybridPredictor
    from world_model.training.matched_dynamics import ContinuousDynamics
    if family == "continuous":
        return ContinuousDynamics(read_json(DYNAMICS / "plan.json")["capacity"]["continuous_width"])
    return CNNHybridPredictor()


def train_predictor(data, family, seed, target, steps=None):
    """The #74/#77 recipe (PAIRS[step % 9]; continuous uses the same horizon; local +
    recursive MSE + carrier bound; hybrid adds symbolic losses) on uniform window
    minibatches; from scratch; last update kept."""
    from scripts.run_issue_71_hybrid_readiness import sample
    from scripts.run_issue_74_matched_dynamics import loss_for
    from world_model.model import Abstraction, PredictionPair
    from world_model.training.cnn_hybrid import PAIRS
    recipe = training_recipe()
    total = steps or recipe["steps"]
    torch.manual_seed(seed)
    model = new_predictor(family).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                                  weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    if family == "continuous":
        data = {k: data[k] for k in ("z", "action", "length")}
    counts, trace = {}, []
    torch.cuda.synchronize()
    began = time.monotonic()
    for step in range(total):
        batch = sample(data, recipe["batch_size"], generator, DEVICE)
        selected = PAIRS[step % 9]
        pair = PredictionPair(selected.delta, Abstraction.CONTINUOUS) if family == "continuous" else selected
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, pair)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite predictor loss {family} seed {seed} step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
        optimizer.step()
        counts[str(pair.identity)] = counts.get(str(pair.identity), 0) + 1
        if (step + 1) % 900 == 0 or step + 1 == total:
            trace.append({"step": step + 1, "loss": float(loss.detach())})
    torch.cuda.synchronize()
    payload = {"schema": SCHEMA_PREDICTOR, "plan_identity": IDENTITY, "family": family, "seed": seed,
               "steps": total, "pair_counts": counts, "loss_trace": trace,
               "gpu_seconds": time.monotonic() - began, "windows": len(data["z"]),
               "parameters": sum(p.numel() for p in model.parameters()),
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if target is not None:
        atomic_torch(target, payload)
    return model.eval(), payload


def load_predictor(output, arm, seed, family):
    payload = torch.load(predictor_path(output, arm, seed, family), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_PREDICTOR or payload.get("plan_identity") != IDENTITY
            or payload["seed"] != seed or payload["family"] != family
            or payload["steps"] != training_recipe()["steps"]):
        raise ValueError(f"predictor {arm}/{seed}/{family} binding differs")
    model = new_predictor(family)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# evaluation records (held-out members only)
# ---------------------------------------------------------------------------

def eval_path(output, arm, inventory, member, seed):
    return Path(output) / "records" / f"eval--{arm}--{inventory}--{member}--seed{seed}.json"


def perception_path(output, arm, inventory, member):
    return Path(output) / "records" / f"perception--{arm}--{inventory}--{member}.json"


def eval_record(universe, costs, adapter, models, arm, inventory, member, seed):
    entry = universe[inventory][member]
    items = entry["inventory"]
    actions = m99.action_batch(items)
    anchor = p94.frozen_anchor(member)
    z0 = wlc.encode_carrier(adapter, anchor, DEVICE).float()
    requests = {}
    for request in REQUESTS:
        family = t96.arm_family(request)
        z_end, timing = m99.rollout_endpoints(models[family], None, z0.to(DEVICE), actions,
                                              t96.arm_spec(request))
        rows = [costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)]
        requests[request] = {"rows": rows, "candidates": len(items), **timing}
    return {"schema": SCHEMA_EVAL, "plan_identity": IDENTITY,
            "cell": {"arm": arm, "inventory": inventory, "member": member, "seed": seed},
            "anchor_sha256": anchor["sha256"], "requests": requests, "engine_seconds": 0}


def perception_record(sources, universe, costs, adapter, arm, inventory, member):
    """Outcome-free: the arm's parsed carriers at positions 0 and 225 of every held-out shot,
    the stage-ap costs, and presence against the engine carriers of the #99 evidence."""
    entry = universe[inventory][member]
    shots, rows = {}, []
    for ordinal in sorted(entry["verdicts"]):
        record = m99.trace_record(sources, inventory, member, entry, ordinal)
        attempt = Path(record["execution"]["native_root"]).parent.parent
        parsed = carriers_for(adapter, attempt, [0, ENDPOINT])
        evidence = m99.load_evidence(ATTRIBUTION, inventory, member, ordinal)
        rows.append(costs.row(ordinal, parsed[1], parsed[0]))
        shots[str(ordinal)] = {
            "parsed_presence_start": [float(v) for v in parsed[0][2::13][:18]],
            "parsed_presence_end": [float(v) for v in parsed[1][2::13][:18]],
            "engine_presence_start": [float(v) for v in evidence["engine_start"][2::13][:18]],
            "engine_presence_end": [float(v) for v in evidence["engine_end"][2::13][:18]]}
    return {"schema": SCHEMA_PERCEPTION, "plan_identity": IDENTITY,
            "cell": {"arm": arm, "inventory": inventory, "member": member},
            "ap_rows": rows, "shots": shots, "engine_seconds": 0}


# ---------------------------------------------------------------------------
# smoke (pre-freeze: nothing kept, no verdict joined)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    sources = t96.load_sources()
    members, universe, _ = t96.build_universe(sources)
    shots = training_shots(sources, universe)
    names = vocabulary()
    checks, timings = [], {}
    chosen = [shots[0], next(s for s in shots if s["source"] != "issue_77_n1")]
    began = time.monotonic()
    for shot in chosen:
        atomic_torch(shot_path(scratch, shot["key"]), build_shot(shot, names))
    timings["shot_build_seconds"] = (time.monotonic() - began) / len(chosen)
    built = [torch.load(shot_path(scratch, s["key"]), weights_only=True) for s in chosen]
    checks.append({"control": "shot_labels_and_images",
                   "shots": [{"key": b["key"], "end": b["end"], "images": list(b["images"].shape),
                              "pig_present_share": float(b["labels"]["presence"][:, names.index("pig:0000")].mean())}
                             for b in built],
                   "ok": all(b["images"].shape[1:] == (3, ENCODER["image_height"], ENCODER["image_width"])
                             for b in built)})
    began = time.monotonic()
    payload = train_encoder(scratch, [shot_path(scratch, s["key"]) for s in chosen], steps_limit=SMOKE_STEPS,
                            target=scratch / "encoder" / "encoder.pt")
    timings["encoder_seconds_per_step"] = (time.monotonic() - began) / SMOKE_STEPS
    checks.append({"control": "encoder_trains", "parameters": payload["parameters"],
                   "loss": payload["reports"][-1]["mean_loss"], "ok": math.isfinite(payload["reports"][-1]["mean_loss"])})
    model = new_encoder()
    model.load_state_dict(payload["model"])
    adapters = {"E": wrap(model), "R0": wlc.load_adapter(DEVICE)}
    began = time.monotonic()
    for arm, adapter in adapters.items():
        for shot, data in zip(chosen, built):
            atomic_torch(carrier_path(scratch, arm, shot["key"]),
                         {"carriers": carriers_for(adapter, shot["attempt"], list(range(data["end"] + 1)))})
    timings["carrier_seconds_per_shot"] = (time.monotonic() - began) / (2 * len(chosen))
    # R0 carriers of an inventory shot must replicate the #99 evidence parse (same parser)
    inventory_shot = chosen[1]
    key = inventory_shot["key"].removeprefix("inv--")
    inventory, member, ordinal = key.split("--")
    evidence = m99.load_evidence(ATTRIBUTION, inventory, member, int(ordinal.removeprefix("o")))
    r0 = torch.load(carrier_path(scratch, "R0", inventory_shot["key"]), weights_only=True)["carriers"]
    delta = float((r0[:ENDPOINT + 1] - evidence["parsed"][:ENDPOINT + 1]).abs().max())
    checks.append({"control": "R0_carriers_replicate_issue_99_evidence", "max_abs_delta": delta,
                   "tolerance": REPLICATION_TOLERANCE, "ok": delta <= REPLICATION_TOLERANCE})
    predictors = {}
    for family in FAMILIES:
        data = window_data(scratch, "E", chosen)
        began = time.monotonic()
        predictors[family], trained = train_predictor(data, family, SEEDS[0], None, steps=SMOKE_STEPS)
        timings[f"{family}_seconds_per_step"] = (time.monotonic() - began) / SMOKE_STEPS
        checks.append({"control": f"{family}_predictor_trains", "windows": trained["windows"],
                       "loss": trained["loss_trace"][-1]["loss"],
                       "ok": math.isfinite(trained["loss_trace"][-1]["loss"])})
    costs = m99.Costs(wlc.load_objective())
    held = sorted(m for m in universe[PRIMARY] if m in HELD_OUT)[0]
    began = time.monotonic()
    record = eval_record(universe, costs, adapters["E"], predictors, "E", PRIMARY, held, SEEDS[0])
    timings["eval_seconds_per_record"] = time.monotonic() - began
    checks.append({"control": "held_out_eval_mechanics", "member": held,
                   "requests": len(record["requests"]),
                   "finite": sum(not row["excluded"] for block in record["requests"].values()
                                 for row in block["rows"]),
                   "ok": len(record["requests"]) == len(REQUESTS)})
    fitted = {s["member"] for s in shots}
    checks.append({"control": "held_out_isolation", "training_members": sorted(fitted),
                   "ok": not (fitted & set(HELD_OUT))})
    projection = {
        "training_shots": len(shots),
        "shot_build_seconds": timings["shot_build_seconds"] * len(shots),
        "carrier_seconds": timings["carrier_seconds_per_shot"] * len(shots) * len(ARMS),
        "encoder_seconds": timings["encoder_seconds_per_step"] * ENCODER["epochs"] * len(shots)
        * (MAX_POSITION // IMAGE_STRIDE + 1) / ENCODER["batch_size"],
        "predictor_seconds": sum(timings[f"{f}_seconds_per_step"] for f in FAMILIES)
        * training_recipe()["steps"] * len(SEEDS) * len(ARMS),
        "eval_seconds": timings["eval_seconds_per_record"] * len(ARMS) * len(SEEDS) * 16}
    projection["total_seconds"] = sum(v for k, v in projection.items() if k != "training_shots")
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timings": timings, "projection": projection,
                "reading": "infrastructure assertions only; nothing kept; no verdict joined", "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected {projection['total_seconds'] / 3600:.2f} h "
        f"({json.dumps({k: round(v) for k, v in projection.items()})})")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_99_attribution_plan": ATTRIBUTION / "plan.json",
        "issue_99_attribution_compute": ATTRIBUTION / "compute.json",
        "issue_99_attribution_runner": ROOT / "scripts/run_decision_chain_attribution.py",
        "issue_96_plan": t96.OUTPUT / "plan.json",
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "issue_77_n1_plan": CAMPAIGN / "plan.json",
        "issue_70_parser_plan": ROOT / ".local-artifacts/issue-70-parser-repair-v6/plan.json",
        "model_module": ROOT / MODEL_MODULE,
        "smoke": Path(output) / "smoke.json",
    }
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at, smoke_evidence, shots, universe):
    attribution = read_json(ATTRIBUTION / "compute.json")
    gate = attribution["Q1_attribution"]["gate_a"]
    if gate["dominant"] != "perception":
        raise ValueError("the frozen Gate A does not implicate perception")
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal", "issue": 99,
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True, "issue_64_authorized": False,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "engine_seconds": 0,
        "gate_a": {"source": "issue-99-decision-chain-attribution-v1 compute.json Q1_attribution",
                   "dominant": gate["dominant"], "drops": gate["drops"],
                   "fix": attribution["Q1_attribution"]["fix"],
                   "q1_token": attribution["Q1_attribution"]["token"],
                   "q1_token_note": ("the attribution's Q1/Q2 tokens are readiness_or_precision_insufficient "
                                     "because its frozen G2 byte-identity guard failed on 4/859 decision "
                                     "frames; Gate A's dominant stage is computed independently of the "
                                     "guards and names the step-3 remedy")},
        "question": ("with the slot encoder replaced (the Gate-A remedy) and predictors retrained on its "
                     "carriers, does any request's predicted cost order held-out grid candidates with "
                     f"AUC >= {TARGET_AUC} and a lower bound > {TARGET_LOWER}; and how much of the change is "
                     "the encoder (E - R0)?"),
        "disclosure": ("Known before freeze: every #99 attribution table (all and held-out cohorts, "
                       "including the current checkpoint's held-out stage-c AUCs under the tie-free cost, "
                       "several of which already meet the target), the #96 and #109 tables. The encoder "
                       "architecture and every training setting below were fixed without any held-out "
                       "verdict join; the smoke joins no verdict."),
        "arms": {"E": ("new spatial slot parser trained on fit-lineage frames, predictors retrained on its "
                       "carriers"),
                 "R0": "frozen issue-70 parser, predictors retrained on its carriers (same data and recipe)",
                 "current": "frozen #77 N1 checkpoints (#99 attribution stage c), reference only"},
        "fit_data": {"members": fit_members(), "shots": [{k: s[k] for k in ("key", "member", "source")}
                                                          for s in shots],
                     "rule": ("every #77 N1 admissible branch of the #77 predictor/controller lineages plus "
                              "every verdict-bearing #93/#94/#87/#89 inventory execution of those lineages; "
                              "verdicts are never read by training"),
                     "positions": f"0..min(last, {MAX_POSITION}) (position k = fixed step 30000 + 50 k)"},
        "encoder": {"module": MODEL_MODULE, "architecture": "issue-99-spatial-slot-parser-v1", **ENCODER,
                    "image_stride": IMAGE_STRIDE,
                    "targets": ("issue-70 repair.targets on the frame's engine sample: presence, projected "
                                "centers, cohort-v2 micro relations, native macro labels"),
                    "loss": ("issue-70 repair.parser_loss with class-balanced pig/block presence weights "
                             "from the training frames"),
                    "adapter": ("issue-70 RepairedAdapter temperatures/thresholds and the 236-value "
                                "carrier; relations and macros never enter the carrier tensor"),
                    "selection": "last epoch; no validation split"},
        "predictors": {"families": list(FAMILIES), "seeds": list(SEEDS), "recipe": training_recipe(),
                       "loss": "run_issue_74_matched_dynamics.loss_for (#74/#77 recipe)",
                       "pairs": "PAIRS[step % 9]; continuous uses the same horizon",
                       "windows": {"starts": list(WINDOW_STARTS), "length": WINDOW_LENGTH,
                                   "padding": "repeat the last position (terminal absorption)"},
                       "initialization": "from scratch, torch.manual_seed(seed)",
                       "selection": "last update"},
        "evaluation": {
            "cohort": {"held_out": list(HELD_OUT)},
            "stage": ("c: the arm's parsed sealed anchor -> request rollout to 225 -> cost "
                      "(m99.rollout_endpoints); requests the 9 hybrid and 3 continuous fixed pairs"),
            "costs": m99.COST_DEFINITIONS, "primary_cost": PRIMARY_COST,
            "cells": "mixed held-out member x 3 seeds; inventories never pooled",
            "interval": "member-clustered percentile bootstrap, 10000 draws, PCG64 7201, DESCRIPTIVE",
            "perception": ("outcome-free: parser presence (>= 0.5) against engine presence per slot kind at "
                           "positions 0 and 225 of every held-out shot, per arm"),
            "endpoint_stage_ap": "the arm's parsed endpoint frame -> cost, held-out shots (diagnostic)",
        },
        "decision_rule": {
            "Q3_target": (f"supported iff guards pass AND some request of arm E on held-out grid cells "
                          f"under the tie-free cost has AUC >= {TARGET_AUC} with lower bound > {TARGET_LOWER}; "
                          f"not_supported_by_this_experiment iff guards pass and no request does; otherwise "
                          f"{STOP_TOKEN}"),
            "Q4_encoder_effect": (f"request-mean paired (E - R0) AUC on held-out grid cells, tie-free cost: "
                                  f"supported iff mean >= {DELTA} and lower > 0; not_supported_by_this_"
                                  f"experiment iff upper < {DELTA}; otherwise {STOP_TOKEN}"),
            "multiplicity": ("Q3 scans 12 requests against one target; the count of requests meeting it is "
                             "reported; held-out grid has 3 member clusters, whose percentile bootstrap "
                             "has 10 distinct resamples and understates uncertainty"),
        },
        "guards": {
            "G1": "no training shot belongs to a held-out member",
            "G2": ("encoder finished every epoch and all 12 predictors reached the frozen step count with "
                   "finite losses"),
            "G3": ("R0 parsed endpoint costs (stage ap) replicate the #99 attribution endpoint records within "
                   f"{REPLICATION_TOLERANCE}"),
            "G4": f"nonfinite share <= {NONFINITE_SHARE_MAX} on held-out grid cells for every arm and request",
            "caps": f"GPU seconds <= {GPU_CAP_SECONDS}, wall seconds <= {WALL_CAP_SECONDS}",
        },
        "universe": {"held_out_mixed": {i: sorted(m for m, e in universe[i].items()
                                                  if m in HELD_OUT and t96.mixed(e)) for i in INVENTORIES},
                     "eval_records": len(ARMS) * len(SEEDS) * sum(
                         1 for i in INVENTORIES for m in universe[i] if m in HELD_OUT),
                     "perception_records": len(ARMS) * sum(
                         1 for i in INVENTORIES for m in universe[i] if m in HELD_OUT)},
        "prohibited": ["reading any held-out verdict before publish", "changing the encoder, windows or "
                       "recipe after training starts", "retrying typed failures",
                       "re-opening any prior disposition (#15-#98, #109) or the #99 attribution"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("decision-only ranking of retained engine-verdicted executions on the 4 held-out "
                           "N1 members (never fit here or by #77; exposed through earlier tickets); a new "
                           "slot encoder and predictors trained on fit lineages of the same two families; "
                           "no engine access, rendering or capture; no closed-loop or competence claim; "
                           "every interval is DESCRIPTIVE"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-99 slot-encoder protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    blob = json_text({k: v for k, v in plan.items() if k not in ("smoke_evidence", "gate_a")})
    for marker in PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    for entry in plan["inputs"]:
        if entry["name"] != "smoke" and sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"frozen input {entry['name']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    for name in ("shots", "encoder", "carriers", "models", "records"):
        if (output / name).exists():
            raise ValueError(f"{name} exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    shots = training_shots(sources, universe)
    plan = frozen_plan(output, utc_now(), evidence, shots, universe)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {len(shots)} fit shots; {plan['universe']['eval_records']} eval and "
        f"{plan['universe']['perception_records']} perception records scheduled")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "status": "running",
            "phases": {}, "gpu_seconds_elapsed": 0.0, "wall_seconds_elapsed": 0.0}


def run(output):
    output = Path(output)
    plan = load_plan(output)
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    shots = training_shots(sources, universe)
    if [{k: s[k] for k in ("key", "member", "source")} for s in shots] != plan["fit_data"]["shots"]:
        raise ValueError("fit shots differ from the frozen plan")
    names = vocabulary()
    ledger = load_ledger(output)
    began, wall_base = time.monotonic(), ledger["wall_seconds_elapsed"]

    def mark(phase, seconds, gpu=0.0, **extra):
        ledger["phases"][phase] = {"seconds": ledger["phases"].get(phase, {}).get("seconds", 0.0) + seconds,
                                   **extra}
        ledger["gpu_seconds_elapsed"] += gpu
        ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
        if (ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS
                or ledger["wall_seconds_elapsed"] > WALL_CAP_SECONDS):
            ledger["status"] = "cap_exceeded"
            write_json(output / "ledger.json", ledger)
            raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")
        write_json(output / "ledger.json", ledger)

    # 1. shot data (outcome-free)
    phase_began = time.monotonic()
    pending = [s for s in shots if not shot_path(output, s["key"]).is_file()]
    with ProcessPoolExecutor(SHOT_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(build_and_save, str(output), shot, names) for shot in pending]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 25 == 0 or done == len(pending):
                elapsed = time.monotonic() - phase_began
                log(f"shots {done}/{len(pending)}; eta {elapsed / done * (len(pending) - done):.0f}s")
    mark("shots", time.monotonic() - phase_began, shots=len(shots))
    # 2. encoder
    if not (output / "encoder" / "encoder.pt").is_file():
        phase_began = time.monotonic()
        payload = train_encoder(output, [shot_path(output, s["key"]) for s in shots])
        mark("encoder", time.monotonic() - phase_began, gpu=payload["gpu_seconds"])
    # 3. carriers per arm
    for arm in ARMS:
        adapter = load_adapter(output, arm)
        phase_began = time.monotonic()
        pending = [s for s in shots if not carrier_path(output, arm, s["key"]).is_file()]
        for done, shot in enumerate(pending, 1):
            end = torch.load(shot_path(output, shot["key"]), weights_only=True)["end"]
            atomic_torch(carrier_path(output, arm, shot["key"]),
                         {"schema": SCHEMA_CARRIERS, "plan_identity": IDENTITY, "arm": arm,
                          "carriers": carriers_for(adapter, shot["attempt"], list(range(end + 1)))})
            if done % 25 == 0 or done == len(pending):
                elapsed = time.monotonic() - phase_began
                log(f"carriers {arm} {done}/{len(pending)}; eta {elapsed / done * (len(pending) - done):.0f}s")
        mark(f"carriers_{arm}", time.monotonic() - phase_began)
        del adapter
        torch.cuda.empty_cache()
    # 4. predictors
    for arm in ARMS:
        data = None
        for seed in SEEDS:
            for family in FAMILIES:
                target = predictor_path(output, arm, seed, family)
                if target.is_file():
                    continue
                if data is None:
                    data = window_data(output, arm, shots)
                    log(f"windows {arm}: {len(data['z'])}")
                phase_began = time.monotonic()
                _, payload = train_predictor(data, family, seed, target)
                log(f"predictor {arm} seed {seed} {family}: {payload['gpu_seconds']:.0f}s, final loss "
                    f"{payload['loss_trace'][-1]['loss']:.5f}")
                mark(f"predictor_{arm}_{seed}_{family}", time.monotonic() - phase_began,
                     gpu=payload["gpu_seconds"])
        del data
    # 5. evaluation and perception records (held-out)
    costs = m99.Costs(wlc.load_objective())
    for arm in ARMS:
        adapter = load_adapter(output, arm)
        phase_began = time.monotonic()
        gpu = 0.0
        for seed in SEEDS:
            models = {family: load_predictor(output, arm, seed, family) for family in FAMILIES}
            for inventory in INVENTORIES:
                for member in sorted(m for m in universe[inventory] if m in HELD_OUT):
                    path = eval_path(output, arm, inventory, member, seed)
                    if path.is_file():
                        continue
                    started = time.monotonic()
                    record = eval_record(universe, costs, adapter, models, arm, inventory, member, seed)
                    record["wall_seconds"] = time.monotonic() - started
                    gpu += sum(block["gpu_seconds"] for block in record["requests"].values())
                    write_json(path, record)
        for inventory in INVENTORIES:
            for member in sorted(m for m in universe[inventory] if m in HELD_OUT):
                path = perception_path(output, arm, inventory, member)
                if not path.is_file():
                    write_json(path, perception_record(sources, universe, costs, adapter, arm, inventory, member))
        mark(f"eval_{arm}", time.monotonic() - phase_began, gpu=gpu)
    ledger["status"] = "terminal"
    ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
    write_json(output / "ledger.json", ledger)
    log(f"run complete: gpu {ledger['gpu_seconds_elapsed']:.0f}s; wall {ledger['wall_seconds_elapsed']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def read_record(path, schema):
    record = read_json(path)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


def current_rows(inventory, member, seed, request):
    record = m99.read_record(m99.model_path(ATTRIBUTION, inventory, member, seed), m99.SCHEMA_MODEL)
    return record["stages"]["c"][request]["rows"]


def held_cells(output, universe, inventory):
    rows = []
    for member in sorted(m for m, e in universe[inventory].items() if m in HELD_OUT and t96.mixed(e)):
        entry = universe[inventory][member]
        for seed in SEEDS:
            views = {}
            for arm in ARMS:
                record = read_record(eval_path(output, arm, inventory, member, seed), SCHEMA_EVAL)
                for request in REQUESTS:
                    for cost in COSTS:
                        views[(cost, arm, request)] = m99.view(entry, record["requests"][request]["rows"], cost)
            for request in REQUESTS:
                for cost in COSTS:
                    views[(cost, "current", request)] = m99.view(entry, current_rows(inventory, member, seed,
                                                                                      request), cost)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def ap_cells(output, universe, inventory):
    rows = []
    for member in sorted(m for m, e in universe[inventory].items() if m in HELD_OUT and t96.mixed(e)):
        entry = universe[inventory][member]
        endpoint = m99.read_record(m99.endpoint_path(ATTRIBUTION, inventory, member), m99.SCHEMA_ENDPOINT)
        views = {}
        for cost in COSTS:
            views[(cost, "a", None)] = m99.view(entry, endpoint["stages"]["a"], cost)
            for arm in ARMS:
                record = read_record(perception_path(output, arm, inventory, member), SCHEMA_PERCEPTION)
                views[(cost, f"ap_{arm}", None)] = m99.view(entry, record["ap_rows"], cost)
        for seed in SEEDS:
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def perception_table(output, universe):
    names = vocabulary()
    kinds = sorted({n.split(":", 1)[0] for n in names})
    out = {}
    for arm in ARMS:
        tally = {when: {kind: {"slots": 0, "agree": 0, "engine_present": 0, "parsed_present": 0,
                               "abs_error": 0.0} for kind in kinds} for when in ("start", "end")}
        shots = 0
        for inventory in INVENTORIES:
            for member in sorted(m for m in universe[inventory] if m in HELD_OUT):
                record = read_record(perception_path(output, arm, inventory, member), SCHEMA_PERCEPTION)
                for shot in record["shots"].values():
                    shots += 1
                    for when in ("start", "end"):
                        for index, name in enumerate(names):
                            block = tally[when][name.split(":", 1)[0]]
                            p, e = shot[f"parsed_presence_{when}"][index], shot[f"engine_presence_{when}"][index]
                            block["slots"] += 1
                            block["agree"] += int((p >= 0.5) == (e >= 0.5))
                            block["engine_present"] += int(e >= 0.5)
                            block["parsed_present"] += int(p >= 0.5)
                            block["abs_error"] += abs(p - e)
        for when in tally.values():
            for block in when.values():
                block["accuracy"] = block["agree"] / block["slots"]
                block["mean_abs_presence_error"] = block.pop("abs_error") / block["slots"]
        out[arm] = {"shots": shots, **tally}
    return out


def ranker_block(rows, key):
    views = [r["views"][key] for r in rows]
    pairs = sum(v["pairs"] for v in views)
    chance = [v["chance_top1"] for v in views if v["chance_top1"] is not None]
    return {"auc": m99.block_of(rows, key),
            "tied_cells": sum(v["all_tied"] for v in views),
            "tied_share": sum(v["all_tied"] for v in views) / len(views) if views else None,
            "tied_pair_share": sum(v["tied_pairs"] for v in views) / pairs if pairs else None,
            "top1_hits": sum(1 for v in views if v["top1_hit"]),
            "chance_top1_mean": float(np.mean(chance)) if chance else None,
            "nonfinite_share": (sum(v["nonfinite"] for v in views)
                                / sum(v["verdict_slots"] for v in views)) if views else None}


def contrasts(rows, cost, first, second):
    n = len(REQUESTS)
    out = {"request_mean": m99.paired(rows, [(1 / n, (cost, first, r)) for r in REQUESTS]
                                      + [(-1 / n, (cost, second, r)) for r in REQUESTS])}
    for request in REQUESTS:
        out[request] = m99.paired(rows, [(1, (cost, first, request)), (-1, (cost, second, request))])
    return out


def compute_tables(output, plan):
    output = Path(output)
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    shots = training_shots(sources, universe)
    ledger = read_json(output / "ledger.json")
    per_inventory = {}
    for inventory in INVENTORIES:
        rows = held_cells(output, universe, inventory)
        endpoint_rows = ap_cells(output, universe, inventory)
        per_inventory[inventory] = {
            "mixed_members": sorted({r["member"] for r in rows}), "cells": len(rows),
            "rankers": {cost: {f"{arm}:{request}": ranker_block(rows, (cost, arm, request))
                               for arm in (*ARMS, "current") for request in REQUESTS} for cost in COSTS},
            "endpoint": {cost: {name: ranker_block(endpoint_rows, (cost, name, None))
                                for name in ("a", *(f"ap_{arm}" for arm in ARMS))} for cost in COSTS},
            "contrasts": {cost: {"E-R0": contrasts(rows, cost, "E", "R0"),
                                 "E-current": contrasts(rows, cost, "E", "current"),
                                 "R0-current": contrasts(rows, cost, "R0", "current")} for cost in COSTS},
        }
    grid = per_inventory[PRIMARY]
    # guards
    training_members = {s["member"] for s in shots}
    encoder = torch.load(output / "encoder" / "encoder.pt", map_location="cpu", weights_only=True)
    predictors = []
    for arm in ARMS:
        for seed in SEEDS:
            for family in FAMILIES:
                payload = torch.load(predictor_path(output, arm, seed, family), map_location="cpu",
                                     weights_only=True)
                predictors.append({"arm": arm, "seed": seed, "family": family, "steps": payload["steps"],
                                   "final_loss": payload["loss_trace"][-1]["loss"],
                                   "windows": payload["windows"], "parameters": payload["parameters"],
                                   "gpu_seconds": payload["gpu_seconds"]})
    replication = 0.0
    for inventory in INVENTORIES:
        for member in sorted(m for m in universe[inventory] if m in HELD_OUT):
            mine = read_record(perception_path(output, "R0", inventory, member), SCHEMA_PERCEPTION)["ap_rows"]
            theirs = m99.read_record(m99.endpoint_path(ATTRIBUTION, inventory, member),
                                     m99.SCHEMA_ENDPOINT)["stages"]["ap"]
            for a, b in zip(mine, theirs, strict=True):
                for cost in COSTS:
                    replication = max(replication, abs(a[cost] - b[cost]))
    worst_nonfinite = max(block["nonfinite_share"] for cost in COSTS
                          for block in grid["rankers"][cost].values())
    guards = {
        "G1": {"observed": {"training_members": sorted(training_members), "held_out": list(HELD_OUT)},
               "pass": not (training_members & set(HELD_OUT))},
        "G2": {"observed": {"encoder_epochs": len(encoder["reports"]), "encoder_complete": encoder["complete"],
                            "predictors": len(predictors),
                            "steps": sorted({p["steps"] for p in predictors})},
               "pass": (encoder["complete"] and len(encoder["reports"]) == ENCODER["epochs"]
                        and len(predictors) == len(ARMS) * len(SEEDS) * len(FAMILIES)
                        and all(p["steps"] == training_recipe()["steps"] and math.isfinite(p["final_loss"])
                                for p in predictors))},
        "G3": {"observed": {"max_abs_ap_cost_delta": replication, "tolerance": REPLICATION_TOLERANCE},
               "pass": replication <= REPLICATION_TOLERANCE},
        "G4": {"observed": {"worst_nonfinite_share": worst_nonfinite},
               "pass": worst_nonfinite <= NONFINITE_SHARE_MAX},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"],
                              "wall_seconds": ledger["wall_seconds_elapsed"], "status": ledger["status"]},
                 "pass": (ledger["status"] == "terminal" and ledger["gpu_seconds_elapsed"] <= GPU_CAP_SECONDS
                          and ledger["wall_seconds_elapsed"] <= WALL_CAP_SECONDS)},
    }
    guards_ok = all(g["pass"] for g in guards.values())
    target_rows = []
    for request in REQUESTS:
        block = grid["rankers"][PRIMARY_COST][f"E:{request}"]["auc"]
        target_rows.append({"request": request, "auc": block,
                            "meets": bool(block and block["mean"] >= TARGET_AUC
                                          and block["interval"][0] > TARGET_LOWER)})
    met = [row["request"] for row in target_rows if row["meets"]]
    q3 = STOP_TOKEN if not guards_ok else ("supported" if met else "not_supported_by_this_experiment")
    effect = grid["contrasts"][PRIMARY_COST]["E-R0"]["request_mean"]["estimate"]
    if not guards_ok or effect is None:
        q4 = STOP_TOKEN
    elif effect["mean"] >= DELTA and effect["interval"][0] > 0:
        q4 = "supported"
    elif effect["interval"][1] < DELTA:
        q4 = "not_supported_by_this_experiment"
    else:
        q4 = STOP_TOKEN
    reference = []
    for arm in ("R0", "current"):
        for request in REQUESTS:
            block = grid["rankers"][PRIMARY_COST][f"{arm}:{request}"]["auc"]
            if block and block["mean"] >= TARGET_AUC and block["interval"][0] > TARGET_LOWER:
                reference.append(f"{arm}:{request}")
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "structure": {"fit_shots": len(shots), "eval_records": plan["universe"]["eval_records"],
                      "perception_records": plan["universe"]["perception_records"]},
        "guards": guards, "guards_pass": guards_ok,
        "Q3_target": {"scope": "arm E, held-out grid, stage c, tie-free cost", "requests": target_rows,
                      "met_by": met, "token": q3, "reference_arms_meeting_target": reference},
        "Q4_encoder_effect": {"estimate": effect, "delta": DELTA, "token": q4},
        "perception": perception_table(output, universe),
        "inventories": per_inventory,
        "training": {"encoder": {"parameters": encoder["parameters"], "reports": encoder["reports"],
                                 "training_frames": encoder["training_frames"],
                                 "gpu_seconds": encoder["gpu_seconds"]},
                     "predictors": predictors},
        "compute": {"phases": ledger["phases"], "run_gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "run_wall_seconds": ledger["wall_seconds_elapsed"], "engine_seconds": 0},
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "inventory", "cost", "statistic", "value", "interval_low", "interval_high",
                     "interval_label", "cells", "members", "detail"])

    def put(identity, inventory, cost, statistic, value, block=None, cells="", members="", detail=""):
        writer.writerow([identity, inventory, cost, statistic, "" if value is None else value,
                         block["interval"][0] if block else "", block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", cells, members, detail])

    put("Q3_token", PRIMARY, PRIMARY_COST, "target token (arm E)", compute["Q3_target"]["token"],
        detail=json.dumps(compute["Q3_target"]["met_by"]))
    e = compute["Q4_encoder_effect"]["estimate"]
    put("Q4_token", PRIMARY, PRIMARY_COST, "encoder effect E - R0 (request-mean)", e and e["mean"], e,
        detail=compute["Q4_encoder_effect"]["token"])
    for guard, block in compute["guards"].items():
        put(f"guard_{guard}", "", "", f"{guard} pass", block["pass"],
            detail=json.dumps(block["observed"], sort_keys=True))
    for inventory, inv in compute["inventories"].items():
        for cost in COSTS:
            for name, block in {**inv["rankers"][cost], **inv["endpoint"][cost]}.items():
                auc = block["auc"]
                put(f"{inventory}_{cost}_auc_{name}", inventory, cost, f"AUC {name}", auc and auc["mean"], auc,
                    auc["units"] if auc else 0, auc["clusters"] if auc else 0,
                    json.dumps({k: block[k] for k in ("tied_share", "tied_pair_share", "top1_hits",
                                                      "chance_top1_mean")}, sort_keys=True))
            for name, parts in inv["contrasts"][cost].items():
                for request, c in parts.items():
                    put(f"{inventory}_{cost}_{name}_{request}", inventory, cost, f"{name} [{request}]",
                        c["estimate"] and c["estimate"]["mean"], c["estimate"], c["cells"], c["members"])
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    q3, q4 = compute["Q3_target"], compute["Q4_encoder_effect"]
    add("# Issue-99 step 3: slot-encoder fix, held-out ranking — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v1 frozen {plan['frozen_at']} before any training or scoring")
    add(f"- validation command: `{VALIDATION_COMMAND}`; zero engine seconds; held-out members "
        f"{', '.join(HELD_OUT)} only; every interval DESCRIPTIVE (member-clustered, 10000 draws, PCG64 7201)")
    add(f"- Gate A input: dominant {plan['gate_a']['dominant']} -> {plan['gate_a']['fix']}; "
        f"{plan['gate_a']['q1_token_note']}")
    add(f"- disclosure: {plan['disclosure']}")
    add("")
    add("## Dispositions")
    add("")
    add(f"- **Q3 target (arm E, held-out grid, tie-free, AUC >= {TARGET_AUC} with lower > {TARGET_LOWER}): "
        f"{q3['token']}**; requests meeting it: {', '.join(q3['met_by']) or 'none'} "
        f"({len(q3['met_by'])}/{len(REQUESTS)}); reference arms meeting it: "
        f"{', '.join(q3['reference_arms_meeting_target']) or 'none'}")
    add(f"- **Q4 encoder effect (request-mean E - R0, held-out grid, tie-free): "
        f"{fmt_interval(q4['estimate'])} -> {q4['token']}**")
    add(f"- multiplicity: {plan['decision_rule']['multiplicity']}")
    add("")
    add("## Guards")
    add("")
    add("| guard | pass | observed |")
    add("|---|---|---|")
    for guard, block in compute["guards"].items():
        add(f"| {guard} | {block['pass']} | `{json.dumps(block['observed'], sort_keys=True)}` |")
    add("")
    add("## Perception on held-out shots (outcome-free; presence >= 0.5 vs engine)")
    add("")
    add("| arm | position | kind | slots | accuracy | engine present | parsed present | mean abs error |")
    add("|---|---|---|---|---|---|---|---|")
    for arm, block in compute["perception"].items():
        for when in ("start", "end"):
            for kind, row in block[when].items():
                if kind in ("slingshot", "world"):
                    continue
                add(f"| {arm} ({block['shots']} shots) | {when} | {kind} | {row['slots']} | "
                    f"{fmt(row['accuracy'], 3)} | {row['engine_present']} | {row['parsed_present']} | "
                    f"{fmt(row['mean_abs_presence_error'], 3)} |")
    add("")
    for cost in COSTS:
        add(f"## Held-out stage-c AUC — cost {cost}")
        add("")
        add("| request | " + " | ".join(f"{i} {a}" for i in INVENTORIES for a in (*ARMS, "current")) + " |")
        add("|---|" + "---|" * (len(INVENTORIES) * (len(ARMS) + 1)))
        for request in REQUESTS:
            add(f"| {request} | " + " | ".join(
                fmt_interval(compute["inventories"][i]["rankers"][cost][f"{a}:{request}"]["auc"])
                for i in INVENTORIES for a in (*ARMS, "current")) + " |")
        add("")
        add(f"### Endpoint rankers (held-out) — cost {cost}")
        add("")
        add("| inventory | cells / members | a (engine endpoint) | ap E | ap R0 |")
        add("|---|---|---|---|---|")
        for inventory, inv in compute["inventories"].items():
            e = inv["endpoint"][cost]
            add(f"| {inventory} | {inv['cells']} / {len(inv['mixed_members'])} | {fmt_interval(e['a']['auc'])} | "
                f"{fmt_interval(e['ap_E']['auc'])} | {fmt_interval(e['ap_R0']['auc'])} |")
        add("")
        add(f"### Paired contrasts (request-mean) — cost {cost}")
        add("")
        add("| inventory | E - R0 | E - current | R0 - current |")
        add("|---|---|---|---|")
        for inventory, inv in compute["inventories"].items():
            c = inv["contrasts"][cost]
            add(f"| {inventory} | {fmt_interval(c['E-R0']['request_mean']['estimate'])} | "
                f"{fmt_interval(c['E-current']['request_mean']['estimate'])} | "
                f"{fmt_interval(c['R0-current']['request_mean']['estimate'])} |")
        add("")
        add(f"### Ties and top-1 (held-out grid) — cost {cost}")
        add("")
        add("| ranker | tied share | tied pair share | top-1 hits / cells | chance |")
        add("|---|---|---|---|---|")
        grid = compute["inventories"][PRIMARY]
        for name, block in grid["rankers"][cost].items():
            add(f"| {name} | {fmt(block['tied_share'], 3)} | {fmt(block['tied_pair_share'], 3)} | "
                f"{block['top1_hits']}/{grid['cells']} | {fmt(block['chance_top1_mean'], 3)} |")
        add("")
    t = compute["training"]
    add("## Training and compute")
    add("")
    add(f"- encoder: {t['encoder']['parameters']} parameters, {t['encoder']['training_frames']} training frames, "
        f"{len(t['encoder']['reports'])} epochs, final loss {t['encoder']['reports'][-1]['mean_loss']:.5f}, "
        f"{t['encoder']['gpu_seconds']:.0f} GPU s")
    add("")
    add("| arm | seed | family | steps | windows | parameters | final loss | GPU s |")
    add("|---|---|---|---|---|---|---|---|")
    for p in t["predictors"]:
        add(f"| {p['arm']} | {p['seed']} | {p['family']} | {p['steps']} | {p['windows']} | {p['parameters']} | "
            f"{p['final_loss']:.5f} | {p['gpu_seconds']:.0f} |")
    add("")
    c = compute["compute"]
    add("| phase | seconds |")
    add("|---|---|")
    for phase, block in c["phases"].items():
        add(f"| {phase} | {block['seconds']:.0f} |")
    add("")
    add(f"- run GPU {c['run_gpu_seconds']:.0f} s, wall {c['run_wall_seconds']:.0f} s (caps {GPU_CAP_SECONDS:.0f} / "
        f"{WALL_CAP_SECONDS:.0f} s); engine seconds 0")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ".")
    add("")
    return "\n".join(lines)


def summary(plan, compute):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "validation_command": VALIDATION_COMMAND, "question": plan["question"],
            "gate_a": plan["gate_a"], "disclosure": plan["disclosure"],
            "guards": compute["guards"], "guards_pass": compute["guards_pass"],
            "Q3_target": compute["Q3_target"], "Q4_encoder_effect": compute["Q4_encoder_effect"],
            "perception": compute["perception"], "inventories": compute["inventories"],
            "training": compute["training"], "compute": compute["compute"],
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False}


def rendered(plan, compute):
    return {"summary.json": json_text(summary(plan, compute)), "findings.md": findings_md(plan, compute),
            "comparisons.csv": comparisons_csv(compute)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    log(f"published: Q3 {compute['Q3_target']['token']} (met by {compute['Q3_target']['met_by']}); "
        f"Q4 {compute['Q4_encoder_effect']['token']}; guards pass {compute['guards_pass']}")
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
    log(f"validation passed: tables re-derived from retained eval/perception records, checkpoints and "
        f"#99 attribution records; byte-compared ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    shots = training_shots(sources, universe)
    by_source = {}
    for shot in shots:
        by_source[shot["source"]] = by_source.get(shot["source"], 0) + 1
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    log(f"  fit members {fit_members()}; fit shots {len(shots)} {by_source}")
    log(f"  held-out shots {len(held_out_shots(universe))}; mixed held-out members "
        f"{ {i: sorted(m for m, e in universe[i].items() if m in HELD_OUT and t96.mixed(e)) for i in INVENTORIES} }")
    log(f"  encoder {asdict(encoder_config())}; windows starts {WINDOW_STARTS} length {WINDOW_LENGTH}")
    log(f"  predictors {len(ARMS)} arms x {len(SEEDS)} seeds x {len(FAMILIES)} families; recipe {training_recipe()}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": (dry_run, False), "smoke": (smoke, True), "prepare": (prepare, False),
         "run": (run, True), "publish": (publish, False), "validate": (validate, False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    function, locked = MODES[mode]
    try:
        if locked:
            with wlc.GPULock():
                return function(args.output)
        return function(args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
