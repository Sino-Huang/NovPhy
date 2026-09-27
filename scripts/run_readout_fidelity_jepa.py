"""Issue-100: description levels the model can actually read.

Binding runner module: scripts/run_readout_fidelity_jepa.py
Exact validation command: python -u -m scripts.run_readout_fidelity_jepa --validate

Three work items, zero engine seconds (every frame and verdict is a retained capture):

1. Readout fidelity audit over the #77 N1 dynamics shard frames (fit lineages) and
   the same frame rule on the held-out lineages 007/008/014/016: per-slot presence
   and per-frame precision / recall / F1 of contact, supports, steady-state and
   structure-unstable against the engine labels, read from encoded carriers and
   after one Delta in {1, 5, 15} prediction, per phase (pre-contact, cascade,
   settled). Encoders: F77 (frozen issue-70 parser + #77 hybrid), E99 (frozen #99
   spatial slot parser + #99 E hybrid) and JEPA (below). Includes a reproduction
   of the unaudited 2026-09-25 manuscript probe numbers.
2. JEPA: context encoder + EMA target encoder (stop-gradient) + the unchanged
   request-conditioned CNNHybridPredictor, micro/macro heads as auxiliary losses,
   trained jointly on the #99 fit data with the #77 update budget; compared with
   the frozen-encoder carriers on rollout error (the tab:res:rollout protocol of
   #77's diagnostic) and readout F1.
3. The #96 within-checkpoint arms (F-Delta-alpha, J, M, T, D) on every encoder,
   reporting J - T and J - D per inventory.

Modes: --dry-run --smoke --prepare --run --publish --validate --gallery (frozen-protocol
chronology; the smoke keeps nothing and joins no verdict). Every interval is
DESCRIPTIVE (member/lineage-clustered percentile bootstrap, 10000 draws, PCG64 7201).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import html
import io
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts import run_decision_chain_attribution as m99
from scripts import run_issue_77_n1_diagnostic as diag
from scripts import run_launch_power_probe as p94
from scripts import run_lookahead_control as wlc
from scripts import run_slot_encoder_fix as s99
from scripts import run_tau_ad_within_checkpoint as t96

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-100-readout-fidelity-jepa-v1"
GALLERY = ROOT / "data/issue-100-readout-fidelity"
FRAME_CACHE = Path.home() / ".cache/novphy/issue-100-frame-cache"
DYNAMICS = t96.DYNAMICS
CAMPAIGN = s99.CAMPAIGN
DIAGNOSTIC = diag.OUTPUT
SLOT_FIX = s99.OUTPUT

IDENTITY = "issue-100-readout-fidelity-jepa-v1"
SCHEMA_PLAN = "issue_100_readout_fidelity_jepa_plan_v1"
SCHEMA_SMOKE = "issue_100_readout_fidelity_jepa_smoke_v1"
SCHEMA_LABELS = "issue_100_audit_labels_v1"
SCHEMA_CARRIERS = "issue_100_audit_carriers_v1"
SCHEMA_JEPA = "issue_100_jepa_checkpoint_v1"
SCHEMA_CONTROLLER = "issue_100_controller_v1"
SCHEMA_AUDIT = "issue_100_audit_record_v1"
SCHEMA_PROBE = "issue_100_probe_record_v1"
SCHEMA_ROLLOUT = "issue_100_rollout_record_v1"
SCHEMA_ARMS = "issue_100_arms_record_v1"
SCHEMA_FREQUENCIES = "issue_100_frequencies_v1"
SCHEMA_LEDGER = "issue_100_ledger_v1"
SCHEMA_COMPUTE = "issue_100_compute_v1"
SCHEMA_REPORT = "issue_100_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_readout_fidelity_jepa --validate"
RUNNER = "scripts/run_readout_fidelity_jepa.py"
MODEL_MODULE = "world_model/training/jepa_slot_encoder.py"

DEVICE = "cuda"
SEEDS = t96.SEEDS
HELD_OUT = m99.HELD_OUT
ENCODERS = ("F77", "E99", "JEPA")
ENCODER_DEFINITIONS = {
    "F77": "frozen issue-70 parser v6 (96x64) + frozen #77 N1 hybrid checkpoints and controllers",
    "E99": ("frozen #99 spatial slot parser (320x240, issue-99-slot-encoder-fix-v1 arm E) + the #99 E "
            "hybrid predictors; controllers trained here with the #77 controller recipe"),
    "JEPA": ("jointly trained here: context encoder (spatial slot parser architecture) + EMA target "
             "encoder (stop-gradient) + CNNHybridPredictor, micro/macro heads and the issue-70 parser "
             "loss as auxiliary losses; controllers trained here with the #77 controller recipe"),
}
DELTAS = t96.DELTAS
ALPHAS = t96.ALPHAS
SOURCES = ("encoder", *(f"{d}-{a}" for d in DELTAS for a in ALPHAS))
PHASES = ("pre_contact", "cascade", "settled")
PRESENCE_KINDS = ("bird", "block", "pig", "platform", "slingshot", "world")
PREDICATES = ("contact", "supports", "steady-state", "structure-unstable")
MICRO = ("contact", "supports")
MACRO = ("steady-state", "structure-unstable")
ALL_PREDICATES = (*(f"presence:{k}" for k in PRESENCE_KINDS), *PREDICATES)
MATCHED_ALPHA = {"presence": "continuous", "contact": "micro", "supports": "micro",
                 "steady-state": "macro", "structure-unstable": "macro"}
COHORTS = ("fit", "held_out")
READ_THRESHOLD = 0.5
MACRO_F1_TARGET = 0.7
MICRO_F1_TARGET = 0.5
LABEL_POSITIVES_MIN = 20

ARMS = (*t96.FIXED_ARMS, "J", "M", "T", "D")
PHASE_1_ARMS = (*t96.FIXED_ARMS, "J", "M")
PHASE_2_ARMS = ("T", "D")
CONTROLLER_ARMS = ("J", "M", "T", "D")
ARM_COHORTS = m99.COHORTS
COSTS = m99.COSTS
PRIMARY_COST = m99.PRIMARY_COST
PRIMARY_ARM_COHORT = "held_out"
DELTA = t96.DELTA
INVENTORIES = t96.INVENTORIES
ENDPOINT = t96.ENDPOINT
ROLLOUT_TIMES = diag.TIMES
ROLLOUT_ENDPOINT = 225
TAB_SYSTEMS = {"hybrid_continuous_h1": "1-continuous", "hybrid_continuous_h5": "5-continuous",
               "hybrid_continuous_h15": "15-continuous", "hybrid_micro_h15": "15-micro",
               "hybrid_macro_h15": "15-macro"}

JEPA = {"steps": 9000, "batch_size": 64, "predictor_learning_rate": 1e-4, "encoder_learning_rate": 1e-3,
        "weight_decay": 1e-4, "grad_clip": 1.0, "ema_start": 0.996, "ema_end": 1.0,
        "parser_loss_weight": 1.0, "head_loss_weight": 0.1, "unroll": 4,
        "window_starts": list(s99.WINDOW_STARTS), "window_length": s99.WINDOW_LENGTH,
        "max_position": s99.MAX_POSITION, "image_height": 240, "image_width": 320,
        "progress_every": 500}
CONTROLLER = {"rounds": 2, "steps_per_round": 1800, "batch_size": 128, "learning_rate": 1e-3,
              "compute_weight": 1e-4, "window": "first window (start 0, 60 transitions) of every fit shot "
                                               "of the #77 controller lineages"}

PROBE = {"capture": "issue-77-n1-001-a06", "encoder": "F77", "seed": 20260908, "frame": 80,
         "frames": 555, "claims": {
             "C1_bird_presence_frame80_below_0.5": "every bird slot's carrier presence < 0.5 at frame 80",
             "C2_pig_presence_frame80_below_0.5": "the pig slot's carrier presence < 0.5 at frame 80",
             "C3_max_gated_micro_555_frames": 0.059,
             "C4_macro_steady_frame80": 0.188,
             "C5_macro_unstable_frame80": 0.041,
             "C6_lineages_with_gated_micro_edge_above_0.5": ["issue-77-n1-009"],
             "C7_lineage_009_positions_mislocalised": "mean engine-present slot center error > 0.05 on the "
                                                      "lineage-009 frames with a gated edge > 0.5"},
         "numeric_tolerance": 0.0005, "mislocalisation_threshold": 0.05}

PARSE_BATCH = 64
DECODE_THREADS = 12
LABEL_WORKERS = 8
PARSE_TOLERANCE = 1e-3
CARRIER_BUILD_TOLERANCE = 1e-4
ROLLOUT_RELATIVE_TOLERANCE = 1e-3
COST_TOLERANCE = t96.COST_TOLERANCE
GPU_CAP_SECONDS = 10 * 3600.0
WALL_CAP_SECONDS = 20 * 3600.0
STOP_TOKEN = t96.STOP_TOKEN
SMOKE_STEPS = 20
PLACEHOLDER_MARKERS = m99.PLACEHOLDER_MARKERS
BOOTSTRAP_DRAWS = t96.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = t96.BOOTSTRAP_SEED
INTERVAL_QUANTILES = t96.INTERVAL_QUANTILES

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
fmt = t96.fmt
fmt_interval = t96.fmt_interval
atomic_torch = s99.atomic_torch


def log(message):
    print(f"[issue-100-readout-fidelity] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def vocabulary():
    return s99.vocabulary()


def plan87():
    return read_json(wlc.EIGHTY_SEVEN / "plan.json")


class Progress:
    """Foreground progress with ETA."""

    def __init__(self, label, total):
        self.label, self.total, self.done, self.began = label, total, 0, time.monotonic()

    def step(self, every=1):
        self.done += 1
        if self.done % every == 0 or self.done == self.total:
            elapsed = time.monotonic() - self.began
            eta = elapsed / self.done * (self.total - self.done)
            log(f"{self.label} {self.done}/{self.total}; elapsed {elapsed:.0f}s; eta {eta:.0f}s")


# ---------------------------------------------------------------------------
# membership (no statistic)
# ---------------------------------------------------------------------------

def audit_shots():
    """Every coverage-admissible #77 N1 branch of the 16 lineages, frozen order."""
    plan = read_json(CAMPAIGN / "plan.json")
    coverage = read_json(CAMPAIGN / "coverage.json")
    shots = []
    for member in sorted(plan["members"], key=lambda m: m["ordinal"]):
        identity = member["identity"]
        for branch in sorted((b for b in plan["branches"] if b["source_member_identity"] == identity),
                             key=lambda b: b["candidate_ordinal"]):
            if coverage["branches"][branch["identity"]]["status"] != "admissible":
                continue
            shots.append({"branch": branch["identity"], "member": identity,
                          "cohort": "held_out" if identity in HELD_OUT else "fit",
                          "attempt": str(CAMPAIGN / "attempts" / branch["identity"])})
    fit = read_json(DYNAMICS / "plan.json")["n1_lineages"]["records"]
    for member in {s["member"] for s in shots if s["cohort"] == "fit"} | set(fit):
        mine = sorted(s["branch"] for s in shots if s["member"] == member)
        if member not in fit or sorted(fit[member]["branches"]) != mine:
            raise ValueError(f"audit branches of {member} differ from the #77 shard membership")
    held = sorted({m["identity"] for m in plan["members"] if m["study_role"] == "held_out_evaluation"})
    if tuple(held) != HELD_OUT:
        raise ValueError("held-out lineages differ from the declared cohort")
    return shots


def shard_positions(last):
    """The #77 build_n1_shot window rows: starts {round(i (last - 60) / 3)}, 60 transitions."""
    starts = sorted({round(i * max(0, last - 60) / 3) for i in range(4)})
    return sorted({p for start in starts for p in range(start, start + min(60, last - start) + 1)})


def controller_lineages():
    return sorted(read_json(DYNAMICS / "plan.json")["n1_lineages"]["controller"])


def fit_data():
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    return sources, universe, s99.training_shots(sources, universe)


def rollout_samples():
    return [s for s in read_json(DIAGNOSTIC / "plan.json")["samples"] if s["sources"]]


# ---------------------------------------------------------------------------
# engine labels and phases of the audit shots
# ---------------------------------------------------------------------------

def label_path(output, branch):
    return Path(output) / "audit" / "labels" / f"{branch}.pt"


def phase_segmentation(relations, macros, macro_mask, names):
    """Engine-only phases: pre-contact before the first bird contact with any non-bird,
    non-slingshot slot; settled from the first later frame after which steady-state
    stays true to the end; cascade in between. Returns (per-position phase, onset, settled)."""
    birds = [i for i, n in enumerate(names) if n.startswith("bird:")]
    others = [i for i, n in enumerate(names) if not n.startswith(("bird:", "slingshot:"))]
    touching = relations[:, birds][:, :, others, 0].flatten(1).any(1)
    count = len(touching)
    onset = int(touching.nonzero()[0]) if bool(touching.any()) else None
    steady = (macros[:, 0] & macro_mask[:, 0]).tolist()
    settled = None
    if onset is not None:
        tail = True
        for p in range(count - 1, onset - 1, -1):
            tail = tail and steady[p]
            if tail:
                settled = p
    phase = torch.zeros(count, dtype=torch.long)
    if onset is not None:
        phase[onset:] = 1
        if settled is not None:
            phase[settled:] = 2
    return phase, onset, settled


def build_labels(shot, names):
    _, _, frames, _ = s99.shot_frames(shot["attempt"])
    segment = read_json(CAMPAIGN / "results" / f"{shot['branch']}.json")["segments"][0]
    positions = list(range(len(frames)))
    labels = s99.shot_labels(segment, frames, positions, names)
    relations = labels["relations"].bool()
    macros, macro_mask = labels["macros"].bool(), labels["macro_mask"].bool()
    phase, onset, settled = phase_segmentation(relations, macros, macro_mask, names)
    elapsed = [0.0] + [frames[p]["fixed_time_seconds"] - frames[p - 1]["fixed_time_seconds"]
                       for p in range(1, len(frames))]
    return {"schema": SCHEMA_LABELS, "plan_identity": IDENTITY, "branch": shot["branch"],
            "member": shot["member"], "frames": len(frames), "action": s99.action_of(segment),
            "presence": labels["presence"].bool(), "centers": labels["centers"].float(),
            "relations": relations, "relation_mask": labels["relation_mask"].bool(),
            "macros": macros, "macro_mask": macro_mask, "phase": phase,
            "contact_onset": onset, "settled_onset": settled,
            "elapsed": torch.tensor(elapsed, dtype=torch.float64)}


def build_labels_and_save(output, shot, names):
    torch.set_num_threads(1)
    atomic_torch(label_path(output, shot["branch"]), build_labels(shot, names))
    return shot["branch"]


def load_labels(output, branch):
    value = torch.load(label_path(output, branch), weights_only=True)
    if value["schema"] != SCHEMA_LABELS or value["branch"] != branch:
        raise ValueError(f"audit labels of {branch} binding differs")
    return value


# ---------------------------------------------------------------------------
# encoders, adapters and whole-shot carriers
# ---------------------------------------------------------------------------

def encoder_tags(encoder):
    return [f"JEPA-seed{seed}" for seed in SEEDS] if encoder == "JEPA" else [encoder]


def tag_of(encoder, seed):
    return f"JEPA-seed{seed}" if encoder == "JEPA" else encoder


def jepa_path(output, seed):
    return Path(output) / "models" / "JEPA" / f"seed-{seed}" / "jepa.pt"


def load_jepa(output, seed):
    from world_model.training.cnn_hybrid import CNNHybridPredictor
    payload = torch.load(jepa_path(output, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_JEPA or payload.get("plan_identity") != IDENTITY
            or payload["seed"] != seed or payload["steps"] != JEPA["steps"]):
        raise ValueError(f"JEPA checkpoint seed {seed} binding differs")
    encoder = s99.new_encoder()
    encoder.load_state_dict(payload["encoder"], strict=True)
    predictor = CNNHybridPredictor()
    predictor.load_state_dict(payload["predictor"], strict=True)
    return encoder.to(DEVICE).eval(), predictor.to(DEVICE).eval()


def wrap(model, identity):
    from scripts import run_issue_70_parser_repair as repair
    plan70 = repair.load_plan(repair.ROOT)
    return repair.RepairedAdapter(model.to(DEVICE).eval(), parser_checkpoint_identity=identity,
                                  temperatures=plan70["temperatures"], thresholds=plan70["thresholds"],
                                  object_kind_temperature=1.0, **repair.carrier_dimensions(plan70))


def load_adapter(output, tag):
    if tag == "F77":
        return wlc.load_adapter(DEVICE)
    if tag == "E99":
        return s99.load_adapter(SLOT_FIX, "E")
    seed = int(tag.removeprefix("JEPA-seed"))
    encoder, _ = load_jepa(output, seed)
    return wrap(encoder, f"{IDENTITY}:JEPA:seed-{seed}")


def load_hybrid(output, encoder, seed):
    if encoder == "F77":
        return wlc.load_predictor(plan87(), seed, "hybrid", DEVICE)
    if encoder == "E99":
        return s99.load_predictor(SLOT_FIX, "E", seed, "hybrid")
    return load_jepa(output, seed)[1]


def parse_images(adapter, images):
    """adapter.parse_batch post-processing on pre-resized uint8 images (B, 3, H, W)."""
    t = adapter.temperatures
    with torch.no_grad():
        output = adapter.model(images.to(next(adapter.model.parameters()).device))
    relation_t = torch.tensor((t["contact"], t["supports"]), device=output["relation_logits"].device)
    macro_t = torch.tensor((t["steady-state"], t["structure-unstable"]), device=output["macro_logits"].device)
    presence = torch.sigmoid(output["presence_logits"] / t["object_presence"]).cpu()
    kinds = torch.softmax(output["kind_logits"] / adapter.object_kind_temperature, dim=-1).cpu()
    relations = torch.sigmoid(output["relation_logits"] / relation_t).cpu()
    macros = torch.sigmoid(output["macro_logits"] / macro_t).cpu()
    centers = output["centers"].cpu()
    return [{"presence": presence[i], "centers": centers[i], "kinds": kinds[i], "relations": relations[i],
             "macros": macros[i]} for i in range(len(images))]


def decode(path, sizes):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {size: torch.from_numpy(np.asarray(rgb.resize(size, Image.Resampling.BILINEAR),
                                                  dtype=np.uint8).copy()).permute(2, 0, 1)
                for size in sizes}


def shot_carriers(adapters, attempt, pool):
    """Every position's carrier (prior = position - 1) for every adapter; one PNG decode."""
    from world_model.data.deployment_temporal import AgentObservation
    _, observation_root, frames, _ = s99.shot_frames(attempt)
    sizes = sorted({(a.model.config.image_width, a.model.config.image_height) for a in adapters.values()})
    parsed = {tag: [] for tag in adapters}
    for begin in range(0, len(frames), PARSE_BATCH):
        chunk = list(range(begin, min(begin + PARSE_BATCH, len(frames))))
        images = list(pool.map(lambda p: decode(observation_root / frames[p]["agent_observation"]["relative_path"],
                                                sizes), chunk))
        for tag, adapter in adapters.items():
            size = (adapter.model.config.image_width, adapter.model.config.image_height)
            parsed[tag].extend(parse_images(adapter, torch.stack([image[size] for image in images])))

    def light(p):
        ref = frames[p]["agent_observation"]
        return AgentObservation(ref["identity"], frames[p]["fixed_step"], frames[p]["fixed_time_seconds"],
                                b"", "agent")

    out = {}
    for tag, adapter in adapters.items():
        rows = parsed[tag]
        out[tag] = torch.stack([m99.build(adapter, None if p == 0 else light(p - 1), light(p), rows[p],
                                          None if p == 0 else rows[p - 1]) for p in range(len(frames))])
    return out


def carrier_path(output, tag, branch):
    return Path(output) / "audit" / "carriers" / tag / f"{branch}.pt"


def load_carriers(output, tag, branch):
    value = torch.load(carrier_path(output, tag, branch), weights_only=True)
    if value["schema"] != SCHEMA_CARRIERS or value["tag"] != tag or value["branch"] != branch:
        raise ValueError(f"carriers {tag}/{branch} binding differ")
    return value["carriers"]


def context_carrier(adapter, attempt):
    """Batch-1 parse of frame 0 (the #77 diagnostic context rule)."""
    from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
    _, observation_root, frames, _ = s99.shot_frames(attempt)
    frame = frames[0]
    observation = AgentObservation(frame["agent_observation"]["identity"], frame["fixed_step"],
                                   frame["fixed_time_seconds"],
                                   (observation_root / frame["agent_observation"]["relative_path"]).read_bytes(),
                                   "agent")
    parsed = adapter.parse_batch((observation,))[0]
    return adapter.build_from_parsed(TemporalObservationContext(None, observation), parsed, None).tensor


# ---------------------------------------------------------------------------
# JEPA training data: windows, labels, frame cache
# ---------------------------------------------------------------------------

def fit_windows(shots_data):
    """#99 windows: starts 0..225 step 15 below the shot end, 60 transitions, pad = repeat last."""
    rows = []
    for index, shot in enumerate(shots_data):
        end = shot["end"]
        for start in JEPA["window_starts"]:
            if start >= end:
                continue
            length = min(JEPA["window_length"], end - start)
            rows.append({"shot": index, "start": start, "length": length,
                         "index": [min(start + i, start + length) for i in range(JEPA["window_length"] + 1)]})
    return rows


def needed_positions(windows_list, shot_count):
    needed = [set() for _ in range(shot_count)]
    for window in windows_list:
        for delta in DELTAS:
            for j in range(JEPA["unroll"] + 1):
                p = window["index"][j * delta]
                needed[window["shot"]].add(p)
                if p > 0:
                    needed[window["shot"]].add(p - 1)
    return [sorted(s) for s in needed]


def frame_index_path(output):
    return Path(output) / "jepa" / "frame-index.pt"


def window_data_path(output):
    return Path(output) / "jepa" / "windows.pt"


def build_window_data(output, shots):
    """Outcome-free labels of the #99 shot cache arranged as windows (JEPA training)."""
    meta, rows = [], []
    for shot in shots:
        value = torch.load(s99.shot_path(SLOT_FIX, shot["key"]), weights_only=True)
        if value["key"] != shot["key"] or value["schema"] != s99.SCHEMA_SHOT:
            raise ValueError(f"#99 shot cache {shot['key']} binding differs")
        _, _, frames, _ = s99.shot_frames(shot["attempt"])
        elapsed = [0.0] + [frames[p]["fixed_time_seconds"] - frames[p - 1]["fixed_time_seconds"]
                           for p in range(1, value["end"] + 1)]
        meta.append({"key": shot["key"], "member": shot["member"], "attempt": shot["attempt"],
                     "end": value["end"], "elapsed": elapsed})
        rows.append({"labels": {k: v for k, v in value["labels"].items()}, "action": value["action"]})
    windows_list = fit_windows(meta)
    tensors = {"shot": torch.tensor([w["shot"] for w in windows_list]),
               "index": torch.tensor([w["index"] for w in windows_list]),
               "length": torch.tensor([w["length"] for w in windows_list]),
               "action": torch.stack([rows[w["shot"]]["action"] for w in windows_list])}
    for key, name, dtype in (("relations", "relations", torch.bool), ("relation_mask", "relations_mask", torch.bool),
                             ("macros", "macros", torch.bool), ("macro_mask", "macros_mask", torch.bool),
                             ("presence", "presence", torch.float32), ("centers", "centers", torch.float32)):
        tensors[name] = torch.stack([rows[w["shot"]]["labels"][key][w["index"]].to(dtype) for w in windows_list])
    atomic_torch(window_data_path(output), {"schema": "issue_100_jepa_windows_v1", "plan_identity": IDENTITY,
                                            "shots": meta, "tensors": tensors})
    return meta, windows_list


def build_frame_cache(output, meta, windows_list, cache=FRAME_CACHE):
    needed = needed_positions(windows_list, len(meta))
    total = sum(len(n) for n in needed)
    cache.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(cache / "frames.npy", mode="w+", dtype=np.uint8,
                                      shape=(total, 3, JEPA["image_height"], JEPA["image_width"]))
    row_of = torch.full((len(meta), JEPA["max_position"] + 1), -1, dtype=torch.long)
    jobs, row = [], 0
    for index, (shot, positions) in enumerate(zip(meta, needed)):
        _, observation_root, frames, _ = s99.shot_frames(shot["attempt"])
        for p in positions:
            jobs.append((row, observation_root / frames[p]["agent_observation"]["relative_path"]))
            row_of[index, p] = row
            row += 1
    size = (JEPA["image_width"], JEPA["image_height"])
    progress = Progress("frame cache", len(jobs))

    def put(job):
        array[job[0]] = decode(job[1], [size])[size].numpy()

    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        for _ in pool.map(put, jobs):
            progress.step(5000)
    array.flush()
    atomic_torch(frame_index_path(output), {"schema": "issue_100_frame_index_v1", "plan_identity": IDENTITY,
                                            "cache": str(cache / "frames.npy"), "rows": total,
                                            "row_of": row_of, "keys": [m["key"] for m in meta]})
    return total


class JepaData:
    def __init__(self, output):
        windows_value = torch.load(window_data_path(output), weights_only=True)
        index_value = torch.load(frame_index_path(output), weights_only=True)
        if windows_value["plan_identity"] != IDENTITY or index_value["plan_identity"] != IDENTITY:
            raise ValueError("JEPA training data binding differs")
        self.meta = windows_value["shots"]
        self.tensors = windows_value["tensors"]
        self.row_of = index_value["row_of"]
        width = JEPA["max_position"] + 1
        self.elapsed = torch.zeros(len(self.meta), width, dtype=torch.float32)
        for i, shot in enumerate(self.meta):
            self.elapsed[i, :len(shot["elapsed"])] = torch.tensor(shot["elapsed"], dtype=torch.float32)
        self.frames = np.load(index_value["cache"], mmap_mode="r")
        if len(self.frames) != index_value["rows"]:
            raise ValueError("frame cache size differs from its index")

    def __len__(self):
        return len(self.tensors["shot"])

    def batch(self, indices, delta):
        """CPU tensors for one step: images of rows 0, D, .., 4D and their priors."""
        rows = [j * delta for j in range(JEPA["unroll"] + 1)]
        shot = self.tensors["shot"][indices]
        positions = self.tensors["index"][indices][:, rows]
        prior = (positions - 1).clamp(min=0)
        current_rows = self.row_of[shot[:, None], positions]
        prior_rows = self.row_of[shot[:, None], prior]
        if bool((current_rows < 0).any()) or bool((prior_rows < 0).any()):
            raise ValueError("frame cache misses a needed position")
        flat = torch.cat((current_rows.flatten(), prior_rows.flatten())).numpy()
        unique, inverse = np.unique(flat, return_inverse=True)
        return {"images": torch.from_numpy(np.ascontiguousarray(self.frames[unique])).pin_memory(),
                "inverse": torch.from_numpy(inverse.reshape(-1)),
                "has_prior": (positions > 0).float(),
                "elapsed": self.elapsed[shot[:, None], positions],
                "rows": rows,
                **{k: self.tensors[k][indices] for k in ("action", "length", "relations", "relations_mask",
                                                         "macros", "macros_mask", "presence", "centers")}}


def presence_weights():
    payload = torch.load(SLOT_FIX / "encoder" / "encoder.pt", map_location="cpu", weights_only=True)
    return torch.tensor(payload["presence_weights"])


def train_jepa(output, data, seed, target_path=None, steps=None):
    """Joint context encoder + predictor training; EMA target encoder; resumable."""
    from scripts.run_issue_70_parser_repair import parser_loss
    from world_model.training import jepa_slot_encoder as js
    from world_model.training.cnn_hybrid import CNNHybridPredictor, PAIRS
    total = steps or JEPA["steps"]
    names = vocabulary()
    expected = js.expected_kind_indices(names)
    weights = presence_weights().to(DEVICE)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    encoder = s99.new_encoder().to(DEVICE)
    predictor = CNNHybridPredictor().to(DEVICE)
    target = js.new_target(encoder)
    optimizer = torch.optim.AdamW(
        [{"params": list(encoder.parameters()), "lr": JEPA["encoder_learning_rate"]},
         {"params": list(predictor.parameters()), "lr": JEPA["predictor_learning_rate"]}],
        weight_decay=JEPA["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    first, trace, counts, gpu = 0, [], {}, 0.0
    progress = None if target_path is None else Path(target_path).with_name("jepa-progress.pt")
    if progress is not None and progress.is_file():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        encoder.load_state_dict(saved["encoder"])
        predictor.load_state_dict(saved["predictor"])
        target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        first, trace, counts, gpu = saved["step"], saved["trace"], saved["pair_counts"], saved["gpu_seconds"]
        log(f"JEPA seed {seed}: resumed at step {first}")
    encoder.train()
    predictor.train()
    pool = ThreadPoolExecutor(1)

    def fetch(step):
        indices = torch.randint(len(data), (JEPA["batch_size"],), generator=generator)
        return pool.submit(data.batch, indices, PAIRS[step % 9].delta)

    pending = fetch(first) if first < total else None
    began = time.monotonic()
    window_loss = {"jepa": 0.0, "parser": 0.0, "n": 0}
    for step in range(first, total):
        batch = pending.result()
        resume_state = generator.get_state()  # before the prefetch draws step + 1
        pending = fetch(step + 1) if step + 1 < total else None
        pair = PAIRS[step % 9]
        size, rows = JEPA["batch_size"], batch["rows"]
        gpu_batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if k != "rows"}
        k = JEPA["unroll"]
        images = gpu_batch["images"][gpu_batch["inverse"]]
        count = size * len(rows)
        current = images[:count].reshape(size, len(rows), *images.shape[1:])
        prior = images[count:].reshape(size, len(rows), *images.shape[1:])
        online_current = current[:, :k].reshape(-1, *current.shape[2:])
        online_prior = prior[:, :k].reshape(-1, *prior.shape[2:])
        context, outputs = js.encode_carriers(encoder, online_current, online_prior,
                                              gpu_batch["has_prior"][:, :k].reshape(-1),
                                              gpu_batch["elapsed"][:, :k].reshape(-1), expected)
        with torch.no_grad():
            targets, _ = js.encode_carriers(target, current[:, 1:].reshape(-1, *current.shape[2:]),
                                            prior[:, 1:].reshape(-1, *prior.shape[2:]),
                                            gpu_batch["has_prior"][:, 1:].reshape(-1),
                                            gpu_batch["elapsed"][:, 1:].reshape(-1), expected)
        context = context.reshape(size, k, -1)
        targets = targets.reshape(size, k, -1)
        labels = {name: gpu_batch[name] for name in ("relations", "relations_mask", "macros", "macros_mask")}
        loss_jepa = js.jepa_loss(predictor, [context[:, j] for j in range(k)],
                                 [targets[:, j] for j in range(k)], gpu_batch["action"], gpu_batch["length"],
                                 labels, pair, rows, JEPA["head_loss_weight"])
        at = torch.tensor(rows[:k], device=DEVICE)
        parser_batch = {"images": None,
                        "presence": gpu_batch["presence"][:, at].reshape(-1, len(names)),
                        "centers": gpu_batch["centers"][:, at].reshape(-1, len(names), 2),
                        "relations": gpu_batch["relations"][:, at].reshape(-1, len(names), len(names), 2).float(),
                        "relation_mask": gpu_batch["relations_mask"][:, at].reshape(-1, len(names), len(names), 2),
                        "macros": gpu_batch["macros"][:, at].reshape(-1, 2).float(),
                        "macro_mask": gpu_batch["macros_mask"][:, at].reshape(-1, 2)}
        loss_parser = parser_loss(lambda _images: outputs, parser_batch, names, weights)
        loss = loss_jepa + JEPA["parser_loss_weight"] * loss_parser
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite JEPA loss seed {seed} step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), JEPA["grad_clip"])
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), JEPA["grad_clip"])
        optimizer.step()
        js.ema_update(target, encoder, js.momentum_at(step + 1, total, JEPA["ema_start"], JEPA["ema_end"]))
        counts[str(pair.identity)] = counts.get(str(pair.identity), 0) + 1
        window_loss["jepa"] += float(loss_jepa.detach())
        window_loss["parser"] += float(loss_parser.detach())
        window_loss["n"] += 1
        if (step + 1) % 90 == 0 or step + 1 == total:
            trace.append({"step": step + 1, "jepa_loss": window_loss["jepa"] / window_loss["n"],
                          "parser_loss": window_loss["parser"] / window_loss["n"]})
            window_loss = {"jepa": 0.0, "parser": 0.0, "n": 0}
        if (step + 1) % JEPA["progress_every"] == 0 or step + 1 == total:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            log(f"JEPA seed {seed} step {step + 1}/{total}: jepa {trace[-1]['jepa_loss']:.5f} parser "
                f"{trace[-1]['parser_loss']:.4f}; {elapsed:.0f}s; eta {elapsed / (step + 1 - first) * (total - step - 1):.0f}s")
            if progress is not None and step + 1 < total:
                atomic_torch(progress, {"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                                        "target": target.state_dict(), "optimizer": optimizer.state_dict(),
                                        "generator": resume_state, "step": step + 1, "trace": trace,
                                        "pair_counts": counts, "gpu_seconds": gpu + elapsed})
    pool.shutdown()
    torch.cuda.synchronize()
    gpu += time.monotonic() - began
    payload = {"schema": SCHEMA_JEPA, "plan_identity": IDENTITY, "seed": seed, "steps": total,
               "pair_counts": counts, "loss_trace": trace, "gpu_seconds": gpu, "windows": len(data),
               "parameters": {"encoder": sum(p.numel() for p in encoder.parameters()),
                              "predictor": sum(p.numel() for p in predictor.parameters()),
                              "target_encoder_trainable": 0},
               "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
               "target": {k: v.detach().cpu() for k, v in target.state_dict().items()},
               "predictor": {k: v.detach().cpu() for k, v in predictor.state_dict().items()}}
    if target_path is not None:
        atomic_torch(target_path, payload)
        if progress is not None and progress.is_file():
            progress.unlink()
    return encoder.eval(), predictor.eval(), target, payload


# ---------------------------------------------------------------------------
# controllers (E99, JEPA) with the #77 controller recipe
# ---------------------------------------------------------------------------

def controller_path(output, encoder, seed):
    return Path(output) / "models" / encoder / f"seed-{seed}" / "controller.pt"


def fit_carrier_path(output, tag, key):
    return Path(output) / "fit-carriers" / tag / f"{key}.pt"


def controller_windows(output, encoder, seed, shots):
    """First window per controller-lineage fit shot, grouped by lineage."""
    tag = tag_of(encoder, seed)
    groups = {}
    for shot in shots:
        if shot["member"] not in controller_lineages():
            continue
        data = torch.load(s99.shot_path(SLOT_FIX, shot["key"]), weights_only=True)
        if encoder == "E99":
            carriers = torch.load(s99.carrier_path(SLOT_FIX, "E", shot["key"]), weights_only=True)["carriers"]
        else:
            carriers = torch.load(fit_carrier_path(output, tag, shot["key"]), weights_only=True)["carriers"]
        window = s99.windows(data, carriers)[0]
        groups.setdefault(shot["member"], []).append({k: window[k] for k in ("z", "action", "length")})
    return {member: {k: torch.stack([w[k] for w in rows]) for k in ("z", "action", "length")}
            for member, rows in sorted(groups.items())}


@torch.no_grad()
def lineage_labels(model, controller, data, aggregated):
    """run_issue_77_n1_train.label_lineage on the lineage's first windows."""
    from world_model.training.matched_dynamics import controller_labels
    records = []
    for position in range(len(data["z"])):
        z = data["z"][position:position + 1].to(DEVICE)
        a = data["action"][position:position + 1].to(DEVICE)
        length = data["length"][position:position + 1].to(DEVICE)
        contexts = z.clone()
        if aggregated:
            current, t = z[:, 0], 0
            while t < int(length[0]):
                pair = controller.pairs[int(controller(current, a, length - t).argmax(-1))]
                current = model.carrier(current, a, pair)
                t += pair.delta
                contexts[:, t] = current
        labels = controller_labels(model, controller, z, a, length, contexts)
        n = int(length[0])
        records.append({"z": contexts[0, :n].cpu(), "action": a.expand(n, -1).cpu(),
                        "remaining": torch.arange(n, 0, -1), "labels": labels[0, :n].cpu()})
    return {k: torch.cat([r[k] for r in records]) for k in records[0]}


def train_controller(model, groups, seed, target=None, steps=None):
    from torch.nn import functional as F
    from scripts.run_issue_71_hybrid_readiness import sample
    from world_model.training.matched_dynamics import MatchedController
    per_round = steps or CONTROLLER["steps_per_round"]
    torch.manual_seed(seed + 7700)
    control = MatchedController(False).to(DEVICE)
    lineages = sorted(groups)
    labels = {}
    began = time.monotonic()
    trace = []
    for round_index in range(CONTROLLER["rounds"]):
        for member in lineages:
            labels[(round_index, member)] = lineage_labels(model, control, groups[member], bool(round_index))
        optimizer = torch.optim.AdamW(control.parameters(), lr=CONTROLLER["learning_rate"])
        generator = torch.Generator().manual_seed(seed + 7700 + round_index)
        for step in range(per_round):
            member = lineages[step % len(lineages)]
            rows = [labels[(r, member)] for r in range(round_index + 1)]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            batch = sample(data, CONTROLLER["batch_size"], generator, DEVICE)
            loss = F.cross_entropy(control(batch["z"], batch["action"], batch["remaining"]), batch["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite controller loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (step + 1) % 600 == 0 or step + 1 == per_round:
                trace.append({"round": round_index, "step": step + 1, "loss": float(loss.detach())})
    torch.cuda.synchronize()
    payload = {"schema": SCHEMA_CONTROLLER, "plan_identity": IDENTITY, "seed": seed,
               "rounds": CONTROLLER["rounds"], "steps_per_round": per_round, "lineages": lineages,
               "label_rows": {f"{r}:{m}": len(v["labels"]) for (r, m), v in labels.items()},
               "loss_trace": trace, "gpu_seconds": time.monotonic() - began,
               "parameters": sum(p.numel() for p in control.parameters()),
               "model": {k: v.detach().cpu() for k, v in control.state_dict().items()}}
    if target is not None:
        atomic_torch(target, payload)
    return control.eval(), payload


def load_controller(output, encoder, seed):
    from world_model.training.matched_dynamics import MatchedController
    if encoder == "F77":
        return wlc.load_controller(seed, "hybrid", DEVICE)
    payload = torch.load(controller_path(output, encoder, seed), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA_CONTROLLER or payload.get("plan_identity") != IDENTITY
            or payload["seed"] != seed or payload["steps_per_round"] != CONTROLLER["steps_per_round"]):
        raise ValueError(f"controller {encoder}/{seed} binding differs")
    control = MatchedController(False)
    control.load_state_dict(payload["model"], strict=True)
    return control.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# readout counts
# ---------------------------------------------------------------------------

def gated(predictor, z):
    """What the predictor reads: gated micro (B, 18, 18, 2) and macro (B, 2) values."""
    from world_model.model import Abstraction
    micro = predictor.symbolic_features(z, Abstraction.MICRO)[:, :-2].reshape(len(z), 18, 18, 2)
    macro = predictor.symbolic_features(z, Abstraction.MACRO)[:, :2]
    return micro, macro


def confusion(pred, truth, mask, phase):
    """[tp, fp, fn, tn] per phase for boolean tensors with a leading position axis."""
    out = []
    for index in range(len(PHASES)):
        rows = phase == index
        m = mask[rows]
        p, t = pred[rows][m], truth[rows][m]
        out.append([int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())])
    return out


@torch.no_grad()
def readout_counts(predictor, z, labels, positions, names):
    idx = torch.as_tensor(positions)
    phase = labels["phase"][idx]
    micro, macro = gated(predictor, z.to(DEVICE))
    micro, macro = micro.cpu(), macro.cpu()
    presence = z[:, 2::13][:, :len(names)].cpu() >= READ_THRESHOLD
    truth_presence = labels["presence"][idx]
    out = {}
    for kind in PRESENCE_KINDS:
        slots = [i for i, n in enumerate(names) if n.split(":", 1)[0] == kind]
        out[f"presence:{kind}"] = confusion(presence[:, slots], truth_presence[:, slots],
                                            torch.ones_like(truth_presence[:, slots]), phase)
    relations, relation_mask = labels["relations"][idx], labels["relation_mask"][idx]
    upper = torch.triu(torch.ones(len(names), len(names), dtype=torch.bool), 1)
    out["contact"] = confusion(micro[..., 0] >= READ_THRESHOLD, relations[..., 0],
                               relation_mask[..., 0] & upper, phase)
    out["supports"] = confusion(micro[..., 1] >= READ_THRESHOLD, relations[..., 1], relation_mask[..., 1], phase)
    macros, macro_mask = labels["macros"][idx], labels["macro_mask"][idx]
    for m, name in enumerate(MACRO):
        out[name] = confusion(macro[:, m] >= READ_THRESHOLD, macros[:, m], macro_mask[:, m], phase)
    return out, micro, macro


def add_counts(total, part):
    for key, phases in part.items():
        block = total.setdefault(key, [[0, 0, 0, 0] for _ in PHASES])
        for i, row in enumerate(phases):
            for j in range(4):
                block[i][j] += row[j]


@torch.no_grad()
def audit_record(output, encoder, seed, lineage, shots, predictor, names):
    from world_model.model import Abstraction, PredictionPair
    tag = tag_of(encoder, seed)
    counts = {source: {} for source in SOURCES}
    per_shot = []
    for shot in shots:
        labels = load_labels(output, shot["branch"])
        carriers = load_carriers(output, tag, shot["branch"])
        positions = shard_positions(labels["frames"] - 1)
        part, micro, _ = readout_counts(predictor, carriers[positions], labels, positions, names)
        add_counts(counts["encoder"], part)
        action = labels["action"][None].to(DEVICE)
        for delta in DELTAS:
            valid = [p for p in positions if p >= delta]
            context = carriers[[p - delta for p in valid]].to(DEVICE)
            for alpha in ALPHAS:
                pair = PredictionPair(delta, Abstraction(alpha))
                predicted = predictor.carrier(context, action.expand(len(valid), -1), pair)
                if not bool(torch.isfinite(predicted).all()):
                    raise ValueError(f"nonfinite one-step prediction {tag}/{shot['branch']}")
                part, _, _ = readout_counts(predictor, predicted.cpu(), labels, valid, names)
                add_counts(counts[f"{delta}-{alpha}"], part)
        per_shot.append({"branch": shot["branch"], "frames": labels["frames"], "audited_positions": len(positions),
                         "contact_onset": labels["contact_onset"], "settled_onset": labels["settled_onset"],
                         "max_gated_contact": float(micro[..., 0].max()),
                         "max_gated_supports": float(micro[..., 1].max())})
    return {"schema": SCHEMA_AUDIT, "plan_identity": IDENTITY,
            "cell": {"encoder": encoder, "seed": seed, "lineage": lineage,
                     "cohort": "held_out" if lineage in HELD_OUT else "fit"},
            "counts": counts, "shots": per_shot, "engine_seconds": 0}


def audit_path(output, encoder, seed, lineage):
    return Path(output) / "records" / f"audit--{encoder}--seed{seed}--{lineage}.json"


@torch.no_grad()
def probe_record(output, shots, names):
    """The 2026-09-25 manuscript probe, recomputed (F77)."""
    seed = PROBE["seed"]
    predictor = load_hybrid(output, "F77", seed)
    branch = PROBE["capture"]
    carriers = load_carriers(output, "F77", branch)
    labels = load_labels(output, branch)
    micro, macro = gated(predictor, carriers.to(DEVICE))
    from world_model.model import Abstraction
    raw_macro = torch.sigmoid(predictor.symbols(carriers.to(DEVICE), Abstraction.MACRO)[0]).cpu()
    frame = PROBE["frame"]
    birds = [i for i, n in enumerate(names) if n.startswith("bird:")]
    pig = names.index("pig:0000")
    flat = micro.flatten(1).cpu()
    at = int(flat.max(1).values.argmax())
    lineage_max = {}
    for other_seed in SEEDS:
        model = predictor if other_seed == seed else load_hybrid(output, "F77", other_seed)
        rows = {}
        for shot in shots:
            c = load_carriers(output, "F77", shot["branch"])
            positions = shard_positions(len(c) - 1)
            m, _ = gated(model, c[positions].to(DEVICE))
            rows.setdefault(shot["member"], []).append(float(m.max()))
        lineage_max[str(other_seed)] = {k: max(v) for k, v in sorted(rows.items())}
    # localization of the lineage-009 frames that carry a gated edge > 0.5
    errors = []
    for shot in (s for s in shots if s["member"] == "issue-77-n1-009"):
        c = load_carriers(output, "F77", shot["branch"])
        lab = load_labels(output, shot["branch"])
        positions = shard_positions(len(c) - 1)
        m, _ = gated(predictor, c[positions].to(DEVICE))
        hits = [p for p, value in zip(positions, m.flatten(1).max(1).values.cpu().tolist()) if value > 0.5]
        for p in hits:
            present = lab["presence"][p] & (c[p, 2::13][:len(names)] >= READ_THRESHOLD)
            centers = c[p, 2:2 + 13 * len(names)].reshape(len(names), 13)[:, 5:7]
            errors.extend(torch.linalg.vector_norm(centers[present] - lab["centers"][p][present], dim=-1).tolist())
    return {"schema": SCHEMA_PROBE, "plan_identity": IDENTITY, "capture": branch, "seed": seed,
            "frames": len(carriers), "frame": frame,
            "bird_presence_frame": [float(carriers[frame, 2 + 13 * i]) for i in birds],
            "bird_engine_presence_frame": [bool(labels["presence"][frame, i]) for i in birds],
            "pig_presence_frame": float(carriers[frame, 2 + 13 * pig]),
            "pig_engine_presence_frame": bool(labels["presence"][frame, pig]),
            "max_gated_micro": float(flat.max()), "max_gated_micro_frame": at,
            "gated_macro_frame": [float(v) for v in macro[frame]],
            "raw_macro_frame": [float(v) for v in raw_macro[frame]],
            "engine_macro_frame": [bool(v) for v in labels["macros"][frame]],
            "engine_macro_mask_frame": [bool(v) for v in labels["macro_mask"][frame]],
            "lineage_max_gated_micro": lineage_max,
            "lineage_009_center_errors": {"frames_with_gated_edge_above_0.5": len(errors) > 0,
                                          "slot_errors": len(errors),
                                          "mean": float(np.mean(errors)) if errors else None},
            "engine_seconds": 0}


# ---------------------------------------------------------------------------
# rollout error (tab:res:rollout protocol of the #77 diagnostic)
# ---------------------------------------------------------------------------

def rollout_path(output, encoder, seed, state):
    return Path(output) / "records" / f"rollout--{encoder}--seed{seed}--{state}.json"


def engine_position_mse(prediction, labels, position):
    names = len(labels["presence"][0])
    present = labels["presence"][position]
    if not bool(present.any()):
        return None
    p = torch.tensor(prediction)[2:2 + 13 * names].reshape(names, 13)[:, 5:7]
    return float((p[present] - labels["centers"][position][present]).square().mean())


@torch.no_grad()
def rollout_record(output, encoder, seed, sample, model, adapter, objective):
    from world_model.training.cnn_hybrid import PAIRS
    reference = sample["sources"][0]
    context = context_carrier(adapter, CAMPAIGN / reference["attempt"])
    candidates = []
    gpu = 0.0
    for source in sample["sources"]:
        tag = tag_of(encoder, seed)
        carriers = load_carriers(output, tag, source["identity"])
        labels = load_labels(output, source["identity"])
        last = len(carriers) - 1
        pairs = {}
        for pair in PAIRS:
            prediction = diag.fixed_curve(model, context, source["action"], pair)
            gpu += prediction["transition_wall_seconds"]
            curve = {}
            for t in ROLLOUT_TIMES:
                output_t = prediction["outputs"].get(str(t))
                if output_t is None:
                    curve[str(t)] = None
                    continue
                target = carriers[min(t, last)].tolist()
                errors = diag.field_errors(output_t, target, objective)
                errors["engine_position_mse"] = engine_position_mse(output_t, labels, min(t, last))
                curve[str(t)] = errors
            pairs[f"{pair.delta}-{pair.abstraction}"] = {"failure": prediction["failure"],
                                                         "linear_macs": prediction["linear_macs"],
                                                         "curve": curve}
        candidates.append({"identity": source["identity"], "ordinal": source["ordinal"], "pairs": pairs})
    return {"schema": SCHEMA_ROLLOUT, "plan_identity": IDENTITY,
            "cell": {"encoder": encoder, "seed": seed, "state": sample["state"]["identity"]},
            "context": context.tolist(), "candidates": candidates, "gpu_seconds": gpu, "engine_seconds": 0}


# ---------------------------------------------------------------------------
# #96 arms
# ---------------------------------------------------------------------------

def arms_path(output, phase, encoder, inventory, member, seed):
    return Path(output) / "records" / f"arms{phase}--{encoder}--{inventory}--{member}--seed{seed}.json"


def rollout_arm(model, controller, z0, actions, spec):
    """t96.rollout (same ops, same order) returning endpoint carriers and pair sequences."""
    from world_model.training.matched_dynamics import linear_macs, pairs_for
    pairs = pairs_for(model)
    count = len(actions)
    z = z0[None].to(DEVICE).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=z.device)
    positions = torch.zeros(count, dtype=torch.long, device=z.device)
    sequences = [[] for _ in range(count)]
    pair_macs = [linear_macs(model, pair) for pair in pairs]
    transition_calls = controller_calls = macs = 0
    torch.cuda.synchronize()
    began = time.monotonic()
    with torch.no_grad():
        while bool(alive.any()):
            index = alive.nonzero(as_tuple=True)[0]
            if spec["kind"] == "fixed":
                chosen = torch.full((len(index),), spec["pair_index"], dtype=torch.long, device=z.device)
            else:
                logits = controller(z[index], actions[index], ENDPOINT - positions[index])
                chosen = t96.select_pairs(spec["kind"], logits, spec)
                controller_calls += len(index)
            z_next = z
            for pair_index, pair in enumerate(pairs):
                rows = index[chosen == pair_index]
                if not len(rows):
                    continue
                z_next = z_next.clone()
                z_next[rows] = model.carrier(z[rows], actions[rows], pair)
                transition_calls += len(rows)
                macs += len(rows) * pair_macs[pair_index]
                for row in rows.tolist():
                    sequences[row].append(pair_index)
                positions[rows] += pair.delta
            z = z_next
            alive = alive & torch.isfinite(z).all(dim=1) & (positions < ENDPOINT)
    torch.cuda.synchronize()
    macs += controller_calls * m99.macs_of(controller)
    return z, sequences, {"transition_calls": transition_calls, "controller_calls": controller_calls,
                          "linear_macs": macs, "gpu_seconds": time.monotonic() - began}


class ArmStack:
    def __init__(self, output):
        self.output = output
        self.costs = m99.Costs(wlc.load_objective())
        self.models, self.controllers, self.adapters, self.anchors = {}, {}, {}, {}

    def model(self, encoder, seed):
        if (encoder, seed) not in self.models:
            self.models[(encoder, seed)] = load_hybrid(self.output, encoder, seed)
        return self.models[(encoder, seed)]

    def controller(self, encoder, seed):
        if (encoder, seed) not in self.controllers:
            self.controllers[(encoder, seed)] = load_controller(self.output, encoder, seed)
        return self.controllers[(encoder, seed)]

    def anchor(self, encoder, seed, member):
        tag = tag_of(encoder, seed)
        if (tag, member) not in self.anchors:
            if tag not in self.adapters:
                self.adapters[tag] = load_adapter(self.output, tag)
            anchor = p94.frozen_anchor(member)
            if sha256_of(anchor["frame_path"]) != anchor["sha256"]:
                raise ValueError(f"{STOP_TOKEN}: anchor of {member} changed")
            self.anchors[(tag, member)] = wlc.encode_carrier(self.adapters[tag], anchor, DEVICE).float()
        return self.anchors[(tag, member)]


def arms_record(stack, encoder, seed, inventory, member, entry, arms, coordinates):
    items = entry["inventory"]
    actions = m99.action_batch(items)
    z0 = stack.anchor(encoder, seed, member)
    out = {}
    for arm in arms:
        spec = t96.arm_spec(arm, coordinates)
        controller = stack.controller(encoder, seed) if arm in CONTROLLER_ARMS else None
        z_end, sequences, timing = rollout_arm(stack.model(encoder, seed), controller, z0, actions, spec)
        rows = [stack.costs.row(item["ordinal"], z_end[slot].cpu(), z0.cpu()) for slot, item in enumerate(items)]
        out[arm] = {"rows": rows, "candidates": len(items),
                    "schedules": ({str(item["ordinal"]): t96.rle(seq) for item, seq in zip(items, sequences)}
                                  if arm in CONTROLLER_ARMS else None),
                    "pair_step_counts": dict(sorted(Counter(t96.PAIR_NAMES[k] for seq in sequences
                                                            for k in seq).items())),
                    **timing}
    return {"schema": SCHEMA_ARMS, "plan_identity": IDENTITY,
            "cell": {"encoder": encoder, "seed": seed, "inventory": inventory, "member": member,
                     "state": entry["state"]},
            "coordinates": coordinates, "arms": out, "engine_seconds": 0}


def frequencies_path(output, encoder):
    return Path(output) / f"frequencies--{encoder}.json"


def arm_frequencies(output, encoder, universe):
    """t96.j_frequencies on this runner's J schedules (outcome-free)."""
    counts = {seed: [0] * len(t96.PAIR_NAMES) for seed in SEEDS}
    for inventory, member, seed in t96.scheduled_cells(universe):
        record = read_record(arms_path(output, 1, encoder, inventory, member, seed), SCHEMA_ARMS)
        for text in record["arms"]["J"]["schedules"].values():
            for index in t96.unrle(text):
                counts[seed][index] += 1
    coordinates = {}
    for seed in SEEDS:
        by_alpha = [sum(counts[seed][d * 3 + a] for d in range(3)) for a in range(3)]
        by_delta = [sum(counts[seed][d * 3 + a] for a in range(3)) for d in range(3)]
        alpha = max(range(3), key=lambda a: (by_alpha[a], -a))
        delta = max(range(3), key=lambda d: (by_delta[d], -d))
        coordinates[str(seed)] = {"alpha_T_index": alpha, "alpha_T": ALPHAS[alpha],
                                  "delta_D_index": delta, "delta_D": DELTAS[delta],
                                  "alpha_marginal_steps": by_alpha, "delta_marginal_steps": by_delta}
    return {"schema": SCHEMA_FREQUENCIES, "plan_identity": IDENTITY, "encoder": encoder,
            "rule": ("J step-pair counts pooled per seed over every scheduled cell of all four inventories and "
                     "every candidate (outcome-free); alpha_T / Delta_D = modal marginal, ties to the lower index"),
            "pairs": list(t96.PAIR_NAMES), "counts": {str(s): counts[s] for s in SEEDS},
            "coordinates": coordinates}


def read_record(path, schema):
    record = read_json(path)
    if record.get("schema") != schema or record.get("plan_identity") != IDENTITY:
        raise ValueError(f"record {path} binding differs")
    return record


# ---------------------------------------------------------------------------
# smoke (pre-freeze: nothing kept, no verdict joined)
# ---------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        raise ValueError("plan.json already frozen; the smoke is a pre-freeze control")
    scratch = output / "smoke-scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    names = vocabulary()
    shots = audit_shots()
    checks, timings = [], {}
    fit_shot = next(s for s in shots if s["branch"] == PROBE["capture"])
    held_shot = next(s for s in shots if s["cohort"] == "held_out")
    # 1. labels
    began = time.monotonic()
    for shot in (fit_shot, held_shot):
        atomic_torch(label_path(scratch, shot["branch"]), build_labels(shot, names))
    timings["label_seconds_per_shot"] = (time.monotonic() - began) / 2
    built = [load_labels(scratch, s["branch"]) for s in (fit_shot, held_shot)]
    checks.append({"control": "labels_and_phases", "shots": [
        {"branch": b["branch"], "frames": b["frames"], "contact_onset": b["contact_onset"],
         "settled_onset": b["settled_onset"]} for b in built], "ok": all(b["frames"] > 1 for b in built)})
    # 2. F77 carriers replicate the #77 shard (G1 mechanics on one shot); E99 parse path
    adapters = {"F77": wlc.load_adapter(DEVICE), "E99": s99.load_adapter(SLOT_FIX, "E")}
    began = time.monotonic()
    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        carriers = shot_carriers(adapters, fit_shot["attempt"], pool)
    timings["carrier_seconds_per_shot_two_encoders"] = time.monotonic() - began
    delta = shard_replication(carriers["F77"], fit_shot)
    checks.append({"control": "G1_F77_carriers_replicate_shard", "branch": fit_shot["branch"],
                   "max_abs_delta": delta, "tolerance": PARSE_TOLERANCE, "ok": delta <= PARSE_TOLERANCE})
    parse_delta = parse_replication(adapters["E99"], fit_shot["attempt"], carriers["E99"])
    checks.append({"control": "batched_parse_replicates_adapter_parse_batch", "max_abs_delta": parse_delta,
                   "tolerance": PARSE_TOLERANCE, "ok": parse_delta <= PARSE_TOLERANCE})
    # 3. differentiable carrier builder equals the adapter build (G6)
    build_delta = carrier_build_replication(adapters["E99"].model, fit_shot["attempt"], carriers["E99"])
    checks.append({"control": "G6_differentiable_carrier_equals_adapter", "max_abs_delta": build_delta,
                   "tolerance": CARRIER_BUILD_TOLERANCE, "ok": build_delta <= CARRIER_BUILD_TOLERANCE})
    # 4. audit mechanics
    atomic_torch(carrier_path(scratch, "F77", fit_shot["branch"]),
                 {"schema": SCHEMA_CARRIERS, "tag": "F77", "branch": fit_shot["branch"], "carriers": carriers["F77"]})
    predictor = load_hybrid(output, "F77", SEEDS[0])
    began = time.monotonic()
    record = audit_record(scratch, "F77", SEEDS[0], fit_shot["member"], [fit_shot], predictor, names)
    timings["audit_seconds_per_shot"] = time.monotonic() - began
    checks.append({"control": "audit_mechanics", "sources": len(record["counts"]),
                   "ok": all(len(record["counts"][s]) == len(ALL_PREDICATES) for s in SOURCES)})
    # 5. rollout replication on one #77 candidate record (G2 mechanics)
    sample = rollout_samples()[0]
    source = sample["sources"][0]
    context = context_carrier(adapters["F77"], CAMPAIGN / source["attempt"])
    retained_targets = read_json(DIAGNOSTIC / "targets" / f"state-{sample['state']['ordinal']:03d}.json")
    context_delta = float((context - torch.tensor(retained_targets["context"])).abs().max())
    retained = read_json(diag_curve_path(SEEDS[0], sample, "hybrid_continuous_h15", source["ordinal"]))
    from world_model.model import Abstraction, PredictionPair
    fresh = diag.fixed_curve(predictor, context, source["action"], PredictionPair(15, Abstraction.CONTINUOUS))
    curve_delta = max(abs(a - b) for t in ROLLOUT_TIMES for a, b in
                      zip(fresh["outputs"][str(t)], retained["prediction"]["outputs"][str(t)]))
    checks.append({"control": "G2_rollout_replicates_77_record", "state": sample["state"]["identity"],
                   "context_max_abs_delta": context_delta, "curve_max_abs_delta": curve_delta,
                   "tolerance": PARSE_TOLERANCE,
                   "ok": context_delta <= PARSE_TOLERANCE and curve_delta <= PARSE_TOLERANCE})
    # 6. JEPA training mechanics on two fit shots (nothing kept)
    sources, universe, fit_shots = fit_data()
    chosen = [fit_shots[0], next(s for s in fit_shots if s["source"] != "issue_77_n1")]
    began = time.monotonic()
    meta, windows_list = build_window_data(scratch, chosen)
    cached = build_frame_cache(scratch, meta, windows_list, cache=scratch / "frame-cache")
    timings["frame_cache_seconds_per_frame"] = (time.monotonic() - began) / cached
    data = JepaData(scratch)
    began = time.monotonic()
    encoder, jepa_predictor, target, payload = train_jepa(scratch, data, SEEDS[0], steps=SMOKE_STEPS)
    timings["jepa_seconds_per_step"] = (time.monotonic() - began) / SMOKE_STEPS
    torch.manual_seed(SEEDS[0])
    initial = s99.new_encoder().state_dict()
    moved = (any(not torch.equal(a.cpu(), b) for a, b in zip(target.state_dict().values(), initial.values()))
             and any(not torch.equal(a, b) for a, b in zip(target.state_dict().values(),
                                                           encoder.state_dict().values())))
    checks.append({"control": "jepa_trains", "windows": len(data), "frames_cached": cached,
                   "final": payload["loss_trace"][-1], "parameters": payload["parameters"],
                   "target_requires_grad": any(p.requires_grad for p in target.parameters()),
                   "ok": (math.isfinite(payload["loss_trace"][-1]["jepa_loss"])
                          and not any(p.requires_grad for p in target.parameters()) and moved)})
    # 7. controller mechanics on the trained smoke JEPA (6 steps)
    fit_carriers = {}
    jepa_adapter = wrap(encoder, f"{IDENTITY}:smoke")
    for shot in chosen:
        end = torch.load(s99.shot_path(SLOT_FIX, shot["key"]), weights_only=True)["end"]
        fit_carriers[shot["key"]] = s99.carriers_for(jepa_adapter, shot["attempt"], list(range(end + 1)))
    groups = {"smoke": {k: torch.stack([s99.windows(torch.load(s99.shot_path(SLOT_FIX, s["key"]), weights_only=True),
                                                    fit_carriers[s["key"]])[0][k] for s in chosen])
                        for k in ("z", "action", "length")}}
    began = time.monotonic()
    _, controller_payload = train_controller(jepa_predictor, groups, SEEDS[0], steps=6)
    timings["controller_seconds_smoke"] = time.monotonic() - began
    checks.append({"control": "controller_trains", "label_rows": controller_payload["label_rows"],
                   "ok": math.isfinite(controller_payload["loss_trace"][-1]["loss"])})
    # 8. #96 arm replication on the #96 G1 cell (G3 mechanics)
    stack = ArmStack(output)
    entry = universe["angle"][t96.G1_MEMBER]
    record = arms_record(stack, "F77", t96.G1_SEED, "angle", t96.G1_MEMBER, entry, ("J", "F-15-continuous"), None)
    worst = 0.0
    for arm in ("J", "F-15-continuous"):
        retained = t96.read_record(t96.OUTPUT, "angle", t96.G1_MEMBER, t96.G1_SEED, arm)["decision"]["ranking"]
        mine = {row["ordinal"]: row["count"] for row in record["arms"][arm]["rows"]}
        worst = max(worst, max(abs(mine[r["ordinal"]] - r["predicted_cost"]) for r in retained
                               if r["predicted_cost"] is not None))
    checks.append({"control": "G3_arms_replicate_96_records", "cell": f"angle/{t96.G1_MEMBER}/seed{t96.G1_SEED}",
                   "max_abs_count_cost_delta": worst, "tolerance": COST_TOLERANCE, "ok": worst <= COST_TOLERANCE})
    fit_count = sum(1 for s in fit_shots if s["member"] in HELD_OUT)
    checks.append({"control": "held_out_isolation", "held_out_shots_in_fit_data": fit_count, "ok": fit_count == 0})
    frames_total = sum(len(n) for n in needed_positions(fit_windows_for(fit_shots), len(fit_shots)))
    projection = {
        "audit_shots": len(shots),
        "label_seconds": timings["label_seconds_per_shot"] * len(shots) / LABEL_WORKERS,
        "carrier_seconds": timings["carrier_seconds_per_shot_two_encoders"] * len(shots) * 2.5,
        "frame_cache_seconds": timings["frame_cache_seconds_per_frame"] * frames_total,
        "jepa_seconds": timings["jepa_seconds_per_step"] * JEPA["steps"] * len(SEEDS),
        "audit_seconds": timings["audit_seconds_per_shot"] * len(shots) * len(SEEDS) * len(ENCODERS),
        "controller_seconds": timings["controller_seconds_smoke"] / 12 * CONTROLLER["steps_per_round"] * 2 * 6,
    }
    projection["total_seconds"] = sum(v for k, v in projection.items() if k != "audit_shots")
    ok = all(check["ok"] for check in checks)
    evidence = {"schema": SCHEMA_SMOKE, "identity": IDENTITY, "run_at": utc_now(), "checks": checks,
                "timings": timings, "projection": projection, "frame_cache_frames": frames_total,
                "reading": "infrastructure assertions only; nothing kept; no verdict joined; no F1 computed",
                "ok": ok}
    write_json(output / "smoke.json", evidence)
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"smoke {'PASSED' if ok else 'FAILED'}; projected {projection['total_seconds'] / 3600:.2f} h "
        f"({json.dumps({k: round(v) for k, v in projection.items()})})")
    for check in checks:
        log(f"  {check['control']}: {check['ok']}")
    if not ok:
        raise ValueError("smoke control differs; STOP RULE: abort before the freeze")
    return 0


def fit_windows_for(fit_shots):
    meta = []
    for shot in fit_shots:
        _, _, frames, _ = s99.shot_frames(shot["attempt"])
        meta.append({"end": min(len(frames) - 1, JEPA["max_position"])})
    return fit_windows(meta)


def diag_curve_path(seed, sample, system, ordinal):
    return DIAGNOSTIC / "fixed" / f"seed-{seed}" / f"state-{sample['state']['ordinal']:03d}" / system / \
        f"candidate-{ordinal:02d}.json"


def shard_replication(carriers, shot):
    """Max |z_shard - z_fresh| over the shard's valid rows of this shot's windows."""
    shard = torch.load(DYNAMICS / "shards" / f"lineage-n1-{int(shot['member'].rsplit('-', 1)[1]):03d}.pt",
                       map_location="cpu", weights_only=True)
    branch_index = shard["record"]["branches"].index(shot["branch"])
    worst = 0.0
    for w, window in enumerate(shard["windows"]):
        if window["shot"] != branch_index:
            continue
        length = int(shard["tensors"]["length"][w])
        rows = shard["tensors"]["z"][w, :length + 1]
        fresh = carriers[window["start"]:window["start"] + length + 1]
        worst = max(worst, float((rows - fresh).abs().max()))
    return worst


def parse_replication(adapter, attempt, carriers, positions=(0, 1, 80, 200)):
    from world_model.data.deployment_temporal import AgentObservation, TemporalObservationContext
    _, observation_root, frames, _ = s99.shot_frames(attempt)

    def observation(p):
        ref = frames[p]["agent_observation"]
        return AgentObservation(ref["identity"], frames[p]["fixed_step"], frames[p]["fixed_time_seconds"],
                                (observation_root / ref["relative_path"]).read_bytes(), "agent")
    worst = 0.0
    for p in positions:
        chosen = [p - 1, p] if p > 0 else [p]
        parsed = adapter.parse_batch(tuple(observation(q) for q in chosen))
        context = TemporalObservationContext(None if p == 0 else observation(p - 1), observation(p))
        z = adapter.build_from_parsed(context, parsed[-1], None if p == 0 else parsed[0]).tensor
        worst = max(worst, float((z - carriers[p]).abs().max()))
    return worst


def carrier_build_replication(model, attempt, carriers, positions=(0, 1, 80, 200, 330)):
    from world_model.training import jepa_slot_encoder as js
    _, observation_root, frames, _ = s99.shot_frames(attempt)
    size = (JEPA["image_width"], JEPA["image_height"])
    positions = [p for p in positions if p < len(frames)]
    current = torch.stack([decode(observation_root / frames[p]["agent_observation"]["relative_path"], [size])[size]
                           for p in positions]).to(DEVICE)
    prior = torch.stack([decode(observation_root / frames[max(p - 1, 0)]["agent_observation"]["relative_path"],
                                [size])[size] for p in positions]).to(DEVICE)
    has_prior = torch.tensor([float(p > 0) for p in positions], device=DEVICE)
    elapsed = torch.tensor([0.0 if p == 0 else frames[p]["fixed_time_seconds"] - frames[p - 1]["fixed_time_seconds"]
                            for p in positions], device=DEVICE)
    with torch.no_grad():
        z, _ = js.encode_carriers(model.eval(), current, prior, has_prior, elapsed,
                                  js.expected_kind_indices(vocabulary()))
    return float((z.cpu() - carriers[list(positions)]).abs().max())


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def input_bindings(output):
    names = {
        "issue_77_dynamics_plan": DYNAMICS / "plan.json",
        "issue_77_n1_plan": CAMPAIGN / "plan.json",
        "issue_77_n1_coverage": CAMPAIGN / "coverage.json",
        "issue_77_diagnostic_plan": DIAGNOSTIC / "plan.json",
        "issue_77_diagnostic_summary": DIAGNOSTIC / "summary.json",
        "issue_87_plan": wlc.EIGHTY_SEVEN / "plan.json",
        "issue_96_plan": t96.OUTPUT / "plan.json",
        "issue_96_frequencies": t96.OUTPUT / "frequencies.json",
        "issue_99_slot_fix_plan": SLOT_FIX / "plan.json",
        "issue_99_slot_fix_encoder": SLOT_FIX / "encoder" / "encoder.pt",
        "issue_70_parser_plan": ROOT / ".local-artifacts/issue-70-parser-repair-v6/plan.json",
        "model_module": ROOT / MODEL_MODULE,
        "spatial_slot_parser_module": ROOT / s99.MODEL_MODULE,
        "smoke": Path(output) / "smoke.json",
    }
    for seed in SEEDS:
        names[f"issue_99_E_hybrid_seed{seed}"] = s99.predictor_path(SLOT_FIX, "E", seed, "hybrid")
        names[f"issue_77_hybrid_controller_seed{seed}"] = DYNAMICS / f"seed-{seed}" / "hybrid" / "controller.pt"
    for path in sorted((DYNAMICS / "shards").glob("lineage-n1-*.pt")):
        names[f"issue_77_shard_{path.stem}"] = path
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def parameter_accounting():
    from world_model.training.cnn_hybrid import CNNHybridPredictor
    from world_model.training.matched_dynamics import MatchedController
    parser = wlc.load_adapter("cpu").model
    predictor = sum(p.numel() for p in CNNHybridPredictor().parameters())
    spatial = sum(p.numel() for p in s99.new_encoder().parameters())
    return {"F77": {"encoder": sum(p.numel() for p in parser.parameters()), "predictor": predictor,
                    "controller": sum(p.numel() for p in MatchedController(False).parameters()),
                    "encoder_training": "issue-70 parser v6 (20 epochs, separate)",
                    "predictor_training": "#77 recipe: 9000 steps x 64 windows over 2409 lineages"},
            "E99": {"encoder": spatial, "predictor": predictor,
                    "controller": sum(p.numel() for p in MatchedController(False).parameters()),
                    "encoder_training": "#99: 12 epochs x 42344 frames, batch 64 (separate, before the predictor)",
                    "predictor_training": "#77 recipe: 9000 steps x 64 windows over the 745 #99 fit shots"},
            "JEPA": {"encoder": spatial, "target_encoder_trainable": 0, "predictor": predictor,
                     "controller": sum(p.numel() for p in MatchedController(False).parameters()),
                     "encoder_training": "joint: 9000 steps x 64 windows (256 context frames per step)",
                     "predictor_training": "joint: 9000 steps x 64 windows over the 745 #99 fit shots"}}


def frozen_plan(output, frozen_at, smoke_evidence, shots, fit_shots, universe):
    samples = rollout_samples()
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal", "issue": 100,
        "frozen_at": frozen_at, "frozen_before_training_and_scoring": True, "issue_64_authorized": False,
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "model_module_sha256_at_freeze":
            sha256_of(ROOT / MODEL_MODULE), "engine_seconds": 0,
        "questions": {
            "Q1_readout_fidelity": ("how well do presence, contact, supports, steady-state and structure-unstable "
                                    "read out of each encoder's carriers and one-step predictions, against engine "
                                    "labels, per phase; are the 2026-09-25 probe numbers reproduced?"),
            "Q2_readout_target": (f"does at least one encoder reach macro F1 >= {MACRO_F1_TARGET} and micro edge "
                                  f"F1 >= {MICRO_F1_TARGET} on held-out lineages?"),
            "Q3_jepa_vs_frozen": ("how do the jointly trained JEPA encoder's rollout error (#77 diagnostic "
                                  "protocol) and readout F1 compare with the frozen-encoder carriers (F77, E99)?"),
            "Q4_alpha_effect": "J - T and J - D per inventory on the better encoder (#96 arms, rerun here)"},
        "disclosure": ("Known before freeze: the unaudited probe numbers quoted in issue #100, every #77/#96/#99/#109 "
                       "table (including the #99 E-arm held-out perception accuracies and ranking AUCs). No readout "
                       "F1, JEPA outcome or new-checkpoint ranking statistic was computed before this freeze; the "
                       "smoke asserts mechanics only."),
        "encoders": ENCODER_DEFINITIONS,
        "audit": {
            "shots": [{k: s[k] for k in ("branch", "member", "cohort")} for s in shots],
            "cohorts": {"fit": "the 12 #77 predictor/controller lineages (the N1 dynamics shards)",
                        "held_out": list(HELD_OUT)},
            "frames": ("the #77 build_n1_shot window rows of every shot: starts sorted({round(i * max(0, last - 60) "
                       "/ 3)}), 60 transitions, valid rows only, union per shot (for fit lineages exactly the "
                       "shard frames)"),
            "labels": ("#99 shot_labels = #77 build_n1_shot semantics: engine presence (active body), projected "
                       "centers, cohort-v2 contact (symmetric) / supports (directed) with availability masks, "
                       "native macro labels with availability masks"),
            "phases": ("engine-only per shot: pre_contact = frames before the first bird contact with any non-bird, "
                       "non-slingshot slot (all frames if none); settled = from the first frame >= onset after which "
                       "steady-state is available and true to the end; cascade = the frames between"),
            "readouts": {
                "presence": "carrier presence_probability >= 0.5 per slot against engine presence; all 18 slots",
                "micro": ("the predictor's gated micro value (symbolic_features: sigmoid(logit) x carrier presence "
                          "of both slots x sigmoid(availability)) >= 0.5; contact on unordered pairs i < j, supports "
                          "on ordered pairs, only where the engine relation mask is available"),
                "macro": "gated macro value sigmoid(logit) x sigmoid(availability) >= 0.5 where the engine mask is available"},
            "sources": {"encoder": "heads on the encoded carrier z_p",
                        "Delta-alpha": ("heads on predictor.carrier(z_{p - Delta}, a, (Delta, alpha)) for every "
                                        "request; headline rows use alpha matched to the predicate level "
                                        f"({MATCHED_ALPHA})")},
            "statistics": ("precision, recall, F1 from counts pooled over the cohort's frames, per seed, then mean "
                           "over seeds; intervals: lineage-clustered percentile bootstrap (10000 draws, PCG64 7201, "
                           "one resample shared by the seeds), DESCRIPTIVE"),
        },
        "probe": PROBE,
        "readout_target": {
            "macro_f1": "mean of the steady-state and structure-unstable F1 (encoder source, all phases, held-out)",
            "micro_edge_f1": "mean of the contact-edge and supports-edge F1 (encoder source, all phases, held-out)",
            "rule": (f"supported iff guards pass AND some encoder has seed-mean macro F1 >= {MACRO_F1_TARGET} and "
                     f"micro edge F1 >= {MICRO_F1_TARGET} (point estimates, as declared in the issue); "
                     f"not_supported_by_this_experiment iff guards pass and none does; otherwise {STOP_TOKEN}"),
            "better_encoder": ("among encoders meeting the target (all encoders if none): the highest mean of held-out "
                               "macro F1 and micro edge F1; ties to the ENCODERS order"),
        },
        "jepa": {**JEPA, "module": MODEL_MODULE,
                 "fit_data": {"rule": "the #99 fit shots (every #77 N1 admissible branch of the 12 fit lineages and "
                                      "every verdict-bearing inventory execution of those lineages); labels from the "
                                      "#99 shot cache; verdicts never read",
                              "shots": [s["key"] for s in fit_shots]},
                 "architecture": ("context encoder = SpatialSlotParser (hidden 128, 2 attention layers x 4 heads, "
                                  "320x240); target encoder = EMA copy, no gradient; predictor = CNNHybridPredictor"),
                 "loss": ("#77 training_loss structure (4 local transitions + recursive, carrier bound .01) with "
                          "targets from the target encoder (stop-gradient) + 0.1 x micro and macro head losses on "
                          "predicted and context carriers for every request + 1.0 x issue-70 parser_loss on the "
                          "context frames (class-balanced presence weights of the #99 encoder payload)"),
                 "requests": "PAIRS[step % 9]", "initialization": "from scratch, torch.manual_seed(seed)",
                 "inference": "the context encoder in the standard RepairedAdapter (same code path as F77/E99)",
                 "selection": "last update; no validation split"},
        "controllers": {**CONTROLLER, "encoders": ["E99", "JEPA"], "lineages": controller_lineages(),
                        "F77": "the frozen #77 controllers"},
        "parameters": parameter_accounting(),
        "rollout": {"protocol": ("#77 diagnostic (tab:res:rollout): held-out states with >= 1 admissible branch, "
                                 "every admissible candidate, one context carrier per state (frame 0 of the lowest-"
                                 "ordinal branch, batch-1 parse by the encoder's adapter), recursive fixed-request "
                                 "rollout to 600 with no truth resets; errors at t in "
                                 f"{list(ROLLOUT_TIMES)} against the encoder's own parsed carrier of the candidate's "
                                 "frame (diag.field_errors, target-availability masks) and against the engine-"
                                 "projected centers of engine-present slots (engine_position_mse); mean over "
                                 "candidates, then states, then seeds"),
                    "states": [s["state"]["identity"] for s in samples],
                    "headline": f"position_mse and engine_position_mse at t = {ROLLOUT_ENDPOINT}, all nine requests",
                    "contrast": ("JEPA - F77 and JEPA - E99 per request: state-clustered paired bootstrap of "
                                 "seed-mean per-state differences, DESCRIPTIVE")},
        "arms": {"arms": list(ARMS), "definitions": {a: t96.ARM_DEFINITIONS[a] for a in ARMS},
                 "T_D_coordinates": "modal alpha / Delta of each encoder's own J frequencies (t96 rule)",
                 "inventories": list(INVENTORIES), "cohorts": list(ARM_COHORTS),
                 "costs": m99.COST_DEFINITIONS, "primary_cost": PRIMARY_COST,
                 "primary_cohort": PRIMARY_ARM_COHORT,
                 "cohort_note": ("E99 and JEPA were trained on inventory executions of the fit lineages, so only the "
                                 "held-out cohort is out-of-sample for them; 'all' is descriptive"),
                 "contrasts": ["J-T", "J-D", "J-M"],
                 "rule": (f"per inventory, better encoder, held-out cohort, tie-free cost: supported iff guards pass "
                          f"AND mean >= {DELTA} AND lower > 0; not_supported_by_this_experiment iff guards pass AND "
                          f"upper < {DELTA}; otherwise {STOP_TOKEN}"),
                 "cells": ("mixed-verdict member x 3 seeds; member-clustered bootstrap, 10000 draws, PCG64 7201, "
                           "DESCRIPTIVE")},
        "guards": {
            "G1": f"F77 audit carriers replicate every valid #77 shard carrier row within {PARSE_TOLERANCE}",
            "G2": (f"F77 rollout per-seed position_mse at t = {ROLLOUT_ENDPOINT} replicates the published #77 "
                   f"diagnostic summary for {sorted(TAB_SYSTEMS)} within relative {ROLLOUT_RELATIVE_TOLERANCE}"),
            "G3": (f"F77 arms replicate the #96 records (count cost of every candidate of every F/J/M/T/D record) "
                   f"within {COST_TOLERANCE}, and the F77 T/D coordinates equal #96's"),
            "G4": "no held-out lineage shot in any JEPA/E99 training or controller window",
            "G5": "3 JEPA checkpoints at 9000 steps with finite losses; 6 controllers complete",
            "G6": (f"the differentiable carrier of each trained JEPA encoder equals its adapter carrier within "
                   f"{CARRIER_BUILD_TOLERANCE} on five frames of {PROBE['capture']}"),
            "G7": f"every micro/macro predicate has >= {LABEL_POSITIVES_MIN} positive engine labels in the held-out audit",
            "caps": f"GPU seconds <= {GPU_CAP_SECONDS}, wall seconds <= {WALL_CAP_SECONDS}"},
        "universe": {"audit_shots": len(shots), "fit_shots": len(fit_shots),
                     "arms_cells": len(t96.scheduled_cells(universe)) * len(ENCODERS),
                     "rollout_states": len(samples)},
        "frame_cache": {"path": str(FRAME_CACHE / "frames.npy"),
                        "note": "derived uint8 cache of fit-shot frames (320x240); rebuildable; not an input"},
        "prohibited": ["reading any verdict before the arms phase finishes", "changing any recipe after training "
                       "starts", "retrying typed failures", "re-opening any prior disposition (#15-#99, #109)"],
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "wall_seconds": WALL_CAP_SECONDS, "stop": STOP_TOKEN},
        "smoke_evidence": smoke_evidence,
        "inputs": input_bindings(output),
        "claim_boundary": ("readout fidelity on retained #77 N1 captures of 16 normal-mechanics lineages; decision-only "
                           "ranking of retained engine-verdicted executions; no engine access, rendering, capture, "
                           "closed-loop or competence claim; every interval is DESCRIPTIVE; nothing here edits the "
                           "ICLR 2026 submission"),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-100 protocol")
    if not plan.get("frozen_before_training_and_scoring") or not plan["smoke_evidence"].get("ok"):
        raise ValueError("plan.json does not declare a passed-smoke freeze")
    blob = json_text({k: v for k, v in plan.items() if k not in ("smoke_evidence", "disclosure", "probe")})
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
    for name in ("audit", "jepa", "models", "records", "fit-carriers"):
        if (output / name).exists():
            raise ValueError(f"{name} exists before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed; STOP RULE: no freeze")
    shots = audit_shots()
    _, universe, fit_shots = fit_data()
    plan = frozen_plan(output, utc_now(), evidence, shots, fit_shots, universe)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {len(shots)} audit shots, {len(fit_shots)} fit shots, "
        f"{plan['universe']['arms_cells']} arm cells")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def load_ledger(output):
    path = Path(output) / "ledger.json"
    if path.is_file():
        return read_json(path)
    return {"schema": SCHEMA_LEDGER, "identity": IDENTITY, "status": "running", "phases": {},
            "gpu_seconds_elapsed": 0.0, "wall_seconds_elapsed": 0.0}


def run(output):
    output = Path(output)
    plan = load_plan(output)
    names = vocabulary()
    shots = audit_shots()
    if [{k: s[k] for k in ("branch", "member", "cohort")} for s in shots] != plan["audit"]["shots"]:
        raise ValueError("audit shots differ from the frozen plan")
    sources, universe, fit_shots = fit_data()
    if [s["key"] for s in fit_shots] != plan["jepa"]["fit_data"]["shots"]:
        raise ValueError("fit shots differ from the frozen plan")
    ledger = load_ledger(output)
    began, wall_base = time.monotonic(), ledger["wall_seconds_elapsed"]

    def mark(phase, seconds, gpu=0.0, **extra):
        ledger["phases"][phase] = {"seconds": ledger["phases"].get(phase, {}).get("seconds", 0.0) + seconds,
                                   "gpu_seconds": ledger["phases"].get(phase, {}).get("gpu_seconds", 0.0) + gpu,
                                   **extra}
        ledger["gpu_seconds_elapsed"] += gpu
        ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
        if ledger["gpu_seconds_elapsed"] > GPU_CAP_SECONDS or ledger["wall_seconds_elapsed"] > WALL_CAP_SECONDS:
            ledger["status"] = "cap_exceeded"
            write_json(output / "ledger.json", ledger)
            raise ValueError(f"{STOP_TOKEN}: compute cap exceeded")
        write_json(output / "ledger.json", ledger)

    # 1. engine labels of the audit shots (outcome-free)
    phase_began = time.monotonic()
    pending = [s for s in shots if not label_path(output, s["branch"]).is_file()]
    if pending:
        progress = Progress("audit labels", len(pending))
        with ProcessPoolExecutor(LABEL_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(build_labels_and_save, str(output), s, names) for s in pending]):
                future.result()
                progress.step(10)
        mark("labels", time.monotonic() - phase_began, shots=len(shots))
    # 2. JEPA training data (outcome-free)
    if not window_data_path(output).is_file() or not frame_index_path(output).is_file():
        phase_began = time.monotonic()
        meta, windows_list = build_window_data(output, fit_shots)
        frames = build_frame_cache(output, meta, windows_list)
        mark("jepa_data", time.monotonic() - phase_began, windows=len(windows_list), frames=frames)
    # 3. JEPA training
    data = None
    for seed in SEEDS:
        if jepa_path(output, seed).is_file():
            continue
        data = data or JepaData(output)
        phase_began = time.monotonic()
        _, _, _, payload = train_jepa(output, data, seed, target_path=jepa_path(output, seed))
        mark(f"jepa_seed{seed}", time.monotonic() - phase_began, gpu=payload["gpu_seconds"])
    del data
    # 4. JEPA carriers of the controller-lineage fit shots
    for seed in SEEDS:
        tag = tag_of("JEPA", seed)
        pending = [s for s in fit_shots if s["member"] in controller_lineages()
                   and not fit_carrier_path(output, tag, s["key"]).is_file()]
        if not pending:
            continue
        adapter = load_adapter(output, tag)
        phase_began = time.monotonic()
        progress = Progress(f"fit carriers {tag}", len(pending))
        for shot in pending:
            end = torch.load(s99.shot_path(SLOT_FIX, shot["key"]), weights_only=True)["end"]
            atomic_torch(fit_carrier_path(output, tag, shot["key"]),
                         {"schema": SCHEMA_CARRIERS, "tag": tag, "branch": shot["key"],
                          "carriers": s99.carriers_for(adapter, shot["attempt"], list(range(end + 1)))})
            progress.step(20)
        mark(f"fit_carriers_{tag}", time.monotonic() - phase_began)
    # 5. controllers
    for encoder in ("E99", "JEPA"):
        for seed in SEEDS:
            target = controller_path(output, encoder, seed)
            if target.is_file():
                continue
            phase_began = time.monotonic()
            groups = controller_windows(output, encoder, seed, fit_shots)
            _, payload = train_controller(load_hybrid(output, encoder, seed), groups, seed, target)
            log(f"controller {encoder} seed {seed}: {payload['gpu_seconds']:.0f}s; final "
                f"{payload['loss_trace'][-1]['loss']:.4f}")
            mark(f"controller_{encoder}_seed{seed}", time.monotonic() - phase_began, gpu=payload["gpu_seconds"])
    # 6. audit carriers (every position of every audit shot, every encoder)
    tags = [t for e in ENCODERS for t in encoder_tags(e)]
    pending = [s for s in shots if not all(carrier_path(output, t, s["branch"]).is_file() for t in tags)]
    if pending:
        phase_began = time.monotonic()
        adapters = {tag: load_adapter(output, tag) for tag in tags}
        progress = Progress("audit carriers", len(pending))
        with ThreadPoolExecutor(DECODE_THREADS) as pool:
            for shot in pending:
                for tag, value in shot_carriers(adapters, shot["attempt"], pool).items():
                    atomic_torch(carrier_path(output, tag, shot["branch"]),
                                 {"schema": SCHEMA_CARRIERS, "tag": tag, "branch": shot["branch"], "carriers": value})
                progress.step(10)
        del adapters
        torch.cuda.empty_cache()
        mark("audit_carriers", time.monotonic() - phase_began, gpu=time.monotonic() - phase_began)
    # 7. audit and probe records
    phase_began = time.monotonic()
    lineages = sorted({s["member"] for s in shots})
    for encoder in ENCODERS:
        for seed in SEEDS:
            todo = [m for m in lineages if not audit_path(output, encoder, seed, m).is_file()]
            if not todo:
                continue
            predictor = load_hybrid(output, encoder, seed)
            for lineage in todo:
                write_json(audit_path(output, encoder, seed, lineage),
                           audit_record(output, encoder, seed, lineage, [s for s in shots if s["member"] == lineage],
                                        predictor, names))
            log(f"audit {encoder} seed {seed}: {len(todo)} lineages")
    if not (output / "records" / "probe.json").is_file():
        write_json(output / "records" / "probe.json",
                   probe_record(output, [s for s in shots if s["cohort"] == "fit"], names))
    mark("audit_records", time.monotonic() - phase_began, gpu=time.monotonic() - phase_began)
    # 8. rollout records
    phase_began = time.monotonic()
    objective = diag.base.TaskObjective.from_vocabulary(names)
    gpu = 0.0
    for encoder in ENCODERS:
        for seed in SEEDS:
            todo = [s for s in rollout_samples()
                    if not rollout_path(output, encoder, seed, s["state"]["identity"]).is_file()]
            if not todo:
                continue
            model = load_hybrid(output, encoder, seed)
            adapter = load_adapter(output, tag_of(encoder, seed))
            for sample in todo:
                record = rollout_record(output, encoder, seed, sample, model, adapter, objective)
                gpu += record["gpu_seconds"]
                write_json(rollout_path(output, encoder, seed, sample["state"]["identity"]), record)
            log(f"rollout {encoder} seed {seed}: {len(todo)} states")
    mark("rollout", time.monotonic() - phase_began, gpu=gpu)
    # 9. arms: phase 1 (F, J, M), frequencies, phase 2 (T, D)
    stack = ArmStack(output)
    cells = t96.scheduled_cells(universe)
    for encoder in ENCODERS:
        phase_began = time.monotonic()
        gpu = 0.0
        for phase, arms in ((1, PHASE_1_ARMS), (2, PHASE_2_ARMS)):
            if phase == 2:
                path = frequencies_path(output, encoder)
                if not path.is_file():
                    write_json(path, arm_frequencies(output, encoder, universe))
                frequencies = read_json(path)
            progress = Progress(f"arms{phase} {encoder}", len(cells))
            for inventory, member, seed in cells:
                path = arms_path(output, phase, encoder, inventory, member, seed)
                if path.is_file():
                    progress.step(60)
                    continue
                coordinates = None if phase == 1 else frequencies["coordinates"][str(seed)]
                record = arms_record(stack, encoder, seed, inventory, member, universe[inventory][member], arms,
                                     coordinates)
                gpu += sum(block["gpu_seconds"] for block in record["arms"].values())
                write_json(path, record)
                progress.step(60)
        mark(f"arms_{encoder}", time.monotonic() - phase_began, gpu=gpu)
    ledger["status"] = "terminal"
    ledger["wall_seconds_elapsed"] = wall_base + time.monotonic() - began
    write_json(output / "ledger.json", ledger)
    log(f"run complete: gpu {ledger['gpu_seconds_elapsed']:.0f}s; wall {ledger['wall_seconds_elapsed']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def nanmean(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def clustered(counts, statistic):
    """counts: array (clusters, seeds, predicates, 3) of tp/fp/fn; statistic maps summed
    counts (..., seeds, predicates, 3) -> (..., seeds). Seed-mean point estimate and a
    cluster-resampled percentile interval (one resample shared by the seeds)."""
    counts = np.asarray(counts, dtype=float)
    point = float(nanmean(statistic(counts.sum(0))))
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(counts), size=(BOOTSTRAP_DRAWS, len(counts)))
    values = nanmean(statistic(counts[draws].sum(1)), axis=-1)
    finite = values[np.isfinite(values)]
    interval = [float(v) for v in np.quantile(finite, INTERVAL_QUANTILES)] if len(finite) else [None, None]
    return {"mean": point if math.isfinite(point) else None, "interval": interval, "clusters": len(counts),
            "defined_draw_share": len(finite) / BOOTSTRAP_DRAWS, "label": "DESCRIPTIVE"}


def first(function):
    return lambda summed: function(summed)[..., 0]


def mean_f1(summed):
    """F1 averaged over the predicate axis, per seed."""
    return nanmean(f1_of(summed), axis=-1)


def split_f1(count):
    """(mean F1 of the first `count` predicates, mean F1 of the rest), per seed."""
    def pieces(summed):
        f1 = f1_of(summed)
        return nanmean(f1[..., :count], axis=-1), nanmean(f1[..., count:], axis=-1)
    return pieces


def score_of(summed):
    macro, micro = split_f1(2)(summed)
    return (macro + micro) / 2


def difference_of(count):
    def difference(summed):
        a, b = split_f1(count)(summed)
        return a - b
    return difference


def f1_of(summed):
    tp, fp, fn = summed[..., 0], summed[..., 1], summed[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)


def precision_of(summed):
    tp, fp = summed[..., 0], summed[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tp + fp > 0, tp / (tp + fp), np.nan)


def recall_of(summed):
    tp, fn = summed[..., 0], summed[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tp + fn > 0, tp / (tp + fn), np.nan)


def load_audit(output, shots):
    lineages = sorted({s["member"] for s in shots})
    return {(encoder, seed, lineage): read_record(audit_path(output, encoder, seed, lineage), SCHEMA_AUDIT)
            for encoder in ENCODERS for seed in SEEDS for lineage in lineages}


def count_array(audit, encoder, lineages, source, phases, predicates):
    """(clusters, seeds, predicates, 3) tp/fp/fn summed over the phases."""
    out = np.zeros((len(lineages), len(SEEDS), len(predicates), 3))
    for c, lineage in enumerate(lineages):
        for s, seed in enumerate(SEEDS):
            counts = audit[(encoder, seed, lineage)]["counts"][source]
            for q, predicate in enumerate(predicates):
                for phase in phases:
                    row = counts[predicate][PHASES.index(phase)]
                    out[c, s, q] += row[:3]
    return out


def fidelity_tables(audit, shots):
    cohorts = {c: sorted({s["member"] for s in shots if s["cohort"] == c}) for c in COHORTS}
    tables = {}
    for encoder in ENCODERS:
        for cohort, lineages in cohorts.items():
            for source in SOURCES:
                for phase in ("all", *PHASES):
                    phases = PHASES if phase == "all" else (phase,)
                    for predicate in ALL_PREDICATES:
                        counts = count_array(audit, encoder, lineages, source, phases, (predicate,))
                        totals = counts.sum((0, 1))[0] / len(SEEDS)
                        tables[(encoder, cohort, source, phase, predicate)] = {
                            "f1": clustered(counts, first(f1_of)),
                            "precision": clustered(counts, first(precision_of)),
                            "recall": clustered(counts, first(recall_of)),
                            "engine_positives_per_seed": float(totals[0] + totals[2]),
                            "predicted_positives_per_seed": float(totals[0] + totals[1])}
    return cohorts, tables


def readout_target(audit, cohorts, guards_ok):
    rows = {}
    held = cohorts["held_out"]
    for encoder in ENCODERS:
        macro = count_array(audit, encoder, held, "encoder", PHASES, MACRO)
        micro = count_array(audit, encoder, held, "encoder", PHASES, MICRO)
        both = np.concatenate((macro, micro), axis=2)
        rows[encoder] = {"macro_f1": clustered(macro, mean_f1), "micro_edge_f1": clustered(micro, mean_f1),
                         "score": clustered(both, score_of)}
        m, u = rows[encoder]["macro_f1"]["mean"], rows[encoder]["micro_edge_f1"]["mean"]
        rows[encoder]["meets"] = bool(m is not None and u is not None and m >= MACRO_F1_TARGET
                                      and u >= MICRO_F1_TARGET)
    meeting = [e for e in ENCODERS if rows[e]["meets"]]
    pool = meeting or list(ENCODERS)
    better = max(pool, key=lambda e: ((rows[e]["score"]["mean"] or -1.0), -ENCODERS.index(e)))
    token = STOP_TOKEN if not guards_ok else ("supported" if meeting else "not_supported_by_this_experiment")
    return {"encoders": rows, "met_by": meeting, "better_encoder": better, "token": token}


def readout_differences(audit, cohorts):
    """JEPA - F77 and JEPA - E99 in held-out composite F1 (encoder source), paired lineage bootstrap."""
    held = cohorts["held_out"]
    out = {}
    for other in ("F77", "E99"):
        for name, predicates in (("macro_f1", MACRO), ("micro_edge_f1", MICRO)):
            a = count_array(audit, "JEPA", held, "encoder", PHASES, predicates)
            b = count_array(audit, other, held, "encoder", PHASES, predicates)
            stacked = np.concatenate((a, b), axis=2)
            out[f"JEPA-{other}:{name}"] = clustered(stacked, difference_of(len(predicates)))
    return out


def probe_table(probe):
    claims = PROBE["claims"]
    tol = PROBE["numeric_tolerance"]
    lineages = sorted(k for k, v in probe["lineage_max_gated_micro"][str(PROBE["seed"])].items() if v > 0.5)
    rows = []

    def put(identity, claimed, observed, reproduced):
        rows.append({"claim": identity, "claimed": claimed, "observed": observed,
                     "verdict": "reproduced" if reproduced else "corrected"})

    put("C1_bird_presence_frame80_below_0.5", claims["C1_bird_presence_frame80_below_0.5"],
        probe["bird_presence_frame"], all(v < 0.5 for v in probe["bird_presence_frame"]))
    put("C2_pig_presence_frame80_below_0.5", claims["C2_pig_presence_frame80_below_0.5"],
        probe["pig_presence_frame"], probe["pig_presence_frame"] < 0.5)
    put("C3_max_gated_micro_555_frames", claims["C3_max_gated_micro_555_frames"], probe["max_gated_micro"],
        probe["frames"] == PROBE["frames"] and abs(probe["max_gated_micro"] - claims["C3_max_gated_micro_555_frames"]) <= tol)
    put("C4_macro_steady_frame80", claims["C4_macro_steady_frame80"], probe["gated_macro_frame"][0],
        abs(probe["gated_macro_frame"][0] - claims["C4_macro_steady_frame80"]) <= tol)
    put("C5_macro_unstable_frame80", claims["C5_macro_unstable_frame80"], probe["gated_macro_frame"][1],
        abs(probe["gated_macro_frame"][1] - claims["C5_macro_unstable_frame80"]) <= tol)
    put("C6_lineages_with_gated_micro_edge_above_0.5", claims["C6_lineages_with_gated_micro_edge_above_0.5"],
        lineages, lineages == claims["C6_lineages_with_gated_micro_edge_above_0.5"])
    error = probe["lineage_009_center_errors"]["mean"]
    put("C7_lineage_009_positions_mislocalised", claims["C7_lineage_009_positions_mislocalised"], error,
        error is not None and error > PROBE["mislocalisation_threshold"])
    return rows


def rollout_tables(output):
    samples = rollout_samples()
    metrics = ("position_mse", "engine_position_mse", "presence_mse", "carrier_mse")
    per_state = {}
    for encoder in ENCODERS:
        for seed in SEEDS:
            for sample in samples:
                record = read_record(rollout_path(output, encoder, seed, sample["state"]["identity"]), SCHEMA_ROLLOUT)
                for pair in t96.PAIR_NAMES:
                    for t in ROLLOUT_TIMES:
                        for metric in metrics:
                            values = [c["pairs"][pair]["curve"][str(t)][metric] for c in record["candidates"]
                                      if c["pairs"][pair]["curve"].get(str(t)) is not None
                                      and c["pairs"][pair]["curve"][str(t)][metric] is not None]
                            per_state[(encoder, seed, sample["state"]["identity"], pair, t, metric)] = (
                                float(np.mean(values)) if values else None)
    states = [s["state"]["identity"] for s in samples]
    curves = {}
    for encoder in ENCODERS:
        for pair in t96.PAIR_NAMES:
            for t in ROLLOUT_TIMES:
                for metric in metrics:
                    seeds = {}
                    for seed in SEEDS:
                        values = [per_state[(encoder, seed, s, pair, t, metric)] for s in states]
                        seeds[str(seed)] = float(np.mean([v for v in values if v is not None])) \
                            if any(v is not None for v in values) else None
                    finite = [v for v in seeds.values() if v is not None]
                    curves[(encoder, pair, t, metric)] = {"per_seed": seeds,
                                                          "mean": float(np.mean(finite)) if len(finite) == len(SEEDS)
                                                          else None}
    contrasts = {}
    for other in ("F77", "E99"):
        for pair in t96.PAIR_NAMES:
            for metric in ("position_mse", "engine_position_mse"):
                values, clusters = [], []
                for s in states:
                    a = [per_state[("JEPA", seed, s, pair, ROLLOUT_ENDPOINT, metric)] for seed in SEEDS]
                    b = [per_state[(other, seed, s, pair, ROLLOUT_ENDPOINT, metric)] for seed in SEEDS]
                    if all(v is not None for v in a + b):
                        values.append(float(np.mean(a) - np.mean(b)))
                        clusters.append(s)
                contrasts[(other, pair, metric)] = t96.bootstrap(values, clusters)
    return per_state, curves, contrasts


def rollout_replication(curves):
    summary = read_json(DIAGNOSTIC / "summary.json")
    rows, worst = [], 0.0
    for system, pair in TAB_SYSTEMS.items():
        for seed in SEEDS:
            published = summary["per_seed"][str(seed)][system]["curves"][str(ROLLOUT_ENDPOINT)]["position_mse"]
            fresh = curves[("F77", pair, ROLLOUT_ENDPOINT, "position_mse")]["per_seed"][str(seed)]
            relative = abs(fresh - published) / abs(published)
            worst = max(worst, relative)
            rows.append({"system": system, "seed": seed, "published": published, "fresh": fresh,
                         "relative_delta": relative})
    return {"rows": rows, "worst_relative_delta": worst}


def arm_cells(output, universe, encoder, inventory, members):
    rows = []
    for member in members:
        entry = universe[inventory].get(member)
        if entry is None or not t96.mixed(entry):
            continue
        for seed in SEEDS:
            views = {}
            blocks = {}
            for phase in (1, 2):
                blocks.update(read_record(arms_path(output, phase, encoder, inventory, member, seed),
                                          SCHEMA_ARMS)["arms"])
            for arm in ARMS:
                for cost in COSTS:
                    views[(cost, arm)] = m99.view(entry, blocks[arm]["rows"], cost)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def arm_tables(output, universe, members):
    cohorts = m99.cohort_members(members)
    out = {}
    for encoder in ENCODERS:
        for inventory in INVENTORIES:
            for cohort in ARM_COHORTS:
                rows = arm_cells(output, universe, encoder, inventory, cohorts[cohort])
                block = {"cells": len(rows), "members": sorted({r["member"] for r in rows})}
                for cost in COSTS:
                    block[cost] = {
                        "auc": {arm: m99.block_of(rows, (cost, arm)) for arm in ARMS},
                        "tied_share": {arm: (sum(r["views"][(cost, arm)]["all_tied"] for r in rows) / len(rows)
                                             if rows else None) for arm in ("J", "T", "D", "M")},
                        "contrasts": {name: m99.paired(rows, [(1, (cost, "J")), (-1, (cost, name.split("-")[1]))])
                                      for name in ("J-T", "J-D", "J-M")}}
                out[(encoder, inventory, cohort)] = block
    return out


def contrast_token(estimate, guards_ok):
    if not guards_ok or estimate is None:
        return STOP_TOKEN
    if estimate["mean"] >= DELTA and estimate["interval"][0] > 0:
        return "supported"
    if estimate["interval"][1] < DELTA:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def arms_replication(output, universe):
    worst, checked = 0.0, 0
    for inventory, member, seed in t96.scheduled_cells(universe):
        blocks = {}
        for phase in (1, 2):
            blocks.update(read_record(arms_path(output, phase, "F77", inventory, member, seed), SCHEMA_ARMS)["arms"])
        for arm in ARMS:
            retained = t96.read_record(t96.OUTPUT, inventory, member, seed, arm)["decision"]["ranking"]
            mine = {row["ordinal"]: row["count"] for row in blocks[arm]["rows"]}
            for row in retained:
                if row["predicted_cost"] is None or mine.get(row["ordinal"]) is None:
                    if (row["predicted_cost"] is None) != (mine.get(row["ordinal"]) is None):
                        worst = math.inf
                    continue
                worst = max(worst, abs(mine[row["ordinal"]] - row["predicted_cost"]))
                checked += 1
    retained = read_json(t96.OUTPUT / "frequencies.json")["coordinates"]
    fresh = read_json(frequencies_path(output, "F77"))["coordinates"]
    same = all(retained[str(s)][k] == fresh[str(s)][k] for s in SEEDS
               for k in ("alpha_T_index", "delta_D_index"))
    return {"max_abs_count_cost_delta": worst, "candidates_checked": checked, "coordinates_equal": same}


def shard_guard(output, shots):
    worst = 0.0
    for shot in (s for s in shots if s["cohort"] == "fit"):
        worst = max(worst, shard_replication(load_carriers(output, "F77", shot["branch"]), shot))
    return worst


def build_guard(output):
    shot = {"attempt": str(CAMPAIGN / "attempts" / PROBE["capture"])}
    worst = 0.0
    for seed in SEEDS:
        encoder, _ = load_jepa(output, seed)
        carriers = load_carriers(output, tag_of("JEPA", seed), PROBE["capture"])
        worst = max(worst, carrier_build_replication(encoder, shot["attempt"], carriers))
    return worst


def training_summary(output):
    jepa, controllers = [], []
    for seed in SEEDS:
        payload = torch.load(jepa_path(output, seed), map_location="cpu", weights_only=True)
        jepa.append({"seed": seed, "steps": payload["steps"], "pair_counts": payload["pair_counts"],
                     "final": payload["loss_trace"][-1], "windows": payload["windows"],
                     "parameters": payload["parameters"], "gpu_seconds": payload["gpu_seconds"]})
    for encoder in ("E99", "JEPA"):
        for seed in SEEDS:
            payload = torch.load(controller_path(output, encoder, seed), map_location="cpu", weights_only=True)
            controllers.append({"encoder": encoder, "seed": seed, "steps_per_round": payload["steps_per_round"],
                                "lineages": payload["lineages"], "label_rows": payload["label_rows"],
                                "final_loss": payload["loss_trace"][-1]["loss"], "parameters": payload["parameters"],
                                "gpu_seconds": payload["gpu_seconds"]})
    return {"jepa": jepa, "controllers": controllers}


def arm_compute(output, universe):
    out = {}
    for encoder in ENCODERS:
        per_arm = {arm: {"records": 0, "candidates": 0, "transition_calls": 0, "controller_calls": 0,
                         "linear_macs": 0, "gpu_seconds": 0.0} for arm in ARMS}
        for inventory, member, seed in t96.scheduled_cells(universe):
            for phase in (1, 2):
                record = read_record(arms_path(output, phase, encoder, inventory, member, seed), SCHEMA_ARMS)
                for arm, block in record["arms"].items():
                    row = per_arm[arm]
                    row["records"] += 1
                    row["candidates"] += block["candidates"]
                    for key in ("transition_calls", "controller_calls", "linear_macs", "gpu_seconds"):
                        row[key] += block[key]
        out[encoder] = per_arm
    return out


def compute_tables(output, plan):
    output = Path(output)
    shots = audit_shots()
    sources, universe, fit_shots = fit_data()
    members = [state["member"] for state in sources["plan93"]["states"]]
    ledger = read_json(output / "ledger.json")
    audit = load_audit(output, shots)
    cohorts, fidelity = fidelity_tables(audit, shots)
    probe = read_record(output / "records" / "probe.json", SCHEMA_PROBE)
    per_state, curves, rollout_contrasts = rollout_tables(output)
    arms = arm_tables(output, universe, members)
    training = training_summary(output)
    g2 = rollout_replication(curves)
    g3 = arms_replication(output, universe)
    held_positives = {}
    for predicate in PREDICATES:
        counts = count_array(audit, "F77", cohorts["held_out"], "encoder", PHASES, (predicate,))
        held_positives[predicate] = float((counts[:, 0, 0, 0] + counts[:, 0, 0, 2]).sum())
    controller_members = set()
    for row in training["controllers"]:
        controller_members |= set(row["lineages"])
    fit_members = {s["member"] for s in fit_shots}
    guards = {
        "G1": {"observed": {"max_abs_delta": shard_guard(output, shots), "tolerance": PARSE_TOLERANCE}},
        "G2": {"observed": {**g2, "tolerance": ROLLOUT_RELATIVE_TOLERANCE}},
        "G3": {"observed": {**g3, "tolerance": COST_TOLERANCE}},
        "G4": {"observed": {"fit_members": sorted(fit_members), "controller_lineages": sorted(controller_members)}},
        "G5": {"observed": {"jepa_steps": [r["steps"] for r in training["jepa"]],
                            "jepa_final": [r["final"] for r in training["jepa"]],
                            "controllers": len(training["controllers"])}},
        "G6": {"observed": {"max_abs_delta": build_guard(output), "tolerance": CARRIER_BUILD_TOLERANCE}},
        "G7": {"observed": {"held_out_engine_positives": held_positives, "minimum": LABEL_POSITIVES_MIN}},
        "caps": {"observed": {"gpu_seconds": ledger["gpu_seconds_elapsed"],
                              "wall_seconds": ledger["wall_seconds_elapsed"], "status": ledger["status"]}},
    }
    guards["G1"]["pass"] = guards["G1"]["observed"]["max_abs_delta"] <= PARSE_TOLERANCE
    guards["G2"]["pass"] = g2["worst_relative_delta"] <= ROLLOUT_RELATIVE_TOLERANCE
    guards["G3"]["pass"] = g3["max_abs_count_cost_delta"] <= COST_TOLERANCE and g3["coordinates_equal"]
    guards["G4"]["pass"] = not ((fit_members | controller_members) & set(HELD_OUT))
    guards["G5"]["pass"] = (all(r["steps"] == JEPA["steps"] and math.isfinite(r["final"]["jepa_loss"])
                                and math.isfinite(r["final"]["parser_loss"]) for r in training["jepa"])
                            and len(training["jepa"]) == len(SEEDS) and len(training["controllers"]) == 2 * len(SEEDS)
                            and all(r["steps_per_round"] == CONTROLLER["steps_per_round"]
                                    and math.isfinite(r["final_loss"]) for r in training["controllers"]))
    guards["G6"]["pass"] = guards["G6"]["observed"]["max_abs_delta"] <= CARRIER_BUILD_TOLERANCE
    guards["G7"]["pass"] = all(v >= LABEL_POSITIVES_MIN for v in held_positives.values())
    guards["caps"]["pass"] = (ledger["status"] == "terminal" and ledger["gpu_seconds_elapsed"] <= GPU_CAP_SECONDS
                              and ledger["wall_seconds_elapsed"] <= WALL_CAP_SECONDS)
    guards_ok = all(g["pass"] for g in guards.values())
    target = readout_target(audit, cohorts, guards_ok)
    better = target["better_encoder"]
    alpha = {}
    for inventory in INVENTORIES:
        block = arms[(better, inventory, PRIMARY_ARM_COHORT)][PRIMARY_COST]["contrasts"]
        alpha[inventory] = {name: {"estimate": block[name]["estimate"], "cells": block[name]["cells"],
                                   "members": block[name]["members"],
                                   "token": contrast_token(block[name]["estimate"], guards_ok)}
                            for name in ("J-T", "J-D")}
    frequencies = {encoder: read_json(frequencies_path(output, encoder)) for encoder in ENCODERS}
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "structure": {"audit_shots": len(shots), "fit_shots": len(fit_shots), "cohorts": cohorts},
        "guards": guards, "guards_pass": guards_ok,
        "probe": {"record": probe, "claims": probe_table(probe)},
        "fidelity": {"|".join(k): v for k, v in fidelity.items()},
        "readout_target": target,
        "readout_differences": readout_differences(audit, cohorts),
        "phase_frames": phase_frames(output, shots),
        "rollout": {"curves": {"|".join(map(str, k)): v for k, v in curves.items()},
                    "contrasts": {"|".join(k): v for k, v in rollout_contrasts.items()}},
        "arms": {"tables": {"|".join(k): v for k, v in arms.items()},
                 "frequencies": {e: {"counts": f["counts"], "coordinates": f["coordinates"]}
                                 for e, f in frequencies.items()},
                 "alpha_effect": {"encoder": better, "cohort": PRIMARY_ARM_COHORT, "cost": PRIMARY_COST,
                                  "inventories": alpha},
                 "compute": arm_compute(output, universe)},
        "training": training, "parameters": plan["parameters"],
        "compute": {"phases": ledger["phases"], "run_gpu_seconds": ledger["gpu_seconds_elapsed"],
                    "run_wall_seconds": ledger["wall_seconds_elapsed"], "engine_seconds": 0},
    }


def phase_frames(output, shots):
    out = {c: {p: 0 for p in PHASES} for c in COHORTS}
    for shot in shots:
        labels = load_labels(output, shot["branch"])
        phase = labels["phase"][shard_positions(labels["frames"] - 1)]
        for i, name in enumerate(PHASES):
            out[shot["cohort"]][name] += int((phase == i).sum())
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def headline_source(predicate, delta):
    level = predicate.split(":", 1)[0] if predicate.startswith("presence:") else predicate
    return f"{delta}-{MATCHED_ALPHA[level]}"


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "section", "encoder", "cohort", "source", "phase", "predicate", "statistic", "value",
                     "interval_low", "interval_high", "interval_label", "clusters", "detail"])

    def put(identity, section, statistic, value, block=None, encoder="", cohort="", source="", phase="",
            predicate="", detail=""):
        writer.writerow([identity, section, encoder, cohort, source, phase, predicate, statistic,
                         "" if value is None else value,
                         "" if not block or block["interval"][0] is None else block["interval"][0],
                         "" if not block or block["interval"][1] is None else block["interval"][1],
                         "DESCRIPTIVE" if block else "", block["clusters"] if block else "", detail])

    target = compute["readout_target"]
    put("Q2_token", "readout_target", "token", target["token"], detail=json.dumps(target["met_by"]))
    put("Q2_better_encoder", "readout_target", "better encoder", target["better_encoder"])
    for encoder, row in target["encoders"].items():
        for name in ("macro_f1", "micro_edge_f1", "score"):
            put(f"Q2_{encoder}_{name}", "readout_target", name, row[name]["mean"], row[name], encoder, "held_out",
                "encoder", "all")
    for name, block in compute["readout_differences"].items():
        put(f"Q3_readout_{name}", "readout_difference", name, block["mean"], block, cohort="held_out", source="encoder")
    for guard, block in compute["guards"].items():
        put(f"guard_{guard}", "guards", f"{guard} pass", block["pass"], detail=json.dumps(block["observed"], sort_keys=True))
    for row in compute["probe"]["claims"]:
        put(f"probe_{row['claim']}", "probe", row["verdict"], json.dumps(row["observed"]),
            detail=json.dumps(row["claimed"]))
    for key, block in compute["fidelity"].items():
        encoder, cohort, source, phase, predicate = key.split("|")
        for statistic in ("f1", "precision", "recall"):
            put(f"fidelity_{key}_{statistic}".replace("|", "_"), "fidelity", statistic, block[statistic]["mean"],
                block[statistic], encoder, cohort, source, phase, predicate,
                json.dumps({"engine_positives_per_seed": block["engine_positives_per_seed"],
                            "predicted_positives_per_seed": block["predicted_positives_per_seed"]}))
    for key, block in compute["rollout"]["curves"].items():
        encoder, pair, t, metric = key.split("|")
        put(f"rollout_{encoder}_{pair}_{t}_{metric}", "rollout", metric, block["mean"], encoder=encoder, source=pair,
            phase=t, detail=json.dumps(block["per_seed"], sort_keys=True))
    for key, block in compute["rollout"]["contrasts"].items():
        other, pair, metric = key.split("|")
        put(f"rollout_JEPA-{other}_{pair}_{metric}", "rollout_contrast", f"JEPA - {other} {metric} at "
            f"{ROLLOUT_ENDPOINT}", block["mean"] if block else None, block, "JEPA", source=pair)
    for key, block in compute["arms"]["tables"].items():
        encoder, inventory, cohort = key.split("|")
        for cost in COSTS:
            for arm, auc in block[cost]["auc"].items():
                put(f"arms_{key}_{cost}_auc_{arm}".replace("|", "_"), "arms", f"AUC {arm}", auc and auc["mean"], auc,
                    encoder, cohort, cost, inventory)
            for name, c in block[cost]["contrasts"].items():
                est = c["estimate"]
                put(f"arms_{key}_{cost}_{name}".replace("|", "_"), "arms", name, est and est["mean"], est, encoder,
                    cohort, cost, inventory, detail=json.dumps({"cells": c["cells"], "members": c["members"]}))
    for inventory, row in compute["arms"]["alpha_effect"]["inventories"].items():
        for name, c in row.items():
            put(f"Q4_{inventory}_{name}", "alpha_effect", f"{name} token", c["token"], c["estimate"],
                compute["arms"]["alpha_effect"]["encoder"], PRIMARY_ARM_COHORT, PRIMARY_COST, inventory)
    return buffer.getvalue()


def f1_cell(block):
    b = block["f1"]
    return "n/a" if b["mean"] is None else fmt_interval(b) if b["interval"][0] is not None else fmt(b["mean"])


def findings_md(plan, compute):
    lines = []
    add = lines.append
    target = compute["readout_target"]
    fid = compute["fidelity"]
    add("# Issue-100: readout fidelity audit and jointly trained JEPA encoder — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v1 frozen {plan['frozen_at']} before any training or scoring")
    add(f"- validation command: `{VALIDATION_COMMAND}`; zero engine seconds; every interval DESCRIPTIVE "
        f"(lineage- or member-clustered percentile bootstrap, 10000 draws, PCG64 7201)")
    add(f"- disclosure: {plan['disclosure']}")
    add("")
    add("## Dispositions")
    add("")
    add(f"- **Q2 readout target (held-out, encoder readout: macro F1 >= {MACRO_F1_TARGET} and micro edge F1 >= "
        f"{MICRO_F1_TARGET}): {target['token']}**; met by: {', '.join(target['met_by']) or 'none'}; better encoder "
        f"(frozen rule): **{target['better_encoder']}**")
    alpha = compute["arms"]["alpha_effect"]
    for inventory, row in alpha["inventories"].items():
        add(f"- **Q4 {inventory} ({alpha['encoder']}, held-out, tie-free):** J − T {fmt_interval(row['J-T']['estimate'])}"
            f" → {row['J-T']['token']}; J − D {fmt_interval(row['J-D']['estimate'])} → {row['J-D']['token']} "
            f"({row['J-T']['cells']} cells / {row['J-T']['members']} members)")
    add("")
    add("## Guards")
    add("")
    add("| guard | pass | observed |")
    add("|---|---|---|")
    for guard, block in compute["guards"].items():
        add(f"| {guard} | {block['pass']} | `{json.dumps(block['observed'], sort_keys=True)[:600]}` |")
    add("")
    add("## Probe reproduction (F77, seed 20260908, capture issue-77-n1-001-a06)")
    add("")
    add("| claim | claimed | observed | verdict |")
    add("|---|---|---|---|")
    for row in compute["probe"]["claims"]:
        add(f"| {row['claim']} | {json.dumps(row['claimed'])} | {json.dumps(row['observed'])} | {row['verdict']} |")
    probe = compute["probe"]["record"]
    add("")
    add(f"- frame {probe['frame']}: engine bird presence {probe['bird_engine_presence_frame']}, engine pig presence "
        f"{probe['pig_engine_presence_frame']}; engine macro {probe['engine_macro_frame']} (mask "
        f"{probe['engine_macro_mask_frame']}); raw (ungated) macro sigmoid {[round(v, 4) for v in probe['raw_macro_frame']]}")
    add(f"- max gated micro {probe['max_gated_micro']:.4f} at frame {probe['max_gated_micro_frame']} of "
        f"{probe['frames']}; per-lineage maxima over the shard frames by seed: "
        f"`{json.dumps({s: {k.rsplit('-', 1)[1]: round(v, 3) for k, v in m.items()} for s, m in probe['lineage_max_gated_micro'].items()})}`")
    add("")
    add("## Readout target (held-out lineages, encoder readout, all phases)")
    add("")
    add("| encoder | macro F1 | micro edge F1 | score | meets |")
    add("|---|---|---|---|---|")
    for encoder, row in target["encoders"].items():
        add(f"| {encoder} | {fmt_interval(row['macro_f1'])} | {fmt_interval(row['micro_edge_f1'])} | "
            f"{fmt_interval(row['score'])} | {row['meets']} |")
    add("")
    for name, block in compute["readout_differences"].items():
        add(f"- {name}: {fmt_interval(block)}")
    add("")
    add(f"Phase frames (audited rows): `{json.dumps(compute['phase_frames'])}`")
    add("")
    for cohort in COHORTS:
        add(f"## Fidelity table — {cohort} lineages (F1, seed mean [lineage-clustered interval])")
        add("")
        add("Encoder rows read the heads (presence: carrier) on the encoded carrier; Δ rows read them after one "
            "request (Δ, α) prediction with α matched to the level (presence: continuous, micro: micro, macro: macro).")
        add("")
        header = "| encoder | source | phase | " + " | ".join(ALL_PREDICATES) + " |"
        add(header)
        add("|---|---|---|" + "---|" * len(ALL_PREDICATES))
        for encoder in ENCODERS:
            for source_label in ("encoder", *(f"Δ{d}" for d in DELTAS)):
                for phase in ("all", *PHASES):
                    cells = []
                    for predicate in ALL_PREDICATES:
                        source = "encoder" if source_label == "encoder" else headline_source(
                            predicate, int(source_label[1:]))
                        cells.append(f1_cell(fid[f"{encoder}|{cohort}|{source}|{phase}|{predicate}"]))
                    add(f"| {encoder} | {source_label} | {phase} | " + " | ".join(cells) + " |")
        add("")
        add(f"### Precision / recall — {cohort}, encoder readout, all phases")
        add("")
        add("| encoder | predicate | precision | recall | engine positives / seed | predicted positives / seed |")
        add("|---|---|---|---|---|---|")
        for encoder in ENCODERS:
            for predicate in ALL_PREDICATES:
                block = fid[f"{encoder}|{cohort}|encoder|all|{predicate}"]
                add(f"| {encoder} | {predicate} | {fmt_interval(block['precision']) if block['precision']['mean'] is not None and block['precision']['interval'][0] is not None else 'n/a'} | "
                    f"{fmt_interval(block['recall']) if block['recall']['mean'] is not None and block['recall']['interval'][0] is not None else 'n/a'} | "
                    f"{block['engine_positives_per_seed']:.0f} | {block['predicted_positives_per_seed']:.0f} |")
        add("")
    add(f"## Rollout error (#77 diagnostic protocol, held-out states, t = {ROLLOUT_ENDPOINT}, three-seed mean)")
    add("")
    add("| request | " + " | ".join(f"{e} own-parse" for e in ENCODERS) + " | "
        + " | ".join(f"{e} engine" for e in ENCODERS) + " |")
    add("|---|" + "---|" * (2 * len(ENCODERS)))
    curves = compute["rollout"]["curves"]
    for pair in t96.PAIR_NAMES:
        own = [fmt(curves[f"{e}|{pair}|{ROLLOUT_ENDPOINT}|position_mse"]["mean"], 5) for e in ENCODERS]
        eng = [fmt(curves[f"{e}|{pair}|{ROLLOUT_ENDPOINT}|engine_position_mse"]["mean"], 5) for e in ENCODERS]
        add(f"| ({pair.replace('-', ', ')}) | " + " | ".join(own) + " | " + " | ".join(eng) + " |")
    add("")
    add("Own-parse = the tab:res:rollout estimand (target = the encoder's own parse of the frame); engine = "
        "error against engine-projected centers of engine-present slots (encoder-independent target).")
    add("")
    add("| request | JEPA − F77 engine | JEPA − E99 engine |")
    add("|---|---|---|")
    contrasts = compute["rollout"]["contrasts"]
    for pair in t96.PAIR_NAMES:
        add(f"| ({pair.replace('-', ', ')}) | {fmt_interval(contrasts[f'F77|{pair}|engine_position_mse'])} | "
            f"{fmt_interval(contrasts[f'E99|{pair}|engine_position_mse'])} |")
    add("")
    add("## #96 arms per encoder (J − T, J − D, J − M; cost tie_free)")
    add("")
    for cohort in ARM_COHORTS:
        add(f"### cohort {cohort}")
        add("")
        add("| encoder | inventory | cells | AUC J | AUC T | AUC D | AUC M | J − T | J − D | J − M |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for encoder in ENCODERS:
            for inventory in INVENTORIES:
                block = compute["arms"]["tables"][f"{encoder}|{inventory}|{cohort}"]
                t = block[PRIMARY_COST]
                add(f"| {encoder} | {inventory} | {block['cells']} | " + " | ".join(
                    fmt_interval(t["auc"][a]) for a in ("J", "T", "D", "M")) + " | " + " | ".join(
                    fmt_interval(t["contrasts"][n]["estimate"]) for n in ("J-T", "J-D", "J-M")) + " |")
        add("")
    add("J step-pair frequencies (outcome-free) and T/D coordinates:")
    add("")
    for encoder, block in compute["arms"]["frequencies"].items():
        add(f"- {encoder}: " + "; ".join(f"seed {s}: α_T {c['alpha_T']}, Δ_D {c['delta_D']}, counts "
                                         f"{block['counts'][s]}" for s, c in block["coordinates"].items()))
    add("")
    add("## Training, parameters and compute")
    add("")
    for encoder, row in compute["parameters"].items():
        add(f"- {encoder}: `{json.dumps(row)}`")
    add("")
    add("| JEPA seed | steps | final jepa loss | final parser loss | windows | GPU s |")
    add("|---|---|---|---|---|---|")
    for row in compute["training"]["jepa"]:
        add(f"| {row['seed']} | {row['steps']} | {row['final']['jepa_loss']:.5f} | {row['final']['parser_loss']:.4f} | "
            f"{row['windows']} | {row['gpu_seconds']:.0f} |")
    add("")
    add("| controller | seed | lineages | final loss | GPU s |")
    add("|---|---|---|---|---|")
    for row in compute["training"]["controllers"]:
        add(f"| {row['encoder']} | {row['seed']} | {len(row['lineages'])} | {row['final_loss']:.4f} | "
            f"{row['gpu_seconds']:.0f} |")
    add("")
    add("| encoder | arm | records | candidates | transition calls | controller calls | linear MACs | GPU s |")
    add("|---|---|---|---|---|---|---|---|")
    for encoder, per_arm in compute["arms"]["compute"].items():
        for arm, row in per_arm.items():
            add(f"| {encoder} | {arm} | {row['records']} | {row['candidates']} | {row['transition_calls']} | "
                f"{row['controller_calls']} | {row['linear_macs']} | {row['gpu_seconds']:.1f} |")
    add("")
    c = compute["compute"]
    add("| phase | seconds | GPU s |")
    add("|---|---|---|")
    for phase, block in c["phases"].items():
        add(f"| {phase} | {block['seconds']:.0f} | {block.get('gpu_seconds', 0.0):.0f} |")
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
            "validation_command": VALIDATION_COMMAND, "questions": plan["questions"], "disclosure": plan["disclosure"],
            "guards": compute["guards"], "guards_pass": compute["guards_pass"], "probe": compute["probe"]["claims"],
            "readout_target": compute["readout_target"], "readout_differences": compute["readout_differences"],
            "fidelity_headline": {k: v for k, v in compute["fidelity"].items()
                                  if k.split("|")[3] == "all" and k.split("|")[2] in
                                  ("encoder", *(headline_source(k.split("|")[4], d) for d in DELTAS))},
            "rollout_endpoint": {k: v for k, v in compute["rollout"]["curves"].items()
                                 if k.split("|")[2] == str(ROLLOUT_ENDPOINT)},
            "rollout_contrasts": compute["rollout"]["contrasts"],
            "alpha_effect": compute["arms"]["alpha_effect"], "training": compute["training"],
            "parameters": compute["parameters"], "compute": compute["compute"],
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
    log(f"published: readout target {compute['readout_target']['token']} (better encoder "
        f"{compute['readout_target']['better_encoder']}); guards pass {compute['guards_pass']}")
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
    log(f"validation passed: tables re-derived from retained audit/probe/rollout/arms records, checkpoints and "
        f"#77/#96 records; byte-compared ({time.monotonic() - began:.1f}s)")
    return 0


# ---------------------------------------------------------------------------
# review gallery (data/): engine vs decoded micro/macro on held-out frames
# ---------------------------------------------------------------------------

def gallery(output):
    output = Path(output)
    load_plan(output)
    names = vocabulary()
    GALLERY.mkdir(parents=True, exist_ok=True)
    predictors = {e: load_hybrid(output, e, SEEDS[0]) for e in ENCODERS}
    entries = []
    shots = [s for s in audit_shots() if s["cohort"] == "held_out" and s["branch"].endswith("-a06")]
    shots.append(next(s for s in audit_shots() if s["branch"] == PROBE["capture"]))
    for shot in shots:
        labels = load_labels(output, shot["branch"])
        _, observation_root, frames, _ = s99.shot_frames(shot["attempt"])
        for phase_index, phase in enumerate(PHASES):
            where = (labels["phase"] == phase_index).nonzero().flatten().tolist()
            if not where:
                entries.append({"branch": shot["branch"], "phase": phase, "frame": None, "status": "phase absent"})
                continue
            p = where[len(where) // 2]
            panels = [draw_panel(observation_root / frames[p]["agent_observation"]["relative_path"],
                                 labels["centers"][p], labels["presence"][p], labels["relations"][p].float(),
                                 [bool(v) for v in labels["macros"][p]], "engine")]
            for encoder in ENCODERS:
                z = load_carriers(output, tag_of(encoder, SEEDS[0]), shot["branch"])[p:p + 1].to(DEVICE)
                with torch.no_grad():
                    micro, macro = gated(predictors[encoder], z)
                centers = z[0, 2:2 + 13 * len(names)].reshape(len(names), 13)[:, 5:7].cpu()
                presence = z[0, 2::13][:len(names)].cpu() >= READ_THRESHOLD
                panels.append(draw_panel(observation_root / frames[p]["agent_observation"]["relative_path"], centers,
                                         presence, micro[0].cpu(), [round(float(v), 3) for v in macro[0]],
                                         f"{encoder} seed {SEEDS[0]}"))
            width = sum(panel.width for panel in panels)
            sheet = Image.new("RGB", (width, panels[0].height), "white")
            x = 0
            for panel in panels:
                sheet.paste(panel, (x, 0))
                x += panel.width
            name = f"{shot['branch']}--{phase}--frame{p:03d}.png"
            sheet.save(GALLERY / name)
            entries.append({"branch": shot["branch"], "phase": phase, "frame": p, "image": name, "status": "rendered"})
    manifest = {"schema": "issue_100_gallery_v1", "identity": IDENTITY, "entries": entries,
                "panels": ["engine labels", *(f"{e} seed {SEEDS[0]} decoded" for e in ENCODERS)],
                "legend": "dots: slot centers (green present); red lines: contact >= 0.5; blue arrows: supports >= 0.5"}
    write_json(GALLERY / "manifest.json", manifest)
    rows = "".join(f"<tr><td>{html.escape(e['branch'])}</td><td>{e['phase']}</td><td>{e['frame']}</td><td>"
                   + (f"<img src='{e['image']}' width='1280'>" if e.get("image") else e["status"]) + "</td></tr>"
                   for e in entries)
    (GALLERY / "index.html").write_text(f"<html><body><h1>{IDENTITY} readout gallery</h1><p>{manifest['legend']}</p>"
                                        f"<table border=1>{rows}</table></body></html>")
    log(f"gallery: {len(entries)} entries under {GALLERY}")
    return 0


def draw_panel(path, centers, presence, relations, macros, title):
    with Image.open(path) as image:
        panel = image.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(panel)
    points = [(float(c[0]) * 320, float(c[1]) * 240) for c in centers]
    count = len(points)
    for i in range(count):
        for j in range(count):
            if i == j or not (presence[i] and presence[j]):
                continue
            if i < j and float(relations[i, j, 0]) >= READ_THRESHOLD:
                draw.line([points[i], points[j]], fill=(220, 30, 30), width=2)
            if float(relations[i, j, 1]) >= READ_THRESHOLD:
                draw.line([points[i], points[j]], fill=(30, 60, 220), width=1)
                draw.ellipse([points[j][0] - 3, points[j][1] - 3, points[j][0] + 3, points[j][1] + 3], outline=(30, 60, 220))
    for i, point in enumerate(points):
        colour = (30, 200, 30) if presence[i] else (160, 160, 160)
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=colour)
    draw.rectangle([0, 0, 320, 24], fill=(255, 255, 255))
    draw.text((3, 2), f"{title}", fill=(0, 0, 0))
    draw.text((3, 12), f"steady {macros[0]}  unstable {macros[1]}", fill=(0, 0, 0))
    return panel


def dry_run(output):
    shots = audit_shots()
    by = Counter(s["cohort"] for s in shots)
    _, universe, fit_shots = fit_data()
    windows_list = fit_windows_for(fit_shots)
    frames = sum(len(n) for n in needed_positions(windows_list, len(fit_shots)))
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    log(f"  audit shots {len(shots)} {dict(by)}; lineages {len({s['member'] for s in shots})}")
    log(f"  fit shots {len(fit_shots)}; JEPA windows {len(windows_list)}; frame cache {frames} frames "
        f"({frames * 3 * 240 * 320 / 1e9:.1f} GB at {FRAME_CACHE})")
    log(f"  JEPA recipe {JEPA}")
    log(f"  controllers {CONTROLLER} on lineages {controller_lineages()}")
    log(f"  rollout states {[s['state']['identity'] for s in rollout_samples()]}; arms cells "
        f"{len(t96.scheduled_cells(universe))} x {len(ENCODERS)} encoders x {len(ARMS)} arms")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": (dry_run, False), "smoke": (smoke, True), "prepare": (prepare, False),
         "run": (run, True), "publish": (publish, True), "validate": (validate, True),
         "gallery": (gallery, True)}


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
