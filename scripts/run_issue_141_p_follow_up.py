"""Issue #141 (#127 P-follow-up): F1 = frozen U + the A-family's discriminative success readout, with the
no-rollout control F1-D, on the sealed NovPhy launch-ranking grid.

Pre-registered in the #141 ticket body (binding); implementation decisions posted on the ticket before their steps
(issuecomment-6480032792 implementation, -6480033291 strata cut points, -6480065190 compute / queues):

- **F1 = U+FiLM**: A2's classifier (``A2Net``, base width 70, carrier_dim 472) on [z0, zhat_225^U_s(z0, a)] and
  ``launch4(a)``; zhat from the frozen seed-s Gate-B unit U (#129 ``UnitRollout``: request F-15-macro, 15 transitions
  to frame 225). U's fit / policy endpoints are cached once per U seed (``features/U-seed<s>.pt``); evaluation and
  #110 endpoints are rolled inside the scoring process.
- **F1-D**: the same classifier, initialisation and sample stream on [z0, z0]; trained independently.
- A2's recipe unchanged (``world_model/baselines/phyre_dqn/novphy.py``): class-balanced BCE, batch 64 = 32 + 32 with
  replacement, 9 000 updates, Adam, cosine, scheduler step before optimizer step; one example per fit shot.
- Selection: lr 3e-4, 1e-4, 1e-3 (declaration order) per arm at seed 20260908 on the 94 policy grid mixed cells
  (``eb.select_configuration``); seeds 20260909/10 at the selected configuration; the selection run is reused.
- Guards (declared scope): global isolation / chronology / scoring-policy / records (imported #128 / #129 records
  hash-checked inside G_records); arm guards G_rank, G_nonfinite, G_trained; extra guards G_u_identity and
  G_rollout_replication (F1), G_alignment and G_classifier_only (both). G-action is inherited from U and reported
  in ``extra.scope``, never an F1 guard.
- Contrasts: F1 - U (arm block, primary); within family F1 - F1-D, F1 - A1, F1 - A2, F1 - U+head (#129); F1 - NN-px /
  NN-car descriptive; distance strata (frozen policy cut points) and #110 novelty (F1, F1-D, #129's U+head)
  descriptive.

Modes (in order): --cut-points (CPU), --selection --gpu {2,3}, --selection-readouts, --smoke, --prepare, --train
--gpu g, --readouts --gpu g, --freeze-readouts, --score, --score-novelty, --publish, --validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import external_baselines as eb  # noqa: E402
from scripts import run_issue_128_direct_predictors as s128  # noqa: E402
from scripts import run_issue_129_slotformer as s129  # noqa: E402
from world_model.baselines.phyre_dqn import novphy as nv  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 141
IDENTITY = "issue-141-p-follow-up-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_141_p_follow_up.py"
TESTS = "tests/test_issue_141_p_follow_up.py"
SCHEMA_MODEL = "issue_141_model_v1"
SCHEMA_FEATURES = "issue_141_features_v1"
SCHEMA_SELECTION = "issue_141_selection_v1"
SCHEMA_SMOKE = "issue_141_smoke_v1"
SCHEMA_PLAN = "issue_141_plan_v1"
SCHEMA_READOUTS = "issue_141_readouts_v1"
SCHEMA_READOUTS_MANIFEST = "issue_141_readouts_manifest_v1"
SCHEMA_RECORD = "issue_141_record_v1"
SCHEMA_REPLICATION = "issue_141_replication_v1"
SCHEMA_CUT_POINTS = "issue_141_cut_points_v1"
SCHEMA_STRATA = "issue_141_strata_v1"
SCHEMA_NOVELTY_RECORD = "issue_141_novelty_record_v1"
SCHEMA_NOVELTY_NN = "issue_141_novelty_nn_reference_v1"
SCHEMA_COMPUTE = "issue_141_compute_v1"
DECISIONS = {
    "pre_registration": "https://github.com/Sino-Huang/NovPhy/issues/141 (ticket body)",
    "program": "https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6479872095",
    "implementation": "https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480032792",
    "strata_cut_points": "https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480033291",
    "compute_and_queues": "https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6480065190",
}

ARMS = {
    "F1": {"label": "U+FiLM", "role": "follow-up", "program_set": False, "second": "U seed-s predicted frame-225 carrier"},
    "F1-D": {"label": "no-rollout control", "role": "control", "program_set": False, "second": "anchor carrier (z0)"},
}
CONFIGS = s128.lr_axis(nv.MATCHED_SCHEDULE["updates"])  # lr3e-4, lr1e-4, lr1e-3 (declaration order), 9 000 updates
BATCH = nv.MATCHED_SCHEDULE["batch_size"]  # 64 = 32 positives + 32 negatives
BASE_WIDTH = nv.A2_MATCHED_BASE_WIDTH  # 70, as A2
CARRIER_DIM = 2 * nv.CARRIER_DIM  # 472 = [z0, second]
SCORE_KEY = "neg_score"
HEAD_KEY = s129.HEAD_KEY  # #129 U+head row key
GRID = "grid"
FIT_EXAMPLES, FIT_POSITIVES, FIT_LEVELS, POLICY_GRID_CELLS = 13_919, 838, 480, 94
SELECTION_QUEUES = {2: "F1", 3: "F1-D"}  # scheduler GPU -> arm (Decision compute_and_queues)
TRAIN_GPU = {"F1": 2, "F1-D": 3}
CHECKPOINT_EVERY = 1000
LOG_EVERY = 500
RESERVE_GPU_HOURS = 60.0  # #141 budget: reserve <= 60 GPU-h, decided by the measured smoke projection
WALL_CAP_HOURS = 48.0  # on 4 GPUs
G_ACTION_TOLERANCE = 1e-6  # replication (a): |G-action(cached U endpoints) - #129 calibration| per seed
CUT_POINTS = (0.3246440972222222, 0.4763107638888889)  # posted (strata_cut_points), 94 policy grid mixed cells
STRATA = ("near", "mid", "far")
STRATA_RANKERS = ("F1", "F1-D", "U+head", "A1", "NN-px", "NN-car", "U")
DISTANCE_METRIC = ("#128 NN-px metric: mean absolute difference in grey levels (uint8, 3 x 240 x 320) between the "
                   "level's anchor frame 0 and the nearest of the 480 fit anchor frames (s128.pixel_orders)")
S128 = ROOT / ".local-artifacts" / s128.IDENTITY
S129 = ROOT / ".local-artifacts" / s129.IDENTITY
IMPORTED_ARMS = {"A1": (S128, "neg_score"), "A2": (S128, "neg_score"), "U+head": (S129, HEAD_KEY)}
IMPORTED_NOVELTY = ("A1", "A2", "A2-native", "A4")  # #128's guarded novelty records (descriptive context)
NN_NAMES = tuple(s128.NN_REFERENCES)  # NN-px, NN-car
CHILD_IDENTITY = {S128: s128.IDENTITY, S129: s129.IDENTITY}
SCOPE = {
    "declared": "F1 is a native-success-readout follow-up on the #129 U+head route, not a member of #127's "
                "world-model disposition set; no new program-level claim is made (#141 ticket body).",
    "g_action": "inherited from U and reported (U's per-seed carrier-space sensitivity on the cached 480 x 16 fit "
                "endpoints), not relaxed and not a G_* entry of F1 or F1-D (#127 issuecomment-6396584235).",
    "arm_guards": ["G_rank", "G_nonfinite", "G_trained"],
    "extra_guards": {"F1": ["G_u_identity", "G_rollout_replication", "G_alignment", "G_classifier_only"],
                     "F1-D": ["G_alignment", "G_classifier_only"]},
    "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
    "port": "no new external code: A2's G-port / G-fid (#128) are inherited and reported, not re-run",
    "tokens": "delta rule (#127) under this declared guard scope",
}


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return s129.device_of(requested)


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------------------------

def cut_points_path(output):
    return Path(output) / "strata" / "cut-points.json"


def strata_path(output):
    return Path(output) / "strata" / "evaluation-distances.json"


def features_path(output, seed):
    return Path(output) / "features" / f"U-seed{seed}.pt"


def selection_dir(output, arm, config):
    return Path(output) / "selection" / arm / config / f"seed-{eb.SELECTION_SEED}"


def main_dir(output, arm, seed):
    return Path(output) / "models" / arm / f"seed-{seed}"


def model_dir(output, arm, seed, selected):
    """Seed 20260908 is the selected configuration's selection run (reused, #127 selection rule step 4)."""
    if seed == eb.SELECTION_SEED:
        return selection_dir(output, arm, selected[arm])
    return main_dir(output, arm, seed)


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / arm / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def replication_path(output, seed):
    return Path(output) / "replication" / f"U--seed{seed}.json"


def novelty_record_path(output, arm, seed):
    return Path(output) / "novelty" / "records" / f"{arm}--seed{seed}.json"


def novelty_nn_path(output, name):
    return Path(output) / "novelty" / "nn" / f"{name}.json"


def imported_record_path(arm, seed):
    return IMPORTED_ARMS[arm][0] / "records" / f"{arm}--seed{seed}.json"


def imported_nn_path(name):
    return S128 / "references" / f"{name}.json"


def imported_novelty_path(arm, seed):
    return S128 / "novelty" / "records" / f"{arm}--seed{seed}.json"


def uhead_path(seed):
    return S129 / "readouts" / "U+head" / f"seed-{seed}" / "head.pt"


def gaction_calibration_path():
    return S129 / "analysis" / "g-action-calibration.json"


# ---------------------------------------------------------------------------------------------------------------
# memorisation strata (cut points frozen before training; policy / fit data only)
# ---------------------------------------------------------------------------------------------------------------

def stratum_of(distance, cuts=CUT_POINTS):
    """near if d <= c1, mid if c1 < d <= c2, far if d > c2."""
    return STRATA[0] if distance <= cuts[0] else STRATA[1] if distance <= cuts[1] else STRATA[2]


def tertile_cuts(distances):
    q = np.quantile(np.asarray(distances, dtype=np.float64), [1 / 3, 2 / 3])
    return float(q[0]), float(q[1])


def nearest_fit_distances(frames):
    """{member: (distance, nearest fit member)} of query anchor frames against the 480 fit anchor frames."""
    orders = s128.pixel_orders(frames, eb.fit_anchor_frames())
    return {m: (order[0][0], order[0][1]) for m, order in orders.items()}


def cut_points(output):
    """--cut-points (CPU; policy / fit data only): tertile cut points of the 94 policy grid mixed cells' nearest-fit
    pixel distance; must equal the posted CUT_POINTS; strata/cut-points.json (written once, re-verified after)."""
    path = cut_points_path(output)
    frames = eb.policy_anchor_frames()
    cells = eb.policy_cells(GRID)
    nearest = nearest_fit_distances({m: frames[m] for m in sorted(cells)})
    cuts = tertile_cuts([d for d, _ in nearest.values()])
    if cuts != CUT_POINTS:
        raise SystemExit(f"cut points {cuts} differ from the posted {CUT_POINTS}")
    payload = {"schema": SCHEMA_CUT_POINTS, "plan_identity": IDENTITY, "cut_points": list(cuts),
               "population": "policy grid mixed cells (eb.policy_cells('grid'))", "levels": len(nearest),
               "metric": DISTANCE_METRIC, "quantiles": [1 / 3, 2 / 3], "rule": "numpy.quantile (linear)",
               "strata": {"near": "d <= c1", "mid": "c1 < d <= c2", "far": "d > c2"},
               "distances": {m: {"distance": d, "nearest_fit": f} for m, (d, f) in sorted(nearest.items())},
               "decision": DECISIONS["strata_cut_points"]}
    if path.is_file():
        stored = eb.read_json(path)
        if {k: v for k, v in stored.items() if k != "written_at"} != json.loads(eb.json_text(payload)):
            raise SystemExit("strata/cut-points.json differs from its re-derivation")
        log(f"cut points re-verified: {cuts}")
        return 0
    payload["written_at"] = eb.utc_now()
    eb.write_json(path, payload)
    log(f"cut points frozen: {cuts}")
    return 0


def write_strata(output, frames, cells):
    """--score only (after the readout freeze and eb.read_evaluation_anchors): each mixed evaluation level's distance
    and stratum under the frozen cut points; written once."""
    path = strata_path(output)
    if path.is_file():
        return eb.read_json(path)
    cuts = tuple(eb.read_json(cut_points_path(output))["cut_points"])
    members = sorted(set().union(*(set(cells[inv]["mixed"]) for inv in eb.INVENTORIES)))
    nearest = nearest_fit_distances({m: frames[m] for m in members})
    levels = {m: {"distance": d, "nearest_fit": f, "stratum": stratum_of(d, cuts)} for m, (d, f) in sorted(nearest.items())}
    grid = sorted(cells[GRID]["mixed"])
    payload = {"schema": SCHEMA_STRATA, "plan_identity": IDENTITY, "cut_points": list(cuts), "metric": DISTANCE_METRIC,
               "levels": levels, "grid_counts": {s: sum(levels[m]["stratum"] == s for m in grid) for s in STRATA},
               "written_at": eb.utc_now()}
    eb.write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------------------------------------------
# U identity and the cached U endpoints (fit / policy only)
# ---------------------------------------------------------------------------------------------------------------

def u_identity(seed):
    """The Gate-B handoff checkpoint of U seed s: file sha256 against the handoff hash (gb.load_model's check)."""
    entry = next((c for c in eb.gb.checkpoints("unit") if c["seed"] == seed), None)
    path = eb.gb.member_path("unit", seed)
    actual = eb.sha256_of(path)
    return {"seed": seed, "checkpoint": rel(path), "sha256": actual,
            "handoff_sha256": None if entry is None else entry["sha256"],
            "usable": bool(entry is not None and entry["usable"]), "request": eb.gb.UNITS["unit"]["request"],
            "pass": bool(entry is not None and entry["usable"] and actual == entry["sha256"])}


def load_features(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA_FEATURES or payload.get("plan_identity") != IDENTITY:
        raise ValueError(f"{path}: feature binding differs")
    return payload


def launch_stack(items, encode):
    return torch.stack([encode(item["action"]) for item in items])


@torch.no_grad()
def build_features(output, seed, device=None, unit=None):
    """features/U-seed<s>.pt (once): U seed s's predicted frame-225 carriers of the fit shots, of the 480 fit anchors
    x 16 grid launches and of every policy grid mixed cell candidate, under deterministic scoring."""
    path = features_path(output, seed)
    if path.is_file():
        return load_features(path)
    device = device_of(device)
    began = time.monotonic()
    shots = eb.FitShots()
    anchors = eb.fit_anchors()["carriers"]
    unit = unit or s129.UnitRollout(seed, device)
    if unit.model is None:
        raise ValueError(f"U seed {seed}: {unit.failure}")
    with harness.deterministic_scoring() as policy:
        fit_end = unit.endpoints(torch.stack([anchors[m] for m in shots.members]), shots.launch5)
        members = sorted(anchors)
        items, actions = s129.grid_actions()
        z0 = torch.stack([anchors[m] for m in members])
        grid_end = unit.endpoints(z0.repeat_interleave(len(items), 0), actions.repeat(len(members), 1))
        pan = eb.policy_anchors()["carriers"]
        pol = {}
        for member, entry in sorted(eb.policy_cells(GRID).items()):
            cand = entry["inventory"]
            launch5 = launch_stack(cand, eb.launch5)
            pol[member] = {"ordinals": [it["ordinal"] for it in cand], "launch5": launch5,
                           "endpoints": unit.endpoints(pan[member][None].expand(len(cand), -1), launch5)}
    seconds = time.monotonic() - began
    payload = {"schema": SCHEMA_FEATURES, "plan_identity": IDENTITY, "seed": seed, "u_identity": u_identity(seed),
               "rollout": {"request": eb.gb.UNITS["unit"]["request"], "delta": int(unit.pair.delta),
                           "transitions": int(math.ceil(s129.sf.ENDPOINT / unit.pair.delta)),
                           "endpoint": int(s129.sf.ENDPOINT)},
               "device": device, "scoring_policy": policy,
               "fit": {"keys": list(shots.keys), "launch5": shots.launch5.clone(), "endpoints": fit_end},
               "fit_grid": {"members": members, "ordinals": [it["ordinal"] for it in items],
                            "endpoints": grid_end.reshape(len(members), len(items), -1)},
               "policy": pol, "seconds": seconds, "written_at": eb.utc_now()}
    eb.atomic_torch(path, payload)
    eb.mark(output, f"features-U-seed{seed}", seconds, seconds if device != "cpu" else 0)
    log(f"U seed {seed} features cached ({seconds:.1f}s on {device})")
    return payload


# ---------------------------------------------------------------------------------------------------------------
# the classifier, its examples and training (A2's recipe, resumable; nonfinite loss = typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def build_classifier(seed):
    """A2Net (base width 70) on the 472-d [z0, second] input; the same seed gives bit-identical weights in both arms."""
    torch.manual_seed(seed)
    return nv.A2Net(BASE_WIDTH, carrier_dim=CARRIER_DIM)


def state_sha(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return f"sha256:{h.hexdigest()}"


def batch_indices(is_solved, seed, step):
    """Batch `step` of a run: A2's sampler (32 + 32 with replacement) on eb.batch_rng(seed, step)."""
    return next(nv.train_indices_sampler(is_solved, BATCH, eb.batch_rng(seed, step)))


def batch_stream_sha(is_solved, seed, updates):
    h = hashlib.sha256()
    for step in range(updates):
        h.update(np.asarray(batch_indices(is_solved, seed, step), dtype=np.int64).tobytes())
    return f"sha256:{h.hexdigest()}"


def fit_observations(arm, members, keys, features=None):
    """One row per fit shot: [z0(level of the shot), U's predicted endpoint of the shot] (F1) or [z0, z0] (F1-D)."""
    anchors = eb.fit_anchors()["carriers"]
    z0 = torch.stack([torch.as_tensor(anchors[m], dtype=torch.float32) for m in members])
    if arm == "F1":
        if features is None or list(features["fit"]["keys"]) != list(keys):
            raise ValueError("F1 features do not follow the fit shot order")
        second = features["fit"]["endpoints"].float()
    elif arm == "F1-D":
        second = z0
    else:
        raise ValueError(f"unknown arm {arm}")
    return torch.cat([z0, second], 1)


def trainset(arm, shots, features, device):
    obs = fit_observations(arm, shots.members, shots.keys, features)
    return nv.TrainSet(task_indices=torch.arange(len(shots), dtype=torch.long), is_solved=shots.verdicts.clone(),
                       actions=shots.launch4.clone(), observations=obs.to(device), levels=tuple(shots.keys))


def classifier_only(model, optimizer):
    """The optimizer holds exactly the classifier's parameters (U is never in the optimizer or the graph)."""
    opt = {id(p) for group in optimizer.param_groups for p in group["params"]}
    own = {id(p) for p in model.parameters()}
    return {"optimizer_parameter_tensors": len(opt), "classifier_parameter_tensors": len(own), "pass": opt == own}


def train_run(arm, config_name, seed, out_dir, data, device="cpu", max_steps=None, features_file=None):
    """Train one arm-seed-configuration (resumable from resume.pt). Returns report.json's content, or
    {"partial": True, ...} when max_steps ends the call early."""
    config = CONFIGS[config_name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final, resume = out_dir / "model.pt", out_dir / "resume.pt"
    if final.is_file():
        return eb.read_json(out_dir / "report.json")
    updates = int(config["updates"])
    features_before = eb.sha256_of(features_file) if features_file else None
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    model = build_classifier(seed).to(device)
    init_sha = state_sha(model.state_dict())
    optimizer, scheduler = nv.make_optimizer(model, learning_rate=config["learning_rate"], updates=updates)
    start, elapsed, started_at = 0, 0.0, eb.utc_now()
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.is_file() else None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
        start, elapsed, started_at, init_sha = state["step"], state["elapsed"], state["started_at"], state["init_sha256"]
        log(f"{arm} {config_name} seed {seed}: resumed at update {start}")
    stop_at = updates if max_steps is None else min(updates, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            loss = float(nv.train_step(model, optimizer, scheduler, data, batch_indices(data.is_solved, seed, step),
                                       device))
            if not math.isfinite(loss):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "loss": repr(loss)}
                break
            if step % LOG_EVERY == 0 or step == updates - 1:
                entry = {"step": step, "lr": optimizer.param_groups[0]["lr"], "loss": loss,
                         "seconds": elapsed + time.monotonic() - began}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(eb.STOP_REQUESTED) and step + 1 < updates
            if stop or ((step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < updates) or (
                    max_steps is not None and step + 1 == stop_at and step + 1 < updates):
                eb.atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                         "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                                         "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                                         "step": step + 1, "elapsed": elapsed + time.monotonic() - began,
                                         "started_at": started_at, "init_sha256": init_sha})
            if stop:
                raise eb.TrainingPaused(f"{arm} {config_name} seed {seed} paused at update {step + 1} of {updates}")
    seconds = elapsed + time.monotonic() - began
    if failure is None and step + 1 < updates:
        return {"partial": True, "steps_done": step + 1, "seconds": seconds}
    complete = failure is None
    parameters = nv.trainable_parameters(model)
    payload = {"schema": SCHEMA_MODEL, "plan_identity": IDENTITY, "arm": arm, "config_name": config_name,
               "config": dict(config), "seed": seed, "updates": updates, "batch": BATCH, "base_width": BASE_WIDTH,
               "carrier_dim": CARRIER_DIM, "complete": complete, "typed_failure": failure,
               "state_dict": model.state_dict() if complete else None, "parameters": parameters}
    eb.atomic_torch(final, payload)
    features_after = eb.sha256_of(features_file) if features_file else None
    gpu = str(device).startswith("cuda")
    report = {"arm": arm, "config_name": config_name, "config": dict(config), "seed": seed, "updates": updates,
              "batch": BATCH, "complete": complete, "typed_failure": failure, "wall_seconds": seconds,
              "gpu_seconds": seconds if gpu else 0.0, "seconds_per_update": seconds / max(1, step + 1),
              "device": torch.cuda.get_device_name(0) if gpu else "cpu",
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if gpu else None,
              "model_sha256": eb.sha256_of(final), "parameters": parameters, "init_sha256": init_sha,
              "batch_stream_sha256": batch_stream_sha(data.is_solved, seed, updates),
              "classifier_only": classifier_only(model, optimizer),
              "features": {"file": rel(features_file) if features_file else None, "sha256_before": features_before,
                           "sha256_after": features_after, "unchanged": features_before == features_after},
              "examples": int(len(data.is_solved)), "positives": int(data.is_solved.sum()),
              "final": trace[-1] if trace else None, "started_at": started_at, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if resume.is_file():
        resume.unlink()
    log(f"{arm} {config_name} seed {seed}: {'complete' if complete else failure['typed_failure']} after {seconds:.0f}s")
    return report


def load_model(path, device="cpu"):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL or payload.get("plan_identity") != IDENTITY:
        raise ValueError(f"{path}: model binding differs")
    if not payload["complete"]:
        return None, payload
    model = nv.A2Net(payload["base_width"], carrier_dim=payload["carrier_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def run_items(output, items, device):
    """Train (arm, config, seed, out_dir) items in order; a stop signal pauses the queue (exit 75)."""
    eb.install_stop_handlers()
    shots = eb.FitShots()
    cache = {}
    for arm, config_name, seed, out_dir in items:
        if (Path(out_dir) / "model.pt").is_file():
            continue
        ffile = features_path(output, seed) if arm == "F1" else None
        if (arm, seed) not in cache:
            features = build_features(output, seed, device) if arm == "F1" else None
            cache[(arm, seed)] = trainset(arm, shots, features, device)
        phase = f"train-{arm}-{config_name}-seed{seed}"
        began = time.monotonic()
        try:
            report = train_run(arm, config_name, seed, out_dir, cache[(arm, seed)], device, features_file=ffile)
        except eb.TrainingPaused as paused:
            seconds = time.monotonic() - began
            eb.mark(output, phase, seconds, seconds if device != "cpu" else 0)
            log(str(paused))
            return eb.EXIT_PAUSED
        seconds = time.monotonic() - began
        eb.mark(output, phase, seconds, seconds if device != "cpu" else 0, report=rel(Path(out_dir) / "report.json"),
                complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# scoring paths (one for every split; deterministic scoring around every call)
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def classifier_logits(model, z0, second, launch4):
    """Success logits of K candidates: rows [z0, second] [K, 472] with launch4 [K, 4] (eval mode, one batch)."""
    model.eval()
    device = next(model.parameters()).device
    obs = torch.cat([z0.float(), second.float()], 1).to(device)
    return model(obs, launch4.to(device=device, dtype=torch.float32)).float().cpu()


@torch.no_grad()
def cell_rows(arm, model, items, z0_carrier, ends=None):
    """Rows of one cell from its anchor carrier [236]: -logit per candidate (eb.score_rows, key neg_score); F1 reads
    U's predicted endpoints `ends` [K, 236] of the same candidates in the same order."""
    z0 = torch.as_tensor(z0_carrier, dtype=torch.float32)[None].expand(len(items), -1)
    if arm == "F1":
        if ends is None or len(ends) != len(items):
            raise ValueError("F1 needs U's endpoints of every candidate")
        second = ends
    else:
        second = z0
    return eb.score_rows(items, classifier_logits(model, z0, second, launch_stack(items, eb.launch4)), key=SCORE_KEY)


def policy_rows(arm, model, features=None):
    """{member: rows} of the 94 policy grid mixed cells (F1: the cached policy endpoints, ordinals checked)."""
    anchors = eb.policy_anchors()["carriers"]
    out = {}
    for member, entry in sorted(eb.policy_cells(GRID).items()):
        items = entry["inventory"]
        ends = None
        if arm == "F1":
            cached = features["policy"][member]
            if cached["ordinals"] != [it["ordinal"] for it in items]:
                raise ValueError(f"policy cell {member}: cached ordinals differ")
            ends = cached["endpoints"]
        out[member] = cell_rows(arm, model, items, anchors[member], ends)
    return out


def fit_costs(arm, model, features=None):
    """Ranking costs [480, 16] = -logit of the 16 grid launches from every fit anchor (G-rank)."""
    anchors = eb.fit_anchors()["carriers"]
    items = eb.grid_items()
    members = sorted(anchors)
    if arm == "F1" and (features["fit_grid"]["members"] != members
                        or features["fit_grid"]["ordinals"] != [it["ordinal"] for it in items]):
        raise ValueError("cached fit-grid endpoints do not follow the fit anchors x grid order")
    out = []
    for i, member in enumerate(members):
        ends = features["fit_grid"]["endpoints"][i] if arm == "F1" else None
        rows = cell_rows(arm, model, items, anchors[member], ends)
        out.append([np.nan if r[SCORE_KEY] is None else r[SCORE_KEY] for r in rows])
    return np.asarray(out, dtype=np.float64)


def macs_block(arm, model, unit=None):
    """MACs per candidate on one fit anchor's 16 grid launches: the classifier, plus (F1) U's 15 transitions."""
    anchors = eb.fit_anchors()["carriers"]
    items = eb.grid_items()
    z0 = anchors[sorted(anchors)[0]][None].expand(len(items), -1).float()
    device = next(model.parameters()).device
    obs = torch.cat([z0, z0], 1).to(device)
    launch4 = launch_stack(items, eb.launch4).to(device)
    classifier = eb.macs_per_candidate(lambda o, a: model(o, a), obs, launch4, candidates=len(items))
    rollout = (eb.macs_per_candidate(unit.endpoints, z0, launch_stack(items, eb.launch5), candidates=len(items))
               if arm == "F1" else 0.0)
    return {"classifier": classifier, "u_rollout": rollout, "total": classifier + rollout}


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, gpu, device=None):
    if not cut_points_path(output).is_file():
        raise SystemExit("strata/cut-points.json missing: the cut points are frozen before any training (--cut-points)")
    if gpu not in SELECTION_QUEUES:
        raise SystemExit(f"--selection queues are GPUs {sorted(SELECTION_QUEUES)}")
    arm = SELECTION_QUEUES[gpu]
    device = device_of(device)
    items = [(arm, name, eb.SELECTION_SEED, selection_dir(output, arm, name)) for name in CONFIGS]
    log(f"selection GPU {gpu}: {arm} x {list(CONFIGS)}")
    return run_items(output, items, device)


def selection_readouts(output, device=None):
    """Policy-split pooled grid AUC of every selection run; eb.select_configuration per arm independently."""
    device = device_of(device)
    began = time.monotonic()
    arms = {}
    with harness.deterministic_scoring() as policy:
        for arm in ARMS:
            features = load_features(features_path(output, eb.SELECTION_SEED)) if arm == "F1" else None
            results, details = {}, {}
            for name in CONFIGS:
                path = selection_dir(output, arm, name) / "model.pt"
                if not path.is_file():
                    raise SystemExit(f"selection run {arm}/{name} is not finished")
                model, payload = load_model(path, device)
                results[name] = {"auc": None, "parameters": payload["parameters"]}
                details[name] = {"model_sha256": eb.sha256_of(path), "typed_failure": payload["typed_failure"]}
                if model is not None:
                    results[name]["auc"] = eb.policy_grid_auc(policy_rows(arm, model, features), SCORE_KEY)
            arms[arm] = {**eb.select_configuration(results), "details": details}
    seconds = time.monotonic() - began
    eb.mark(output, "selection-readouts", seconds, seconds if device != "cpu" else 0)
    selected = {arm: block["selected"] for arm, block in arms.items()}
    eb.write_json(Path(output) / "selection.json", {
        "schema": SCHEMA_SELECTION, "plan_identity": IDENTITY, "configs": CONFIGS, "seed": eb.SELECTION_SEED,
        "selected": selected, "arms": arms, "scoring_policy": policy, "policy_cells": len(eb.policy_cells(GRID)),
        "seconds": seconds, "written_at": eb.utc_now()})
    log(f"selection: {selected}")
    return 0 if all(v is not None for v in selected.values()) else 1


def selected_configs(output):
    path = Path(output) / "selection.json"
    if not path.is_file():
        raise ValueError("selection.json missing; run --selection-readouts")
    selected = eb.read_json(path)["selected"]
    if any(v is None for v in selected.values()):
        raise ValueError(f"an arm has no selected configuration: {selected}")
    return selected


# ---------------------------------------------------------------------------------------------------------------
# --smoke (GPU 2 via the scheduler): accounting, U identity / replication, alignment, classifier-only, pairing,
# CPU tests, measured throughput, projection against the 60 GPU-h reserve
# ---------------------------------------------------------------------------------------------------------------

def uhead_manifest_entry(seed):
    return eb.read_json(S129 / "readouts-manifest.json")["entries"][f"U+head--seed{seed}"]


def replication_pre_freeze(features, seed, device):
    """(a) G-action of the cached fit endpoints vs #129's calibration; (b) #129's frozen U+head head on the cached
    policy endpoints vs #129's policy_head_auc (4 d.p.)."""
    calibration = eb.read_json(gaction_calibration_path())["U"]["per_seed"][str(seed)]["G_action_carrier_236"]
    value = float(eb.action_sensitivity(features["fit_grid"]["endpoints"]))
    entry = uhead_manifest_entry(seed)
    head_ok = eb.sha256_of(uhead_path(seed)) == entry["head_sha256"]
    head, _ = s129.load_head(uhead_path(seed), device)
    anchors = eb.policy_anchors()["carriers"]
    rows = {}
    with harness.deterministic_scoring():
        for member, cell in sorted(eb.policy_cells(GRID).items()):
            cached = features["policy"][member]
            z0 = anchors[member][None].expand(len(cell["inventory"]), -1)
            rows[member] = eb.score_rows(cell["inventory"], s129.head_logits(head, z0, cached["endpoints"],
                                                                             cached["launch5"]), key=HEAD_KEY)
    auc = eb.policy_grid_auc(rows, HEAD_KEY)
    reference = eb.read_json(S129 / "readouts" / "U+head" / f"seed-{seed}" / "readouts.json")["policy_head_auc"]
    g_action = {"value": value, "calibration": calibration, "abs_delta": abs(value - calibration),
                "tolerance": G_ACTION_TOLERANCE, "pass": abs(value - calibration) <= G_ACTION_TOLERANCE}
    uhead = {"policy_head_auc": auc, "issue_129": reference, "head_sha256_matches_129_manifest": head_ok,
             "pass": bool(head_ok and auc is not None and round(auc, 4) == round(reference, 4))}
    return {"g_action": g_action, "u_head_policy_auc": uhead, "pass": bool(g_action["pass"] and uhead["pass"])}


def alignment_check(shots, features_by_seed):
    """Every cached row follows the fit shot order / grid order / policy cell order, and launch4 / launch5 of a row are
    built from the same action dict."""
    launch5 = torch.stack([eb.launch5(a) for a in shots.actions])
    launch4 = torch.stack([eb.launch4(a) for a in shots.actions])
    out = {"fit_launch5_from_actions": bool(torch.equal(launch5, shots.launch5)),
           "fit_launch4_from_actions": bool(torch.equal(launch4, shots.launch4)), "seeds": {}}
    cells = eb.policy_cells(GRID)
    for seed, f in features_by_seed.items():
        out["seeds"][str(seed)] = {
            "fit_keys": list(f["fit"]["keys"]) == list(shots.keys),
            "fit_launch5": bool(torch.equal(f["fit"]["launch5"], shots.launch5)),
            "fit_rows": int(f["fit"]["endpoints"].shape[0]) == len(shots),
            "fit_grid_members": f["fit_grid"]["members"] == sorted(eb.fit_anchors()["carriers"]),
            "fit_grid_ordinals": f["fit_grid"]["ordinals"] == [it["ordinal"] for it in eb.grid_items()],
            "policy_cells": sorted(f["policy"]) == sorted(cells),
            "policy_ordinals": all(f["policy"][m]["ordinals"] == [it["ordinal"] for it in cells[m]["inventory"]]
                                   and torch.equal(f["policy"][m]["launch5"], launch_stack(cells[m]["inventory"],
                                                                                           eb.launch5))
                                   for m in cells if m in f["policy"]),
            "nonfinite_rows": int((~torch.isfinite(f["fit"]["endpoints"])).any(1).sum())}
    out["pass"] = bool(out["fit_launch5_from_actions"] and out["fit_launch4_from_actions"] and all(
        all(v for k, v in s.items() if k != "nonfinite_rows") for s in out["seeds"].values()))
    return out


def classifier_only_check(output, shots, device, steps=20):
    """A short throwaway F1 run with U loaded: the optimizer holds exactly the classifier's parameters, U's state is
    unchanged, no U parameter receives a gradient, and the feature file is unchanged."""
    unit = s129.UnitRollout(eb.SELECTION_SEED, device)
    before = state_sha(unit.model.state_dict())
    ffile = features_path(output, eb.SELECTION_SEED)
    fsha = eb.sha256_of(ffile)
    data = trainset("F1", shots, load_features(ffile), device)
    model = build_classifier(eb.SELECTION_SEED).to(device)
    optimizer, scheduler = nv.make_optimizer(model, learning_rate=CONFIGS["lr3e-4"]["learning_rate"],
                                             updates=CONFIGS["lr3e-4"]["updates"])
    for step in range(steps):
        nv.train_step(model, optimizer, scheduler, data, batch_indices(data.is_solved, eb.SELECTION_SEED, step), device)
    check = classifier_only(model, optimizer)
    u_grads = sum(p.grad is not None for p in unit.model.parameters())
    out = {**check, "updates": steps, "u_state_sha256_before": before,
           "u_state_sha256_after": state_sha(unit.model.state_dict()), "u_parameters_with_grad": int(u_grads),
           "features_unchanged": eb.sha256_of(ffile) == fsha}
    out["pass"] = bool(check["pass"] and out["u_state_sha256_before"] == out["u_state_sha256_after"]
                       and u_grads == 0 and out["features_unchanged"])
    return out


def pairing_check(output):
    """F1 and F1-D selection runs of each configuration: identical initial weights and batch-index streams."""
    out = {}
    for name in CONFIGS:
        a = eb.read_json(selection_dir(output, "F1", name) / "report.json")
        b = eb.read_json(selection_dir(output, "F1-D", name) / "report.json")
        out[name] = {"init": a["init_sha256"] == b["init_sha256"],
                     "batch_stream": a["batch_stream_sha256"] == b["batch_stream_sha256"]}
    return {"configs": out, "pass": all(v["init"] and v["batch_stream"] for v in out.values())}


def run_tests():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    return {"command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {TESTS}", "exit_code": proc.returncode,
            "tail": proc.stdout.strip().splitlines()[-1:] if proc.stdout else []}


def readout_measure(output, arm, seed, run, device):
    """Seconds of one full readout computation (policy rows, G-rank, MACs) of a trained run; nothing is written."""
    began = time.monotonic()
    model, _ = load_model(Path(run) / "model.pt", device)
    if model is None:
        return None
    features = load_features(features_path(output, seed)) if arm == "F1" else None
    unit = s129.UnitRollout(seed, device) if arm == "F1" else None
    with harness.deterministic_scoring():
        eb.policy_grid_auc(policy_rows(arm, model, features), SCORE_KEY)
        eb.g_rank(fit_costs(arm, model, features))
        macs_block(arm, model, unit)
    return time.monotonic() - began


# Projection terms that cannot be measured before the freeze (the evaluation / novelty populations are sealed until
# --score; Decision compute_and_queues): one process each, measured per-cell rates scaled by an upper bound.
SCORING_CELLS_UPPER_BOUND = 2000  # mixed cells over the four inventories (the grid alone has 311)
NOVELTY_LEVELS_UPPER_BOUND = 600
PROCESS_OVERHEAD_SECONDS = 600.0  # start-up and data loading per GPU job


def projection(output, selected, throughput, readout_seconds, cell_seconds):
    """GPU-h of everything left after the smoke (main training, readouts, scoring, novelty, validation)."""
    jobs = []
    for arm in ARMS:
        rate = throughput[arm][selected[arm]]["seconds_per_update"]
        for seed in eb.SEEDS[1:]:
            jobs.append((f"train-{arm}-seed{seed}", rate * CONFIGS[selected[arm]]["updates"]))
        jobs += [(f"readouts-{arm}-seed{seed}", readout_seconds[arm]) for seed in eb.SEEDS]
    score = SCORING_CELLS_UPPER_BOUND * len(eb.SEEDS) * sum(cell_seconds.values())
    novelty = NOVELTY_LEVELS_UPPER_BOUND * len(eb.SEEDS) * (sum(cell_seconds.values()) + cell_seconds["F1"])
    validate = SCORING_CELLS_UPPER_BOUND * cell_seconds["F1"]
    overhead = PROCESS_OVERHEAD_SECONDS * 8  # train x2, readouts x2, score, novelty, validate, spare
    total = sum(s for _, s in jobs) + score + novelty + validate + overhead
    train_wall = max(sum(s for n, s in jobs if n.startswith(f"train-{a}") or n.startswith(f"readouts-{a}"))
                     for a in ARMS)
    wall = train_wall + score + novelty + validate + overhead
    return {"remaining_gpu_hours": total / 3600, "terms_seconds": {
                "training_and_readouts": dict(jobs), "score_estimate": score, "novelty_estimate": novelty,
                "validate_estimate": validate, "process_overhead_estimate": overhead},
            "reserve_gpu_hours": RESERVE_GPU_HOURS, "wall_hours_on_4_gpus": wall / 3600, "wall_cap_hours": WALL_CAP_HOURS,
            "within_reserve": bool(total / 3600 <= RESERVE_GPU_HOURS and wall / 3600 <= WALL_CAP_HOURS),
            "rule": "measured selection seconds/update and readout seconds; scoring from the measured per-cell rate "
                    f"x upper bounds ({SCORING_CELLS_UPPER_BOUND} cells, {NOVELTY_LEVELS_UPPER_BOUND} novelty levels)"}


@torch.no_grad()
def cell_seconds(output, selected, device):
    """Median seconds to score one 16-candidate grid cell from its anchor: F1 incl. U's rollout, F1-D."""
    anchors = eb.policy_anchors()["carriers"]
    cells = sorted(eb.policy_cells(GRID).items())[:20]
    unit = s129.UnitRollout(eb.SELECTION_SEED, device)
    out = {}
    with harness.deterministic_scoring():
        for arm in ARMS:
            model, _ = load_model(selection_dir(output, arm, selected[arm]) / "model.pt", device)
            if model is None:
                out[arm] = 0.0
                continue

            def decide(mc, arm=arm, model=model):
                member, entry = mc
                items = entry["inventory"]
                ends = (unit.endpoints(anchors[member][None].expand(len(items), -1), launch_stack(items, eb.launch5))
                        if arm == "F1" else None)
                return cell_rows(arm, model, items, anchors[member], ends)
            out[arm] = eb.wall_per_decision(decide, cells)
    return out


def smoke(output, device=None):
    output = Path(output)
    device = device_of(device)
    began = time.monotonic()
    shots = eb.FitShots()
    accounting = {"fit_shots": len(shots), "fit_positive": int(shots.verdicts.sum()), "fit_levels": len(shots.levels),
                  "fit_anchors": len(eb.fit_anchors()["carriers"]), "policy_grid_mixed_cells": len(eb.policy_cells(GRID)),
                  "classifier_parameters": nv.trainable_parameters(build_classifier(eb.SELECTION_SEED))}
    accounting["pass"] = bool((accounting["fit_shots"], accounting["fit_positive"], accounting["fit_levels"],
                               accounting["fit_anchors"], accounting["policy_grid_mixed_cells"])
                              == (FIT_EXAMPLES, FIT_POSITIVES, FIT_LEVELS, FIT_LEVELS, POLICY_GRID_CELLS))
    selection_record = eb.read_json(output / "selection.json")
    selected = selected_configs(output)
    features, u = {}, {}
    for seed in eb.SEEDS:
        features[seed] = build_features(output, seed, device)
        identity = u_identity(seed)
        u[str(seed)] = {"identity": identity, "features_identity_matches": features[seed]["u_identity"] == identity,
                        "features_sha256": eb.sha256_of(features_path(output, seed)),
                        "replication": replication_pre_freeze(features[seed], seed, device),
                        "rollout": features[seed]["rollout"]}
    u_pass = all(v["identity"]["pass"] and v["features_identity_matches"] and v["replication"]["pass"]
                 and v["rollout"]["transitions"] == 15 for v in u.values())
    alignment = alignment_check(shots, features)
    only = classifier_only_check(output, shots, device)
    pairing = pairing_check(output)
    reports = {arm: {name: eb.read_json(selection_dir(output, arm, name) / "report.json") for name in CONFIGS}
               for arm in ARMS}
    reports_ok = all(r["classifier_only"]["pass"] and r["features"]["unchanged"] and r["examples"] == FIT_EXAMPLES
                     and r["positives"] == FIT_POSITIVES for by in reports.values() for r in by.values())
    tests = run_tests()
    throughput = {arm: {name: {k: r[k] for k in ("seconds_per_update", "gpu_seconds", "wall_seconds", "updates",
                                                 "device", "complete")} for name, r in by.items()}
                  for arm, by in reports.items()}
    readout_seconds = {arm: readout_measure(output, arm, eb.SELECTION_SEED,
                                            selection_dir(output, arm, selected[arm]), device) or 0.0 for arm in ARMS}
    per_cell = cell_seconds(output, selected, device)
    proj = projection(output, selected, throughput, readout_seconds, per_cell)
    spent = {"selection_training_gpu_seconds": sum(r["gpu_seconds"] for by in reports.values() for r in by.values()),
             "feature_pass_gpu_seconds": {str(s): f["seconds"] if f["device"] != "cpu" else 0.0
                                          for s, f in features.items()}}
    record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "u": u, "u_pass": u_pass,
              "alignment": alignment, "classifier_only": only, "pairing": pairing,
              "selection_reports_pass": reports_ok, "tests": tests, "throughput": throughput,
              "readout_seconds": readout_seconds, "cell_seconds": per_cell, "selection": selection_record,
              "selected": selected, "spent": spent, "projection": proj, "device": device,
              "pass": bool(accounting["pass"] and u_pass and alignment["pass"] and only["pass"] and pairing["pass"]
                           and reports_ok and tests["exit_code"] == 0 and proj["within_reserve"]),
              "written_at": eb.utc_now()}
    seconds = time.monotonic() - began
    eb.mark(output, "smoke", seconds, seconds if device != "cpu" else 0)
    eb.write_json(output / "smoke.json", record)
    log(f"smoke: pass {record['pass']}; selected {selected}; projection {proj['remaining_gpu_hours']:.2f} GPU-h "
        f"(reserve {RESERVE_GPU_HOURS})")
    return 0 if record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/phyre_dqn/novphy.py", "world_model/baselines/phyre_dqn/__init__.py",
             "scripts/run_issue_128_direct_predictors.py", "scripts/run_issue_129_slotformer.py",
             "world_model/baselines/slotformer/__init__.py", "world_model/baselines/slotformer/novphy.py"]
VENDOR_DIRS = ["world_model/baselines/phyre_dqn/vendor", "world_model/baselines/slotformer/vendor"]


def imported_files():
    """Every imported artefact (hash-pinned in the plan; read for content only after the freeze)."""
    paths = [imported_record_path(arm, seed) for arm in IMPORTED_ARMS for seed in eb.SEEDS]
    paths += [imported_nn_path(name) for name in NN_NAMES]
    paths += [imported_novelty_path(arm, seed) for arm in IMPORTED_NOVELTY for seed in eb.SEEDS]
    return paths


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += [p for d in VENDOR_DIRS for p in sorted((ROOT / d).rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts]
    paths += [cut_points_path(output), output / "selection.json", output / "smoke.json"]
    paths += [features_path(output, seed) for seed in eb.SEEDS]
    paths += [selection_dir(output, arm, name) / f for arm in ARMS for name in CONFIGS for f in ("model.pt", "report.json")]
    paths += [eb.gb.member_path("unit", seed) for seed in eb.SEEDS]
    paths += [uhead_path(seed) for seed in eb.SEEDS]
    paths += [S129 / "readouts" / "U+head" / f"seed-{seed}" / "readouts.json" for seed in eb.SEEDS]
    paths += [S129 / "readouts-manifest.json", gaction_calibration_path()]
    paths += imported_files()
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise ValueError(f"pinned inputs missing: {missing}")
    return [{"name": f"f1_{rel(p)}", "artifact": rel(p), "sha256": eb.sha256_of(p)} for p in paths]


def train_queues():
    return {str(TRAIN_GPU[arm]): [f"{arm}--seed{seed}" for seed in eb.SEEDS[1:]] for arm in ARMS}


def readout_queues():
    return {str(TRAIN_GPU[arm]): [f"{arm}--seed{seed}" for seed in eb.SEEDS] for arm in ARMS}


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        load_plan(output)
        log("plan.json is frozen; pins re-verified")
        return 0
    smoke_record = eb.read_json(output / "smoke.json")
    if not smoke_record["pass"]:
        raise SystemExit("smoke did not pass; no freeze")
    selected = smoke_record["selected"]
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {arm: {**ARMS[arm], "configuration": selected[arm], "config": CONFIGS[selected[arm]],
                       "classifier": {"class": "A2Net", "base_width": BASE_WIDTH, "carrier_dim": CARRIER_DIM},
                       "parameters": smoke_record["accounting"]["classifier_parameters"], "seeds": list(eb.SEEDS),
                       "reused_seed": {"seed": eb.SELECTION_SEED,
                                       "run": rel(selection_dir(output, arm, selected[arm]))}} for arm in ARMS},
        "recipe": {"loss": "class-balanced BCE with logits (A2)", "batch": BATCH, "sampler": "nv.train_indices_sampler "
                   "(32 + 32 with replacement) on eb.batch_rng(seed, b)", "optimizer": "Adam", "schedule":
                   "CosineAnnealingLR(T_max=updates), scheduler.step() before optimizer.step()",
                   "examples": FIT_EXAMPLES, "positives": FIT_POSITIVES, "levels": FIT_LEVELS},
        "u": {"name": "unit", "request": eb.gb.UNITS["unit"]["request"], "transitions": 15,
              "pairing": "F1 seed s reads U seed s", "features": {str(s): rel(features_path(output, s))
                                                                 for s in eb.SEEDS}},
        "queues": {"train": train_queues(), "readouts": readout_queues(), "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "scope": SCOPE,
        "contrasts": {"per_arm": dict(eb.CONTRAST_REFERENCES), "primary": "F1 - U",
                      "within_family": ["F1 - F1-D", "F1 - A1", "F1 - A2", "F1 - U+head"],
                      "descriptive": ["F1 - NN-px", "F1 - NN-car"],
                      "imported": {arm: rel(imported_record_path(arm, eb.SELECTION_SEED).parent) for arm in IMPORTED_ARMS}},
        "strata": {"cut_points": list(CUT_POINTS), "metric": DISTANCE_METRIC, "rankers": list(STRATA_RANKERS),
                   "population": "grid mixed cells", "tokens": None},
        "novelty": {"arms": ["F1", "F1-D", "U+head"], "rule": "P-novelty for F1 / F1-D (guard-passing, every seed); "
                    "#129's frozen U+head every seed; NN exact-launch rule, missing support unavailable",
                    "imported": list(IMPORTED_NOVELTY), "tokens": None},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA,
        "budget": {"reserve_gpu_hours": RESERVE_GPU_HOURS, "projection": smoke_record["projection"]},
        "decisions": DECISIONS,
        "disclosures": [
            "F1 and F1-D were each selected on the policy split among three pre-declared learning rates; U had no "
            "such tuning (#127).",
            "F1's training compute includes U's own training (Decision compute_and_queues).",
            "G-action is inherited from U (below the 0.05 threshold in carrier space) and reported, not an F1 guard.",
        ],
    }
    eb.write_json(path, plan)
    log(f"plan.json frozen ({len(plan['pins'])} pins); selected {selected}")
    return 0


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = eb.read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("plan_identity") != IDENTITY:
        raise ValueError("plan.json binding differs")
    eb.verify_pins(plan["pins"])
    return plan


def parse_name(name):
    arm, seed = name.split("--seed")
    return arm, int(seed)


# ---------------------------------------------------------------------------------------------------------------
# --train / --readouts / --freeze-readouts
# ---------------------------------------------------------------------------------------------------------------

def train(output, gpu, device=None):
    plan = load_plan(output)
    names = plan["queues"]["train"].get(str(gpu), [])
    items = []
    for name in names:
        arm, seed = parse_name(name)
        items.append((arm, plan["selected"][arm], seed, main_dir(output, arm, seed)))
    log(f"train GPU {gpu}: {names}")
    return run_items(output, items, device_of(device))


def report_of_factory(output, plan):
    """ledger phase -> its training report (each arm-seed counted once; #123 A3 rule)."""
    def report_of(phase):
        if not phase.startswith("train-"):
            return None
        head, seed = phase[len("train-"):].rsplit("-seed", 1)
        arm, config = head.rsplit("-lr", 1)
        seed, config = int(seed), f"lr{config}"
        if seed == eb.SELECTION_SEED:
            return selection_dir(output, arm, config) / "report.json"
        return main_dir(output, arm, seed) / "report.json"
    return report_of


def readouts_arm(output, plan, arm, seed, device):
    out_dir = readouts_dir(output, arm, seed)
    if (out_dir / "readouts.json").is_file():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    path = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
    model, payload = load_model(path, device)
    result = {"schema": SCHEMA_READOUTS, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
              "config_name": payload["config_name"], "model": rel(path), "model_sha256": eb.sha256_of(path),
              "parameters": payload["parameters"],
              "typed_failure": payload["typed_failure"]["typed_failure"] if payload["typed_failure"] else None}
    if model is not None:
        features = load_features(features_path(output, seed)) if arm == "F1" else None
        unit = s129.UnitRollout(seed, device) if arm == "F1" else None
        with harness.deterministic_scoring() as policy:
            result["policy_grid_auc"] = eb.policy_grid_auc(policy_rows(arm, model, features), SCORE_KEY)
            costs = fit_costs(arm, model, features)
            result["G_rank"] = eb.g_rank(costs)
            result["fit_anchor_nonfinite_share"] = float(np.isnan(costs).mean())
            result["macs"] = macs_block(arm, model, unit)
            result["macs_per_candidate"] = result["macs"]["total"]
            result["scoring_policy"] = policy
        if arm == "F1":
            value = float(eb.action_sensitivity(features["fit_grid"]["endpoints"]))
            result["G_action_inherited"] = {
                "value": value, "threshold": eb.GUARD["action_sensitivity_min"],
                "meets_threshold": value >= eb.GUARD["action_sensitivity_min"],
                "rule": "U's carrier-space action sensitivity on the cached 480 x 16 fit endpoints (inherited, "
                        "reported; not an F1 guard)"}
            result["features_sha256"] = eb.sha256_of(features_path(output, seed))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{arm}-seed{seed}", seconds, seconds if device != "cpu" else 0)
    eb.write_json(out_dir / "readouts.json", result)  # written last (atomic)
    log(f"readouts {arm} seed {seed}: policy AUC {result.get('policy_grid_auc')}, G-rank "
        f"{(result.get('G_rank') or {}).get('pass')}")
    return 0


def readouts(output, gpu, device=None):
    plan = load_plan(output)
    device = device_of(device)
    for name in plan["queues"]["readouts"].get(str(gpu), []):
        arm, seed = parse_name(name)
        readouts_arm(output, plan, arm, seed, device)
    return 0


def readout_items():
    return [(arm, seed) for arm in ARMS for seed in eb.SEEDS]


def freeze_readouts(output):
    output = Path(output)
    plan = load_plan(output)
    path = output / "readouts-manifest.json"
    if path.is_file():
        load_manifest(output)
        log("readouts are frozen; manifest re-verified")
        return 0
    if eb.evaluation_anchors_path(output).is_file():
        raise SystemExit("evaluation anchors already read: G-chronology would fail")
    entries = {}
    for arm, seed in readout_items():
        rpath = readouts_dir(output, arm, seed) / "readouts.json"
        if not rpath.is_file():
            raise SystemExit(f"readouts of {arm} seed {seed} missing")
        mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
        entry = {"model": rel(mpath), "model_sha256": eb.sha256_of(mpath), "readouts": rel(rpath),
                 "readouts_sha256": eb.sha256_of(rpath)}
        if arm == "F1":
            entry.update(features=rel(features_path(output, seed)), features_sha256=eb.sha256_of(features_path(output, seed)))
        entries[f"{arm}--seed{seed}"] = entry
    for seed in eb.SEEDS:  # #129's frozen U+head heads (re-scored only on the #110 novelty levels)
        entries[f"U+head--seed{seed}"] = {"head": rel(uhead_path(seed)), "head_sha256": eb.sha256_of(uhead_path(seed))}
    eb.write_json(path, {"schema": SCHEMA_READOUTS_MANIFEST, "plan_identity": IDENTITY, "frozen_at": eb.utc_now(),
                         "entries": entries})
    log(f"readouts frozen ({len(entries)} entries)")
    return 0


def load_manifest(output):
    path = Path(output) / "readouts-manifest.json"
    if not path.is_file():
        raise ValueError("readouts are not frozen; run --freeze-readouts first")
    manifest = eb.read_json(path)
    for name, entry in manifest["entries"].items():
        for kind in ("model", "readouts", "features", "head"):
            if kind in entry and eb.sha256_of(ROOT / entry[kind]) != entry[f"{kind}_sha256"]:
                raise ValueError(f"{name}: frozen {kind} changed")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# --score (once; a process that did not train; GPU 0; deterministic)
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def u_endpoints(unit, carriers, cells):
    """{inventory: {member: U's predicted frame-225 carriers [K, 236]}} of every mixed cell from its anchor."""
    out = {}
    for inv in eb.INVENTORIES:
        out[inv] = {}
        for member, entry in sorted(cells[inv]["mixed"].items()):
            items = entry["inventory"]
            z0 = torch.as_tensor(carriers[member], dtype=torch.float32)[None].expand(len(items), -1)
            out[inv][member] = unit.endpoints(z0, launch_stack(items, eb.launch5))
    return out


def replication_block(seed, ends, carriers, cells, costs, reference):
    """EndpointCosts rows of U's endpoints (the ones F1 reads) vs the Gate-B U record of the same seed:
    harness.endpoint_replication per cell; every cell must pass; exact-equality share reported."""
    if reference.get("typed_failure"):
        return {"pass": False, "reason": f"Gate-B U record typed failure {reference['typed_failure']}"}
    inventories, ok = {}, True
    for inv in eb.INVENTORIES:
        worst = dict.fromkeys(harness.PARSED_QUANTITIES, 0.0)
        failed, rows, exact = [], 0, 0
        for member, entry in sorted(cells[inv]["mixed"].items()):
            fresh = eb.endpoint_rows(entry["inventory"], ends[inv][member], carriers[member], costs)
            ref = reference["inventories"][inv][member]["rows"]
            check = harness.endpoint_replication(ref, fresh)
            for key in worst:
                worst[key] = max(worst[key], check["max_abs_delta"][key])
            if not check["pass"]:
                failed.append(member)
            theirs = {r["ordinal"]: r for r in ref}
            for row in fresh:
                rows += 1
                other = theirs.get(row["ordinal"])
                exact += bool(other is not None and all(other.get(k) == row.get(k) for k in
                                                        ("excluded", *harness.PARSED_QUANTITIES)))
        inventories[inv] = {"cells": len(cells[inv]["mixed"]), "failed_cells": failed, "max_abs_delta": worst,
                            "rows": rows, "exact_rows": exact, "pass": not failed}
        ok = ok and not failed
    return {"inventories": inventories, "tolerance": harness.PARSE_TOLERANCE,
            "rule": "harness.endpoint_replication (parsed quantities) per mixed cell, every inventory", "pass": ok}


def frozen_model(output, manifest, arm, seed, device):
    entry = manifest["entries"][f"{arm}--seed{seed}"]
    path = ROOT / entry["model"]
    if eb.sha256_of(path) != entry["model_sha256"]:
        raise ValueError(f"{arm}/{seed}: model differs from the frozen readout manifest")
    model, payload = load_model(path, device)
    return model, payload, entry


@torch.no_grad()
def score_arm(arm, seed, model, carriers, cells, ends=None, unit=None):
    """Rows of every mixed cell of every inventory from its Gate-B anchor; wall per decision on the grid cells (F1
    includes U's rollout)."""
    inventories = {inv: {member: {"rows": cell_rows(arm, model, entry["inventory"], carriers[member],
                                                    ends[inv][member] if arm == "F1" else None)}
                         for member, entry in sorted(cells[inv]["mixed"].items())} for inv in eb.INVENTORIES}

    def decide(mc):
        member, entry = mc
        items = entry["inventory"]
        e = None
        if arm == "F1":
            e = unit.endpoints(torch.as_tensor(carriers[member], dtype=torch.float32)[None].expand(len(items), -1),
                               launch_stack(items, eb.launch5))
        return cell_rows(arm, model, items, carriers[member], e)
    wall = eb.wall_per_decision(decide, sorted(cells[GRID]["mixed"].items()))
    return inventories, wall


def score(output, device=None):
    output = Path(output)
    load_plan(output)
    manifest = load_manifest(output)
    device = device_of(device)
    costs = s129.objective_costs()
    with harness.deterministic_scoring() as policy:
        apath = eb.evaluation_anchors_path(output)
        anchors = (torch.load(apath, map_location="cpu", weights_only=True) if apath.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        write_strata(output, anchors["frames"], cells)
        carriers = anchors["carriers"]
        for seed in eb.SEEDS:
            todo = [arm for arm in ARMS if not record_path(output, arm, seed).is_file()]
            need_rep = not replication_path(output, seed).is_file()
            if not todo and not need_rep:
                continue
            began = time.monotonic()
            unit = s129.UnitRollout(seed, device)
            ends = u_endpoints(unit, carriers, cells) if unit.model is not None else None
            if need_rep:
                rep = ({"pass": False, "reason": f"U seed {seed}: {unit.failure}"} if ends is None else
                       replication_block(seed, ends, carriers, cells, costs, eb.rc.gate_b_rows("unit", seed)))
                eb.write_json(replication_path(output, seed), {
                    "schema": SCHEMA_REPLICATION, "plan_identity": IDENTITY, "seed": seed, "u_identity": u_identity(seed),
                    "reference": rel(eb.GATE_B / "records" / f"unit--seed{seed}.json"), **rep,
                    "scoring_policy": policy, "written_at": eb.utc_now()})
                log(f"U seed {seed} replication vs Gate-B: {rep['pass']}")
            for arm in todo:
                t0 = time.monotonic()
                model, payload, entry = frozen_model(output, manifest, arm, seed, device)
                base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
                        "config_name": payload["config_name"], "model_sha256": entry["model_sha256"],
                        "u_checkpoint_sha256": u_identity(seed)["sha256"] if arm == "F1" else None}
                if model is None:
                    record = {**base, "typed_failure": payload["typed_failure"]["typed_failure"], "inventories": {},
                              "wall_per_decision": None}
                elif arm == "F1" and ends is None:
                    record = {**base, "typed_failure": f"u_{unit.failure}", "inventories": {}, "wall_per_decision": None}
                else:
                    inventories, wall = score_arm(arm, seed, model, carriers, cells, ends, unit)
                    record = {**base, "typed_failure": None, "inventories": inventories, "wall_per_decision": wall}
                seconds = time.monotonic() - t0
                record.update(scoring_policy=policy, scored_at=eb.utc_now(), readouts_frozen_at=manifest["frozen_at"],
                              anchors_read_at=anchors["read_at"], score_seconds=seconds)
                eb.write_json(record_path(output, arm, seed), record)
                log(f"scored {arm} seed {seed} ({seconds:.0f}s)")
            seconds = time.monotonic() - began
            eb.mark(output, f"score-seed{seed}", seconds, seconds if device != "cpu" else 0)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# guards (the publication and --score-novelty use exactly this computation)
# ---------------------------------------------------------------------------------------------------------------

def load_records(output):
    return {arm: {seed: eb.read_json(record_path(output, arm, seed)) for seed in eb.SEEDS} for arm in ARMS}


def records_aligned(records, cells):
    """Every scored record holds exactly the mixed cells of every inventory, rows in inventory ordinal order."""
    for record in records.values():
        if record.get("typed_failure"):
            continue
        for inv in eb.INVENTORIES:
            got = record["inventories"].get(inv, {})
            if sorted(got) != sorted(cells[inv]["mixed"]):
                return False
            for member, entry in cells[inv]["mixed"].items():
                if [r["ordinal"] for r in got[member]["rows"]] != [it["ordinal"] for it in entry["inventory"]]:
                    return False
    return True


def arm_guards(output, records, cells):
    """{arm: (guards, extra guards)}: eb.arm_guard_block over G-rank / G-nonfinite / G-trained plus the declared
    extra guards (smoke checks, training reports, post-freeze replication, record alignment)."""
    output = Path(output)
    smoke_record = eb.read_json(output / "smoke.json")
    plan = eb.read_json(output / "plan.json")
    out = {}
    for arm in ARMS:
        per_seed, reports = {}, []
        for seed in eb.SEEDS:
            rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
            per_seed[seed] = {"G_rank": rd.get("G_rank"), "typed_failure": rd.get("typed_failure")
                              or records[arm][seed].get("typed_failure")}
            report = model_dir(output, arm, seed, plan["selected"]) / "report.json"
            reports.append(eb.read_json(report) if report.is_file() else None)
        guards = eb.arm_guard_block(per_seed, ["G_rank", "G_nonfinite", "G_trained"],
                                    nonfinite=eb.nonfinite_share(records[arm].values(), SCORE_KEY))
        only = bool(smoke_record["classifier_only"]["pass"] and all(
            r is not None and r["classifier_only"]["pass"] and r["features"]["unchanged"] for r in reports))
        extra = {"G_alignment": bool(smoke_record["alignment"]["pass"] and records_aligned(records[arm], cells)),
                 "G_classifier_only": only}
        if arm == "F1":
            reps = [eb.read_json(replication_path(output, s)) if replication_path(output, s).is_file() else None
                    for s in eb.SEEDS]
            extra = {"G_u_identity": bool(all(v["identity"]["pass"] and v["features_identity_matches"]
                                              for v in smoke_record["u"].values())
                                          and all(r is not None and r["u_identity"]["pass"] for r in reps)),
                     "G_rollout_replication": bool(all(v["replication"]["pass"] for v in smoke_record["u"].values())
                                                   and all(r is not None and r["pass"] for r in reps)),
                     **extra}
        out[arm] = (guards, extra)
    return out


def arm_guards_pass(guards, extra):
    merged = {**guards, **extra}
    names = [k for k in merged if k.startswith("G_")]
    return bool(names) and all(merged[k] is True for k in names)


# ---------------------------------------------------------------------------------------------------------------
# --score-novelty (P-novelty; DESCRIPTIVE; after the primary scoring pass)
# ---------------------------------------------------------------------------------------------------------------

def novelty_nn_records(output, cells, anchors):
    """NN-px / NN-car on the mixed novelty levels (#128's exact-launch rule; a launch no fit level executed is
    unavailable: excluded, counted). Written once; seed-free."""
    names = [n for n in NN_NAMES if not novelty_nn_path(output, n).is_file()]
    if names:
        levels = {lv: cells["levels"][lv] for lv in cells["mixed"]}
        orders = s128.nn_orders({lv: anchors["frames"][lv] for lv in levels},
                                {lv: anchors["carriers"][lv] for lv in levels})
        verdicts = s128.fit_verdicts_by_launch(eb.FitShots())
        for name in names:
            record = s128.nn_record(name, {GRID: levels}, orders[name], verdicts)
            rows = [r for c in record["inventories"][GRID].values() for r in c["rows"]]
            record.update(schema=SCHEMA_NOVELTY_NN, plan_identity=IDENTITY,
                          unavailable={"rows": sum(r["excluded"] for r in rows), "of": len(rows)},
                          scored_at=eb.utc_now())
            eb.write_json(novelty_nn_path(output, name), record)
    return {name: eb.read_json(novelty_nn_path(output, name)) for name in NN_NAMES}


@torch.no_grad()
def score_novelty(output, device=None):
    """F1 / F1-D (guard-passing, every seed) and #129's frozen U+head (every seed) on the 16 grid launches of every
    mixed #110 novelty level from its E99 anchor, exactly as the primary paths score a cell; resumable per record."""
    output = Path(output)
    if not (output / "readouts-manifest.json").is_file():
        raise SystemExit("--score-novelty refuses: readouts are not frozen (readouts-manifest.json missing)")
    missing = [f"{a}--seed{s}" for a in ARMS for s in eb.SEEDS if not record_path(output, a, s).is_file()]
    if missing:
        raise SystemExit(f"--score-novelty refuses: primary records missing ({', '.join(missing)}); run --score first")
    load_plan(output)
    manifest = load_manifest(output)
    device = device_of(device)
    records = load_records(output)
    guards = arm_guards(output, records, evaluation_cells())
    status = {arm: {"scored": arm_guards_pass(*guards[arm]),
                    "failed_guards": sorted(k for k, v in {**guards[arm][0], **guards[arm][1]}.items()
                                            if k.startswith("G_") and v is not True)} for arm in ARMS}
    status["U+head"] = {"scored": True, "reason": "#141 memorisation safeguard 3: #129's frozen U+head (control)"}
    costs = s129.objective_costs()
    with harness.deterministic_scoring() as policy:
        cells = eb.novelty_cells()
        apath = eb.novelty_anchors_path(output)
        anchors = (torch.load(apath, map_location="cpu", weights_only=True) if apath.is_file()
                   else eb.read_novelty_anchors(output, policy, cells))
        eb.novelty_reference_records(output, policy, cells, anchors["carriers"])
        novelty_nn_records(output, cells, anchors)
        carriers = anchors["carriers"]
        for seed in eb.SEEDS:
            arms = [a for a in (*ARMS, "U+head") if status[a]["scored"] and not novelty_record_path(output, a, seed).is_file()]
            if not arms:
                continue
            unit = s129.UnitRollout(seed, device)
            for arm in arms:
                began = time.monotonic()
                entry = manifest["entries"][f"{arm}--seed{seed}"]
                base = {"schema": SCHEMA_NOVELTY_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
                        "model_sha256": entry.get("model_sha256") or entry["head_sha256"],
                        "u_checkpoint_sha256": u_identity(seed)["sha256"] if arm != "F1-D" else None,
                        "scoring_policy": policy, "anchors_read_at": anchors["read_at"]}
                grid, failure = {}, None
                if arm == "U+head":
                    if eb.sha256_of(uhead_path(seed)) != entry["head_sha256"]:
                        raise ValueError(f"#129 U+head seed {seed}: head differs from the frozen manifest")
                    head, _ = s129.load_head(uhead_path(seed), device)
                    if unit.model is None or head is None:
                        failure = unit.failure or "head_typed_failure"
                    else:
                        grid = {lv: {"rows": s129.cell_rows("U+head", unit.endpoints, head,
                                                            cells["levels"][lv]["inventory"], carriers[lv], costs)}
                                for lv in cells["mixed"]}
                else:
                    model, payload, _ = frozen_model(output, manifest, arm, seed, device)
                    if model is None:
                        failure = payload["typed_failure"]["typed_failure"]
                    elif arm == "F1" and unit.model is None:
                        failure = f"u_{unit.failure}"
                    else:
                        for lv in cells["mixed"]:
                            items = cells["levels"][lv]["inventory"]
                            ends = None
                            if arm == "F1":
                                z0 = torch.as_tensor(carriers[lv], dtype=torch.float32)[None].expand(len(items), -1)
                                ends = unit.endpoints(z0, launch_stack(items, eb.launch5))
                            grid[lv] = {"rows": cell_rows(arm, model, items, carriers[lv], ends)}
                record = {**base, "typed_failure": failure, "inventories": {GRID: grid}, "scored_at": eb.utc_now()}
                eb.write_json(novelty_record_path(output, arm, seed), record)
                seconds = time.monotonic() - began
                eb.mark(output, f"score-novelty-{arm}-seed{seed}", seconds, seconds if device != "cpu" else 0)
                log(f"novelty: scored {arm} seed {seed} ({seconds:.0f}s)")
    eb.write_json(output / "novelty" / "arms.json", {"plan_identity": IDENTITY, "arms": status})
    return 0


def novelty_section(output, records):
    """DESCRIPTIVE P-novelty block (no tokens, never in a disposition) when novelty records exist."""
    output = Path(output)
    arms_path = output / "novelty" / "arms.json"
    if not arms_path.is_file():
        return None
    status = eb.read_json(arms_path)["arms"]
    scored = [a for a in (*ARMS, "U+head") if all(novelty_record_path(output, a, s).is_file() for s in eb.SEEDS)]
    if not scored:
        return None
    cells = eb.novelty_cells()
    anchors = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)
    refs = eb.novelty_reference_records(output, None, cells, anchors["carriers"])  # read back, never rewritten
    keys = {"F1": SCORE_KEY, "F1-D": SCORE_KEY, "U+head": HEAD_KEY}
    views = {a: ({s: eb.read_json(novelty_record_path(output, a, s)) for s in eb.SEEDS}, keys[a]) for a in scored}
    nn = {name: eb.read_json(novelty_nn_path(output, name)) for name in NN_NAMES if novelty_nn_path(output, name).is_file()}
    views.update({name: ({s: rec for s in eb.SEEDS}, SCORE_KEY) for name, rec in nn.items()})
    imported = {a: ({s: eb.read_json(imported_novelty_path(a, s)) for s in eb.SEEDS}, SCORE_KEY)
                for a in IMPORTED_NOVELTY if all(imported_novelty_path(a, s).is_file() for s in eb.SEEDS)}
    last_primary = max(r["scored_at"] for by in records.values() for r in by.values())
    first = min(r["scored_at"] for a in scored for r in views[a][0].values())
    chronology = {"anchors_read_at": anchors["read_at"], "last_primary_scored_at": last_primary,
                  "first_novelty_scored_at": first,
                  "pass": bool(last_primary <= anchors["read_at"] <= first)}
    return {"label": "DESCRIPTIVE (P-novelty; no tokens, never in a disposition)",
            "table": eb.novelty_table({**views, **imported, **refs}, cells), "arms": status,
            "nn_unavailable": {name: rec["unavailable"] for name, rec in nn.items()},
            "imported": {a: [rel(imported_novelty_path(a, s)) for s in eb.SEEDS] for a in imported},
            "scoring_policy_pass": all(r.get("scoring_policy") == harness.SCORING_POLICY
                                       for a in scored for r in views[a][0].values()),
            "chronology": chronology}


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ["records/*--seed*.json", "replication/*.json", "strata/evaluation-distances.json",
                   "novelty/records/*--seed*.json", "novelty/references/*.json", "novelty/nn/*.json"]
PUBLISHED = ("compute.json", "summary.json", "findings.md")


def evaluation_cells():
    return eb.evaluation_cells()


def references():
    return eb.references()


def reference_compute():
    return eb.reference_compute()


def imported_views():
    """{view: ({seed: record}, key)} of A1, A2 (#128), U+head (#129) and the seed-free NN-px / NN-car (#128)."""
    views = {arm: ({s: eb.read_json(imported_record_path(arm, s)) for s in eb.SEEDS}, key)
             for arm, (_, key) in IMPORTED_ARMS.items()}
    for name in NN_NAMES:
        record = eb.read_json(imported_nn_path(name))
        views[name] = ({s: record for s in eb.SEEDS}, "neg_score")
    return views


def imported_check():
    """Each imported record matches its child's records-manifest.json, whose validation receipt passes."""
    files, children = [], {}
    for base in (S128, S129):
        guard = eb.records_guard(base, CHILD_IDENTITY[base])
        children[rel(base)] = {"records_guard_pass": bool(guard["pass"])}
    manifests = {base: {e["artifact"]: e["sha256"] for e in eb.read_json(base / "records-manifest.json")["records"]}
                 for base in (S128, S129)}
    for path in [imported_record_path(a, s) for a in IMPORTED_ARMS for s in eb.SEEDS] + [imported_nn_path(n) for n in NN_NAMES]:
        base = S128 if str(path).startswith(str(S128)) else S129
        listed = manifests[base].get(rel(path))
        files.append({"artifact": rel(path), "listed": listed is not None, "match": listed == eb.sha256_of(path)})
    ok = all(f["match"] for f in files) and all(c["records_guard_pass"] for c in children.values())
    return {"pass": bool(ok), "files": files, "children": children}


def imported_guards_pass():
    """#128's published guards_pass of A1 / A2 (guarding F1 - A1 / F1 - A2)."""
    arms = eb.read_json(S128 / "summary.json")["arms"]
    return {arm: bool(arms[arm]["guards_pass"]) for arm in ("A1", "A2")}


def a2_inherited():
    """A2's G-port / G-fid as #128 published them (inherited by the classifier; reported in extra.scope)."""
    summary = eb.read_json(S128 / "summary.json")
    return {"G_port": summary["arms"]["A2"]["guards"].get("G_port"),
            "G_fid": summary["arms"]["A2"]["guards"].get("G_fid"), "gfid": summary["gfid"].get("A2")}


def global_guards(output, plan, records, cells):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    flat = [r for by in records.values() for r in by.values()]
    anchors_read = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)["read_at"]
    first_scored = min(r["scored_at"] for r in flat)
    cuts = eb.read_json(cut_points_path(output))["written_at"]
    first_training = min(eb.read_json(selection_dir(output, arm, name) / "report.json")["started_at"]
                         for arm in ARMS for name in CONFIGS)
    chronology = bool(cuts <= first_training and plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read
                      <= first_scored)
    novelty = None
    if eb.novelty_anchors_path(output).is_file():
        nread = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)["read_at"]
        nscored = [eb.read_json(p)["scored_at"] for p in sorted((output / "novelty" / "records").glob("*.json"))]
        novelty = {"anchors_read_at": nread, "last_primary_scored_at": max(r["scored_at"] for r in flat),
                   "pass": bool(nread >= max(r["scored_at"] for r in flat) and all(s >= nread for s in nscored))}
        chronology = chronology and novelty["pass"]
    roster = {level["identity"] for level in eb.gb.evaluation_levels()}
    scored = [r for r in flat if not r["typed_failure"]]
    isolation = bool(all(set(r["inventories"][GRID]) <= roster for r in scored)
                     and all(k.startswith("issue-109-sc-fit") for k in eb.FitShots().keys)
                     and not set(eb.policy_cells(GRID)) & roster)
    reps = [eb.read_json(replication_path(output, s)) for s in eb.SEEDS if replication_path(output, s).is_file()]
    policies = [r["scoring_policy"] for r in flat] + [r["scoring_policy"] for r in reps]
    own = eb.records_guard(output, IDENTITY)
    imported = imported_check()
    return {
        "G_isolation": {"pass": isolation, "rule": "fit split trains, policy split selects, evaluation anchors are "
                        "read only by --score after --freeze-readouts; no policy level is an evaluation level"},
        "G_chronology": {"pass": chronology, "cut_points_written_at": cuts, "first_training_started_at": first_training,
                         "plan_frozen_at": plan["frozen_at"], "readouts_frozen_at": manifest["frozen_at"],
                         "anchors_read_at": anchors_read, "first_scored_at": first_scored, "novelty": novelty},
        "G_scoring_policy": {"pass": all(p == harness.SCORING_POLICY for p in policies)},
        "G_records": {**own, "pass": bool(own["pass"] and imported["pass"]), "own_pass": bool(own["pass"]),
                      "imported": imported},
    }


def arm_compute(output, plan, arm, records, reference):
    """Training GPU-s per seed (F1: U's own training + U feature pass + classifier; F1-D: classifier), MACs per
    candidate (F1 incl. U's rollout), wall per decision (Decision compute_and_queues). reference: eb.reference_compute()."""
    u_training = reference["U"]["training"] if arm == "F1" else {}
    seconds, breakdown, macs = {}, {}, []
    for seed in eb.SEEDS:
        report = eb.read_json(model_dir(output, arm, seed, plan["selected"]) / "report.json")
        parts = {"classifier": float(report["gpu_seconds"])}
        if arm == "F1":
            f = torch.load(features_path(output, seed), map_location="cpu", weights_only=True)
            parts["u_feature_pass"] = float(f["seconds"]) if f["device"] != "cpu" else 0.0
            parts["u_training"] = float(u_training[str(seed)]["training_gpu_seconds"])
        seconds[str(seed)] = sum(parts.values())
        breakdown[str(seed)] = parts
        rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
        if rd.get("macs_per_candidate") is not None:
            macs.append(rd["macs"])
    walls = [r["wall_per_decision"] for r in records.values() if r.get("wall_per_decision") is not None]
    return {"trainable_parameters": plan["arms"][arm]["parameters"], "training_gpu_seconds": seconds,
            "macs_per_candidate": float(np.mean([m["total"] for m in macs])) if macs else None,
            "wall_per_decision": float(np.median(walls)) if walls else None,
            "training_breakdown_gpu_seconds": breakdown,
            "macs_breakdown": {k: float(np.mean([m[k] for m in macs])) for k in ("classifier", "u_rollout")} if macs
            else None, "frozen_u_parameters": reference["U"]["trainable_parameters"] if arm == "F1" else 0}


def grid_contrasts(rows, f1_pass, f1d_pass, global_pass, imported_pass):
    """Within-family (guarded) and descriptive (NN, no token) contrasts on the grid rows."""
    f1_ok = bool(global_pass and f1_pass)
    within = {"F1 - F1-D": eb.within_family(rows, "F1", "F1-D", f1_ok and f1d_pass),
              "F1 - A1": eb.within_family(rows, "F1", "A1", f1_ok and imported_pass["A1"]),
              "F1 - A2": eb.within_family(rows, "F1", "A2", f1_ok and imported_pass["A2"]),
              "F1 - U+head": eb.within_family(rows, "F1", "U+head", f1_ok)}
    descriptive = {f"F1 - {name}": {**eb.within_family(rows, "F1", name, False), "label": f"F1 - {name}",
                                    "descriptive": True} for name in NN_NAMES}
    return within, descriptive


def strata_table(strata, rows, names=STRATA_RANKERS):
    """Per stratum (frozen cut points, grid mixed cells): eb.ranker_table blocks of the named rankers and counts."""
    out = {"cut_points": strata["cut_points"], "metric": strata["metric"], "rankers": list(names), "strata": {},
           "label": "DESCRIPTIVE (memorisation safeguard 2; no tokens; no stratum is selected)"}
    for stratum in STRATA:
        members = {m for m, lv in strata["levels"].items() if lv["stratum"] == stratum}
        srows = [r for r in rows if r["member"] in members]
        out["strata"][stratum] = {"cells": len({r["member"] for r in srows}),
                                  "table": eb.ranker_table(srows, list(names)) if srows else None}
    return out


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = evaluation_cells()
    refs = references()
    gg = global_guards(output, plan, records, cells)
    global_pass = all(g["pass"] is True for g in gg.values())
    guards = arm_guards(output, records, cells)
    views = {"F1": (records["F1"], SCORE_KEY), "F1-D": (records["F1-D"], SCORE_KEY), **imported_views()}
    a_pass = imported_guards_pass()
    reference = reference_compute()
    compute = {arm: arm_compute(output, plan, arm, records[arm], reference) for arm in ARMS}
    blocks, tables, within, descriptive, strata = {}, {}, {}, {}, None
    for inv in eb.INVENTORIES:
        rows = eb.view_rows(cells, inv, views, refs)
        tables[inv] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", *views])}
        if inv == GRID:
            for arm in ARMS:
                blocks[arm] = eb.arm_block(rows, arm, records[arm], SCORE_KEY, guards[arm][0], global_pass,
                                           compute[arm], ARMS[arm]["role"], ARMS[arm]["program_set"],
                                           extra_guards=guards[arm][1])
            within, descriptive = grid_contrasts(rows, blocks["F1"]["guards"]["pass"], blocks["F1-D"]["guards"]["pass"],
                                                 global_pass, a_pass)
            strata = strata_table(eb.read_json(strata_path(output)), rows)
        else:
            tables[inv]["contrasts"] = {f"{arm} - U": eb.contrast(rows, arm, "U", global_pass and
                                                                  blocks[arm]["guards"]["pass"]) for arm in ARMS}
    readouts_view = {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_grid_auc", "G_rank", "G_action_inherited", "macs",
                                                "typed_failure", "config_name")}
                     for a, s in readout_items()}
    smoke_record = eb.read_json(output / "smoke.json")
    scope = {**SCOPE, "g_action_inherited": {s: readouts_view[f"F1--seed{s}"].get("G_action_inherited")
                                             for s in eb.SEEDS},
             "a2_inherited": a2_inherited()}
    selection = eb.read_json(output / "selection.json")
    extra = {
        "scope": scope, "inventories": tables, "within_family_descriptive": descriptive, "strata": strata,
        "selection": plan["selected"], "selection_results": {a: b["results"] for a, b in selection["arms"].items()},
        "readouts": readouts_view,
        "replication": {"pre_freeze": {s: v["replication"] for s, v in smoke_record["u"].items()},
                        "post_freeze": {str(s): {k: v for k, v in eb.read_json(replication_path(output, s)).items()
                                                 if k in ("pass", "inventories", "rule", "tolerance", "reason")}
                                        for s in eb.SEEDS}},
        "imported_guards_pass": a_pass, "novelty": novelty_section(output, records),
        "decisions": DECISIONS,
    }
    gfid = {arm: {"pass": None, "note": SCOPE["port"]} for arm in ARMS}
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, within, gg, gfid, extra)
    ledger, gpu_total, wall_total = eb.ledger_totals(output, report_of_factory(output, plan))
    compute_json = {"schema": SCHEMA_COMPUTE, "plan_identity": IDENTITY, "ledger": ledger,
                    "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
                    "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": reference,
                    "projection_at_smoke": plan["budget"]["projection"]}
    return summary, compute_json


def _fmt_ranker(block):
    return eb.rc.fmt_auc(block["auc"]) if block else "n/a"


def findings_text(summary):
    x = summary["extra"]
    lines = [f"# #{ISSUE} P-follow-up F1 (U + the A-family's success readout): findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configurations {x['selection']}.", "",
             f"Scope: {SCOPE['declared']} G-action: {SCOPE['g_action']}", "",
             *eb.format_arm_rows(summary["arms"]), "", "## Within family (guarded)", ""]
    for label, block in summary["within_family"].items():
        lines.append(f"- {label}: {eb.rc.fmt_block(block['estimate'])} ({block['token']} / {block['reverse_token']})")
    lines += ["", "## No-model references (DESCRIPTIVE, no tokens)", ""]
    grid = x["inventories"][GRID]["rankers"]
    for name in NN_NAMES:
        block = x["within_family_descriptive"][f"F1 - {name}"]
        lines.append(f"- {name}: grid AUC {_fmt_ranker(grid.get(name))}; F1 - {name} {eb.rc.fmt_block(block['estimate'])}")
    strata = x.get("strata")
    if strata:
        lines += ["", f"## Distance strata (DESCRIPTIVE; cut points {strata['cut_points']}, grid mixed cells)", "",
                  "| stratum | cells | " + " | ".join(strata["rankers"]) + " |",
                  "| --- | --- |" + " --- |" * len(strata["rankers"])]
        for stratum, block in strata["strata"].items():
            table = (block["table"] or {}).get("rankers", {})
            lines.append(f"| {stratum} | {block['cells']} | " + " | ".join(_fmt_ranker(table.get(n)) for n in
                                                                       strata["rankers"]) + " |")
    novelty = x.get("novelty")
    if novelty is not None:
        lines += ["", "## Novelty levels (P-novelty; DESCRIPTIVE, no tokens, never in a disposition)", "",
                  *eb.format_novelty_rows(novelty["table"]), ""]
        for arm, entry in novelty["arms"].items():
            if not entry["scored"]:
                lines.append(f"- {arm}: not scored (failed {', '.join(entry['failed_guards'])})")
        for name, n in novelty["nn_unavailable"].items():
            lines.append(f"- {name}: {n['rows']} of {n['of']} candidates unavailable (launch never executed in a fit level)")
    lines += ["", "Intervals: level-clustered percentile bootstrap (10 000 draws, PCG64 7201); tokens follow the #127 "
              "delta rule under the declared guard scope.", ""]
    return "\n".join(lines)


def rendered(summary, compute_json):
    return {"compute.json": eb.json_text(compute_json), "summary.json": eb.json_text(summary),
            "findings.md": findings_text(summary)}


def write_publication(output):
    output = Path(output)
    with harness.deterministic_scoring():
        summary, compute_json = publication(output)
    for name, text in rendered(summary, compute_json).items():
        (output / name).write_text(text)
    return summary


def publish(output):
    """records-manifest.json and the tables; every token is the stop token until --validate binds G-records."""
    output = Path(output)
    eb.write_json(output / "records-manifest.json", eb.records_manifest(output, IDENTITY, RECORD_PATTERNS))
    summary = write_publication(output)
    log(f"published; G-records {summary['global_guards']['G_records']['pass']}")
    return 0


def spot_check(output, device=None):
    """Re-score F1 seed 20260908 (U rollout + classifier) on every mixed cell of every inventory; scores must
    replicate (relative 1e-4, #128's rule)."""
    output = Path(output)
    device = device_of(device)
    arm, seed = "F1", eb.SELECTION_SEED
    stored = eb.read_json(record_path(output, arm, seed))
    if stored["typed_failure"]:
        arm = "F1-D"
        stored = eb.read_json(record_path(output, arm, seed))
        if stored["typed_failure"]:
            raise ValueError("no complete record to re-score")
    manifest = load_manifest(output)
    anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
    cells = eb.evaluation_cells()
    with harness.deterministic_scoring():
        unit = s129.UnitRollout(seed, device)
        ends = u_endpoints(unit, anchors["carriers"], cells) if arm == "F1" else None
        model, _, _ = frozen_model(output, manifest, arm, seed, device)
        fresh, _ = score_arm(arm, seed, model, anchors["carriers"], cells, ends, unit)
    worst = worst_rel = 0.0
    for inv in eb.INVENTORIES:
        for member, block in stored["inventories"][inv].items():
            for old, new in zip(block["rows"], fresh[inv][member]["rows"]):
                if (old["score"] is None) != (new["score"] is None) or old["ordinal"] != new["ordinal"]:
                    worst = worst_rel = math.inf
                elif old["score"] is not None:
                    delta = abs(old["score"] - new["score"])
                    worst = max(worst, delta)
                    worst_rel = max(worst_rel, delta / max(1.0, abs(old["score"])))
    return {"record": f"{arm}--seed{seed}", "inventories": list(eb.INVENTORIES), "max_abs_delta": worst,
            "max_relative_delta": worst_rel, "tolerance_relative": 1e-4, "pass": bool(worst_rel <= 1e-4)}


def validate(output, device=None):
    """Hash-check the records against the manifest, re-derive every published byte, re-score one record, write the
    receipt bound to the records manifest, re-derive the publication."""
    output = Path(output)
    load_plan(output)
    spot = None
    try:
        manifest = eb.read_json(output / "records-manifest.json")
        for entry in manifest["records"]:
            if eb.sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
                raise ValueError(f"record {entry['artifact']} differs from records-manifest.json")
        if manifest != eb.records_manifest(output, IDENTITY, RECORD_PATTERNS):
            raise ValueError("records-manifest.json does not list exactly the current records")
        with harness.deterministic_scoring():
            summary, compute_json = publication(output)
        for name, text in rendered(summary, compute_json).items():
            if (output / name).read_bytes() != text.encode():
                raise ValueError(f"published {name} differs from the recomputation")
        spot = spot_check(output, device)
        if not spot["pass"]:
            raise ValueError(f"re-scoring does not replicate the retained record: {spot}")
    except Exception as error:
        for name in PUBLISHED:
            (output / name).unlink(missing_ok=True)
        eb.write_receipt(output, IDENTITY, spot, f"{type(error).__name__}: {error}")
        write_publication(output)
        log(f"validation FAILED: publication re-derived with every token {eb.STOP_TOKEN}")
        raise
    eb.write_receipt(output, IDENTITY, spot)
    summary = write_publication(output)
    if not summary["global_guards"]["G_records"]["pass"]:
        raise ValueError("G-records does not pass after the receipt (imported records or receipt binding)")
    log(f"validation passed: {len(manifest['records'])} records, spot check {spot}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

MODES = ("cut-points", "selection", "selection-readouts", "smoke", "prepare", "train", "readouts", "freeze-readouts",
         "score", "score-novelty", "publish", "validate")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index = the scheduler GPU of the job")
    parser.add_argument("--workers", type=int, default=0, help="accepted for the contract; data live on the device")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    mode = next(m for m in MODES if getattr(args, m.replace("-", "_")))
    if mode == "cut-points":
        return cut_points(output)
    if mode == "selection":
        return selection(output, args.gpu, args.device)
    if mode == "selection-readouts":
        return selection_readouts(output, args.device)
    if mode == "smoke":
        return smoke(output, args.device)
    if mode == "prepare":
        return prepare(output)
    if mode == "train":
        return train(output, args.gpu, args.device)
    if mode == "readouts":
        return readouts(output, args.gpu, args.device)
    if mode == "freeze-readouts":
        return freeze_readouts(output)
    if mode == "score":
        return score(output, args.device)
    if mode == "score-novelty":
        return score_novelty(output, args.device)
    if mode == "publish":
        return publish(output)
    return validate(output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
