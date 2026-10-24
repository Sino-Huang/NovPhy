"""Issue #130 (#127 child C1): DINO-WM against U on the sealed NovPhy launch-ranking grid.

Arms (Decisions P-other on #130; the #127 frozen arm matrix):
- **C1-dino** (primary, program set): stopped at the readiness smoke before any predictor training (typed
  ``readiness_ceiling_below_threshold``: R1 policy ceiling 0.5948 < 0.65, issuecomment-6383216316). It is published
  as a dropped arm (no records, every token the stop token) and vetoes the #127 program claim; its readiness record
  is pinned in the plan.
- **C1-car** (secondary): the official DINO-WM predictor (ViTPredictor, causal frame blocks, concat_dim 1 action
  tiling, z_loss, AdamW; ``world_model/baselines/dino_wm/novphy.py``, G-port) on E99 carriers as 18 slot tokens
  (13 slot features + the tiled 2-d header, fit-split z-scored), launch at frame 0 + no-ops (frame-skip 5, 45 steps to
  frame 225), history 3; score = tie-free ``EndpointCosts`` of the decoded predicted frame-225 carrier. Configuration
  selected on the policy split among ``CONFIGS`` (seed 20260908, full budget); the selected run is reused as seed
  20260908.
- Within family, C1-dino - C1-car cannot be computed (C1-dino not run): published as an absent estimate (stop token).

Modes (round3/child-runner-contract.md): --selection, --selection-readouts, --smoke, --prepare, --train, --readouts,
--freeze-readouts, --score, --score-novelty (P-novelty, round3/score-novelty-mode.md; DESCRIPTIVE), --publish,
--validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_130.json); tests use CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import external_baselines as eb  # noqa: E402
from world_model.baselines.dino_wm import novphy as dn  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 130
IDENTITY = "issue-130-dino-wm-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_130_dino_wm.py"
TESTS = "tests/test_issue_130_dino_wm.py"
SCHEMA_MODEL = "issue_130_model_v1"
SCHEMA_PLAN = "issue_130_plan_v1"
SCHEMA_RECORD = "issue_130_record_v1"
SCHEMA_SMOKE = "issue_130_smoke_v1"
SCHEMA_SELECTION = "issue_130_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_130_readouts_manifest_v1"
SCHEMA_NOVELTY_RECORD = "issue_130_novelty_record_v1"
DECISIONS = {
    "band": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382394238",
    "license": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382422982",
    "tier_e_schedule": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382562140",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605",
    "c1_dino_stop": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383216316",
    "tier_e_resumable": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383358026",
    "selection": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386706483",
    "novelty": "https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386831073",
}

ARM = "C1-car"
DINO = "C1-dino"
DINO_STOP = "readiness_ceiling_below_threshold"
BATCH = 32  # train.yaml / the released PushT hydra.yaml training.batch_size (effective)
EPOCHS = 100  # train.yaml / hydra.yaml training.epochs
PUSHT_RELEASE_SAMPLES = 2 * 1_981_721  # released PushT checkpoint: epoch 2 x its train windows (num_hist+1 frames, fs 5)
CONFIGS = {  # pre-declared (Decision P-other, selection); native size, batch 32, seed 20260908
    "car-lr5e-4": {"predictor_lr": 5e-4, "action_encoder_lr": 5e-4, "epochs": EPOCHS, "samples": None},
    "car-lr5e-5": {"predictor_lr": 5e-5, "action_encoder_lr": 5e-4, "epochs": EPOCHS, "samples": None},
    "car-release": {"predictor_lr": 5e-4, "action_encoder_lr": 5e-4, "epochs": None, "samples": PUSHT_RELEASE_SAMPLES},
}
SELECTION_QUEUES = {0: ["car-lr5e-4"], 1: ["car-lr5e-5"], 2: ["car-release"]}
ARMS = {ARM: {"role": "secondary", "program_set": False}}
CHECKPOINT_EVERY = 20_000
LOG_EVERY = 500
CHILD_CAP_GPU_HOURS = 88.0  # upper end of the #127 C1 child share (now C1-car + fidelity only)
SEED_CAP_GPU_HOURS = 20.0  # #127: C1 cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
GRID = "grid"
READINESS = OUTPUT / "readiness/readiness.json"


def log(message):
    eb.log(ISSUE, message)


def rel(path):
    """Repo-relative path string (absolute for paths outside the repo, e.g. test outputs)."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------
# configuration and paths
# ---------------------------------------------------------------------------------------------------------------

def steps_per_epoch(windows, batch=BATCH):
    """The official DataLoader (shuffle False over the once-permuted TrajSlicerDataset, drop_last False)."""
    return math.ceil(windows / batch)


def config_spec(name, windows):
    c = CONFIGS[name]
    steps = (c["epochs"] * steps_per_epoch(windows) if c["epochs"] is not None
             else math.ceil(c["samples"] / BATCH))
    return {"name": name, "arm": ARM, "batch": BATCH, "epochs": c["epochs"], "samples": c["samples"], "steps": steps,
            "predictor_lr": c["predictor_lr"], "action_encoder_lr": c["action_encoder_lr"], "windows": windows}


def arm_spec(selected, windows):
    return {**config_spec(selected, windows), "arm": ARM}


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, seed, selected):
    """Seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if seed == eb.SELECTION_SEED:
        return selection_dir(output, selected)
    return Path(output) / "models" / ARM / f"seed-{seed}"


def readouts_dir(output, seed):
    return Path(output) / "readouts" / ARM / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


# ---------------------------------------------------------------------------------------------------------------
# fit data: stride-5 carriers, windows, batches
# ---------------------------------------------------------------------------------------------------------------

class FitCarriers:
    """Every fit shot's E99 carriers at frame-skip-5 positions 0 .. min(225, end) (NaN beyond), the fit z-score
    statistics, the launches and the training windows (shot, start step) of num_hist + num_pred = 4 frames."""

    def __init__(self, shots=None, limit=None):
        shots = shots or eb.FitShots()
        n = len(shots) if limit is None else min(limit, len(shots))
        frames = dn.STEPS + 1
        last = np.minimum(dn.HORIZON_FRAME, np.asarray(shots.ends[:n], dtype=np.int64)) // dn.FRAMESKIP  # last step
        carriers = torch.full((n, frames, dn.CARRIER_DIM), float("nan"))
        index_shot = np.repeat(np.arange(n), last + 1)
        index_step = np.concatenate([np.arange(k + 1) for k in last])
        rows = shots.carrier_rows(index_shot, index_step * dn.FRAMESKIP).float()
        carriers[torch.from_numpy(index_shot), torch.from_numpy(index_step)] = rows
        self.carriers = carriers
        self.mean, self.std = dn.carrier_stats(rows)
        self.launch5 = shots.launch5[:n].clone()
        window = dn.NUM_HIST + dn.NUM_PRED
        self.windows = torch.tensor([(s, t) for s in range(n) for t in range(int(last[s]) - window + 2)],
                                    dtype=torch.long)
        self.last = torch.from_numpy(last)
        self.shots = n

    def to(self, device):
        """The carriers and windows stay in (pinned) host memory and each batch is gathered on the host (so a
        selection job needs well under 1 GB of GPU memory beside the Tier-E job); statistics and launches move."""
        self.device = device
        for k in ("mean", "std", "launch5"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def batch(self, index):
        """tokens [B, 4, 18, 15] and actions [B, 4, 25] of windows[index] (index: host tensor)."""
        device = getattr(self, "device", "cpu")
        shot, start = self.windows[index, 0], self.windows[index, 1]
        steps = start[:, None] + torch.arange(dn.NUM_HIST + dn.NUM_PRED)[None]
        carriers = self.carriers[shot[:, None], steps].to(device, non_blocking=True)
        tokens = dn.carrier_to_tokens(carriers, self.mean, self.std)
        # dn.window_actions on the device: only a window starting at step 0 holds the launch (frame 0 of step 0)
        acts = torch.zeros(len(shot), dn.NUM_HIST + dn.NUM_PRED, dn.ACTION_DIM, device=tokens.device)
        first = (start == 0).to(device)
        acts[first, 0, :dn.LAUNCH_DIM] = self.launch5[shot.to(device)[first]]
        return tokens, acts


class FixedOrderBatches:
    """Batch b = slice b mod spe of ONE permutation (eb.batch_rng(seed, 0)) reused every epoch: the official
    TrajSlicerDataset permutes its slices once at construction and the DataLoader does not shuffle; the last batch
    of an epoch is partial (drop_last False). Deterministic and resumable at any step."""

    def __init__(self, seed, n, batch=BATCH):
        self.n, self.batch = n, batch
        self.spe = steps_per_epoch(n, batch)
        self.perm = torch.from_numpy(eb.batch_rng(seed, 0).permutation(n))

    def __call__(self, b):
        i = b % self.spe
        return self.perm[i * self.batch:(i + 1) * self.batch]


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def make_optimizers(model, spec):
    opts = dn.make_optimizers(model)
    for group in opts["predictor"].param_groups:
        group["lr"] = spec["predictor_lr"]
    for group in opts["action_encoder"].param_groups:
        group["lr"] = spec["action_encoder_lr"]
    return opts


def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one C1-car model (resumable from out_dir/resume.pt); data: FitCarriers on the device."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    steps = spec["steps"]
    torch.manual_seed(seed)
    model = dn.build(ARM).to(device)
    optimizers = make_optimizers(model, spec)
    start, elapsed = 0, 0.0
    resume = out_dir / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for name, opt in optimizers.items():
            opt.load_state_dict(state["optimizers"][name])
        start, elapsed = state["step"], state["elapsed"]
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
        log(f"{spec['name']} seed {seed}: resumed at step {start}")
    batches = FixedOrderBatches(seed, len(data.windows))
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            tokens, acts = data.batch(batches(step))
            _, loss, components = dn.train_step(model, optimizers, tokens, acts)
            if not bool(torch.isfinite(loss)):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": "nonfinite z_loss"}
                break
            if step % LOG_EVERY == 0 or step == steps - 1:
                entry = {"step": step, "loss": float(loss.detach()), "seconds": elapsed + time.monotonic() - began}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(eb.STOP_REQUESTED) and step + 1 < steps
            if stop or ((step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < steps):
                eb.atomic_torch(resume, {
                    "model": model.state_dict(), "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "step": step + 1, "elapsed": elapsed + time.monotonic() - began})
            if stop:
                log(f"{spec['name']} seed {seed}: stop signal; resume.pt written at step {step + 1}")
                raise eb.TrainingPaused(f"{spec['name']} seed {seed} paused at step {step + 1} of {steps}")
    seconds = elapsed + time.monotonic() - began
    if failure is None and step + 1 < steps:  # max_steps reached (tests, throughput)
        return {"partial": True, "steps_done": step + 1 - start, "seconds": time.monotonic() - began}
    complete = failure is None
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": failure,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "carrier_mean": data.mean.cpu(), "carrier_std": data.std.cpu(),
               "parameters": dn.trainable_parameters(model)}
    eb.atomic_torch(final, payload)
    report = {"spec": spec, "seed": seed, "steps": steps, "complete": complete, "typed_failure": failure,
              "wall_seconds": seconds, "gpu_seconds": seconds if device.startswith("cuda") else 0.0,
              "device": device, "seconds_per_step": seconds / max(1, steps),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": trace[-1] if trace else None, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if resume.is_file() and complete:
        resume.unlink()
    log(f"{spec['name']} seed {seed}: {'complete' if complete else failure['typed_failure']} after {seconds:.0f}s")
    return report


class CarWM:
    """A trained C1-car model with its carrier statistics: carriers in, carriers out."""

    def __init__(self, model, mean, std):
        self.model, self.mean, self.std = model, mean, std
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def rollout(self, z0, launch5, batch=2048):
        """Predicted carriers [B, 46, 236] (frames 0, 5, .., 225; frame 0 = the anchor) from anchors z0 [B, 236]."""
        out = []
        for i in range(0, len(z0), batch):
            z = z0[i:i + batch].to(self.device).float()
            tok = dn.carrier_to_tokens(z, self.mean, self.std)[:, None]
            acts = dn.step_actions(launch5[i:i + batch].cpu()).to(self.device)
            rolled = self.model.rollout(tok, acts)[..., :dn.SLOT_TOKEN_DIM]
            out.append(dn.tokens_to_carrier(rolled, self.mean, self.std).float().cpu())
        return torch.cat(out)

    def endpoints(self, z0, launch5):
        return self.rollout(z0, launch5)[:, -1]


def load_trained(path, device=None):
    """(CarWM, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL:
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    device = device_of(device)
    model = dn.build(ARM)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    return CarWM(model, payload["carrier_mean"].to(device), payload["carrier_std"].to(device)), payload


def run_queue(output, items, device=None, data=None):
    """Train (spec, seed, out_dir) items in order; a stop signal pauses the queue (exit 75)."""
    eb.install_stop_handlers()
    device = device_of(device)
    for spec, seed, out_dir in items:
        if (Path(out_dir) / "model.pt").is_file():
            continue
        data = data or FitCarriers().to(device)
        began = time.monotonic()
        try:
            report = train_run(out_dir, spec, seed, data, device)
        except eb.TrainingPaused as paused:
            seconds = time.monotonic() - began
            eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device != "cpu" else 0)
            log(str(paused))
            return eb.EXIT_PAUSED
        seconds = time.monotonic() - began
        eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device != "cpu" else 0,
                report=rel(Path(out_dir, "report.json")), complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# prediction, costs, policy AUC (deterministic, float32)
# ---------------------------------------------------------------------------------------------------------------

def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


def fit_anchor_endpoints(wm):
    """Predicted frame-225 carriers [480, 16, 236] of the 16 grid launches from the 480 fit anchors (G-rank,
    G-action) and the anchor carriers [480, 236] (sorted members)."""
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)
    items, actions = grid_actions()
    z0 = torch.stack([anchors[m] for m in members]).float()
    ends = wm.endpoints(z0.repeat_interleave(len(items), 0), actions.repeat(len(members), 1))
    return members, z0, ends.reshape(len(members), len(items), -1)


def tie_free_matrix(ends, z0, costs):
    """[L, K] tie-free costs (nonfinite -> nan) of endpoints [L, K, 236] from anchors [L, 236]."""
    out = np.full(ends.shape[:2], np.nan)
    items, _ = grid_actions()
    ordinals = [it["ordinal"] for it in items]
    for i in range(ends.shape[0]):
        for row in eb.endpoint_rows(items, ends[i], z0[i], costs):
            if not row["excluded"]:
                out[i, ordinals.index(row["ordinal"])] = row[eb.PRIMARY_COST]
    return out


def policy_endpoint_rows(wm, costs):
    anchors = eb.policy_anchors()["carriers"]
    out = {}
    for member, entry in sorted(eb.policy_cells(GRID).items()):
        items = entry["inventory"]
        actions = torch.stack([eb.launch5(item["action"]) for item in items])
        z0 = anchors[member].float()
        ends = wm.endpoints(z0[None].expand(len(items), -1), actions)
        out[member] = eb.endpoint_rows(items, ends, z0, costs)
    return out


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, queue, device=None):
    device = device_of(device)
    data = FitCarriers().to(device)
    names = SELECTION_QUEUES[queue]
    items = [(config_spec(name, len(data.windows)), eb.SELECTION_SEED, selection_dir(output, name)) for name in names]
    log(f"selection queue {queue}: {names} ({len(data.windows)} windows)")
    return run_queue(output, items, device, data)


def selection_readouts(output, device=None):
    """Policy-split pooled grid AUC of the primary score per configuration (deterministic scoring); selection.json."""
    costs = objective_costs()
    results, details = {}, {}
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        for name in CONFIGS:
            path = selection_dir(output, name) / "model.pt"
            if not path.is_file():
                raise ValueError(f"selection run {name} is not finished")
            wm, payload = load_trained(path, device)
            if wm is None:
                results[name] = {"auc": None, "parameters": payload["parameters"]}
                details[name] = {"typed_failure": payload["typed_failure"]}
                continue
            rows = policy_endpoint_rows(wm, costs)
            results[name] = {"auc": eb.policy_grid_auc(rows, eb.PRIMARY_COST), "parameters": payload["parameters"]}
            details[name] = {"typed_failure": None, "model_sha256": eb.sha256_of(path), "cells": len(rows)}
    choice = eb.select_configuration(results)
    seconds = time.monotonic() - began
    eb.mark(output, "selection-readouts", seconds, seconds if device_of(device) != "cpu" else 0)
    eb.write_json(Path(output) / "selection.json", {
        "schema": SCHEMA_SELECTION, "plan_identity": IDENTITY, "configs": CONFIGS, **choice, "details": details,
        "scoring_policy": policy, "policy_cells": len(eb.policy_cells(GRID)), "written_at": eb.utc_now()})
    log(f"selection: {choice['selected']} ({json.dumps(results)})")
    return 0 if choice["selected"] is not None else 1


def selected_config(output):
    path = Path(output) / "selection.json"
    if not path.is_file():
        raise ValueError("selection.json missing; run --selection-readouts")
    selected = eb.read_json(path)["selected"]
    if selected is None:
        raise ValueError("no configuration was selected (every selection run failed)")
    return selected


# ---------------------------------------------------------------------------------------------------------------
# --smoke
# ---------------------------------------------------------------------------------------------------------------

def lpt(items, bins=4):
    queues, loads = {b: [] for b in range(bins)}, [0.0] * bins
    for name, seconds in sorted(items, key=lambda t: (-t[1], t[0])):
        b = int(np.argmin(loads))
        queues[b].append(name)
        loads[b] += seconds
    return queues, loads


def remaining_items(output, selected, windows):
    return [(seed, arm_spec(selected, windows), model_dir(output, seed, selected))
            for seed in eb.SEEDS if seed != eb.SELECTION_SEED]


def readiness_check():
    record = eb.read_json(READINESS)
    return {"artifact": rel(READINESS), "sha256": eb.sha256_of(READINESS),
            "ceiling_auc": record["ceiling_auc"], "threshold": record["threshold"], "pass": record["pass"],
            "typed_reason": record["typed_reason"]}


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    shots = eb.FitShots()
    data = FitCarriers(shots)
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    accounting = {"fit_shots": len(shots), "fit_levels": len(shots.levels), "fit_positive": int(shots.verdicts.sum()),
                  "fit_anchors": len(eb.fit_anchors()["carriers"]), "policy_grid_mixed_cells": len(eb.policy_cells(GRID)),
                  "windows": len(data.windows), "steps_per_epoch": steps_per_epoch(len(data.windows)),
                  "short_shots": int((data.last < dn.STEPS).sum())}
    gport = eb.read_json(output / "gport/gport.json")
    gport_check = {"pass": bool(gport["pass"]), "vendor_hashes_match": gport["vendor_hashes_match"],
                   "sha256": eb.sha256_of(output / "gport/gport.json")}
    readiness = readiness_check()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    tests_record = {"command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {TESTS}", "exit_code": tests.returncode,
                    "tail": tests.stdout.strip().splitlines()[-1:] if tests.stdout else []}
    throughput = {name: {k: eb.read_json(selection_dir(output, name) / "report.json")[k]
                         for k in ("seconds_per_step", "steps", "gpu_seconds")} for name in CONFIGS}
    rate = throughput[selected]["seconds_per_step"]
    training = [(f"{ARM}--seed{seed}", spec["steps"] * rate) for seed, spec, _ in remaining_items(
        output, selected, len(data.windows))]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values())
    fidelity_seconds = 13 * 3600.0  # the Tier-E run (~12.3 h measured rate) + the stopped run 1
    readout_seconds = 0.5 * 3600.0
    score_seconds = 1.0 * 3600.0
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds + fidelity_seconds
    per_seed = arm_spec(selected, len(data.windows))["steps"] * rate
    wall = max(loads + [fidelity_seconds]) + readout_seconds + score_seconds
    projection = {"selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "fidelity_gpu_h_estimate": fidelity_seconds / 3600, "readouts_gpu_h_estimate": readout_seconds / 3600,
                  "score_gpu_h_estimate": score_seconds / 3600, "total_gpu_h": total / 3600,
                  "child_cap_gpu_h": CHILD_CAP_GPU_HOURS, "per_seed_gpu_h": per_seed / 3600,
                  "seed_cap_gpu_h": SEED_CAP_GPU_HOURS, "wall_h_on_4_gpus": wall / 3600, "wall_cap_h": WALL_CAP_HOURS,
                  "training_queues": queues, "training_queue_loads_h": [x / 3600 for x in loads]}
    projection["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed / 3600 <= SEED_CAP_GPU_HOURS
                                     and wall / 3600 <= WALL_CAP_HOURS)
    record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
              "readiness_c1_dino": readiness, "tests": tests_record, "throughput": throughput,
              "selection": selection_record, "selected": selected, "projection": projection, "device": device,
              "pass": bool(gport_check["pass"] and tests.returncode == 0 and projection["within_caps"]),
              "written_at": eb.utc_now()}
    seconds = time.monotonic() - began
    eb.mark(output, "smoke", seconds, seconds if device != "cpu" else 0)
    eb.write_json(output / "smoke.json", record)
    log(f"smoke: pass {record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h, wall "
        f"{projection['wall_h_on_4_gpus']:.1f} h; selected {selected}")
    return 0 if record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/dino_wm/__init__.py", "world_model/baselines/dino_wm/novphy.py",
             "scripts/b127_dino_wm_port_reference.py", "scripts/b127_dino_wm_fidelity.py",
             "scripts/b127_dino_wm_features.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/dino_wm/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "gport/reference.pt", output / "fidelity/band.json",
              output / "smoke.json", output / "selection.json", READINESS, output / "readiness/r1.pt",
              output / "features/MANIFEST.json"]  # gfid.json is read live (it lands after the freeze; G_fid)
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"c1_{p.relative_to(ROOT)}", "artifact": rel(p), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file() and "__pycache__" not in str(p)]


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        load_plan(output)
        log("plan.json is frozen; pins re-verified")
        return 0
    smoke_record = eb.read_json(output / "smoke.json")
    if not smoke_record["pass"]:
        raise SystemExit("smoke did not pass; no freeze (P-cap / fix first)")
    selected = smoke_record["selected"]
    windows = smoke_record["accounting"]["windows"]
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {ARM: {**ARMS[ARM], "spec": arm_spec(selected, windows), "seeds": list(eb.SEEDS)}},
        "dropped": {DINO: {"typed_reason": DINO_STOP, "program_set": True, "readiness": readiness_check(),
                           "decision": DECISIONS["c1_dino_stop"]}},
        "reused": {ARM: {"seed": eb.SELECTION_SEED, "run": rel(selection_dir(output, selected))}},
        "queues": {"train": {str(k): v for k, v in smoke_record["projection"]["training_queues"].items()},
                   "readouts": {"0": [f"{ARM}--seed{s}" for s in eb.SEEDS]}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD,
        "arm_guards": ["G_rank", "G_action", "G_nonfinite", "G_trained"], "extra_guards": ["G_port", "G_fid"],
        "not_applicable": {"G_collapse": "C1-car predicts E99 carriers directly (no learned latent; decision on #130)",
                           "G_readout": "no readout: the cost is computed on the decoded carrier",
                           "native_head": "DINO-WM has no success head (planning by goal matching); no U + head"},
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES),
                      "within_family": {"C1-dino - C1-car": "not computable: C1-dino not run (typed stop)"},
                      "wrong_anchor": [ARM]},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "C1-car's configuration was selected on the policy split among three pre-declared configurations; U had "
            "no such tuning (#127).",
            "Training windows start at frame-skip-5 phase 0 only (native DINO-WM slices every frame offset).",
            "The released config's predictor LR (5e-4) differs from the paper's Table 12 (5e-5); both are candidates.",
            "C1-dino stopped at the readiness smoke (R1 policy ceiling 0.5948 < 0.65); it vetoes the program claim.",
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


# ---------------------------------------------------------------------------------------------------------------
# --train
# ---------------------------------------------------------------------------------------------------------------

def train(output, queue, device=None):
    plan = load_plan(output)
    items = []
    for name in plan["queues"]["train"].get(str(queue), []):
        seed = int(name.split("--seed")[1])
        items.append((plan["arms"][ARM]["spec"], seed, model_dir(output, seed, plan["selected"])))
    log(f"train queue {queue}: {[i[1] for i in items]}")
    return run_queue(output, items, device)


def training_report_of(output, plan):
    """ledger phase -> its training report (each arm-seed counted once; #123 A3 rule)."""
    def report_of(phase):
        if not phase.startswith("train-"):
            return None
        name, seed = phase[len("train-"):].rsplit("-seed", 1)
        if name in CONFIGS:
            if plan and name == plan["selected"]:  # main seeds of the selected configuration (and its reused run)
                return model_dir(output, int(seed), plan["selected"]) / "report.json"
            return selection_dir(output, name, int(seed)) / "report.json"
        return None
    return report_of


# ---------------------------------------------------------------------------------------------------------------
# --readouts (fit / policy data only) and --freeze-readouts
# ---------------------------------------------------------------------------------------------------------------

def readouts_arm(output, plan, seed, device=None):
    out_dir = readouts_dir(output, seed)
    if (out_dir / "readouts.json").is_file():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_of(device)
    costs = objective_costs()
    began = time.monotonic()
    path = model_dir(output, seed, plan["selected"]) / "model.pt"
    result = {"arm": ARM, "seed": seed, "plan_identity": IDENTITY, "model": rel(path),
              "model_sha256": eb.sha256_of(path)}
    with harness.deterministic_scoring() as policy:
        result["scoring_policy"] = policy
        wm, payload = load_trained(path, device)
        if wm is None:
            result.update(typed_failure=payload["typed_failure"]["typed_failure"])
        else:
            members, z0, ends = fit_anchor_endpoints(wm)
            matrix = tie_free_matrix(ends, z0, costs)
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=float(eb.action_sensitivity(ends)),
                          fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          policy_auc=eb.policy_grid_auc(policy_endpoint_rows(wm, costs), eb.PRIMARY_COST),
                          trainable_parameters=dn.trainable_parameters(wm.model))
            items, actions = grid_actions()
            result["macs_per_candidate"] = eb.macs_per_candidate(
                wm.endpoints, z0[0][None].expand(len(items), -1).contiguous(), actions, candidates=len(items))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{ARM}-seed{seed}", seconds, seconds if device != "cpu" else 0)
    eb.write_json(out_dir / "readouts.json", result)
    log(f"readouts {ARM} seed {seed}: {json.dumps({k: result.get(k) for k in ('policy_auc', 'G_action')})}")
    return 0


def readouts(output, queue, device=None):
    plan = load_plan(output)
    for name in plan["queues"]["readouts"].get(str(queue), []):
        readouts_arm(output, plan, int(name.split("--seed")[1]), device)
    return 0


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
    for seed in eb.SEEDS:
        rdir = readouts_dir(output, seed)
        if not (rdir / "readouts.json").is_file():
            raise SystemExit(f"readouts of {ARM} seed {seed} missing")
        mpath = model_dir(output, seed, plan["selected"]) / "model.pt"
        entries[f"{ARM}--seed{seed}"] = {"readouts": rel(rdir / "readouts.json"),
                                         "readouts_sha256": eb.sha256_of(rdir / "readouts.json"),
                                         "model": rel(mpath), "model_sha256": eb.sha256_of(mpath)}
    entries[f"{DINO}--readiness"] = {"readouts": rel(READINESS),
                                     "readouts_sha256": eb.sha256_of(READINESS)}
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
        for kind in ("model", "readouts"):
            if kind in entry and eb.sha256_of(ROOT / entry[kind]) != entry[f"{kind}_sha256"]:
                raise ValueError(f"{name}: frozen {kind} changed")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# --score (once; a process that did not train; GPU 0; deterministic)
# ---------------------------------------------------------------------------------------------------------------

def cell_rows(wm, items, z0, costs):
    """EndpointCosts rows of one cell / level: every candidate's predicted frame-225 carrier from the anchor z0
    (the primary scoring path; --score-novelty scores a novelty level with exactly this)."""
    actions = torch.stack([eb.launch5(it["action"]) for it in items])
    z0 = z0.float()
    return eb.endpoint_rows(items, wm.endpoints(z0[None].expand(len(items), -1), actions), z0, costs)


@torch.no_grad()
def score_arm(output, plan, seed, anchors, cells, device=None):
    """One C1-car record: rows of every mixed cell of every inventory, wrong-anchor rows (480 fit anchors, grid),
    position error (descriptive), wall time per decision."""
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": ARM, "seed": seed}
    wm, payload = load_trained(model_dir(output, seed, plan["selected"]) / "model.pt", device)
    if wm is None:
        return {**base, "typed_failure": payload["typed_failure"]["typed_failure"]}
    carriers = anchors["carriers"]
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {}
        for member, entry in sorted(cells[inventory]["mixed"].items()):
            inventories[inventory][member] = {"rows": cell_rows(wm, entry["inventory"], carriers[member], costs)}
    record = {**base, "typed_failure": None, "inventories": inventories}
    fit = eb.fit_anchors()["carriers"]
    items, actions = grid_actions()
    record["wrong_anchor"] = {m: {"rows": eb.endpoint_rows(items, wm.endpoints(
        fit[m].float()[None].expand(len(items), -1), actions), fit[m].float(), costs)} for m in sorted(fit)}
    cache = eb.position_cache()
    branches = eb.position_branches(cells)
    z0 = torch.stack([carriers[m].float() for m, _ in branches])
    acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
    rolled = wm.rollout(z0, acts)
    positions = {}
    for b, (_, item) in enumerate(branches):
        e = cache["entries"][item["key"]]
        positions[item["key"]] = {str(t): {"pred": eb.carrier_position_mse(rolled[b, t // dn.FRAMESKIP],
                                                                           e["presence"][j], e["centers"][j]),
                                           "floor": None} for j, t in enumerate(eb.POSITION_TIMES)}
    record["positions"] = positions
    members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(
        lambda m: wm.endpoints(carriers[m].float()[None].expand(len(items), -1), actions), members)
    return record


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    readouts_frozen_at = eb.read_json(output / "readouts-manifest.json")["frozen_at"]
    with harness.deterministic_scoring() as policy:
        anchors_path = eb.evaluation_anchors_path(output)
        anchors = (torch.load(anchors_path, map_location="cpu", weights_only=True) if anchors_path.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        for seed in eb.SEEDS:
            path = record_path(output, ARM, seed)
            if path.is_file():
                continue
            began = time.monotonic()
            record = score_arm(output, plan, seed, anchors, cells, device)
            seconds = time.monotonic() - began
            record.update(scoring_policy=policy, scored_at=eb.utc_now(), readouts_frozen_at=readouts_frozen_at,
                          anchors_read_at=anchors["read_at"], score_seconds=seconds)
            eb.write_json(path, record)
            eb.mark(output, f"score-{ARM}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(f"scored {ARM} seed {seed} ({seconds:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --score-novelty (P-novelty; once, after --score; DESCRIPTIVE, never in a disposition)
# ---------------------------------------------------------------------------------------------------------------

def novelty_record_path(output, arm, seed):
    return Path(output) / "novelty" / "records" / f"{arm}--seed{seed}.json"


def score_novelty(output, device=None):
    """Score C1-car (if its published guards pass) on the 16 grid launches of every mixed novelty level from that
    level's E99 anchor carrier, exactly as --score scores a Gate-B cell (cell_rows). C1-dino is a typed stop and is
    listed as not scored. Refuses before the model freeze and the primary records; never re-scores a primary
    record; resumable per record."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    missing = [s for s in eb.SEEDS if not record_path(output, ARM, s).is_file()]
    if missing:
        raise SystemExit(f"primary records of {ARM} missing for seeds {missing}: run --score first")
    records = load_records(output)
    guards = car_guards(output, records)
    status = {"schema": "issue_130_novelty_status_v1", "plan_identity": IDENTITY, "decision": DECISIONS["novelty"],
              "scored": [], "not_scored": {DINO: f"typed stop {DINO_STOP} (never trained)"}}
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        cells = eb.novelty_cells()
        path = eb.novelty_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_novelty_anchors(output, policy, cells))
        eb.novelty_reference_records(output, policy, cells, anchors["carriers"])
        if not guards["pass"]:
            status["not_scored"][ARM] = f"guards fail: {guards['failed']}"
        else:
            costs = objective_costs()
            for seed in eb.SEEDS:
                out = novelty_record_path(output, ARM, seed)
                if out.is_file():
                    continue
                mpath = model_dir(output, seed, plan["selected"]) / "model.pt"
                wm, _ = load_trained(mpath, device)
                grid = {level: {"rows": cell_rows(wm, cells["levels"][level]["inventory"], anchors["carriers"][level],
                                                  costs)} for level in cells["mixed"]}
                eb.write_json(out, {"schema": SCHEMA_NOVELTY_RECORD, "plan_identity": IDENTITY, "arm": ARM,
                                    "seed": seed, "model_sha256": eb.sha256_of(mpath), "typed_failure": None,
                                    "inventories": {"grid": grid}, "scoring_policy": policy,
                                    "anchors_read_at": anchors["read_at"], "scored_at": eb.utc_now()})
                log(f"novelty: scored {ARM} seed {seed} ({len(grid)} mixed levels)")
            status["scored"].append(ARM)
    status["guards"] = guards
    eb.write_json(output / "novelty" / "status.json", status)
    seconds = time.monotonic() - began
    eb.mark(output, "score-novelty", seconds, seconds if device_of(device) != "cpu" else 0)
    return 0


def novelty_section(output, records):
    """The DESCRIPTIVE P-novelty block of the publication (None until --score-novelty ran)."""
    output = Path(output)
    status_path = output / "novelty" / "status.json"
    if not status_path.is_file():
        return None
    status = eb.read_json(status_path)
    cells = eb.novelty_cells()
    anchors = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)
    with harness.deterministic_scoring() as policy:
        refs = eb.novelty_reference_records(output, policy, cells, anchors["carriers"])  # read back, never rewritten
    views = dict(refs)
    if ARM in status["scored"]:
        views[ARM] = ({s: eb.read_json(novelty_record_path(output, ARM, s)) for s in eb.SEEDS}, eb.PRIMARY_COST)
    last_primary = max(r["scored_at"] for r in records.values() if "scored_at" in r)
    table = eb.novelty_table(views, cells)
    return {"table": table, "scored": status["scored"], "not_scored": status["not_scored"],
            "G_chronology": {"pass": bool(anchors["read_at"] >= last_primary), "novelty_anchors_read_at":
                             anchors["read_at"], "last_primary_scored_at": last_primary},
            "label": "DESCRIPTIVE (P-novelty): no tokens, never part of a disposition"}


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ["records/*.json", "novelty/records/*.json", "novelty/references/*.json"]


def load_records(output):
    return {seed: eb.read_json(record_path(output, ARM, seed)) for seed in eb.SEEDS}


def dino_records():
    """C1-dino's stand-in records: the typed stop on every seed (no rows; view_rows yields no estimate)."""
    return {seed: {"arm": DINO, "seed": seed, "typed_failure": DINO_STOP} for seed in eb.SEEDS}


def global_guards(output, plan, records):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    scored = [r for r in records.values() if "scored_at" in r]
    anchors_read = min(r["anchors_read_at"] for r in scored)
    first_scored = min(r["scored_at"] for r in scored)
    chronology = bool(plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read <= first_scored)
    return {
        "G_isolation": {"pass": True, "rule": "fit split trains, policy split selects and gives the C1-dino readiness "
                        "ceiling; evaluation anchors are read only by --score after --freeze-readouts (G_chronology)"},
        "G_chronology": {"pass": chronology, "plan_frozen_at": plan["frozen_at"],
                         "readouts_frozen_at": manifest["frozen_at"], "anchors_read_at": anchors_read,
                         "first_scored_at": first_scored},
        "G_scoring_policy": {"pass": all(r["scoring_policy"] == harness.SCORING_POLICY for r in scored)},
        "G_records": eb.records_guard(output, IDENTITY),
    }


def gfid_status(output):
    path = Path(output) / "fidelity/gfid.json"
    if not path.is_file():
        return {"pass": False, "reason": "fidelity/gfid.json missing (Tier E not run to completion)"}
    gfid = eb.read_json(path)
    return {"pass": bool(gfid.get("pass")), "band": gfid.get("band"), "decided_by_run": gfid.get("decided_by_run"),
            "success_rate": next((r["success_rate"] for r in gfid.get("runs", [])
                                  if r["run_dir"] == gfid.get("decided_by_run")), None),
            "sha256": eb.sha256_of(path)}


def car_guards(output, records):
    """C1-car's published guard result (the same computation as the publication's arm block): the arm guards
    (G_rank, G_action, G_nonfinite, G_trained) and the extra guards G_port, G_fid; returns the arm block's G_* map,
    its pass and the failing guards."""
    output = Path(output)
    per_seed = {}
    for seed in eb.SEEDS:
        rd = eb.read_json(readouts_dir(output, seed) / "readouts.json")
        per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": rd.get("G_action"), "G_collapse": None,
                          "policy_ceiling_auc": None,
                          "typed_failure": rd.get("typed_failure") or records[seed].get("typed_failure")}
    block = eb.arm_guard_block(per_seed, ["G_rank", "G_action", "G_nonfinite", "G_trained"],
                               nonfinite=eb.nonfinite_share(records.values(), eb.PRIMARY_COST))
    gport = eb.read_json(output / "gport/gport.json")
    gfid = gfid_status(output)
    extra = {"G_port": bool(gport["pass"]), "G_fid": bool(gfid["pass"])}
    merged = {**block, **extra}
    failed = [k for k, v in merged.items() if k.startswith("G_") and v is not True]
    return {"arm": block, "extra": extra, "gfid": gfid, "failed": failed, "pass": not failed}


def arm_compute(output, plan):
    seconds, macs, walls, params = {}, [], [], None
    records = load_records(output)
    for seed in eb.SEEDS:
        report = model_dir(output, seed, plan["selected"]) / "report.json"
        if report.is_file():
            seconds[str(seed)] = eb.read_json(report)["gpu_seconds"]
        rd = eb.read_json(readouts_dir(output, seed) / "readouts.json")
        if rd.get("macs_per_candidate") is not None:
            macs.append(rd["macs_per_candidate"])
            params = rd["trainable_parameters"]
        if records[seed].get("wall_per_decision") is not None:
            walls.append(records[seed]["wall_per_decision"])
    return {"trainable_parameters": params, "training_gpu_seconds": seconds,
            "macs_per_candidate": float(np.mean(macs)) if macs else None,
            "wall_per_decision": float(np.median(walls)) if walls else None}


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = eb.evaluation_cells()
    refs = eb.references()
    guards_global = global_guards(output, plan, records)
    global_pass = all(g["pass"] is True for g in guards_global.values())
    guards_car = car_guards(output, records)
    gfid, extra = guards_car["gfid"], guards_car["extra"]
    views = {ARM: (records, eb.PRIMARY_COST), DINO: (dino_records(), eb.PRIMARY_COST)}
    readiness = readiness_check()
    dino_guards = {"G_readout": bool(readiness["pass"]), "readiness": readiness, "pass": False}
    blocks, tables = {}, {}
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", ARM])}
        if inventory == GRID:
            blocks[ARM] = eb.arm_block(rows, ARM, records, eb.PRIMARY_COST, guards_car["arm"], global_pass,
                                       arm_compute(output, plan), ARMS[ARM]["role"], False, extra_guards=extra)
            blocks[DINO] = eb.arm_block(rows, DINO, dino_records(), eb.PRIMARY_COST, dino_guards, global_pass,
                                        {"trainable_parameters": None, "training_gpu_seconds": {},
                                         "macs_per_candidate": None, "wall_per_decision": None},
                                        "primary", True, extra_guards=dict(extra),
                                        dropped=DINO_STOP)
            within = {"C1-dino - C1-car": {**eb.within_family(rows, DINO, ARM, False),
                                           "note": "not computable: C1-dino was not run (typed stop "
                                                   f"{DINO_STOP}); every token is the stop token"}}
        else:
            tables[inventory]["contrasts"] = {f"{ARM} - U": eb.contrast(
                rows, ARM, "U", global_pass and blocks[ARM]["guards"]["pass"])}
    wrong = {ARM: eb.wrong_anchor_table(cells, records, ARM, eb.PRIMARY_COST,
                                        global_pass and blocks[ARM]["guards"]["pass"])}
    curves = eb.reference_position_curves()
    positions = eb.position_table({ARM: records, "U": curves["U"], "C*": curves["C*"]}, cells)
    gfid_block = {ARM: gfid, DINO: gfid}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, within, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": positions, "selection": plan["selected"],
        "dropped": plan["dropped"], "novelty": novelty_section(output, records),
        "readouts": {f"{ARM}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, s) / "readouts.json").items()
                                         if k in ("policy_auc", "G_action", "G_rank", "typed_failure")}
                     for s in eb.SEEDS}})
    compute = {"schema": "issue_130_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {ARM: blocks[ARM]["compute"]}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} C1 DINO-WM: findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; C1-car configuration {summary['extra']['selection']}; "
             f"G-fid (DINO-WM Tier E) pass {summary['gfid'][ARM]['pass']}.", "",
             f"C1-dino: dropped before training ({DINO_STOP}; R1 policy ceiling "
             f"{summary['extra']['dropped'][DINO]['readiness']['ceiling_auc']:.4f} < 0.65); vetoes the program claim.",
             "", *eb.format_arm_rows(summary["arms"]), "", "## Within family", ""]
    for label, block in summary["within_family"].items():
        lines.append(f"- {label}: {eb.rc.fmt_block(block['estimate'])} ({block['token']} / {block['reverse_token']}); "
                     f"{block.get('note', '')}")
    lines += ["", "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    novelty = summary["extra"].get("novelty")
    if novelty is not None:
        lines += ["", "## P-novelty (#110 novelty levels; DESCRIPTIVE: no tokens, never part of a disposition)", "",
                  f"Scored: {', '.join(novelty['scored']) or 'none'}; not scored: "
                  + "; ".join(f"{a} ({why})" for a, why in novelty["not_scored"].items())
                  + f"; G-chronology (novelty anchors read after the primary records) {novelty['G_chronology']['pass']}.",
                  "", *eb.format_novelty_rows(novelty["table"])]
    lines += ["", "Intervals are descriptive (member-clustered bootstrap); tokens follow the #127 delta rule.", ""]
    return "\n".join(lines)


def publish(output):
    output = Path(output)
    eb.write_json(output / "records-manifest.json", eb.records_manifest(output, IDENTITY, RECORD_PATTERNS))
    summary, compute = publication(output)
    eb.write_json(output / "compute.json", compute)
    eb.write_json(output / "summary.json", summary)
    (output / "findings.md").write_text(findings_text(summary))
    log(f"published; G-records {summary['global_guards']['G_records']['pass']}")
    return 0


def validate(output, device=None):
    """Hash-check the records against the manifest, re-score one record (exact row equality), write the receipt,
    then re-derive the publication twice and require byte-identical files."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    manifest = eb.read_json(output / "records-manifest.json")
    error, spot = None, None
    if manifest != eb.records_manifest(output, IDENTITY, manifest["patterns"]):
        error = "records differ from records-manifest.json"
    else:
        seed = eb.SELECTION_SEED
        stored = eb.read_json(record_path(output, ARM, seed))
        anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
        with harness.deterministic_scoring():
            fresh = score_arm(output, plan, seed, anchors, eb.evaluation_cells(), device)
        fresh = json.loads(eb.json_text(fresh))
        same = all(fresh["inventories"][inv] == stored["inventories"][inv] for inv in eb.INVENTORIES)
        spot = {"record": f"{ARM}--seed{seed}", "pass": bool(same), "rule": "exact row equality, every inventory"}
    eb.write_receipt(output, IDENTITY, spot, error)
    publish(output)
    first = {name: (output / name).read_bytes() for name in ("summary.json", "compute.json", "findings.md")}
    publish(output)
    second = {name: (output / name).read_bytes() for name in first}
    stable = first == second
    receipt = eb.read_json(output / eb.VALIDATION_RECEIPT)
    ok = bool(receipt["pass"] and stable)
    log(f"validate: receipt {receipt['pass']}, publication stable {stable}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("selection", "selection-readouts", "smoke", "prepare", "train", "readouts", "freeze-readouts",
                 "score", "score-novelty", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--workers", type=int, default=0, help="accepted for the contract; data live on the device")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.selection:
        return selection(output, args.gpu, args.device)
    if args.selection_readouts:
        return selection_readouts(output, args.device)
    if args.smoke:
        return smoke(output, args.device)
    if args.prepare:
        return prepare(output)
    if args.train:
        return train(output, args.gpu, args.device)
    if args.readouts:
        return readouts(output, args.gpu, args.device)
    if args.freeze_readouts:
        return freeze_readouts(output)
    if args.score:
        return score(output, args.device)
    if args.score_novelty:
        return score_novelty(output, args.device)
    if args.publish:
        return publish(output)
    return validate(output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
