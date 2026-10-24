"""Issue #131 (#127 child D1): THICK on E99 carriers against U on the sealed NovPhy launch-ranking grid.

Arms (Decisions P-other on #131; the #127 frozen arm matrix):
- **D1** (primary, program set): native THICK (the G-port-verified PyTorch port ``world_model/baselines/thick/port.py``
  in the ``thick_pinpad`` configuration with THICK's MLP observation pathway on the 236-d carrier, adapter
  ``world_model/baselines/thick/novphy.py``; 21 852 337 parameters), offline on the fit split: 46-frame episodes
  (frames 0, 5, ..., 225), the launch at t0, 200 000 updates x 16 episodes. Score = tie-free ``EndpointCosts`` of the
  low level's decoded frame-225 carrier (deterministic rollout: latent modes, zero GateL0RD noise). Configuration
  (``ctxt_sparsity`` in {1, 10, 25}) selected on the policy split (seed 20260908, full native budget); the selected
  run is reused as main seed 20260908.
- **D1-m** (secondary, matched): U's B1 recipe (9 000 updates x 64 episodes) and 1.84 M +- 5 % trainable parameters:
  every hidden width of the native model times ``MATCHED_WIDTH_FACTOR`` (discrete latents 32 x 32 and the 16-d
  context kept), every other hyperparameter the selected configuration's.
- Exploratory views of D1 (never in a claim): the context decoder's endpoint carrier (``coarse_tie_free``), the high
  level's jump prediction from the post-launch state (``hl_tie_free``, plus its predicted elapsed steps), and the
  commit points (rollout steps whose GateL0RD context gates open: THICK's learned adaptive Delta).
- THICK has no native success head on NovPhy (there is no reward signal: the reward heads are trained on zeros), so
  there is no native-head score and no U + head control.

Readiness gate (#131): selection refuses to train unless ``gport/gport.json`` passes and the G-fid timing run
(``fidelity/gfid-timing.json``) measured the G-fid feasible; ``--score`` refuses unless ``fidelity/gfid.json`` passes.

Modes (round3/child-runner-contract.md): --selection, --selection-readouts, --smoke, --prepare, --train,
--readouts, --freeze-readouts, --score, --score-novelty (P-novelty, descriptive), --publish, --validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_131.json); tests use CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
import json
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
from world_model.baselines.thick import novphy as tn  # noqa: E402
from world_model.baselines.thick import port  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 131
IDENTITY = "issue-131-thick-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_131_thick.py"
TESTS = "tests/test_issue_131_thick_runner.py"
SCHEMA_MODEL = "issue_131_model_v1"
SCHEMA_PLAN = "issue_131_plan_v1"
SCHEMA_RECORD = "issue_131_record_v1"
SCHEMA_SMOKE = "issue_131_smoke_v1"
SCHEMA_SELECTION = "issue_131_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_131_readouts_manifest_v1"
DECISIONS = {
    "route": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382162858",
    "license": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382307800",
    "band": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382475412",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382505350",
    "gfid_route": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756572",
    "cap": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756903",
    "runner": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383193006",
    "precision": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383381590",
    "novelty": "https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386831318",
}

CONFIGS = dict(tn.CONFIGURATIONS)  # sparsity-1 / -10 / -25 (pre-declared, Decision P-other adapter, item 6)
SELECTION_QUEUES = {0: ["sparsity-1"], 1: ["sparsity-10"], 2: ["sparsity-25"]}  # one config per scheduler job
NATIVE_RECIPE = {"steps": 200_000, "batch": tn.BATCH}  # upstream steps 1e6 / train_every 5; 16 episodes
MATCHED_RECIPE = {"steps": 9000, "batch": 64}  # U's B1 recipe
U_PARAMETERS = 1_837_690  # eb.reference_compute()["unit"] trainable parameters (#127 matched target, +- 5 %)
MATCHED_TOLERANCE = 0.05
MATCHED_WIDTH_FACTOR = 0.166  # the factor of [0.150, 0.180] (step 0.001) whose model is closest to U_PARAMETERS
ARMS = {"D1": {"size": "native", "role": "primary", "program_set": True},
        "D1-m": {"size": "matched", "role": "secondary (matched budget and parameters)", "program_set": False}}
EXPLORATORY = {"D1-coarse": "coarse_tie_free", "D1-hl": "hl_tie_free"}  # views of D1's records (no claims)
COLLAPSE_INDEX = list(range(0, len(tn.POSITIONS), 3))  # frames 0, 15, ..., 225 (the #123 probe stride)
CHECKPOINT_EVERY = 2000
LOG_EVERY = 100
CHILD_CAP_GPU_HOURS = 146.0  # the upper end of the #127 D1 child share
SEED_CAP_GPU_HOURS = 24.0  # #127: native cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
GRID = "grid"
ARM_CHECKS = ["G_rank", "G_action", "G_collapse", "G_nonfinite", "G_trained"]


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------
# configurations and paths
# ---------------------------------------------------------------------------------------------------------------

def matched_overrides(factor=MATCHED_WIDTH_FACTOR):
    """Every hidden width of the native model times ``factor`` (rounded): RSSM hidden / deter, the encoder and both
    decoders' MLP layers, the reward heads, THICKNet's width (body, high-level action prior / posterior) and its
    readouts. The discrete latents (32 x 32), the 16-d context and every depth stay native; ``ctxt_rnn_out_size``
    is unused by THICK's headless GateL0RD."""
    r = lambda x: max(1, int(round(x * factor)))  # noqa: E731
    native = tn.build_config()
    return {"rssm.hidden": r(native.rssm.hidden), "rssm.deter": r(native.rssm.deter),
            "encoder.mlp_layers": tuple(r(u) for u in native.encoder.mlp_layers),
            "decoder.mlp_layers": tuple(r(u) for u in native.decoder.mlp_layers),
            "reward_head.units": r(native.reward_head.units), "thick_feat": r(native.thick_feat),
            "thick_readouts.units": r(native.thick_readouts.units)}


def config_spec(name):
    return {"name": name, "arm": "D1", "size": "native", "configuration": name, "overrides": {},
            "steps": NATIVE_RECIPE["steps"], "batch": NATIVE_RECIPE["batch"]}


def arm_spec(arm, selected):
    """The training spec of an arm at the selected configuration."""
    base = config_spec(selected)
    if arm == "D1":
        return base
    return {**base, "arm": arm, "name": f"{arm}@{selected}", "size": ARMS[arm]["size"],
            "overrides": matched_overrides(), "steps": MATCHED_RECIPE["steps"], "batch": MATCHED_RECIPE["batch"]}


def build_model(spec):
    return tn.build_model(tn.build_config(spec["configuration"], spec["overrides"]))


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, arm, seed, selected=None):
    """D1 seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if arm == "D1" and seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("model_dir(D1, 20260908) needs the selected configuration")
        return selection_dir(output, selected)
    return Path(output) / "models" / arm / f"seed-{seed}"


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / arm / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


# ---------------------------------------------------------------------------------------------------------------
# fit episodes
# ---------------------------------------------------------------------------------------------------------------

def fit_store(shots, chunk=1024):
    """Every fit shot's episode: carriers [N, 46, 236] float32 (frames 0, 5, ..., 225; positions past the end
    repeat the final carrier) and launch5 [N, 5], in the frozen shot order (CPU)."""
    carriers = []
    for start in range(0, len(shots.keys), chunk):
        index = range(start, min(len(shots.keys), start + chunk))
        pairs = [(s, p) for s in index for p in tn.episode_positions(shots.ends[s])]
        rows = shots.carrier_rows([s for s, _ in pairs], [p for _, p in pairs])
        carriers.append(torch.as_tensor(np.asarray(rows), dtype=torch.float32).reshape(
            len(index), len(tn.POSITIONS), tn.CARRIER_DIM))
    return {"carriers": torch.cat(carriers), "launch5": torch.as_tensor(shots.launch5, dtype=torch.float32)}


def episode_batch(data, rows, device):
    rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    batch = tn.episode_batch(data["carriers"][rows], data["launch5"][rows])
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss / gradient norm -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

# Training precision of every D1 GPU run (Decision P-route, G-fid on the port, #131 issuecomment-6382756572; it
# supersedes the adapter decision's "TF32 off"): float32 parameters and activations with TF32 matmul / convolution on
# the GPU, as the official TF 2.4.2 stack selects on the RTX 5090. Training only: every readout, selection-readout,
# scoring and validation path runs under harness.deterministic_scoring() (TF32 off). On the CPU the flags are inert.
TRAINING_PRECISION = {"dtype": "float32", "matmul_allow_tf32": True, "cudnn_allow_tf32": True,
                      "source": DECISIONS["gfid_route"]}


def training_flags():
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def _set_training_flags(flags):
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]


def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one model (resumable from out_dir/resume.pt) under TRAINING_PRECISION (the prior flags are restored).
    Batch b = tn.sample_shots(eb.batch_rng(seed, b)); every latent / GateL0RD draw comes from one torch generator
    seeded with ``seed`` whose state is checkpointed."""
    saved = training_flags()
    _set_training_flags(TRAINING_PRECISION)
    try:
        return _train_run(out_dir, spec, seed, data, device, max_steps)
    finally:
        _set_training_flags(saved)


def _train_run(out_dir, spec, seed, data, device, max_steps):
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    precision = {**TRAINING_PRECISION, "device": device, "flags": training_flags()}
    steps, n = spec["steps"], len(data["launch5"])
    torch.manual_seed(seed)
    model = build_model(spec).to(device).train()
    model_opt, thick_opt = model.make_optimizers()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = port.TorchNoise(generator)
    start, elapsed = 0, 0.0
    resume = out_dir / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model_opt.load_state_dict(state["model_opt"])
        thick_opt.load_state_dict(state["thick_opt"])
        generator.set_state(state["generator"].cpu())
        start, elapsed = state["step"], state["elapsed"]
        log(f"{spec['name']} seed {seed}: resumed at step {start}")
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            batch = episode_batch(data, tn.sample_shots(eb.batch_rng(seed, step), n, spec["batch"]), device)
            try:
                out = model.train_step(batch, model_opt, thick_opt, noise=noise)
            except FloatingPointError as error:
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": str(error)}
                break
            if step % LOG_EVERY == 0 or step == steps - 1:
                entry = {"step": step, "seconds": elapsed + time.monotonic() - began,
                         "model_loss": float(out["model_loss"]), "thick_loss": float(out["thick_loss"]),
                         "model_grad_norm": float(out["model_grad_norm"]),
                         "thick_grad_norm": float(out["thick_grad_norm"]),
                         **{k: float(v) for k, v in out["losses"].items()},
                         **{f"thick_{k}" if not k.startswith("thick") else k: float(v)
                            for k, v in out["thick_losses"].items()}}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(eb.STOP_REQUESTED) and step + 1 < steps
            if stop or ((step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < steps):
                eb.atomic_torch(resume, {"model": model.state_dict(), "model_opt": model_opt.state_dict(),
                                         "thick_opt": thick_opt.state_dict(), "generator": generator.get_state(),
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
               "parameters": tn.parameter_count(model), "precision": precision}
    eb.atomic_torch(final, payload)
    report = {"spec": spec, "seed": seed, "steps": steps, "complete": complete, "typed_failure": failure,
              "wall_seconds": seconds, "gpu_seconds": seconds if device.startswith("cuda") else 0.0,
              "device": device, "seconds_per_step": seconds / max(1, steps),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": trace[-1] if trace else None, "precision": precision, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if resume.is_file() and complete:
        resume.unlink()
    log(f"{spec['name']} seed {seed}: {'complete' if complete else failure['typed_failure']} after {seconds:.0f}s")
    return report


def load_trained(path, device=None):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL:
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    model = build_model(payload["spec"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device_of(device)).eval(), payload


def run_queue(output, items, device=None):
    """Train (spec, seed, out_dir) items in order; a stop signal pauses the queue (exit 75)."""
    eb.install_stop_handlers()
    data = None
    for spec, seed, out_dir in items:
        if (Path(out_dir) / "model.pt").is_file():
            continue
        data = data or fit_store(eb.FitShots())
        began = time.monotonic()
        try:
            report = train_run(out_dir, spec, seed, data, device)
        except eb.TrainingPaused as paused:
            seconds = time.monotonic() - began
            eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(str(paused))
            return eb.EXIT_PAUSED
        seconds = time.monotonic() - began
        eb.mark(output, f"train-{spec['name']}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0,
                report=str(Path(out_dir, "report.json").relative_to(ROOT)) if Path(out_dir).is_relative_to(ROOT)
                else str(Path(out_dir, "report.json")), complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# prediction (deterministic rollouts), costs, policy AUC
# ---------------------------------------------------------------------------------------------------------------

def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


@torch.no_grad()
def predict(model, z0, actions, carriers=False, jump=False, batch=4096):
    """Deterministic rollouts (tn.rollout) of anchors z0 [B, 236] under launches [B, 5], in batches; CPU tensors:
    ``endpoint`` [B, 236], ``coarse`` [B, 236], ``update_counts`` [B], ``commit_steps`` (lists), optionally
    ``carriers`` [B, 46, 236] and the high-level jump (``hl_carrier`` [B, 236], ``hl_time`` [B])."""
    parts = {"endpoint": [], "coarse": [], "update_counts": [], "commit_steps": [], "carriers": [], "hl_carrier": [],
             "hl_time": []}
    for i in range(0, len(z0), batch):
        out = tn.rollout(model, z0[i:i + batch], actions[i:i + batch])
        parts["endpoint"].append(out["endpoint"].float().cpu())
        parts["coarse"].append(out["coarse_endpoint"].float().cpu())
        parts["update_counts"].append(out["update_counts"].cpu())
        parts["commit_steps"] += out["commit_steps"]
        if carriers:
            parts["carriers"].append(out["carriers"].float().cpu())
        if jump:
            hl = tn.high_level_jump(model, z0[i:i + batch], actions[i:i + batch])
            parts["hl_carrier"].append(hl["carrier"].float().cpu())
            parts["hl_time"].append(hl["thick_time"].float().cpu())
    return {k: (v if k == "commit_steps" else torch.cat(v) if v else None) for k, v in parts.items()}


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


@torch.no_grad()
def fit_anchor_predictions(model):
    """Rollouts of the 16 grid launches from the 480 fit anchors (G-rank, G-action, exploratory statistics)."""
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)
    items, actions = grid_actions()
    z0 = torch.stack([anchors[m] for m in members])
    out = predict(model, z0.repeat_interleave(len(items), 0), actions.repeat(len(members), 1), jump=True)
    return members, z0, out


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


def candidate_rows(items, out, z0, costs):
    """Cost rows of one anchor's candidates: the primary tie-free cost of the fine endpoint plus the exploratory
    keys (coarse endpoint and high-level jump tie-free costs, commit counts, predicted high-level elapsed steps)."""
    primary = lambda ends: [r[eb.PRIMARY_COST] for r in eb.rc.cost_rows(items, ends, z0, costs)]  # noqa: E731
    extra = {"coarse_tie_free": primary(out["coarse"]), "commit_count": out["update_counts"].double().tolist()}
    if out["hl_carrier"] is not None:
        extra.update(hl_tie_free=primary(out["hl_carrier"]), hl_time=out["hl_time"].double().tolist())
    return eb.rc.cost_rows(items, out["endpoint"], z0, costs, extra)


@torch.no_grad()
def policy_endpoint_rows(model, costs):
    """{member: rows} of the policy grid mixed cells (primary + exploratory keys)."""
    anchors = eb.policy_anchors()["carriers"]
    out = {}
    for member, entry in sorted(eb.policy_cells(GRID).items()):
        items = entry["inventory"]
        actions = torch.stack([eb.launch5(item["action"]) for item in items])
        z0 = anchors[member]
        out[member] = candidate_rows(items, predict(model, z0[None].expand(len(items), -1), actions, jump=True),
                                     z0, costs)
    return out


@torch.no_grad()
def posterior_features(model, data, frames=COLLAPSE_INDEX, batch=512):
    """G-collapse latent (Decision P-other, runner): the low level's decoder input ``rssm.get_feat`` (one-hot
    stochastic mode 32 x 32 + deterministic + 16-d context) of the posterior after observing each fit episode
    deterministically (latent modes, zero GateL0RD noise), at frames 0, 15, ..., 225. Returns z [N * F, D] float64
    (CPU) and the episode index of every row."""
    model.eval()
    device = next(model.parameters()).device
    rssm, feats = model.rssm, []
    n = len(data["launch5"])
    for start in range(0, n, batch):
        d = model.preprocess(episode_batch(data, range(start, min(n, start + batch)), device))
        post, _ = rssm.observe(model.encoder(d), d["action"], d["is_first"], None, sample=False, rnn_sample=False,
                               noise=tn.ZeroNoise())
        feats.append(rssm.get_feat(post)[:, frames].double().cpu())
    z = torch.cat(feats)
    episodes = torch.arange(n).repeat_interleave(len(frames))
    return z.reshape(-1, z.shape[-1]), episodes


def commit_statistics(update_counts, commit_steps):
    """Exploratory: how often and where the context gates open along the 45-step rollouts."""
    counts = torch.as_tensor(update_counts).double()
    histogram = np.zeros(tn.ROLLOUT_STEPS, dtype=np.int64)
    for steps in commit_steps:
        for s in steps:
            histogram[s - 1] += 1
    return {"candidates": int(len(counts)), "mean_commits": float(counts.mean()) if len(counts) else None,
            "share_with_commit": float((counts > 0).double().mean()) if len(counts) else None,
            "commit_step_histogram": histogram.tolist()}


# ---------------------------------------------------------------------------------------------------------------
# readiness gate (#131)
# ---------------------------------------------------------------------------------------------------------------

def readiness(output):
    """G-port passes (world model and agent records) and the G-fid timing run measured the G-fid feasible."""
    output = Path(output)
    out = {}
    for name in ("gport/gport.json", "gport/agent/gport.json"):
        path = output / name
        out[name] = bool(path.is_file() and eb.read_json(path).get("pass") is True)
    timing = output / "fidelity/gfid-timing.json"
    out["fidelity/gfid-timing.json"] = bool(timing.is_file() and eb.read_json(timing).get("feasible") is True)
    out["pass"] = all(out.values())
    return out


def gfid_pass(output):
    path = Path(output) / "fidelity/gfid.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, queue, device=None):
    gate = readiness(output)
    if not gate["pass"]:
        log(f"selection refused: readiness gate {gate}")
        return 1
    names = SELECTION_QUEUES[queue]
    items = [(config_spec(name), eb.SELECTION_SEED, selection_dir(output, name)) for name in names]
    log(f"selection queue {queue}: {names}")
    return run_queue(output, items, device)


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
            model, payload = load_trained(path, device)
            if model is None:
                results[name] = {"auc": None, "parameters": payload["parameters"]}
                details[name] = {"typed_failure": payload["typed_failure"]}
                continue
            rows = policy_endpoint_rows(model, costs)
            results[name] = {"auc": eb.policy_grid_auc(rows, eb.PRIMARY_COST), "parameters": payload["parameters"]}
            details[name] = {"typed_failure": None, "model_sha256": eb.sha256_of(path), "cells": len(rows),
                             "exploratory_auc": {view: eb.policy_grid_auc(rows, key)
                                                 for view, key in EXPLORATORY.items()}}
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
    """Longest-processing-time assignment: items [(name, seconds)] -> {bin: [names]}, loads."""
    queues, loads = {b: [] for b in range(bins)}, [0.0] * bins
    for name, seconds in sorted(items, key=lambda t: (-t[1], t[0])):
        b = int(np.argmin(loads))
        queues[b].append(name)
        loads[b] += seconds
    return queues, loads


def measure_seconds_per_step(spec, data, device, steps=200):
    """GPU seconds per update of spec (a throwaway run, never saved)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        warm = train_run(tmp, spec, 0, data, device, max_steps=20)
        if not warm.get("partial"):
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        began = time.monotonic()
        report = train_run(tmp, spec, 0, data, device, max_steps=steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.monotonic() - began) / report.get("steps_done", steps)


def remaining_items(output, selected):
    """(arm, seed, spec, out_dir) of every training run the plan still needs."""
    out = []
    for arm in ARMS:
        for seed in eb.SEEDS:
            if arm == "D1" and seed == eb.SELECTION_SEED:
                continue
            out.append((arm, seed, arm_spec(arm, selected), model_dir(output, arm, seed)))
    return out


def fidelity_gpu_hours(output):
    """G-fid GPU-h for the child projection: the measured re-projection of the G-fid timing run when present, else
    the frozen G-fid plan's TF32 projection of its design."""
    output = Path(output)
    timing = output / "fidelity/gfid-timing.json"
    if timing.is_file():
        record = eb.read_json(timing)
        for key in ("projected_gpu_h", "projected_gpu_hours", "gpu_h"):
            if isinstance(record.get(key), (int, float)):
                return float(record[key]), f"fidelity/gfid-timing.json:{key}"
    plan = eb.read_json(output / "fidelity/plan.json")
    design = plan["proposal"]["design"]
    return float(plan["projections"]["tf32"]["gpu_hours_by_design"][design]), "fidelity/plan.json (tf32)"


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    launch = torch.as_tensor(shots.launch5)
    accounting = {"fit_shots": len(shots), "fit_levels": len(shots.levels), "fit_positive": int(shots.verdicts.sum()),
                  "fit_anchors": len(eb.fit_anchors()["carriers"]), "policy_grid_mixed_cells": len(eb.policy_cells(GRID)),
                  "episode_frames": len(tn.POSITIONS), "shots_ending_before_225": int((np.asarray(shots.ends) <
                                                                                       tn.END_FRAME).sum()),
                  "launch5_abs_max": float(launch.abs().max())}
    gport = eb.read_json(output / "gport/gport.json")
    gport_check = {"world_model_pass": gport.get("pass") is True,
                   "agent_pass": eb.read_json(output / "gport/agent/gport.json").get("pass") is True
                   if (output / "gport/agent/gport.json").is_file() else None,
                   "port_current": gport.get("port_sha256") == eb.sha256_of(ROOT / "world_model/baselines/thick/port.py")}
    gport_check["pass"] = bool(gport_check["world_model_pass"] and gport_check["port_current"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    tests_record = {"command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {TESTS}", "exit_code": tests.returncode,
                    "tail": tests.stdout.strip().splitlines()[-1:] if tests.stdout else []}
    throughput = {}
    for name in CONFIGS:
        report = eb.read_json(selection_dir(output, name) / "report.json")
        throughput[name] = {"seconds_per_step": report["seconds_per_step"], "steps": report["steps"],
                            "gpu_seconds": report["gpu_seconds"]}
    matched = arm_spec("D1-m", selected)
    throughput["D1-m"] = {"seconds_per_step": measure_seconds_per_step(matched, fit_store(shots), device),
                          "steps": matched["steps"]}
    remaining = remaining_items(output, selected)
    native_rate = throughput[selected]["seconds_per_step"]
    training = [(f"{arm}--seed{seed}", spec["steps"] * (native_rate if arm == "D1" else
                                                         throughput["D1-m"]["seconds_per_step"]))
                for arm, seed, spec, _ in remaining]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values() if "gpu_seconds" in t)
    readout_seconds = 3 * 3600.0  # G-rank / G-action / G-collapse / policy AUCs of 6 arm-seeds (estimate)
    score_seconds = 2 * 3600.0  # one scoring pass incl. wrong anchors and positions (estimate)
    fidelity_h, fidelity_source = fidelity_gpu_hours(output)
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds + fidelity_h * 3600
    per_seed_native = NATIVE_RECIPE["steps"] * native_rate
    wall = max(loads) + readout_seconds / 4 + score_seconds
    projection = {"fidelity_gpu_h": fidelity_h, "fidelity_source": fidelity_source,
                  "selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h_estimate": readout_seconds / 3600, "score_gpu_h_estimate": score_seconds / 3600,
                  "total_gpu_h": total / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "native_per_seed_gpu_h": per_seed_native / 3600, "seed_cap_gpu_h": SEED_CAP_GPU_HOURS,
                  "wall_h_on_4_gpus": wall / 3600, "wall_cap_h": WALL_CAP_HOURS,
                  "training_queues": queues, "training_queue_loads_h": [x / 3600 for x in loads]}
    projection["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed_native / 3600 <= SEED_CAP_GPU_HOURS
                                     and wall / 3600 <= WALL_CAP_HOURS)
    smoke_record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
                    "readiness": readiness(output), "tests": tests_record, "throughput": throughput,
                    "selection": selection_record, "selected": selected, "projection": projection, "device": device,
                    "pass": bool(gport_check["pass"] and tests.returncode == 0 and projection["within_caps"]
                                 and accounting["launch5_abs_max"] <= 1.0),
                    "written_at": eb.utc_now()}
    seconds = time.monotonic() - began
    eb.mark(output, "smoke", seconds, seconds if device != "cpu" else 0)
    eb.write_json(output / "smoke.json", smoke_record)
    log(f"smoke: pass {smoke_record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h, wall "
        f"{projection['wall_h_on_4_gpus']:.1f} h; selected {selected}")
    return 0 if smoke_record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/thick/__init__.py", "world_model/baselines/thick/port.py",
             "world_model/baselines/thick/novphy.py", "scripts/b127_thick_port_reference.py",
             "scripts/b127_thick_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/thick/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "gport/agent/gport.json", output / "fidelity/band.json",
              output / "fidelity/plan.json", output / "fidelity/gfid-timing.json", output / "smoke.json",
              output / "selection.json"]  # decisions.md is append-only: not pinned; gfid.json is read at publish
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"d1_{p.relative_to(ROOT)}", "artifact": str(p.relative_to(ROOT)), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file() and "__pycache__" not in str(p)]


def readout_items():
    return [(arm, seed) for arm in ARMS for seed in eb.SEEDS]


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
    queues = smoke_record["projection"]["training_queues"]
    readout_queues, _ = lpt([(f"{a}--seed{s}", 1.0 if a == "D1-m" else 2.0) for a, s in readout_items()])
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {arm: {**ARMS[arm], "spec": arm_spec(arm, selected), "seeds": list(eb.SEEDS)} for arm in ARMS},
        "training_precision": TRAINING_PRECISION, "scoring_policy": harness.SCORING_POLICY,
        "matched": {"width_factor": MATCHED_WIDTH_FACTOR, "overrides": matched_overrides(),
                    "target_parameters": U_PARAMETERS, "tolerance": MATCHED_TOLERANCE},
        "reused": {"D1": {"seed": eb.SELECTION_SEED, "run": str(selection_dir(output, selected).relative_to(ROOT))}},
        "readout": {"rollout": "tn.rollout: anchor observed at t0 (obs_step), 45 img_steps (launch at the first), "
                               "latent modes, zero GateL0RD noise; primary = tie-free EndpointCosts of the fine "
                               "decoder's mean carrier at frame 225",
                    "exploratory": EXPLORATORY, "commit_points": "rollout steps whose GateL0RD gates open",
                    "collapse_latent": "rssm.get_feat of the deterministic posterior on fit episodes at frames "
                                       "0, 15, ..., 225 (1024 one-hot stochastic + deter + 16 context)",
                    "action_sensitivity": "eb.action_sensitivity on the decoded frame-225 carriers, 480 fit "
                                          "anchors x 16 grid launches"},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES),
                      "within_family": ["D1 - J_dec", "D1 - D1-m", "D1-coarse - D1", "D1-hl - D1"],
                      "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "D1's configuration (ctxt_sparsity) was selected on the policy split among three pre-declared "
            "configurations; U had no such tuning (#127).",
            "THICK has no native success head on NovPhy (no reward signal; reward heads trained on zeros): no "
            "native-head score and no U + head control.",
            "The deterministic readout uses zero GateL0RD noise (upstream eval_ctxt_mean); upstream "
            "ctxt_always_sample would keep the noise on at evaluation.",
            "Training (selection, main seeds, D1-m, throughput) is float32 with TF32 matmul / convolution on the GPU, "
            "as the G-fid runs (Decision P-route, superseding the adapter decision's 'TF32 off'); every readout, "
            "selection-readout, scoring and validation path runs under deterministic_scoring() (TF32 off).",
            "D1-m scales every hidden width by one factor to U's parameter count; depths, latents and context native.",
            "Scoring waits for G-fid pass (#131 readiness gate); a G-fid failure drops D1 (fidelity_not_reproduced).",
            "P-novelty: after the primary scoring pass, every guard-passing arm (D1, D1-m; not the exploratory "
            "views) is scored on the #110 novelty levels from the E99 v1 anchor parse (--score-novelty); "
            "descriptive only, no tokens, never in a disposition.",
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
    names = plan["queues"]["train"].get(str(queue), [])
    items = []
    for name in names:
        arm, seed = name.split("--seed")
        seed = int(seed)
        items.append((plan["arms"][arm]["spec"], seed, model_dir(output, arm, seed, plan["selected"])))
    log(f"train queue {queue}: {names}")
    return run_queue(output, items, device)


def training_report_of(output, plan):
    """ledger phase -> its training report (each arm-seed counted once; #123 A3 rule)."""
    def report_of(phase):
        if not phase.startswith("train-"):
            return None
        name, seed = phase[len("train-"):].rsplit("-seed", 1)
        for arm in ARMS:
            spec = plan["arms"][arm]["spec"] if plan else None
            if spec and spec["name"] == name and not (arm == "D1" and int(seed) == eb.SELECTION_SEED):
                return model_dir(output, arm, int(seed), plan["selected"]) / "report.json"
        if name in CONFIGS:
            return selection_dir(output, name, int(seed)) / "report.json"
        return None
    return report_of


# ---------------------------------------------------------------------------------------------------------------
# --readouts (fit / policy data only) and --freeze-readouts
# ---------------------------------------------------------------------------------------------------------------

def readouts_arm(output, plan, arm, seed, device=None, data=None):
    out_dir = readouts_dir(output, arm, seed)
    if (out_dir / "readouts.json").is_file():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_of(device)
    costs = objective_costs()
    began = time.monotonic()
    result = {"arm": arm, "seed": seed, "plan_identity": IDENTITY}
    path = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
    with harness.deterministic_scoring() as policy:
        result["scoring_policy"] = policy
        model, payload = load_trained(path, device)
        result["model"] = str(path.relative_to(ROOT))
        result["model_sha256"] = eb.sha256_of(path)
        if model is None:
            result.update(typed_failure=payload["typed_failure"]["typed_failure"])
        else:
            members, z0, out = fit_anchor_predictions(model)
            k = len(eb.grid_items())
            ends = out["endpoint"].reshape(len(members), k, -1)
            matrix = tie_free_matrix(ends, z0, costs)
            data = data or fit_store(eb.FitShots())
            z, episodes = posterior_features(model, data)
            collapse = {"effective_rank": eb.effective_rank(z), "within_episode_share": eb.within_episode_share(
                z, episodes), "dim": int(z.shape[1]), "rows": int(z.shape[0]), "latent": "rssm.get_feat (posterior)"}
            del z
            items, actions = grid_actions()
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=float(eb.action_sensitivity(ends)),
                          G_collapse=collapse, fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          trainable_parameters=tn.parameter_count(model),
                          exploratory={"commit_points_fit_anchors": commit_statistics(out["update_counts"],
                                                                                       out["commit_steps"]),
                                       "hl_time_mean_fit_anchors": float(out["hl_time"].mean())})
            prow = policy_endpoint_rows(model, costs)
            result["policy_auc"] = eb.policy_grid_auc(prow, eb.PRIMARY_COST)
            result["policy_auc_exploratory"] = {view: eb.policy_grid_auc(prow, key) for view, key in EXPLORATORY.items()}
            result["macs_per_candidate"] = eb.macs_per_candidate(
                lambda a, b: tn.rollout(model, a, b), z0[0][None].expand(len(items), -1), actions,
                candidates=len(items))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{arm}-seed{seed}", seconds, seconds if device != "cpu" else 0)
    eb.write_json(out_dir / "readouts.json", result)  # written last (atomic)
    log(f"readouts {arm} seed {seed}: {json.dumps({k: result.get(k) for k in ('policy_auc', 'G_action')})}")
    return 0


def readouts(output, queue, device=None):
    plan = load_plan(output)
    data = None
    for name in plan["queues"]["readouts"].get(str(queue), []):
        arm, seed = name.split("--seed")
        if not (readouts_dir(output, arm, int(seed)) / "readouts.json").is_file():
            data = data or fit_store(eb.FitShots())
        readouts_arm(output, plan, arm, int(seed), device, data)
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
    for arm, seed in readout_items():
        rdir = readouts_dir(output, arm, seed)
        if not (rdir / "readouts.json").is_file():
            raise SystemExit(f"readouts of {arm} seed {seed} missing")
        mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
        entries[f"{arm}--seed{seed}"] = {"readouts": str((rdir / "readouts.json").relative_to(ROOT)),
                                         "readouts_sha256": eb.sha256_of(rdir / "readouts.json"),
                                         "model": str(mpath.relative_to(ROOT)), "model_sha256": eb.sha256_of(mpath)}
    eb.write_json(path, {"schema": SCHEMA_READOUTS_MANIFEST, "plan_identity": IDENTITY, "frozen_at": eb.utc_now(),
                         "entries": entries})
    log(f"readouts frozen ({len(entries)} arm-seeds)")
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
# --score (once; a process that did not train; GPU 0; deterministic; only after G-fid passed)
# ---------------------------------------------------------------------------------------------------------------

def _position_rows(at, branches, cache, times):
    positions = {}
    for b, (_, item) in enumerate(branches):
        e = cache["entries"][item["key"]]
        positions[item["key"]] = {}
        for j, t in enumerate(times):
            positions[item["key"]][str(t)] = {
                "pred": eb.carrier_position_mse(at[t][b], e["presence"][j], e["centers"][j]), "floor": None}
    return positions


@torch.no_grad()
def score_arm(output, plan, arm, seed, anchors, cells, device=None):
    """One record: rows of every mixed cell of every inventory (primary + exploratory keys), the grid commit points,
    wrong-anchor rows (480 fit anchors, grid), position error (descriptive), wall time per decision."""
    device = device_of(device)
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed}
    model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
    if model is None:
        return {**base, "typed_failure": payload["typed_failure"]["typed_failure"]}
    carriers = anchors["carriers"]
    inventories, commits = {}, {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {}
        for member, entry in sorted(cells[inventory]["mixed"].items()):
            items = entry["inventory"]
            actions = torch.stack([eb.launch5(it["action"]) for it in items])
            out = predict(model, carriers[member][None].expand(len(items), -1), actions, jump=True)
            inventories[inventory][member] = {"rows": candidate_rows(items, out, carriers[member], costs)}
            if inventory == GRID:
                commits[member] = out["commit_steps"]
    record = {**base, "typed_failure": None, "inventories": inventories, "commit_points": commits}
    fit = eb.fit_anchors()["carriers"]
    items, actions = grid_actions()
    record["wrong_anchor"] = {
        m: {"rows": eb.endpoint_rows(items, predict(model, fit[m][None].expand(len(items), -1), actions)["endpoint"],
                                     fit[m], costs)} for m in sorted(fit)}
    cache = eb.position_cache()
    branches = eb.position_branches(cells)
    z0 = torch.stack([carriers[m] for m, _ in branches])
    acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
    rolled = predict(model, z0, acts, carriers=True)["carriers"]
    at = {t: rolled[:, t // tn.STRIDE] for t in eb.POSITION_TIMES}
    record["positions"] = _position_rows(at, branches, cache, eb.POSITION_TIMES)
    members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(
        lambda m: tn.rollout(model, carriers[m][None].expand(len(items), -1), actions), members)
    return record


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#131 readiness gate); evaluation not read")
        return 1
    readouts_frozen_at = eb.read_json(output / "readouts-manifest.json")["frozen_at"]
    with harness.deterministic_scoring() as policy:
        anchors_path = eb.evaluation_anchors_path(output)
        anchors = (torch.load(anchors_path, map_location="cpu", weights_only=True) if anchors_path.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        for arm, seed in readout_items():
            path = record_path(output, arm, seed)
            if path.is_file():
                continue
            began = time.monotonic()
            record = score_arm(output, plan, arm, seed, anchors, cells, device)
            seconds = time.monotonic() - began
            record.update(scoring_policy=policy, scored_at=eb.utc_now(), readouts_frozen_at=readouts_frozen_at,
                          anchors_read_at=anchors["read_at"], score_seconds=seconds)
            eb.write_json(path, record)
            eb.mark(output, f"score-{arm}-seed{seed}", seconds, seconds if device_of(device) != "cpu" else 0)
            log(f"scored {arm} seed {seed} ({seconds:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --score-novelty (P-novelty, DESCRIPTIVE; once, after --score; GPU 0; deterministic; resumable per record)
# ---------------------------------------------------------------------------------------------------------------

SCHEMA_NOVELTY_RECORD = "issue_131_novelty_record_v1"
NOVELTY_ARMS = tuple(ARMS)  # D1 and D1-m; the exploratory views D1-coarse / D1-hl are not scored on novelty


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


@torch.no_grad()
def novelty_level_rows(model, items, z0, costs):
    """Rows of one novelty level from its carrier anchor: the primary scoring path's tie-free cost of the decoded
    frame-225 carrier (score_arm's rows without the exploratory keys)."""
    actions = torch.stack([eb.launch5(it["action"]) for it in items])
    return eb.rc.cost_rows(items, predict(model, z0[None].expand(len(items), -1), actions)["endpoint"], z0, costs)


def score_novelty(output, device=None, cells=None):
    """P-novelty: every guard-passing arm, every seed, on the mixed novelty levels (grid) from each level's carrier
    anchor; refuses before the primary records and the readouts freeze exist; never touches a primary record."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    missing = [f"{a}--seed{s}" for a, s in readout_items() if not record_path(output, a, s).is_file()]
    if missing:
        log(f"score-novelty refused: primary records missing ({missing})")
        return 1
    records = load_records(output)
    guards_of, extra = arm_guards(output, records), guard_extras(output)
    status = {arm: {"guards_pass": arm_guard_pass(guards_of[arm], extra),
                    "failed": sorted(k for k, v in {**guards_of[arm], **extra}.items()
                                     if k.startswith("G_") and v is not True)} for arm in NOVELTY_ARMS}
    costs = objective_costs()
    with harness.deterministic_scoring() as policy:
        cells = eb.novelty_cells() if cells is None else cells
        path = eb.novelty_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_novelty_anchors(output, policy, cells))
        eb.novelty_reference_records(output, policy, cells, anchors["carriers"])
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_131_novelty_arms_v1", "arms": status,
                                                          "anchors_read_at": anchors["read_at"]})
        for arm in NOVELTY_ARMS:
            if not status[arm]["guards_pass"]:
                log(f"score-novelty: {arm} not scored (failed guards {status[arm]['failed']})")
                continue
            for seed in eb.SEEDS:
                rpath = novelty_record_path(output, arm, seed)
                if rpath.is_file():
                    continue
                began = time.monotonic()
                mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
                model, _ = load_trained(mpath, device)
                grid = {level: {"rows": novelty_level_rows(model, cells["levels"][level]["inventory"],
                                                           anchors["carriers"][level], costs)}
                        for level in sorted(cells["mixed"])}
                seconds = time.monotonic() - began
                eb.write_json(rpath, {"schema": SCHEMA_NOVELTY_RECORD, "arm": arm, "seed": seed,
                                      "model_sha256": eb.sha256_of(mpath), "inventories": {GRID: grid},
                                      "scoring_policy": policy, "scored_at": eb.utc_now(),
                                      "anchors_read_at": anchors["read_at"]})
                eb.mark(output, f"score-novelty-{arm}-seed{seed}", seconds,
                        seconds if device_of(device) != "cpu" else 0)
                log(f"novelty scored {arm} seed {seed} ({seconds:.0f}s)")
    return 0


def novelty_section(output, records, cells=None):
    """DESCRIPTIVE novelty table (no tokens, never in a disposition) when the novelty pass has run, else None."""
    output = Path(output)
    arms_path = novelty_dir(output) / "arms.json"
    if not arms_path.is_file():
        return None
    status = eb.read_json(arms_path)["arms"]
    scored = [arm for arm in NOVELTY_ARMS if status[arm]["guards_pass"]]
    cells = eb.novelty_cells() if cells is None else cells
    anchors_read = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)["read_at"]
    views = {arm: ({seed: eb.read_json(novelty_record_path(output, arm, seed)) for seed in eb.SEEDS},
                   eb.PRIMARY_COST) for arm in scored}
    refs = eb.novelty_reference_records(output, None, cells)  # read back (written once by --score-novelty)
    last_primary = max(r["scored_at"] for recs in records.values() for r in recs.values())
    return {"label": "DESCRIPTIVE (P-novelty)", "scored_arms": scored,
            "not_scored": {arm: status[arm]["failed"] for arm in NOVELTY_ARMS if arm not in scored},
            "chronology": {"pass": bool(anchors_read >= last_primary), "novelty_anchors_read_at": anchors_read,
                           "last_primary_scored_at": last_primary},
            "table": eb.novelty_table({**views, **refs}, cells)}


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ["records/*.json", "novelty/arms.json", "novelty/records/*.json", "novelty/references/*.json"]


def load_records(output):
    return {arm: {seed: eb.read_json(record_path(output, arm, seed)) for seed in eb.SEEDS} for arm in ARMS}


def global_guards(output, plan, records):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    anchors_read = min(r["anchors_read_at"] for recs in records.values() for r in recs.values())
    first_scored = min(r["scored_at"] for recs in records.values() for r in recs.values())
    chronology = bool(plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read <= first_scored)
    policies = [r["scoring_policy"] for recs in records.values() for r in recs.values()]
    return {
        "G_isolation": {"pass": True, "rule": "fit split trains, policy split selects; evaluation anchors are read "
                        "only by --score after --freeze-readouts (G_chronology) and after G-fid passed"},
        "G_chronology": {"pass": chronology, "plan_frozen_at": plan["frozen_at"],
                         "readouts_frozen_at": manifest["frozen_at"], "anchors_read_at": anchors_read,
                         "first_scored_at": first_scored},
        "G_scoring_policy": {"pass": all(p == harness.SCORING_POLICY for p in policies)},
        "G_records": eb.records_guard(output, IDENTITY),
    }


def arm_compute(output, plan, arm):
    seconds, macs, walls, params = {}, [], [], None
    for seed in eb.SEEDS:
        report = model_dir(output, arm, seed, plan["selected"]) / "report.json"
        if report.is_file():
            seconds[str(seed)] = eb.read_json(report)["gpu_seconds"]
        rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
        if rd.get("macs_per_candidate") is not None:
            macs.append(rd["macs_per_candidate"])
            params = rd["trainable_parameters"]
        rec = eb.read_json(record_path(output, arm, seed))
        if rec.get("wall_per_decision") is not None:
            walls.append(rec["wall_per_decision"])
    return {"trainable_parameters": params, "training_gpu_seconds": seconds,
            "macs_per_candidate": float(np.mean(macs)) if macs else None,
            "wall_per_decision": float(np.median(walls)) if walls else None}


def record_commit_statistics(records):
    """Exploratory: commit points of the scored grid candidates (all seeds)."""
    counts, steps = [], []
    for record in records.values():
        if record.get("typed_failure"):
            continue
        for member_steps in record["commit_points"].values():
            for s in member_steps:
                counts.append(len(s))
                steps.append(s)
    return commit_statistics(counts, steps)


def guard_extras(output):
    output = Path(output)
    gport_ok = all(eb.read_json(output / p).get("pass") is True for p in ("gport/gport.json", "gport/agent/gport.json")
                   if (output / p).is_file()) and (output / "gport/gport.json").is_file()
    return {"G_port": bool(gport_ok), "G_fid": gfid_pass(output)}


def arm_guards(output, records):
    """{arm: eb.arm_guard_block} of the arm checks from the readouts and the primary records (publication and
    --score-novelty use this one computation)."""
    guards_of = {}
    for arm in ARMS:
        per_seed = {}
        for seed in eb.SEEDS:
            rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
            per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": rd.get("G_action"),
                              "G_collapse": rd.get("G_collapse"), "policy_ceiling_auc": None,
                              "typed_failure": rd.get("typed_failure") or records[arm][seed].get("typed_failure")}
        guards_of[arm] = eb.arm_guard_block(per_seed, ARM_CHECKS,
                                            nonfinite=eb.nonfinite_share(records[arm].values(), eb.PRIMARY_COST))
    return guards_of


def arm_guard_pass(guards, extra_guards):
    """eb.arm_block's arm pass: every G_* entry of the arm guards and the extra guards (G_port, G_fid) is True."""
    merged = {**guards, **extra_guards}
    names = [k for k in merged if k.startswith("G_")]
    return bool(names) and all(merged[k] is True for k in names)


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = eb.evaluation_cells()
    refs = eb.references()
    guards_global = global_guards(output, plan, records)
    global_pass = all(g["pass"] is True for g in guards_global.values())
    extra_guards = guard_extras(output)
    views = {arm: (records[arm], eb.PRIMARY_COST) for arm in ARMS}
    views.update({view: (records["D1"], key) for view, key in EXPLORATORY.items()})
    blocks, tables = {}, {}
    guards_of = arm_guards(output, records)
    within = {}
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", *views])}
        if inventory == GRID:
            for arm in ARMS:
                blocks[arm] = eb.arm_block(rows, arm, records[arm], eb.PRIMARY_COST, guards_of[arm], global_pass,
                                           arm_compute(output, plan, arm), ARMS[arm]["role"], ARMS[arm]["program_set"],
                                           extra_guards=extra_guards)
            d1_ok = global_pass and blocks["D1"]["guards"]["pass"]
            within = {"D1 - J_dec": eb.within_family(rows, "D1", "J_dec", d1_ok),
                      "D1 - D1-m": eb.within_family(rows, "D1", "D1-m", d1_ok and blocks["D1-m"]["guards"]["pass"]),
                      "D1-coarse - D1": eb.within_family(rows, "D1-coarse", "D1", d1_ok),
                      "D1-hl - D1": eb.within_family(rows, "D1-hl", "D1", d1_ok)}
        else:
            tables[inventory]["contrasts"] = {
                f"{arm} - U": eb.contrast(rows, arm, "U", global_pass and blocks[arm]["guards"]["pass"])
                for arm in ARMS}
    wrong = {arm: eb.wrong_anchor_table(cells, records[arm], arm, eb.PRIMARY_COST,
                                        global_pass and blocks[arm]["guards"]["pass"]) for arm in ARMS}
    curves = eb.reference_position_curves()
    positions = eb.position_table({"D1": records["D1"], "D1-m": records["D1-m"], "U": curves["U"],
                                   "C*": curves["C*"]}, cells)
    gfid_path = output / "fidelity/gfid.json"
    gfid_block = {"D1": {"pass": extra_guards["G_fid"], "record": str(gfid_path.relative_to(ROOT)),
                         "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, within, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": positions, "selection": plan["selected"],
        "commit_points": {arm: record_commit_statistics(records[arm]) for arm in ARMS},
        "readouts": {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "policy_auc_exploratory", "G_action", "G_collapse",
                                                "exploratory", "typed_failure")}
                     for a, s in readout_items()},
        "novelty": novelty_section(output, records),
        "native_success_head": None, "u_plus_head": "not applicable: THICK has no native success head on NovPhy"})
    compute = {"schema": "issue_131_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} D1 THICK: findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configuration "
             f"{summary['extra']['selection']}.", "", *eb.format_arm_rows(summary["arms"]), "", "## Within family", ""]
    for label, block in summary["within_family"].items():
        lines.append(f"- {label}: {eb.rc.fmt_block(block['estimate'])} ({block['token']} / {block['reverse_token']})")
    lines += ["", "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    lines += ["", "## Commit points (exploratory; grid candidates, all seeds)", ""]
    for arm, stats in summary["extra"]["commit_points"].items():
        lines.append(f"- {arm}: mean commits {stats['mean_commits']}, share with a commit {stats['share_with_commit']}")
    novelty = summary["extra"]["novelty"]
    if novelty is not None:
        lines += ["", f"## Novelty pairs ({novelty['table']['label']})", "",
                  f"Not scored (guards): {novelty['not_scored'] or 'none'}; novelty anchors read after the primary "
                  f"records: {novelty['chronology']['pass']}.", "", *eb.format_novelty_rows(novelty["table"])]
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
        arm, seed = "D1", eb.SELECTION_SEED
        stored = eb.read_json(record_path(output, arm, seed))
        anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
        with harness.deterministic_scoring():
            fresh = score_arm(output, plan, arm, seed, anchors, eb.evaluation_cells(), device)
        fresh = json.loads(eb.json_text(fresh))
        same = all(fresh["inventories"][inv] == stored["inventories"][inv] for inv in eb.INVENTORIES)
        spot = {"record": f"{arm}--seed{seed}", "pass": bool(same), "rule": "exact row equality, every inventory"}
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
    parser.add_argument("--workers", type=int, default=0, help="accepted for the contract; data live in memory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B127_D1_CPU_THREADS", "8")))  # shared host: avoid oversubscription
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
