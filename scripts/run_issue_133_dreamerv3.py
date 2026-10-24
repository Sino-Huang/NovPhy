"""Issue #133 (#127 coverage arm T1, rule P-coverage): DreamerV3 on E99 carriers against U on the sealed NovPhy
launch-ranking grid.

Arm (Decision P-other adapter on #133, issuecomment-6388683322):
- **T1** (coverage, secondary; never in the program claim): the official NM512/dreamerv3-torch world model
  (vendored @ 6ef8646; adapter ``world_model/baselines/dreamerv3/novphy.py``) at the dreamerv3-torch ``defaults``
  with the official vector-observation pathway on the 236-d carrier, no reward / continue heads (14 671 084
  parameters), offline on the fit split: 46-frame episodes (frames 0, 5, ..., 225), the launch at t0, upstream
  16 x 64 sample_episodes rows, DreamerV3's replay ratio over the dataset (320 137 updates at train_ratio 512). Score
  = tie-free ``EndpointCosts`` of the decoded frame-225 carrier (deterministic rollout: latent modes). Configuration
  selected on the policy split among three pre-declared ones (``dn.CONFIGURATIONS``; seed 20260908, full budget);
  the selected run is reused as main seed 20260908.
- No native success head (no reward head) -> no native-head score and no U + head control. No matched-budget arm.

Gates (#133): ``--selection`` and ``--train`` refuse without the orchestrator's ``funded.json`` (Decision
P-coverage funding), a passing ``gport/gport.json`` and a passing ``fidelity/gfid.json`` (PushT Tier T);
``--score`` refuses unless G-fid passed.

Modes (round3/child-runner-contract.md): --project (CPU, pre-funding GPU-h projection), --selection,
--selection-readouts, --smoke, --prepare, --train, --readouts, --freeze-readouts, --score, --score-novelty
(P-novelty, descriptive), --publish, --validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_133.json, prepared, not enqueued before funding); tests use CUDA_VISIBLE_DEVICES="".
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
from world_model.baselines.dreamerv3 import novphy as dn  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 133
IDENTITY = "issue-133-dreamerv3-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_133_dreamerv3.py"
TESTS = "tests/test_issue_133_dreamerv3.py"
SCHEMA_MODEL = "issue_133_model_v1"
SCHEMA_PLAN = "issue_133_plan_v1"
SCHEMA_RECORD = "issue_133_record_v1"
SCHEMA_SMOKE = "issue_133_smoke_v1"
SCHEMA_SELECTION = "issue_133_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_133_readouts_manifest_v1"
SCHEMA_PROJECTION = "issue_133_projection_v1"
DECISIONS = {
    "coverage": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388525040",
    "license": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388618813",
    "route": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388683089",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388683322",
    "band": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388719519",
    "fidelity": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388719795",
    "selection": "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388865221",
}

CONFIGS = dict(dn.CONFIGURATIONS)  # native / lr3e-4 / rr128 (pre-declared)
SELECTION_QUEUES = {0: ["native"], 1: ["lr3e-4"], 2: ["rr128"]}  # one configuration per scheduler job
FIT_SHOTS = 13_919  # eb.FitShots() (13 920 minus one typed-failure branch); checked by --smoke
BATCH, LENGTH = 16, 64  # upstream batch_size x batch_length
ARMS = {"T1": {"size": "native", "role": "coverage (secondary, P-coverage; never in the program claim)",
               "program_set": False}}
ARM = "T1"
COLLAPSE_INDEX = list(range(0, len(dn.POSITIONS), 3))  # frames 0, 15, ..., 225 (the #123 probe stride)
POSITION_INDEX = [t // dn.STRIDE for t in eb.POSITION_TIMES]
CHECKPOINT_EVERY = 2000
LOG_EVERY = 100
CHILD_CAP_GPU_HOURS = 76.0  # the upper end of the #133 estimate (P-coverage funds against the projection)
SEED_CAP_GPU_HOURS = 24.0  # #127: native cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
GRID = "grid"
ARM_CHECKS = ["G_rank", "G_action", "G_collapse", "G_nonfinite", "G_trained"]
# fp32 with torch's default flags (upstream precision 32: matmul TF32 off, cuDNN TF32 on; the vector model has no
# convolution). Every readout / selection-readout / scoring / validation path runs under deterministic_scoring().
TRAINING_PRECISION = {"dtype": "float32", "flags": "torch defaults (upstream precision 32)"}


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------------------------
# configurations and paths
# ---------------------------------------------------------------------------------------------------------------

def native_steps(configuration, shots=FIT_SHOTS):
    """DreamerV3's replay ratio over the fit dataset (dn.updates_for over shots x 46 steps)."""
    return dn.updates_for(shots * len(dn.POSITIONS), dn.build_config(CONFIGS[configuration]))


def config_spec(name):
    return {"name": name, "arm": ARM, "size": "native", "configuration": name, "overrides": {},
            "steps": native_steps(name), "batch": BATCH, "length": LENGTH}


def arm_spec(arm, selected):
    """The training spec of the arm at the selected configuration."""
    if arm != ARM:
        raise KeyError(arm)
    return config_spec(selected)


def build_model(spec, device="cpu"):
    return dn.build_model(spec["configuration"], spec.get("overrides"), device=device)


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, arm, seed, selected=None):
    """T1 seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if arm == ARM and seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("model_dir(T1, 20260908) needs the selected configuration")
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
    """Every fit shot's episode: carriers [N, 46, 236] float32 (frames 0, 5, ..., 225; positions past the end repeat
    the final carrier) and launch5 [N, 5], in the frozen shot order (CPU)."""
    carriers = []
    for start in range(0, len(shots.keys), chunk):
        index = range(start, min(len(shots.keys), start + chunk))
        pairs = [(s, p) for s in index for p in dn.episode_positions(shots.ends[s])]
        rows = shots.carrier_rows([s for s, _ in pairs], [p for _, p in pairs])
        carriers.append(torch.as_tensor(np.asarray(rows), dtype=torch.float32).reshape(
            len(index), len(dn.POSITIONS), dn.CARRIER_DIM))
    return {"carriers": torch.cat(carriers), "launch5": torch.as_tensor(shots.launch5, dtype=torch.float32)}


def training_batch(data, step, seed, spec):
    """Batch ``step`` of a run: upstream sample_episodes rows drawn from eb.batch_rng(seed, step)."""
    lengths = np.full(len(data["launch5"]), len(dn.POSITIONS))
    rows = dn.sample_segments(eb.batch_rng(seed, step), lengths, spec["length"], spec["batch"])
    return dn.training_batch(data["carriers"], data["launch5"], rows)


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss / gradient norm -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one model (resumable from out_dir/resume.pt; the official update via dn.train_loop). Batch b =
    training_batch(data, b, seed, spec); the posterior samples draw from the global torch generators seeded with
    ``seed`` (checkpointed)."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    steps = spec["steps"]
    torch.manual_seed(seed)
    model = build_model(spec, device)
    try:
        out = dn.train_loop(model, out_dir, steps, lambda b: training_batch(data, b, seed, spec),
                            lambda: bool(eb.STOP_REQUESTED), device, max_steps=max_steps, log_every=LOG_EVERY,
                            checkpoint_every=CHECKPOINT_EVERY)
    except dn.TrainingPaused as paused:
        log(f"{spec['name']} seed {seed}: stop signal; {paused}")
        raise eb.TrainingPaused(f"{spec['name']} seed {seed} {paused}") from None
    if out["failure"] is None and out["stop"] < steps:  # max_steps reached (tests, throughput)
        return {"partial": True, "steps_done": out["steps_done"], "seconds": out["seconds"]}
    complete = out["failure"] is None
    precision = {**TRAINING_PRECISION, "device": device,
                 "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": out["failure"],
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "parameters": dn.parameter_count(model), "precision": precision}
    eb.atomic_torch(final, payload)
    seconds = out["elapsed"]
    report = {"spec": spec, "seed": seed, "steps": steps, "complete": complete, "typed_failure": out["failure"],
              "wall_seconds": seconds, "gpu_seconds": seconds if device.startswith("cuda") else 0.0,
              "device": device, "seconds_per_step": seconds / max(1, steps),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": out["trace"][-1] if out["trace"] else None, "precision": precision, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if (out_dir / "resume.pt").is_file() and complete:
        (out_dir / "resume.pt").unlink()
    log(f"{spec['name']} seed {seed}: {'complete' if complete else out['failure']['typed_failure']} after "
        f"{seconds:.0f}s")
    return report


def load_trained(path, device=None):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL:
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    device = device_of(device)
    model = build_model(payload["spec"], device)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


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
def predict(model, z0, actions, keep=(), batch=4096):
    """Deterministic rollouts (dn.rollout) of anchors z0 [B, 236] under launches [B, 5], in batches; CPU tensors:
    ``endpoint`` [B, 236] (frame 225) and ``carriers`` {rollout index: [B, 236]} for ``keep``."""
    ends, kept = [], {k: [] for k in keep}
    for i in range(0, len(z0), batch):
        out = dn.rollout(model, z0[i:i + batch], actions[i:i + batch], keep=keep)
        ends.append(out["endpoint"].float().cpu())
        for k in keep:
            kept[k].append(out["carriers"][k].float().cpu())
    return {"endpoint": torch.cat(ends), "carriers": {k: torch.cat(v) for k, v in kept.items()}}


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


@torch.no_grad()
def fit_anchor_predictions(model):
    """Rollouts of the 16 grid launches from the 480 fit anchors (G-rank, G-action)."""
    anchors = eb.fit_anchors()["carriers"]
    members = sorted(anchors)
    items, actions = grid_actions()
    z0 = torch.stack([anchors[m] for m in members])
    out = predict(model, z0.repeat_interleave(len(items), 0), actions.repeat(len(members), 1))
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


@torch.no_grad()
def level_rows(model, items, z0, costs):
    """Rows of one level's candidates from its carrier anchor: the tie-free cost of the decoded frame-225 carrier
    (the primary scoring path; shared by policy AUCs, --score and --score-novelty)."""
    actions = torch.stack([eb.launch5(it["action"]) for it in items])
    return eb.endpoint_rows(items, predict(model, z0[None].expand(len(items), -1), actions)["endpoint"], z0, costs)


@torch.no_grad()
def policy_endpoint_rows(model, costs):
    """{member: rows} of the policy grid mixed cells."""
    anchors = eb.policy_anchors()["carriers"]
    return {member: level_rows(model, entry["inventory"], anchors[member], costs)
            for member, entry in sorted(eb.policy_cells(GRID).items())}


@torch.no_grad()
def posterior_features(model, data, frames=COLLAPSE_INDEX, batch=512):
    """G-collapse latent (Decision P-other adapter, item 7): the RSSM feature ``get_feat`` (one-hot stochastic mode
    32 x 32 + deter 512) of the deterministic posterior of each fit episode at frames 0, 15, ..., 225. Returns z
    [N * F, 1536] float64 (CPU) and the episode index of every row."""
    device = next(model.parameters()).device
    feats = []
    n = len(data["launch5"])
    for start in range(0, n, batch):
        rows = slice(start, min(n, start + batch))
        feats.append(dn.posterior_features(model, data["carriers"][rows].to(device), data["launch5"][rows],
                                           frames).double().cpu())
    z = torch.cat(feats)
    episodes = torch.arange(n).repeat_interleave(len(frames))
    return z.reshape(-1, z.shape[-1]), episodes


# ---------------------------------------------------------------------------------------------------------------
# gates (#133)
# ---------------------------------------------------------------------------------------------------------------

def funded(output):
    """The orchestrator's funding marker (Decision P-coverage funding): present, a JSON object, not ``funded: false``."""
    path = Path(output) / "funded.json"
    if not path.is_file():
        return False
    try:
        marker = eb.read_json(path)
    except ValueError:
        return False
    return isinstance(marker, dict) and marker.get("funded", True) is True


def gport_pass(output):
    path = Path(output) / "gport/gport.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def gfid_pass(output):
    path = Path(output) / "fidelity/gfid.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def readiness(output):
    """Funding marker, G-port pass and G-fid pass (validation ladder: the PushT fidelity first)."""
    out = {"funded.json": funded(output), "gport/gport.json": gport_pass(output),
           "fidelity/gfid.json": gfid_pass(output)}
    out["pass"] = all(out.values())
    return out


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, queue, device=None):
    gate = readiness(output)
    if not gate["pass"]:
        log(f"selection refused: readiness gate {gate}; nothing trained")
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
# --project (CPU, before funding) and --smoke
# ---------------------------------------------------------------------------------------------------------------

def lpt(items, bins=4):
    """Longest-processing-time assignment: items [(name, seconds)] -> {bin: [names]}, loads."""
    queues, loads = {b: [] for b in range(bins)}, [0.0] * bins
    for name, seconds in sorted(items, key=lambda t: (-t[1], t[0])):
        b = int(np.argmin(loads))
        queues[b].append(name)
        loads[b] += seconds
    return queues, loads


def remaining_items(output, selected):
    """(arm, seed, spec, out_dir) of every training run the plan still needs."""
    return [(ARM, seed, arm_spec(ARM, selected), model_dir(output, ARM, seed)) for seed in eb.SEEDS
            if seed != eb.SELECTION_SEED]


READOUT_HOURS_PER_ARM_SEED = 0.5  # G-rank / G-action / G-collapse / policy AUC of one arm-seed (estimate)
SCORE_HOURS = 1.0  # one scoring pass of 3 arm-seeds incl. wrong anchors, positions and novelty (estimate)


def child_projection(per_update_gpu_s, fidelity_gpu_h, selected_steps=None):
    """GPU-h of everything left from per-configuration GPU s/update: three selection runs, two more seeds at the
    selected configuration (unknown before selection: the most expensive one), readouts, scoring, the fidelity."""
    steps = {name: native_steps(name) for name in CONFIGS}
    selection_s = {name: steps[name] * per_update_gpu_s[name] for name in CONFIGS}
    main_steps = selected_steps if selected_steps is not None else max(steps.values())
    rate = max(per_update_gpu_s.values())
    training = [(f"{ARM}--seed{seed}", main_steps * rate) for seed in eb.SEEDS if seed != eb.SELECTION_SEED]
    readout_s = READOUT_HOURS_PER_ARM_SEED * 3600 * len(eb.SEEDS)
    score_s = SCORE_HOURS * 3600
    queues, loads = lpt(training)
    total = sum(selection_s.values()) + sum(s for _, s in training) + readout_s + score_s + fidelity_gpu_h * 3600
    wall = max(selection_s.values()) + max(loads) + readout_s / 4 + score_s + fidelity_gpu_h * 3600
    per_seed = main_steps * rate
    out = {"steps": steps, "selection_gpu_h": {k: v / 3600 for k, v in selection_s.items()},
           "training_gpu_h": sum(s for _, s in training) / 3600, "readouts_gpu_h_estimate": readout_s / 3600,
           "score_gpu_h_estimate": score_s / 3600, "fidelity_gpu_h": fidelity_gpu_h, "total_gpu_h": total / 3600,
           "child_cap_gpu_h": CHILD_CAP_GPU_HOURS, "native_per_seed_gpu_h": per_seed / 3600,
           "seed_cap_gpu_h": SEED_CAP_GPU_HOURS, "wall_h_on_4_gpus_serial_ladder": wall / 3600,
           "wall_cap_h": WALL_CAP_HOURS, "training_queues": queues, "training_queue_loads_h": [x / 3600 for x in loads]}
    out["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed / 3600 <= SEED_CAP_GPU_HOURS
                              and wall / 3600 <= WALL_CAP_HOURS)
    return out


def cpu_seconds_per_update(configuration="native", repeats=2, shots=64):
    """CPU s/update of T1 at a configuration (random carriers; native batch 16 x 64)."""
    import tempfile
    g = torch.Generator().manual_seed(0)
    data = {"carriers": torch.randn(shots, len(dn.POSITIONS), dn.CARRIER_DIM, generator=g),
            "launch5": torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(shots, 1)}
    spec = config_spec(configuration)
    with tempfile.TemporaryDirectory() as tmp:
        train_run(tmp, spec, 0, data, "cpu", max_steps=1)
        began = time.monotonic()
        report = train_run(tmp, spec, 0, data, "cpu", max_steps=repeats)
        return (time.monotonic() - began) / report["steps_done"]


SCHEMA_TIMING = "issue_133_timing_v1"
PLAN_CASES = 12 * 50  # MPC iterations (band max_iter) x evaluation cases; one case = one CEM sub-plan of 30 steps


def _rss_gib():
    """(current, peak) host RSS of this process in GiB (/proc/self/status VmRSS, getrusage ru_maxrss)."""
    import resource
    current = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = int(line.split()[1]) / 2 ** 20
    return current, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _time_updates(model, batch_fn, updates, device, chunk=100):
    """Official updates model.train_step(batch_fn(step)), timed per chunk (CUDA synchronised), batch building timed
    separately; GPU peak memory over the phase."""
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    chunks, batch_seconds, losses = [], 0.0, []
    began = time.monotonic()
    start = began
    for step in range(updates):
        t = time.monotonic()
        batch = batch_fn(step)
        batch_seconds += time.monotonic() - t
        losses.append(model.train_step(batch)["model_loss"])
        if (step + 1) % chunk == 0 or step + 1 == updates:
            _sync(device)
            now = time.monotonic()
            chunks.append({"updates": (step % chunk) + 1, "seconds": now - start})
            start = now
    _sync(device)
    total = time.monotonic() - began
    steady = [c["seconds"] / c["updates"] for c in chunks[1:]] or [c["seconds"] / c["updates"] for c in chunks]
    out = {"updates": updates, "seconds": total, "seconds_per_update_all": total / updates,
           "seconds_per_update_steady_median": float(np.median(steady)), "chunks": chunks,
           "batch_build_seconds_per_update": batch_seconds / updates,
           "first_loss": float(losses[0]), "last_loss": float(losses[-1]),
           "finite": bool(np.all(np.isfinite(losses)))}
    if str(device).startswith("cuda"):
        out.update(peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                   peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    out["host_rss_gib"], out["host_peak_rss_gib"] = _rss_gib()
    return out


def timing(output, updates=1000, cem_steps=30, device=None, t1_overrides=None, data=None):
    """Decision P-coverage timing probe (no funded.json needed; <= 1 GPU-h): ``updates`` official updates of the
    native T1 model on the fit split (the runner's own batches) and of the PushT fidelity model on synthetic batches
    of the real shape (16 x 64 frames of 64 x 64 x 3 uint8 values, proprio, 10-d actions; the cache is a funded
    stage), then one CEM planning case (``cem_steps`` CEM steps of 300 samples x horizon 5 through PlanningModel,
    goal encoding included; the PushT env rollouts are not part of it). rr128's update is the native update (same
    model, batch and optimizer; only the update count differs), so it is not timed separately. Writes timing.json
    and a ledger entry; models, optimizer states and batches are discarded."""
    from scripts import b127_dreamerv3_fidelity as fid
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    result = {"schema": SCHEMA_TIMING, "plan_identity": IDENTITY, "decision": "P-coverage timing probe (#133)",
              "device": device, "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
              "torch": torch.__version__, "updates": updates, "rr128": "same update as native (not timed)"}
    t = time.monotonic()
    data = data if data is not None else fit_store(eb.FitShots())
    result["fit_store_seconds"] = time.monotonic() - t
    spec = {**config_spec("native"), "overrides": t1_overrides or {}}
    torch.manual_seed(eb.SELECTION_SEED)
    model = build_model(spec, device)
    result["t1"] = {"parameters": dn.parameter_count(model), "batch": [spec["batch"], spec["length"]],
                    **_time_updates(model, lambda b: training_batch(data, b, eb.SELECTION_SEED, spec), updates, device)}
    model = data = None  # free the T1 model and the fit store before the PushT phase
    torch.manual_seed(fid.SEED)
    model = fid.build_pusht_model(device)
    config = model._config
    b, length = config.batch_size, config.batch_length
    rng = np.random.default_rng(0)
    first = torch.zeros(b, length)
    first[:, 0] = 1.0
    pool = [{"image": torch.from_numpy(rng.integers(0, 256, (b, length, fid.IMAGE, fid.IMAGE, 3), dtype=np.uint8)),
             "proprio": torch.randn(b, length, fid.PROPRIO_DIM), "action": torch.randn(b, length, 10),
             "is_first": first, "is_terminal": torch.zeros(b, length)} for _ in range(4)]
    result["pusht"] = {"parameters": dn.parameter_count(model), "batch": [b, length],
                       "data": "synthetic batches of the real shape (uint8 images cast to float per update)",
                       **_time_updates(model, lambda s: {**pool[s % 4], "image": pool[s % 4]["image"].float()},
                                       updates, device)}
    pool = None
    planner = fid.PlanningModel(model.eval())
    obs = {"visual": torch.rand(1, 1, 3, 224, 224, device=device) * 2 - 1,
           "proprio": torch.randn(1, 1, fid.PROPRIO_DIM, device=device)}
    samples = {k: v.expand(300, *v.shape[1:]).contiguous() for k, v in obs.items()}
    _sync(device)
    t = time.monotonic()
    with torch.no_grad():
        planner.encode_obs(obs)  # goal encoding
        for _ in range(cem_steps):
            z, _ = planner.rollout(samples, torch.randn(300, 5, 10, device=device))
            ((z["visual"][:, -1:] - z["visual"][:1, -1:]) ** 2).mean(dim=(1, 2, 3)).argsort()[:30]
    _sync(device)
    case = time.monotonic() - t
    result["cem_case"] = {"cem_steps": cem_steps, "samples": 300, "horizon": 5, "seconds": case,
                          "seconds_per_30_steps": case * 30 / cem_steps}
    result["host_rss_gib"], result["host_peak_rss_gib"] = _rss_gib()
    result["seconds"] = time.monotonic() - began
    result["written_at"] = eb.utc_now()
    eb.write_json(output / "timing.json", result)
    eb.mark(output, "timing", result["seconds"], result["seconds"] if device.startswith("cuda") else 0.0)
    log(f"timing: T1 {result['t1']['seconds_per_update_steady_median']:.4f} s/update, PushT "
        f"{result['pusht']['seconds_per_update_steady_median']:.4f} s/update, CEM case {case:.2f} s "
        f"({result['seconds']:.0f} s total)")
    projection_path = output / "projection.json"
    if device.startswith("cuda") and projection_path.is_file():  # the measured projection becomes primary
        projection = eb.read_json(projection_path)
        projection.update(projection_measured=measured_projection(result), primary="projection_measured",
                          timing_sha256=eb.sha256_of(output / "timing.json"))
        eb.write_json(projection_path, projection)
        child = projection["projection_measured"]["child"]
        log(f"projection.json: measured {child['total_gpu_h']:.1f} GPU-h; within caps {child['within_caps']}")
    return 0


def measured_projection(timing_record, plan_env_allowance_h=0.25):
    """The child projection from timing.json (measured steady-state s/update; PushT planning = 600 CEM cases x the
    measured case time + the env allowance)."""
    from scripts import b127_dreamerv3_fidelity as fid
    rate = timing_record["t1"]["seconds_per_update_steady_median"]
    pusht_rate = timing_record["pusht"]["seconds_per_update_steady_median"]
    native = dn.updates_for(fid.PUSHT_TRAIN_FRAMES / fid.FRAMESKIP, fid.pusht_config())
    fidelity = {"train_gpu_h": native * pusht_rate / 3600,
                "plan_gpu_h": PLAN_CASES * timing_record["cem_case"]["seconds_per_30_steps"] / 3600
                + plan_env_allowance_h, "native_updates": native, "seconds_per_update": pusht_rate}
    fidelity["total_gpu_h"] = fidelity["train_gpu_h"] + fidelity["plan_gpu_h"]
    return {"fidelity": fidelity, "child": child_projection({name: rate for name in CONFIGS}, fidelity["total_gpu_h"]),
            "t1_seconds_per_update": rate, "source": "timing.json (measured)"}


def project(output):
    """projection.json (CPU, before funding): when timing.json exists, the measured projection (primary); always the
    [INFERENCE] CPU models: T1 CPU s/update scaled by the GPU/CPU ratio measured on #131's native THICK update
    (upper model), and the scan-latency model (THICK's measured GPU s/update x 64 / 46 sequential steps; lower
    model); fidelity/projection.json carries the PushT G-fid projection under both models."""
    from scripts import b127_dreamerv3_fidelity as fid
    output = Path(output)
    fidelity = fid.project(output)
    cpu = cpu_seconds_per_update("native")
    ratio = fidelity["gpu_over_cpu"]
    rates = {name: cpu * ratio for name in CONFIGS}  # same model and batch in every configuration
    latency = {name: fidelity["latency_model_gpu_seconds_per_update"] for name in CONFIGS}
    upper = child_projection(rates, fidelity["total_gpu_h"])
    lower = child_projection(latency, fidelity["total_gpu_h_latency_model"])
    result = {"schema": SCHEMA_PROJECTION, "plan_identity": IDENTITY, "label": fidelity["label"],
              "t1_cpu_seconds_per_update": cpu, "gpu_over_cpu": ratio,
              "t1_gpu_seconds_per_update_upper": rates, "t1_gpu_seconds_per_update_latency_model": latency,
              "parameters": dn.parameter_count(dn.build_model()), "fidelity": fidelity,
              "projection_upper": upper, "projection_latency_model": lower, "primary": "projection_upper",
              "measured_at": eb.utc_now()}
    timing_path = output / "timing.json"
    if timing_path.is_file():
        record = eb.read_json(timing_path)
        if str(record.get("device", "")).startswith("cuda"):
            result["projection_measured"] = measured_projection(record)
            result["timing_sha256"] = eb.sha256_of(timing_path)
            result["primary"] = "projection_measured"
    eb.write_json(output / "projection.json", result)
    if result["primary"] == "projection_measured":
        child = result["projection_measured"]["child"]
        log(f"projection (measured): {child['total_gpu_h']:.1f} GPU-h; within caps {child['within_caps']}")
    else:
        log(f"projection [INFERENCE]: {lower['total_gpu_h']:.1f}-{upper['total_gpu_h']:.1f} GPU-h (fidelity "
            f"{fidelity['total_gpu_h_latency_model']:.1f}-{fidelity['total_gpu_h']:.1f}); within caps "
            f"{lower['within_caps']} / {upper['within_caps']}")
    return 0


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    launch = torch.as_tensor(shots.launch5)
    accounting = {"fit_shots": len(shots), "fit_shots_expected": FIT_SHOTS, "fit_levels": len(shots.levels),
                  "fit_positive": int(shots.verdicts.sum()), "fit_anchors": len(eb.fit_anchors()["carriers"]),
                  "policy_grid_mixed_cells": len(eb.policy_cells(GRID)), "episode_frames": len(dn.POSITIONS),
                  "shots_ending_before_225": int((np.asarray(shots.ends) < dn.END_FRAME).sum()),
                  "launch5_abs_max": float(launch.abs().max())}
    gport = eb.read_json(output / "gport/gport.json")
    adapter = eb.sha256_of(ROOT / "world_model/baselines/dreamerv3/novphy.py").split(":")[-1]
    gport_check = {"pass": gport.get("pass") is True,
                   "adapter_current": gport.get("adapter_sha256", "").split(":")[-1] == adapter}
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
    gfid = eb.read_json(output / "fidelity/gfid.json")
    fidelity_report = (gfid.get("model") or {}).get("report") or {}
    fidelity_h = (fidelity_report.get("gpu_seconds") or 0.0) / 3600 + sum(
        (r.get("wall_seconds") or 0.0) for r in gfid.get("runs", [])) / 3600
    remaining = remaining_items(output, selected)
    rate = throughput[selected]["seconds_per_step"]
    training = [(f"{arm}--seed{seed}", spec["steps"] * rate) for arm, seed, spec, _ in remaining]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values())
    readout_seconds = READOUT_HOURS_PER_ARM_SEED * 3600 * len(eb.SEEDS)
    score_seconds = SCORE_HOURS * 3600
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds + fidelity_h * 3600
    per_seed = arm_spec(ARM, selected)["steps"] * rate
    wall = max(loads) + readout_seconds / 4 + score_seconds
    projection = {"fidelity_gpu_h_spent": fidelity_h, "selection_gpu_h": selection_seconds / 3600,
                  "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h_estimate": readout_seconds / 3600, "score_gpu_h_estimate": score_seconds / 3600,
                  "total_gpu_h": total / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "native_per_seed_gpu_h": per_seed / 3600, "seed_cap_gpu_h": SEED_CAP_GPU_HOURS,
                  "wall_h_on_4_gpus": wall / 3600, "wall_cap_h": WALL_CAP_HOURS, "training_queues": queues,
                  "training_queue_loads_h": [x / 3600 for x in loads]}
    projection["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and per_seed / 3600 <= SEED_CAP_GPU_HOURS
                                     and wall / 3600 <= WALL_CAP_HOURS)
    smoke_record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
                    "readiness": readiness(output), "tests": tests_record, "throughput": throughput,
                    "selection": selection_record, "selected": selected, "projection": projection, "device": device,
                    "pass": bool(gport_check["pass"] and gport_check["adapter_current"] and tests.returncode == 0
                                 and projection["within_caps"] and accounting["fit_shots"] == FIT_SHOTS
                                 and accounting["launch5_abs_max"] <= 1.0),
                    "written_at": eb.utc_now()}
    eb.write_json(output / "smoke.json", smoke_record)
    log(f"smoke: pass {smoke_record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h, wall "
        f"{projection['wall_h_on_4_gpus']:.1f} h; selected {selected}")
    return 0 if smoke_record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/dreamerv3/__init__.py", "world_model/baselines/dreamerv3/novphy.py",
             "scripts/b127_dreamerv3_port_reference.py", "scripts/b127_dreamerv3_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/dreamerv3/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "fidelity/band.json", output / "fidelity/gfid.json",
              output / "smoke.json", output / "selection.json", output / "funded.json"]  # decisions.md: append-only
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"t1_{p.relative_to(ROOT)}", "artifact": str(p.relative_to(ROOT)), "sha256": eb.sha256_of(p)}
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
    readout_queues, _ = lpt([(f"{a}--seed{s}", 1.0) for a, s in readout_items()])
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {arm: {**ARMS[arm], "spec": arm_spec(arm, selected), "seeds": list(eb.SEEDS)} for arm in ARMS},
        "training_precision": TRAINING_PRECISION, "scoring_policy": harness.SCORING_POLICY,
        "reused": {ARM: {"seed": eb.SELECTION_SEED, "run": str(selection_dir(output, selected).relative_to(ROOT))}},
        "readout": {"rollout": "dn.rollout: anchor observed at t0 (obs_step from the initial state, latent mode), 45 "
                               "img_steps (launch at the first, no-ops after; latent modes); primary = tie-free "
                               "EndpointCosts of the decoder's mode (symexp) carrier at frame 225",
                    "collapse_latent": "dynamics.get_feat of the deterministic posterior on fit episodes at frames "
                                       "0, 15, ..., 225 (1024 one-hot stochastic + 512 deter)",
                    "action_sensitivity": "eb.action_sensitivity on the decoded frame-225 carriers, 480 fit anchors x "
                                          "16 grid launches"},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": [], "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "T1 is a coverage arm (P-coverage): secondary, reported, never in the program claim.",
            "T1's configuration was selected on the policy split among three pre-declared configurations (native, "
            "lr3e-4, rr128); U had no such tuning (#127).",
            "No reward / continue heads (offline, task-free data, as DINO-WM's DreamerV3 baseline): no native-head "
            "score and no U + head control.",
            "The deterministic readout uses latent modes (sample=False); upstream imagination samples.",
            "Training length = DreamerV3's replay ratio (train_ratio) over the fit dataset; upstream trains online.",
            "Scoring waits for G-fid pass (PushT Tier T); a G-fid failure drops T1 (fidelity_not_reproduced).",
            "P-novelty: after the primary scoring pass, T1 (if its guards pass) is scored on the #110 novelty levels "
            "from the E99 v1 anchor parse (--score-novelty); descriptive only, no tokens, never in a disposition.",
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
    if not funded(output):
        log("train refused: funded.json absent; nothing trained")
        return 1
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
        if name in CONFIGS:
            seed = int(seed)
            if seed == eb.SELECTION_SEED:
                return selection_dir(output, name, seed) / "report.json"
            if plan and name == plan["selected"]:
                return model_dir(output, ARM, seed, plan["selected"]) / "report.json"
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
                z, episodes), "dim": int(z.shape[1]), "rows": int(z.shape[0]), "latent": "dynamics.get_feat (posterior)"}
            del z
            items, actions = grid_actions()
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=float(eb.action_sensitivity(ends)),
                          G_collapse=collapse, fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          trainable_parameters=dn.parameter_count(model))
            prow = policy_endpoint_rows(model, costs)
            result["policy_auc"] = eb.policy_grid_auc(prow, eb.PRIMARY_COST)
            result["macs_per_candidate"] = eb.macs_per_candidate(
                lambda a, b: dn.rollout(model, a, b), z0[0][None].expand(len(items), -1), actions,
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
    """One record: rows of every mixed cell of every inventory, wrong-anchor rows (480 fit anchors, grid), position
    error (descriptive), wall time per decision."""
    device = device_of(device)
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed}
    model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
    if model is None:
        return {**base, "typed_failure": payload["typed_failure"]["typed_failure"]}
    carriers = anchors["carriers"]
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {member: {"rows": level_rows(model, entry["inventory"], carriers[member], costs)}
                                  for member, entry in sorted(cells[inventory]["mixed"].items())}
    record = {**base, "typed_failure": None, "inventories": inventories}
    fit = eb.fit_anchors()["carriers"]
    items, actions = grid_actions()
    record["wrong_anchor"] = {m: {"rows": level_rows(model, items, fit[m], costs)} for m in sorted(fit)}
    cache = eb.position_cache()
    branches = eb.position_branches(cells)
    z0 = torch.stack([carriers[m] for m, _ in branches])
    acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
    rolled = predict(model, z0, acts, keep=POSITION_INDEX)["carriers"]
    at = {t: rolled[t // dn.STRIDE] for t in eb.POSITION_TIMES}
    record["positions"] = _position_rows(at, branches, cache, eb.POSITION_TIMES)
    members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(
        lambda m: dn.rollout(model, carriers[m][None].expand(len(items), -1), actions), members)
    return record


def score(output, device=None):
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#133 gate); evaluation not read")
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

SCHEMA_NOVELTY_RECORD = "issue_133_novelty_record_v1"
NOVELTY_ARMS = tuple(ARMS)


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


def score_novelty(output, device=None, cells=None):
    """P-novelty: every guard-passing arm, every seed, on the mixed novelty levels (grid) from each level's carrier
    anchor (the primary scoring path, level_rows); refuses before the primary records and the readouts freeze exist;
    never touches a primary record."""
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
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_133_novelty_arms_v1", "arms": status,
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
                grid = {level: {"rows": level_rows(model, cells["levels"][level]["inventory"],
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


def guard_extras(output):
    return {"G_port": gport_pass(output), "G_fid": gfid_pass(output)}


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
    blocks, tables = {}, {}
    guards_of = arm_guards(output, records)
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", *views])}
        if inventory == GRID:
            for arm in ARMS:
                blocks[arm] = eb.arm_block(rows, arm, records[arm], eb.PRIMARY_COST, guards_of[arm], global_pass,
                                           arm_compute(output, plan, arm), ARMS[arm]["role"], ARMS[arm]["program_set"],
                                           extra_guards=extra_guards)
        else:
            tables[inventory]["contrasts"] = {
                f"{arm} - U": eb.contrast(rows, arm, "U", global_pass and blocks[arm]["guards"]["pass"])
                for arm in ARMS}
    wrong = {arm: eb.wrong_anchor_table(cells, records[arm], arm, eb.PRIMARY_COST,
                                        global_pass and blocks[arm]["guards"]["pass"]) for arm in ARMS}
    curves = eb.reference_position_curves()
    positions = eb.position_table({ARM: records[ARM], "U": curves["U"], "C*": curves["C*"]}, cells)
    gfid_path = output / "fidelity/gfid.json"
    gfid_block = {ARM: {"pass": extra_guards["G_fid"], "record": str(gfid_path.relative_to(ROOT)),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, {}, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": positions, "selection": plan["selected"],
        "coverage_arm": True,
        "readouts": {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "G_action", "G_collapse", "typed_failure")}
                     for a, s in readout_items()},
        "novelty": novelty_section(output, records),
        "native_success_head": None, "u_plus_head": "not applicable: no reward head (task-free offline training)"})
    compute = {"schema": "issue_133_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} T1 DreamerV3 (coverage arm): findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configuration "
             f"{summary['extra']['selection']}.", "", *eb.format_arm_rows(summary["arms"]), "",
             "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    novelty = summary["extra"]["novelty"]
    if novelty is not None:
        lines += ["", f"## Novelty pairs ({novelty['table']['label']})", "",
                  f"Not scored (guards): {novelty['not_scored'] or 'none'}; novelty anchors read after the primary "
                  f"records: {novelty['chronology']['pass']}.", "", *eb.format_novelty_rows(novelty["table"])]
    lines += ["", "T1 is a coverage arm (P-coverage): never in the program claim. Intervals are descriptive "
              "(member-clustered bootstrap); tokens follow the #127 delta rule.", ""]
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
        arm, seed = ARM, eb.SELECTION_SEED
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
    for mode in ("project", "timing", "selection", "selection-readouts", "smoke", "prepare", "train", "readouts",
                 "freeze-readouts", "score", "score-novelty", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--workers", type=int, default=0, help="accepted for the contract; data live in memory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--timing-updates", type=int, default=1000, help="--timing: updates per model")
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B127_T1_CPU_THREADS", "8")))  # shared host: avoid oversubscription
    if args.timing:
        return timing(output, args.timing_updates, device=args.device)
    if args.project:
        return project(output)
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
