"""Issue #135 (#127 triggered Tier-C arm T2): DDLP (Deep Dynamic Latent Particles) on NovPhy decision frames against
U on the sealed launch-ranking grid.

Arm (Decisions on #135: adapter issuecomment-6396857866, recipe issuecomment-6396890056, readiness
issuecomment-6396890266):
- **T2** (triggered Tier-C, secondary; never in the program claim): the official taldatech/ddlp ``ObjectDynamicsDLP``
  (vendored @ 5686275; adapter ``world_model/baselines/ddlp/novphy.py``) at the released PHYRE-128 checkpoint's
  hparams, trained on fit-shot frame windows (128 x 128, stride 5, the decision frame x4 as the static burn-in
  context, the launch as an extra PINT particle), batch 12 (4 x 3), the official schedule. Rollout: 45 deterministic
  PINT steps from the decision frame to frame 225. Score = tie-free ``EndpointCosts`` of the particle -> E99-carrier
  readout (fit frames only) at frame 225 against the readout of the decision frame. Configuration selected on the
  policy split among three pre-declared ones (native / paper-schedule / repo-config; seed 20260908, full budget);
  the selected run is reused as main seed 20260908.
- Readiness stop before any DDLP training: R0 (the DDLP particle model in the official static mode) -> readout ->
  policy ceiling; < 0.65 -> typed stop ``readiness_ceiling_below_threshold``.

Gates: every GPU mode (--readiness, --selection, --selection-readouts, --train, --readouts, --score,
--score-novelty) refuses without the orchestrator's ``funded.json`` (Decision P-budget funding); --selection also
needs G-port, G-fid (PHYRE-128 Tier E) and the readiness record to have passed; --score needs G-fid.
``--timing`` is the measured GPU probe (<= 0.5 GPU-h; enqueued only after its own posted decision).

Modes (round3/child-runner-contract.md): --data (CPU frame cache), --project (CPU, [INFERENCE] projection),
--timing, --readiness, --selection, --selection-readouts, --smoke, --prepare, --train, --readouts,
--freeze-readouts, --score, --score-novelty (P-novelty, descriptive), --publish, --validate. GPU work only through
the night-window scheduler (scripts/gpu_window_issue_135*.json, prepared, not enqueued before funding); the env is
``b127-ddlp`` (Decision P-route issuecomment-6396701779); tests use CUDA_VISIBLE_DEVICES="".
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
from world_model.baselines.ddlp import novphy as dd  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 135
IDENTITY = "issue-135-ddlp-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_135_ddlp.py"
TESTS = "tests/test_issue_135_ddlp.py"
SCHEMA_MODEL = "issue_135_model_v1"
SCHEMA_PLAN = "issue_135_plan_v1"
SCHEMA_RECORD = "issue_135_record_v1"
SCHEMA_SMOKE = "issue_135_smoke_v1"
SCHEMA_SELECTION = "issue_135_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_135_readouts_manifest_v1"
SCHEMA_PROJECTION = "issue_135_projection_v1"
SCHEMA_TIMING = "issue_135_timing_v1"
SCHEMA_READINESS = "issue_135_readiness_v1"
SCHEMA_READOUT = "issue_135_readout_v1"
SCHEMA_FRAMES = "issue_135_frames_manifest_v1"
DECISIONS = {
    "license": "https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396690148",
    "route": "https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396701779",
    "band": "https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396727991",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396857866",
    "recipe": "https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396890056",
    "readiness": "https://github.com/Sino-Huang/NovPhy/issues/135#issuecomment-6396890266",
    "trigger": "https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6396584478",
    "g_action_calibration": "https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6396584235",
}

CACHE = Path(os.environ.get("B135_FRAME_CACHE", Path.home() / ".cache" / "novphy-issue-135-ddlp-frames"))
FIT_SHOTS = 13_919  # eb.FitShots() (13 920 minus one typed-failure branch); checked by --smoke
BATCH = dd.BATCH  # 12 = 4 processes x batch_size 3 (accel_conf.yml)
PAPER_EPOCH_STEPS = (2574 * (100 - 16 + 1)) // BATCH  # the paper's PHYRE epoch at batch 12: 18 232 steps
MODEL_CONFIGS = {"native": dd.NATIVE, "repo-phyre": dd.REPO_PHYRE}
CONFIGS = {  # declaration order (Decision P-other recipe)
    "native": {"model": "native", "epochs": 40, "epoch_steps": None},
    "paper-schedule": {"model": "native", "epochs": 40, "epoch_steps": PAPER_EPOCH_STEPS},
    "repo-config": {"model": "repo-phyre", "epochs": 40, "epoch_steps": None},
}
CUT_ORDER = ["repo-config"]  # P-cap: three configurations -> two
SELECTION_QUEUES = {0: ["native"], 1: ["paper-schedule"], 2: ["repo-config"]}
R0 = {"name": "r0-static", "model": "native", "epochs": 3, "epoch_steps": 10_000, "static": True}
ARMS = {"T2": {"size": "native", "role": "triggered Tier-C (secondary; never in the program claim)",
               "program_set": False}}
ARM = "T2"
ARM_CHECKS = ["G_rank", "G_action", "G_readout", "G_nonfinite", "G_trained"]
READOUT_POSITIONS = tuple(range(0, dd.ENDPOINT + 1, 15))  # fit readout frames: 0, 15, ..., 225 (clamped)
POSITION_STEPS = {t: t // dd.STRIDE for t in eb.POSITION_TIMES}  # rollout step of each position time
GRID = "grid"
THRESHOLD = eb.GUARD["readout_ceiling_auc_min"]  # 0.65
TYPED_STOP = "readiness_ceiling_below_threshold"
CHECKPOINT_EVERY = 2000
LOG_EVERY = 500
CHILD_CAP_GPU_HOURS = 164.0  # the upper end of the CoverageScout estimate for DDLP (82-164)
WALL_CAP_HOURS = 48.0  # on 4 GPUs
ROLLOUT_CHUNK = 256  # rollouts per forward
ENCODE_CHUNK = 128  # frames per encoder forward
TRAINING_PRECISION = {"dtype": "float32", "flags": "torch defaults (the repository's accelerate mixed_precision 'no')"}
CALIBRATION = ROOT / ".local-artifacts/issue-129-slotformer-v1/analysis/g-action-calibration.json"


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ---------------------------------------------------------------------------------------------------------------
# configurations and paths
# ---------------------------------------------------------------------------------------------------------------

def per_shot(model="native"):
    return dd.windows_per_episode(MODEL_CONFIGS[model])


def window_epoch_steps(shots=FIT_SHOTS, model="native"):
    """Steps of one pass over the NovPhy training windows (the official shuffled loader, drop_last)."""
    return shots * per_shot(model) // BATCH


def config_spec(name, shots=FIT_SHOTS):
    c = R0 if name == R0["name"] else CONFIGS[name]
    epoch_steps = c["epoch_steps"] or window_epoch_steps(shots, c["model"])
    return {"name": name, "arm": ARM, "configuration": name, "model": c["model"], "batch": BATCH,
            "epochs": c["epochs"], "epoch_steps": epoch_steps, "steps": c["epochs"] * epoch_steps,
            "static": bool(c.get("static", False)), "launch": not c.get("static", False)}


def arm_spec(arm, selected):
    if arm != ARM:
        raise KeyError(arm)
    return config_spec(selected)


def model_config(spec):
    return MODEL_CONFIGS[spec["model"]]


def build_model(spec, device="cpu"):
    return dd.build_model(model_config(spec), launch=spec["launch"], device=device)


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / name / f"seed-{seed}"


def model_dir(output, arm, seed, selected=None):
    """T2 seed 20260908 is the reused selection run of the selected configuration (#127 selection rule step 4)."""
    if arm == ARM and seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("the selected configuration is needed for the reused seed")
        return selection_dir(output, selected, seed)
    return Path(output) / "models" / arm / f"seed-{seed}"


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / arm / f"seed-{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def readiness_dir(output):
    return Path(output) / "readiness"


# ---------------------------------------------------------------------------------------------------------------
# --data: the 128 x 128 frame cache (CPU; fit / policy frames only)
# ---------------------------------------------------------------------------------------------------------------

def frame_paths(output=OUTPUT):
    return {"fit": CACHE / "fit-frames.u8", "progress": CACHE / "progress.json",
            "manifest": Path(output) / "data" / "frames-manifest.json",
            "policy": Path(output) / "data" / "policy-frames.pt",
            "fit_anchors": Path(output) / "data" / "fit-anchor-frames.pt"}


def fit_shape(n):
    return (n, len(dd.POSITIONS), 3, dd.IMAGE, dd.IMAGE)


def open_fit_frames(n, mode="r", path=None):
    return np.memmap(path or frame_paths()["fit"], dtype=np.uint8, mode=mode, shape=fit_shape(n))


_WORKER = {}


def _shot_frames(args):
    """Worker: uint8 [k, 46, 3, 128, 128] of fit shots [start, stop) (the #123 fit pixel store, dd.resize_frames)."""
    start, stop, ends = args
    if "data" not in _WORKER:
        from scripts import run_hjepa_comparison as rc
        torch.set_num_threads(1)
        _WORKER["data"] = rc.FitData(rc.OUTPUT)
    data = _WORKER["data"]
    out = np.empty((stop - start, len(dd.POSITIONS), 3, dd.IMAGE, dd.IMAGE), dtype=np.uint8)
    for i, shot in enumerate(range(start, stop)):
        frames = torch.stack([data.frame(shot, p) for p in dd.frame_positions(ends[i])])
        out[i] = dd.resize_frames(frames).numpy()
    return start, stop, out


def build_data(output=OUTPUT, workers=4, chunk=64):
    """Resumable per chunk; then the policy / fit-anchor frames and frames-manifest.json (sha256 of every file)."""
    import multiprocessing as mp
    paths = frame_paths(output)
    shots = eb.FitShots()
    n = len(shots)
    CACHE.mkdir(parents=True, exist_ok=True)
    progress = eb.read_json(paths["progress"]) if paths["progress"].is_file() else {"done": [], "n": n}
    if progress["n"] != n:
        raise ValueError("the fit shot count changed since the cache was started")
    mm = open_fit_frames(n, "r+" if paths["fit"].is_file() else "w+")
    done = set(progress["done"])
    ends = np.asarray(shots.ends)
    jobs = [(s, min(n, s + chunk), ends[s:min(n, s + chunk)].tolist()) for s in range(0, n, chunk) if s not in done]
    began = time.monotonic()
    if jobs:
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, (start, stop, arr) in enumerate(pool.imap_unordered(_shot_frames, jobs)):
                mm[start:stop] = arr
                mm.flush()
                done.add(start)
                progress["done"] = sorted(done)
                eb.write_json(paths["progress"], progress)
                if k % 20 == 0:
                    log(f"data: {len(done) * chunk}/{n} shots ({time.monotonic() - began:.0f}s)")
    del mm
    from scripts import run_hjepa_comparison as rc
    pa = eb.policy_anchor_frames()
    cells = eb.policy_cells(GRID)
    pshots = eb.policy_shots()
    store = rc.PixelStore(rc.OUTPUT, "policy")
    keys = sorted({item["key"] for entry in cells.values() for item in entry["inventory"]})
    endpoints = {k: dd.resize_frames(store.frame(store.row[k], min(dd.ENDPOINT, int(pshots[k]["end"])))) for k in keys}
    eb.atomic_torch(paths["policy"], {"schema": "issue_135_policy_frames_v1",
                                      "anchors": {m: dd.resize_frames(f) for m, f in sorted(pa.items())},
                                      "endpoints": endpoints,
                                      "rule": "policy anchor frame 0; true endpoint min(225, end), #123 policy store"})
    fa = eb.fit_anchor_frames()
    eb.atomic_torch(paths["fit_anchors"], {"schema": "issue_135_fit_anchor_frames_v1",
                                           "frames": {m: dd.resize_frames(f) for m, f in sorted(fa.items())}})
    manifest = {"schema": SCHEMA_FRAMES, "written_at": eb.utc_now(), "shots": n, "shape": list(fit_shape(n)),
                "positions": list(dd.POSITIONS), "resize": "cv2 INTER_AREA 320x240 -> 128x128 (dd.resize_frames)",
                "keys_sha256": "sha256:" + __import__("hashlib").sha256("\n".join(shots.keys).encode()).hexdigest(),
                "files": {name: {"path": str(paths[name]), "bytes": paths[name].stat().st_size,
                                 "sha256": eb.sha256_of(paths[name])} for name in ("fit", "policy", "fit_anchors")},
                "cpu_seconds": time.monotonic() - began}
    eb.write_json(paths["manifest"], manifest)
    log(f"data: frame cache complete ({n} shots, {manifest['files']['fit']['bytes'] / 1e9:.1f} GB)")
    return 0


def verify_frames(output=OUTPUT, hash_files=False):
    paths = frame_paths(output)
    manifest = eb.read_json(paths["manifest"])
    for name, entry in manifest["files"].items():
        p = Path(entry["path"])
        if p.stat().st_size != entry["bytes"] or (hash_files and eb.sha256_of(p) != entry["sha256"]):
            raise ValueError(f"frame cache {name} differs from frames-manifest.json")
    return manifest


def load_data(output=OUTPUT, shots=None):
    """{"frames": memmap [N, 46, 3, 128, 128], "launch5": [N, 5], "shots": eb.FitShots}."""
    shots = shots or eb.FitShots()
    verify_frames(output)
    return {"frames": open_fit_frames(len(shots)), "launch5": torch.as_tensor(shots.launch5, dtype=torch.float32),
            "shots": shots}


def load_policy_frames(output=OUTPUT):
    return torch.load(frame_paths(output)["policy"], map_location="cpu", weights_only=True)


def load_fit_anchor_frames(output=OUTPUT):
    return torch.load(frame_paths(output)["fit_anchors"], map_location="cpu", weights_only=True)["frames"]


# ---------------------------------------------------------------------------------------------------------------
# training windows (the official shuffled loader over every (shot, offset) window)
# ---------------------------------------------------------------------------------------------------------------

class Windows:
    """Batch ``step`` = windows [i * B, (i + 1) * B) of data-epoch e's permutation (eb.batch_rng(seed, 1e9 + e))."""

    def __init__(self, frames, launch5, seed, config, batch=BATCH):
        self.frames, self.launch5, self.seed, self.batch = frames, launch5, seed, batch
        self.horizon = config["timestep_horizon"] + 1
        self.per_shot = dd.windows_per_episode(config)
        self.n = len(launch5) * self.per_shot
        self.steps_per_epoch = self.n // batch
        self._perm = (None, None)

    def indices(self, step):
        e, i = divmod(step, self.steps_per_epoch)
        if self._perm[0] != e:
            self._perm = (e, eb.batch_rng(self.seed, 10 ** 9 + e).permutation(self.n))
        return self._perm[1][i * self.batch:(i + 1) * self.batch]

    def batch_of(self, step, device="cpu"):
        idx = self.indices(step)
        shots, offsets = idx // self.per_shot, idx % self.per_shot
        x = np.stack([self.frames[s][list(dd.EPISODE_FRAMES[o:o + self.horizon])] for s, o in zip(shots, offsets)])
        x = torch.from_numpy(x).to(device).float() / 255.0
        return x, self.launch5[torch.as_tensor(shots)].to(device), shots, offsets


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; nonfinite loss -> typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def _rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one model (resumable from out_dir/resume.pt). Batch b = Windows(seed).batch_of(b); the official step
    (dd.train_step, or dd.train_step_static for R0); StepLR at every epoch end; nonfinite loss -> typed failure."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    cfg = model_config(spec)
    torch.manual_seed(seed)
    model = build_model(spec, device)
    optimizer, scheduler = dd.build_optimizer(model, cfg)
    recon = dd.recon_loss_func(cfg, device)
    windows = Windows(data["frames"], data["launch5"], seed, cfg)
    epoch_steps, steps = spec["epoch_steps"], spec["steps"]
    start, elapsed, trace = 0, 0.0, []
    resume = out_dir / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, elapsed, trace = state["step"], state["elapsed"], state["trace"]
        _set_rng_state(state["rng"])
        log(f"{spec['name']} seed {seed}: resumed at step {start}")

    def save_resume(step):
        eb.atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict(), "step": step, "rng": _rng_state(),
                                 "elapsed": elapsed + time.monotonic() - began, "trace": trace})

    began = time.monotonic()
    failure, step = None, start
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    model.train()
    while step < stop_at:
        if eb.STOP_REQUESTED:
            save_resume(step)
            raise eb.TrainingPaused(f"{spec['name']} seed {seed} paused at step {step}")
        x, launch5, _, _ = windows.batch_of(step, device)
        epoch = step // epoch_steps
        if spec["static"]:
            terms = dd.train_step_static(model, optimizer, x, epoch, cfg, recon)
        else:
            terms = dd.train_step(model, optimizer, x, launch5, epoch, step, epoch_steps, cfg, recon)
        if not math.isfinite(terms["loss"]):
            failure = {"typed_failure": "nonfinite_loss", "step": step, "terms": terms}
            break
        step += 1
        if step % epoch_steps == 0:
            scheduler.step()
        if step % LOG_EVERY == 0 or step == steps:
            trace.append({"step": step, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **terms})
            log(f"{spec['name']} seed {seed}: step {step}/{steps} loss {terms['loss']:.3f} "
                f"({(elapsed + time.monotonic() - began) / max(1, step - 0):.3f} s/step)")
        if step % CHECKPOINT_EVERY == 0 and step < steps:
            save_resume(step)
    seconds = elapsed + time.monotonic() - began
    if failure is None and step < steps:  # max_steps reached (tests, timing)
        save_resume(step)
        return {"partial": True, "steps_done": step, "seconds": seconds}
    complete = failure is None
    precision = {**TRAINING_PRECISION, "device": device, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": failure,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "parameters": dd.trainable_parameters(model), "precision": precision}
    eb.atomic_torch(final, payload)
    report = {"spec": spec, "seed": seed, "steps": steps, "steps_done": step, "complete": complete,
              "typed_failure": failure, "wall_seconds": seconds,
              "gpu_seconds": seconds if device.startswith("cuda") else 0.0, "device": device,
              "seconds_per_step": seconds / max(1, step),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": trace[-1] if trace else None, "precision": precision, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if resume.is_file():
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
    model = build_model(payload["spec"], device_of(device))
    model.load_state_dict(payload["state_dict"])
    return model.eval().requires_grad_(False), payload


def run_queue(output, items, device=None, data=None):
    """Train (spec, seed, out_dir) items in order; a stop signal pauses the queue (exit 75)."""
    eb.install_stop_handlers()
    for spec, seed, out_dir in items:
        if (Path(out_dir) / "model.pt").is_file():
            continue
        data = data or load_data(output)
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
                report=rel(Path(out_dir, "report.json")), complete=report["complete"])
    return 0


# ---------------------------------------------------------------------------------------------------------------
# particles, readout, rollouts, costs
# ---------------------------------------------------------------------------------------------------------------

def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


def model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def frame_tokens(model, frames_u8, chunk=ENCODE_CHUNK):
    """(tokens [N, K, F], bg [N, d]) of single frames uint8 [N, 3, 128, 128] (dd.frame_particles), CPU float32."""
    device = model_device(model)
    toks, bgs = [], []
    for i in range(0, len(frames_u8), chunk):
        p = dd.frame_particles(model, dd.to_float(torch.as_tensor(frames_u8[i:i + chunk]).to(device)))
        t, b = dd.particle_tokens(p)
        toks.append(t.float().cpu())
        bgs.append(b.float().cpu())
    return torch.cat(toks), torch.cat(bgs)


def readout_items(shots):
    """(shot, cache index, position) of every fit readout frame: positions 0, 15, ..., 225 clamped to the shot's last
    frame (deduplicated); cache index j stores frame min(5 j, end)."""
    out = []
    for s, end in enumerate(np.asarray(shots.ends)):
        seen = set()
        for p in READOUT_POSITIONS:
            q = min(p, int(end))
            if q in seen:
                continue
            seen.add(q)
            j = q // dd.STRIDE if p <= end else math.ceil(int(end) / dd.STRIDE)
            out.append((s, j, q))
    return out


def fit_readout_tokens(model, data, chunk=1024):
    shots = data["shots"]
    items = readout_items(shots)
    toks, bgs = [], []
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        frames = np.stack([data["frames"][s][j] for s, j, _ in part])
        t, b = frame_tokens(model, frames)
        toks.append(t)
        bgs.append(b)
    targets = shots.carrier_rows([s for s, _, _ in items], [q for _, _, q in items])
    return torch.cat(toks), torch.cat(bgs), torch.as_tensor(np.asarray(targets), dtype=torch.float32), len(items)


def fit_readout(model, data, seed=eb.SELECTION_SEED):
    """ParticleReadout on the fit readout frames (fit split only; deterministic scoring policy is the caller's)."""
    device = model_device(model)
    tokens, bg, targets, n = fit_readout_tokens(model, data)
    probe, report = dd.fit_readout(tokens, bg, targets, seed, device=str(device), log=log)
    report.update(fit_frames=n, fit_shots=len(data["shots"]))
    return probe, report


def save_readout(path, probe, report, model_sha256):
    eb.atomic_torch(path, {"schema": SCHEMA_READOUT, "state": {k: v.cpu() for k, v in probe.state_dict().items()},
                           "token_dim": int(probe.embed[0].in_features), "bg_dim": int(probe.embed_bg.in_features),
                           "fit": report, "model_sha256": model_sha256})


def load_readout(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_READOUT:
        raise ValueError(f"{path}: schema differs")
    probe = dd.ParticleReadout(payload["token_dim"], payload["bg_dim"])
    probe.load_state_dict(payload["state"])
    return probe.to(device).eval().requires_grad_(False), payload


@torch.no_grad()
def read_carriers(probe, tokens, bg):
    device = next(probe.parameters()).device
    return dd.apply_readout(probe, tokens, bg, device=device)


@torch.no_grad()
def predict(model, probe, frames_u8, launch5, keep=(dd.STEPS,), chunk=ROLLOUT_CHUNK):
    """Rollouts of anchors frames_u8 [L, 3, 128, 128] under launches [L, K, 5] (K candidates per anchor):
    {"anchor": readout carriers of the decision frames [L, 236], "carriers": {step: [L, K, 236]}} (CPU)."""
    device = model_device(model)
    n_anchor, k = launch5.shape[:2]
    per = max(1, chunk // k)
    anchors, kept = [], {s: [] for s in keep}
    for i in range(0, n_anchor, per):
        frames = dd.to_float(torch.as_tensor(frames_u8[i:i + per]).to(device))
        enc = dd.encode_context(model, dd.context(frames))
        t0, b0 = dd.particle_tokens(enc, t=-1)
        anchors.append(read_carriers(probe, t0, b0))
        lp = launch5[i:i + per].reshape(-1, launch5.shape[-1]).to(device)
        out = dd.rollout_particles(model, dd.expand_particles(enc, k), lp, steps=max(keep))
        for s in keep:
            t, b = dd.particle_tokens(out, t=dd.STATIC - 1 + s)
            kept[s].append(read_carriers(probe, t, b).reshape(-1, k, dd.CARRIER_DIM))
    return {"anchor": torch.cat(anchors), "carriers": {s: torch.cat(v) for s, v in kept.items()}}


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


@torch.no_grad()
def level_rows(model, probe, items, frame_u8, costs):
    """Rows of one level's candidates from its decision frame: tie-free cost of readout(particles at frame 225)
    against readout(particles of the decision frame) (the primary scoring path, shared by the policy AUCs, --score,
    --score-novelty and --validate)."""
    actions = torch.stack([eb.launch5(it["action"]) for it in items])
    out = predict(model, probe, torch.as_tensor(frame_u8)[None], actions[None])
    return eb.endpoint_rows(items, out["carriers"][dd.STEPS][0], out["anchor"][0], costs)


@torch.no_grad()
def fit_anchor_predictions(model, probe, output=OUTPUT):
    """Rollouts of the 16 grid launches from the 480 fit anchor frames (G-rank, G-action)."""
    frames = load_fit_anchor_frames(output)
    members = sorted(frames)
    _, actions = grid_actions()
    out = predict(model, probe, torch.stack([frames[m] for m in members]),
                  actions[None].expand(len(members), -1, -1))
    return members, out["anchor"], out["carriers"][dd.STEPS]


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
def policy_endpoint_rows(model, probe, costs, pframes):
    """{member: rows} of the policy grid mixed cells (predicted endpoints)."""
    return {member: level_rows(model, probe, entry["inventory"], pframes["anchors"][member], costs)
            for member, entry in sorted(eb.policy_cells(GRID).items())}


@torch.no_grad()
def policy_ceiling(model, probe, costs, pframes):
    """G-readout: pooled policy grid AUC of the readout of the TRUE endpoint frames (#123 / #130 ceiling) + the
    readout's error against the E99 carriers of the same frames."""
    cells = eb.policy_cells(GRID)
    members = sorted(cells)
    a_tok, a_bg = frame_tokens(model, torch.stack([pframes["anchors"][m] for m in members]))
    c0 = dict(zip(members, read_carriers(probe, a_tok, a_bg)))
    flat = [(m, item) for m in members for item in cells[m]["inventory"]]
    e_tok, e_bg = frame_tokens(model, torch.stack([pframes["endpoints"][item["key"]] for _, item in flat]))
    c_true = read_carriers(probe, e_tok, e_bg)
    shots = eb.policy_shots()
    e99 = torch.stack([shots[item["key"]]["carriers"][(shots[item["key"]]["positions"] ==
                                                       min(dd.ENDPOINT, int(shots[item["key"]]["end"]))).nonzero()[0, 0]]
                       for _, item in flat]).float()
    rows, k = {}, 0
    for m in members:
        inv = cells[m]["inventory"]
        rows[m] = eb.endpoint_rows(inv, c_true[k:k + len(inv)], c0[m], costs)
        k += len(inv)
    presence, center = costs.presence, list(costs.center)
    return {"ceiling_auc": eb.policy_grid_auc(rows, eb.PRIMARY_COST), "policy_grid_cells": len(members),
            "policy_candidates": len(flat),
            "readout_error_vs_e99_on_policy_endpoints": {
                "pig_presence_mae": float((c_true[:, presence] - e99[:, presence]).abs().mean()),
                "pig_center_mae": float((c_true[:, center] - e99[:, center]).abs().mean()),
                "carrier_mae": float((c_true - e99).abs().mean())}}


def e99_true_ceiling(costs):
    """The E99 true-carrier ceiling on the same policy cells (#130 reference 0.7223)."""
    cells = eb.policy_cells(GRID)
    anchors = eb.policy_anchors()["carriers"]
    shots = eb.policy_shots()
    rows = {}
    for m, entry in sorted(cells.items()):
        ends = torch.stack([shots[it["key"]]["carriers"][(shots[it["key"]]["positions"] ==
                                                          min(dd.ENDPOINT, int(shots[it["key"]]["end"]))).nonzero()[0, 0]]
                            for it in entry["inventory"]]).float()
        rows[m] = eb.endpoint_rows(entry["inventory"], ends, anchors[m].float(), costs)
    return eb.policy_grid_auc(rows, eb.PRIMARY_COST)


def g_action_calibration(value):
    """#127 decision (issuecomment-6396584235): the carrier-space G-action beside the frozen guard (descriptive)."""
    if not CALIBRATION.is_file():
        return {"arm": value, "source": None}
    cal = eb.read_json(CALIBRATION)
    truth = cal["means"]["engine_truth"]
    return {"arm": value, "engine_truth": truth, "fraction_of_engine_truth": None if value is None else value / truth,
            "U": cal["means"]["U"], "C*": cal["means"]["C*"], "B1": cal["means"]["B1"],
            "threshold_frozen": cal["threshold_frozen"], "threshold_attainable_by_a_perfect_model":
                cal["threshold_attainable_by_a_perfect_model"], "source": rel(CALIBRATION),
            "source_sha256": eb.sha256_of(CALIBRATION), "label": "DESCRIPTIVE (no guard change)"}


# ---------------------------------------------------------------------------------------------------------------
# gates (#135)
# ---------------------------------------------------------------------------------------------------------------

def funded(output):
    """The orchestrator's funding marker (Decision P-budget funding): present, a JSON object, not ``funded: false``."""
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


def readiness_pass(output):
    path = readiness_dir(output) / "readiness.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def refuse_unfunded(output, mode):
    if funded(output):
        return False
    log(f"{mode} refused: funded.json absent (no posted funding decision); no GPU work")
    return True


def selection_gate(output):
    """Funding marker, G-port, G-fid (PHYRE-128 Tier E) and the readiness stop (validation ladder)."""
    out = {"funded.json": funded(output), "gport/gport.json": gport_pass(output),
           "fidelity/gfid.json": gfid_pass(output), "readiness/readiness.json": readiness_pass(output)}
    out["pass"] = all(out.values())
    return out


# ---------------------------------------------------------------------------------------------------------------
# --readiness (R0 static DLPv2 -> readout -> policy ceiling; pre-registered stop)
# ---------------------------------------------------------------------------------------------------------------

def readiness(output, device=None):
    output = Path(output)
    if refuse_unfunded(output, "readiness"):
        return 1
    if not gport_pass(output):
        log("readiness refused: G-port has not passed")
        return 1
    rdir = readiness_dir(output)
    if (rdir / "readiness.json").is_file():
        log("readiness.json exists (one run, never re-run)")
        return 0
    device = device_of(device)
    data = load_data(output)
    spec = config_spec(R0["name"])
    code = run_queue(output, [(spec, eb.SELECTION_SEED, rdir / "r0")], device, data)
    if code != 0:
        return code
    began = time.monotonic()
    costs = objective_costs()
    model, payload = load_trained(rdir / "r0" / "model.pt", device)
    with harness.deterministic_scoring() as policy:
        if model is None:
            ceiling = {"ceiling_auc": None}
            report = {"typed_failure": payload["typed_failure"]}
        else:
            probe, fit_report = fit_readout(model, data)
            save_readout(rdir / "readout.pt", probe, fit_report, eb.sha256_of(rdir / "r0" / "model.pt"))
            ceiling = policy_ceiling(model, probe, costs, load_policy_frames(output))
            report = {"readout": {"artifact": rel(rdir / "readout.pt"), "sha256": eb.sha256_of(rdir / "readout.pt"),
                                  **fit_report}}
        e99 = e99_true_ceiling(costs)
    value = ceiling["ceiling_auc"]
    passed = value is not None and value >= THRESHOLD
    record = {"schema": SCHEMA_READINESS, "arm": ARM, "written_at": eb.utc_now(), "scoring_policy": policy,
              "threshold": THRESHOLD, **ceiling, "pass": bool(passed), "typed_reason": None if passed else TYPED_STOP,
              "consequence": "T2 proceeds to selection" if passed else "T2 stops before any DDLP training (typed)",
              "e99_true_carrier_ceiling_auc": e99, "r0": {"spec": spec, "model": rel(rdir / "r0" / "model.pt"),
                                                          "model_sha256": eb.sha256_of(rdir / "r0" / "model.pt"),
                                                          "report": eb.read_json(rdir / "r0" / "report.json")},
              **report, "decision": DECISIONS["readiness"],
              "data": {"readout_fit": "fit split only: every fit shot at 0, 15, ..., 225 (clamped), E99 targets",
                       "ceiling": "policy split: eb.policy_cells('grid') mixed cells; anchor frame 0; true endpoint "
                                  "min(225, end) from the #123 policy pixel store", "evaluation": "not read"},
              "seconds": time.monotonic() - began}
    eb.write_json(rdir / "readiness.json", record)
    eb.mark(output, "readiness-ceiling", record["seconds"], record["seconds"] if device != "cpu" else 0)
    log(f"readiness: ceiling {value} (threshold {THRESHOLD}) -> {'PASS' if passed else TYPED_STOP}; E99 {e99}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def selection(output, queue, device=None):
    gate = selection_gate(output)
    if not gate["pass"]:
        log(f"selection refused: gate {gate}; nothing trained")
        return 1
    names = SELECTION_QUEUES[queue]
    items = [(config_spec(name), eb.SELECTION_SEED, selection_dir(output, name)) for name in names]
    log(f"selection queue {queue}: {names}")
    return run_queue(output, items, device)


def readout_for(output, model, model_path, out_path, data):
    """The run's readout (fit split only): reused when out_path exists for the same model, else fitted and saved."""
    sha = eb.sha256_of(model_path)
    if Path(out_path).is_file():
        probe, payload = load_readout(out_path, model_device(model))
        if payload["model_sha256"] == sha:
            return probe, payload["fit"]
    probe, report = fit_readout(model, data)
    save_readout(out_path, probe, report, sha)
    return probe, report


def selection_readouts(output, device=None):
    """Per configuration: readout (fit frames), policy grid AUC of the primary score and the readout ceiling
    (deterministic scoring); selection.json."""
    output = Path(output)
    if refuse_unfunded(output, "selection-readouts"):
        return 1
    costs = objective_costs()
    results, details = {}, {}
    data = load_data(output)
    pframes = load_policy_frames(output)
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
            probe, fit_report = readout_for(output, model, path, selection_dir(output, name) / "readout.pt", data)
            rows = policy_endpoint_rows(model, probe, costs, pframes)
            results[name] = {"auc": eb.policy_grid_auc(rows, eb.PRIMARY_COST), "parameters": payload["parameters"]}
            details[name] = {"typed_failure": None, "model_sha256": eb.sha256_of(path), "cells": len(rows),
                             "readout": fit_report, **policy_ceiling(model, probe, costs, pframes)}
    choice = eb.select_configuration(results)
    seconds = time.monotonic() - began
    eb.mark(output, "selection-readouts", seconds, seconds if device_of(device) != "cpu" else 0)
    eb.write_json(output / "selection.json", {
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
# --project (CPU, [INFERENCE]) / --timing (GPU, measured) / --smoke
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
    return [(ARM, seed, arm_spec(ARM, selected), model_dir(output, ARM, seed)) for seed in eb.SEEDS
            if seed != eb.SELECTION_SEED]


def child_projection(rates, extra_gpu_h, names=None):
    """GPU-h of the child from per-configuration GPU s/step: selection runs of ``names`` (default all three), two
    more seeds at the most expensive of them (unknown before selection), R0 and readouts / scoring (extra_gpu_h)."""
    names = list(CONFIGS) if names is None else names
    steps = {n: config_spec(n)["steps"] for n in names}
    selection_s = {n: steps[n] * rates[n] for n in names}
    per_seed = max(selection_s.values())
    training = [(f"{ARM}--seed{s}", per_seed) for s in eb.SEEDS if s != eb.SELECTION_SEED]
    queues, loads = lpt(training)
    total = sum(selection_s.values()) + sum(s for _, s in training) + extra_gpu_h * 3600
    wall = max(selection_s.values()) + max(loads) + extra_gpu_h * 3600
    return {"configurations": names, "steps": steps, "selection_gpu_h": {k: v / 3600 for k, v in selection_s.items()},
            "training_gpu_h": sum(s for _, s in training) / 3600, "other_gpu_h": extra_gpu_h,
            "total_gpu_h": total / 3600, "native_per_seed_gpu_h_max": per_seed / 3600,
            "wall_h_on_4_gpus_serial_ladder": wall / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
            "wall_cap_h": WALL_CAP_HOURS,
            "within_caps": bool(total / 3600 <= CHILD_CAP_GPU_HOURS and wall / 3600 <= WALL_CAP_HOURS)}


def _step_flops(spec):
    from torch.utils.flop_counter import FlopCounterMode
    cfg = model_config(spec)
    model = build_model(spec, "cpu")
    optimizer, _ = dd.build_optimizer(model, cfg)
    x = torch.rand(3, cfg["timestep_horizon"] + 1, 3, dd.IMAGE, dd.IMAGE)
    with FlopCounterMode(display=False) as fc:
        if spec["static"]:
            dd.train_step_static(model, optimizer, x, 2, cfg)
        else:
            dd.train_step(model, optimizer, x, torch.rand(3, 5), 2, 100, 1000, cfg)
    return fc.get_total_flops() * BATCH / 3


def project(output, tflops=(10.0, 40.0)):
    """projection.json: [INFERENCE] FLOP model (FlopCounterMode per batch-12 step / an assumed sustained 10-40 TFLOP/s
    on one RTX 5090); when timing.json (GPU) exists, the measured projection is primary."""
    output = Path(output)
    flops = {name: _step_flops(config_spec(name)) for name in CONFIGS}
    flops[R0["name"]] = _step_flops(config_spec(R0["name"]))
    result = {"schema": SCHEMA_PROJECTION, "plan_identity": IDENTITY, "label": "[INFERENCE] FLOP model",
              "step_tflop_batch12": {k: v / 1e12 for k, v in flops.items()}, "assumed_sustained_tflops": list(tflops),
              "measured_at": eb.utc_now()}
    for tag, rate in (("upper", tflops[0]), ("lower", tflops[1])):
        rates = {k: v / (rate * 1e12) for k, v in flops.items()}
        r0 = config_spec(R0["name"])["steps"] * rates[R0["name"]] / 3600
        result[f"projection_{tag}"] = {"seconds_per_step": rates, "r0_gpu_h": r0,
                                       "three": child_projection(rates, r0 + 6.0),
                                       "two_after_cut": child_projection(rates, r0 + 6.0,
                                                                         [n for n in CONFIGS if n not in CUT_ORDER])}
    result["primary"] = "projection_upper"
    timing_path = output / "timing.json"
    if timing_path.is_file():
        record = eb.read_json(timing_path)
        if str(record.get("device", "")).startswith("cuda"):
            result["projection_measured"] = measured_projection(record)
            result["timing_sha256"] = eb.sha256_of(timing_path)
            result["primary"] = "projection_measured"
    eb.write_json(output / "projection.json", result)
    up, lo = result["projection_upper"]["three"], result["projection_lower"]["three"]
    log(f"projection [INFERENCE]: three configurations {lo['total_gpu_h']:.0f}-{up['total_gpu_h']:.0f} GPU-h "
        f"(cap {CHILD_CAP_GPU_HOURS}); primary {result['primary']}")
    return 0


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _time_steps(spec, data, steps, device, chunk=50):
    """``steps`` official training steps of ``spec`` on the fit windows (timed per chunk, CUDA synchronised)."""
    cfg = model_config(spec)
    torch.manual_seed(eb.SELECTION_SEED)
    model = build_model(spec, device)
    optimizer, _ = dd.build_optimizer(model, cfg)
    windows = Windows(data["frames"], data["launch5"], eb.SELECTION_SEED, cfg)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    chunks, load_s, losses = [], 0.0, []
    start = time.monotonic()
    for step in range(steps):
        t = time.monotonic()
        x, l5, _, _ = windows.batch_of(step, device)
        load_s += time.monotonic() - t
        if spec["static"]:
            terms = dd.train_step_static(model, optimizer, x, 2, cfg)
        else:
            terms = dd.train_step(model, optimizer, x, l5, 2, 20_000 + step, spec["epoch_steps"], cfg)
        losses.append(terms["loss"])
        if (step + 1) % chunk == 0 or step + 1 == steps:
            _sync(device)
            now = time.monotonic()
            chunks.append({"steps": (step % chunk) + 1, "seconds": now - start})
            start = now
    steady = [c["seconds"] / c["steps"] for c in chunks[1:]] or [c["seconds"] / c["steps"] for c in chunks]
    out = {"steps": steps, "seconds_per_step_steady_median": float(np.median(steady)), "chunks": chunks,
           "batch_load_seconds_per_step": load_s / steps, "finite": bool(np.all(np.isfinite(losses))),
           "parameters": dd.trainable_parameters(model)}
    if str(device).startswith("cuda"):
        out["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    return model, out


def timing(output, steps=300, device=None):
    """--timing probe (<= 0.5 GPU-h; enqueued only after its own posted decision; no funded.json needed): ``steps``
    official steps of each configuration and of R0 on the real fit windows, the rollout of one decision (16
    candidates) and the frame-encoding throughput. Writes timing.json + ledger; keeps no model."""
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    data = load_data(output)
    result = {"schema": SCHEMA_TIMING, "plan_identity": IDENTITY, "device": device,
              "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
              "torch": torch.__version__, "steps": steps, "configs": {}}
    for name in [*CONFIGS, R0["name"]]:
        model, out = _time_steps(config_spec(name), data, steps, device)
        result["configs"][name] = out
        if name == "native":
            model.eval()
            probe = dd.ParticleReadout(6 + model.learned_feature_dim, model.bg_learned_feature_dim).to(device)
            probe.eval().requires_grad_(False)
            _, actions = grid_actions()
            frames = torch.as_tensor(np.stack([data["frames"][s][0] for s in range(8)]))
            _sync(device)
            t = time.monotonic()
            predict(model, probe, frames, actions[None].expand(8, -1, -1))
            _sync(device)
            result["rollout_seconds_per_decision"] = (time.monotonic() - t) / 8
            t = time.monotonic()
            frame_tokens(model, torch.as_tensor(np.stack([data["frames"][s][3] for s in range(512)])))
            _sync(device)
            result["encode_seconds_per_frame"] = (time.monotonic() - t) / 512
            result["readout_fit_frames"] = len(readout_items(data["shots"]))
        del model
    result["seconds"] = time.monotonic() - began
    result["written_at"] = eb.utc_now()
    eb.write_json(output / "timing.json", result)
    eb.mark(output, "timing", result["seconds"], result["seconds"] if device.startswith("cuda") else 0.0)
    log("timing: " + ", ".join(f"{k} {v['seconds_per_step_steady_median']:.4f} s/step"
                               for k, v in result["configs"].items()))
    if device.startswith("cuda") and (output / "projection.json").is_file():
        projection = eb.read_json(output / "projection.json")
        projection.update(projection_measured=measured_projection(result), primary="projection_measured",
                          timing_sha256=eb.sha256_of(output / "timing.json"))
        eb.write_json(output / "projection.json", projection)
    return 0


# decisions (one anchor x its candidates) per readout pass and per scoring pass [INFERENCE]: 480 fit anchors
# (G-rank / G-action) + 94 policy cells (+ the ceiling encodings); scoring: the mixed cells of four inventories +
# 480 wrong anchors + the position branches
READOUT_DECISIONS = 480 + 94
SCORING_DECISIONS = 2000


def measured_projection(record):
    rates = {k: v["seconds_per_step_steady_median"] for k, v in record["configs"].items()}
    r0 = config_spec(R0["name"])["steps"] * rates[R0["name"]] / 3600
    encode_h = record["readout_fit_frames"] * record["encode_seconds_per_frame"] / 3600
    readout_h = encode_h + READOUT_DECISIONS * record["rollout_seconds_per_decision"] / 3600
    scoring_h = SCORING_DECISIONS * record["rollout_seconds_per_decision"] / 3600
    other = r0 + encode_h + 5 * readout_h + 3 * scoring_h  # R0 readout; 3 selection runs + 2 seeds; 3 seeds scored
    return {"seconds_per_step": rates, "r0_gpu_h": r0, "readout_gpu_h_each": readout_h,
            "scoring_gpu_h_per_seed": scoring_h, "other_gpu_h": other,
            "three": child_projection(rates, other),
            "two_after_cut": child_projection(rates, other, [n for n in CONFIGS if n not in CUT_ORDER]),
            "source": "timing.json (measured)"}


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    launch = torch.as_tensor(shots.launch5)
    accounting = {"fit_shots": len(shots), "fit_shots_expected": FIT_SHOTS, "fit_levels": len(shots.levels),
                  "fit_positive": int(shots.verdicts.sum()), "fit_anchors": len(eb.fit_anchors()["carriers"]),
                  "policy_grid_mixed_cells": len(eb.policy_cells(GRID)), "episode_frames": len(dd.EPISODE_FRAMES),
                  "windows_per_shot": per_shot(), "windows": len(shots) * per_shot(),
                  "shots_ending_before_225": int((np.asarray(shots.ends) < dd.ENDPOINT).sum()),
                  "readout_fit_frames": len(readout_items(shots)), "launch5_abs_max": float(launch.abs().max())}
    gport = eb.read_json(output / "gport/gport.json")
    adapter = eb.sha256_of(ROOT / "world_model/baselines/ddlp/novphy.py")
    gport_check = {"pass": gport.get("pass") is True, "adapter_current": gport.get("adapter") == adapter}
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
    remaining = remaining_items(output, selected)
    rate = throughput[selected]["seconds_per_step"]
    training = [(f"{arm}--seed{seed}", spec["steps"] * rate) for arm, seed, spec, _ in remaining]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values())
    readout_seconds, score_seconds = 3 * 1.0 * 3600, 2.0 * 3600  # [INFERENCE] per #135 timing
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds
    wall = max(loads) + readout_seconds / 4 + score_seconds
    projection = {"selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h_estimate": readout_seconds / 3600, "score_gpu_h_estimate": score_seconds / 3600,
                  "total_gpu_h": total / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "native_per_seed_gpu_h": arm_spec(ARM, selected)["steps"] * rate / 3600,
                  "wall_h_on_4_gpus": wall / 3600, "wall_cap_h": WALL_CAP_HOURS, "training_queues": queues,
                  "training_queue_loads_h": [x / 3600 for x in loads]}
    projection["within_caps"] = bool(total / 3600 <= CHILD_CAP_GPU_HOURS and wall / 3600 <= WALL_CAP_HOURS)
    smoke_record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
                    "gate": selection_gate(output), "tests": tests_record, "throughput": throughput,
                    "readiness": eb.read_json(readiness_dir(output) / "readiness.json"),
                    "selection": selection_record, "selected": selected, "projection": projection, "device": device,
                    "pass": bool(gport_check["pass"] and gport_check["adapter_current"] and tests.returncode == 0
                                 and projection["within_caps"] and accounting["fit_shots"] == FIT_SHOTS
                                 and readiness_pass(output) and gfid_pass(output)),
                    "written_at": eb.utc_now()}
    eb.write_json(output / "smoke.json", smoke_record)
    log(f"smoke: pass {smoke_record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h, wall "
        f"{projection['wall_h_on_4_gpus']:.1f} h; selected {selected}")
    return 0 if smoke_record["pass"] else 1


# ---------------------------------------------------------------------------------------------------------------
# --prepare (freeze) and load_plan
# ---------------------------------------------------------------------------------------------------------------

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/ddlp/__init__.py", "world_model/baselines/ddlp/novphy.py",
             "scripts/b127_ddlp_port_reference.py", "scripts/b127_ddlp_fidelity.py", "scripts/b127_ddlp_mega.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/ddlp/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "fidelity/band.json", output / "fidelity/gfid.json",
              output / "smoke.json", output / "selection.json", output / "funded.json",
              readiness_dir(output) / "readiness.json", readiness_dir(output) / "readout.pt",
              frame_paths(output)["manifest"], frame_paths(output)["policy"], frame_paths(output)["fit_anchors"]]
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json", "readout.pt")]
    return [{"name": f"t2_{rel(p)}", "artifact": rel(p), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file() and "__pycache__" not in str(p)]


def arm_seeds():
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
    readout_queues, _ = lpt([(f"{a}--seed{s}", 1.0) for a, s in arm_seeds()])
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS, "cut_order": CUT_ORDER,
        "arms": {arm: {**ARMS[arm], "spec": arm_spec(arm, selected), "seeds": list(eb.SEEDS)} for arm in ARMS},
        "training_precision": TRAINING_PRECISION, "scoring_policy": harness.SCORING_POLICY,
        "reused": {ARM: {"seed": eb.SELECTION_SEED, "run": rel(selection_dir(output, selected))}},
        "readout": {"rollout": "decision frame x4 (static burn-in) encoded as ObjectDynamicsDLP.sample does, 45 "
                               "deterministic PINT steps with the launch particle; primary = tie-free EndpointCosts "
                               "of readout(particles at frame 225) vs readout(decision-frame particles)",
                    "readout": "ParticleReadout fitted per arm-seed on fit frames 0, 15, ..., 225 (E99 targets)",
                    "g_readout": "policy ceiling on true endpoint frames >= 0.65 (mean over seeds)",
                    "action_sensitivity": "eb.action_sensitivity on readout carriers at frame 225, 480 fit anchors x "
                                          "16 grid launches; carrier-space calibration reported beside it"},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": [], "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "T2 is a triggered Tier-C arm (#127 ladder; trigger issuecomment-6396584478): secondary, reported, never "
            "in the program claim.",
            "T2's configuration was selected on the policy split among three pre-declared configurations (native, "
            "paper-schedule, repo-config); U had no such tuning (#127).",
            "The launch enters PINT as an extra particle (learned linear map of launch5), re-set after every rollout "
            "step; the static burn-in context is the decision frame x4 (the pre-launch scene is static).",
            "Deterministic rollout (posterior / PINT means); the readout is fitted on fit frames only and frozen "
            "before any evaluation read; z0 is the readout of the decision frame (the arm sees pixels only).",
            "128 x 128 full-frame resize (cv2 INTER_AREA; the interpolation the official PHYRE preparation names); "
            "beta_KL 0.15 from the released checkpoint (Table 4 prints 0.1).",
            "No native success head (DDLP has none): no native-head score and no U + head control; G-collapse not "
            "applied (explicit particles through a readout; G-readout applies).",
            "P-novelty: after the primary scoring pass, T2 (if its guards pass) is scored on the #110 novelty levels "
            "from each level's decision frame (--score-novelty); descriptive only, no tokens.",
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
    if refuse_unfunded(output, "train"):
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
        seed = int(seed)
        if name == R0["name"]:
            return readiness_dir(output) / "r0" / "report.json"
        if name in CONFIGS:
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
        result["model"] = rel(path)
        result["model_sha256"] = eb.sha256_of(path)
        if model is None:
            result.update(typed_failure=payload["typed_failure"]["typed_failure"])
        else:
            data = data or load_data(output)
            source = path.parent / "readout.pt" if seed == eb.SELECTION_SEED else out_dir / "readout.pt"
            probe, fit_report = readout_for(output, model, path, source, data)
            if source != out_dir / "readout.pt":
                (out_dir / "readout.pt").write_bytes(source.read_bytes())
            members, z0, ends = fit_anchor_predictions(model, probe, output)
            matrix = tie_free_matrix(ends, z0, costs)
            action = float(eb.action_sensitivity(ends))
            pframes = load_policy_frames(output)
            ceiling = policy_ceiling(model, probe, costs, pframes)
            items, actions = grid_actions()
            fa = load_fit_anchor_frames(output)
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=action,
                          G_action_calibration=g_action_calibration(action), G_collapse=None,
                          G_collapse_note="not applied: explicit particle state read through a readout (G-readout)",
                          policy_ceiling_auc=ceiling["ceiling_auc"], ceiling=ceiling, readout=fit_report,
                          readout_artifact=rel(out_dir / "readout.pt"),
                          readout_sha256=eb.sha256_of(out_dir / "readout.pt"),
                          fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          trainable_parameters=dd.trainable_parameters(model))
            prow = policy_endpoint_rows(model, probe, costs, pframes)
            result["policy_auc"] = eb.policy_grid_auc(prow, eb.PRIMARY_COST)
            frame0 = fa[members[0]]
            result["macs_per_candidate"] = eb.macs_per_candidate(
                lambda f, a: predict(model, probe, f, a), frame0[None], actions[None], candidates=len(items))
    seconds = time.monotonic() - began
    result.update(seconds=seconds, written_at=eb.utc_now())
    eb.mark(output, f"readouts-{arm}-seed{seed}", seconds, seconds if device != "cpu" else 0)
    eb.write_json(out_dir / "readouts.json", result)  # written last (atomic)
    log(f"readouts {arm} seed {seed}: "
        f"{json.dumps({k: result.get(k) for k in ('policy_auc', 'policy_ceiling_auc', 'G_action')})}")
    return 0


def readouts(output, queue, device=None):
    if refuse_unfunded(output, "readouts"):
        return 1
    plan = load_plan(output)
    data = None
    for name in plan["queues"]["readouts"].get(str(queue), []):
        arm, seed = name.split("--seed")
        if not (readouts_dir(output, arm, int(seed)) / "readouts.json").is_file():
            data = data or load_data(output)
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
    for arm, seed in arm_seeds():
        rdir = readouts_dir(output, arm, seed)
        if not (rdir / "readouts.json").is_file():
            raise SystemExit(f"readouts of {arm} seed {seed} missing")
        mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
        entry = {"readouts": rel(rdir / "readouts.json"), "readouts_sha256": eb.sha256_of(rdir / "readouts.json"),
                 "model": rel(mpath), "model_sha256": eb.sha256_of(mpath)}
        if (rdir / "readout.pt").is_file():
            entry.update(readout=rel(rdir / "readout.pt"), readout_sha256=eb.sha256_of(rdir / "readout.pt"))
        entries[f"{arm}--seed{seed}"] = entry
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
        for kind in ("model", "readouts", "readout"):
            if kind in entry and eb.sha256_of(ROOT / entry[kind]) != entry[f"{kind}_sha256"]:
                raise ValueError(f"{name}: frozen {kind} changed")
    return manifest


def frozen_predictor(output, plan, arm, seed, device=None):
    """(model, probe) of a frozen arm-seed (None, reason) for a typed failure."""
    model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
    if model is None:
        return None, payload["typed_failure"]["typed_failure"]
    probe, _ = load_readout(readouts_dir(output, arm, seed) / "readout.pt", model_device(model))
    return model, probe


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
    """One record: rows of every mixed cell of every inventory, wrong-anchor rows (480 fit anchor frames, grid),
    position error (descriptive), wall time per decision."""
    device = device_of(device)
    costs = objective_costs()
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed}
    model, probe = frozen_predictor(output, plan, arm, seed, device)
    if model is None:
        return {**base, "typed_failure": probe}
    frames = {m: dd.resize_frames(f) for m, f in anchors["frames"].items()}
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {member: {"rows": level_rows(model, probe, entry["inventory"], frames[member], costs)}
                                  for member, entry in sorted(cells[inventory]["mixed"].items())}
    record = {**base, "typed_failure": None, "inventories": inventories}
    fit = load_fit_anchor_frames(output)
    items, actions = grid_actions()
    record["wrong_anchor"] = {m: {"rows": level_rows(model, probe, items, fit[m], costs)} for m in sorted(fit)}
    cache = eb.position_cache()
    branches = eb.position_branches(cells)
    keep = tuple(sorted(set(POSITION_STEPS.values())))
    acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
    out = predict(model, probe, torch.stack([frames[m] for m, _ in branches]), acts[:, None], keep=keep)
    at = {t: out["carriers"][s][:, 0] for t, s in POSITION_STEPS.items()}
    record["positions"] = _position_rows(at, branches, cache, eb.POSITION_TIMES)
    members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(
        lambda m: predict(model, probe, frames[m][None], actions[None]), members)
    return record


def score(output, device=None):
    output = Path(output)
    if refuse_unfunded(output, "score"):
        return 1
    plan = load_plan(output)
    load_manifest(output)
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#135 gate); evaluation not read")
        return 1
    readouts_frozen_at = eb.read_json(output / "readouts-manifest.json")["frozen_at"]
    with harness.deterministic_scoring() as policy:
        anchors_path = eb.evaluation_anchors_path(output)
        anchors = (torch.load(anchors_path, map_location="cpu", weights_only=True) if anchors_path.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        for arm, seed in arm_seeds():
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

SCHEMA_NOVELTY_RECORD = "issue_135_novelty_record_v1"
NOVELTY_ARMS = tuple(ARMS)


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


def score_novelty(output, device=None, cells=None):
    """P-novelty: every guard-passing arm, every seed, on the mixed novelty levels (grid) from each level's decision
    frame (the primary scoring path, level_rows); refuses before the primary records and the readouts freeze exist;
    never touches a primary record."""
    output = Path(output)
    if refuse_unfunded(output, "score-novelty"):
        return 1
    plan = load_plan(output)
    load_manifest(output)
    missing = [f"{a}--seed{s}" for a, s in arm_seeds() if not record_path(output, a, s).is_file()]
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
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_135_novelty_arms_v1", "arms": status,
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
                model, probe = frozen_predictor(output, plan, arm, seed, device)
                grid = {level: {"rows": level_rows(model, probe, cells["levels"][level]["inventory"],
                                                   dd.resize_frames(anchors["frames"][level]), costs)}
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
    refs = eb.novelty_reference_records(output, None, cells)
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
        "G_isolation": {"pass": True, "rule": "fit split trains and fits the readout, policy split selects and gives "
                        "the readout ceilings; evaluation anchors are read only by --score after --freeze-readouts "
                        "(G_chronology) and after G-fid passed"},
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
            per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": rd.get("G_action"), "G_collapse": None,
                              "policy_ceiling_auc": rd.get("policy_ceiling_auc"),
                              "typed_failure": rd.get("typed_failure") or records[arm][seed].get("typed_failure")}
        guards_of[arm] = eb.arm_guard_block(per_seed, ARM_CHECKS,
                                            nonfinite=eb.nonfinite_share(records[arm].values(), eb.PRIMARY_COST))
    return guards_of


def arm_guard_pass(guards, extra_guards):
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
    gfid_block = {ARM: {"pass": extra_guards["G_fid"], "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output, plan))
    readouts_view = {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "policy_ceiling_auc", "G_action", "G_action_calibration",
                                                "typed_failure")} for a, s in arm_seeds()}
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, {}, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": positions, "selection": plan["selected"],
        "tier_c_arm": True, "readiness": eb.read_json(readiness_dir(output) / "readiness.json"),
        "readouts": readouts_view, "novelty": novelty_section(output, records),
        "native_success_head": None, "u_plus_head": "not applicable: DDLP has no success head"})
    compute = {"schema": "issue_135_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} T2 DDLP (triggered Tier-C arm): findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configuration "
             f"{summary['extra']['selection']}.", "", *eb.format_arm_rows(summary["arms"]), "",
             "## G-action in carrier space (DESCRIPTIVE, #127 issuecomment-6396584235)", ""]
    for name, rd in summary["extra"]["readouts"].items():
        cal = rd.get("G_action_calibration") or {}
        lines.append(f"- {name}: G-action {rd.get('G_action')}; engine truth {cal.get('engine_truth')}; "
                     f"fraction {cal.get('fraction_of_engine_truth')}")
    lines += ["", "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    novelty = summary["extra"]["novelty"]
    if novelty is not None:
        lines += ["", f"## Novelty pairs ({novelty['table']['label']})", "",
                  f"Not scored (guards): {novelty['not_scored'] or 'none'}; novelty anchors read after the primary "
                  f"records: {novelty['chronology']['pass']}.", "", *eb.format_novelty_rows(novelty["table"])]
    lines += ["", "T2 is a triggered Tier-C arm: never in the program claim. Intervals are descriptive "
              "(member-clustered bootstrap); tokens follow the #127 delta rule.", ""]
    return "\n".join(lines)


def not_run_reason(output):
    """The typed reason T2 is not run (no plan.json): the readiness stop, or the orchestrator's not-run.json marker
    (e.g. {"typed_reason": "cap", "decision": url}); None while T2 may still run."""
    output = Path(output)
    rpath = readiness_dir(output) / "readiness.json"
    if rpath.is_file() and not eb.read_json(rpath).get("pass"):
        return eb.read_json(rpath).get("typed_reason") or TYPED_STOP
    marker = output / "not-run.json"
    if marker.is_file():
        return eb.read_json(marker)["typed_reason"]
    return None


def publish_not_run(output, reason):
    """summary.json of a T2 that was not run (typed): no arm block, extra.pending_arms {T2: reason} (listed as not
    run by the program summary), the G-port / G-fid / readiness records that exist."""
    output = Path(output)
    gfid_path = output / "fidelity/gfid.json"
    rpath = readiness_dir(output) / "readiness.json"
    gfid_block = {ARM: {"pass": gfid_pass(output), "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    summary = eb.child_summary(ISSUE, IDENTITY, None, {}, {}, {}, gfid_block, {
        "pending_arms": {ARM: reason}, "tier_c_arm": True, "G_port": gport_pass(output),
        "readiness": eb.read_json(rpath) if rpath.is_file() else None,
        "not_run_marker": eb.read_json(output / "not-run.json") if (output / "not-run.json").is_file() else None,
        "decisions": DECISIONS})
    eb.write_json(output / "summary.json", summary)
    (output / "findings.md").write_text(
        f"# #{ISSUE} T2 DDLP (triggered Tier-C arm): not run\n\nTyped reason: `{reason}`. G-port {gport_pass(output)}; "
        f"G-fid {gfid_pass(output)}. No records; T2 is listed as not run in the program summary.\n")
    log(f"published not-run summary ({reason})")
    return 0


def publish(output):
    output = Path(output)
    if not (output / "plan.json").is_file():
        reason = not_run_reason(output)
        if reason is None:
            raise SystemExit("no plan.json and no typed not-run reason: nothing to publish")
        return publish_not_run(output, reason)
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
    for mode in ("data", "project", "timing", "readiness", "selection", "selection-readouts", "smoke", "prepare",
                 "train", "readouts", "freeze-readouts", "score", "score-novelty", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--workers", type=int, default=4, help="--data: decode workers")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--timing-steps", type=int, default=300, help="--timing: steps per configuration")
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B135_CPU_THREADS", "4")))  # shared host: avoid oversubscription
    if args.data:
        return build_data(output, args.workers)
    if args.project:
        return project(output)
    if args.timing:
        return timing(output, args.timing_steps, device=args.device)
    if args.readiness:
        return readiness(output, args.device)
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
