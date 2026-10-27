"""Issue #138 (#127 coverage arm T6, rule P-coverage): V-JEPA 2-AC (action-conditioned latent predictor over frozen
V-JEPA 2 features) on NovPhy decision frames against U on the sealed launch-ranking grid.

Arm (Decisions on #138: P-license / P-route / P-adapter / P-fid, see DECISIONS):
- **T6 V-JEPA 2-AC** (coverage, secondary; never in the program claim): the released ViT-g/16 AC encoder (frozen;
  ``vjepa2-ac-vitg.pt``) encodes every NovPhy frame (official ``VideoTransform``, 256 x 256, a 2-frame tubelet per
  frame, bf16 autocast; cached in fp16); the official ``VisionTransformerPredictorAC`` (depth 24, width 1024) is
  trained from scratch on fit-shot clips of 8 stride-5 frames with the official teacher-forcing + rollout L1 loss,
  AdamW + WSD / cosine-WD schedules, 94 500 updates x 256 clips (32 micro-batches of 8); the launch enters at t0
  (``eb.launch5`` as the action token), no-ops after, zero state tokens; scoring rolls the predictor 45 steps to frame
  225 (sliding context of 7 frames). Score = tie-free ``EndpointCosts`` of R1(predicted frame-225 latent) against
  R1(decision-frame latent); R1 = the #130 ``PatchReadout`` recipe fitted on the normalised true features of fit frames
  at ``rc.probe_positions`` (readiness job), shared by every configuration and seed. Configuration selected on the
  policy split among native / ar4 / lr-half (seed 20260908, full native budget); the selected run is reused as main
  seed 20260908.
- Readiness stop (pre-registered, C1-dino's R1 procedure): the R1 policy ceiling on true-frame features < 0.65 ->
  typed stop ``readiness_ceiling_below_threshold`` (not-run publication).

Gates: ``--selection`` / ``--train`` / ``--features training`` (and every later GPU mode) refuse without the
orchestrator's ``funded.json``; ``--selection`` also needs G-port, G-fid (Diving-48 Tier E) and the readiness record
to have passed; ``--score`` refuses unless G-fid passed. ``--timing`` (<= 1 GPU-h) and ``--features readiness`` /
``--readiness`` need no funded.json but their own posted decisions before they are enqueued. Without plan.json,
``--publish`` writes the not-run child summary (readiness stop, or the orchestrator's ``not-run.json``
``{"typed_reason": "cap" | "fidelity_not_reproduced", "decision": url}``).

Modes (round3/child-runner-contract.md): --features {readiness,training}, --readiness, --project (CPU, [INFERENCE]),
--timing, --selection, --selection-readouts, --smoke, --prepare, --train, --readouts, --freeze-readouts, --score,
--score-novelty (P-novelty, descriptive), --publish, --validate. GPU work only through the night-window scheduler
(scripts/gpu_window_issue_138*.json, prepared, not enqueued by the child); env ``novphy``; tests use
CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from world_model.baselines.vjepa2 import novphy as vn  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 138
IDENTITY = "issue-138-vjepa2-ac-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_138_vjepa2_ac.py"
TESTS = "tests/test_issue_138_vjepa2_ac.py"
SCHEMA_MODEL = "issue_138_model_v1"
SCHEMA_PLAN = "issue_138_plan_v1"
SCHEMA_RECORD = "issue_138_record_v1"
SCHEMA_SMOKE = "issue_138_smoke_v1"
SCHEMA_SELECTION = "issue_138_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_138_readouts_manifest_v1"
SCHEMA_PROJECTION = "issue_138_projection_v1"
SCHEMA_TIMING = "issue_138_timing_v1"
SCHEMA_READINESS = "issue_138_readiness_v1"
SCHEMA_FEATURES = "issue_138_feature_manifest_v1"
SCHEMA_R1 = "issue_138_r1_v1"
DECISIONS = {
    "license": "https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436220188",
    "route": "https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436321490",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436355731",
    "fid": "https://github.com/Sino-Huang/NovPhy/issues/138#issuecomment-6436495373",
}

CACHE = Path(os.environ.get("B138_FEATURE_CACHE", Path.home() / ".cache" / "novphy-issue-138-vjepa2-features"))
FIT_SHOTS = 13_919  # eb.FitShots() (13 920 minus one typed-failure branch); checked by --smoke
CONFIGS = {  # declaration order (Decision P-adapter)
    "native": {"updates": vn.NATIVE_UPDATES, "batch": vn.GLOBAL_BATCH, "auto_steps": 2, "optimization": {}},
    "ar4": {"updates": vn.NATIVE_UPDATES, "batch": vn.GLOBAL_BATCH, "auto_steps": 4, "optimization": {}},
    "lr-half": {"updates": vn.NATIVE_UPDATES, "batch": vn.GLOBAL_BATCH, "auto_steps": 2,
                "optimization": {"lr": 2.125e-4, "start_lr": 3.75e-5}},
}
SELECTION_QUEUES = {0: ["native"], 1: ["ar4"], 2: ["lr-half"]}
ARMS = {"T6": {"size": "native", "role": "coverage (secondary, P-coverage; never in the program claim)",
               "program_set": False}}
ARM = "T6"
ARM_CHECKS = ["G_rank", "G_action", "G_readout", "G_nonfinite", "G_trained"]
POSITION_STEPS = {t: t // vn.STRIDE for t in eb.POSITION_TIMES}  # rollout step of each position time
CLIP_STARTS = len(vn.POSITIONS) - vn.CLIP_FRAMES + 1  # 39 clip starts per shot
GRID = "grid"
THRESHOLD = eb.GUARD["readout_ceiling_auc_min"]  # 0.65
TYPED_STOP = "readiness_ceiling_below_threshold"
CHECKPOINT_EVERY = 250
LOG_EVERY = 50
FEATURE_BATCH = 64
CHILD_CAP_GPU_HOURS = 90.0  # upper end of the #127 scout estimate for V-JEPA 2-AC (44-90)
SEED_CAP_GPU_HOURS = 24.0  # #127: native cap per seed
WALL_CAP_HOURS = 48.0  # on 4 GPUs
REMAINDER_GPU_HOURS = 73.0  # #127 issuecomment-6436095504: program remainder after DIAMOND
TRAINING_PRECISION = {"dtype": "bfloat16 autocast + GradScaler on CUDA (meta.dtype bfloat16, upstream); float32 on CPU",
                      "features": "frozen encoder under bf16 autocast, cached fp16; layer_norm at load"}
CALIBRATION = ROOT / ".local-artifacts/issue-129-slotformer-v1/analysis/g-action-calibration.json"


def log(message):
    eb.log(ISSUE, message)


def device_of(requested=None):
    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def config_spec(name):
    c = CONFIGS[name]
    return {"name": name, "updates": c["updates"], "batch": c["batch"], "micro_batch": vn.MICRO_BATCH,
            "clip_frames": vn.CLIP_FRAMES, "auto_steps": c["auto_steps"],
            "optimization": {**vn.OPTIMIZATION, **c["optimization"]}, "predictor": vn.PREDICTOR,
            "loss": {**vn.LOSS, "auto_steps": c["auto_steps"]}}


def arm_spec(arm, selected):
    return config_spec(selected)


def selection_dir(output, name, seed=eb.SELECTION_SEED):
    return Path(output) / "selection" / f"{name}--seed{seed}"


def model_dir(output, arm, seed, selected=None):
    if seed == eb.SELECTION_SEED:
        if selected is None:
            raise ValueError("the main seed-20260908 run is the selected configuration's selection run")
        return selection_dir(output, selected)
    return Path(output) / "models" / f"{arm}--seed{seed}"


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / f"{arm}--seed{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def readiness_dir(output):
    return Path(output) / "readiness"


# ---------------------------------------------------------------------------------------------------------------
# feature cache (fit: training positions u R1 probe positions; policy: anchors + true endpoints)
# ---------------------------------------------------------------------------------------------------------------

def training_positions(end):
    """Clip frames of a fit shot: stride-5 positions 0..225, clamped to the shot's end (later frames repeat it)."""
    return sorted({min(p, int(end)) for p in vn.POSITIONS})


def fit_positions(end):
    from scripts import run_hjepa_comparison as rc
    return sorted(set(training_positions(end)) | set(rc.probe_positions(int(end))))


def fit_layout(ends):
    """first [S+1] row offsets and the concatenated positions of every fit shot (frozen eb.FitShots order)."""
    first, positions = [0], []
    for end in ends:
        p = fit_positions(end)
        positions += p
        first.append(first[-1] + len(p))
    return np.asarray(first, dtype=np.int64), np.asarray(positions, dtype=np.int16)


class FitIndex:
    """(shot, position) -> row of the fit feature memmap; raises for anything that is not a cached fit frame."""

    def __init__(self, first, positions):
        self.first, self.positions = np.asarray(first), np.asarray(positions)

    def rows(self, shots, positions):
        shots, positions = np.asarray(shots, dtype=np.int64), np.asarray(positions)
        out = np.empty(len(shots), dtype=np.int64)
        for i, (s, p) in enumerate(zip(shots, positions)):
            lo, hi = int(self.first[s]), int(self.first[s + 1])
            k = int(np.searchsorted(self.positions[lo:hi], p))
            if k >= hi - lo or int(self.positions[lo + k]) != int(p):
                raise KeyError(f"fit shot {s} position {p} is not in the feature cache")
            out[i] = lo + k
        return out

    def clip_table(self, ends):
        """[S, 46] fit rows of each shot's stride-5 positions clamped to its end."""
        shots = np.repeat(np.arange(len(ends)), len(vn.POSITIONS))
        pos = np.asarray([min(p, int(e)) for e in ends for p in vn.POSITIONS])
        return self.rows(shots, pos).reshape(len(ends), len(vn.POSITIONS))


def r1_items(ends):
    """The #123 R1 frame set (C1-dino's readiness): every fit shot at rc.probe_positions(end)."""
    from scripts import run_hjepa_comparison as rc
    return [(shot, p) for shot, end in enumerate(ends) for p in rc.probe_positions(int(end))]


def policy_layout():
    """("anchor", member, 0) for the policy anchors, then ("endpoint", key, min(225, end)) per complete policy shot."""
    anchors = eb.policy_anchors()["keys"]
    shots = eb.policy_shots()
    items = [("anchor", m, 0) for m in sorted(anchors)]
    items += [("endpoint", k, int(min(vn.ENDPOINT, int(shots[k]["end"])))) for k in sorted(shots)]
    return items


def feature_paths(output=OUTPUT):
    f = Path(output) / "features"
    return {"fit": CACHE / "fit-features.f16", "policy": CACHE / "policy-features.f16",
            "fit_index": f / "fit-index.npz", "policy_index": f / "policy-index.json", "plan": f / "plan.json",
            "progress": f / "progress.json", "manifest": f / "MANIFEST.json"}


FRAME_BYTES = vn.TOKENS * vn.EMBED_DIM * 2


def open_features(output, kind, mode="r"):
    p = feature_paths(output)
    rows = (int(np.load(p["fit_index"])["first"][-1]) if kind == "fit"
            else len(eb.read_json(p["policy_index"])["items"]))
    return np.memmap(p[kind], dtype=np.float16, mode=mode, shape=(rows, vn.TOKENS, vn.EMBED_DIM))


def plan_features(output=OUTPUT, shots=None):
    """features/{fit-index.npz, policy-index.json, plan.json}: the cache layout and sizes (CPU)."""
    shots = shots or eb.FitShots()
    p = feature_paths(output)
    p["fit_index"].parent.mkdir(parents=True, exist_ok=True)
    first, positions = fit_layout(shots.ends)
    np.savez(p["fit_index"], keys=np.array(shots.keys), first=first, positions=positions,
             ends=np.asarray(shots.ends, dtype=np.int16))
    items = policy_layout()
    eb.write_json(p["policy_index"], {"schema": "issue_138_policy_feature_index_v1", "items": items})
    n_fit, n_r1 = int(first[-1]), len(r1_items(shots.ends))
    eb.write_json(p["plan"], {
        "schema": "issue_138_feature_plan_v1", "written_at": eb.utc_now(), "cache_dir": str(CACHE),
        "encoder": {"checkpoint": rel(vn.AC_CHECKPOINT), "sha256": vn.AC_CHECKPOINT_SHA256, "key": "encoder",
                    "model": vn.ENCODER["model_name"], "autocast": "bfloat16", "stored": "float16 raw tokens"},
        "frame_bytes": FRAME_BYTES,
        "fit": {"shots": len(shots), "rows": n_fit, "readiness_rows": n_r1, "training_only_rows": n_fit - n_r1,
                "bytes": n_fit * FRAME_BYTES, "readiness_bytes": n_r1 * FRAME_BYTES},
        "policy": {"rows": len(items), "bytes": len(items) * FRAME_BYTES}})
    log(f"feature plan: fit {n_fit} rows ({n_r1} readiness), policy {len(items)}; "
        f"{n_fit * FRAME_BYTES / 1e9:.0f} GB fit ({n_r1 * FRAME_BYTES / 1e9:.0f} GB readiness)")
    return 0


def phase_rows(output, phase, shots):
    """Fit rows a feature phase writes: readiness = the R1 rows; training = every other row."""
    index = np.load(feature_paths(output)["fit_index"])
    fi = FitIndex(index["first"], index["positions"])
    items = r1_items(shots.ends)
    r1 = np.unique(fi.rows([s for s, _ in items], [q for _, q in items]))
    if phase == "readiness":
        return r1
    return np.setdiff1d(np.arange(int(index["first"][-1])), r1)


class FitFrames(torch.utils.data.Dataset):
    """uint8 [3, 240, 320] fit frames of (shot, position) items (the #123 fit pixel store; one store per worker)."""

    def __init__(self, items):
        self.items, self.data = items, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from scripts import run_hjepa_comparison as rc
        if self.data is None:
            self.data = rc.FitData(rc.OUTPUT)
        shot, position = self.items[i]
        return self.data.frame(int(shot), int(position))


def policy_frames(items):
    from scripts import run_hjepa_comparison as rc
    anchors = eb.policy_anchor_frames()
    store = rc.PixelStore(rc.OUTPUT, "policy")
    return torch.stack([anchors[name] if kind == "anchor" else store.frame(store.row[name], q)
                        for kind, name, q in items])


def features(output, phase, device=None, workers=8):
    """--features readiness|training: frozen-encoder tokens into the fp16 caches (resumable per batch: SIGTERM ->
    progress saved -> exit 75); at the end of a phase the cache files are hashed into features/MANIFEST.json."""
    output = Path(output)
    if phase == "training" and refuse_unfunded(output, "features training"):
        return 1
    device = device_of(device)
    shots = eb.FitShots()
    p = feature_paths(output)
    if not p["fit_index"].is_file():
        plan_features(output, shots)
    eb.install_stop_handlers()
    CACHE.mkdir(parents=True, exist_ok=True)
    progress = eb.read_json(p["progress"]) if p["progress"].is_file() else {"readiness": 0, "training": 0,
                                                                              "policy_done": False}
    if progress.get(f"{phase}_complete"):
        log(f"features {phase}: complete")
        return 0
    index = np.load(p["fit_index"])
    items = eb.read_json(p["policy_index"])["items"]
    for kind, rows in (("fit", int(index["first"][-1])), ("policy", len(items))):
        if not p[kind].is_file():  # sparse preallocation
            np.memmap(p[kind], dtype=np.float16, mode="w+", shape=(rows, vn.TOKENS, vn.EMBED_DIM)).flush()
    rows = phase_rows(output, phase, shots)
    shot_of = np.repeat(np.arange(len(index["first"]) - 1), np.diff(index["first"]))
    began = time.monotonic()
    encoder, load_msg = vn.load_encoder(device)
    if phase == "readiness" and not progress["policy_done"]:
        mm = open_features(output, "policy", "r+")
        frames = policy_frames(items)
        for i in range(0, len(frames), FEATURE_BATCH):
            mm[i:i + FEATURE_BATCH] = vn.encode(encoder, frames[i:i + FEATURE_BATCH]).half().cpu().numpy()
        mm.flush()
        progress["policy_done"] = True
        eb.write_json(p["progress"], progress)
        log(f"policy features: {len(items)} frames")
    mm = open_features(output, "fit", "r+")
    start = progress[phase]
    todo = rows[start:]
    loader = torch.utils.data.DataLoader(
        FitFrames([(int(shot_of[r]), int(index["positions"][r])) for r in todo]), batch_size=FEATURE_BATCH,
        num_workers=workers, prefetch_factor=4 if workers else None)
    done = start
    for k, frames in enumerate(loader):
        n = frames.shape[0]
        mm[rows[done:done + n]] = vn.encode(encoder, frames).half().cpu().numpy()
        done += n
        if k % 200 == 199 or eb.STOP_REQUESTED:
            mm.flush()
            progress[phase] = int(done)
            eb.write_json(p["progress"], progress)
            log(f"fit features {phase}: {done}/{len(rows)} ({time.monotonic() - began:.0f}s)")
            if eb.STOP_REQUESTED:
                seconds = time.monotonic() - began
                eb.mark(output, f"features-{phase}", seconds, seconds if device != "cpu" else 0)
                return eb.EXIT_PAUSED
    mm.flush()
    progress[phase] = int(done)
    progress[f"{phase}_complete"] = done == len(rows)
    eb.write_json(p["progress"], progress)
    seconds = time.monotonic() - began
    eb.mark(output, f"features-{phase}", seconds, seconds if device != "cpu" else 0)
    manifest = eb.read_json(p["manifest"]) if p["manifest"].is_file() else {"schema": SCHEMA_FEATURES, "phases": {}}
    manifest["phases"][phase] = {
        "rows": int(len(rows)), "written_at": eb.utc_now(), "encoder_load": load_msg,
        "files": {kind: {"path": str(p[kind]), "bytes": p[kind].stat().st_size, "sha256": eb.sha256_of(p[kind])}
                  for kind in ("fit", "policy")},
        "indexes": {kind: {"path": rel(p[kind]), "sha256": eb.sha256_of(p[kind])}
                    for kind in ("fit_index", "policy_index", "plan")}}
    eb.write_json(p["manifest"], manifest)
    log(f"features {phase} complete ({seconds:.0f}s)")
    return 0


def verify_features(output, phase):
    p = feature_paths(output)
    manifest = eb.read_json(p["manifest"])
    entry = manifest["phases"][phase]
    for kind, idx in entry["indexes"].items():
        if eb.sha256_of(ROOT / idx["path"]) != idx["sha256"]:
            raise ValueError(f"{kind} differs from the feature manifest")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# model build / training (resumable)
# ---------------------------------------------------------------------------------------------------------------

def build_model(spec, device="cpu"):
    return vn.build_predictor(use_activation_checkpointing=str(device).startswith("cuda")).to(device)


class ClipData:
    """Fit-split clip batches from the feature cache: update b draws 256 (shot, start) pairs from
    eb.batch_rng(seed, b) (uniform shot, uniform start in 0..38) and returns 32 micro-batches (h, actions, states)."""

    def __init__(self, store, table, launch5):
        self.store, self.table = store, np.asarray(table)
        self.launch5 = torch.as_tensor(np.asarray(launch5), dtype=torch.float32)

    def draw(self, seed, b, batch):
        rng = eb.batch_rng(seed, b)
        shots = rng.integers(0, len(self.table), size=batch)
        starts = rng.integers(0, CLIP_STARTS, size=batch)
        return shots, starts

    def tokens(self, shots, starts, frames=vn.CLIP_FRAMES):
        rows = self.table[shots[:, None], starts[:, None] + np.arange(frames)[None]]  # [B, T]
        flat = rows.reshape(-1)
        order = np.argsort(flat, kind="stable")
        out = np.empty((len(flat), vn.TOKENS, vn.EMBED_DIM), dtype=np.float16)
        out[order] = self.store[flat[order]]
        return torch.from_numpy(out).view(len(shots), frames * vn.TOKENS, vn.EMBED_DIM)

    def batch(self, seed, b, spec, device="cpu"):
        shots, starts = self.draw(seed, b, spec["batch"])
        h = self.tokens(shots, starts)
        actions = vn.clip_actions(self.launch5[shots], starts)
        micro = []
        for i in range(0, spec["batch"], spec["micro_batch"]):
            sl = slice(i, i + spec["micro_batch"])
            micro.append((vn.normalize(h[sl].to(device).float()), actions[sl].to(device),
                          vn.clip_states(len(range(*sl.indices(spec["batch"])))).to(device)))
        return micro


def load_data(output=OUTPUT, shots=None):
    shots = shots or eb.FitShots()
    verify_features(output, "training")
    index = np.load(feature_paths(output)["fit_index"])
    fi = FitIndex(index["first"], index["positions"])
    return ClipData(open_features(output, "fit"), fi.clip_table(shots.ends), shots.launch5)


def _rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _scaler(device):
    return torch.amp.GradScaler("cuda") if str(device).startswith("cuda") else None


def _autocast(device):
    return torch.bfloat16 if str(device).startswith("cuda") else None


def train_run(out_dir, spec, seed, data, device=None, max_steps=None):
    """Train one predictor (resumable from out_dir/resume.pt). Update b = data.batch(seed, b); the official update
    (vn.train_step); nonfinite loss -> typed failure."""
    device = device_of(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{spec['name']} seed {seed}: model.pt exists; skipped")
        return eb.read_json(out_dir / "report.json")
    torch.manual_seed(seed)
    model = build_model(spec, device)
    optimizer, scheduler, wd_scheduler = vn.build_optimizer(model, spec)
    scaler = _scaler(device)
    steps = spec["updates"]
    start, elapsed, trace = 0, 0.0, []
    resume = out_dir / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler._step, wd_scheduler._step = state["scheduler_step"], state["wd_scheduler_step"]
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start, elapsed, trace = state["step"], state["elapsed"], state["trace"]
        _set_rng_state(state["rng"])
        log(f"{spec['name']} seed {seed}: resumed at step {start}")

    def save_resume(step):
        eb.atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scheduler_step": scheduler._step, "wd_scheduler_step": wd_scheduler._step,
                                 "scaler": scaler.state_dict() if scaler is not None else None, "step": step,
                                 "rng": _rng_state(), "elapsed": elapsed + time.monotonic() - began, "trace": trace})

    began = time.monotonic()
    failure, step = None, start
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    model.train()
    optimizer.zero_grad()
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(data.batch, seed, step, spec, "cpu") if step < stop_at else None
        while step < stop_at:
            if eb.STOP_REQUESTED:
                save_resume(step)
                raise eb.TrainingPaused(f"{spec['name']} seed {seed} paused at step {step}")
            micro = pending.result()
            pending = pool.submit(data.batch, seed, step + 1, spec, "cpu") if step + 1 < stop_at else None
            micro = [(h.to(device), a.to(device), s.to(device)) for h, a, s in micro]
            terms = vn.train_step(model, optimizer, scheduler, wd_scheduler, micro, spec["auto_steps"], scaler,
                                  _autocast(device))
            if not math.isfinite(terms["loss"]):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "terms": terms}
                break
            step += 1
            if step % LOG_EVERY == 0 or step == steps:
                trace.append({"step": step, **terms})
                log(f"{spec['name']} seed {seed}: step {step}/{steps} loss {terms['loss']:.4f} "
                    f"({(elapsed + time.monotonic() - began) / max(1, step - start):.3f} s/step)")
            if step % CHECKPOINT_EVERY == 0 and step < steps:
                save_resume(step)
    seconds = elapsed + time.monotonic() - began
    if failure is None and step < steps:  # max_steps reached (tests, timing)
        save_resume(step)
        return {"partial": True, "steps_done": step, "seconds": seconds}
    complete = failure is None
    precision = {**TRAINING_PRECISION, "device": device}
    payload = {"schema": SCHEMA_MODEL, "spec": spec, "seed": seed, "complete": complete, "typed_failure": failure,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None,
               "parameters": vn.trainable_parameters(model), "precision": precision}
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
    model = vn.build_predictor().to(device_of(device))
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
# encoder, R1 readout, rollouts, costs
# ---------------------------------------------------------------------------------------------------------------

_ENCODER = {}


def frozen_encoder(device):
    if device not in _ENCODER:
        _ENCODER.clear()
        _ENCODER[device] = vn.load_encoder(device)[0]
    return _ENCODER[device]


def anchor_tokens(encoder, frames_u8, chunk=FEATURE_BATCH):
    """Normalised decision-frame latents [N, 256, 1408] (encoded exactly as the cache: bf16 autocast, fp16)."""
    out = [vn.normalize(vn.encode(encoder, frames_u8[i:i + chunk]).half().float())
           for i in range(0, len(frames_u8), chunk)]
    return torch.cat(out)


def objective_costs():
    from scripts import run_lookahead_control as wlc
    return harness.EndpointCosts(wlc.load_objective())


def grid_actions():
    items = eb.grid_items()
    return items, torch.stack([eb.launch5(item["action"]) for item in items])


def save_r1(path, probe, report):
    eb.atomic_torch(path, {"schema": SCHEMA_R1, "state": {k: v.cpu() for k, v in probe.state_dict().items()},
                           "fit": report})


def load_r1(output, device="cpu"):
    payload = torch.load(readiness_dir(output) / "r1.pt", map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_R1:
        raise ValueError("r1.pt: schema differs")
    probe = vn.new_readout()
    probe.load_state_dict(payload["state"])
    return probe.to(device).eval().requires_grad_(False), payload


def carriers(probe, z):
    """R1 carriers [N, 236] of normalised latents z [N, 256, 1408] (CPU)."""
    return vn.apply_readout(probe, z, str(next(probe.parameters()).device))


@torch.no_grad()
def predict(model, probe, z0, launch5, keep=(vn.STEPS,)):
    """Rollouts of normalised anchor latents z0 [L, 256, 1408] under launches [L, K, 5]: {"anchor": R1 carriers
    [L, 236], "carriers": {step: R1 carriers [L, K, 236]}} (CPU)."""
    device = next(model.parameters()).device
    n_l, n_k = launch5.shape[:2]
    out = {"anchor": carriers(probe, z0), "carriers": {}}
    per = {s: [] for s in keep}
    for i in range(n_l):
        z = z0[i:i + 1].to(device).expand(n_k, -1, -1)
        rolled = vn.rollout(model, z, launch5[i].to(device), keep=keep)
        for s in keep:
            per[s].append(carriers(probe, rolled[s]))
    for s in keep:
        out["carriers"][s] = torch.stack(per[s])
    return out


@torch.no_grad()
def level_rows(model, probe, items, z0, costs):
    """Rows of one level's candidates from its normalised decision-frame latent z0 [256, 1408]: tie-free cost of
    R1(predicted frame-225 latent) against R1(z0) (the primary scoring path, shared by the policy AUCs, --score,
    --score-novelty and --validate)."""
    actions = torch.stack([eb.launch5(it["action"]) for it in items])
    out = predict(model, probe, z0[None], actions[None])
    return eb.endpoint_rows(items, out["carriers"][vn.STEPS][0], out["anchor"][0], costs)


def tie_free_matrix(ends, c0, costs):
    """[L, K] tie-free costs (nonfinite -> nan) of endpoint carriers [L, K, 236] from anchor carriers [L, 236]."""
    out = np.full(ends.shape[:2], np.nan)
    items, _ = grid_actions()
    ordinals = [it["ordinal"] for it in items]
    for i in range(ends.shape[0]):
        for row in eb.endpoint_rows(items, ends[i], c0[i], costs):
            if not row["excluded"]:
                out[i, ordinals.index(row["ordinal"])] = row[eb.PRIMARY_COST]
    return out


def policy_anchor_latents(output):
    """{member: normalised anchor latent} of the policy grid anchors from the policy feature cache."""
    items = eb.read_json(feature_paths(output)["policy_index"])["items"]
    mm = open_features(output, "policy")
    return {name: vn.normalize(torch.from_numpy(np.asarray(mm[i])).float())
            for i, (kind, name, _) in enumerate(items) if kind == "anchor"}


@torch.no_grad()
def policy_endpoint_rows(model, probe, costs, z0s):
    return {member: level_rows(model, probe, entry["inventory"], z0s[member], costs)
            for member, entry in sorted(eb.policy_cells(GRID).items())}


def _e99_endpoint(shots, key):
    shot = shots[key]
    return shot["carriers"][(shot["positions"] == min(vn.ENDPOINT, int(shot["end"]))).nonzero()[0, 0]]


def e99_true_ceiling(costs):
    """The E99 true-carrier ceiling on the same policy cells (#130 reference 0.7223)."""
    cells = eb.policy_cells(GRID)
    anchors = eb.policy_anchors()["carriers"]
    shots = eb.policy_shots()
    rows = {}
    for m, entry in sorted(cells.items()):
        ends = torch.stack([_e99_endpoint(shots, it["key"]) for it in entry["inventory"]]).float()
        rows[m] = eb.endpoint_rows(entry["inventory"], ends, anchors[m].float(), costs)
    return eb.policy_grid_auc(rows, eb.PRIMARY_COST)


def ceiling_from(probe, costs, policy_store, items):
    """G-readout / readiness (C1-dino's procedure): pooled policy grid AUC of the tie-free cost of R1(true endpoint
    features) against R1(anchor features), + R1's error against the E99 carriers of the same endpoints."""
    row_of = {(kind, name): i for i, (kind, name, _) in enumerate(items)}
    cells = eb.policy_cells(GRID)
    members = sorted(cells)
    norm = vn.NormalizedFeatures(policy_store)
    c0 = dict(zip(members, carriers(probe, torch.from_numpy(norm[[row_of[("anchor", m)] for m in members]]))))
    flat = [(m, item) for m in members for item in cells[m]["inventory"]]
    c_true = carriers(probe, torch.from_numpy(norm[[row_of[("endpoint", it["key"])] for _, it in flat]]))
    shots = eb.policy_shots()
    e99 = torch.stack([_e99_endpoint(shots, it["key"]) for _, it in flat]).float()
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


def g_action_calibration(value):
    """#127 decision (issuecomment-6396584235): the carrier-space G-action beside the frozen guard (descriptive)."""
    if not CALIBRATION.is_file():
        return {"arm": value, "source": None}
    cal = eb.read_json(CALIBRATION)
    truth = cal["means"]["engine_truth"]
    return {"arm": value, "engine_truth": truth, "fraction_of_engine_truth": None if value is None else value / truth,
            "threshold_frozen": cal["threshold_frozen"], "source": rel(CALIBRATION),
            "source_sha256": eb.sha256_of(CALIBRATION), "label": "DESCRIPTIVE (no guard change)"}


# ---------------------------------------------------------------------------------------------------------------
# gates (#138)
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


def readiness_pass(output):
    path = readiness_dir(output) / "readiness.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def refuse_unfunded(output, mode):
    if funded(output):
        return False
    log(f"{mode} refused: funded.json absent (no posted funding decision); no GPU work")
    return True


def selection_gate(output):
    """Funding marker, G-port, G-fid (Diving-48 Tier E) and the readiness record (validation ladder)."""
    out = {"funded.json": funded(output), "gport/gport.json": gport_pass(output),
           "fidelity/gfid.json": gfid_pass(output), "readiness/readiness.json": readiness_pass(output)}
    out["pass"] = all(out.values())
    return out


# ---------------------------------------------------------------------------------------------------------------
# --readiness (C1-dino's R1 procedure on frozen V-JEPA 2 features of true frames; pre-registered stop)
# ---------------------------------------------------------------------------------------------------------------

def readiness_decision(ceiling, threshold=THRESHOLD):
    passed = ceiling is not None and ceiling >= threshold
    return {"pass": bool(passed), "typed_reason": None if passed else TYPED_STOP,
            "consequence": "T6 Phase 0 continues" if passed else "T6 stops before any predictor training (typed)"}


def readiness(output, device=None):
    """R1 (#130 PatchReadout / fit_readout) on the normalised features of the fit R1 frames (E99 targets, seed
    20260908) under deterministic_scoring(), then the policy ceiling; readiness/r1.pt + readiness.json (one run)."""
    output = Path(output)
    rdir = readiness_dir(output)
    if (rdir / "readiness.json").is_file():
        log("readiness.json exists (one run, never re-run)")
        return 0
    rdir.mkdir(parents=True, exist_ok=True)
    device = device_of(device)
    manifest = verify_features(output, "readiness")
    p = feature_paths(output)
    began = time.monotonic()
    shots = eb.FitShots()
    index = np.load(p["fit_index"])
    if list(map(str, index["keys"])) != shots.keys:
        raise ValueError("feature index and eb.FitShots disagree on the shot order")
    fi = FitIndex(index["first"], index["positions"])
    items = r1_items(shots.ends)
    sh = np.array([s for s, _ in items])
    pos = np.array([q for _, q in items])
    rows = fi.rows(sh, pos)
    targets = shots.carrier_rows(sh, pos)
    costs = objective_costs()
    with harness.deterministic_scoring() as policy:
        probe, fit_report = vn.fit_readout(vn.NormalizedFeatures(open_features(output, "fit")), rows, targets,
                                           eb.SELECTION_SEED, device=device, log=log)
        save_r1(rdir / "r1.pt", probe, fit_report)
        ceiling = ceiling_from(probe, costs, open_features(output, "policy"),
                               eb.read_json(p["policy_index"])["items"])
        e99 = e99_true_ceiling(costs)
    record = {"schema": SCHEMA_READINESS, "arm": ARM, "written_at": eb.utc_now(), "scoring_policy": policy,
              "threshold": THRESHOLD, **ceiling, **readiness_decision(ceiling["ceiling_auc"]),
              "e99_true_carrier_ceiling_auc": e99,
              "r1": {"artifact": rel(rdir / "r1.pt"), "sha256": eb.sha256_of(rdir / "r1.pt"), "fit_frames": len(items),
                     "fit_shots": len(shots), "seed": eb.SELECTION_SEED, **fit_report},
              "decision": DECISIONS["adapter"],
              "data": {"r1_fit": "fit split only: eb.FitShots x rc.probe_positions, normalised frozen V-JEPA 2 AC "
                                 "encoder features, E99 targets from the #123 fit memmap",
                       "ceiling": "policy split: eb.policy_cells('grid') mixed cells, eb.policy_anchor_frames, #123 "
                                  "policy pixel store at min(225, end)", "evaluation": "not read"},
              "feature_manifest_sha256": eb.sha256_of(p["manifest"]),
              "feature_files": manifest["phases"]["readiness"]["files"], "seconds": time.monotonic() - began}
    eb.write_json(rdir / "readiness.json", record)
    eb.mark(output, "readiness", record["seconds"], record["seconds"] if device != "cpu" else 0)
    log(f"readiness: ceiling {ceiling['ceiling_auc']} (threshold {THRESHOLD}) -> "
        f"{'PASS' if record['pass'] else TYPED_STOP}; E99 {e99}")
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


def selection_readouts(output, device=None):
    """Per configuration: the policy grid AUC of the primary score (deterministic scoring); selection.json."""
    output = Path(output)
    if refuse_unfunded(output, "selection-readouts"):
        return 1
    device = device_of(device)
    costs = objective_costs()
    results, details = {}, {}
    z0s = policy_anchor_latents(output)
    probe, _ = load_r1(output, device)
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
            rows = policy_endpoint_rows(model, probe, costs, z0s)
            results[name] = {"auc": eb.policy_grid_auc(rows, eb.PRIMARY_COST), "parameters": payload["parameters"]}
            details[name] = {"typed_failure": None, "model_sha256": eb.sha256_of(path), "cells": len(rows)}
    choice = eb.select_configuration(results)
    seconds = time.monotonic() - began
    eb.mark(output, "selection-readouts", seconds, seconds if device != "cpu" else 0)
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


# decisions (one anchor x its 16 grid candidates) per pass: selection readouts = the 94 policy cells per
# configuration; readouts per arm-seed = 480 fit anchors (G-rank / G-action) + 94 policy cells; scoring per seed =
# the mixed cells of the four inventories (311 + 76 + 78 + 47) + 480 wrong anchors + the position branches
# (~311 x 16 single-candidate rollouts ~ 311 decisions) + novelty (29 x 8 x 2 levels)
POLICY_DECISIONS = 94
READOUT_DECISIONS = 480 + 94
SCORING_DECISIONS = (311 + 76 + 78 + 47) + 480 + 311 + 29 * 8 * 2
FEATURE_ROWS = {"readiness": 273_692 + 3_600, "training_only": 413_346}  # replaced by features/plan.json if present
R1_STEPS = 53_450  # C1-dino's R1: 25 epochs x 2 138 batches of 128 (same frame set)


def child_projection(rates, decision_seconds, frame_seconds, r1_step_seconds, gfid_gpu_h, feature_rows):
    """GPU-h of the child from per-configuration s/update, s/decision, s/encoded frame and s/R1 step: the feature
    caches (readiness + training rows), the readiness R1 fit, three selection runs, two more seeds at the most
    expensive configuration, selection readouts, readouts and scoring of 3 seeds (+ novelty), G-fid, the probe."""
    names = list(CONFIGS)
    steps = {n: CONFIGS[n]["updates"] for n in names}
    selection_s = {n: steps[n] * rates[n] for n in names}
    per_seed = max(selection_s.values())
    training = [(f"{ARM}--seed{s}", per_seed) for s in eb.SEEDS if s != eb.SELECTION_SEED]
    queues, loads = lpt(training)
    features_s = (feature_rows["readiness"] + feature_rows["training_only"]) * frame_seconds
    readiness_s = R1_STEPS * r1_step_seconds
    sel_readouts = len(names) * POLICY_DECISIONS * decision_seconds
    readouts = 3 * READOUT_DECISIONS * decision_seconds
    scoring = 3 * SCORING_DECISIONS * decision_seconds
    other = features_s + readiness_s + sel_readouts + readouts + scoring + gfid_gpu_h * 3600 + 3600
    total = sum(selection_s.values()) + sum(s for _, s in training) + other
    wall = features_s + readiness_s + max(selection_s.values()) + max(loads) + sel_readouts + readouts / 3 + scoring
    return {"steps": steps, "seconds_per_update": rates, "seconds_per_decision": decision_seconds,
            "seconds_per_encoded_frame": frame_seconds, "seconds_per_r1_step": r1_step_seconds,
            "features_gpu_h": features_s / 3600, "readiness_gpu_h": readiness_s / 3600,
            "selection_gpu_h": {k: v / 3600 for k, v in selection_s.items()},
            "training_gpu_h": sum(s for _, s in training) / 3600, "selection_readouts_gpu_h": sel_readouts / 3600,
            "readouts_gpu_h": readouts / 3600, "score_gpu_h": scoring / 3600, "gfid_gpu_h": gfid_gpu_h,
            "timing_probe_gpu_h": 1.0, "total_gpu_h": total / 3600, "native_per_seed_gpu_h_max": per_seed / 3600,
            "seed_cap_gpu_h": SEED_CAP_GPU_HOURS, "training_queues": queues,
            "training_queue_loads_h": [x / 3600 for x in loads], "wall_h_on_4_gpus_serial_ladder": wall / 3600,
            "child_cap_gpu_h": CHILD_CAP_GPU_HOURS, "wall_cap_h": WALL_CAP_HOURS,
            "remainder_gpu_h": REMAINDER_GPU_HOURS,
            "fits_remainder": bool(total / 3600 <= REMAINDER_GPU_HOURS),
            "within_caps": bool(total / 3600 <= CHILD_CAP_GPU_HOURS and wall / 3600 <= WALL_CAP_HOURS
                                and per_seed / 3600 <= SEED_CAP_GPU_HOURS)}


def _clip_flops(auto_steps):
    """FLOPs of one clip's official loss forward + backward (CPU FlopCounterMode, micro-batch of 1) and its CPU s."""
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    model = vn.build_predictor()
    h = vn.normalize(torch.randn(1, vn.CLIP_FRAMES * vn.TOKENS, vn.EMBED_DIM))
    a = vn.clip_actions(torch.randn(1, vn.LAUNCH_DIM), [0])
    s = vn.clip_states(1)
    with FlopCounterMode(display=False) as fc:
        loss, _, _ = vn.clip_loss(model, h, a, s, auto_steps)
        loss.backward()
    t = time.monotonic()
    loss, _, _ = vn.clip_loss(model, h, a, s, auto_steps)
    loss.backward()
    return fc.get_total_flops(), time.monotonic() - t


def _decision_flops():
    """FLOPs of one decision on CPU: the anchor through the frozen ViT-g (one frame), 45 rollout steps of 16
    candidates (sliding context; the 1-step to 7-step contexts counted exactly), R1 on 17 latents."""
    from torch.utils.flop_counter import FlopCounterMode
    torch.manual_seed(0)
    model = vn.build_predictor().eval().requires_grad_(False)
    probe = vn.new_readout().eval().requires_grad_(False)
    flops = {}
    with torch.no_grad():
        z = vn.normalize(torch.randn(1, vn.TOKENS, vn.EMBED_DIM))
        for ctx in range(1, vn.CONTEXT + 1):
            x = z.repeat(1, ctx, 1)
            a = torch.zeros(1, ctx, vn.LAUNCH_DIM)
            with FlopCounterMode(display=False) as fc:
                model(x, a, a)
            flops[ctx] = fc.get_total_flops()
        with FlopCounterMode(display=False) as fr:
            probe(z)
        encoder = vn.build_encoder().eval().requires_grad_(False)
        with FlopCounterMode(display=False) as fe:
            encoder(torch.zeros(1, 3, vn.TUBELET, vn.CROP, vn.CROP))
    rollout = sum(flops[min(t + 1, vn.CONTEXT)] for t in range(vn.STEPS)) * 16
    return {"rollout_flops": rollout, "readout_flops": 17 * fr.get_total_flops(),
            "encoder_flops_per_frame": fe.get_total_flops(),
            "decision_flops": rollout + 17 * fr.get_total_flops() + fe.get_total_flops()}


def _r1_step_flops():
    from torch.utils.flop_counter import FlopCounterMode
    probe = vn.new_readout()
    x = torch.randn(vn.readout_module().R1["batch"], vn.TOKENS, vn.EMBED_DIM)
    with FlopCounterMode(display=False) as fc:
        probe.forward_z(x).pow(2).mean().backward()
    return fc.get_total_flops()


# [INFERENCE] RTX 5090 mapping of measured FLOPs (sustained bf16 tensor throughput: lower = optimistic 100 TFLOP/s,
# upper = pessimistic 40 TFLOP/s; dense bf16 peak ~209 TFLOP/s) and a feature-cache read bound of 3 GB/s per GPU.
GPU_MODEL = {"lower": {"tflops": 100.0, "read_gb_s": 3.0}, "upper": {"tflops": 40.0, "read_gb_s": 3.0}}
GFID_GPU_H = {"lower": 0.3, "upper": 0.7}  # Decision P-fid (Diving-48 ViT-L Tier E)


def feature_rows(output):
    path = feature_paths(output)["plan"]
    if path.is_file():
        plan = eb.read_json(path)
        return {"readiness": plan["fit"]["readiness_rows"] + plan["policy"]["rows"],
                "training_only": plan["fit"]["training_only_rows"]}
    return dict(FEATURE_ROWS)


def project(output):
    """projection.json: [INFERENCE] CPU micro-benchmark (FLOPs of one official clip update per configuration x 256
    clips, one decision, one encoded frame, one R1 step) mapped to an RTX 5090 by GPU_MODEL, with a read bound for the
    feature cache (1.48 GB per update); when timing.json (GPU) exists, the measured projection is primary."""
    output = Path(output)
    cpu, rates = {}, {}
    for name in CONFIGS:
        flops, seconds = _clip_flops(CONFIGS[name]["auto_steps"])
        cpu[name] = {"clip_flops": flops, "update_flops": flops * vn.GLOBAL_BATCH, "cpu_seconds_per_clip": seconds}
    dec = _decision_flops()
    r1 = _r1_step_flops()
    rows = feature_rows(output)
    update_bytes = vn.GLOBAL_BATCH * vn.CLIP_FRAMES * FRAME_BYTES
    r1_bytes = vn.readout_module().R1["batch"] * FRAME_BYTES
    result = {"schema": SCHEMA_PROJECTION, "plan_identity": IDENTITY, "measured_at": eb.utc_now(),
              "label": "[INFERENCE] CPU micro-benchmark (4 threads, fp32, FlopCounterMode): FLOPs of one official "
                       "clip update (8 frames, TF + rollout, forward + backward) x 256 clips per update, one decision "
                       "(anchor encode + 45-step rollouts of 16 candidates + R1), one ViT-g frame, one R1 step; mapped "
                       "to an RTX 5090 at 100 (lower) / 40 (upper) sustained bf16 TFLOP/s, each update / R1 step "
                       "bounded below by reading its features from the cache at 3 GB/s",
              "cpu": cpu, "decision": dec, "r1_step_flops": r1, "feature_rows": rows, "update_read_bytes": update_bytes,
              "r1_read_bytes": r1_bytes, "gpu_model": GPU_MODEL, "native_updates": vn.NATIVE_UPDATES,
              "global_batch": vn.GLOBAL_BATCH}
    for tag, model in GPU_MODEL.items():
        tf = model["tflops"] * 1e12
        rd = model["read_gb_s"] * 1e9
        rates = {n: max(cpu[n]["update_flops"] / tf, update_bytes / rd) for n in CONFIGS}
        result[f"projection_{tag}"] = child_projection(
            rates, dec["decision_flops"] / tf, dec["encoder_flops_per_frame"] / tf, max(3 * r1 / tf, r1_bytes / rd),
            GFID_GPU_H[tag], rows)
    result["primary"] = "projection_upper"
    timing_path = output / "timing.json"
    if timing_path.is_file():
        record = eb.read_json(timing_path)
        if str(record.get("device", "")).startswith("cuda"):
            result["projection_measured"] = measured_projection(record, rows)
            result["timing_sha256"] = eb.sha256_of(timing_path)
            result["primary"] = "projection_measured"
    eb.write_json(output / "projection.json", result)
    lo, up = (result[f"projection_{t}"]["total_gpu_h"] for t in ("lower", "upper"))
    log(f"projection [INFERENCE]: {lo:.0f}-{up:.0f} GPU-h (remainder {REMAINDER_GPU_HOURS}, cap "
        f"{CHILD_CAP_GPU_HOURS}); primary {result['primary']}")
    return 0


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timing(output, updates=6, decisions=4, frames=512, device=None):
    """--timing probe (<= 1 GPU-h; enqueued only after its own posted decision; no funded.json needed), exclusive
    GPU: (1) the frozen ViT-g encoder on ``frames`` fit frames (frames / s, after a warm-up batch); (2) ``updates``
    full official updates (256 clips, 32 micro-batches, bf16 autocast + GradScaler) of native and ar4 (lr-half = native
    per update) on clips of those encoded frames (64 shots x 8 frames, in GPU memory: the cache read bound is modelled,
    not measured); (3) the scoring path (anchor encode + 45-step rollouts of 16 grid candidates + R1) from
    ``decisions`` fit anchors under deterministic_scoring(); (4) 50 R1 steps (batch 128). Keeps no model."""
    device = device_of(device)
    output = Path(output)
    began = time.monotonic()
    shots = eb.FitShots()
    result = {"schema": SCHEMA_TIMING, "plan_identity": IDENTITY, "device": device,
              "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
              "torch": torch.__version__, "configs": {}}
    encoder = frozen_encoder(device)
    pick = [(s, p) for s in range(frames // vn.CLIP_FRAMES) for p in vn.POSITIONS[:vn.CLIP_FRAMES]]
    pix = torch.stack([torch.as_tensor(shots.frame(s, min(p, int(shots.ends[s])))) for s, p in pick])
    vn.encode(encoder, pix[:FEATURE_BATCH])
    _sync(device)
    t = time.monotonic()
    tokens = torch.cat([vn.encode(encoder, pix[i:i + FEATURE_BATCH]).half()
                        for i in range(0, len(pix), FEATURE_BATCH)])
    _sync(device)
    result["encoder"] = {"frames": len(pix), "seconds": time.monotonic() - t,
                         "seconds_per_frame": (time.monotonic() - t) / len(pix)}
    clips = vn.normalize(tokens.float().view(-1, vn.CLIP_FRAMES * vn.TOKENS, vn.EMBED_DIM))
    launch = torch.as_tensor(np.asarray(shots.launch5))[: len(clips)].float()
    for name in ("native", "ar4"):
        spec = config_spec(name)
        torch.manual_seed(eb.SELECTION_SEED)
        model = build_model(spec, device)
        optimizer, scheduler, wd_scheduler = vn.build_optimizer(model, spec)
        scaler = _scaler(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        seconds, losses = [], []
        for u in range(updates + 1):
            g = eb.batch_rng(eb.SELECTION_SEED, u)
            idx = g.integers(0, len(clips), size=spec["batch"])
            micro = [(clips[idx[i:i + spec["micro_batch"]]].to(device),
                      vn.clip_actions(launch[idx[i:i + spec["micro_batch"]]], [0] * spec["micro_batch"]).to(device),
                      vn.clip_states(spec["micro_batch"]).to(device))
                     for i in range(0, spec["batch"], spec["micro_batch"])]
            _sync(device)
            t = time.monotonic()
            losses.append(vn.train_step(model, optimizer, scheduler, wd_scheduler, micro, spec["auto_steps"], scaler,
                                        _autocast(device))["loss"])
            _sync(device)
            if u > 0:  # first update = warm-up
                seconds.append(time.monotonic() - t)
        result["configs"][name] = {"updates": updates, "seconds_per_update_steady_median": float(np.median(seconds)),
                                   "seconds": seconds, "finite": bool(np.all(np.isfinite(losses))),
                                   "parameters": vn.trainable_parameters(model),
                                   "peak_gpu_allocated_gib": (torch.cuda.max_memory_allocated() / 2 ** 30
                                                              if device.startswith("cuda") else None)}
        if name == "native":
            model.eval().requires_grad_(False)
            probe = vn.new_readout().to(device).eval().requires_grad_(False)
            fa = eb.fit_anchor_frames()
            members = sorted(fa)[:decisions + 1]
            items, _ = grid_actions()
            costs = objective_costs()
            with harness.deterministic_scoring():
                level_rows(model, probe, items, anchor_tokens(encoder, fa[members[0]][None])[0], costs)  # warm-up
                _sync(device)
                t = time.monotonic()
                for m in members[1:]:
                    level_rows(model, probe, items, anchor_tokens(encoder, fa[m][None])[0], costs)
                _sync(device)
            result["scoring"] = {"decisions": decisions, "candidates": len(items), "seconds": time.monotonic() - t,
                                 "seconds_per_decision": (time.monotonic() - t) / decisions}
        del model, optimizer
    probe = vn.new_readout().to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    x = clips[:1].view(-1, vn.TOKENS, vn.EMBED_DIM)[:1].expand(vn.readout_module().R1["batch"], -1, -1).to(device)
    seconds = []
    for k in range(51):
        _sync(device)
        t = time.monotonic()
        loss = probe.forward_z(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        _sync(device)
        if k:
            seconds.append(time.monotonic() - t)
    result["r1"] = {"steps": 50, "seconds_per_step_median": float(np.median(seconds))}
    result["configs"]["lr-half"] = {**result["configs"]["native"], "note": "same update as native"}
    result["seconds"] = time.monotonic() - began
    result["written_at"] = eb.utc_now()
    eb.write_json(output / "timing.json", result)
    eb.mark(output, "timing", result["seconds"], result["seconds"] if device.startswith("cuda") else 0.0)
    log("timing: " + ", ".join(f"{k} {v['seconds_per_update_steady_median']:.2f} s/update"
                               for k, v in result["configs"].items())
        + f"; {result['scoring']['seconds_per_decision']:.2f} s/decision; "
          f"{result['encoder']['seconds_per_frame'] * 1e3:.2f} ms/frame")
    if device.startswith("cuda") and (output / "projection.json").is_file():
        projection = eb.read_json(output / "projection.json")
        projection.update(projection_measured=measured_projection(result, feature_rows(output)),
                          primary="projection_measured", timing_sha256=eb.sha256_of(output / "timing.json"))
        eb.write_json(output / "projection.json", projection)
    return 0


def measured_projection(record, rows):
    """The measured projection: GPU s/update (max with the 3 GB/s cache read bound), s/decision, s/frame, s/R1 step
    (max with its read bound); G-fid at the upper [INFERENCE] value."""
    read = GPU_MODEL["upper"]["read_gb_s"] * 1e9
    update_bytes = vn.GLOBAL_BATCH * vn.CLIP_FRAMES * FRAME_BYTES
    r1_bytes = vn.readout_module().R1["batch"] * FRAME_BYTES
    rates = {k: max(v["seconds_per_update_steady_median"], update_bytes / read) for k, v in record["configs"].items()}
    return {"source": "timing.json (measured)",
            **child_projection(rates, record["scoring"]["seconds_per_decision"],
                               record["encoder"]["seconds_per_frame"],
                               max(record["r1"]["seconds_per_step_median"], r1_bytes / read), GFID_GPU_H["upper"],
                               rows)}


def smoke(output, device=None):
    device = device_of(device)
    output = Path(output)
    shots = eb.FitShots()
    selected = selected_config(output)
    selection_record = eb.read_json(output / "selection.json")
    accounting = {"fit_shots": len(shots), "fit_shots_expected": FIT_SHOTS, "fit_levels": len(shots.levels),
                  "fit_positive": int(shots.verdicts.sum()), "fit_anchors": len(eb.fit_anchors()["carriers"]),
                  "policy_grid_mixed_cells": len(eb.policy_cells(GRID)), "clip_frames": vn.CLIP_FRAMES,
                  "clip_starts_per_shot": CLIP_STARTS,
                  "shots_ending_before_225": int((np.asarray(shots.ends) < vn.ENDPOINT).sum()),
                  "r1_fit_frames": len(r1_items(shots.ends)), "feature_plan": eb.read_json(feature_paths(output)["plan"])}
    gport = eb.read_json(output / "gport/gport.json")
    adapter = eb.sha256_of(ROOT / "world_model/baselines/vjepa2/novphy.py")
    gport_check = {"pass": gport.get("pass") is True, "adapter_current": gport.get("adapter_sha256") == adapter}
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
    training = [(f"{arm}--seed{seed}", spec["updates"] * rate) for arm, seed, spec, _ in remaining]
    selection_seconds = sum(t["gpu_seconds"] for t in throughput.values())
    decision_s = eb.read_json(output / "timing.json")["scoring"]["seconds_per_decision"]
    readout_seconds = 3 * READOUT_DECISIONS * decision_s
    score_seconds = 3 * SCORING_DECISIONS * decision_s
    queues, loads = lpt(training)
    total = selection_seconds + sum(s for _, s in training) + readout_seconds + score_seconds
    wall = max(loads) + readout_seconds / 3 + score_seconds
    projection = {"selection_gpu_h": selection_seconds / 3600, "training_gpu_h": sum(s for _, s in training) / 3600,
                  "readouts_gpu_h": readout_seconds / 3600, "score_gpu_h": score_seconds / 3600,
                  "total_gpu_h": total / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "native_per_seed_gpu_h": arm_spec(ARM, selected)["updates"] * rate / 3600,
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

OWN_FILES = [RUNNER, TESTS, "world_model/baselines/vjepa2/__init__.py", "world_model/baselines/vjepa2/novphy.py",
             "scripts/b127_vjepa2_port_reference.py", "scripts/b127_vjepa2_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/vjepa2/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "fidelity/band.json", output / "fidelity/gfid.json",
              output / "smoke.json", output / "selection.json", output / "funded.json", output / "timing.json",
              readiness_dir(output) / "readiness.json", readiness_dir(output) / "r1.pt",
              feature_paths(output)["manifest"], feature_paths(output)["fit_index"],
              feature_paths(output)["policy_index"]]
    paths += [selection_dir(output, name) / f for name in CONFIGS for f in ("model.pt", "report.json")]
    return [{"name": f"t6_{rel(p)}", "artifact": rel(p), "sha256": eb.sha256_of(p)}
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
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS,
        "arms": {arm: {**ARMS[arm], "spec": arm_spec(arm, selected), "seeds": list(eb.SEEDS)} for arm in ARMS},
        "training_precision": TRAINING_PRECISION, "scoring_policy": harness.SCORING_POLICY,
        "reused": {ARM: {"seed": eb.SELECTION_SEED, "run": rel(selection_dir(output, selected))}},
        "readout": {"rollout": "decision-frame latent (frozen ViT-g AC encoder, normalised), 45 autoregressive "
                               "step_predictor steps (launch at step 0, no-ops after, zero states) over a sliding "
                               "context of 7 frames; primary = tie-free EndpointCosts of R1(predicted frame-225 latent) "
                               "vs R1(decision-frame latent)",
                    "readout": "R1 = #130 PatchReadout fitted once on normalised true fit-frame features at "
                               "rc.probe_positions (E99 targets), shared by every configuration and seed "
                               "(readiness/r1.pt)",
                    "g_readout": "policy ceiling on true-frame features >= 0.65 (readiness record)",
                    "action_sensitivity": "eb.action_sensitivity on R1 carriers at frame 225, 480 fit anchors x 16 "
                                          "grid launches; carrier-space calibration reported beside it"},
        "queues": {"train": {str(k): v for k, v in queues.items()},
                   "readouts": {str(k): v for k, v in readout_queues.items()}, "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": [], "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "T6 is a coverage arm (#127 rule P-coverage; funded by the measured projection): secondary, reported, "
            "never in the program claim.",
            "T6's configuration was selected on the policy split among three pre-declared configurations (native, ar4, "
            "lr-half); U had no such tuning (#127).",
            "Frozen released ViT-g AC encoder at 256 x 256 (4:3 -> 1:1, 16 px patches): a pig is a few pixels wide; the "
            "readiness ceiling on true-frame features measures what R1 can recover.",
            "The AC predictor is trained from scratch on NovPhy (as V-JEPA 2-AC on DROID); the launch is the action "
            "token at t0, zero state tokens; the config's predictor mask size crashes upstream, so the predictor is "
            "built with the hub's num_frames 64 (identical mask content).",
            "G-fid reproduces a published V-JEPA 2 encoder number (Diving-48 ViT-L); no V-JEPA 2-AC number is "
            "reproducible with released assets (robot success rates).",
            "No native success head: no native-head score and no U + head control; G-collapse not applied (frozen "
            "encoder latent read through R1; G-readout applies).",
            "P-novelty: after the primary scoring pass, T6 (if its guards pass) is scored on the #110 novelty levels "
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

def readouts_arm(output, plan, arm, seed, device=None):
    out_dir = readouts_dir(output, arm, seed)
    if (out_dir / "readouts.json").is_file():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_of(device)
    costs = objective_costs()
    began = time.monotonic()
    result = {"arm": arm, "seed": seed, "plan_identity": IDENTITY}
    path = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
    readiness_record = eb.read_json(readiness_dir(output) / "readiness.json")
    with harness.deterministic_scoring() as policy:
        result["scoring_policy"] = policy
        model, payload = load_trained(path, device)
        result["model"] = rel(path)
        result["model_sha256"] = eb.sha256_of(path)
        if model is None:
            result.update(typed_failure=payload["typed_failure"]["typed_failure"])
        else:
            probe, _ = load_r1(output, device)
            encoder = frozen_encoder(device)
            fa = eb.fit_anchor_frames()
            members = sorted(fa)
            z0 = anchor_tokens(encoder, torch.stack([fa[m] for m in members]))
            items, actions = grid_actions()
            out = predict(model, probe, z0, actions[None].expand(len(members), -1, -1))
            ends = out["carriers"][vn.STEPS]
            matrix = tie_free_matrix(ends, out["anchor"], costs)
            action = float(eb.action_sensitivity(ends))
            result.update(typed_failure=None, G_rank=eb.g_rank(matrix), G_action=action,
                          G_action_calibration=g_action_calibration(action), G_collapse=None,
                          G_collapse_note="not applied: frozen-encoder latent read through R1 (G-readout)",
                          policy_ceiling_auc=readiness_record["ceiling_auc"],
                          readout_artifact=readiness_record["r1"]["artifact"],
                          readout_sha256=readiness_record["r1"]["sha256"],
                          fit_anchor_nonfinite_share=float(np.isnan(matrix).mean()),
                          trainable_parameters=vn.trainable_parameters(model))
            prow = policy_endpoint_rows(model, probe, costs, policy_anchor_latents(output))
            result["policy_auc"] = eb.policy_grid_auc(prow, eb.PRIMARY_COST)
            result["macs_per_candidate"] = eb.macs_per_candidate(
                lambda z, a: predict(model, probe, z, a), z0[:1], actions[None], candidates=len(items))
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
    for name in plan["queues"]["readouts"].get(str(queue), []):
        arm, seed = name.split("--seed")
        readouts_arm(output, plan, arm, int(seed), device)
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
    readout = readiness_dir(output) / "r1.pt"
    entries = {}
    for arm, seed in arm_seeds():
        rdir = readouts_dir(output, arm, seed)
        if not (rdir / "readouts.json").is_file():
            raise SystemExit(f"readouts of {arm} seed {seed} missing")
        mpath = model_dir(output, arm, seed, plan["selected"]) / "model.pt"
        entries[f"{arm}--seed{seed}"] = {
            "readouts": rel(rdir / "readouts.json"), "readouts_sha256": eb.sha256_of(rdir / "readouts.json"),
            "model": rel(mpath), "model_sha256": eb.sha256_of(mpath), "readout": rel(readout),
            "readout_sha256": eb.sha256_of(readout)}
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
    """(model, probe) of a frozen arm-seed, or (None, reason) for a typed failure."""
    model, payload = load_trained(model_dir(output, arm, seed, plan["selected"]) / "model.pt", device)
    if model is None:
        return None, payload["typed_failure"]["typed_failure"]
    probe, _ = load_r1(output, str(next(model.parameters()).device))
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
    encoder = frozen_encoder(device)
    members_all = sorted(anchors["frames"])
    z = dict(zip(members_all, anchor_tokens(encoder, torch.stack([anchors["frames"][m] for m in members_all]))))
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {member: {"rows": level_rows(model, probe, entry["inventory"], z[member], costs)}
                                  for member, entry in sorted(cells[inventory]["mixed"].items())}
    record = {**base, "typed_failure": None, "inventories": inventories}
    fit = eb.fit_anchor_frames()
    fm = sorted(fit)
    zf = dict(zip(fm, anchor_tokens(encoder, torch.stack([fit[m] for m in fm]))))
    items, actions = grid_actions()
    record["wrong_anchor"] = {m: {"rows": level_rows(model, probe, items, zf[m], costs)} for m in fm}
    cache = eb.position_cache()
    branches = eb.position_branches(cells)
    keep = tuple(sorted(set(POSITION_STEPS.values())))
    acts = torch.stack([eb.launch5(it["action"]) for _, it in branches])
    out = predict(model, probe, torch.stack([z[m] for m, _ in branches]), acts[:, None], keep=keep)
    at = {t: out["carriers"][s][:, 0] for t, s in POSITION_STEPS.items()}
    record["positions"] = _position_rows(at, branches, cache, eb.POSITION_TIMES)
    grid_members = sorted(cells[GRID]["mixed"])[:20]
    record["wall_per_decision"] = eb.wall_per_decision(
        lambda m: predict(model, probe, anchor_tokens(encoder, anchors["frames"][m][None]), actions[None]),
        grid_members)
    return record


def score(output, device=None):
    output = Path(output)
    if refuse_unfunded(output, "score"):
        return 1
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#138 gate); evaluation not read")
        return 1
    plan = load_plan(output)
    load_manifest(output)
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

SCHEMA_NOVELTY_RECORD = "issue_138_novelty_record_v1"
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
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_138_novelty_arms_v1", "arms": status,
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
                encoder = frozen_encoder(device_of(device))
                grid = {level: {"rows": level_rows(model, probe, cells["levels"][level]["inventory"],
                                                   anchor_tokens(encoder, anchors["frames"][level][None])[0], costs)}
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
        "G_isolation": {"pass": True, "rule": "fit split trains and fits R1, policy split selects and gives the R1 "
                        "ceiling; evaluation anchors are read only by --score after --freeze-readouts (G_chronology) "
                        "and after G-fid passed"},
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
        "coverage_arm": True, "readiness": eb.read_json(readiness_dir(output) / "readiness.json"),
        "readouts": readouts_view, "novelty": novelty_section(output, records),
        "native_success_head": None, "u_plus_head": "not applicable: V-JEPA 2-AC has no success head"})
    compute = {"schema": "issue_138_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} T6 V-JEPA 2-AC (coverage arm): findings", "",
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
    lines += ["", "T6 is a coverage arm: never in the program claim. Intervals are descriptive (member-clustered "
              "bootstrap); tokens follow the #127 delta rule.", ""]
    return "\n".join(lines)


def not_run_reason(output):
    """The typed reason T6 is not run (no plan.json): the readiness stop, or the orchestrator's not-run.json marker
    (e.g. {"typed_reason": "cap", "decision": url}); None while T6 may still run."""
    output = Path(output)
    rpath = readiness_dir(output) / "readiness.json"
    if rpath.is_file() and not eb.read_json(rpath).get("pass"):
        return eb.read_json(rpath).get("typed_reason") or TYPED_STOP
    marker = output / "not-run.json"
    if marker.is_file():
        return eb.read_json(marker)["typed_reason"]
    return None


def publish_not_run(output, reason):
    """summary.json of a T6 that was not run (typed): no arm block, extra.pending_arms {T6: reason} (listed as not
    run by the program summary), the G-port / G-fid / readiness / projection records that exist."""
    output = Path(output)
    gfid_path = output / "fidelity/gfid.json"
    rpath = readiness_dir(output) / "readiness.json"
    ppath = output / "projection.json"
    gfid_block = {ARM: {"pass": gfid_pass(output), "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    readiness_record = eb.read_json(rpath) if rpath.is_file() else None
    projection = None
    if ppath.is_file():
        full = eb.read_json(ppath)
        projection = {"primary": full["primary"], "sha256": eb.sha256_of(ppath),
                      "total_gpu_h": full[full["primary"]]["total_gpu_h"],
                      "range_gpu_h": [full[f"projection_{t}"]["total_gpu_h"] for t in ("lower", "upper")
                                      if f"projection_{t}" in full],
                      "remainder_gpu_h": REMAINDER_GPU_HOURS}
    summary = eb.child_summary(ISSUE, IDENTITY, None, {}, {}, {}, gfid_block, {
        "pending_arms": {ARM: reason}, "coverage_arm": True, "G_port": gport_pass(output),
        "readiness": readiness_record, "projection": projection,
        "not_run_marker": eb.read_json(output / "not-run.json") if (output / "not-run.json").is_file() else None,
        "decisions": DECISIONS})
    eb.write_json(output / "summary.json", summary)
    ceiling = "" if readiness_record is None else (
        f" Readiness ceiling {readiness_record['ceiling_auc']} (threshold {readiness_record['threshold']}; E99 "
        f"true-carrier ceiling {readiness_record['e99_true_carrier_ceiling_auc']}).")
    proj = "" if projection is None else (
        f" Projection ({projection['primary']}): {projection['total_gpu_h']:.0f} GPU-h against a remainder of "
        f"{REMAINDER_GPU_HOURS:.0f} GPU-h.")
    (output / "findings.md").write_text(
        f"# #{ISSUE} T6 V-JEPA 2-AC (coverage arm): not run\n\nTyped reason: `{reason}`.{ceiling}{proj} G-port "
        f"{gport_pass(output)}; G-fid {gfid_pass(output)}. No records; T6 is listed as not run in the program "
        f"summary.\n")
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
    modes.add_argument("--features", choices=("readiness", "training"))
    modes.add_argument("--feature-plan", action="store_true", help="CPU: write the feature-cache layout and sizes")
    for mode in ("readiness", "project", "timing", "selection", "selection-readouts", "smoke", "prepare", "train",
                 "readouts", "freeze-readouts", "score", "score-novelty", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--workers", type=int, default=8, help="--features: frame decode workers")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--timing-updates", type=int, default=6, help="--timing: updates per configuration")
    parser.add_argument("--timing-decisions", type=int, default=4, help="--timing: fit anchors scored")
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B138_CPU_THREADS", "4")))  # shared host: avoid oversubscription
    if args.features:
        return features(output, args.features, args.device, args.workers)
    if args.feature_plan:
        return plan_features(output)
    if args.readiness:
        return readiness(output, args.device)
    if args.project:
        return project(output)
    if args.timing:
        return timing(output, args.timing_updates, args.timing_decisions, device=args.device)
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
