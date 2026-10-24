"""Issue #133 (T1 DreamerV3) G-fid, Tier T: the DINO-WM Table 1 DreamerV3 PushT row (0.30), rebuilt.

DINO-WM (arXiv 2411.04983v2, Sec. 4.2, A.9) trained NM512/dreamerv3-torch offline on its PushT dataset without reward
or task information and ran MPC-CEM over the learned world model; it released no baseline code. This harness rebuilds
it from the vendored official world model (``world_model/baselines/dreamerv3``) and DINO-WM's unchanged planner
(``planning/``, ``PlanWorkspace``, ``conf/plan_pusht.yaml``) at dino_wm @ 0a9492f. Every reconstruction choice is
Decision P-other on #133 (issuecomment-6388719795); the band is frozen in ``fidelity/band.json`` (Decision P-fid,
issuecomment-6388719519).

Stages (each refuses to run without the orchestrator's ``.local-artifacts/issue-133-dreamerv3-v1/funded.json``):
    --build-cache  (env ``b127-dino_wm``, CPU) the train split through DINO-WM's own ``PushTDataset`` (normalised
                   2-d actions, normalised 4-d proprio) into memmaps on local NVMe (``ISSUE133_PUSHT_CACHE``, default
                   ~/.cache/novphy-issue-133-pusht64): frames resized 224 -> 64 (``to_dreamer_image``), proprio,
                   actions. Resumable per episode (SIGTERM -> progress.json -> exit 75).
    --train        (env ``novphy``, one GPU through the scheduler) the official ``WorldModel._train`` on the cache:
                   batch 16 x 64 (upstream sample_episodes rows over the 5 frameskip phases of every episode), 10-d
                   actions (5 frames x 2), seed 0, ``dn.updates_for`` updates (replay ratio 512). Resumable
                   (SIGTERM -> resume.pt -> exit 75).
    --plan         (env ``b127-dino_wm``, one GPU through the scheduler) DINO-WM's ``PlanWorkspace`` with
                   ``plan_pusht.yaml`` (max_iter from band.json) over ``PlanningModel`` (the DreamerV3 latent);
                   resumable through #130's MPC-iteration checkpoint (``install_mpc_checkpointing``); writes gfid.json.
    --record RUN_DIR  re-derives ``fidelity/gfid.json`` (CPU).
Not gated (CPU, nothing recorded):
    --smoke        the planning pipeline on CPU with an untrained model (2 cases, 4 samples, 2 CEM steps); with
                   --run-dir / --stop-after it exercises the resume path (tests).
    --targets-check RUN_DIR  builds the official PlanWorkspace targets (50 cases, seed 99) and compares them with
                   #130's ``plan_targets.pkl`` (the same start / goal pairs as DINO-WM's own Tier-E run).
    --project      CPU micro-benchmarks scaled to GPU-h ([INFERENCE]) -> ``fidelity/projection.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / ".local-artifacts/issue-133-dreamerv3-v1"
FIDELITY = OUTPUT / "fidelity"
MODEL_DIR = FIDELITY / "model"
OFFICIAL = Path(os.environ.get("DINO_WM_REF", "/mnt/array/sukaih/novphy-baselines-src/dino_wm@0a9492f"))
EXTERNAL = ROOT / "data/external/dino_wm"
CKPT_BASE = EXTERNAL / "checkpoints"
DATASET_DIR = EXTERNAL / "datasets"
CACHE = Path(os.environ.get("ISSUE133_PUSHT_CACHE", Path.home() / ".cache" / "novphy-issue-133-pusht64"))
DINO_COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
SCHEMA = "issue_127_gfid_v1"
SCHEMA_CACHE = "issue_133_pusht64_cache_v1"
SCHEMA_MODEL = "issue_133_pusht_model_v1"
IMAGE = 64
FRAMESKIP = 5
ACTION_DIM = 2  # DINO-WM's PushT relative action
PROPRIO_DIM = 4  # agent position + velocity
SEED = 0  # upstream configs.yaml defaults seed
BLOCKS = ("defaults", "dmc_vision")
OVERRIDES = {"encoder.mlp_keys": "proprio", "decoder.mlp_keys": "proprio"}
SHAPES = {"image": (IMAGE, IMAGE, 3), "proprio": (PROPRIO_DIM,), "is_first": (), "is_last": (), "is_terminal": ()}
SMOKE = {"n_evals": 2, "planner.sub_planner.num_samples": 4, "planner.sub_planner.topk": 2,
         "planner.sub_planner.opt_steps": 2, "n_plot_samples": 1}
ISSUE130_TARGETS = ROOT / ".local-artifacts/issue-130-dino-wm-v1/fidelity/runs/tier-e-2/plan_targets.pkl"
STOP = []


def _dn():
    """The adapter module (imported as a package-relative module so that the py3.10 ``b127-dino_wm`` env never runs
    the NovPhy ``world_model/__init__.py``)."""
    try:
        from world_model.baselines.dreamerv3 import novphy
        return novphy
    except ImportError:
        base = str(ROOT / "world_model" / "baselines")
        if base not in sys.path:
            sys.path.insert(0, base)
        from dreamerv3 import novphy
        return novphy


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def funded(output=OUTPUT):
    """The orchestrator's funding marker (Decision P-coverage funding): present, a JSON object, not ``funded: false``."""
    path = Path(output) / "funded.json"
    if not path.is_file():
        return False
    try:
        marker = read_json(path)
    except ValueError:
        return False
    return isinstance(marker, dict) and marker.get("funded", True) is True


def refuse_unfunded(stage, output=OUTPUT):
    if funded(output):
        return False
    print(f"[issue-133] {stage} refused: {Path(output) / 'funded.json'} absent (no P-coverage funding decision); "
          "nothing ran", flush=True)
    return True


def mark(phase, wall, gpu, output=OUTPUT, **extra):
    """#123 ledger entry (output/ledger/<phase>.json, accumulated)."""
    path = Path(output) / "ledger" / f"{phase}.json"
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})


def install_stop_handlers():
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda s, _f: STOP.append(s))


# ---------------------------------------------------------------------------------------------------------------
# model and images
# ---------------------------------------------------------------------------------------------------------------

def to_dreamer_image(x01):
    """[N, 3, H, W] float in [0, 1] -> uint8 [N, 64, 64, 3]: antialiased bilinear resize, rounded (cache and planner)."""
    x = F.interpolate(torch.as_tensor(x01, dtype=torch.float32), size=(IMAGE, IMAGE), mode="bilinear",
                      align_corners=False, antialias=True)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def pusht_config(device="cpu"):
    return _dn().build_config(OVERRIDES, blocks=BLOCKS, device=device, num_actions=FRAMESKIP * ACTION_DIM)


def build_pusht_model(device="cpu"):
    """The PushT DreamerV3 world model (official dmc_vision CNN at 64 x 64 + the proprio MLP, no reward / continue
    heads; G-port arm pusht-64)."""
    dn = _dn()
    return dn.WorldModel(SHAPES, pusht_config(device)).to(device)


def load_pusht_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL or not payload.get("complete"):
        raise ValueError(f"{path}: not a complete PushT DreamerV3 model")
    model = build_pusht_model(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


class PlanningModel(torch.nn.Module):
    """DINO-WM's world-model interface over DreamerV3 (Decision P-other, item 6). enc(o): the posterior mode from the
    initial state given that single frame; rollout: ``img_step`` modes under the 10-d actions; z: the categorical
    probabilities (32 x 32, unimix) of the stochastic state as ``visual`` [b, t, 1, 1024]; ``proprio`` is zero (the
    latent already embeds proprio), so DINO-WM's objective is the MSE between probabilities. ``decoder = None``: the
    evaluator skips its plots."""

    def __init__(self, wm):
        super().__init__()
        self.wm = wm
        self.decoder = None

    def _posterior(self, visual, proprio):
        n = visual.shape[0]
        image = to_dreamer_image(visual.float() * 0.5 + 0.5).float()  # DINO-WM's Normalize(0.5, 0.5) undone
        device = next(self.wm.parameters()).device
        obs = self.wm.preprocess({"image": image[:, None], "proprio": proprio.float()[:, None],
                                  "is_first": torch.ones(n, 1), "is_terminal": torch.zeros(n, 1)})
        embed = self.wm.encoder(obs)[:, 0]
        state, _ = self.wm.dynamics.obs_step(None, None, embed, obs["is_first"][:, 0].to(device), sample=False)
        return state

    def latent(self, state):
        probs = self.wm.dynamics.get_dist(state).base_dist.probs
        return probs.reshape(probs.shape[0], -1)

    @torch.no_grad()
    def encode_obs(self, obs):
        visual, proprio = obs["visual"], obs["proprio"]
        b, t = visual.shape[:2]
        z = self.latent(self._posterior(visual.reshape(b * t, *visual.shape[2:]), proprio.reshape(b * t, -1)))
        z = z.reshape(b, t, 1, -1)
        return {"visual": z, "proprio": torch.zeros(b, t, 1, device=z.device)}

    @torch.no_grad()
    def rollout(self, obs_0, act):
        state = self._posterior(obs_0["visual"][:, 0], obs_0["proprio"][:, 0])
        zs = [self.latent(state)]
        for k in range(act.shape[1]):
            state = self.wm.dynamics.img_step(state, act[:, k].float(), sample=False)
            zs.append(self.latent(state))
        z = torch.stack(zs, 1)[:, :, None]
        z_obses = {"visual": z, "proprio": torch.zeros(z.shape[0], z.shape[1], 1, device=z.device)}
        return z_obses, z


# ---------------------------------------------------------------------------------------------------------------
# --build-cache (env b127-dino_wm)
# ---------------------------------------------------------------------------------------------------------------

def official_imports():
    """sys.path = the clone (as upstream runs it); import-only stand-ins for mujoco_py / d4rl (#130's shim)."""
    from scripts import b127_dino_wm_fidelity as dwf
    dwf.install_import_stubs()
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    return dwf


def cache_paths(cache):
    cache = Path(cache)
    return {"frames": cache / "frames.u8", "proprio": cache / "proprio.f32", "action": cache / "action.f32",
            "index": cache / "index.json", "progress": cache / "progress.json"}


def open_cache(cache, mode="r", total=None):
    p = cache_paths(cache)
    total = total if total is not None else read_json(p["index"])["frames"]
    return {"frames": np.memmap(p["frames"], np.uint8, mode, shape=(total, IMAGE, IMAGE, 3)),
            "proprio": np.memmap(p["proprio"], np.float32, mode, shape=(total, PROPRIO_DIM)),
            "action": np.memmap(p["action"], np.float32, mode, shape=(total, ACTION_DIM))}


def build_cache(cache=CACHE, data_path=None, limit=None, stop_after=None):
    """Every train episode (``limit``: the first n, tests) through DINO-WM's PushTDataset (transform None: frames in
    [0, 1] at 224) into the memmaps; resumable per episode (progress.json is written after a flush)."""
    official_imports()
    from datasets.pusht_dset import PushTDataset
    data_path = Path(data_path or DATASET_DIR / "pusht_noise" / "train")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    p = cache_paths(cache)
    dset = PushTDataset(n_rollout=limit, transform=None, data_path=str(data_path), normalize_action=True,
                        with_velocity=True)
    lengths = [int(dset.get_seq_length(i)) for i in range(len(dset))]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int).tolist()
    index = {"schema": SCHEMA_CACHE, "source": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT)
             else str(data_path), "dino_wm_commit": DINO_COMMIT, "episodes": len(lengths), "lengths": lengths,
             "offsets": offsets, "frames": int(sum(lengths)), "image": [IMAGE, IMAGE, 3],
             "resize": "F.interpolate bilinear antialias=True align_corners=False on [0,1] frames, round to uint8",
             "action": "PushTDataset normalised relative action (2)", "proprio": "PushTDataset normalised (4)"}
    if p["index"].is_file():
        old = read_json(p["index"])
        if old["lengths"] != lengths or old["source"] != index["source"]:
            raise SystemExit(f"{cache}: an existing cache of a different source / episode set")
        mode = "r+"
    else:
        write_json(p["index"], index)
        mode = "w+"
    arrays = open_cache(cache, mode, index["frames"])
    done = read_json(p["progress"])["episodes_done"] if p["progress"].is_file() else 0
    install_stop_handlers()
    began = time.monotonic()
    for i in range(done, len(lengths)):
        obs, act, _, _ = dset[i]
        o, t = offsets[i], lengths[i]
        arrays["frames"][o:o + t] = to_dreamer_image(obs["visual"]).numpy()
        arrays["proprio"][o:o + t] = obs["proprio"].numpy()
        arrays["action"][o:o + t] = act.numpy()
        last = i + 1 == len(lengths)
        stop = bool(STOP) or (stop_after is not None and i + 1 - done >= stop_after)
        if (i + 1) % 100 == 0 or last or stop:
            for a in arrays.values():
                a.flush()
            write_json(p["progress"], {"episodes_done": i + 1, "complete": last, "updated_at": utc_now()})
        if stop and not last:
            print(f"[issue-133] cache paused after episode {i + 1} of {len(lengths)}", flush=True)
            return 75
    for a in arrays.values():
        a.flush()
    manifest = {"schema": SCHEMA_CACHE, "index_sha256": sha256_of(p["index"]),
                **{f"{k}_sha256": sha256_of(p[k]) for k in ("frames", "proprio", "action")},
                **{f"{k}_bytes": p[k].stat().st_size for k in ("frames", "proprio", "action")},
                "built_at": utc_now(), "seconds_last_launch": time.monotonic() - began}
    write_json(cache / "MANIFEST.json", manifest)
    write_json(p["progress"], {"episodes_done": len(lengths), "complete": True, "updated_at": utc_now()})
    print(f"[issue-133] cache complete: {len(lengths)} episodes, {index['frames']} frames", flush=True)
    return 0


def cache_complete(cache=CACHE):
    p = cache_paths(cache)
    return p["progress"].is_file() and read_json(p["progress"]).get("complete") is True and (
        Path(cache) / "MANIFEST.json").is_file()


# ---------------------------------------------------------------------------------------------------------------
# --train (env novphy)
# ---------------------------------------------------------------------------------------------------------------

def phase_episodes(lengths, offsets, frameskip=FRAMESKIP):
    """The model-step episodes: for every source episode and phase p in 0..frameskip-1, frames p, p+5, ... (every
    frame belongs to exactly one). Returns base frame index [M] and length [M] (episodes with >= 1 step)."""
    base, count = [], []
    for o, t in zip(offsets, lengths):
        for p in range(frameskip):
            n = len(range(p, t, frameskip))
            if n:
                base.append(o + p)
                count.append(n)
    return np.asarray(base, dtype=np.int64), np.asarray(count, dtype=np.int64)


def pusht_batch(arrays, base, rows, frameskip=FRAMESKIP):
    """Official training inputs of segment rows over phase episodes: image (uint8 values as float) [B, L, 64, 64, 3],
    proprio [B, L, 4], action [B, L, 10] (step k > 0: the 5 raw actions from frame k-1 to frame k; step 0: zeros),
    is_first (segment starts), is_terminal 0."""
    images, proprio, actions, firsts = [], [], [], []
    for segments in rows:
        idx, prev, first = [], [], []
        for episode, start, stop in segments:
            k = np.arange(start, stop)
            idx.append(base[episode] + frameskip * k)
            prev.append(np.where(k > 0, base[episode] + frameskip * (k - 1), -1))
            f = np.zeros(stop - start, np.float32)
            f[0] = 1.0
            first.append(f)
        idx, prev = np.concatenate(idx), np.concatenate(prev)
        images.append(torch.from_numpy(np.asarray(arrays["frames"][idx])))
        proprio.append(torch.from_numpy(np.asarray(arrays["proprio"][idx])))
        window = np.clip(prev[:, None] + np.arange(frameskip)[None], 0, None)
        act = np.asarray(arrays["action"][window.reshape(-1)]).reshape(len(idx), frameskip * ACTION_DIM)
        act[prev < 0] = 0.0
        actions.append(torch.from_numpy(act))
        firsts.append(torch.from_numpy(np.concatenate(first)))
    is_first = torch.stack(firsts)
    return {"image": torch.stack(images).float(), "proprio": torch.stack(proprio), "action": torch.stack(actions),
            "is_first": is_first, "is_terminal": torch.zeros_like(is_first)}


def batch_rng(seed, b):
    return np.random.Generator(np.random.PCG64([seed, b]))  # eb.batch_rng


def train(cache=CACHE, out_dir=MODEL_DIR, device=None, max_steps=None, steps=None, output=OUTPUT):
    """Train the PushT world model (resumable). Returns 0 (complete), 75 (paused) or 1 (typed failure)."""
    dn = _dn()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    if (out_dir / "model.pt").is_file():
        print("[issue-133] PushT model.pt exists; skipped", flush=True)
        return 0
    index = read_json(cache_paths(cache)["index"])
    arrays = open_cache(cache, "r", index["frames"])
    base, count = phase_episodes(index["lengths"], index["offsets"])
    config = pusht_config(device)
    native = dn.updates_for(index["frames"] / FRAMESKIP, config)
    steps = native if steps is None else steps
    torch.manual_seed(SEED)
    model = build_pusht_model(device)
    install_stop_handlers()

    def batch_fn(step):
        rows = dn.sample_segments(batch_rng(SEED, step), count, config.batch_length, config.batch_size)
        return pusht_batch(arrays, base, rows)

    began = time.monotonic()
    try:
        out = dn.train_loop(model, out_dir, steps, batch_fn, lambda: bool(STOP), device, max_steps=max_steps)
    except dn.TrainingPaused as paused:
        wall = time.monotonic() - began
        mark("fidelity-train", wall, wall if device != "cpu" else 0.0, output)
        print(f"[issue-133] PushT training {paused}", flush=True)
        return 75
    wall = time.monotonic() - began
    if out["failure"] is None and out["stop"] < steps:  # max_steps reached (tests, throughput)
        return 0
    complete = out["failure"] is None
    payload = {"schema": SCHEMA_MODEL, "complete": complete, "typed_failure": out["failure"], "updates": steps,
               "native_updates": native, "seed": SEED, "blocks": list(BLOCKS), "overrides": OVERRIDES,
               "parameters": dn.parameter_count(model),
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()} if complete else None}
    dn.atomic_save(out_dir / "model.pt", payload)
    report = {"complete": complete, "typed_failure": out["failure"], "updates": steps, "native_updates": native,
              "wall_seconds": out["elapsed"], "gpu_seconds": out["elapsed"] if device != "cpu" else 0.0,
              "seconds_per_update": out["elapsed"] / max(1, steps), "device": device, "torch": torch.__version__,
              "parameters": payload["parameters"], "model_sha256": sha256_of(out_dir / "model.pt"),
              "cache_manifest": read_json(Path(cache) / "MANIFEST.json") if (Path(cache) / "MANIFEST.json").is_file()
              else None, "final": out["trace"][-1] if out["trace"] else None, "finished_at": utc_now()}
    write_json(out_dir / "report.json", report)
    if (out_dir / "resume.pt").is_file() and complete:
        (out_dir / "resume.pt").unlink()
    mark("fidelity-train", wall, wall if device != "cpu" else 0.0, output,
         report=str((out_dir / "report.json").relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else None)
    print(f"[issue-133] PushT training {'complete' if complete else out['failure']['typed_failure']}", flush=True)
    return 0 if complete else 1


# ---------------------------------------------------------------------------------------------------------------
# --plan / --smoke / --targets-check (env b127-dino_wm)
# ---------------------------------------------------------------------------------------------------------------

CHECKPOINT = "mpc_checkpoint.pt"
FINAL = "mpc_final.pt"


def plan_cfg(run_dir, max_iter, overrides=None):
    """The cfg_dict official ``plan.py`` builds from ``plan_pusht.yaml`` (hydra drops ``defaults`` / ``hydra``;
    ``cfg_to_dict``), with model_name pusht (its hydra.yaml gives the env, dataset and frameskip), our checkpoint base
    path, ``planner.max_iter`` and the run directory."""
    from omegaconf import OmegaConf
    from utils import cfg_to_dict
    cfg = OmegaConf.load(OFFICIAL / "conf" / "plan_pusht.yaml")
    for key in ("defaults", "hydra"):
        cfg.pop(key, None)
    settings = {"model_name": "pusht", "ckpt_base_path": str(CKPT_BASE), "planner.max_iter": max_iter,
                **(overrides or {})}
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value, merge=False)
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["saved_folder"] = str(run_dir)
    cfg_dict["wandb_logging"] = False  # official main sets True with WANDB_MODE disabled -> the same DummyWandbRun
    return cfg_dict


def planning_workspace(run_dir, wm, max_iter, overrides=None):
    """Official ``planning_main`` with the DreamerV3 PlanningModel in place of ``load_model`` (same order: seed 99,
    dataset, model, env, PlanWorkspace) in ``run_dir`` (cwd, as hydra's run dir)."""
    import gym
    import hydra
    from omegaconf import OmegaConf
    import plan as official_plan
    from env.venv import SubprocVectorEnv
    from utils import seed

    cfg_dict = plan_cfg(run_dir, max_iter, overrides)
    os.chdir(run_dir)
    model_cfg = OmegaConf.load(CKPT_BASE / "outputs" / cfg_dict["model_name"] / "hydra.yaml")
    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist, num_pred=model_cfg.num_pred,
                               frameskip=model_cfg.frameskip)
    dset = dset["valid"]
    model = wm()
    env = SubprocVectorEnv([lambda: gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)
                            for _ in range(cfg_dict["n_evals"])])
    return official_plan.PlanWorkspace(cfg_dict=cfg_dict, wm=model, dset=dset, env=env, env_name=model_cfg.env.name,
                                       frameskip=model_cfg.frameskip, wandb_run=None), cfg_dict


def _environment():
    os.environ["DATASET_DIR"] = str(DATASET_DIR)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def run_planning(run_dir, model_path=None, max_iter=12, overrides=None, stop_after=None):
    """The official MPC-CEM evaluation of the DreamerV3 world model (``model_path`` None: an untrained model, smoke)."""
    _environment()
    dwf = official_imports()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dwf.install_mpc_checkpointing(run_dir, stop_after)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def wm():
        if model_path is None:
            torch.manual_seed(SEED)
            return PlanningModel(build_pusht_model(device).eval())
        return PlanningModel(load_pusht_model(model_path, device)[0])

    workspace, _ = planning_workspace(run_dir, wm, max_iter, overrides)
    return workspace.perform_planning()


def launch(run_dir, model_path, max_iter, overrides=None, stop_after=None):
    """The planning in a forked child; SIGTERM / SIGUSR1 are forwarded and turn into exit 75 (resume from the last
    MPC-iteration checkpoint; #130's pattern)."""
    stopped = []
    pid = os.fork()
    if pid == 0:
        try:
            run_planning(run_dir, model_path, max_iter, overrides, stop_after)
            os._exit(0)
        except BaseException as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            os._exit(0 if isinstance(exc, SystemExit) and not exc.code else 1)

    def forward(signum, _frame):
        stopped.append(signum)
        os.kill(pid, signal.SIGTERM)

    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, forward)
    while True:
        try:
            _, status = os.waitpid(pid, 0)
            break
        except InterruptedError:
            continue
    code = os.waitstatus_to_exitcode(status)
    return 75 if stopped and not (Path(run_dir) / FINAL).is_file() else code


def run_dir_for_run():
    """The unfinished run to resume (an MPC checkpoint, no final record), else a new tier-t-N."""
    base = FIDELITY / "runs"
    base.mkdir(parents=True, exist_ok=True)
    runs = sorted(base.glob("tier-t-*"), key=lambda p: int(p.name.split("-")[-1]))
    if runs and (runs[-1] / CHECKPOINT).is_file() and not (runs[-1] / FINAL).is_file():
        return runs[-1]
    n = 1 + max([int(p.name.split("-")[-1]) for p in runs] or [0])
    return base / f"tier-t-{n}"


def targets_check(run_dir, reference=ISSUE130_TARGETS):
    """Official targets (50 cases, seed 99) built with an untrained DreamerV3 model == #130's plan_targets.pkl."""
    import pickle
    _environment()
    official_imports()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    band = read_json(FIDELITY / "band.json")

    def wm():
        torch.manual_seed(SEED)
        return PlanningModel(build_pusht_model("cpu").eval())

    workspace, _ = planning_workspace(run_dir, wm, band["protocol"]["max_iter"])
    workspace.env.close()
    mine = pickle.loads((run_dir / "plan_targets.pkl").read_bytes())
    theirs = pickle.loads(Path(reference).read_bytes())

    def same(a, b):
        if isinstance(a, dict):
            return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if torch.is_tensor(a) or torch.is_tensor(b):
            return torch.equal(torch.as_tensor(a), torch.as_tensor(b))
        return np.array_equal(np.asarray(a), np.asarray(b))

    result = {k: bool(same(mine[k], theirs[k])) for k in ("obs_0", "obs_g", "state_0", "state_g", "gt_actions",
                                                         "goal_H")}
    write_json(run_dir / "targets-check.json", {"reference": str(reference), "equal": result,
                                                "pass": all(result.values()), "checked_at": utc_now()})
    print(f"[issue-133] targets equal #130's: {result}", flush=True)
    return 0 if all(result.values()) else 1


def parse_logs(run_dir):
    official_imports()
    from scripts import b127_dino_wm_fidelity as dwf
    return dwf.parse_logs(run_dir)


def record(run_dir, wall_seconds=None, exit_code=None):
    import platform
    band_path = FIDELITY / "band.json"
    band = read_json(band_path)
    curve, success, final = parse_logs(run_dir)
    n = band["protocol"]["n_evals"]
    lo, hi = band["band"]
    previous = read_json(FIDELITY / "gfid.json") if (FIDELITY / "gfid.json").is_file() else {}
    rel = str(Path(run_dir).resolve().relative_to(ROOT))
    runs = [r for r in previous.get("runs", []) if r["run_dir"] != rel]
    runs.append({"run_dir": rel, "recorded_at": utc_now(), "success_rate": success,
                 "successes": None if success is None else round(success * n), "n_evals": n, "mpc_curve": curve,
                 "mpc_iterations_run": len(curve), "final_eval": final, "wall_seconds": wall_seconds,
                 "exit_code": exit_code, "in_band": None if success is None else bool(lo <= success <= hi)})
    model_report = read_json(MODEL_DIR / "report.json") if (MODEL_DIR / "report.json").is_file() else None
    result = {"schema": SCHEMA, "issue": 133, "arm_family": "dreamerv3", "tier": "T", "paper": band["paper"],
              "band": band["band"], "band_rule": band["band_rule"], "band_sha256": sha256_of(band_path),
              "protocol": band["protocol"],
              "harness": {"dino_wm": {"repo": "https://github.com/gaoyuezhou/dino_wm", "commit": DINO_COMMIT},
                          "script": "scripts/b127_dreamerv3_fidelity.py", "script_sha256": sha256_of(Path(__file__)),
                          "decisions": ["https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388719519",
                                        "https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6388719795"]},
              "model": {"path": str((MODEL_DIR / "model.pt").relative_to(ROOT)),
                        "sha256": sha256_of(MODEL_DIR / "model.pt") if (MODEL_DIR / "model.pt").is_file() else None,
                        "report": model_report},
              "shims": ["sys.modules stand-ins for mujoco_py and d4rl.offline_env (import-only, #130)",
                        "WANDB_MODE=disabled (DummyWandbRun)", "SDL_VIDEODRIVER=dummy",
                        "PlanningModel.decoder = None (the evaluator's plots are skipped; never in success)",
                        "MPC-iteration checkpoint/resume (#130 install_mpc_checkpointing)"],
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              "runs": runs}
    scored = [r for r in runs if r["success_rate"] is not None]
    result["pass"] = bool(scored) and bool(scored[-1]["in_band"])
    result["decided_by_run"] = scored[-1]["run_dir"] if scored else None
    write_json(FIDELITY / "gfid.json", result)
    print(f"[issue-133] G-fid Tier T: success {success} band {band['band']} -> pass {result['pass']}", flush=True)
    return result


def add_wall(run_dir, seconds):
    path = Path(run_dir) / "wall.json"
    block = read_json(path) if path.is_file() else {"seconds": 0.0, "launches": 0}
    block["seconds"] += seconds
    block["launches"] += 1
    write_json(path, block)
    return block


# ---------------------------------------------------------------------------------------------------------------
# --project (CPU micro-benchmarks scaled to GPU-h; [INFERENCE])
# ---------------------------------------------------------------------------------------------------------------

THICK_SELECTION_LOGS = ROOT / ".local-artifacts/issue-131-thick-v1/selection"
PUSHT_TRAIN_FRAMES = 2_336_736  # pusht_noise/train seq_lengths sum (18 685 episodes)


def thick_gpu_seconds_per_update():
    """Measured GPU s/update of #131's THICK selection runs (native, batch 16 x 46; shared GPUs): median over the
    configurations of the last logged (seconds / step)."""
    rates = []
    for log in sorted(THICK_SELECTION_LOGS.glob("*/seed-*/train-log.jsonl")):
        lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        if lines and lines[-1]["step"] > 0:
            rates.append(lines[-1]["seconds"] / lines[-1]["step"])
    return (float(np.median(rates)) if rates else None), len(rates)


LENGTH_STEPS, THICK_STEPS = 64, 46  # sequential RSSM steps per update: DreamerV3 rows vs #131's THICK episodes
CACHE_FRAMES_PER_SECOND = 327 / 0.38  # measured: 3-episode cache build (decode + resize + write)


def cpu_seconds(fn, repeats=2, warmup=1):
    for _ in range(warmup):
        fn()
    began = time.monotonic()
    for _ in range(repeats):
        fn()
    return (time.monotonic() - began) / repeats


def thick_cpu_seconds_per_update(repeats=2):
    """CPU s/update of #131's native THICK on random carriers (same threads as the DreamerV3 benchmarks)."""
    import tempfile
    from scripts import run_issue_131_thick as thick
    from world_model.baselines.thick import novphy as tn
    g = torch.Generator().manual_seed(0)
    data = {"carriers": torch.randn(32, len(tn.POSITIONS), tn.CARRIER_DIM, generator=g) * 0.1,
            "launch5": torch.tensor([[-0.2, 0.1, 1.0, 0.0, 1.0]]).repeat(32, 1)}
    spec = thick.config_spec("sparsity-1")
    with tempfile.TemporaryDirectory() as tmp:
        thick.train_run(tmp, spec, 0, data, "cpu", max_steps=1)
        began = time.monotonic()
        report = thick.train_run(tmp, spec, 0, data, "cpu", max_steps=repeats)
        return (time.monotonic() - began) / report.get("steps_done", repeats)


def project(output=OUTPUT, repeats=2):
    """fidelity/projection.json: PushT training (CPU s/update x the THICK GPU/CPU ratio x native updates) and the
    planning stage (CPU time of one 300-sample CEM rollout x the same ratio x 12 MPC iterations x 50 cases x 30 CEM
    steps, plus the 50-env rollouts per CEM step at the CPU rate measured nowhere: an allowance) [INFERENCE]."""
    dn = _dn()
    config = pusht_config("cpu")
    native = dn.updates_for(PUSHT_TRAIN_FRAMES / FRAMESKIP, config)
    torch.manual_seed(SEED)
    model = build_pusht_model("cpu")
    rng = np.random.default_rng(0)
    b, t = config.batch_size, config.batch_length

    def batch():
        first = np.zeros((b, t), np.float32)
        first[:, 0] = 1
        return {"image": torch.from_numpy(rng.integers(0, 256, (b, t, IMAGE, IMAGE, 3)).astype(np.float32)),
                "proprio": torch.randn(b, t, PROPRIO_DIM), "action": torch.randn(b, t, 10),
                "is_first": torch.from_numpy(first), "is_terminal": torch.zeros(b, t)}

    pusht_cpu = cpu_seconds(lambda: model.train_step(batch()), repeats)
    wm = PlanningModel(model.eval())
    obs = {"visual": torch.rand(300, 1, 3, 224, 224) * 2 - 1, "proprio": torch.randn(300, 1, PROPRIO_DIM)}
    cem_cpu = cpu_seconds(lambda: wm.rollout(obs, torch.randn(300, 5, 10)), repeats)
    thick_gpu, thick_runs = thick_gpu_seconds_per_update()
    thick_cpu = thick_cpu_seconds_per_update(repeats)
    ratio = thick_gpu / thick_cpu if thick_gpu else None
    band = read_json(FIDELITY / "band.json")["protocol"]
    rollouts = band["max_iter"] * band["n_evals"] * 30  # MPC iterations x cases x CEM steps
    env_allowance_h = 0.25  # 12 x (30 + 1) evaluator env rollouts of 50 PushT envs, 25-300 env steps each
    train_h = native * pusht_cpu * ratio / 3600 if ratio else None
    plan_h = rollouts * cem_cpu * ratio / 3600 + env_allowance_h if ratio else None
    # second model: a GPU update of these RSSMs is bound by the sequential scan's kernel latency; THICK's measured GPU
    # s/update over 46 steps scaled to the 64-step DreamerV3 rows (CNN compute ignored: a lower estimate)
    latency = thick_gpu * LENGTH_STEPS / THICK_STEPS if thick_gpu else None
    result = {"schema": "issue_133_fidelity_projection_v1", "label": "[INFERENCE] CPU micro-benchmarks scaled by the "
              "measured GPU/CPU ratio of #131's native THICK update (a similar-size RSSM; conv-heavy PushT updates "
              "gain more on a GPU, so this training estimate is an upper one); the scan-latency model (THICK's GPU "
              "s/update x 64 / 46 steps) is the lower one",
              "threads": torch.get_num_threads(), "native_updates": native, "pusht_cpu_seconds_per_update": pusht_cpu,
              "cem_rollout_300x5_cpu_seconds": cem_cpu, "thick_gpu_seconds_per_update": thick_gpu,
              "thick_gpu_runs": thick_runs, "thick_cpu_seconds_per_update": thick_cpu, "gpu_over_cpu": ratio,
              "train_gpu_h": train_h, "plan_gpu_h": plan_h, "cem_rollouts": rollouts,
              "env_allowance_h": env_allowance_h, "total_gpu_h": (train_h + plan_h) if ratio else None,
              "latency_model_gpu_seconds_per_update": latency,
              "train_gpu_h_latency_model": native * latency / 3600 if latency else None,
              "total_gpu_h_latency_model": native * latency / 3600 + plan_h if latency and ratio else None,
              "cache_build_cpu_h_estimate": PUSHT_TRAIN_FRAMES / CACHE_FRAMES_PER_SECOND / 3600,
              "cache_build_rate_source": "3-episode build (327 frames in 0.38 s; b127-dino_wm, 4 threads, 2026-10-25)",
              "measured_at": utc_now()}
    write_json(Path(output) / "fidelity" / "projection.json", result)
    print(f"[issue-133] fidelity projection: {json.dumps({k: result[k] for k in ('train_gpu_h', 'plan_gpu_h')})}",
          flush=True)
    return result


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-cache", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--record", metavar="RUN_DIR")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--targets-check", metavar="RUN_DIR")
    mode.add_argument("--project", action="store_true")
    parser.add_argument("--run-dir", default=None, help="--smoke: the run directory (resumed if it has a checkpoint)")
    parser.add_argument("--max-iter", type=int, default=1, help="--smoke only")
    parser.add_argument("--stop-after", type=int, default=None, help="--smoke only: exit 75 after this MPC iteration")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if args.record:
        run_dir = Path(args.record).resolve()
        wall = read_json(run_dir / "wall.json")["seconds"] if (run_dir / "wall.json").is_file() else None
        record(run_dir, wall)
        return 0
    if args.smoke:
        run_dir = Path(args.run_dir or Path("/tmp/issue-133-gfid-smoke") / time.strftime("%Y%m%d-%H%M%S")).resolve()
        code = launch(run_dir, None, args.max_iter, SMOKE, args.stop_after)
        if code == 0:
            curve, success, _ = parse_logs(run_dir)
            print(f"[issue-133] smoke ok: curve {curve} final {success}", flush=True)
        return code
    if args.targets_check:
        return targets_check(Path(args.targets_check).resolve())
    if args.project:
        project()
        return 0
    stage = "--build-cache" if args.build_cache else "--train" if args.train else "--plan"
    if refuse_unfunded(stage, OUTPUT):
        return 1
    if not (FIDELITY / "band.json").is_file():
        raise SystemExit("fidelity/band.json missing: the band is frozen and posted before any run")
    if args.build_cache:
        return build_cache()
    if args.train:
        if not cache_complete():
            raise SystemExit(f"{CACHE}: the cache is incomplete; run --build-cache first")
        return train(device=args.device)
    if not (MODEL_DIR / "model.pt").is_file():
        raise SystemExit("fidelity/model/model.pt missing; run --train first")
    band = read_json(FIDELITY / "band.json")
    run_dir = run_dir_for_run()
    began = time.monotonic()
    code = launch(run_dir, MODEL_DIR / "model.pt", band["protocol"]["max_iter"])
    wall = time.monotonic() - began
    mark("fidelity-plan", wall, wall if torch.cuda.is_available() else 0.0, OUTPUT)
    total = add_wall(run_dir, wall)
    if code == 0 and (run_dir / FINAL).is_file():
        record(run_dir, total["seconds"], code)
    return code


if __name__ == "__main__":
    sys.exit(main())
