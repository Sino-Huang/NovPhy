"""#136 T4 DF-car: the NovPhy adapter around the vendored official Diffusion Forcing transition model
(``vendor/``, buoyancy99/diffusion-forcing, ``paper`` branch @ dd65516, MIT with its fork-credit clause).

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template
[repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in
`README.md` and the `LICENSE` file to credit the author. (Upstream fork-credit sentence, kept as upstream modules do;
``vendor/README.md`` and ``vendor/LICENSE`` are verbatim.)

Model (Decisions P-route / P-adapter on #136): the official ``DiffusionTransitionModel`` with ``TransitionMlp`` (the
paper's low-dimensional RNN: residual MLP -> stacked ``GRUCell`` latent z -> linear head), imported unchanged from
``vendor/algorithms/diffusion_forcing/models`` by file location (the package ``__init__`` files import Lightning /
wandb / GluonTS, which the novphy env lacks). x = the 236-d E99 carrier normalised by the fit-split per-feature mean /
std (the time-series loader's rule: std clamped at 1e-7); the external condition = launch5 entering the transition
out of t0 (the official condition slot ``conditions[1]``; ``_preprocess_batch`` zeroes slot 0), zeros (no-op) after;
46 frames 0, 5, ..., 225 (stride 5; positions past a shot's end repeat its final carrier, as #131 / #134); frame 0 is
the context.

Training (``Trainer.train_step``) = ``DiffusionForcingBase.training_step`` + Lightning's automatic optimisation as
the official pipeline runs it: an independent noise level per sequence element (``DiffusionTransitionModel.forward``
draws ``torch.randint`` per time step), constant SNR weight, mask-weighted mean; AdamW; Lightning's gradient-norm clip
(``gradient_clip_val``); ``optimizer_step``'s manual warm-up applied *after* each step (so update 0 runs at the full
LR, update 1 at LR / warmup, ...); 16-mixed precision on CUDA (fp16 autocast + GradScaler, unscale before the clip,
as Lightning's MixedPrecisionPlugin), fp32 on CPU. The python ``random() <= gt_cond_prob`` draws of the official step
are omitted: with ``gt_cond_prob = gt_first_frame = 0`` they select the random noise level with probability 1.

Early stopping (configuration ``ts-native``): the official time-series rule — validation CRPS-sum (``crps_quantile_sum``,
19 quantiles of 100 sampled paths, frames after the context) after every epoch of 100 batches, Lightning
``EarlyStopping(patience=6, mode="min")`` and the best checkpoint (``ModelCheckpoint(save_top_k=1)``); the validation
set is a fixed subset of fit shots (the official validation set is a random subset of the training series).

Sampling (``sample_paths``) = the official ``DiffusionForcingBase.validation_step`` prediction loop with
``context_frames = 1``, ``chunk_size = -1`` (the whole 45-step horizon) and ``uncertainty_scale = 1``: the context
frame at noise level 0, then the causal-uncertainty pyramid (row m, frame t at DDIM index clip(m - t, 0, 50), capped
at 49; 95 rows x 45 frames). ``ddim_step`` / the context step re-state ``ddim_sample_step`` / ``forward`` with the
noise passed in, so the scoring can use common random numbers (``CommonNoise``: one base draw [S, 236] per noise call,
shared by every candidate and level); with ``GlobalNoise`` they consume the global generator exactly as the official
code does (G-port).

Score: the frame-225 carrier of each of S sampled paths -> ``harness.EndpointCosts`` row; the candidate's row is the
mean over its S samples (``expected_row``); any nonfinite sample -> excluded typed row.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parent / "vendor"
MODELS_DIR = VENDOR / "algorithms" / "diffusion_forcing" / "models"
STRIDE = 5
END_FRAME = 225
POSITIONS = tuple(range(0, END_FRAME + 1, STRIDE))  # 46 frames
FRAMES = len(POSITIONS)
CONTEXT_FRAMES = 1
HORIZON = FRAMES - CONTEXT_FRAMES  # 45
CARRIER_DIM = 236
ACT_DIM = 5
STD_MIN = 1e-7  # datasets/timeseries/gluonts_dataset.py: torch.clamp(std, min=1e-7)

# configurations/algorithm/df_base.yaml `diffusion` block (paper branch)
DF_BASE_DIFFUSION = {
    "network_size": 32, "beta_schedule": "cosine", "objective": "pred_x0", "use_snr": False, "use_cum_snr": False,
    "snr_clip": 5.0, "cum_snr_decay": 0.9, "timesteps": 1000, "self_condition": False, "ddim_sampling_eta": 1.0,
    "p2_loss_weight_gamma": 0, "p2_loss_weight_k": 1, "schedule_fn_kwargs": {}, "sampling_timesteps": 50,
    "mask_unet": False, "num_gru_layers": 1, "num_mlp_layers": 0, "return_all_timesteps": False, "clip_noise": 20.0,
}
# time series: df_prediction.yaml (algorithm) + exp_prediction.yaml (experiment) + df_base.yaml defaults
TIME_SERIES = {
    "recipe": "time_series", "z_dim": 1024, "lr": 3e-4, "weight_decay": 1e-3, "betas": (0.9, 0.999),
    "warmup_steps": 2000, "gradient_clip_val": 10.0, "batch_size": 32, "precision": "16-mixed",
    "uncertainty_scale": 1, "gt_cond_prob": 0.0, "gt_first_frame": 0.0,
    "diffusion": {**DF_BASE_DIFFUSION, "network_size": 64, "objective": "pred_v", "num_gru_layers": 4,
                  "num_mlp_layers": 1},
}
# maze planning: df_planning.yaml + exp_planning.yaml (network size 32 = maze2d large; batch 2048 = paper App. D.10;
# frame_stack 1 instead of 10 and the launch through the external-condition path: Decision P-adapter)
PLANNING = {
    "recipe": "planning", "z_dim": 128, "lr": 2e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999),
    "warmup_steps": 100, "gradient_clip_val": None, "batch_size": 2048, "precision": "16-mixed",
    "uncertainty_scale": 1, "gt_cond_prob": 0.0, "gt_first_frame": 0.0,
    "diffusion": {**DF_BASE_DIFFUSION, "network_size": 32, "num_gru_layers": 1, "num_mlp_layers": 4,
                  "ddim_sampling_eta": 0.0, "beta_schedule": "linear", "cum_snr_decay": 0.98},
}
EARLY_STOPPING = {"patience": 6, "epoch_batches": 100, "validation_shots": 16, "samples": 100}  # exp_prediction.yaml
CONFIGURATIONS = {  # pre-declared (Decision P-adapter on #136); declaration order = selection tie order
    "ts-native": {**TIME_SERIES, "steps": 200_000, "early_stopping": EARLY_STOPPING},
    "ts-100k": {**TIME_SERIES, "steps": 100_000, "early_stopping": None},
    "planning": {**PLANNING, "steps": 120_005, "early_stopping": None},
}
SAMPLES = 32  # sampled paths per candidate for the expected cost (Decision P-adapter)
SAMPLE_SEED = 20261026  # CommonNoise base draws (CPU generator)
VALIDATION_SEED = 20261027  # early-stopping validation: shot subset and sampling noise


def config_of(name, overrides=None):
    config = {**CONFIGURATIONS[name], **(overrides or {})}
    config["diffusion"] = dict(config["diffusion"])
    return config


# ---------------------------------------------------------------------------------------------------------------
# the vendored model (loaded by file location)
# ---------------------------------------------------------------------------------------------------------------

_PACKAGE = "b127_dforce_vendor_models"


def vendor_models():
    """The vendored ``algorithms/diffusion_forcing/models`` package (torch + einops + packaging only)."""
    if _PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(_PACKAGE, MODELS_DIR / "__init__.py",
                                                      submodule_search_locations=[str(MODELS_DIR)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{_PACKAGE}.diffusion_transition")


def diffusion_cfg(config):
    return SimpleNamespace(**config["diffusion"])


class DiffusionForcingCarrier(nn.Module):
    """The official ``transition_model`` plus the ``data_mean`` / ``data_std`` buffers (DiffusionForcingBase names,
    so state dicts load into the official module)."""

    def __init__(self, config, mean, std):
        super().__init__()
        self.config = config
        self.transition_model = vendor_models().DiffusionTransitionModel(
            [CARRIER_DIM], [config["z_dim"]], ACT_DIM, diffusion_cfg(config))
        self.register_buffer("data_mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("data_std", torch.as_tensor(std, dtype=torch.float32).clone())

    def normalize(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        return (xs - self.data_mean.reshape(shape).to(xs.device)) / self.data_std.reshape(shape).to(xs.device)

    def unnormalize(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        return xs * self.data_std.reshape(shape).to(xs.device) + self.data_mean.reshape(shape).to(xs.device)


def build_model(config, mean, std, device="cpu"):
    """Built on CPU from the caller's torch seed (device-independent initial weights), then moved."""
    return DiffusionForcingCarrier(config, mean, std).to(device)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_device(model):
    return next(model.parameters()).device


def normalisation(carriers):
    """Per-feature mean / std [236] over every fit carrier frame (float64 accumulation; std clamped at 1e-7)."""
    flat = carriers.reshape(-1, carriers.shape[-1]).double()
    return flat.mean(0).float(), torch.clamp(flat.std(0), min=STD_MIN).float()


# ---------------------------------------------------------------------------------------------------------------
# sequences, batches
# ---------------------------------------------------------------------------------------------------------------

def episode_positions(end):
    """Frames 0, 5, ..., 225 of a shot whose last stored position is ``end`` (later positions repeat ``end``)."""
    return [min(p, int(end)) for p in POSITIONS]


def condition_sequence(launch5, frames=FRAMES):
    """[B, frames, 5]: slot 1 = the launch (conditions the transition out of t0), every other slot 0 (no-op)."""
    launch5 = torch.as_tensor(launch5, dtype=torch.float32)
    out = torch.zeros(len(launch5), frames, ACT_DIM)
    out[:, 1] = launch5
    return out


def batch_rows(rng, shots, batch):
    """Shot indices of one batch, uniform with replacement (numpy generator)."""
    return rng.integers(0, shots, size=batch)


def sequence_batch(carriers, launch5, rows, device="cpu"):
    """(xs [B, 46, 236], conditions [B, 46, 5], nonterminals [B, 46]) of fit shots ``rows`` — the official batch."""
    rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    xs = carriers[rows].float()
    conditions = condition_sequence(torch.as_tensor(launch5, dtype=torch.float32)[rows], xs.shape[1])
    return xs.to(device), conditions.to(device), torch.ones(xs.shape[:2], device=device)


def preprocess(model, xs, conditions, nonterminals):
    """``DiffusionForcingBase._preprocess_batch`` (frame_stack 1, no learnable init z): time-major normalised xs,
    conditions with slot 0 zeroed, cumulative masks [T, B], zero init z."""
    masks = torch.cumprod(nonterminals.bool().permute(1, 0), dim=0).contiguous()
    conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
    conditions = conditions.permute(1, 0, 2).contiguous()
    xs = model.normalize(xs).permute(1, 0, 2).contiguous()
    init_z = torch.zeros(xs.shape[1], model.config["z_dim"], device=xs.device)
    return xs, conditions, masks, init_z


def training_loss(model, xs, conditions, nonterminals):
    """``DiffusionForcingBase.training_step``'s loss: per frame ``transition_model(z, x_t, c_t)`` with its own random
    noise level, then the mask-weighted mean. Returns (loss, per-frame unreduced losses [T, B, 236], xs_pred)."""
    xs, conditions, masks, z = preprocess(model, xs, conditions, nonterminals)
    tm = model.transition_model
    preds, losses, cum_snr = [], [], None
    for t in range(xs.shape[0]):
        z, x_pred, loss, cum_snr = tm(z, xs[t], conditions[t], deterministic_t=None, cum_snr=cum_snr)
        preds.append(x_pred)
        losses.append(loss)
    losses = torch.stack(losses)
    weight = masks.reshape(*masks.shape, *([1] * (losses.ndim - masks.ndim)))
    return (losses * weight).mean(), losses, torch.stack(preds)


# ---------------------------------------------------------------------------------------------------------------
# training (Lightning's automatic optimisation, re-stated)
# ---------------------------------------------------------------------------------------------------------------

class Trainer:
    """Model, AdamW (``configure_optimizers``), the warm-up rule (``optimizer_step``), the gradient clip and, on
    CUDA with 16-mixed, the autocast + GradScaler of Lightning's MixedPrecisionPlugin."""

    def __init__(self, config, mean, std, device="cpu", model=None):
        self.config = config
        self.device = str(device)
        self.model = model if model is not None else build_model(config, mean, std, device)
        self.params = list(self.model.transition_model.parameters())
        self.optimizer = torch.optim.AdamW(self.params, lr=config["lr"], weight_decay=config["weight_decay"],
                                           betas=tuple(config["betas"]))
        self.mixed = self.device.startswith("cuda") and config["precision"] == "16-mixed"
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed else None
        self.global_step = 0

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.mixed)

    def train_step(self, xs, conditions, nonterminals):
        """One optimizer step; returns {"loss", "grad_norm", "lr"} (lr used by this step)."""
        self.model.train()
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            loss, _, _ = training_loss(self.model, xs, conditions, nonterminals)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        clip = self.config["gradient_clip_val"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, clip) if clip else None
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        warmup = self.config["warmup_steps"]
        if self.global_step < warmup:
            scale = min(1.0, float(self.global_step + 1) / warmup)
            for group in self.optimizer.param_groups:
                group["lr"] = scale * self.config["lr"]
        self.global_step += 1
        return {"loss": loss.detach(), "grad_norm": None if grad_norm is None else grad_norm.detach(), "lr": lr}

    def state(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "global_step": self.global_step}

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]


class TrainingPaused(Exception):
    """A stop was requested; resume.pt holds the state after the last completed update."""


def atomic_save(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def rng_state(device):
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if cuda else None}


def set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"], device)


def early_stop_update(state, value, patience):
    """Lightning ``EarlyStopping`` / ``ModelCheckpoint(save_top_k=1)`` (mode min, min_delta 0, check_finite):
    returns (improved, stop) and updates ``state`` {"best", "wait", "best_step", "history"}."""
    finite = bool(np.isfinite(value))
    improved = finite and (state["best"] is None or value < state["best"])
    if improved:
        state["best"], state["wait"] = float(value), 0
    elif finite:
        state["wait"] += 1
    stop = (not finite) or state["wait"] >= patience
    return improved, stop


def train_loop(trainer, out_dir, steps, batch_fn, should_stop, validate=None, early=None, max_steps=None,
               log_every=100, checkpoint_every=2000):
    """Updates ``trainer.train_step(*batch_fn(step))`` for steps [start, steps), resuming from ``out_dir/resume.pt``
    (model, AdamW, GradScaler, warm-up step, torch generators, early-stopping state). With ``early`` (the
    EARLY_STOPPING dict) ``validate()`` runs after every ``epoch_batches`` updates; an improvement writes ``best.pt``;
    ``patience`` validations without improvement end the run (``stopped_early``). A nonfinite loss ends the run as a
    typed failure (``nonfinite_loss``). ``should_stop()`` True -> resume.pt -> ``TrainingPaused``; ``max_steps`` bounds
    this call (tests, throughput). Returns {"start", "stop", "steps_done", "seconds", "elapsed", "failure", "trace",
    "early"}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = out_dir / "resume.pt"
    device = trainer.device
    start, elapsed = 0, 0.0
    es = {"best": None, "wait": 0, "best_step": None, "history": [], "stopped_early": False}
    if resume.is_file():
        state = torch.load(resume, map_location=device, weights_only=False)
        trainer.load_state(state["trainer"])
        set_rng_state(state["rng"], device)
        start, elapsed, es = state["step"], state["elapsed"], state["early"]
    stop_at = steps if max_steps is None else min(steps, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1

    def save_resume(next_step):
        atomic_save(resume, {"trainer": trainer.state(), "rng": rng_state(device), "step": next_step,
                             "elapsed": elapsed + time.monotonic() - began, "early": es})

    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            out = trainer.train_step(*batch_fn(step))
            loss = float(out["loss"])
            if not np.isfinite(loss):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "loss": str(loss)}
                break
            entry = None
            if step % log_every == 0 or step == steps - 1:
                entry = {"step": step, "seconds": elapsed + time.monotonic() - began, "loss": loss, "lr": out["lr"],
                         "grad_norm": None if out["grad_norm"] is None else float(out["grad_norm"])}
            if early is not None and (step + 1) % early["epoch_batches"] == 0:
                value = float(validate())
                improved, stop_early = early_stop_update(es, value, early["patience"])
                es["history"].append({"step": step + 1, "crps_sum": value, "improved": improved})
                if improved:
                    es["best_step"] = step + 1
                    atomic_save(out_dir / "best.pt", {"model": trainer.model.state_dict(), "step": step + 1,
                                                      "crps_sum": value})
                entry = {**(entry or {"step": step, "seconds": elapsed + time.monotonic() - began, "loss": loss}),
                         "validation_crps_sum": value, "improved": improved}
                if stop_early:
                    es["stopped_early"] = True
            if entry is not None:
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            if es["stopped_early"]:
                break
            stop = bool(should_stop()) and step + 1 < steps
            if stop or ((step + 1) % checkpoint_every == 0 and step + 1 < steps):
                save_resume(step + 1)
            if stop:
                raise TrainingPaused(f"paused at update {step + 1} of {steps}")
    finished = es["stopped_early"] or step + 1 >= steps
    if failure is None and not finished and step + 1 > start:  # bounded call: keep the state for the next one
        save_resume(step + 1)
    seconds = time.monotonic() - began
    return {"start": start, "stop": step + 1, "steps_done": step + 1 - start, "seconds": seconds,
            "elapsed": elapsed + seconds, "failure": failure, "trace": trace, "early": es, "finished": finished}


# ---------------------------------------------------------------------------------------------------------------
# sampling: the official prediction loop with the causal-uncertainty pyramid
# ---------------------------------------------------------------------------------------------------------------

class GlobalNoise:
    """Noise from the global torch generator, in the official call order (``torch.randn`` / ``randn_like``)."""

    def __call__(self, shape, device):
        return torch.randn(shape, device=device)


class GeneratorNoise:
    """Independent noise per row from a private generator on ``device`` (early-stopping validation)."""

    def __init__(self, seed, device):
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, shape, device):
        return torch.randn(shape, device=device, generator=self.generator)


def ddim_times(tm):
    """``ddim_sample_step``'s (time, time_next) pairs."""
    times = torch.linspace(-1, tm.num_timesteps - 1, steps=tm.sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    return list(zip(times[:-1], times[1:]))


def ddim_step(tm, x, z_cond, external_cond, index, noise_fn, pairs):
    """``DiffusionTransitionModel.ddim_sample_step`` (no guidance outputs) with the noise drawn from ``noise_fn``."""
    if index == 0:
        x = torch.clamp(x, -tm.clip_noise, tm.clip_noise)
    time_, time_next = pairs[index]
    time_cond = torch.full((x.shape[0],), time_, device=x.device, dtype=torch.long)
    pred = tm.model_predictions(x, time_cond, z_cond, external_cond=external_cond, x_self_cond=None)
    if time_next < 0:
        return pred.pred_x_start, pred.pred_z
    alpha = tm.alphas_cumprod[time_]
    alpha_next = tm.alphas_cumprod[time_next]
    sigma = tm.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    c = (1 - alpha_next - sigma**2).sqrt()
    noise = torch.clamp(noise_fn(x.shape, x.device), -tm.clip_noise, tm.clip_noise)
    return pred.pred_x_start * alpha_next.sqrt() + c * pred.pred_noise + sigma * noise, pred.pred_z


def context_step(tm, z, x, external_cond, noise_fn):
    """``transition_model(z, x, c, deterministic_t=0)`` of the official context loop (noise level 0) with the noise
    drawn from ``noise_fn``: returns (z_next, x_start)."""
    t = torch.full((x.shape[0],), 0, device=x.device).long()
    noise = torch.clamp(noise_fn(x.shape, x.device), -tm.clip_noise, tm.clip_noise)
    noised = tm.q_sample(x_start=x, t=t, noise=noise)
    pred = tm.model_predictions(noised, t, z, external_cond=external_cond, x_self_cond=None)
    return pred.pred_z, pred.pred_x_start


def pyramid(horizon, sampling_timesteps, uncertainty_scale):
    """The causal-uncertainty schedule of ``validation_step``: [rows, horizon] DDIM indices."""
    height = sampling_timesteps + int(horizon * uncertainty_scale)
    grid = np.zeros((height, horizon), dtype=int)
    for m in range(height):
        for t in range(horizon):
            grid[m, t] = m - int(t * uncertainty_scale)
    return np.clip(grid, a_min=0, a_max=sampling_timesteps, dtype=int)


class CommonNoise:
    """Common random numbers: the k-th noise call returns base draw k [S, 236] (CPU generator seeded ``seed``, cached
    per device) repeated over the rows' leading groups (rows = groups x S, sample index fastest). Every candidate,
    level and batch composition sees the same draws."""

    def __init__(self, samples=SAMPLES, seed=SAMPLE_SEED):
        self.samples, self.generator = samples, torch.Generator().manual_seed(seed)
        self.base, self.on_device, self.calls = [], {}, 0

    def reset(self):
        self.calls = 0

    def __call__(self, shape, device):
        k, self.calls = self.calls, self.calls + 1
        while len(self.base) <= k:
            self.base.append(torch.randn(self.samples, shape[-1], generator=self.generator))
        cached = self.on_device.setdefault(str(device), [])
        while len(cached) <= k:
            cached.append(self.base[len(cached)].to(device))
        base = cached[k]
        return base.unsqueeze(0).expand(shape[0] // self.samples, *base.shape).reshape(shape)


@torch.no_grad()
def sample_paths(model, z0, launch5, noise_fn, horizon=HORIZON):
    """Sampled carrier paths [1 + horizon, B, 236] (unnormalised; index 0 = the model's frame-0 reconstruction,
    index k = frame 5k) from anchors z0 [B, 236] under launches launch5 [B, 5] (the official validation_step
    prediction loop: context at noise level 0, then the pyramid over the whole horizon)."""
    tm = model.transition_model
    device = model_device(model)
    z0 = z0.to(device).float()
    batch = z0.shape[0]
    x0 = model.normalize(z0)
    conditions = condition_sequence(launch5, CONTEXT_FRAMES + horizon).to(device).permute(1, 0, 2).contiguous()
    z = torch.zeros(batch, model.config["z_dim"], device=device)
    z, x_context = context_step(tm, z, x0, conditions[0], noise_fn)
    chunk = [noise_fn((batch, CARRIER_DIM), device) for _ in range(horizon)]
    steps = tm.sampling_timesteps
    grid = pyramid(horizon, steps, model.config["uncertainty_scale"])
    pairs = ddim_times(tm)
    for m in range(grid.shape[0]):
        z_chunk = z.detach()
        for t in range(horizon):
            i = min(int(grid[m, t]), steps - 1)
            chunk[t], z_chunk = ddim_step(tm, chunk[t], z_chunk, conditions[CONTEXT_FRAMES + t], i, noise_fn, pairs)
    return model.unnormalize(torch.stack([x_context] + chunk))


# ---------------------------------------------------------------------------------------------------------------
# early-stopping validation (the official time-series CRPS-sum)
# ---------------------------------------------------------------------------------------------------------------

QUANTILES = (np.arange(20) / 20.0)[1:]


def crps_quantile_sum(pred, truth):
    """``algorithms/common/metrics/crps_quantile.py`` (GluonTS nursery ``CRPS``, 19 quantiles) re-stated:
    pred [samples, time, batch, feature], truth [time, batch, feature]."""
    quantiles = torch.as_tensor([float(q) for q in QUANTILES])
    pred, truth = pred.sum(dim=-1), truth.sum(dim=-1)
    pq = torch.stack([pred.kthvalue(int(q * pred.shape[0]), dim=0)[0] for q in QUANTILES], dim=2)
    pq = pq.permute((1, 0, 2))
    y_true = truth.transpose(0, 1).unsqueeze(-1)
    losses = 2 * torch.sum(torch.abs((pq - y_true) * ((y_true <= pq).type(torch.uint8) - quantiles)), axis=-1)
    denom = torch.as_tensor(0.0) + torch.sum(torch.abs(y_true))
    total = torch.as_tensor(0.0) + torch.mul(losses, torch.ones_like(losses)).sum() / len(quantiles)
    return total / denom


def validation_rows(shots, count=EARLY_STOPPING["validation_shots"], seed=VALIDATION_SEED):
    """The fixed early-stopping validation subset: ``count`` fit-shot indices (sorted)."""
    return np.sort(np.random.default_rng(seed).choice(shots, size=count, replace=False))


@torch.no_grad()
def validation_crps(model, carriers, launch5, rows, samples=EARLY_STOPPING["samples"], seed=VALIDATION_SEED,
                    autocast=False):
    """CRPS-sum of ``samples`` sampled paths per validation shot (the official ``validation_step`` with
    ``calc_crps_sum``: rows sample-major, then ``on_validation_epoch_end``), frames after the context."""
    model.eval()
    device = model_device(model)
    xs, conditions, _ = sequence_batch(carriers, launch5, rows, device)
    z0 = xs[:, 0].repeat(samples, 1)
    launch = conditions[:, 1].repeat(samples, 1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        paths = sample_paths(model, z0, launch, GeneratorNoise(seed, device))
    preds = paths.float().view(paths.shape[0], samples, -1, CARRIER_DIM).permute(1, 0, 2, 3).cpu()
    gt = xs.permute(1, 0, 2).cpu()
    return float(crps_quantile_sum(preds[:, CONTEXT_FRAMES:], gt[CONTEXT_FRAMES:]))


# ---------------------------------------------------------------------------------------------------------------
# scoring: expected tie-free cost over S sampled paths
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def candidate_paths(model, z0, launch5, samples=SAMPLES, noise=None, keep=(HORIZON,)):
    """Paths of candidates: anchors z0 [R, 236] (one per candidate row) and launches [R, 5]; returns
    {frame index: [R, S, 236] CPU} for frame indices ``keep`` (45 = frame 225). Common random numbers."""
    model.eval()
    noise = noise or CommonNoise(samples)
    noise.reset()
    rows = z0.shape[0]
    z = z0.float().repeat_interleave(samples, dim=0)
    a = torch.as_tensor(launch5, dtype=torch.float32).repeat_interleave(samples, dim=0)
    paths = sample_paths(model, z, a, noise)
    return {k: paths[k].reshape(rows, samples, CARRIER_DIM).cpu() for k in keep}


def _cost_keys():
    from world_model.planning import scoring_harness as harness
    return (harness.PRIMARY_COST, harness.SECONDARY_COST, "pig_presence", "pig_displacement")


def expected_row(ordinal, samples, z0, costs):
    """The expected EndpointCosts row of one candidate over its S frame-225 samples [S, 236]; any nonfinite sample
    -> excluded typed row."""
    keys = _cost_keys()
    s = len(samples)
    rows = [costs.row(ordinal, x, z0) for x in samples]
    bad = sum(1 for r in rows if r["excluded"])
    if bad:
        return {"ordinal": ordinal, "excluded": True, **dict.fromkeys(keys), "samples": s, "nonfinite_samples": bad,
                "typed_failure": "nonfinite_samples"}
    return {"ordinal": ordinal, "excluded": False, **{k: float(np.mean([r[k] for r in rows])) for k in keys},
            "samples": s, "nonfinite_samples": 0, "typed_failure": None}


def expected_rows(items, samples, z0, costs):
    """Rows of candidates ``items`` from their samples [K, S, 236] and the anchor z0 [236]."""
    return [expected_row(it["ordinal"], samples[i], z0, costs) for i, it in enumerate(items)]


def sample_means(samples):
    """Per-candidate mean carrier over the sample axis (G-action / position input); NaN where a sample is NaN."""
    return samples.mean(dim=-2)
