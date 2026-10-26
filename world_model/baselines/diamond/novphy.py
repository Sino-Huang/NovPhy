"""Issue #137 (#127 coverage arm T5): the vendored official DIAMOND diffusion world model on NovPhy decision frames.

Adapter (Decision P-adapter on #137; the vendored code at ``vendor/`` (eloialonso/diamond ``main`` @ 5bcd159) is
unchanged):

* **Model** = the official ``models.diffusion.Denoiser`` (EDM preconditioning, offset noise, log-normal training sigma)
  around ``InnerModel`` (U-Net 2D, frame stacking of the last 4 frames, adaptive-group-norm conditioning on the noise
  level and the last 4 actions) at ``config/agent/default.yaml`` (= paper Table 2). The action is the 5-d launch
  vector ``eb.launch5`` (zeros = no-op), embedded through the official table as the ``csgo`` branch embeds multi-hot
  actions: ``act @ act_emb[0].weight`` (``LaunchEmbedding``; ``num_actions`` = 5, same parameters and init as the
  official ``nn.Embedding(5, 64)``).
* **Frames.** 64 x 64 RGB (DIAMOND's native resolution) in [-1, 1] (``TorchEnv``'s ``u8 / 255 * 2 - 1``): the #123
  pixel stores (uint8 320 x 240) resized with exact area averaging (cv2 ``INTER_AREA`` semantics, the official
  ``AtariPreprocessing`` resize; full frame, aspect 4:3 -> 1:1), ``resize_frames``.
* **Episodes.** Stride 5 (positions 0, 5, ..., 225; shots ending earlier repeat their last frame), preceded by 3
  copies of the decision frame (the scene is static before the launch): 49 frames; actions are zeros except the
  launch at episode index 3 (the decision frame), so the 4-frame context of the first prediction is the decision
  frame x4 with actions [no-op, no-op, no-op, launch]. Training windows: the official ``BatchSampler`` rule on a
  static dataset (episodes weighted by length = uniform; padding only before the start, zero frames / actions,
  ``mask_padding`` False) and ``make_segment`` padding; ``Denoiser.forward`` is the loss.
* **Training step** = ``Trainer.train_component``: ``loss.backward()``, ``clip_grad_norm_`` (1.0), AdamW
  (``configure_opt``: decay on Linear / Conv weights only, lr 1e-4, wd 1e-2, eps 1e-8), ``zero_grad``, warm-up
  ``LambdaLR`` (100 steps).
* **Rollout** = ``WorldModelEnv.step`` semantics (set the current action in the last slot, sample the next frame
  with the official Euler 3-step sampler, roll the 4-frame / 4-action buffers): 45 steps to frame 225. ``sample_next``
  is ``DiffusionSampler.sample`` with its initial noise supplied, so paths can share noise across candidates (common
  random numbers); G-port checks it against the official call.
* **Readout** (G-readout applies): ``FrameReadout`` maps a 64 x 64 frame to the 236-d E99 carrier: 64 patch tokens
  (8 x 8 x 3 pixels + the patch centre) and the mean-colour token, read by the #135 ``ParticleReadout`` architecture
  with the #130 R1 recipe (19 slot queries cross-attending; dim 256, 8 heads, MLP 512, AdamW 1e-3 / wd 1e-4, batch 128,
  25 epochs, MSE on z-scored carriers); fitted once on true fit frames only and shared by every configuration / seed.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import types
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

VENDOR = Path(__file__).resolve().parent / "vendor"
SRC = VENDOR / "src"
_PACKAGE = "_b137_diamond_vendor"

IMAGE = 64
FRAME_HW = (240, 320)
STRIDE = 5
ENDPOINT = 225
STEPS = ENDPOINT // STRIDE  # 45 autoregressive steps
POSITIONS = tuple(range(0, ENDPOINT + 1, STRIDE))  # 46 stored frames per shot
CONDITIONING = 4  # num_steps_conditioning (L)
EPISODE_FRAMES = tuple([0] * (CONDITIONING - 1) + list(range(len(POSITIONS))))  # 49 indices into POSITIONS
LAUNCH_INDEX = CONDITIONING - 1  # the episode index of the decision frame (the launch acts here)
LAUNCH_DIM = 5
CARRIER_DIM = 236
CARRIER_HEADER = 2
SLOTS = 18
SLOT_WIDTH = 13

# config/agent/default.yaml (denoiser) and config/trainer.yaml (denoiser training, sampler) at 5bcd159
AGENT = {"sigma_data": 0.5, "sigma_offset_noise": 0.3,
         "inner_model": {"img_channels": 3, "num_steps_conditioning": CONDITIONING, "cond_channels": 256,
                         "depths": [2, 2, 2, 2], "channels": [64, 64, 64, 64], "attn_depths": [0, 0, 0, 0]}}
SIGMA_DISTRIBUTION = {"loc": -0.4, "scale": 1.2, "sigma_min": 2e-3, "sigma_max": 20.0}
OPTIMIZER = {"lr": 1e-4, "weight_decay": 1e-2, "eps": 1e-8}
TRAINING = {"batch_size": 32, "lr_warmup_steps": 100, "max_grad_norm": 1.0, "num_autoregressive_steps": 1}
SAMPLER = {"num_steps_denoising": 3, "sigma_min": 2e-3, "sigma_max": 5.0, "rho": 7, "order": 1, "s_churn": 0.0,
           "s_tmin": 0.0, "s_tmax": float("inf"), "s_noise": 1.0}
NATIVE_UPDATES = 10_000 + 999 * 400  # steps_first_epoch + (1000 epochs - 1) x steps_per_epoch (paper Table 3)


# ---------------------------------------------------------------------------------------------------------------
# vendored modules (loaded by file location; no omegaconf / wandb needed)
# ---------------------------------------------------------------------------------------------------------------

_VENDOR = None


def vendor() -> types.SimpleNamespace:
    """The official Denoiser / sampler classes, imported unchanged from vendor/src under a private package name.
    ``denoiser.py``'s top-level ``from data import Batch`` / ``from utils import LossAndLogs`` are served, during this
    load only, by vendor ``src/data/batch.py`` and the upstream ``LossAndLogs`` alias (upstream ``utils.py`` imports
    omegaconf / wandb)."""
    global _VENDOR
    if _VENDOR is not None:
        return _VENDOR

    def package(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module

    package(_PACKAGE, SRC)
    package(f"{_PACKAGE}.data", SRC / "data")
    package(f"{_PACKAGE}.models", SRC / "models")
    batch = importlib.import_module(f"{_PACKAGE}.data.batch")
    data_stub = types.ModuleType("data")
    data_stub.Batch = batch.Batch
    utils_stub = types.ModuleType("utils")
    utils_stub.LossAndLogs = Tuple[Tensor, Dict[str, Any]]  # upstream src/utils.py
    saved = {name: sys.modules.get(name) for name in ("data", "utils")}
    sys.modules.update(data=data_stub, utils=utils_stub)
    try:
        diffusion = importlib.import_module(f"{_PACKAGE}.models.diffusion")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    sampler = importlib.import_module(f"{_PACKAGE}.models.diffusion.diffusion_sampler")
    _VENDOR = types.SimpleNamespace(Batch=batch.Batch, Denoiser=diffusion.Denoiser,
                                    DenoiserConfig=diffusion.DenoiserConfig, InnerModelConfig=diffusion.InnerModelConfig,
                                    SigmaDistributionConfig=diffusion.SigmaDistributionConfig,
                                    DiffusionSampler=diffusion.DiffusionSampler,
                                    DiffusionSamplerConfig=diffusion.DiffusionSamplerConfig,
                                    build_sigmas=sampler.build_sigmas)
    return _VENDOR


# ---------------------------------------------------------------------------------------------------------------
# model and optimizer
# ---------------------------------------------------------------------------------------------------------------

class LaunchEmbedding(nn.Embedding):
    """The official action table applied to a real-valued action vector (the ``csgo`` branch's multi-hot form
    ``act.float() @ act_emb[0].weight``): act [b, t, 5] -> [b, t, e]; zeros (no-op) embed to zero."""

    def forward(self, act: Tensor) -> Tensor:
        return act.float() @ self.weight


def build_denoiser(agent: dict = AGENT, sigma_distribution: dict = SIGMA_DISTRIBUTION):
    """The official ``Denoiser`` (``num_actions`` = 5) with ``LaunchEmbedding`` as its action table (the module's
    class is swapped: same parameter, same init, no extra RNG draw) and the official training sigma sampler."""
    v = vendor()
    inner = v.InnerModelConfig(**agent["inner_model"], num_actions=LAUNCH_DIM)
    denoiser = v.Denoiser(v.DenoiserConfig(inner_model=inner, sigma_data=agent["sigma_data"],
                                           sigma_offset_noise=agent["sigma_offset_noise"]))
    denoiser.inner_model.act_emb[0].__class__ = LaunchEmbedding
    denoiser.setup_training(v.SigmaDistributionConfig(**sigma_distribution))
    return denoiser


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_opt(model: nn.Module, lr: float, weight_decay: float, eps: float) -> torch.optim.AdamW:
    """Upstream ``utils.configure_opt`` (credits to karpathy/minGPT): weight decay on Linear / Conv / LSTM weights,
    none on biases, LayerNorm / Embedding / GroupNorm weights; parameters in sorted-name order per group."""
    decay, no_decay = set(), set()
    whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTMCell, nn.LSTM)
    blacklist = (nn.LayerNorm, nn.Embedding, nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if "bias" in pn:
                no_decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, whitelist):
                decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, blacklist):
                no_decay.add(fpn)
    params = dict(model.named_parameters())
    if decay & no_decay or params.keys() - (decay | no_decay):
        raise ValueError("configure_opt: a parameter is in both or neither group")
    groups = [{"params": [params[n] for n in sorted(decay)], "weight_decay": weight_decay},
              {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, eps=eps)


def get_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Upstream ``utils.get_lr_sched``: linear warm-up from 0, then constant."""
    def lr_lambda(current_step: int):
        return 1 if current_step >= num_warmup_steps else current_step / max(1, num_warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=-1)


def build_optimizer(model: nn.Module, optimizer: dict = OPTIMIZER, training: dict = TRAINING):
    opt = configure_opt(model, **optimizer)
    return opt, get_lr_sched(opt, training["lr_warmup_steps"])


def seq_length(training: dict = TRAINING) -> int:
    """Window length of the denoiser loader: L + 1 + num_autoregressive_steps (``Trainer.__init__``)."""
    return CONDITIONING + 1 + training["num_autoregressive_steps"]


def train_step(model, opt, lr_sched, batch, max_grad_norm: float = TRAINING["max_grad_norm"]) -> dict:
    """One ``Trainer.train_component`` iteration (grad_acc_steps 1): loss, backward, clip, step, zero_grad, LR step."""
    loss, _ = model(batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()
    lr = lr_sched.get_last_lr()[0]
    lr_sched.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm), "lr": float(lr)}


# ---------------------------------------------------------------------------------------------------------------
# frames, episodes and training windows
# ---------------------------------------------------------------------------------------------------------------

def _area_weights(n_in: int, n_out: int) -> Tensor:
    """[n_out, n_in] area-averaging weights (cv2 INTER_AREA for a downscale: the fraction of each source pixel
    covered by the destination pixel, divided by the scale)."""
    scale = n_in / n_out
    w = torch.zeros(n_out, n_in, dtype=torch.float64)
    for i in range(n_out):
        a, b = i * scale, (i + 1) * scale
        for j in range(int(math.floor(a)), min(n_in, int(math.ceil(b)))):
            w[i, j] = (min(b, j + 1) - max(a, j)) / scale
    return w


_WEIGHTS = {}


def resize_frames(frames: Tensor) -> Tensor:
    """uint8 [N, 3, 240, 320] (or [3, 240, 320]) -> uint8 [N, 3, 64, 64]: exact area averaging (float64), rounded
    half-to-even (cv2 ``saturate_cast``)."""
    single = frames.dim() == 3
    frames = frames[None] if single else frames
    h, w = frames.shape[-2:]
    if (h, w) not in _WEIGHTS:
        _WEIGHTS[(h, w)] = (_area_weights(h, IMAGE), _area_weights(w, IMAGE))
    wy, wx = _WEIGHTS[(h, w)]
    out = torch.round(wy @ frames.double() @ wx.T).clamp_(0, 255).to(torch.uint8)
    return out[0] if single else out


def to_unit(frames_u8) -> Tensor:
    """uint8 frames -> float in [-1, 1] (``TorchEnv._to_tensor``)."""
    return torch.as_tensor(frames_u8).float().div(255).mul(2).sub(1)


def frame_positions(end: int) -> list[int]:
    """The stored positions of a shot: 0, 5, ..., 225, clamped to its last frame (the episode repeats it)."""
    return [min(p, int(end)) for p in POSITIONS]


def sample_segments(rng: np.random.Generator, n_shots: int, batch: int, length: int):
    """Official ``BatchSampler.sample`` on a static dataset of equal-length episodes (sample weights ignored ->
    episodes weighted by length = uniform; ``can_sample_beyond_end`` False): (episode ids, window starts)."""
    episode_len = len(EPISODE_FRAMES)
    episode_ids = rng.choice(n_shots, size=batch, replace=True)
    timesteps = rng.integers(0, episode_len, size=batch)
    stops = np.minimum(episode_len, timesteps + 1 + rng.integers(0, length, size=batch))
    return episode_ids, stops - length


def make_batch(frames, launch5: Tensor, shots, starts, length: int, device="cpu"):
    """The official ``Batch`` of windows [start, start + length) of the given episodes (``make_segment`` padding:
    zero frames / actions before the episode start, ``mask_padding`` False there)."""
    b = len(shots)
    obs = torch.zeros(b, length, 3, IMAGE, IMAGE)
    act = torch.zeros(b, length, LAUNCH_DIM)
    mask = torch.zeros(b, length, dtype=torch.bool)
    for i, (s, start) in enumerate(zip(shots, starts)):
        lo = max(0, -int(start))
        idx = [EPISODE_FRAMES[int(start) + t] for t in range(lo, length)]
        obs[i, lo:] = to_unit(np.asarray(frames[int(s)][idx]))
        mask[i, lo:] = True
        if lo <= LAUNCH_INDEX - int(start) < length:
            act[i, LAUNCH_INDEX - int(start)] = launch5[int(s)]
    zeros = torch.zeros(b, length)
    return vendor().Batch(obs.to(device), act.to(device), zeros.to(device), zeros.long().to(device),
                          zeros.long().to(device), mask.to(device), [{} for _ in range(b)], [])


# ---------------------------------------------------------------------------------------------------------------
# sampler and rollout
# ---------------------------------------------------------------------------------------------------------------

def make_sampler(denoiser, sampler: dict = SAMPLER):
    v = vendor()
    return v.DiffusionSampler(denoiser, v.DiffusionSamplerConfig(**sampler))


@torch.no_grad()
def sample_next(sampler, prev_obs: Tensor, prev_act: Tensor, x: Tensor) -> Tensor:
    """``DiffusionSampler.sample`` (official body) with its initial noise ``x`` [b, c, h, w] supplied."""
    b, t, c, h, w = prev_obs.size()
    prev_obs = prev_obs.reshape(b, t * c, h, w)
    s_in = torch.ones(b, device=prev_obs.device)
    cfg = sampler.cfg
    gamma_ = min(cfg.s_churn / (len(sampler.sigmas) - 1), 2 ** 0.5 - 1)
    for sigma, next_sigma in zip(sampler.sigmas[:-1], sampler.sigmas[1:]):
        gamma = gamma_ if cfg.s_tmin <= sigma <= cfg.s_tmax else 0
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * cfg.s_noise
            x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        denoised = sampler.denoiser.denoise(x, sigma, prev_obs, prev_act)
        d = (x - denoised) / sigma_hat
        dt = next_sigma - sigma_hat
        if cfg.order == 1 or next_sigma == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = sampler.denoiser.denoise(x_2, next_sigma * s_in, prev_obs, prev_act)
            d_2 = (x_2 - denoised_2) / next_sigma
            x = x + (d + d_2) / 2 * dt
    return x


@torch.no_grad()
def rollout(denoiser, frames0: Tensor, launch5: Tensor, noise: Tensor, keep=(STEPS,), sampler=None) -> dict:
    """``WorldModelEnv.step`` x max(keep) from the decision frames ``frames0`` [B, 3, 64, 64] (in [-1, 1]; context =
    the frame x4, actions [0, 0, 0, launch]) under launches [B, 5], no-ops after; ``noise`` [steps, B, 3, 64, 64] is
    the sampler's initial noise of each step. -> {step: predicted frames [B, 3, 64, 64]} (step k = frame 5 k)."""
    sampler = sampler or make_sampler(denoiser)
    b = frames0.shape[0]
    obs = frames0[:, None].repeat(1, CONDITIONING, 1, 1, 1)
    act = torch.zeros(b, CONDITIONING, LAUNCH_DIM, device=frames0.device)
    out = {}
    for k in range(max(keep)):
        act[:, -1] = launch5 if k == 0 else 0
        nxt = sample_next(sampler, obs, act, noise[k])
        obs = obs.roll(-1, dims=1)
        act = act.roll(-1, dims=1)
        obs[:, -1] = nxt
        if k + 1 in keep:
            out[k + 1] = nxt
    return out


# ---------------------------------------------------------------------------------------------------------------
# readout: 64 x 64 frame -> E99 carrier (fit frames only; the #135 ParticleReadout architecture, #130 R1 recipe)
# ---------------------------------------------------------------------------------------------------------------

PATCH = 8
READOUT = {"dim": 256, "heads": 8, "hidden": 512, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128, "epochs": 25}
TOKEN_DIM = 3 * PATCH * PATCH + 2


def carrier_stats(rows):
    rows = torch.as_tensor(rows).float()
    return rows.mean(0), rows.std(0).clamp_min(1e-4)


def _patch_centres() -> Tensor:
    n = IMAGE // PATCH
    c = (torch.arange(n, dtype=torch.float32) + 0.5) / n * 2 - 1
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], 1)  # [64, 2] (x, y) in (-1, 1)


def frame_tokens(frames: Tensor) -> tuple[Tensor, Tensor]:
    """Frames [N, 3, 64, 64] in [-1, 1] -> (patch tokens [N, 64, 194] (8 x 8 x 3 pixels, patch centre x, y),
    mean-colour token [N, 3])."""
    n = frames.shape[0]
    g = IMAGE // PATCH
    p = frames.reshape(n, 3, g, PATCH, g, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, 3 * PATCH * PATCH)
    xy = _patch_centres().to(frames.device)[None].expand(n, -1, -1)
    return torch.cat([p, xy], -1), frames.mean((2, 3))


class FrameReadout(nn.Module):
    """#135 ``ParticleReadout`` (token MLP, background token, 19 queries, one cross-attention, MLP, header / slot
    heads) over the patch tokens of a 64 x 64 frame; outputs the 236-d E99 carrier (z-scored internally)."""

    def __init__(self, mean=None, std=None, dim: int = READOUT["dim"]):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.embed_bg = nn.Linear(3, dim)
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(1 + SLOTS, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, READOUT["heads"], batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, READOUT["hidden"]), nn.GELU(),
                                 nn.Linear(READOUT["hidden"], dim))
        self.head_header = nn.Linear(dim, CARRIER_HEADER)
        self.head_slot = nn.Linear(dim, SLOT_WIDTH)
        self.register_buffer("mean", torch.zeros(CARRIER_DIM) if mean is None else mean.clone().float())
        self.register_buffer("std", torch.ones(CARRIER_DIM) if std is None else std.clone().float())

    def forward_z(self, frames: Tensor) -> Tensor:
        tokens, bg = frame_tokens(frames)
        k = self.norm(torch.cat([self.embed(tokens), self.embed_bg(bg)[:, None]], 1))
        q = self.queries.unsqueeze(0).expand(frames.shape[0], -1, -1)
        h = q + self.attn(q, k, k, need_weights=False)[0]
        h = h + self.mlp(h)
        return torch.cat([self.head_header(h[:, 0]), self.head_slot(h[:, 1:]).flatten(1)], 1)

    def forward(self, frames: Tensor) -> Tensor:
        return self.forward_z(frames) * self.std + self.mean


def fit_readout(frames_u8, targets, seed: int, device="cpu", epochs=None, log=None, state=None, checkpoint=None):
    """Fit ``FrameReadout`` on uint8 frames [N, 3, 64, 64] -> E99 targets [N, 236] (MSE on z-scored targets,
    AdamW; epoch e's order = randperm under a generator seeded once). ``checkpoint(state)`` is called after every
    epoch with a resumable state (model, optimizer, generator, epoch); ``state`` resumes from it."""
    epochs = READOUT["epochs"] if epochs is None else epochs
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    mean, std = carrier_stats(targets)
    probe = FrameReadout(mean, std).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=READOUT["lr"], weight_decay=READOUT["weight_decay"])
    yz = (torch.as_tensor(targets).float() - mean) / std
    frames_u8 = torch.as_tensor(frames_u8)
    n, b = len(frames_u8), READOUT["batch"]
    steps, last, first_epoch = 0, None, 0
    if state is not None:
        probe.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        steps, last, first_epoch = state["steps"], state["last"], state["epoch"]
    for epoch in range(first_epoch, epochs):
        order = torch.randperm(n, generator=generator)
        for i in range(0, n - b + 1, b):
            idx = order[i:i + b]
            loss = F.mse_loss(probe.forward_z(to_unit(frames_u8[idx]).to(device)), yz[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            last = float(loss.detach())
        if log is not None:
            log(f"readout epoch {epoch + 1}/{epochs}: loss {last:.5f}")
        if checkpoint is not None and epoch + 1 < epochs:
            checkpoint({"model": probe.state_dict(), "optimizer": optimizer.state_dict(),
                        "generator": generator.get_state(), "steps": steps, "last": last, "epoch": epoch + 1})
    probe.eval().requires_grad_(False)
    return probe, {"steps": steps, "final_loss": last, "samples": n, "epochs": epochs, "seed": seed,
                   "patch": PATCH, **{k: READOUT[k] for k in READOUT}}


@torch.no_grad()
def apply_readout(probe: FrameReadout, frames: Tensor, chunk: int = 2048) -> Tensor:
    """Readout carriers [N, 236] (CPU float32) of frames [N, 3, 64, 64] in [-1, 1]."""
    device = probe.mean.device
    out = [probe(frames[i:i + chunk].to(device)).float().cpu() for i in range(0, len(frames), chunk)]
    return torch.cat(out) if out else torch.empty(0, CARRIER_DIM)
