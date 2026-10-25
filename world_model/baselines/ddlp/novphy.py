"""Issue #135 (#127 Tier-C T2): the vendored official DDLP (ObjectDynamicsDLP + PINT) on NovPhy decision frames.

Adapter (Decision P-other on #135; the vendored code is unchanged):

* **Frames.** The #123 pixel stores (uint8 [3, 240, 320]) resized to the PHYRE configuration's 128 x 128 with
  ``cv2.resize(..., interpolation=cv2.INTER_AREA)`` (the interpolation the official ``phyre_preparation.py`` names;
  full frame, aspect 4:3 -> 1:1; no colour inversion: that was PHYRE-specific, white background), float in [0, 1].
* **Episodes.** Frames at stride 5 (0.1 s at 50 Hz = the 10 fps of the paper's PHYRE data, inside DDLP's
  small-displacement tracking assumption): positions 0, 5, ..., 225 (shots ending earlier repeat their last frame, as
  the official PHYRE loader pads), preceded by ``STATIC - 1`` copies of the decision frame: the scene is static
  before the launch, so the burn-in context of ``tau = 4`` frames (``num_static_frames``) is the decision frame
  four times and the launch acts at t0 = context index 3. Episode = 4 + 45 = 49 frames; training windows are the
  official ``timestep_horizon + 1`` = 16 consecutive frames at every offset (the official ``PhyreDataset`` indexing).
* **Launch.** One extra *launch particle* per frame, appended after the K foreground particles (the background
  token stays last, as official): its attributes [xy 2, scale 2, obj_on 1, depth 1, features d] are a learned linear
  map of ``eb.launch5``; the official ``ParticleFeatureProjection`` projects it like any particle; PINT's particle
  relative-position bias covers K + 2 tokens; its predicted next state is discarded (it is re-set to the launch at
  every step of a rollout and excluded from every loss: the losses see exactly the official K-particle outputs).
* **Rollout.** Deterministic (posterior / PINT means, the official ``sample`` default): encode the 4-frame static
  context with the official tracking posterior + background encoder exactly as ``ObjectDynamicsDLP.sample`` does, then
  45 PINT steps (official ``DynamicsDLP.sample``, one step at a time so the launch particle can be re-set; sliding
  context of ``timestep_horizon`` frames as official) -> particles at frames 5, ..., 225.
* **Readout** (G-readout applies): ``ParticleReadout`` maps a particle set (K tokens [xy, scale, obj_on, depth,
  features] + the background-feature token) to the 236-d E99 carrier (19 slot queries cross-attending over the
  particle tokens, no particle-index embedding: permutation-invariant); fitted on fit frames only (particles of the
  true frame, encoded as the static context) with the #130 R1 recipe; frozen before any evaluation read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendor.models import ObjectDynamicsDLP
from .vendor.modules.dynamics_modules import DynamicsDLP
from .vendor.utils.loss_functions import calc_reconstruction_loss

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT / "data/external/ddlp"
RELEASED_PHYRE_HPARAMS = EXTERNAL / "checkpoints/ddlp-phyre/hparams.json"

IMAGE = 128
FRAME_HW = (240, 320)
STRIDE = 5
ENDPOINT = 225
STEPS = ENDPOINT // STRIDE  # 45 PINT steps
POSITIONS = tuple(range(0, ENDPOINT + 1, STRIDE))  # 46 stored frames per shot
STATIC = 4  # tau = num_static_frames (burn-in), the paper's value for every dataset
EPISODE_FRAMES = tuple([0] * (STATIC - 1) + list(range(len(POSITIONS))))  # 49 indices into POSITIONS
LAUNCH_INDEX = STATIC - 1  # the context frame at which the launch acts (t0)
LAUNCH_DIM = 5
CARRIER_DIM = 236
CARRIER_HEADER = 2
SLOTS = 18
SLOT_WIDTH = 13

# The released PHYRE-128 checkpoint's hparams.json (= the paper's Table 4 PHYRE column); the NovPhy native config.
# Training-only keys (ds, root, device, num_epochs, eval / animation settings) are not part of the model.
NATIVE = {
    "batch_size": 3, "lr": 0.0002, "kp_activation": "tanh", "pad_mode": "replicate", "n_kp": 1,
    "recon_loss_type": "mse", "sigma": 1.0, "beta_kl": 0.15, "beta_rec": 1.0, "patch_size": 16, "n_kp_enc": 25,
    "learned_feature_dim": 4, "n_kp_prior": 64, "weight_decay": 0.0, "kp_range": [-1, 1], "warmup_epoch": 1,
    "dropout": 0.0, "anchor_s": 0.125, "kl_balance": 0.001, "image_size": 128, "ch": 3,
    "enc_channels": [32, 32, 64, 64, 128], "prior_channels": [32, 32, 64], "timestep_horizon": 15,
    "predict_delta": True, "beta_dyn": 0.15, "scale_std": 1.0, "offset_std": 1.0, "obj_on_alpha": 0.1,
    "obj_on_beta": 0.1, "beta_dyn_rec": 1.0, "num_static_frames": 4, "pint_layers": 6, "pint_heads": 8,
    "pint_dim": 512, "use_resblock": False, "scheduler_gamma": 0.95, "adam_betas": [0.9, 0.999], "adam_eps": 0.0001,
    "train_enc_prior": True, "start_dyn_epoch": 0, "use_correlation_heatmaps": True, "enable_enc_attn": False,
    "filtering_heuristic": "variance",
}
NATIVE_GPUS = 4  # accel_conf.yml num_processes: 4 processes x batch_size 3, DDP-averaged = one batch of 12
# The repository's configs/phyre.json (train_ddlp.py -d phyre) differs from the released checkpoint only in these keys.
REPO_PHYRE = {**NATIVE, "enc_channels": [32, 64, 128, 256], "prior_channels": [16, 32, 64], "n_kp_prior": 30,
              "scale_std": 0.3, "offset_std": 0.2}
BATCH = NATIVE["batch_size"] * NATIVE_GPUS  # 12


def released_hparams(path=RELEASED_PHYRE_HPARAMS) -> dict:
    return json.loads(Path(path).read_text())


def model_kwargs(config: dict) -> dict:
    """ObjectDynamicsDLP constructor arguments of a config (exactly the official train_ddlp.py call)."""
    c = config
    return dict(cdim=c["ch"], enc_channels=c["enc_channels"], prior_channels=c["prior_channels"],
                image_size=c["image_size"], n_kp=c["n_kp"], learned_feature_dim=c["learned_feature_dim"],
                pad_mode=c["pad_mode"], sigma=c["sigma"], dropout=c["dropout"], patch_size=c["patch_size"],
                n_kp_enc=c["n_kp_enc"], n_kp_prior=c["n_kp_prior"], kp_range=c["kp_range"],
                kp_activation=c["kp_activation"], anchor_s=c["anchor_s"], use_resblock=c["use_resblock"],
                timestep_horizon=c["timestep_horizon"], predict_delta=c["predict_delta"], scale_std=c["scale_std"],
                offset_std=c["offset_std"], obj_on_alpha=c["obj_on_alpha"], obj_on_beta=c["obj_on_beta"],
                pint_layers=c["pint_layers"], pint_heads=c["pint_heads"], pint_dim=c["pint_dim"],
                use_correlation_heatmaps=c["use_correlation_heatmaps"], enable_enc_attn=c["enable_enc_attn"],
                filtering_heuristic=c["filtering_heuristic"])


# ---------------------------------------------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------------------------------------------

def resize_frames(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [N, 3, 240, 320] (or [3, 240, 320]) -> uint8 [N, 3, 128, 128] (cv2 INTER_AREA, full frame)."""
    import cv2
    single = frames.dim() == 3
    frames = frames[None] if single else frames
    hwc = frames.permute(0, 2, 3, 1).contiguous().numpy()
    out = np.stack([cv2.resize(f, (IMAGE, IMAGE), interpolation=cv2.INTER_AREA) for f in hwc])
    out = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
    return out[0] if single else out


def to_float(frames_u8: torch.Tensor) -> torch.Tensor:
    return frames_u8.float() / 255.0


def frame_positions(end: int) -> list[int]:
    """The stored positions of a shot: 0, 5, ..., 225, clamped to its last frame (the episode pads with it)."""
    return [min(p, int(end)) for p in POSITIONS]


def windows_per_episode(config: dict) -> int:
    return len(EPISODE_FRAMES) - (config["timestep_horizon"] + 1) + 1


def window(episode_frames: torch.Tensor, offset: int, config: dict) -> torch.Tensor:
    """[46, 3, H, W] stored frames of one shot -> the official [T + 1, 3, H, W] window at ``offset`` of its episode."""
    idx = list(EPISODE_FRAMES[offset:offset + config["timestep_horizon"] + 1])
    return episode_frames[idx]


def context(frames0: torch.Tensor) -> torch.Tensor:
    """Decision frames [B, 3, H, W] -> the static burn-in context [B, STATIC, 3, H, W]."""
    return frames0[:, None].expand(-1, STATIC, -1, -1, -1)


# ---------------------------------------------------------------------------------------------------------------
# the launch particle
# ---------------------------------------------------------------------------------------------------------------

class LaunchDynamics(nn.Module):
    """The official ``DynamicsDLP`` (PINT) with one launch particle per frame (index K, before the background token).

    ``launch5`` is set by the caller (``DDLPNovPhy.launch_context``) for the duration of a forward / rollout."""

    def __init__(self, dyn: DynamicsDLP, n_kp: int, features_dim: int):
        super().__init__()
        self.dyn = dyn
        self.n_kp = n_kp
        self.features_dim = features_dim
        self.launch_map = nn.Linear(LAUNCH_DIM, 2 + 2 + 1 + 1 + features_dim)
        self.launch5 = None

    def launch_particle(self):
        """[B, 1, 1, ...] attributes of the launch particle: xy, scale, obj_on, depth, features."""
        a = self.launch_map(self.launch5)[:, None, None]
        xy, scale, obj_on, depth, feat = torch.split(a, [2, 2, 1, 1, self.features_dim], dim=-1)
        return xy, scale, obj_on[..., 0], depth, feat

    def extend(self, z, z_scale, z_obj_on, z_depth, z_features):
        """Append the launch particle to [B, T, K, .] particle tensors (z_obj_on [B, T, K])."""
        t = z.shape[1]
        xy, scale, obj_on, depth, feat = self.launch_particle()
        return (torch.cat([z, xy.expand(-1, t, -1, -1)], 2), torch.cat([z_scale, scale.expand(-1, t, -1, -1)], 2),
                torch.cat([z_obj_on, obj_on.expand(-1, t, -1)], 2), torch.cat([z_depth, depth.expand(-1, t, -1, -1)], 2),
                torch.cat([z_features, feat.expand(-1, t, -1, -1)], 2))

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features):
        out = self.dyn(*self.extend(z, z_scale, z_obj_on, z_depth, z_features), z_bg_features)
        k = self.n_kp
        return {name: (v if name.endswith("bg_features") else v[:, :, :k]) for name, v in out.items()}

    def sample(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, steps=10, deterministic=False):
        z, z_scale, z_obj_on, z_depth, z_features = self.extend(z, z_scale, z_obj_on, z_depth, z_features)
        xy, scale, obj_on, depth, feat = self.launch_particle()
        k = self.n_kp
        for _ in range(steps):
            z, z_scale, z_obj_on, z_depth, z_features, z_bg_features = self.dyn.sample(
                z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, steps=1, deterministic=deterministic)
            z = torch.cat([z[:, :-1], torch.cat([z[:, -1:, :k], xy], 2)], 1)
            z_scale = torch.cat([z_scale[:, :-1], torch.cat([z_scale[:, -1:, :k], scale], 2)], 1)
            z_obj_on = torch.cat([z_obj_on[:, :-1], torch.cat([z_obj_on[:, -1:, :k], obj_on], 2)], 1)
            z_depth = torch.cat([z_depth[:, :-1], torch.cat([z_depth[:, -1:, :k], depth], 2)], 1)
            z_features = torch.cat([z_features[:, :-1], torch.cat([z_features[:, -1:, :k], feat], 2)], 1)
        return z[:, :, :k], z_scale[:, :, :k], z_obj_on[:, :, :k], z_depth[:, :, :k], z_features[:, :, :k], z_bg_features


class DDLPNovPhy(ObjectDynamicsDLP):
    """The official ``ObjectDynamicsDLP``; with ``launch=True`` its PINT carries the launch particle."""

    def __init__(self, config: dict, launch: bool = True):
        super().__init__(**model_kwargs(config))
        self.config = dict(config)
        self.launch = launch
        if launch:
            dyn = DynamicsDLP(self.learned_feature_dim, self.bg_learned_feature_dim, hidden_dim=self.pint_dim,
                              projection_dim=self.pint_dim, n_head=self.pint_heads, n_layer=self.pint_layers,
                              block_size=self.timestep_horizon, kp_activation=self.kp_activation,
                              predict_delta=self.predict_delta, max_delta=1.0, positional_bias=True,
                              max_particles=self.n_kp_enc + 2)
            self.dyn_module = LaunchDynamics(dyn, self.n_kp_enc, self.learned_feature_dim)

    def set_launch(self, launch5):
        if self.launch:
            self.dyn_module.launch5 = launch5

    def forward_launch(self, x, launch5, **kwargs):
        self.set_launch(launch5)
        try:
            return self.forward(x, **kwargs)
        finally:
            self.set_launch(None)


def build_model(config: dict, launch: bool = True, seed: int | None = None, device="cpu") -> DDLPNovPhy:
    if seed is not None:
        torch.manual_seed(seed)
    return DDLPNovPhy(config, launch=launch).to(device)


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------------------------
# the official training step (train_ddlp.py) and its schedule
# ---------------------------------------------------------------------------------------------------------------

def build_optimizer(model: ObjectDynamicsDLP, config: dict):
    """train_ddlp.py: Adam over model.get_parameters(), StepLR(step_size=1 epoch, gamma) stepped per epoch."""
    optimizer = torch.optim.Adam(model.get_parameters(), lr=config["lr"], betas=tuple(config["adam_betas"]),
                                 eps=config["adam_eps"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["scheduler_gamma"])
    return optimizer, scheduler


def dynamics_discount(iteration: int, iter_per_epoch: int, config: dict) -> torch.Tensor:
    """train_ddlp.py's per-step dynamics-loss ramp: discount_i = clamp(iteration / ((i + 1) * iter_per_step), 0, 1),
    iter_per_step = max(warmup_epoch, 1) * max(10 000, iter_per_epoch) // timestep_horizon."""
    horizon = config["timestep_horizon"]
    warmup_iters = max(config["warmup_epoch"], 1) * max(10_000, iter_per_epoch)
    per_step = warmup_iters // horizon
    max_iterations = [per_step * (i + 1) for i in range(horizon)]
    return (iteration / torch.tensor(max_iterations)).clamp(0.0, 1.0)


def phase(epoch: int, config: dict) -> dict:
    """train_ddlp.py: warm-up for epoch < warmup_epoch; noisy for epoch < warmup_epoch + 1."""
    return {"warmup": epoch < config["warmup_epoch"], "noisy": epoch < config["warmup_epoch"] + 1}


def recon_loss_func(config: dict, device="cpu"):
    if config["recon_loss_type"] == "vgg":
        from .vendor.utils.loss_functions import VGGDistance
        return VGGDistance(device=device)
    return calc_reconstruction_loss


def losses(model: DDLPNovPhy, x: torch.Tensor, launch5, epoch: int, iteration: int, iter_per_epoch: int,
           config: dict, recon=None) -> tuple[dict, dict]:
    """(model_output, all_losses) of one official training iteration (``start_dyn_epoch`` 0 -> calc_elbo)."""
    flags = phase(epoch, config)
    recon = recon_loss_func(config, x.device) if recon is None else recon
    kwargs = dict(x_prior=x, warmup=flags["warmup"], noisy=flags["noisy"], bg_masks_from_fg=False, forward_dyn=True,
                  train_enc_prior=config["train_enc_prior"], num_static_frames=config["num_static_frames"])
    out = model.forward_launch(x, launch5, **kwargs) if isinstance(model, DDLPNovPhy) else model(x, **kwargs)
    discount = dynamics_discount(iteration, iter_per_epoch, config).to(x.device)
    all_losses = model.calc_elbo(x, out, warmup=flags["warmup"], beta_kl=config["beta_kl"], beta_dyn=config["beta_dyn"],
                                 beta_rec=config["beta_rec"], kl_balance=config["kl_balance"],
                                 dynamic_discount=discount, recon_loss_type=config["recon_loss_type"],
                                 recon_loss_func=recon, beta_dyn_rec=config["beta_dyn_rec"],
                                 num_static=config["num_static_frames"], noisy=flags["noisy"])
    return out, all_losses


def train_step(model, optimizer, x, launch5, epoch, iteration, iter_per_epoch, config, recon=None) -> dict:
    """One official optimizer step (zero_grad, backward, step); returns the detached loss terms."""
    _, all_losses = losses(model, x, launch5, epoch, iteration, iter_per_epoch, config, recon)
    optimizer.zero_grad()
    all_losses["loss"].backward()
    optimizer.step()
    return {k: float(v.detach()) for k, v in all_losses.items() if torch.is_tensor(v)}


def losses_static(model: ObjectDynamicsDLP, x: torch.Tensor, epoch: int, config: dict, recon=None) -> tuple[dict, dict]:
    """(model_output, all_losses) of one official static iteration (train_ddlp.py with epoch < start_dyn_epoch:
    ``forward_dyn=False``, ``calc_elbo_static`` with a zero dynamics discount): the DLPv2 particle model alone."""
    flags = phase(epoch, config)
    recon = recon_loss_func(config, x.device) if recon is None else recon
    out = model(x, x_prior=x, warmup=flags["warmup"], noisy=flags["noisy"], bg_masks_from_fg=False, forward_dyn=False,
                train_enc_prior=config["train_enc_prior"], num_static_frames=config["num_static_frames"])
    discount = 0.0 * torch.ones(size=(config["timestep_horizon"],), device=x.device)
    all_losses = model.calc_elbo_static(x, out, warmup=flags["warmup"], beta_kl=config["beta_kl"],
                                        beta_dyn=config["beta_dyn"], beta_rec=config["beta_rec"],
                                        kl_balance=config["kl_balance"], dynamic_discount=discount,
                                        recon_loss_type=config["recon_loss_type"], recon_loss_func=recon,
                                        noisy=flags["noisy"])
    return out, all_losses


def train_step_static(model, optimizer, x, epoch, config, recon=None) -> dict:
    _, all_losses = losses_static(model, x, epoch, config, recon)
    optimizer.zero_grad()
    all_losses["loss"].backward()
    optimizer.step()
    return {k: float(v.detach()) for k, v in all_losses.items() if torch.is_tensor(v)}


# ---------------------------------------------------------------------------------------------------------------
# encoding and rollouts (deterministic, as the official sample default)
# ---------------------------------------------------------------------------------------------------------------

PARTICLE_KEYS = ("z", "z_scale", "z_obj_on", "z_depth", "z_features", "z_bg")


@torch.no_grad()
def encode_context(model: ObjectDynamicsDLP, x: torch.Tensor) -> dict:
    """Particles [B, T, ...] of frames x [B, T, 3, H, W]: the encoding half of ``ObjectDynamicsDLP.sample``
    (tracking posterior, deterministic; background encoder with the particle-glimpse mask)."""
    b, t = x.shape[:2]
    fg = model.fg_sequential_opt(x, deterministic=True, x_prior=x, reshape=True)
    x_in = x.reshape(-1, *x.shape[2:])
    mask = model.get_bg_mask_from_particle_glimpses(fg["z"], fg["obj_on"], mask_size=x_in.shape[-1])
    z_bg = model.bg_module(x_in, mask, True)["z_bg"]
    return {"z": fg["z"].view(b, t, *fg["z"].shape[1:]), "z_scale": fg["z_scale"].view(b, t, *fg["z_scale"].shape[1:]),
            "z_obj_on": fg["obj_on"].view(b, t, *fg["obj_on"].shape[1:]),
            "z_depth": fg["z_depth"].view(b, t, *fg["z_depth"].shape[1:]),
            "z_features": fg["z_features"].view(b, t, *fg["z_features"].shape[1:]),
            "z_bg": z_bg.view(b, t, *z_bg.shape[1:])}


@torch.no_grad()
def rollout_particles(model: DDLPNovPhy, enc: dict, launch5: torch.Tensor | None, steps: int = STEPS) -> dict:
    """PINT rollout of encoded context particles {key: [B, T, ...]} under launches [B, 5] (deterministic, the
    official ``DynamicsDLP.sample``): {key: [B, T + steps, ...]}."""
    model.set_launch(launch5)
    try:
        out = model.dyn_module.sample(enc["z"], enc["z_scale"], enc["z_obj_on"], enc["z_depth"], enc["z_features"],
                                      enc["z_bg"], steps=steps, deterministic=True)
    finally:
        model.set_launch(None)
    return dict(zip(PARTICLE_KEYS, out))


@torch.no_grad()
def rollout(model: DDLPNovPhy, frames0: torch.Tensor, launch5: torch.Tensor | None, steps: int = STEPS) -> dict:
    """Deterministic rollout from decision frames [B, 3, 128, 128] (float) under launches [B, 5]: particles of the
    static context and of the ``steps`` predicted frames, {key: [B, STATIC + steps, ...]}."""
    return rollout_particles(model, encode_context(model, context(frames0)), launch5, steps)


def expand_particles(enc: dict, repeats: int) -> dict:
    """Each row of {key: [B, ...]} repeated ``repeats`` times (row-major: anchor-major, candidate-minor)."""
    return {k: v.repeat_interleave(repeats, 0) for k, v in enc.items()}


@torch.no_grad()
def frame_particles(model: ObjectDynamicsDLP, frames: torch.Tensor) -> dict:
    """Particles of single true frames [B, 3, 128, 128]: the last frame of their static context (the same encoding
    the rollout applies to the decision frame) -> {key: [B, ...]}."""
    enc = encode_context(model, context(frames))
    return {k: v[:, -1] for k, v in enc.items()}


def particle_tokens(p: dict, t=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens [B, K, 6 + d], background features [B, d_bg]) of particles {key: [B, ...]} (or time index t of
    {key: [B, T, ...]})."""
    sel = (lambda v: v) if t is None else (lambda v: v[:, t])
    z, s, o, d, f, bg = (sel(p[k]) for k in PARTICLE_KEYS)
    return torch.cat([z, s, o.unsqueeze(-1), d, f], -1), bg


# ---------------------------------------------------------------------------------------------------------------
# readout: particles -> E99 carrier (fit frames only; the #130 R1 recipe)
# ---------------------------------------------------------------------------------------------------------------

READOUT = {"dim": 256, "heads": 8, "hidden": 512, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128, "epochs": 25}


def carrier_stats(rows: torch.Tensor):
    rows = torch.as_tensor(rows).float()
    mean = rows.mean(0)
    std = rows.std(0).clamp_min(1e-4)
    return mean, std


class ParticleReadout(nn.Module):
    def __init__(self, token_dim: int, bg_dim: int, mean=None, std=None, dim: int = READOUT["dim"]):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(token_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.embed_bg = nn.Linear(bg_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(1 + SLOTS, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, READOUT["heads"], batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, READOUT["hidden"]), nn.GELU(),
                                 nn.Linear(READOUT["hidden"], dim))
        self.head_header = nn.Linear(dim, CARRIER_HEADER)
        self.head_slot = nn.Linear(dim, SLOT_WIDTH)
        self.register_buffer("mean", torch.zeros(CARRIER_DIM) if mean is None else mean.clone().float())
        self.register_buffer("std", torch.ones(CARRIER_DIM) if std is None else std.clone().float())

    def forward_z(self, tokens, bg):
        k = self.norm(torch.cat([self.embed(tokens.float()), self.embed_bg(bg.float())[:, None]], 1))
        q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        h = q + self.attn(q, k, k, need_weights=False)[0]
        h = h + self.mlp(h)
        return torch.cat([self.head_header(h[:, 0]), self.head_slot(h[:, 1:]).flatten(1)], 1)

    def forward(self, tokens, bg):
        return self.forward_z(tokens, bg) * self.std + self.mean


def fit_readout(tokens, bg, targets, seed: int, device="cpu", epochs=None, log=None):
    """Fit ParticleReadout on (tokens [N, K, F], bg [N, d_bg]) -> targets [N, 236] (MSE on z-scored targets, AdamW)."""
    epochs = READOUT["epochs"] if epochs is None else epochs
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mean, std = carrier_stats(targets)
    probe = ParticleReadout(tokens.shape[-1], bg.shape[-1], mean, std).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=READOUT["lr"], weight_decay=READOUT["weight_decay"])
    yz = (torch.as_tensor(targets).float() - mean) / std
    n, b = len(tokens), READOUT["batch"]
    steps, last = 0, None
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        for i in range(0, n - b + 1, b):
            idx = order[i:i + b]
            loss = F.mse_loss(probe.forward_z(tokens[idx].to(device), bg[idx].to(device)), yz[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            last = float(loss.detach())
        if log is not None:
            log(f"readout epoch {epoch + 1}/{epochs}: loss {last:.5f}")
    probe.eval().requires_grad_(False)  # frozen readout (views of its parameters carry no autograd state)
    return probe, {"steps": steps, "final_loss": last, "samples": n, "epochs": epochs, "seed": seed,
                   **{k: READOUT[k] for k in READOUT}}


@torch.no_grad()
def apply_readout(probe: ParticleReadout, tokens, bg, device="cpu", chunk=4096) -> torch.Tensor:
    out = [probe(tokens[i:i + chunk].to(device), bg[i:i + chunk].to(device)).cpu() for i in range(0, len(tokens), chunk)]
    return torch.cat(out) if out else torch.empty(0, CARRIER_DIM)


def finite_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x).all(-1) if x.dim() > 1 else torch.isfinite(x)


