"""Issue #123: the vendored H-JEPA on NovPhy (arms, clip sampler, launch encoding, training step,
top-level rollout, readout probes and the collapse / action diagnostics).

Nothing here changes the H-JEPA objective (``vendor.hjepa_utils.hjepa_forward``: per-level teacher-forced
latent MSE + lambda * SIGReg, summed over levels, end to end, no EMA / stop-gradient). Adaptations, each
disclosed in docs/issue-123-hjepa-comparison.md:

- layouts H3-Delta (steps of 1 / 5 / 15 carrier frames: s2 = 5, s3 = 3, w = 1, c = 3), flat L1 and H2-fs5;
- the anchor-mixture clip sampler (half of every batch starts at the launch frame 0) or H-JEPA's uniform one;
- the launch as the level-1 action at frame 0 (``[drag_x/480, drag_y/480, release/1000, tap/1000, 1]``),
  zero no-ops after it, no z-scoring (``persist`` repeats it at every frame);
- sparse level-1 inputs: the loader gathers only the frames some kept step reads (``PlannedHJEPA``), with the
  crop starts drawn per batch exactly as ``HJEPA._draw_crops_and_level1_frames`` does (one start per level
  and batch, uniform); the official code encodes exactly these frames too (ARCHITECTURE.md sparse encode);
- the level-1 ViT runs with activation checkpointing (identical gradients; no ViT dropout) at 320 x 240,
  patch 16, with its 20 x 20 position table interpolated by the official ``interpolate_pos_encoding`` path.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vendor import hjepa_utils
from .vendor.models.hjepa import HJEPA
from .vendor.models.module import MLP

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CARRIER_DIM = 236
ACTION_DIM = 5
FRAME_HEIGHT, FRAME_WIDTH = 240, 320
PATCH = 16
VIT_IMAGE_SIZE = 320  # square table (20 x 20 patches); interpolated to the 15 x 20 grid of a 320 x 240 frame
ENDPOINT = 225
POSITION_TIMES = (15, 30, 60, 150, 225)
HISTORY = 3
SIGREG_LAMBDAS = (0.09, 0.18)
LR_LEVEL1, LR_UPPER = 5e-5, 1e-5
WEIGHT_DECAY = 1e-3
WARMUP_FRACTION = 0.01
GRAD_CLIP = 1.0
B1 = {"steps": 9000, "batch": 64}
B3 = {"epochs": 5, "batch": 128}
QUEUE_SIZE = 4000

# layout: level-1 frameskip (carrier frames per level-1 step) and the upper strides
LAYOUTS = {
    "H3": {"frameskip": 1, "strides": (5, 3), "frames_per_top_step": 15},
    "L1": {"frameskip": 1, "strides": (), "frames_per_top_step": 1},
    "H2fs5": {"frameskip": 5, "strides": (3,), "frames_per_top_step": 15},
}

# sizes: level-1 encoder, per-level predictor shapes
SIZES = {
    # the paper's pixel model: ViT-Tiny + MLP projector, L1 predictor 4/8/1024, upper 6/16/2048 (Push-T / Cube), D 192
    "paper": {"D": 192, "encoder": {"type": "vit", "scale": "tiny"}, "encoder_projector": "mlp",
              "upper_encoder_hidden": 384, "predictor_projector": "mlp",
              "level1_predictor": {"depth": 4, "heads": 8, "mlp_dim": 1024, "dim_head": 64},
              "upper_predictor": {"depth": 6, "heads": 16, "mlp_dim": 2048, "dim_head": 64}},
    # paper-sized predictors on carriers: MLP(236 -> 2048 -> 192) + MLP projector
    "native": {"D": 192, "encoder": {"type": "mlp", "input_dim": CARRIER_DIM, "hidden_dim": 2048},
               "encoder_projector": "mlp", "upper_encoder_hidden": 384, "predictor_projector": "mlp",
               "level1_predictor": {"depth": 4, "heads": 8, "mlp_dim": 1024, "dim_head": 64},
               "upper_predictor": {"depth": 6, "heads": 16, "mlp_dim": 2048, "dim_head": 64}},
    # parameter-matched carrier models (target 1,837,690 = the Gate-B unit, +-2 %): MLP(236 -> 384 -> 112) level-1
    # encoder, latent-MLP(112 -> 224 -> 112) upper encoders, no projectors
    "matched": {"D": 112, "encoder": {"type": "mlp", "input_dim": CARRIER_DIM, "hidden_dim": 384},
                "encoder_projector": "identity", "upper_encoder_hidden": 224, "predictor_projector": "identity",
                "level1_predictor": {"depth": 2, "heads": 4, "mlp_dim": 448, "dim_head": 28},
                "upper_predictor": {"depth": 2, "heads": 4, "mlp_dim": 448, "dim_head": 28}},
    "matched-lewm": {"D": 112, "encoder": {"type": "mlp", "input_dim": CARRIER_DIM, "hidden_dim": 384},
                     "encoder_projector": "identity", "upper_encoder_hidden": 224, "predictor_projector": "identity",
                     "level1_predictor": {"depth": 7, "heads": 4, "mlp_dim": 480, "dim_head": 28},
                     "upper_predictor": None},
    "matched-hwm": {"D": 112, "encoder": {"type": "mlp", "input_dim": CARRIER_DIM, "hidden_dim": 384},
                    "encoder_projector": "identity", "upper_encoder_hidden": None, "predictor_projector": "identity",
                    "level1_predictor": {"depth": 2, "heads": 4, "mlp_dim": 512, "dim_head": 28},
                    "upper_predictor": {"depth": 2, "heads": 4, "mlp_dim": 512, "dim_head": 28}},
    "matched-fs5": {"D": 112, "encoder": {"type": "mlp", "input_dim": CARRIER_DIM, "hidden_dim": 384},
                    "encoder_projector": "identity", "upper_encoder_hidden": 224, "predictor_projector": "identity",
                    "level1_predictor": {"depth": 3, "heads": 4, "mlp_dim": 544, "dim_head": 28},
                    "upper_predictor": {"depth": 3, "heads": 4, "mlp_dim": 544, "dim_head": 28}},
}

# arm -> spec. kind: hjepa (latent-MLP upper encoders, SIGReg at every level), hwm (identity upper encoders,
# no upper SIGReg), lewm (one level)
ARMS = {
    "HJ-px": {"input": "pixels", "kind": "hjepa", "layout": "H3", "size": "paper", "budget": "B3",
              "sampler": "mixture", "action": "sparse", "role": "primary, faithful"},
    "HJ-car-m": {"input": "carriers", "kind": "hjepa", "layout": "H3", "size": "matched", "budget": "B1",
                 "sampler": "mixture", "action": "sparse", "role": "primary, controlled"},
    "HJ-px-B1": {"input": "pixels", "kind": "hjepa", "layout": "H3", "size": "paper", "budget": "B1",
                 "sampler": "mixture", "action": "sparse", "role": "budget-matched faithful"},
    "LeWM-px": {"input": "pixels", "kind": "lewm", "layout": "L1", "size": "paper", "budget": "B3",
                "sampler": "mixture", "action": "sparse", "role": "temporal lever (pixels)"},
    "HWM-px": {"input": "pixels", "kind": "hwm", "layout": "H3", "size": "paper", "budget": "B3",
               "sampler": "mixture", "action": "sparse", "role": "representational lever (pixels)"},
    "LeWM-car-m": {"input": "carriers", "kind": "lewm", "layout": "L1", "size": "matched-lewm", "budget": "B1",
                   "sampler": "mixture", "action": "sparse", "role": "temporal lever (carriers)"},
    "HWM-car-m": {"input": "carriers", "kind": "hwm", "layout": "H3", "size": "matched-hwm", "budget": "B1",
                  "sampler": "mixture", "action": "sparse", "role": "representational lever (carriers)"},
    "HJ-car-native": {"input": "carriers", "kind": "hjepa", "layout": "H3", "size": "native", "budget": "B3",
                      "sampler": "mixture", "action": "sparse", "role": "native architecture on our input"},
    "HJ-car-m-uniform": {"input": "carriers", "kind": "hjepa", "layout": "H3", "size": "matched", "budget": "B1",
                         "sampler": "uniform", "action": "sparse", "role": "ablation: H-JEPA's uniform sampler"},
    "HJ-car-m-persist": {"input": "carriers", "kind": "hjepa", "layout": "H3", "size": "matched", "budget": "B1",
                         "sampler": "mixture", "action": "persist", "role": "ablation: launch at every step"},
    "HJ-car-m-fs5": {"input": "carriers", "kind": "hjepa", "layout": "H2fs5", "size": "matched-fs5",
                     "budget": "B1", "sampler": "mixture", "action": "sparse",
                     "role": "ablation: 2 levels, level-1 frameskip 5 (no Delta-1 level)"},
}
HJ_ARMS = ("HJ-px", "HJ-car-m", "HJ-px-B1", "HJ-car-native", "HJ-car-m-uniform", "HJ-car-m-persist", "HJ-car-m-fs5")


class Cfg(dict):
    """Attribute access over a nested dict: the subset of the OmegaConf interface create_world_model uses."""

    def __init__(self, data=(), **kwargs):
        super().__init__()
        for key, value in dict(data, **kwargs).items():
            self[key] = Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def num_levels(arm):
    return 1 + len(LAYOUTS[ARMS[arm]["layout"]]["strides"])


def arm_config(arm, lam):
    """The create_world_model config of an arm at SIGReg weight lam (H-JEPA's per-environment config layout)."""
    spec = ARMS[arm]
    size = SIZES[spec["size"]]
    layout = LAYOUTS[spec["layout"]]
    D = size["D"]
    levels = num_levels(arm)

    def predictor(shape):
        return {**shape, "dropout": 0.1, "emb_dropout": 0.0, "hidden_dim": D,
                "projector": {"type": size["predictor_projector"]}}

    encoder = dict(size["encoder"])
    if encoder["type"] == "mlp":
        encoder["output_dim"] = D
    cfg = {"num_levels": levels, "img_size": VIT_IMAGE_SIZE, "patch_size": PATCH,
           "data": {"dataset": {"level1": {"frameskip": layout["frameskip"]}}},
           "optimizer": {"type": "AdamW", "lr": LR_LEVEL1, "weight_decay": WEIGHT_DECAY},
           "level1": {"lr": LR_LEVEL1,
                      "wm": {"history_size": HISTORY, "rollout_n": 1, "embed_dim": D, "use_proprio": False,
                             "action_dim": ACTION_DIM},
                      "encoder": {**encoder, "projector": {"type": size["encoder_projector"]}},
                      "predictor": predictor(size["level1_predictor"]),
                      "loss": {"embed": {"sigreg": {"enabled": True, "weight": lam}}}}}
    for level, stride in enumerate(layout["strides"], start=2):
        if spec["kind"] == "hwm":
            upper_encoder = {"type": "identity", "input_dim": D, "projector": {"type": "identity"}}
        else:
            upper_encoder = {"type": "latent_mlp", "input_dim": D, "hidden_dim": size["upper_encoder_hidden"],
                             "final_ln": False, "output_dim": D, "projector": {"type": "identity"}}
        cfg[f"level{level}"] = {
            "lr": LR_UPPER, "stride": stride, "window_size": 1,
            "wm": {"history_size": HISTORY, "rollout_n": 1, "embed_dim": D, "use_proprio": False},
            "encoder": upper_encoder,
            "action_encoder": {"output_dim": 4, "input_dim": D, "max_chunk": stride + 1, "queue_size": QUEUE_SIZE},
            "action_embed": {"patch_embed": False, "input_dim": 4},
            "predictor": predictor(size["upper_predictor"]),
            "loss": {"embed": {"sigreg": {"enabled": spec["kind"] != "hwm", "weight": lam}}}}
    return Cfg(cfg)


class PlannedHJEPA(HJEPA):
    """HJEPA whose per-batch crop starts and level-1 frame set come from the loader (``plan``), and whose
    level-1 input holds only those frames in sorted order. With ``plan = None`` it is the official class.

    The planned input is not aligned with the level-1 time axis, so the "*_level{N}" aliases of the raw
    input (decoder / probe target streams only; no loss of these arms reads them) are not created."""

    plan = None  # (seq_len, {level: start or None}, sorted level-1 frame indices)

    def _add_level_target_aliases(self, info, level):
        if self.plan is None:
            super()._add_level_target_aliases(info, level)

    def _draw_crops_and_level1_frames(self, seq_len, max_level, device):
        if self.plan is None:
            return super()._draw_crops_and_level1_frames(seq_len, max_level, device)
        return dict(self.plan[1]), list(self.plan[2])

    def _encode_level1(self, info, key, frame_indices):
        if self.plan is None:
            return super()._encode_level1(info, key, frame_indices)
        seq_len = int(self.plan[0])
        ref = info[key]
        if ref.size(1) != len(frame_indices):
            raise ValueError("planned level-1 input must hold exactly the planned frames")
        idx = torch.as_tensor(frame_indices, device=ref.device, dtype=torch.long)
        level_out = self.jepas[0].encode(dict(info), key=key)
        for output_name in ("embed", "pixel_embed", "proprio_embed"):
            sparse = level_out.get(f"{output_name}_0")
            if sparse is None:
                continue
            full = sparse.new_zeros(sparse.size(0), seq_len, *sparse.shape[2:])
            full[:, idx] = sparse
            info[f"{output_name}_1"] = full
        if "action_0" in level_out:
            info["action_1"] = level_out["action_0"]


def build_model(arm, lam, checkpointing=None):
    """The vendored create_world_model for the arm; returns (PlannedHJEPA, {sigreg modules})."""
    model, losses = hjepa_utils.create_world_model(arm_config(arm, lam))
    model.__class__ = PlannedHJEPA
    if checkpointing is None:
        checkpointing = ARMS[arm]["input"] == "pixels"
    if checkpointing:
        vit = model.get_level(1).encoder.encoder
        vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, losses


def parameter_count(arm, lam=SIGREG_LAMBDAS[0]):
    model, _ = build_model(arm, lam, checkpointing=False)
    return sum(p.numel() for p in model.parameters())


def normalize_uint8_images_on_device(batch):
    """main_hjepa._normalize_uint8_images_on_device (official, verbatim semantics)."""
    for key, value in list(batch.items()):
        if not key.startswith("pixels"):
            continue
        if not torch.is_tensor(value) or value.dtype != torch.uint8:
            continue
        if value.ndim < 4 or value.size(-3) != 3:
            raise ValueError(f"Expected uint8 pixel tensor '{key}' to have channel-first RGB shape")
        view_shape = [1] * value.ndim
        view_shape[-3] = 3
        mean = value.new_tensor(IMAGENET_MEAN, dtype=torch.float32).view(*view_shape)
        std = value.new_tensor(IMAGENET_STD, dtype=torch.float32).view(*view_shape)
        batch[key] = value.float().div(255.0).sub(mean).div(std)


class Wrapper(nn.Module):
    """The ``self`` hjepa_forward expects: ``model``, the sigreg_level{N} modules and ``log_dict``."""

    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, module in losses.items():
            setattr(self, name, module)
        self.logged = {}

    def log_dict(self, values, **_):
        self.logged = {k: float(v) for k, v in values.items()}


def training_cfg(arm, lam):
    return arm_config(arm, lam)


def make_optimizer(model, cfg):
    """main_hjepa._build_hjepa_optimizer_factory + spt create_optimizer({type: AdamW, lr, weight_decay})."""
    groups = [{"params": [p for p in model.get_level(level).parameters() if p.requires_grad],
               "lr": cfg[f"level{level}"].lr, "name": f"level{level}"} for level in range(1, int(cfg.num_levels) + 1)]
    return torch.optim.AdamW(groups, lr=LR_LEVEL1, weight_decay=WEIGHT_DECAY)


class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
    """spt 0.1.7 LinearWarmupCosineAnnealingLR (warmup_start_lr 0, eta_min 0), stepped per optimizer step."""

    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        return [(base_lr) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps)
                                          / (self.max_steps - self.warmup_steps))) / 2 for base_lr in self.base_lrs]


def warmup_steps(total_steps):
    """spt default for LinearWarmupCosineAnnealingLR: max(1, int(0.01 * estimated_stepping_batches))."""
    return max(1, int(WARMUP_FRACTION * total_steps))


def train_step(wrapper, cfg, optimizer, scheduler, batch, autocast=True):
    """One spt manual-optimisation step: forward under bf16-mixed autocast, backward, global grad-norm clip
    1.0 over the optimizer's parameters, AdamW step, scheduler step, zero_grad(set_to_none)."""
    device_type = "cuda" if next(wrapper.parameters()).is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast):
        output = hjepa_utils.hjepa_forward(wrapper, batch, "fit", cfg,
                                           normalize_batch=normalize_uint8_images_on_device)
    loss = output["loss"]
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "grad_norm": float(norm), **wrapper.logged}


# ---------------------------------------------------------------------------
# clip geometry, sampler and launch encoding
# ---------------------------------------------------------------------------

class Geometry:
    """Clip geometry of an arm (H-JEPA's dataset recursion n_{l-1} = (n_l - 1) s_l + w_l, n_L = c + 1)."""

    def __init__(self, arm):
        layout = LAYOUTS[ARMS[arm]["layout"]]
        self.frameskip = layout["frameskip"]
        self.strides = tuple(layout["strides"])
        self.levels = 1 + len(self.strides)
        self.target = HISTORY + 1
        n = self.target
        for stride in reversed(self.strides):
            n = (n - 1) * stride + 1
        self.level1_steps = n  # level-1 steps per clip
        self.span = n * self.frameskip  # carrier frames per clip
        lengths = [n]
        for stride in self.strides:
            lengths.append((lengths[-1] - 1) // stride + 1)
        self.lengths = lengths
        self.top_steps = ENDPOINT // layout["frames_per_top_step"]
        self.frames_per_top_step = layout["frames_per_top_step"]

    def valid_starts(self, end):
        """Clip starts of a shot whose positions are 0..end (every start with a full span)."""
        return max(0, end + 1 - self.span + 1)

    def draw_crops(self, rng):
        """One crop start per level (None when the level holds exactly target steps), as HJEPA does."""
        starts = {}
        for level, length in enumerate(self.lengths, start=1):
            starts[level] = None if length <= self.target else int(rng.integers(0, length - self.target + 1))
        return starts

    def needed(self, starts):
        """HJEPA._draw_crops_and_level1_frames: the level-1 steps any kept step reads."""
        kept = {}
        for level, length in enumerate(self.lengths, start=1):
            s = starts[level]
            kept[level] = set(range(length)) if s is None else set(range(s, s + self.target))
        needed = set(kept[self.levels])
        for level in range(self.levels, 1, -1):
            stride = self.strides[level - 2]
            needed = kept[level - 1] | {t * stride for t in needed}
        return sorted(needed)


def level1_actions(launch, start, geometry, persist):
    """Level-1 actions of a clip starting at carrier frame ``start``: the launch at frame 0, no-ops after
    (``persist``: the launch at every frame); level-1 step k concatenates the frameskip frame actions."""
    frames = torch.arange(start, start + geometry.span)
    actions = torch.zeros(geometry.span, ACTION_DIM)
    mask = torch.ones_like(frames, dtype=torch.bool) if persist else frames == 0
    actions[mask] = launch.to(actions.dtype)
    return actions.reshape(geometry.level1_steps, geometry.frameskip * ACTION_DIM)


def plan_batch(rng, shot_ends, cumulative, geometry, sampler, batch):
    """(shot index, start) of every clip of a batch plus the shared crop starts and level-1 step set.

    mixture: the first batch // 2 clips start at the launch frame of a uniformly drawn shot, the rest are
    uniform over all valid (shot, start) pairs; uniform: every clip uniform over valid pairs."""
    total = int(cumulative[-1])
    anchors = batch // 2 if sampler == "mixture" else 0
    shots = rng.integers(0, len(shot_ends), size=anchors)
    flat = rng.integers(0, total, size=batch - anchors)
    uniform_shots = np.searchsorted(cumulative, flat, side="right") - 1
    uniform_starts = flat - cumulative[uniform_shots]
    clips = [(int(s), 0) for s in shots] + [(int(s), int(t)) for s, t in zip(uniform_shots, uniform_starts)]
    starts = geometry.draw_crops(rng)
    return clips, starts, geometry.needed(starts)


def steps_for(arm, valid_clips):
    budget = ARMS[arm]["budget"]
    if budget == "B1":
        return B1["steps"], B1["batch"]
    return (valid_clips // B3["batch"]) * B3["epochs"], B3["batch"]


# ---------------------------------------------------------------------------
# evaluation-time encoding and top-level rollout (eval mode, no grad)
# ---------------------------------------------------------------------------

def encode_levels(model, x):
    """Per-level encodings of single frames (stride 1, window 1; final_probing_decoding_eval semantics).
    x: uint8 [N, 3, H, W] pixels or float [N, 236] carriers. Returns a list of [N, D] tensors."""
    if x.dtype == torch.uint8:
        batch = {"pixels": x}
        normalize_uint8_images_on_device(batch)
        x = batch["pixels"]
    z = model.get_level(1).encoder(x.float())
    out = [z]
    for level in range(2, model.num_levels + 1):
        z = model.get_level(level).encoder(z)
        out.append(z)
    return out


def top_action_embeddings(model, launches, geometry, persist):
    """Embedded top-level actions [N, top_steps, D] of each launch over the 225-frame decision horizon."""
    n = launches.size(0)
    steps1 = ENDPOINT // geometry.frameskip
    frames = torch.zeros(n, ENDPOINT, ACTION_DIM, device=launches.device)
    if persist:
        frames[:] = launches[:, None]
    else:
        frames[:, 0] = launches
    act = frames.reshape(n, steps1, geometry.frameskip * ACTION_DIM)
    jepa = model.get_level(1)
    act = jepa.action_embed(jepa.action_encoder(act))
    for level in range(2, model.num_levels + 1):
        jepa = model.get_level(level)
        chunked = jepa._chunk_temporal_info({"ref": act[..., :1], "action": act}, key="ref",
                                            stride=jepa.temporal_stride, window_size=jepa.temporal_window_size)
        act = jepa.action_embed(jepa.action_encoder(chunked["action"], chunked["action_mask"]))
    if act.size(1) != geometry.top_steps:
        raise ValueError("top-level action count differs from the decision horizon")
    return act


def rollout_top(model, z0, act_emb):
    """JEPA.rollout semantics at the top level (history window HISTORY): states 0..T, [N, T+1, D]."""
    jepa = model.get_level(model.num_levels)
    emb = z0[:, None]
    for t in range(act_emb.size(1)):
        lo = max(0, t + 1 - HISTORY)
        pred = jepa.predict(emb[:, lo:t + 1], act_emb[:, lo:t + 1])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return emb


# ---------------------------------------------------------------------------
# readout probes (H-JEPA App. D: MLP(D -> 512, LayerNorm, GELU -> out), AdamW, z-scored targets)
# ---------------------------------------------------------------------------

PROBE = {"hidden": 512, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128, "epochs": 25}


class Probe(nn.Module):
    def __init__(self, input_dim, output_dim, mean, std):
        super().__init__()
        self.net = MLP(input_dim=input_dim, hidden_dim=PROBE["hidden"], output_dim=output_dim)
        self.register_buffer("mean", mean.clone().float())
        self.register_buffer("std", std.clone().float())

    def forward(self, z):
        return self.net(z) * self.std + self.mean


def target_stats(y):
    mean = y.mean(0)
    std = y.std(0)
    return mean, torch.where(std > 1e-6, std, torch.ones_like(std))


def fit_probe(x, y, seed, epochs=PROBE["epochs"], extra_loss=None, device="cuda"):
    """Fit Probe(x -> y) with MSE on z-scored targets (deterministic given the seed and the device policy).
    extra_loss(probe) -> scalar is added at every step when given (R4 ranking term)."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mean, std = target_stats(y)
    probe = Probe(x.size(1), y.size(1), mean, std).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=PROBE["lr"], weight_decay=PROBE["weight_decay"])
    x, yz = x.to(device), ((y - mean) / std).to(device)
    n = x.size(0)
    steps = 0
    last = None
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        for i in range(0, n - PROBE["batch"] + 1, PROBE["batch"]):
            idx = order[i:i + PROBE["batch"]]
            loss = F.mse_loss(probe.net(x[idx]), yz[idx])
            if extra_loss is not None:
                loss = loss + extra_loss(probe)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            last = float(loss.detach())
    probe.eval()
    return probe, {"steps": steps, "final_loss": last, "samples": n}


class SlotProbe(nn.Module):
    """P_c: slot presence logits and normalized centres of the 18 vocabulary slots from a latent."""

    def __init__(self, input_dim, slots=18):
        super().__init__()
        self.slots = slots
        self.net = MLP(input_dim=input_dim, hidden_dim=PROBE["hidden"], output_dim=slots * 3)

    def forward(self, z):
        out = self.net(z).view(-1, self.slots, 3)
        return out[..., 0], out[..., 1:]


def fit_slot_probe(x, presence, centers, seed, epochs=PROBE["epochs"], device="cuda"):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    probe = SlotProbe(x.size(1), presence.size(1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=PROBE["lr"], weight_decay=PROBE["weight_decay"])
    x, presence, centers = x.to(device), presence.to(device).float(), centers.to(device).float()
    n = x.size(0)
    last = None
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(device)
        for i in range(0, n - PROBE["batch"] + 1, PROBE["batch"]):
            idx = order[i:i + PROBE["batch"]]
            logits, pred = probe(x[idx])
            present = presence[idx] > 0.5
            loss = F.binary_cross_entropy_with_logits(logits, presence[idx])
            if bool(present.any()):
                loss = loss + F.mse_loss(pred[present], centers[idx][present])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last = float(loss.detach())
    probe.eval()
    return probe, {"final_loss": last, "samples": n}


def position_mse(pred_centers, presence, centers):
    """#100 engine_position_mse on probe centres: MSE over engine-present slots and both coordinates."""
    present = presence > 0.5
    if not bool(present.any()):
        return None
    return float((pred_centers[present] - centers[present]).square().mean())


# ---------------------------------------------------------------------------
# collapse and action diagnostics (H-JEPA App. K / L)
# ---------------------------------------------------------------------------

def effective_rank(z):
    """Roy & Vetterli effective rank exp(H(p)), p = singular values / their sum, of the centred embeddings."""
    z = z.double()
    s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def within_episode_share(z, episodes):
    """chi (App. L Eq. 10): E_b[tr Var_t(z | b)] / tr Var(z), episodes = shots (frame-weighted)."""
    z = z.double()
    total = float(((z - z.mean(0)) ** 2).sum(1).mean())
    _, inverse = torch.unique(torch.as_tensor(episodes), return_inverse=True)
    groups = int(inverse.max()) + 1
    counts = torch.zeros(groups, dtype=z.dtype).index_add_(0, inverse, torch.ones(len(z), dtype=z.dtype))
    means = torch.zeros(groups, z.size(1), dtype=z.dtype).index_add_(0, inverse, z) / counts[:, None]
    within = float(((z - means[inverse]) ** 2).sum(1).mean())
    return within / total if total > 0 else 0.0


def action_sensitivity(endpoints):
    """Relative change of the predicted endpoint when the launch is replaced by another grid launch:
    mean over anchors and ordered launch pairs (i != j) of ||z_i - z_j|| / ||z_i||. endpoints [A, K, D]."""
    z = endpoints.double()
    diff = torch.cdist(z, z)  # [A, K, K]
    norms = z.norm(dim=-1)  # [A, K]
    k = z.size(1)
    off = ~torch.eye(k, dtype=torch.bool)
    rel = diff / norms[:, :, None].clamp_min(1e-12)
    return float(rel[:, off].mean())
