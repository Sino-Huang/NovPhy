"""Issue #138 (#127 coverage arm T6): V-JEPA 2-AC on NovPhy decision frames.

Adapter (Decisions on #138: P-route issuecomment-6436321490, P-adapter issuecomment-6436355731; the vendored code at
``vendor/`` (facebookresearch/vjepa2 @ 204698b) differs from upstream only by package-relative imports):

* **Encoder** (frozen): the ViT-g/16 ``encoder`` of the released ``vjepa2-ac-vitg.pt`` built and loaded as the hub
  entry ``vjepa2_ac_vit_giant`` does. Each frame is encoded on its own as ``forward_target`` does (repeated to a
  2-frame tubelet; 256 tokens x 1408) under bf16 autocast (the config's ``meta.dtype``); ``normalize`` is the
  ``normalize_reps`` layer norm.
* **Preprocessing** = the official ``VideoTransform`` (``app/vjepa_droid/transforms.py``, the config's scale
  [1.777, 1.777] / ratio [0.75, 1.35]) on a 320 x 240 frame: the random crop always takes its deterministic fallback
  (the whole frame), bilinear ``interpolate`` to 256 x 256, ImageNet mean / std in 0..255 units.
* **Predictor** = the official ``VisionTransformerPredictorAC`` at the config's size (depth 24, width 1024, 16 heads,
  RoPE, frame-causal), ``action_embed_dim`` 5 (``eb.launch5`` at t0, zeros after; zero state tokens), built with
  ``num_frames`` 64 as the hub does (the config's ``num_frames`` 8 builds a 4-frame mask and crashes the 7-frame
  teacher-forcing forward upstream).
* **Loss / step** = ``app/vjepa_droid/train.py``'s ``forward_predictions`` + ``loss_fn`` (teacher forcing + an
  ``auto_steps`` rollout, layer-normed predictions, L1) and its update (``WSDSchedule`` / ``CosineWDSchedule`` stepped
  first, AdamW groups of ``init_opt``, bf16 autocast + ``GradScaler`` on CUDA); one update = 256 clips as 32
  micro-batches of 8 with ``loss / 32`` accumulated (= DDP's mean over 32 equal ranks).
* **Rollout** = ``step_predictor`` of ``notebooks/utils/world_model_wrapper.py`` (predict, keep the last frame's
  tokens, layer norm) for 45 steps to frame 225 over a sliding context of the last <= 7 frames.
* **Readout** (R1, as C1-dino): the #130 ``PatchReadout`` / ``fit_readout`` (imported unchanged from
  ``world_model.baselines.dino_wm.novphy``) on normalised features [256, 1408].
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
VENDOR = Path(__file__).resolve().parent / "vendor"
EXTERNAL = ROOT / "data/external/vjepa2"
AC_CHECKPOINT = EXTERNAL / "vjepa2-ac-vitg.pt"
AC_CHECKPOINT_SHA256 = "sha256:0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b"
_PACKAGE = "world_model.baselines.vjepa2.vendor"

# frames
FRAME_HW = (240, 320)
CROP = 256
PATCH = 16
TUBELET = 2
GRID = CROP // PATCH  # 16
TOKENS = GRID * GRID  # 256 tokens per frame
EMBED_DIM = 1408  # ViT-g
NORMALIZE = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # make_transforms default
STRIDE = 5
ENDPOINT = 225
STEPS = ENDPOINT // STRIDE  # 45
POSITIONS = tuple(range(0, ENDPOINT + 1, STRIDE))  # 46 frames per shot
CLIP_FRAMES = 8  # dataset_fpcs
CONTEXT = CLIP_FRAMES - 1  # longest teacher-forcing context (frames 0..6)
LAUNCH_DIM = 5
MASK_FRAMES = 64  # the hub's num_frames (frame-causal mask over 32 frames)

# configs/train/vitg16/droid-256px-8f.yaml @ 204698b
ENCODER = {"model_name": "vit_giant_xformers", "img_size": CROP, "patch_size": PATCH, "num_frames": 64,
           "tubelet_size": TUBELET, "use_sdpa": True, "uniform_power": False, "use_rope": True}  # hub kwargs
PREDICTOR = {"depth": 24, "embed_dim": 1024, "num_heads": 16, "is_frame_causal": True, "use_rope": True,
             "uniform_power": True, "use_sdpa": True, "use_silu": False, "wide_silu": True, "use_extrinsics": False}
OPTIMIZATION = {"ipe": 300, "epochs": 315, "warmup": 15, "anneal": 15, "lr": 4.25e-4, "start_lr": 7.5e-5,
                "final_lr": 0.0, "weight_decay": 0.04, "final_weight_decay": 0.04, "betas": (0.9, 0.999),
                "eps": 1e-8}
LOSS = {"auto_steps": 2, "loss_exp": 1.0, "normalize_reps": True}
GLOBAL_BATCH = 8 * 32  # batch_size x world_size of the released checkpoint
MICRO_BATCH = 8
NATIVE_UPDATES = OPTIMIZATION["ipe"] * OPTIMIZATION["epochs"]  # 94 500


# ---------------------------------------------------------------------------------------------------------------
# vendored modules
# ---------------------------------------------------------------------------------------------------------------

def _drop_path_unused(*args, **kwargs):
    raise RuntimeError("timm drop_path called: T6 modules use drop_path_rate 0 (vendor stub)")


_VENDOR = None


def vendor() -> types.SimpleNamespace:
    """The official modules from ``vendor/src``; ``modules.py``'s ``from timm.models.layers import drop_path`` is
    served by a raising stub during this load only when timm is absent (env ``novphy`` has no timm / torchvision)."""
    global _VENDOR
    if _VENDOR is not None:
        return _VENDOR
    stubbed = []
    if importlib.util.find_spec("timm") is None:
        layers = types.ModuleType("timm.models.layers")
        layers.drop_path = _drop_path_unused
        stubs = {"timm": types.ModuleType("timm"), "timm.models": types.ModuleType("timm.models"),
                 "timm.models.layers": layers}
        stubbed = list(stubs)
        sys.modules.update(stubs)
    try:
        ac = importlib.import_module(f"{_PACKAGE}.src.models.ac_predictor")
        vt = importlib.import_module(f"{_PACKAGE}.src.models.vision_transformer")
        schedulers = importlib.import_module(f"{_PACKAGE}.src.utils.schedulers")
        modules = importlib.import_module(f"{_PACKAGE}.src.models.utils.modules")
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)
    _VENDOR = types.SimpleNamespace(ac_predictor=ac, vision_transformer=vt, schedulers=schedulers, modules=modules)
    return _VENDOR


# ---------------------------------------------------------------------------------------------------------------
# frames and the frozen encoder
# ---------------------------------------------------------------------------------------------------------------

def preprocess(frames_u8):
    """uint8 [N, 3, 240, 320] -> float32 [N, 3, 256, 256]: the official VideoTransform's fallback crop (whole
    frame), bilinear interpolate (align_corners False, no antialias) and (x - 255 mean) / (255 std)."""
    x = torch.as_tensor(frames_u8).to(torch.float32)
    x = F.interpolate(x, size=(CROP, CROP), mode="bilinear", align_corners=False)
    mean = torch.tensor(NORMALIZE[0], dtype=torch.float32, device=x.device) * 255.0
    std = torch.tensor(NORMALIZE[1], dtype=torch.float32, device=x.device) * 255.0
    return (x - mean[:, None, None]) / std[:, None, None]


def build_encoder():
    kwargs = {k: value for k, value in ENCODER.items() if k != "model_name"}
    kwargs["img_size"] = (CROP, CROP)
    return vendor().vision_transformer.__dict__[ENCODER["model_name"]](**kwargs)


def _clean_backbone_key(state_dict):
    """src/hub/backbones.py::_clean_backbone_key."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def load_checkpoint(path=AC_CHECKPOINT):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def load_encoder(device="cpu", checkpoint=None):
    """(encoder, load message) with the released AC encoder (hub: load_state_dict(strict=False)); frozen, eval."""
    checkpoint = load_checkpoint() if checkpoint is None else checkpoint
    encoder = build_encoder()
    msg = encoder.load_state_dict(_clean_backbone_key(checkpoint["encoder"]), strict=False)
    encoder = encoder.to(device).eval().requires_grad_(False)
    return encoder, {"missing_keys": list(msg.missing_keys), "unexpected_keys": list(msg.unexpected_keys)}


@torch.no_grad()
def encode(encoder, frames_u8, autocast=True):
    """Raw encoder tokens fp32 [N, 256, 1408] of uint8 frames [N, 3, 240, 320] (forward_target per frame: the frame
    repeated to a 2-frame tubelet), under bf16 autocast on the encoder's device when ``autocast``."""
    device = next(encoder.parameters()).device
    x = preprocess(torch.as_tensor(frames_u8).to(device))
    x = x.unsqueeze(2).repeat(1, 1, TUBELET, 1, 1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
        h = encoder(x)
    return h.float()


def normalize(h):
    """normalize_reps: F.layer_norm over the feature dimension (no affine)."""
    return F.layer_norm(h, (h.size(-1),))


# ---------------------------------------------------------------------------------------------------------------
# the action-conditioned predictor, its losses and its update
# ---------------------------------------------------------------------------------------------------------------

def build_predictor(use_activation_checkpointing=False, action_dim=LAUNCH_DIM):
    """VisionTransformerPredictorAC as init_video_model builds it (config sizes), num_frames = MASK_FRAMES
    (action_dim 7 = the released DROID predictor, for loading its weights in G-port)."""
    v = vendor()
    p = PREDICTOR
    return v.ac_predictor.vit_ac_predictor(
        img_size=CROP, patch_size=PATCH, num_frames=MASK_FRAMES, tubelet_size=TUBELET, embed_dim=EMBED_DIM,
        predictor_embed_dim=p["embed_dim"], action_embed_dim=action_dim, depth=p["depth"],
        is_frame_causal=p["is_frame_causal"], num_heads=p["num_heads"], uniform_power=p["uniform_power"],
        use_rope=p["use_rope"], use_sdpa=p["use_sdpa"], use_silu=p["use_silu"], wide_silu=p["wide_silu"],
        use_extrinsics=p["use_extrinsics"], use_activation_checkpointing=use_activation_checkpointing)


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_actions(launch5, starts, frames=CLIP_FRAMES):
    """Actions [B, frames - 1, 5] of clips starting at step s: the launch iff the absolute step is 0, else zeros."""
    launch5 = torch.as_tensor(launch5, dtype=torch.float32)
    steps = torch.as_tensor(starts)[:, None] + torch.arange(frames - 1)[None]
    return launch5[:, None, :] * (steps == 0)[..., None].to(torch.float32)


def clip_states(batch, frames=CLIP_FRAMES):
    """NovPhy has no proprioceptive state: zero state inputs [B, frames, 5]."""
    return torch.zeros(batch, frames, LAUNCH_DIM)


def forward_predictions(predictor, z, actions, states, auto_steps, tokens_per_frame=TOKENS, normalize_reps=True):
    """train.py::forward_predictions (no extrinsics): teacher forcing over frames 0..T-2, then the auto-regressive
    rollout from [frame 0, the TF prediction of frame 1] to auto_steps predicted frames."""
    def _step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        return _z

    _z, _a, _s = z[:, :-tokens_per_frame], actions, states[:, :-1]
    z_tf = _step_predictor(_z, _a, _s)
    _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
    for n in range(1, auto_steps):
        _a, _s = actions[:, : n + 1], states[:, : n + 1]
        _z_nxt = _step_predictor(_z, _a, _s)[:, -tokens_per_frame:]
        _z = torch.cat([_z, _z_nxt], dim=1)
    z_ar = _z[:, tokens_per_frame:]
    return z_tf, z_ar


def loss_fn(z, h, tokens_per_frame=TOKENS, loss_exp=LOSS["loss_exp"]):
    """train.py::loss_fn."""
    _h = h[:, tokens_per_frame: z.size(1) + tokens_per_frame]
    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp


def clip_loss(predictor, h, actions, states, auto_steps):
    """(loss, jloss, sloss) of normalised clip features h [B, T * 256, 1408]."""
    z_tf, z_ar = forward_predictions(predictor, h, actions, states, auto_steps)
    jloss = loss_fn(z_tf, h)
    sloss = loss_fn(z_ar, h)
    return jloss + sloss, jloss, sloss


def optimization(spec):
    """The spec's optimisation block (OPTIMIZATION with the configuration's overrides)."""
    return {**OPTIMIZATION, **spec.get("optimization", {})}


def build_optimizer(predictor, spec):
    """(optimizer, lr scheduler, wd scheduler) = init_opt's AdamW groups (predictor groups only: the frozen encoder
    has no gradients) + the vendored WSDSchedule / CosineWDSchedule over the native update count."""
    o = optimization(spec)
    groups = [
        {"params": [p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)]},
        {"params": [p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)],
         "WD_exclude": True, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(groups, betas=o["betas"], eps=o["eps"])
    s = vendor().schedulers
    ipe = o["ipe"]
    scheduler = s.WSDSchedule(optimizer, warmup_steps=int(o["warmup"] * ipe), anneal_steps=int(o["anneal"] * ipe),
                              start_lr=o["start_lr"], ref_lr=o["lr"], final_lr=o["final_lr"],
                              T_max=int(o["epochs"] * ipe))
    wd_scheduler = s.CosineWDSchedule(optimizer, ref_wd=o["weight_decay"], final_wd=o["final_weight_decay"],
                                      T_max=int(o["epochs"] * ipe))
    return optimizer, scheduler, wd_scheduler


def train_step(predictor, optimizer, scheduler, wd_scheduler, micro_batches, auto_steps, scaler=None,
               autocast_dtype=None):
    """One official update (train.py::train_step) over micro-batches [(h, actions, states)]: schedulers first, then
    forward / backward of every micro-batch with loss / len(micro_batches), the (scaled) optimizer step, zero_grad."""
    new_lr = scheduler.step()
    new_wd = wd_scheduler.step()
    n = len(micro_batches)
    totals = {"loss": 0.0, "jloss": 0.0, "sloss": 0.0}
    for h, actions, states in micro_batches:
        context = (torch.autocast(device_type=h.device.type, dtype=autocast_dtype) if autocast_dtype is not None
                   else contextlib.nullcontext())
        with context:
            loss, jloss, sloss = clip_loss(predictor, h, actions, states, auto_steps)
        scaled = loss / n
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        for key, value in (("loss", loss), ("jloss", jloss), ("sloss", sloss)):
            totals[key] += float(value.detach()) / n
    if scaler is not None:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()
    return {**totals, "lr": new_lr, "wd": new_wd}


# ---------------------------------------------------------------------------------------------------------------
# rollout (scoring)
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def rollout(predictor, z0, launch5, steps=STEPS, context=CONTEXT, keep=(STEPS,)):
    """Autoregressive latents from normalised decision-frame tokens z0 [B, 256, 1408] under launches [B, 5] (launch
    at step 0, no-ops after): step_predictor (predict, last frame's tokens, layer norm) over a sliding context of the
    last <= ``context`` frames. Returns {step: [B, 256, 1408]} for the steps in ``keep``."""
    b = z0.shape[0]
    launch5 = torch.as_tensor(launch5, dtype=z0.dtype, device=z0.device)
    frames, out = [z0], {}
    for t in range(steps):
        window = frames[-context:]
        first = t + 1 - len(window)  # absolute step of the window's first frame
        actions = torch.zeros(b, len(window), LAUNCH_DIM, dtype=z0.dtype, device=z0.device)
        if first == 0:
            actions[:, 0] = launch5
        states = torch.zeros_like(actions)
        nxt = predictor(torch.cat(window, dim=1), actions, states)[:, -TOKENS:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),))
        frames.append(nxt)
        if t + 1 in keep:
            out[t + 1] = nxt
    return out


# ---------------------------------------------------------------------------------------------------------------
# R1 readout (as C1-dino; the #130 recipe imported unchanged)
# ---------------------------------------------------------------------------------------------------------------

def readout_module():
    from world_model.baselines.dino_wm import novphy as dn
    return dn


class NormalizedFeatures:
    """Read-only view of a raw fp16 feature store [N, 256, 1408] that yields normalised float32 rows (the R1 input
    space = the predictor's output space)."""

    def __init__(self, store):
        self.store = store
        self.shape = store.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(np.ascontiguousarray(self.store[index])).float()
        return normalize(x).numpy()


def new_readout():
    return readout_module().PatchReadout(TOKENS, EMBED_DIM)


def fit_readout(features, rows, targets, seed, device="cpu", epochs=None, log=None):
    """#130 fit_readout on normalised features (features: NormalizedFeatures or a normalised array)."""
    return readout_module().fit_readout(features, rows, targets, seed, device=device, epochs=epochs, log=log)


@torch.no_grad()
def apply_readout(probe, features, device="cpu"):
    return readout_module().apply_readout(probe, features, device)


def module_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters())
