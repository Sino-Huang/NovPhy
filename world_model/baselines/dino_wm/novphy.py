"""#130 C1 DINO-WM adapter: the official predictor (vendored ViTPredictor + ProprioceptiveEmbedding) with the
predictor path of ``models/visual_world_model.py`` and the optimizer setup of ``train.py`` ported line for line
(G-port: ``scripts/b127_dino_wm_port_reference.py``), on precomputed DINOv2 ViT-S/14 patch features (C1-dino) or on
E99 carrier slot tokens (C1-car).

Official semantics kept (gaoyuezhou/dino_wm @ 0a9492f; decisions on #130):
- ``concat_dim=1``: the action embedding (and, in the official PushT configuration only, the proprio embedding) is
  tiled onto every patch token; the predictor input dim is visual_dim + embeddings, with no input projection.
- Training: windows of ``num_hist + num_pred`` = 4 frames at frame-skip 5; ``z_src = z[:, :3]``, ``z_tgt = z[:, 1:]``;
  the loss is the MSE between predicted and target tokens over every non-action channel (official ``z_loss``).
- Actions: the per-frame actions of one frame-skip step are concatenated (``TrajSlicerDataset`` "concat"); the NovPhy
  launch is ``eb.launch5`` at frame 0 and the zero no-op on every other frame, so step 0 = [launch5, 0 x 20], every
  later step = 0 (25-d).
- Rollout: the official ``VWorldModel.rollout`` (history window of the last ``num_hist`` frames, predicted action
  channels replaced by the next action's embedding); from the single anchor frame 45 steps reach frame 225.
- Optimizers: AdamW(predictor, 5e-4) and AdamW(action_encoder + proprio_encoder, 5e-4); no scheduler, no clipping.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .vendor.models.proprio import ProprioceptiveEmbedding
from .vendor.models.vit import ViTPredictor

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT / "data/external/dino_wm"
DINOV2_REPO = EXTERNAL / "torch_hub/hub/facebookresearch_dinov2_main"  # facebookresearch/dinov2 @ 7764ea0
DINOV2_WEIGHTS = EXTERNAL / "dinov2_vits14_pretrain.pth"
DINOV2_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_NAME = "dinov2_vits14"

# frames and features
FRAME_HW = (240, 320)
PATCH = 14
DINO_HW = (238, 322)  # aspect-preserving resize of the full 320 x 240 frame to the nearest multiples of 14
GRID = (DINO_HW[0] // PATCH, DINO_HW[1] // PATCH)  # (17, 23)
NUM_PATCHES = GRID[0] * GRID[1]  # 391
FEATURE_DIM = 384
OFFICIAL_IMG = 224  # the official configuration: 224 x 224 dataset images, encoder resize to 196 (14 x 14 patches)
OFFICIAL_ENCODER_HW = ((OFFICIAL_IMG // 16) * PATCH,) * 2

# schedule
FRAMESKIP = 5
HORIZON_FRAME = 225
STEPS = HORIZON_FRAME // FRAMESKIP  # 45
NUM_HIST = 3
NUM_PRED = 1
LAUNCH_DIM = 5
ACTION_DIM = LAUNCH_DIM * FRAMESKIP  # 25
ACTION_EMB_DIM = 10
PROPRIO_EMB_DIM = 10
PREDICTOR = {"depth": 6, "heads": 16, "mlp_dim": 2048, "dropout": 0.1, "emb_dropout": 0.0, "pool": "mean"}
LR = {"predictor": 5e-4, "action_encoder": 5e-4}

# E99 carriers
CARRIER_DIM = 236
CARRIER_HEADER = 2
SLOTS = 18
SLOT_WIDTH = 13
SLOT_TOKEN_DIM = SLOT_WIDTH + CARRIER_HEADER  # 15

ARMS = {
    "C1-dino": {"input": "dinov2", "num_patches": NUM_PATCHES, "visual_dim": FEATURE_DIM},
    "C1-car": {"input": "carriers", "num_patches": SLOTS, "visual_dim": SLOT_TOKEN_DIM},
}


# ---------------------------------------------------------------------------
# DINOv2 features (frozen encoder; DINO-WM's own Normalize(0.5, 0.5) and torchvision-equivalent bilinear resize)
# ---------------------------------------------------------------------------

def load_dinov2(device="cpu"):
    """dinov2_vits14 from the pinned hub clone with the official weights (what hubconf pretrained=True loads)."""
    model = torch.hub.load(str(DINOV2_REPO), DINOV2_NAME, source="local", pretrained=False)
    state = torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


def resize(x, size):
    """torchvision ``transforms.Resize(size)`` on float tensors (bilinear, antialias, align_corners False)."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def preprocess(frames, size=DINO_HW):
    """uint8 [N, 3, 240, 320] -> float [N, 3, *size]: /255, Normalize(0.5, 0.5) (datasets/img_transforms.py), then
    the full frame resized aspect-preserving to multiples of 14 (decision P-other on #130; no centre crop)."""
    x = frames.float() / 255.0
    x = (x - 0.5) / 0.5
    return resize(x, size)


@torch.no_grad()
def patch_features(encoder, images):
    """DinoV2Encoder.forward with feature_key x_norm_patchtokens: [N, P, 384]."""
    return encoder.forward_features(images)["x_norm_patchtokens"]


def official_encoder_input(images_224):
    """VWorldModel.encoder_transform of the official configuration: Resize(196) of the 224 x 224 dataset images."""
    return resize(images_224, OFFICIAL_ENCODER_HW)


# ---------------------------------------------------------------------------
# launch / no-op schedule (native frame-skip action concatenation)
# ---------------------------------------------------------------------------

def frame_actions(launch5, frames=HORIZON_FRAME):
    """Per-frame actions [N, frames, 5]: the launch at frame 0, the zero no-op at every later frame."""
    launch5 = torch.as_tensor(launch5, dtype=torch.float32)
    out = torch.zeros(launch5.shape[0], frames, LAUNCH_DIM)
    out[:, 0] = launch5
    return out


def step_actions(launch5, steps=STEPS):
    """Frame-skip step actions [N, steps, 25] = rearrange(frame actions, '(h f) d -> h (f d)', f=5)."""
    return rearrange(frame_actions(launch5, steps * FRAMESKIP), "n (h f) d -> n h (f d)", f=FRAMESKIP)


def window_actions(launch5, start_steps, length=NUM_HIST + NUM_PRED):
    """Actions [N, length, 25] of training windows starting at step s (frame 5 s) of each shot."""
    full = step_actions(launch5, STEPS + length)
    idx = torch.as_tensor(start_steps)[:, None] + torch.arange(length)[None]
    return torch.gather(full, 1, idx[..., None].expand(-1, -1, ACTION_DIM))


# ---------------------------------------------------------------------------
# C1-car: carrier <-> slot tokens (18 slot tokens; the 2-d frame header tiled onto every slot, as concat_dim=1
# tiles frame-level vectors onto every patch); columns z-scored with fit-split statistics
# ---------------------------------------------------------------------------

def carrier_stats(rows):
    """Per-column mean / std of fit carriers (std floor as hn.target_stats)."""
    rows = torch.as_tensor(rows, dtype=torch.float64)
    mean, std = rows.mean(0), rows.std(0)
    return mean.float(), torch.where(std > 1e-6, std, torch.ones_like(std)).float()


def carrier_to_tokens(carriers, mean, std):
    z = (carriers - mean) / std
    header = z[..., :CARRIER_HEADER]
    slots = z[..., CARRIER_HEADER:].reshape(*z.shape[:-1], SLOTS, SLOT_WIDTH)
    return torch.cat([slots, header.unsqueeze(-2).expand(*header.shape[:-1], SLOTS, CARRIER_HEADER)], -1)


def tokens_to_carrier(tokens, mean, std):
    slots = tokens[..., :SLOT_WIDTH].reshape(*tokens.shape[:-2], SLOTS * SLOT_WIDTH)
    header = tokens[..., SLOT_WIDTH:].mean(-2)
    return torch.cat([header, slots], -1) * std + mean


# ---------------------------------------------------------------------------
# the world model (port of VWorldModel's predictor path; proprio optional)
# ---------------------------------------------------------------------------

class DinoWM(nn.Module):
    def __init__(self, num_patches, visual_dim, action_dim=ACTION_DIM, proprio_dim=None,
                 action_emb_dim=ACTION_EMB_DIM, proprio_emb_dim=PROPRIO_EMB_DIM, num_hist=NUM_HIST,
                 num_pred=NUM_PRED, predictor=None):
        super().__init__()
        predictor = dict(PREDICTOR if predictor is None else predictor)
        self.num_hist, self.num_pred = num_hist, num_pred
        self.visual_dim = visual_dim
        # official construction order (Trainer.init_models): proprio encoder, action encoder, predictor
        self.proprio_encoder = (ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=proprio_dim,
                                                        emb_dim=proprio_emb_dim, use_3d_pos=False)
                                if proprio_dim else None)
        self.action_encoder = ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=action_dim,
                                                      emb_dim=action_emb_dim, use_3d_pos=False)
        self.proprio_dim = proprio_emb_dim if proprio_dim else 0
        self.action_dim = action_emb_dim
        self.predictor = ViTPredictor(num_patches=num_patches, num_frames=num_hist,
                                      dim=visual_dim + self.proprio_dim + self.action_dim, **predictor)

    # VWorldModel.encode (concat_dim == 1) on precomputed visual tokens
    def encode(self, visual, act, proprio=None):
        """visual [b, t, P, Dv], act [b, t, action_dim], proprio [b, t, proprio_dim] -> z [b, t, P, Dv + emb]."""
        parts = [visual]
        if self.proprio_encoder is not None:
            proprio_emb = self.proprio_encoder(proprio)
            parts.append(repeat(proprio_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual.shape[2]))
        act_emb = self.action_encoder(act)
        parts.append(repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual.shape[2]))
        return torch.cat(parts, dim=3)

    def predict(self, z):
        t = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = self.predictor(z)
        return rearrange(z, "b (t p) d -> b t p d", t=t)

    def separate(self, z):
        """{"visual", "proprio"} without the action channels (VWorldModel.separate_emb, concat_dim 1)."""
        z_visual = z[..., :self.visual_dim]
        out = {"visual": z_visual}
        if self.proprio_encoder is not None:
            out["proprio"] = z[:, :, 0, self.visual_dim:self.visual_dim + self.proprio_dim]
        return out

    def forward(self, visual, act, proprio=None):
        """VWorldModel.forward's predictor branch: (z_pred, loss, components)."""
        z = self.encode(visual, act, proprio)
        z_src = z[:, :self.num_hist]
        z_tgt = z[:, self.num_pred:]
        z_pred = self.predict(z_src)
        a = self.action_dim
        z_loss = F.mse_loss(z_pred[..., :-a], z_tgt[..., :-a].detach())
        components = {"z_loss": z_loss,
                      "z_visual_loss": F.mse_loss(z_pred[..., :self.visual_dim], z_tgt[..., :self.visual_dim].detach())}
        if self.proprio_encoder is not None:
            p = slice(self.visual_dim, self.visual_dim + self.proprio_dim)
            components["z_proprio_loss"] = F.mse_loss(z_pred[..., p], z_tgt[..., p].detach())
        components["loss"] = z_loss
        return z_pred, z_loss, components

    def replace_actions(self, z, act):
        act_emb = self.action_encoder(act)
        z[..., -self.action_dim:] = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
        return z

    def rollout(self, visual0, act, proprio0=None):
        """VWorldModel.rollout: visual0 [b, n, P, Dv] (n context frames), act [b, n + t, action_dim] ->
        z [b, n + t + 1, P, D] (frames 0 .. n + t)."""
        n = visual0.shape[1]
        z = self.encode(visual0, act[:, :n], proprio0)
        action = act[:, n:]
        t = 0
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist:])
            z_new = self.replace_actions(z_pred[:, -1:, ...], action[:, t:t + 1, :])
            z = torch.cat([z, z_new], dim=1)
            t += 1
        z_pred = self.predict(z[:, -self.num_hist:])
        return torch.cat([z, z_pred[:, -1:, ...]], dim=1)


def build(arm, predictor=None):
    spec = ARMS[arm]
    return DinoWM(spec["num_patches"], spec["visual_dim"], predictor=predictor)


def official_pusht_model():
    """The official PushT configuration (released hydra.yaml): 196 patches x 384, proprio 4, action 2 x 5."""
    return DinoWM(num_patches=196, visual_dim=FEATURE_DIM, action_dim=10, proprio_dim=4)


def make_optimizers(model):
    """Trainer.init_optimizers (has_predictor): AdamW(predictor), AdamW(action_encoder + proprio_encoder)."""
    encoders = [model.action_encoder.parameters()]
    if model.proprio_encoder is not None:
        encoders.append(model.proprio_encoder.parameters())
    return {"predictor": torch.optim.AdamW(model.predictor.parameters(), lr=LR["predictor"]),
            "action_encoder": torch.optim.AdamW(itertools.chain(*encoders), lr=LR["action_encoder"])}


def train_step(model, optimizers, visual, act, proprio=None):
    """Trainer.train's step: forward, zero_grad, backward, predictor then action-encoder step."""
    model.train()
    z_pred, loss, components = model(visual, act, proprio)
    for opt in optimizers.values():
        opt.zero_grad()
    loss.backward()
    optimizers["predictor"].step()
    optimizers["action_encoder"].step()
    return z_pred, loss, components


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# R1 readout (C1-dino): DINOv2 patch features -> E99 carrier; slot-query cross-attention over the patch grid with a
# learned 2-D position embedding, so a pig that covers a fraction of one 14 px patch can be attended to and located
# ---------------------------------------------------------------------------

R1 = {"heads": 8, "hidden": 512, "lr": 1e-3, "weight_decay": 1e-4, "batch": 128, "epochs": 25}


class PatchReadout(nn.Module):
    def __init__(self, num_patches=NUM_PATCHES, dim=FEATURE_DIM, mean=None, std=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.queries = nn.Parameter(torch.randn(1 + SLOTS, dim) * 0.02)  # header query + one query per slot
        self.attn = nn.MultiheadAttention(dim, R1["heads"], batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, R1["hidden"]), nn.GELU(), nn.Linear(R1["hidden"], dim))
        self.head_header = nn.Linear(dim, CARRIER_HEADER)
        self.head_slot = nn.Linear(dim, SLOT_WIDTH)
        self.register_buffer("mean", torch.zeros(CARRIER_DIM) if mean is None else mean.clone().float())
        self.register_buffer("std", torch.ones(CARRIER_DIM) if std is None else std.clone().float())

    def forward_z(self, x):
        """z-scored carrier [B, 236] of patch features x [B, P, D]."""
        k = self.norm(x.float()) + self.pos
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = q + self.attn(q, k, k, need_weights=False)[0]
        h = h + self.mlp(h)
        return torch.cat([self.head_header(h[:, 0]), self.head_slot(h[:, 1:]).flatten(1)], 1)

    def forward(self, x):
        return self.forward_z(x) * self.std + self.mean


def fit_readout(features, rows, targets, seed, device="cuda", epochs=None, log=None):
    """Fit PatchReadout on features[rows] -> targets (MSE on z-scored targets; AdamW; #123 probe schedule).
    features: an indexable [N, P, D] store (fp16 memmap); rows: int64 [M] row indices; targets [M, 236]."""
    import numpy as np
    epochs = R1["epochs"] if epochs is None else epochs
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mean, std = carrier_stats(targets)
    probe = PatchReadout(features.shape[1], features.shape[2], mean, std).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=R1["lr"], weight_decay=R1["weight_decay"])
    yz = ((torch.as_tensor(targets).float() - mean) / std)
    rows = np.asarray(rows, dtype=np.int64)
    n, b = len(rows), R1["batch"]
    steps, last = 0, None

    def load(idx):
        pick = rows[idx]
        sort = np.argsort(pick)  # sorted reads from the memmap; the batch is the same set
        return torch.from_numpy(np.ascontiguousarray(features[pick[sort]])), torch.from_numpy(idx[sort])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        for epoch in range(epochs):
            order = torch.randperm(n, generator=generator).numpy()
            batches = [order[i:i + b] for i in range(0, n - b + 1, b)]
            pending = [pool.submit(load, batches[0])] if batches else []
            for k in range(len(batches)):
                if k + 1 < len(batches):
                    pending.append(pool.submit(load, batches[k + 1]))
                x, idx = pending.pop(0).result()
                loss = F.mse_loss(probe.forward_z(x.to(device).float()), yz[idx].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                last = float(loss.detach())
            if log is not None:
                log(f"R1 epoch {epoch + 1}/{epochs}: loss {last:.5f}")
    probe.eval()
    return probe, {"steps": steps, "final_loss": last, "samples": n, "epochs": epochs, **{k: R1[k] for k in R1}}


@torch.no_grad()
def apply_readout(probe, features, device="cuda", chunk=256):
    out = []
    for i in range(0, len(features), chunk):
        out.append(probe(torch.as_tensor(features[i:i + chunk]).to(device).float()).cpu())
    return torch.cat(out) if out else torch.empty(0, CARRIER_DIM)
