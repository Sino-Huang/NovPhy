"""Issue #129 (B1 SlotFormer-E99): the vendored official SlotFormer slot Transformer on E99 carriers.

The dynamics model, its forward pass and its loss are the official ``SingleStepSlotFormer`` (SlotFormer's PHYRE
planning model: burn-in 1, iterative overlapping over a ``cond_len`` window), used unchanged through
``vendor/single_step_slotformer.py`` and ``vendor/slotformer.py``. The adapter only fixes the data plumbing
(Decision P-other on #129):

- slots: the 18 E99 vocabulary slots x 13 ``ENTITY_FEATURES`` of the 236-d carrier, raw (no z-scoring), every
  slot always present (absent objects are slots with presence ~0, as SAVi's empty slots);
- the 2 header columns (has-prior flag, elapsed seconds) are parse-schedule metadata, not objects: dropped from the
  tokens; a predicted carrier gets the header every non-anchor fit carrier has, ``ROLLED_HEADER``;
- the launch as one extra token at t0, ``[launch5, 0 x 8]`` in slot space (the intervention injected into the
  slot set before the rollout, as SlotFormer's PHYRE planning inserts the action object); its rolled values are
  fed back like every slot, and its target at every step is the same token (the injected object persists);
- frames 0, s, 2s, ..., 225 (stride s = 15: 15 rollout steps; s = 5: 45); a shot whose last position precedes
  225 is padded with its last real strided frame and its loss truncated by ``vid_len`` (the official PHYRE path:
  ``fix_video_len`` + ``SlotFormer.calc_train_loss``);
- loss: the official slot reconstruction MSE (``slot_recon_loss`` x ``slot_recon_loss_w`` = 1); no image loss;
- optimizer: the official ``SlotBaseMethod._configure_optimizers`` recipe (Adam, weight decay 0, no clipping,
  ``CosineAnnealingWarmupRestarts`` from lr / 100 with 5 % linear warm-up, stepped every optimizer step).
"""
from __future__ import annotations

import torch
from torch import nn

from .vendor.nerv_lr import CosineAnnealingWarmupRestarts
from .vendor.single_step_slotformer import SingleStepSlotFormer

CARRIER_DIM = 236
CARRIER_HEADER = 2
CARRIER_SLOTS = 18
SLOT_WIDTH = 13
LAUNCH_DIM = 5
TOKENS = CARRIER_SLOTS + 1  # 18 carrier slots + the launch token
LAUNCH_TOKEN = CARRIER_SLOTS
ENDPOINT = 225
STRIDES = (15, 5)
# header of every non-anchor fit carrier: has-prior flag 1, one frame (0.02 s) since the prior parse
ROLLED_HEADER = (1.0, 0.02)

# SlotFormer's PHYRE configuration (slotformer_phyre_params-fold0.py): burn-in 1, cond_len 6, sine temporal P.E.,
# no slot P.E., pre-norm Transformer encoder; the native size is its 8 x 256 encoder (ffn 4 d, 8 heads)
COND_LEN = 6
SIZES = {
    "native": {"d_model": 256, "num_layers": 8, "num_heads": 8, "ffn_dim": 1024},
    # U's B1 parameter budget (1.84 M +- 5 %; U = 1 837 690): the same depth, heads and ffn ratio, d_model 136
    "matched": {"d_model": 136, "num_layers": 8, "num_heads": 8, "ffn_dim": 544},
}
U_PARAMETERS = 1_837_690
MATCHED_TOLERANCE = 0.05

# official optimizer settings (slotformer_phyre_params-fold0.py; nerv BaseParams): Adam, lr 2e-4, 5 % warm-up,
# cosine to lr / 100, weight decay 0, no gradient clipping, batch 64
LR = 2e-4
WARMUP_STEPS_PCT = 0.05
MIN_LR_DIVISOR = 100.  # min_lr = lr / 100. (the official expression, bit-identical)
BATCH = 64
LOSS_WEIGHTS = {"slot_recon_loss": 1.0}


def positions(stride: int, endpoint: int = ENDPOINT) -> list[int]:
    """The frames a sequence holds: the anchor (0), then every ``stride``-th frame up to ``endpoint``."""
    if endpoint % stride:
        raise ValueError(f"stride {stride} does not divide the endpoint {endpoint}")
    return list(range(0, endpoint + 1, stride))


def rollout_steps(stride: int) -> int:
    return len(positions(stride)) - 1


def launch_token(launch5: torch.Tensor) -> torch.Tensor:
    """[..., 5] launch vector -> [..., 13] slot-space token (launch values, then zeros)."""
    if launch5.shape[-1] != LAUNCH_DIM:
        raise ValueError(f"launch vector has {launch5.shape[-1]} values, expected {LAUNCH_DIM}")
    pad = launch5.new_zeros(*launch5.shape[:-1], SLOT_WIDTH - LAUNCH_DIM)
    return torch.cat([launch5, pad], dim=-1)


def carrier_slots(carriers: torch.Tensor) -> torch.Tensor:
    """[..., 236] carriers -> [..., 18, 13] slots (the header columns dropped)."""
    if carriers.shape[-1] != CARRIER_DIM:
        raise ValueError(f"carrier has {carriers.shape[-1]} columns, expected {CARRIER_DIM}")
    return carriers[..., CARRIER_HEADER:].reshape(*carriers.shape[:-1], CARRIER_SLOTS, SLOT_WIDTH)


def pack_slots(carriers: torch.Tensor, launch5: torch.Tensor) -> torch.Tensor:
    """Carriers [B, T, 236] and launches [B, 5] -> SlotFormer slots [B, T, 19, 13]: the 18 carrier slots and the
    launch token, which is the same at every frame (the t0 input and the target of every rolled step)."""
    slots = carrier_slots(carriers)
    token = launch_token(launch5)[:, None, None, :].expand(-1, slots.shape[1], 1, -1)
    return torch.cat([slots, token.to(slots.dtype)], dim=2)


def unpack_carriers(slots: torch.Tensor) -> torch.Tensor:
    """Predicted slots [..., 19, 13] -> carriers [..., 236] (``ROLLED_HEADER``, then the 18 carrier slots)."""
    if slots.shape[-2:] != (TOKENS, SLOT_WIDTH):
        raise ValueError(f"slots have shape {tuple(slots.shape[-2:])}, expected {(TOKENS, SLOT_WIDTH)}")
    header = slots.new_tensor(ROLLED_HEADER).expand(*slots.shape[:-2], CARRIER_HEADER)
    return torch.cat([header, slots[..., :CARRIER_SLOTS, :].flatten(-2)], dim=-1)


def sequence_plan(ends, stride: int, endpoint: int = ENDPOINT):
    """Per shot: the carrier positions to read (``positions(stride)``, those past the shot's last position replaced
    by its last real strided frame, as PHYRE's ``fix_video_len``) and ``vid_len``, the number of real frames."""
    grid = torch.tensor(positions(stride, endpoint))
    ends = torch.as_tensor(ends, dtype=torch.long).reshape(-1)
    last = (torch.clamp(ends, max=endpoint) // stride) * stride
    read = torch.minimum(grid[None, :], last[:, None])
    vid_len = torch.clamp(ends // stride + 1, max=len(grid))
    return read, vid_len


def training_batch(carrier_rows, shots, ends, launch5: torch.Tensor, stride: int) -> dict:
    """The official ``data_dict`` of fit shots: ``slots`` [B, T, 19, 13] and ``vid_len`` [B].

    ``carrier_rows(shots, positions)`` is ``rc.FitData.carrier_rows`` (flat shot / position index arrays)."""
    read, vid_len = sequence_plan(ends, stride)
    shots = torch.as_tensor(shots, dtype=torch.long).reshape(-1)
    flat_shots = shots[:, None].expand_as(read).reshape(-1)
    rows = carrier_rows(flat_shots.numpy(), read.reshape(-1).numpy()).reshape(*read.shape, CARRIER_DIM)
    return {"slots": pack_slots(rows.float(), launch5.float()), "vid_len": vid_len}


class SlotFormerE99(SingleStepSlotFormer):
    """The official ``SingleStepSlotFormer`` without the SAVi pixel decoder (the E99 slots are the state)."""

    def __init__(self, size: str | dict = "native", stride: int = 15, num_slots: int = TOKENS,
                 slot_size: int = SLOT_WIDTH, cond_len: int = COND_LEN, rollout_len: int | None = None):
        shape = SIZES[size] if isinstance(size, str) else dict(size)
        self.size_name = size if isinstance(size, str) else "custom"
        self.stride = stride
        rollout_dict = dict(num_slots=num_slots, slot_size=slot_size, history_len=1, cond_len=cond_len,
                            t_pe="sin", slots_pe="", norm_first=True, **shape)
        super().__init__(
            resolution=(1, 1), clip_len=1,
            slot_dict=dict(num_slots=num_slots, slot_size=slot_size),
            dec_dict={},
            rollout_dict=rollout_dict,
            loss_dict=dict(rollout_len=rollout_steps(stride) if rollout_len is None else rollout_len,
                           use_img_recon_loss=False))

    def _build_decoder(self):
        """No pixel decoder: the official one only serves the image loss and visualisation."""

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def total_loss(self, data_dict: dict) -> tuple[torch.Tensor, dict]:
        """nerv ``BaseMethod._loss_function``: the weighted sum of the ``*_loss`` terms of ``loss_function``."""
        out = self.loss_function(data_dict)
        loss = sum(value * LOSS_WEIGHTS[name] for name, value in out.items() if name.endswith("_loss"))
        return loss, out

    def predict_slots(self, anchor_carriers: torch.Tensor, launch5: torch.Tensor,
                      steps: int | None = None) -> torch.Tensor:
        """Anchor carriers [B, 236] + launches [B, 5] -> rolled slots [B, steps, 19, 13]."""
        first = pack_slots(anchor_carriers[:, None, :], launch5)
        return self.rollout(first, self.rollout_len if steps is None else steps)

    def predict_carriers(self, anchor_carriers: torch.Tensor, launch5: torch.Tensor,
                         steps: int | None = None) -> torch.Tensor:
        """Predicted carriers [B, steps, 236] at frames stride, 2 stride, ..., steps x stride."""
        return unpack_carriers(self.predict_slots(anchor_carriers, launch5, steps))

    def predict_endpoint(self, anchor_carriers: torch.Tensor, launch5: torch.Tensor) -> torch.Tensor:
        """The predicted frame-225 carrier [B, 236] (``EndpointCosts`` applies directly)."""
        return self.predict_carriers(anchor_carriers, launch5)[:, -1]


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model: nn.Module, total_steps: int, lr: float = LR, warmup_pct: float = WARMUP_STEPS_PCT):
    """The official optimizer and per-step scheduler (``SlotBaseMethod._configure_optimizers`` with nerv
    ``BaseMethod._configure_optimizers``, ``weight_decay`` 0): Adam on the trainable parameters and
    ``CosineAnnealingWarmupRestarts(total_steps, max_lr=lr, min_lr=lr / 100, warmup_steps=warmup_pct total_steps)``."""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, total_steps, max_lr=lr, min_lr=lr / MIN_LR_DIVISOR,
                                              warmup_steps=warmup_pct * total_steps)
    return optimizer, scheduler


# ---------------------------------------------------------------------------------------------------------------
# native success head: SlotFormer's PHYRE readout (vendor/phyre_readout.py, readout_phyre_params-fold0.py recipe)
# ---------------------------------------------------------------------------------------------------------------

# the official readout reads the slots at sel_slots = [0, 3] (initial + a rolled frame); here the two frames are the
# anchor and the predicted frame-225 state (#127), both packed with the launch token. The official d_model equals its
# slot size (128); 8 heads cannot split our width 13, so the head keeps the official absolute width 128 (in_proj
# 13 -> 128), 4 layers, 8 heads, ffn 512 (Decision P-other on #129)
HEAD_DICT = dict(num_slots=TOKENS, slot_size=SLOT_WIDTH, t_pe="sin", d_model=128, num_layers=4, num_heads=8,
                 ffn_dim=512, norm_first=True, sel_slots=[0, 1])
HEAD_LR = 1e-3
HEAD_WARMUP_PCT = 0.1
HEAD_BATCH = 256
HEAD_EPOCHS = 50
HEAD_LOSS_WEIGHTS = {"vqa_loss": 1.0}


def build_head(readout_dict: dict | None = None) -> nn.Module:
    from .vendor.phyre_readout import PHYREReadout
    return PHYREReadout(readout_dict=dict(HEAD_DICT if readout_dict is None else readout_dict))


def head_input(anchor_carriers: torch.Tensor, end_carriers: torch.Tensor, launch5: torch.Tensor) -> torch.Tensor:
    """[B, 236] anchors, [B, 236] (predicted) frame-225 carriers, [B, 5] launches -> head slots [B, 2, 19, 13]."""
    return pack_slots(torch.stack([anchor_carriers, end_carriers], dim=1), launch5)


def head_total_loss(head: nn.Module, data_dict: dict) -> tuple[torch.Tensor, dict]:
    """nerv ``_loss_function`` over the readout's ``calc_train_loss`` (``vqa_loss`` x ``vqa_loss_w`` = 1)."""
    out = head.loss_function(data_dict)
    loss = sum(value * HEAD_LOSS_WEIGHTS[name] for name, value in out.items() if name.endswith("_loss"))
    return loss, out
