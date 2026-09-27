"""Jointly trained JEPA slot encoder for the 236-value deployment carrier (issue #100).

The #77 hybrid reads a frozen, separately supervised slot parser. Here the slot
encoder is trained together with the request-conditioned predictor:

- a context (online) encoder, the ``SpatialSlotParser`` architecture, whose
  outputs become the carrier through ``carrier_from_outputs`` (a differentiable
  twin of ``TemporalVisualCarrierAdapter.build_from_parsed``), so the predictor
  loss reaches the encoder weights;
- a target encoder, an exponential moving average of the context encoder that
  receives no gradient; prediction targets are its carriers (stop-gradient);
- the unchanged ``CNNHybridPredictor`` (FiLM / AdaLN-Zero request conditioning);
- auxiliary losses: the micro and macro heads on context and predicted carriers
  for every request, and the issue-70 parser loss on the context encoder outputs
  against engine-projected targets (keeps the fixed carrier layout readable by
  the cost, the rollout metrics and the presence gate of the micro adapter).

At inference the context encoder is wrapped in the standard adapter, so anchor
carriers, rollouts and costs use exactly the code path of the frozen encoders.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from world_model.data.deployment_temporal import ENTITY_FEATURES, OBJECT_KIND_VOCABULARY
from world_model.model import Abstraction
from world_model.training.cnn_hybrid import DIM, symbolic_loss

KIND_DIVISOR = len(OBJECT_KIND_VOCABULARY) - 1


def expected_kind_indices(vocabulary):
    """OBJECT_KIND_VOCABULARY index of each slot's declared kind (adapter rule)."""
    out = []
    for name in vocabulary:
        kind = name.split(":", 1)[0]
        out.append(OBJECT_KIND_VOCABULARY.index(kind) if kind in OBJECT_KIND_VOCABULARY[:-1]
                   else len(OBJECT_KIND_VOCABULARY) - 1)
    return out


def carrier_from_outputs(output, prior, has_prior, elapsed, expected_index, *,
                         presence_temperature=1.0, kind_temperature=1.0, threshold=0.5):
    """Differentiable ``build_from_parsed`` for a batch.

    output / prior: parser output dicts for the current / prior frames (B rows);
    has_prior (B,) in {0, 1}; elapsed (B,) seconds between prior and current
    (ignored where has_prior is 0); expected_index: per-slot declared-kind index.
    Hard gates (presence >= threshold, kind argmax) are constants of the forward
    pass, exactly as in the adapter; presence, kind probabilities, centers and
    motion carry gradient.
    """
    presence = torch.sigmoid(output["presence_logits"] / presence_temperature)
    kinds = torch.softmax(output["kind_logits"] / kind_temperature, dim=-1)
    count = presence.shape[1]
    if len(ENTITY_FEATURES) * count + 2 > DIM:
        raise ValueError("carrier layout does not fit the slot count")
    kind_index = kinds.argmax(-1)
    kind_confidence = kinds.gather(-1, kind_index[..., None]).squeeze(-1)
    expected = kinds[:, torch.arange(count, device=kinds.device),
                     torch.as_tensor(expected_index, device=kinds.device)]
    centers = output["centers"]
    center_available = (presence >= threshold).to(presence.dtype)
    prior_presence = torch.sigmoid(prior["presence_logits"] / presence_temperature)
    has_prior = has_prior.to(presence.dtype)
    motion_available = has_prior[:, None] * (prior_presence >= threshold).to(presence.dtype) * center_available
    safe_elapsed = torch.where(has_prior > 0, elapsed.to(presence.dtype), torch.ones_like(has_prior))
    motion = (centers - prior["centers"]) / safe_elapsed[:, None, None] * motion_available[..., None]
    ones = torch.ones_like(presence)
    slots = torch.stack((presence, ones, kind_index.to(presence.dtype) / KIND_DIVISOR, kind_confidence, ones,
                         centers[..., 0] * center_available, centers[..., 1] * center_available,
                         center_available, motion[..., 0], motion[..., 1], motion_available, expected, ones), -1)
    world = torch.stack((has_prior, elapsed.to(presence.dtype) * has_prior), -1)
    carrier = torch.cat((world, slots.flatten(1)), -1)
    if carrier.shape[1] < DIM:
        carrier = F.pad(carrier, (0, DIM - carrier.shape[1]))
    return carrier


def encode_carriers(encoder, current, prior, has_prior, elapsed, expected_index):
    """current/prior uint8 images (B, 3, H, W) -> (carriers (B, 236), current outputs)."""
    batch = len(current)
    output = encoder(torch.cat((current, prior)))
    now = {k: v[:batch] for k, v in output.items()}
    before = {k: v[batch:] for k, v in output.items()}
    return carrier_from_outputs(now, before, has_prior, elapsed, expected_index), now


def new_target(encoder):
    target = copy.deepcopy(encoder)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target.eval()


@torch.no_grad()
def ema_update(target, online, momentum):
    for t, o in zip(target.parameters(), online.parameters(), strict=True):
        t.mul_(momentum).add_(o.detach(), alpha=1.0 - momentum)
    for t, o in zip(target.buffers(), online.buffers(), strict=True):
        t.copy_(o)


def momentum_at(step, steps, start=0.996, end=1.0):
    """I-JEPA linear EMA schedule."""
    return start + (end - start) * min(step, steps) / steps


def head_losses(predictor, z, labels, positions):
    """Micro and macro head losses (the #77 symbolic_loss) for every request."""
    return (symbolic_loss(predictor, z, labels, positions, Abstraction.MICRO)
            + symbolic_loss(predictor, z, labels, positions, Abstraction.MACRO))


def jepa_loss(predictor, context, targets, action, length, labels, pair, rows, head_weight=0.1):
    """#77 training_loss structure with stop-gradient EMA targets.

    context[j] (B, 236): online carriers at window rows rows[j] (j = 0..3);
    targets[j] (B, 236): EMA carriers at rows[j + 1] (detached);
    labels: window label tensors (relations/macros and masks, B x 61 x ...);
    rows: the five window rows 0, Delta, ..., 4 Delta.
    """
    losses, recursive, current = [], [], None
    for step in range(4):
        end = rows[step + 1]
        keep = length >= end
        if not bool(keep.any()):
            continue
        subset = {k: v[keep] for k, v in labels.items()}
        target = targets[step].detach()
        local = predictor.carrier(context[step], action, pair)
        at_end = torch.full((int(keep.sum()),), end, device=action.device, dtype=torch.long)
        loss = F.mse_loss(local[keep], target[keep])
        loss = loss + head_weight * head_losses(predictor, local[keep], subset, at_end)
        loss = loss + head_weight * head_losses(predictor, context[step][keep], subset, at_end - pair.delta)
        losses.append(loss)
        current = local if step == 0 else predictor.carrier(current, action, pair)
        recursive.append(F.mse_loss(current[keep], target[keep])
                         + .01 * F.relu(current[keep].abs() - 2).square().mean())
    if not losses:
        raise ValueError("batch has no complete requested-horizon transition")
    return torch.stack(losses).mean() + torch.stack(recursive).mean()


class JEPAWorld(nn.Module):
    """Container for the trainable parts: context encoder + predictor."""

    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
