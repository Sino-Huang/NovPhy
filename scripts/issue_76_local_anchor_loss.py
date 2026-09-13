"""Increase only horizon-750 local supervision; preserve full recursive losses."""
import torch
from torch.nn import functional as F

from scripts.issue_76_full_duration_loss import full_continuous_loss, full_hybrid_loss


def local_anchor_loss(model, batch, pair, *, pure):
    loss = (full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
            if pure else full_hybrid_loss(model, batch, pair))
    if loss is None or pair.delta != 750:
        return loss
    local = []
    for offset in range(1, min(5, batch["z"].shape[1])):
        mask = batch["available"][:, offset - 1] & batch["available"][:, offset]
        if not bool(mask.any()):
            break
        prediction = model.carrier(batch["z"][:, offset - 1], batch["action"], pair)
        local.append(F.mse_loss(prediction[mask], batch["z"][mask, offset]))
    return loss + 9 * torch.stack(local).mean()
