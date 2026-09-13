"""Add observed endpoint supervision only to horizon-750 transitions."""
from scripts.issue_76_local_anchor_loss import local_anchor_loss


def endpoint_anchor_loss(model, batch, pair, *, pure):
    loss = local_anchor_loss(model, batch, pair, pure=pure)
    if loss is None or pair.delta != 750:
        return loss
    mask = batch["available"][:, -1]
    if not bool(mask.any()):
        return loss
    current = batch["z"][:, 0]
    for _ in range(1, batch["z"].shape[1]):
        current = model.carrier(current, batch["action"], pair)
    return loss + (current[mask] - batch["z"][mask, -1]).square().mean()
