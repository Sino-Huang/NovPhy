"""Extend recursive supervision to the native endpoint; keep local/symbolic terms."""
import torch
from torch.nn import functional as F

from scripts.issue_76_matched_batches import indexed_batch
from world_model.model import Abstraction
from world_model.training.native_history_fit import ENDPOINT, symbolic_loss


def trajectory_batch(shards, carriers, rows, horizon, *, symbolic, endpoint=ENDPOINT):
    batch = indexed_batch(shards, carriers, rows, horizon, symbolic=symbolic)
    count = endpoint // horizon + 1
    values = carriers[0].new_zeros(len(rows), count, carriers[0].shape[-1])
    available = torch.zeros(len(rows), count, dtype=torch.bool)
    for entry in rows[:, 0].unique().tolist():
        shard, carrier = shards[entry], carriers[entry]
        steps = shard["tensors"]["fixed_steps"].tolist()
        ranges = [(s, {steps[i]: i for i in range(s["start"], s["stop"])})
                  for s in shard["segment_ranges"]]
        for row in (rows[:, 0] == entry).nonzero().flatten().tolist():
            start = int(rows[row, 1])
            _, lookup = next((s, lookup) for s, lookup in ranges if s["start"] <= start < s["stop"])
            for offset in range(count):
                end = lookup.get(steps[start] + offset * horizon)
                if end is not None:
                    values[row, offset] = carrier[end]
                    available[row, offset] = True
    batch.update(z=values, available=available)
    return batch


def full_continuous_loss(model, z, available, action, pair):
    local_terms, recursive_terms = [], []
    current = z[:, 0]
    for offset in range(1, z.shape[1]):
        mask = available[:, offset - 1] & available[:, offset]
        if not bool(mask.any()):
            break
        current = model.carrier(current, action, pair)
        target = z[:, offset]
        if offset <= 4:
            local = current if offset == 1 else model.carrier(z[:, offset - 1], action, pair)
            local_terms.append(F.mse_loss(local[mask], target[mask]))
        recursive_terms.append(F.mse_loss(current[mask], target[mask])
                               + .01 * F.relu(current[mask].abs() - 2).square().mean())
    if not recursive_terms:
        return None
    return torch.stack(local_terms).mean() + torch.stack(recursive_terms).mean()


def full_hybrid_loss(model, batch, pair):
    loss = full_continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
    if loss is None or pair.abstraction == Abstraction.CONTINUOUS:
        return loss
    name, mask_name = ("relations", "relation_mask") if pair.abstraction == Abstraction.MICRO else ("macros", "macro_mask")
    terms = []
    for offset in range(5):
        valid = batch["available"][:, offset]
        if bool(valid.any()):
            terms.append(symbolic_loss(model, batch["z"][valid, offset], batch[name][valid, offset],
                                       batch[mask_name][valid, offset], pair.abstraction))
    return loss + torch.stack(terms).mean()
