"""Regroup the native predictor's exact sampled starts, without new exposure."""
from collections import defaultdict

import torch

from scripts.run_issue_76_native_refit import generator, pair_for_update


def sampled_schedule(lengths, seed, pure, updates, batch_size=32):
    """Rows are (training-entry index, local start); zero lengths stay skipped."""
    schedule = {}
    for update in range(updates):
        entry = update % len(lengths)
        if not lengths[entry]:
            continue
        starts = torch.randint(lengths[entry], (batch_size,), generator=generator(seed, update))
        schedule[update] = torch.stack((torch.full_like(starts, entry), starts), -1)
    return schedule


def mixed_schedule(original, seed, pure):
    """Permute only within the exact abstraction/horizon pair, using local RNG."""
    groups = defaultdict(list)
    for update in original:
        groups[pair_for_update(pure, update)].append(update)
    mixed = {}
    for pair_index, updates in enumerate(groups.values()):
        rows = torch.cat([original[u] for u in updates])
        permutation = torch.randperm(len(rows), generator=generator(seed, 100000 + pair_index))
        for update, batch in zip(updates, rows[permutation].split(len(original[updates[0]]))):
            mixed[update] = batch
    return dict(sorted(mixed.items()))


def indexed_batch(shards, carriers, rows, horizon, *, symbolic):
    """Exact native targets for explicit starts, allowing multiple lineages."""
    values = carriers[0].new_zeros(len(rows), 5, carriers[0].shape[-1])
    batch = {"z": values, "available": torch.zeros(len(rows), 5, dtype=torch.bool),
             "action": torch.zeros(len(rows), 5)}
    names = ("relations", "relation_mask", "macros", "macro_mask") if symbolic else ()
    for name in names:
        source = shards[0]["tensors"][name]
        batch[name] = source.new_zeros((len(rows), 5) + source.shape[1:])
    for entry in rows[:, 0].unique().tolist():
        shard, carrier = shards[entry], carriers[entry]
        steps = shard["tensors"]["fixed_steps"].tolist()
        ranges = [(s, {steps[i]: i for i in range(s["start"], s["stop"])})
                  for s in shard["segment_ranges"]]
        for row in (rows[:, 0] == entry).nonzero().flatten().tolist():
            start = int(rows[row, 1])
            segment, lookup = next((s, lookup) for s, lookup in ranges
                                   if s["start"] <= start < s["stop"])
            batch["action"][row] = segment["action"]
            for offset in range(5):
                end = lookup.get(steps[start] + offset * horizon)
                if end is not None:
                    values[row, offset] = carrier[end]
                    batch["available"][row, offset] = True
                    for name in names:
                        batch[name][row, offset] = shard["tensors"][name][end]
    return batch
