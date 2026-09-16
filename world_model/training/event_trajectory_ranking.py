"""Predicted trajectory features and outcome-trained pair imitation."""

import torch

from world_model.training.native_history_fit import ENDPOINT


@torch.no_grad()
def fixed_trajectory(model, initial, action, pair, *, endpoint=ENDPOINT,
                     checkpoints=(0, 3750, 7500, 10500)):
    if endpoint % pair.delta or any(value % pair.delta for value in checkpoints):
        raise ValueError("trajectory checkpoints must align with the fixed horizon")
    current = initial
    states = {0: initial}
    for elapsed in range(pair.delta, endpoint + 1, pair.delta):
        current = model.carrier(current, action, pair)
        if not bool(torch.isfinite(current).all()):
            raise ValueError(f"nonfinite fixed trajectory at native step {elapsed}")
        if elapsed in checkpoints:
            states[elapsed] = current
    return current, torch.stack([states[value] for value in checkpoints], dim=1)


@torch.no_grad()
def adaptive_trajectory(model, selector, initial, action, *, endpoint=ENDPOINT):
    pairs = model.pairs
    horizons = torch.tensor([pair.delta for pair in pairs], device=initial.device)
    current = initial.clone()
    remaining = torch.full((len(initial),), endpoint, device=initial.device, dtype=torch.long)
    counts = torch.zeros(len(initial), len(pairs), device=initial.device, dtype=torch.long)
    for _ in range(endpoint // min(pair.delta for pair in pairs)):
        active = (remaining > 0).nonzero().flatten()
        if not len(active):
            break
        logits = selector(current[active], action[active], remaining[active])
        logits = logits.masked_fill(horizons[None] > remaining[active, None], -torch.inf)
        choices = logits.argmax(1)
        for index, pair in enumerate(pairs):
            rows = active[choices == index]
            if not len(rows):
                continue
            current[rows] = model.carrier(current[rows], action[rows], pair)
            remaining[rows] -= pair.delta
            counts[rows, index] += 1
        if not bool(torch.isfinite(current[active]).all()):
            raise ValueError("nonfinite adaptive trajectory")
    if bool((remaining != 0).any()):
        raise ValueError("adaptive trajectory did not reach the native endpoint")
    return current, counts


@torch.no_grad()
def trajectory_pair_teacher(pair_logits, labels, pair_costs, compute_weight):
    losses = -pair_logits.log_softmax(-1).gather(
        2, labels[:, None, None].expand(-1, pair_logits.shape[1], 1)).squeeze(-1)
    return (losses + compute_weight * pair_costs[None]).argmin(1)
