"""Controller-free prediction primitives; no dataset access or fitting."""
import torch

from world_model.training.native_history_data import VISUAL_DIM, VOCABULARY
from world_model.training.native_history_model import STATE_DIM

CURVE_TIMES = (750, 1500, 3000, 6000, 11250)
FAILURE_MSE = 1e9


def first_shot_indices(shard, *, times=CURVE_TIMES):
    """Exact first-shot indices, retaining unavailable times as None.

    Local contexts need an independently observed frame one horizon earlier;
    missing frames are never interpolated or borrowed from another shot.
    """
    segment = shard["segment_ranges"][0]
    start, stop = segment["start"], segment["stop"]
    steps = shard["tensors"]["fixed_steps"][start:stop].tolist()
    if not steps or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("first shot requires strictly increasing observed steps")
    lookup = {step - steps[0]: start + i for i, step in enumerate(steps)}
    return {"initial": start, "targets": {t: lookup.get(t) for t in times},
            "local_contexts": {h: {t: lookup.get(t - h) for t in times}
                               for h in (50, 250, 750)}}


@torch.no_grad()
def recursive_curves(model, initial, action, pair, *, times=CURVE_TIMES):
    """Batched uninterrupted imagined paths, without any future observations.

    Failure masks are cumulative per member, including failures between saved
    times. Failed paths are not restarted. Infrastructure exceptions propagate
    to the driver; they are not silently counted as model prediction failures.
    Transition counts include every attempted transition for every batch row.
    """
    if pair not in model.pairs or not times or tuple(sorted(set(times))) != tuple(times):
        raise ValueError("expected a permitted fixed pair and increasing unique times")
    if any(t <= 0 or t % pair.delta for t in times):
        raise ValueError("curve times require exact fixed-pair paths")
    if initial.ndim != 2 or initial.shape[1] != STATE_DIM or action.shape != (len(initial), 5):
        raise ValueError("expected batched native carriers and actions")
    current = initial.clone()
    failed = ~torch.isfinite(current).all(-1) | ~torch.isfinite(action).all(-1)
    curves = {}
    for elapsed in range(pair.delta, times[-1] + 1, pair.delta):
        current = model.carrier(current, action, pair)
        failed = failed | ~torch.isfinite(current).all(-1)
        if elapsed in times:
            curves[elapsed] = {"predicted": current.clone(), "failed": failed.clone(),
                               "transitions_per_member": elapsed // pair.delta}
    return curves


@torch.no_grad()
def observed_context_prediction(model, context, action, pair):
    """One local step from an actual observation exactly one horizon earlier.

    The caller must establish exact observed-context availability. This output
    is diagnostic only and must never be injected into a recursive path.
    """
    return recursive_curves(model, context, action, pair, times=(pair.delta,))[pair.delta]


@torch.no_grad()
def field_metrics(predicted, target, actual_presence, actual_centers, *, failed=None):
    """Per-member metrics; engine labels are evaluation targets, never inputs.

    No-present-object center error is undefined (None), not zero. Nonfinite
    predictions or metric overflow retain a penalty carrier MSE and an explicit
    failure. Invalid targets are a data error and stop scoring.
    """
    size = len(predicted)
    if predicted.shape != (size, STATE_DIM) or target.shape != predicted.shape:
        raise ValueError("expected matching native carrier batches")
    if actual_presence.shape != (size, len(VOCABULARY)) or actual_centers.shape != (size, len(VOCABULARY), 2):
        raise ValueError("engine targets must use the native slot vocabulary")
    present = actual_presence.bool()
    if not (torch.isfinite(target).all() and torch.isfinite(actual_presence).all()
            and torch.isfinite(actual_centers[present]).all()):
        raise ValueError("nonfinite available evaluation target")
    bad = ~torch.isfinite(predicted).all(-1)
    if failed is not None:
        bad = bad | failed
    error = (predicted - target).square()
    visual = predicted[:, :VISUAL_DIM]
    presence = visual[:, 2::13]
    centers = torch.stack((visual[:, 7::13], visual[:, 8::13]), -1)
    center_error = torch.where(present[..., None], (centers - actual_centers).square(), 0.)
    center_denominator = present.sum(-1) * 2
    values = {
        "carrier_mse": error.mean(-1),
        "visual_mse": error[:, :VISUAL_DIM].mean(-1),
        "memory_mse": error[:, VISUAL_DIM:].mean(-1),
        "presence_mse_to_engine": (presence - actual_presence).square().mean(-1),
        "center_mse_to_engine": center_error.sum((1, 2)) / center_denominator.clamp_min(1),
        "carrier_absolute_bound_excess": (predicted.abs() - 2).clamp_min(0).amax(-1),
    }
    for kind in ("pig", "block"):
        slots = [i for i, name in enumerate(VOCABULARY) if name.startswith(kind + ":")]
        values[kind + "_count_absolute_error"] = (
            presence[:, slots].clamp(0, 1).sum(-1) - actual_presence[:, slots].sum(-1)).abs()
    bad = bad | ~torch.stack(tuple(values.values()), -1).isfinite().all(-1)
    columns = {key: value.cpu().tolist() for key, value in values.items()}
    records = []
    for i, is_bad in enumerate(bad.cpu().tolist()):
        if is_bad:
            records.append({"failure": "nonfinite_prediction_or_metric", "carrier_mse": FAILURE_MSE})
        else:
            record = {key: value[i] for key, value in columns.items()}
            if not bool(present[i].any()):
                record["center_mse_to_engine"] = None
            record["failure"] = None
            records.append(record)
    return records
