"""Frozen segmentation and non-monotonicity estimands for issue #83.

Post-#79 probe (analysis-scale: frozen checkpoints are re-loaded, never
retrained). The library imports the frozen #79 representation/estimand module
``world_model.training.clevrer_boundary`` and adds:

- Contact-activity segmentation of any frame timeline from a scene's frozen
  collision timeline (Phase-0 data, frozen before any error is computed): a
  frame is ``pre_first_collision`` before the scene's first collision event,
  ``active_cascade`` from the first through the last event (objects in transit
  between contact events remain inside the active cascade), and
  ``post_settlement`` after the scene's last event.
- Batched recursive per-frame error traces in two availability variants: the
  frozen ``inside_camera_view`` mask (identical semantics to #79's
  ``field_errors``) and an ``all_frames`` variant that scores every declared
  object against the raw annotation values without a visibility mask.
- Descriptive statistics: the post-settlement vs pre/active error-ratio
  contrast with a frozen uniformity band (E2) and the strict monotonicity
  predicate over an endpoint grid (E3).

No new training, no new captures; #79's published verdicts are inputs, never
recomputed here.
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.training import clevrer_boundary as clevrer
from world_model.training.cnn_hybrid import DIM, SLOTS

WINDOW_TRACE_LAST = clevrer.WINDOW_LENGTH - 1     # last window-relative frame with truth (60)
GRID_TRACE_LAST = 120                             # recursive span of the endpoint-grid traces
GRID_ENDPOINTS = (15, 30, 60, 90, 120)            # finer sensitivity grid (frozen)
BROAD_ENDPOINTS = clevrer.ENDPOINTS               # (15, 30, 60, 120) - #79 frozen anchor points
HORIZONS = clevrer.HORIZONS                       # (1, 5, 15)
SEGMENTS = ("pre_first_collision", "active_cascade", "post_settlement")
RATIO_UNIFORM_LOWER = 0.80                        # frozen E2 uniformity band on post/pre+active
RATIO_UNIFORM_UPPER = 1.25
MIN_STRATUM_UNITS = 8                             # frozen precision minimum per stratum
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_RNG = 7201
MIN_VALID_DRAWS = 9000


class ClevrerNonmonotonicityError(ValueError):
    """Typed blocker for segmentation, trace, or estimand contract violations."""


# ---------------------------------------------------------------- segmentation


def cascade_bounds(collision_frames: torch.Tensor) -> tuple[int, int]:
    """(first, last) collision frame of one scene's frozen timeline."""
    frames = [int(f) for f in collision_frames.tolist()]
    if not frames:
        raise ClevrerNonmonotonicityError("segmentation requires at least one frozen collision event")
    for frame in frames:
        if not 0 <= frame < clevrer.FRAMES_PER_CLIP:
            raise ClevrerNonmonotonicityError(
                f"collision frame {frame} outside the {clevrer.FRAMES_PER_CLIP}-frame clip")
    return min(frames), max(frames)


def segment_of_frame(bounds: tuple[int, int], frame: int) -> str:
    """Contact-activity segment of one absolute frame from the frozen cascade bounds."""
    first, last = bounds
    if frame < first:
        return "pre_first_collision"
    if frame <= last:
        return "active_cascade"
    return "post_settlement"


def segment_timeline(bounds: tuple[int, int], start: int, relative_frames) -> list[str]:
    """Labels for the absolute frames ``start + relative`` of one window/eval timeline."""
    return [segment_of_frame(bounds, start + int(relative)) for relative in relative_frames]


def segment_census(bounds_by_scene: dict[int, tuple[int, int]], units, endpoint: int) -> dict:
    """Outcome-free strata sizes at one endpoint over (scene, start-or-context) units.

    ``units`` are (scene_index, start) pairs whose scored frame is
    ``start + endpoint``; counts are data facts of the frozen collision
    timelines and are published in the plan before any error is computed.
    """
    counts = {name: 0 for name in SEGMENTS}
    for scene_index, start in units:
        bounds = bounds_by_scene[int(scene_index)]
        counts[segment_of_frame(bounds, int(start) + endpoint)] += 1
    return {"endpoint": endpoint, "units": int(sum(counts.values())), **counts}


# ---------------------------------------------------------------- error traces


def batched_masked_errors(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-carrier #79-convention field errors over a batch.

    Row-wise identical semantics to ``clevrer_boundary.field_errors``: position
    and velocity under the target frame's ``inside_camera_view`` mask (columns
    7 and 10), kind under the slot-valid mask, plus carrier and presence MSE
    over the full carrier. Empty masks yield NaN so batch results stay
    rectangular; ``finite`` marks rows whose carriers are finite.
    """
    if prediction.shape != target.shape or prediction.shape[1] != DIM:
        raise ClevrerNonmonotonicityError("batched errors require (N, DIM) prediction/target carriers")
    slots_p = prediction[:, 2:].reshape(-1, SLOTS, clevrer.ENTITY_FEATURES)
    slots_t = target[:, 2:].reshape(-1, SLOTS, clevrer.ENTITY_FEATURES)
    position_mask = slots_t[:, :, 7] > .5
    velocity_mask = slots_t[:, :, 10] > .5
    kind_mask = slots_t[:, :, 12] > .5

    def masked_mean(squared: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        values = torch.where(mask, squared, torch.zeros_like(squared))
        total = values.flatten(1).sum(1)
        denominator = mask.flatten(1).sum(1) * squared.shape[2]
        return torch.where(denominator > 0, total / denominator.clamp(min=1),
                           torch.full_like(total, float("nan")))

    squared_position = (slots_p[:, :, list(clevrer.POSITION_COLUMNS)]
                        - slots_t[:, :, list(clevrer.POSITION_COLUMNS)]).square()
    squared_velocity = (slots_p[:, :, list(clevrer.VELOCITY_COLUMNS)]
                        - slots_t[:, :, list(clevrer.VELOCITY_COLUMNS)]).square()
    squared_kind = (slots_p[:, :, list(clevrer.KIND_COLUMNS)]
                    - slots_t[:, :, list(clevrer.KIND_COLUMNS)]).square()
    return {
        "carrier_mse": (prediction - target).square().mean(1),
        "presence_mse": (slots_p[:, :, 0] - slots_t[:, :, 0]).square().mean(1),
        "position_mse": masked_mean(squared_position, position_mask),
        "velocity_mse": masked_mean(squared_velocity, velocity_mask),
        "kind_mse": masked_mean(squared_kind, kind_mask),
        "position_available": position_mask.flatten(1).sum(1) * len(clevrer.POSITION_COLUMNS),
        "velocity_available": velocity_mask.flatten(1).sum(1) * len(clevrer.VELOCITY_COLUMNS),
        "finite": torch.isfinite(prediction).all(1),
    }


def batched_all_frames_errors(prediction: torch.Tensor, raw_position: torch.Tensor,
                              declared: torch.Tensor, raw_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``all_frames`` sensitivity variant: no visibility mask.

    Mean squared error over the declared objects' position (and velocity)
    columns against the raw annotation values in carrier units, with the fixed
    denominator ``declared_objects x 2`` per frame (frozen definition).
    """
    if prediction.shape[1] != DIM or raw_position.shape[1] != SLOTS or declared.shape[1] != SLOTS:
        raise ClevrerNonmonotonicityError(
            "all-frames errors require (N, DIM) carriers and (N, SLOTS, 2) raw targets")
    slots_p = prediction[:, 2:].reshape(-1, SLOTS, clevrer.ENTITY_FEATURES)
    denominator = declared.sum(1) * 2

    def mean_over_declared(raw: torch.Tensor, columns: tuple[int, ...]) -> torch.Tensor:
        squared = (slots_p[:, :, list(columns)] - raw).square().sum(2)
        total = (squared * declared).sum(1)
        return torch.where(denominator > 0, total / denominator.clamp(min=1),
                           torch.full_like(total, float("nan")))

    return (mean_over_declared(raw_position, clevrer.POSITION_COLUMNS),
            mean_over_declared(raw_velocity, clevrer.VELOCITY_COLUMNS))


@torch.no_grad()
def trace_rollout(model, pair, initial: torch.Tensor, targets: torch.Tensor, action: torch.Tensor,
                  raw_position: torch.Tensor | None = None, raw_velocity: torch.Tensor | None = None,
                  declared: torch.Tensor | None = None) -> dict:
    """Recursive fixed-pair trace over one shared batch of cells; no truth resets.

    ``initial`` is (N, DIM); ``targets`` is (N, K, DIM) holding the true
    carriers at relative frames ``h, 2h, ... K*h`` of each cell. One shared
    initial carrier per cell; each prediction feeds the next transition. When
    the raw annotation tensors (N, K, SLOTS, 2)/(N, K, SLOTS, 2) and the
    (N, SLOTS) declared mask are supplied, the ``all_frames`` variant is
    computed in the same pass. The fixed-pair transition is per-row
    independent, so a nonfinite carrier poisons only its own row (cumulative
    per-row ``finite`` flags; NaN errors from the divergence step on); the
    rollout always returns exactly K rows.
    """
    if len(targets.shape) != 3 or initial.shape[0] != targets.shape[0] or initial.shape[1] != DIM:
        raise ClevrerNonmonotonicityError(
            "trace rollout requires matching (N, DIM) initial carriers and (N, K, DIM) targets")
    if tuple(action.shape) != (initial.shape[0], 5):
        raise ClevrerNonmonotonicityError("trace rollout requires an (N, 5) null action")
    if (raw_position is None) != (raw_velocity is None) or (raw_position is None) != (declared is None):
        raise ClevrerNonmonotonicityError("all-frames variant requires raw positions, velocities and declared mask")
    steps = targets.shape[1]
    carrier, rows = initial, []
    finite = torch.ones(initial.shape[0], dtype=torch.bool, device=initial.device)
    for step in range(steps):
        carrier = model.carrier(carrier, action, pair)
        finite = finite & torch.isfinite(carrier).all(1)
        row = {key: value.clone() for key, value in batched_masked_errors(carrier, targets[:, step]).items()}
        if raw_position is not None:
            position, velocity = batched_all_frames_errors(
                carrier, raw_position[:, step], declared, raw_velocity[:, step])
            row["position_all_frames"], row["velocity_all_frames"] = position, velocity
        row["finite"] = finite.clone()
        rows.append(row)
    return {"rows": rows, "carrier": carrier}


# ---------------------------------------------------------------- statistics


def paired_interval(values) -> dict:
    """Paired bootstrap over paired units; DESCRIPTIVE (issue-72/79 convention)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RNG)
    draws = values[rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    return {"mean": float(values.mean()), "paired_units": int(len(values)),
            "descriptive_95_percent_interval": np.quantile(draws, [.025, .975]).tolist()}


def ratio_statistic(values, is_post) -> dict:
    """Post-settlement vs pre/active mean-error ratio with a paired bootstrap interval.

    ``values`` holds one number per paired unit (mean over seeds); ``is_post``
    is the unit's frozen segment label at the scored frame. Resampling is over
    the joint unit roster (strata labels stay fixed), recomputing both strata
    means and their ratio per draw. Draws with an empty stratum or a zero
    pre/active mean are discarded and counted.
    """
    values = np.asarray(values, dtype=float)
    is_post = np.asarray(is_post, dtype=bool)
    if len(values) != len(is_post):
        raise ClevrerNonmonotonicityError("ratio statistic requires one label per unit value")
    post, pre = values[is_post], values[~is_post]
    result = {"post_units": int(len(post)), "pre_active_units": int(len(pre)),
              "uniform_band": [RATIO_UNIFORM_LOWER, RATIO_UNIFORM_UPPER]}
    if len(post) == 0 or len(pre) == 0:
        result.update({"ratio": None, "post_mean": None, "pre_active_mean": None,
                       "descriptive_95_percent_interval": None, "valid_draws": 0,
                       "interval_reliable": False})
        return result
    post_mean, pre_mean = float(post.mean()), float(pre.mean())
    rng = np.random.default_rng(BOOTSTRAP_RNG)
    size = len(values)
    draws = np.empty(BOOTSTRAP_DRAWS)
    valid = attempts = 0
    while valid < BOOTSTRAP_DRAWS and attempts < BOOTSTRAP_DRAWS * 2:
        attempts += 1
        indices = rng.integers(size, size=size)
        sample = values[indices]
        sample_labels = is_post[indices]
        sample_post, sample_pre = sample[sample_labels], sample[~sample_labels]
        if len(sample_post) == 0 or len(sample_pre) == 0 or sample_pre.mean() <= 0:
            continue
        draws[valid] = sample_post.mean() / sample_pre.mean()
        valid += 1
    reliable = valid >= MIN_VALID_DRAWS
    result.update({"post_mean": post_mean, "pre_active_mean": pre_mean, "ratio": post_mean / pre_mean,
                   "descriptive_95_percent_interval": (np.quantile(draws[:valid], [.025, .975]).tolist()
                                                       if reliable else None),
                   "valid_draws": int(valid), "interval_reliable": bool(reliable)})
    return result


def monotonicity(curve: dict) -> dict:
    """Strict-increase predicate over consecutive grid endpoints (#79 S3 convention)."""
    endpoints = sorted(int(key) for key in curve)
    points = [(endpoint, curve[str(endpoint)] if str(endpoint) in curve else curve[endpoint])
              for endpoint in endpoints]
    if any(value is None for _, value in points):
        return {"monotone_increasing": None, "first_violation": None, "endpoints": endpoints}
    for (first, first_value), (second, second_value) in zip(points[:-1], points[1:]):
        if not second_value > first_value:
            return {"monotone_increasing": False, "first_violation": [first, second], "endpoints": endpoints}
    return {"monotone_increasing": True, "first_violation": None, "endpoints": endpoints}
