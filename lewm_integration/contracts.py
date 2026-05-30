from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_COORDINATE_CONVENTION = "wrapper_relative_to_slingshot"
DEFAULT_ACTION_MIN = -100.0
DEFAULT_ACTION_MAX = 100.0


class ContractValidationError(ValueError):
    """Raised when NovPhy/LeWM integration inputs violate the MVP contract."""


class TransitionReason(str, Enum):
    STATIC = "static"
    WON = "won"
    LOST = "lost"
    TIMEOUT = "timeout"
    CRASH = "crash"
    INVALID_ACTION = "invalid_action"


@dataclass(frozen=True)
class ValidatedAction2D:
    value: npt.NDArray[np.float32]
    was_clipped: bool
    coordinate_convention: str = DEFAULT_COORDINATE_CONVENTION


@dataclass(frozen=True)
class ObservationMetadata:
    height: int
    width: int
    channels: int
    dtype: str
    color_mode: str = "rgb"


def _as_numeric_array(action: Any) -> npt.NDArray[np.float32]:
    try:
        arr = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("Action must be numeric and convertible to float32.") from exc
    if arr.ndim == 0:
        raise ContractValidationError("Action must contain exactly two numeric values.")
    flat = arr.reshape(-1)
    if flat.shape != (2,):
        raise ContractValidationError(f"Expected action shape (2,), got {tuple(flat.shape)}.")
    if not np.isfinite(flat).all():
        raise ContractValidationError("Action values must be finite.")
    return flat.astype(np.float32, copy=False)


def validate_action_2d(
    action: Any,
    *,
    min_value: float = DEFAULT_ACTION_MIN,
    max_value: float = DEFAULT_ACTION_MAX,
    coordinate_convention: str = DEFAULT_COORDINATE_CONVENTION,
) -> ValidatedAction2D:
    """Validate and clamp the canonical 2D NovPhy action vector."""

    values = _as_numeric_array(action)
    clipped = np.clip(values, min_value, max_value)
    was_clipped = not np.array_equal(values, clipped)
    return ValidatedAction2D(
        value=clipped.astype(np.float32, copy=False),
        was_clipped=was_clipped,
        coordinate_convention=coordinate_convention,
    )


def validate_pixels_shape(pixels: Any) -> ObservationMetadata:
    arr = np.asarray(pixels)
    if arr.ndim != 3:
        raise ContractValidationError(f"Expected pixel observation shape (H, W, C), got {arr.shape}.")
    height, width, channels = arr.shape
    if height <= 0 or width <= 0:
        raise ContractValidationError("Pixel observations must have positive height and width.")
    if channels not in (1, 3, 4):
        raise ContractValidationError(f"Unsupported channel count {channels}; expected 1, 3, or 4.")
    return ObservationMetadata(
        height=int(height),
        width=int(width),
        channels=int(channels),
        dtype=str(arr.dtype),
        color_mode="rgb" if channels in (3, 4) else "grayscale",
    )


def reason_to_termination(reason: TransitionReason) -> tuple[bool, bool]:
    if reason in (TransitionReason.WON, TransitionReason.LOST, TransitionReason.CRASH, TransitionReason.INVALID_ACTION):
        return True, False
    if reason is TransitionReason.TIMEOUT:
        return False, True
    return False, False


def ensure_transition_reason(value: str | TransitionReason) -> TransitionReason:
    if isinstance(value, TransitionReason):
        return value
    try:
        return TransitionReason(value)
    except ValueError as exc:
        valid = ", ".join(reason.value for reason in TransitionReason)
        raise ContractValidationError(f"Unknown transition reason '{value}'. Valid reasons: {valid}.") from exc
