from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .contracts import TransitionReason, reason_to_termination, validate_action_2d, validate_pixels_shape


class RuntimeContractError(RuntimeError):
    """Raised when the fake/runtime adapter contract is used incorrectly."""


@dataclass(frozen=True)
class FakeRuntimeConfig:
    seed: int = 0
    image_shape: tuple[int, int, int] = (64, 64, 3)
    max_steps: int = 2
    terminal_reason: TransitionReason = TransitionReason.WON
    timeout_after_steps: int | None = None


class FakeNovPhyRuntime:
    def __init__(self, config: FakeRuntimeConfig | None = None) -> None:
        self.config: FakeRuntimeConfig = config or FakeRuntimeConfig()
        self._rng: np.random.Generator = np.random.default_rng(self.config.seed)
        self._step_count: int = 0
        self._score: float = 0.0
        self._terminated: bool = False
        self._truncated: bool = False
        self._reset_called: bool = False
        self._obs: npt.NDArray[np.uint8] = self._make_obs(step_index=0)

    def _make_obs(self, step_index: int) -> npt.NDArray[np.uint8]:
        height, width, channels = self.config.image_shape
        base = np.asarray(
            self._rng.integers(0, 32, size=(height, width, channels), dtype=np.uint8),
            dtype=np.uint8,
        )
        obs = np.asarray(base + np.uint8(step_index % 223), dtype=np.uint8)
        _ = validate_pixels_shape(obs)
        return obs

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> tuple[npt.NDArray[np.uint8], dict[str, object]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(self.config.seed)
        self._step_count = 0
        self._score = 0.0
        self._terminated = False
        self._truncated = False
        self._reset_called = True
        self._obs = self._make_obs(step_index=0)
        return self._obs.copy(), {"task_id": task_id, "seed": seed if seed is not None else self.config.seed}

    def step(self, action: Any) -> tuple[npt.NDArray[np.uint8], float, bool, bool, dict[str, str | bool | int | float]]:
        if not self._reset_called:
            raise RuntimeContractError("reset() must be called before step().")
        if self._terminated or self._truncated:
            raise RuntimeContractError("Cannot call step() after the episode has finished.")

        validated = validate_action_2d(action)
        self._step_count += 1
        self._score += 1.0
        reward = float(self._score)

        if self.config.timeout_after_steps is not None and self._step_count >= self.config.timeout_after_steps:
            reason = TransitionReason.TIMEOUT
        elif self._step_count >= self.config.max_steps:
            reason = self.config.terminal_reason
        else:
            reason = TransitionReason.STATIC

        terminated, truncated = reason_to_termination(reason)
        self._terminated = terminated
        self._truncated = truncated
        self._obs = self._make_obs(step_index=self._step_count)

        info = {
            "transition_reason": reason.value,
            "was_clipped": validated.was_clipped,
            "action_coordinate_convention": validated.coordinate_convention,
            "static_wait_steps": 1,
            "score": self._score,
        }
        return self._obs.copy(), reward, terminated, truncated, info

    def render_pixels(self) -> npt.NDArray[np.uint8]:
        return self._obs.copy()

    def get_symbolic_state(self, optional: bool = True) -> dict[str, Any] | None:
        if optional:
            return None
        return {"birds_remaining": max(0, self.config.max_steps - self._step_count), "score": self._score}

    def get_score(self) -> float:
        return self._score

    def is_done(self) -> bool:
        return self._terminated or self._truncated

    def close(self) -> None:
        return None
