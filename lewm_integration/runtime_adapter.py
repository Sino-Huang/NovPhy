from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import numpy.typing as npt

from .contracts import TransitionReason, ensure_transition_reason, reason_to_termination, validate_action_2d, validate_pixels_shape
from .fake_runtime import FakeNovPhyRuntime, FakeRuntimeConfig


class RuntimeUnavailableError(RuntimeError):
    """Raised when the local checkout cannot construct a real NovPhy runtime."""


@dataclass(frozen=True)
class AdapterStepResult:
    observation: npt.NDArray[np.uint8]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class NovPhyAdapter:
    def __init__(
        self,
        *,
        runtime: Any | None = None,
        use_fake_runtime: bool = False,
        fake_runtime_config: FakeRuntimeConfig | None = None,
        speed: int = 50,
    ) -> None:
        self.speed = speed
        self._runtime = runtime or self._create_runtime(
            use_fake_runtime=use_fake_runtime,
            fake_runtime_config=fake_runtime_config,
            speed=speed,
        )

    @staticmethod
    def _create_runtime(*, use_fake_runtime: bool, fake_runtime_config: FakeRuntimeConfig | None, speed: int) -> Any:
        if use_fake_runtime:
            return FakeNovPhyRuntime(fake_runtime_config or FakeRuntimeConfig())

        try:
            wrapper_module = importlib.import_module("SBEnvironment.SBEnvironmentWrapper")
            wrapper_class = getattr(wrapper_module, "SBEnvironmentWrapper")
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Real NovPhy runtime is unavailable in this checkout. Expected external SBEnvironmentWrapper sources are missing; "
                "use --use-fake-runtime for smoke tests or provide an importable NovPhy runtime package."
            ) from exc

        return wrapper_class(reward_type="score", speed=speed)

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> tuple[npt.NDArray[np.uint8], dict[str, object]]:
        if hasattr(self._runtime, "reset"):
            result = self._runtime.reset(task_id=task_id, seed=seed) if isinstance(self._runtime, FakeNovPhyRuntime) else self._runtime.reset()
        else:
            raise RuntimeUnavailableError("Configured runtime does not expose reset().")

        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        elif isinstance(result, tuple) and len(result) == 4:
            observation, reward, done, info = result
            info = {**dict(info), "initial_reward": float(reward), "initial_done": bool(done)}
        else:
            raise RuntimeUnavailableError("Unsupported reset() return signature from runtime.")

        obs_array = np.asarray(observation, dtype=np.uint8)
        _ = validate_pixels_shape(obs_array)
        return obs_array, dict(info)

    def step(self, action: Any) -> AdapterStepResult:
        validated = validate_action_2d(action)
        result = self._runtime.step(validated.value.tolist())

        if not isinstance(result, tuple):
            raise RuntimeUnavailableError("Unsupported step() return signature from runtime.")

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            info_dict = dict(info)
        elif len(result) == 4:
            observation, reward, done, info = result
            info_dict = dict(info)
            reason = ensure_transition_reason(info_dict.get("transition_reason", "won" if done else "static"))
            terminated, truncated = reason_to_termination(reason)
            info_dict.setdefault("transition_reason", reason.value)
        else:
            raise RuntimeUnavailableError("Unsupported step() return signature from runtime.")

        observation_array = np.asarray(observation, dtype=np.uint8)
        _ = validate_pixels_shape(observation_array)
        info_dict.setdefault("executed_action", validated.value.astype(np.float32, copy=False).tolist())
        info_dict.setdefault("action_coordinate_convention", validated.coordinate_convention)
        info_dict.setdefault("was_clipped", validated.was_clipped)
        info_dict.setdefault("static_wait_steps", 0)
        return AdapterStepResult(
            observation=observation_array,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info_dict,
        )

    def render_pixels(self) -> npt.NDArray[np.uint8]:
        if hasattr(self._runtime, "render_pixels"):
            pixels = self._runtime.render_pixels()
        elif hasattr(self._runtime, "render"):
            pixels = self._runtime.render()
        else:
            raise RuntimeUnavailableError("Configured runtime does not expose render_pixels() or render().")
        return np.asarray(pixels, dtype=np.uint8)

    def get_symbolic_state(self, optional: bool = True) -> dict[str, object] | None:
        if hasattr(self._runtime, "get_symbolic_state"):
            state = self._runtime.get_symbolic_state(optional=optional)
            return None if state is None else dict(state)
        return None if optional else {}

    def get_score(self) -> float:
        if hasattr(self._runtime, "get_score"):
            return float(self._runtime.get_score())
        return 0.0

    def is_done(self) -> bool:
        if hasattr(self._runtime, "is_done"):
            return bool(self._runtime.is_done())
        return False

    def close(self) -> None:
        if hasattr(self._runtime, "close"):
            self._runtime.close()
