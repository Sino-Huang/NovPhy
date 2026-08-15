"""EMA / stop-gradient target encoder.

The target branch is a detached exponential-moving-average copy of the online
encoder.  It never receives a gradient and never appears in an optimizer's
parameter list; ``JepaBackbone.trainable_parameters`` is the single place that
decides what is optimized.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from world_model.data.types import ContractValueError


class EmaTargetEncoder(nn.Module):
    """A no-grad EMA copy of an online module, with a cosine momentum ramp."""

    def __init__(
        self,
        online: nn.Module,
        *,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(online, nn.Module):
            raise ContractValueError("online module", "must be a torch module")
        _require_unit_momentum(base_momentum, "base_momentum")
        _require_unit_momentum(final_momentum, "final_momentum")
        if base_momentum > final_momentum:
            raise ContractValueError("base_momentum", "must not exceed final_momentum")

        self.module = copy.deepcopy(online)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self._base_momentum = float(base_momentum)
        self._final_momentum = float(final_momentum)

    @property
    def base_momentum(self) -> float:
        return self._base_momentum

    @property
    def final_momentum(self) -> float:
        return self._final_momentum

    def momentum_at(self, step: int, total_steps: int) -> float:
        """Return the cosine-ramped momentum for a step, clamped to the horizon."""
        _require_nonnegative_step(step, "step")
        if type(total_steps) is not int or total_steps <= 0:
            raise ContractValueError("total_steps", "must be a positive integer")
        progress = min(step, total_steps) / total_steps
        span = self._final_momentum - self._base_momentum
        return self._final_momentum - span * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        """Blend the target toward ``online`` in place: ``t <- m*t + (1-m)*o``."""
        if not isinstance(online, nn.Module):
            raise ContractValueError("online module", "must be a torch module")
        _require_unit_momentum(momentum, "momentum")
        blend = float(momentum)

        online_parameters = dict(online.named_parameters())
        for name, parameter in self.module.named_parameters():
            source = online_parameters.get(name)
            if source is None:
                raise ContractValueError("ema parameter", f"{name!r} is absent from the online module")
            if parameter.is_floating_point():
                parameter.mul_(blend).add_(source.detach(), alpha=1.0 - blend)
            else:
                # Integral parameters have no meaningful interpolation.
                parameter.copy_(source.detach())

        online_buffers = dict(online.named_buffers())
        for name, buffer in self.module.named_buffers():
            source_buffer = online_buffers.get(name)
            if source_buffer is None:
                raise ContractValueError("ema buffer", f"{name!r} is absent from the online module")
            buffer.copy_(source_buffer.detach())

    def forward(self, *args: object, **kwargs: object) -> object:
        """Run the target module with gradients disabled (stop-gradient)."""
        with torch.no_grad():
            return self.module(*args, **kwargs)


def _require_unit_momentum(value: float, field: str) -> None:
    if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
        raise ContractValueError(field, "must lie in the closed unit interval")


def _require_nonnegative_step(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ContractValueError(field, "must be a nonnegative integer")
