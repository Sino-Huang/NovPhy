"""Assembled JEPA backbone: online encoder, EMA target, and predictor."""
from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from world_model.data.types import ContractValueError
from world_model.model.config import JepaConfig, PredictionPair
from world_model.model.ema import EmaTargetEncoder
from world_model.model.encoder import ContextEncoder, EncoderOutput
from world_model.model.predictor import DualOutputPredictor, PredictorOutput
from world_model.model import encoder as _encoder_module


class JepaBackbone(nn.Module):
    """The Milestone 1a + 1b assembly.

    - ``encoder``     — online context encoder (the only encoder that trains)
    - ``target``      — EMA/stop-grad copy of the encoder
    - ``predictor``   — dual-output ``F_theta^{Delta,alpha}``

    ``trainable_parameters()`` is the single source of truth for what an
    optimizer may touch: online encoder and predictor.  Nothing else.
    """

    def __init__(self, config: JepaConfig) -> None:
        super().__init__()
        if type(config) is not JepaConfig:
            raise ContractValueError("jepa config", "must be a JepaConfig")
        self._config = config

        self.encoder: ContextEncoder = _encoder_module.build_encoder(config.encoder)
        self.target = EmaTargetEncoder(
            self.encoder,
            base_momentum=config.ema_base_momentum,
            final_momentum=config.ema_final_momentum,
        )
        self.predictor = DualOutputPredictor(config.predictor)

    @property
    def config(self) -> JepaConfig:
        return self._config

    @property
    def latent_dim(self) -> int:
        return self._config.latent_dim

    def encode(self, images: torch.Tensor) -> EncoderOutput:
        """Online encoding: this is what trains."""
        return self.encoder(images)

    def encode_target(self, images: torch.Tensor) -> EncoderOutput:
        """Target encoding: always detached."""
        return self.target(images)  # type: ignore[return-value]

    def predict(
        self, latent: torch.Tensor, action: torch.Tensor, pair: PredictionPair
    ) -> PredictorOutput:
        """Predict the carrier (and the selected mode readout) for one pair."""
        return self.predictor(latent, action, pair)

    def rollout(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        pairs: tuple[PredictionPair, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Chain carrier to carrier; mode heads are never constructed here."""
        return self.predictor.rollout(latent, action, pairs)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield online-encoder and predictor parameters only."""
        yield from self.encoder.parameters()
        yield from self.predictor.parameters()

    def update_target(self, *, step: int, total_steps: int) -> None:
        """Advance the EMA target by the cosine-ramped momentum for this step."""
        self.target.update(
            self.encoder, momentum=self.target.momentum_at(step, total_steps)
        )


__all__ = [
    "ContextEncoder",
    "DualOutputPredictor",
    "EmaTargetEncoder",
    "EncoderOutput",
    "JepaBackbone",
    "JepaConfig",
    "PredictionPair",
    "PredictorOutput",
    "build_encoder",
]
