"""Context encoder for the JEPA carrier.

The encoder maps an RGB frame to the continuous latent ``z`` that the rollout
carries.  It is deliberately batch-independent: GroupNorm and LayerNorm only,
never BatchNorm, so the EMA/stop-grad target branch cannot be perturbed by the
composition of a batch.

``EncoderOutput.tokens`` is a side output reserved for the Milestone 1c SPSG
work.  It is *not* the state carrier and must never cross a rollout step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

import torch
from torch import nn

from world_model.data.types import ContractValueError
from world_model.model.config import EncoderConfig


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """One encoder forward pass.

    latent : torch.Tensor  [B, latent_dim]  — the sole rollout state carrier
    tokens : torch.Tensor  [B, N, C]        — pre-pool grid, side output only
    """

    latent: torch.Tensor
    tokens: torch.Tensor


class AttentionPool(nn.Module):
    """Pool a token grid into a single carrier vector with a learned query."""

    def __init__(self, token_dim: int, latent_dim: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.projection = nn.Linear(token_dim, latent_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.output = nn.Linear(latent_dim, latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.projection(self.norm(tokens))
        query = self.query.expand(projected.shape[0], -1, -1)
        pooled, _ = self.attention(query, projected, projected, need_weights=False)
        return self.output(pooled.squeeze(1))


class ContextEncoder(nn.Module):
    """Strided convolutional trunk with GroupNorm, followed by attention pooling."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        if type(config) is not EncoderConfig:
            raise ContractValueError("encoder config", "must be an EncoderConfig")
        self._config = config

        groups = config.group_norm_groups
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(groups, config.stem_channels),
            nn.SiLU(),
        )

        stages: list[nn.Module] = []
        in_channels = config.stem_channels
        for out_channels in config.stage_channels:
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
            ]
            for _ in range(config.blocks_per_stage - 1):
                layers.extend(
                    (
                        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
                        nn.GroupNorm(groups, out_channels),
                        nn.SiLU(),
                    )
                )
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)

        self.pool = AttentionPool(in_channels, config.latent_dim, config.pool_heads)

    @property
    def config(self) -> EncoderConfig:
        return self._config

    @property
    def latent_dim(self) -> int:
        return self._config.latent_dim

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        """Encode a batch of channel-first RGB frames into carrier and tokens."""
        if not isinstance(images, torch.Tensor):
            raise ContractValueError("images", "must be a torch tensor")
        if images.ndim != 4:
            raise ContractValueError("images", "must have shape [B, C, H, W]")
        if images.shape[1] != self._config.input_channels:
            raise ContractValueError(
                "images", f"must carry {self._config.input_channels} channels"
            )

        features = self.stages(self.stem(images))
        tokens = features.flatten(2).transpose(1, 2)
        return EncoderOutput(latent=self.pool(tokens), tokens=tokens)


_ENCODER_REGISTRY: Final[dict[str, Callable[[EncoderConfig], nn.Module]]] = {
    "conv_gn_v1": ContextEncoder,
}


def build_encoder(config: EncoderConfig) -> nn.Module:
    """Resolve and construct the encoder backbone named by ``config``.

    Milestone 2 introduces a frozen pretrained ViT; it registers here and needs
    no change to the predictor, the EMA wrapper, or the training loop.
    """
    if type(config) is not EncoderConfig:
        raise ContractValueError("encoder config", "must be an EncoderConfig")
    factory = _ENCODER_REGISTRY.get(config.name)
    if factory is None:
        raise ContractValueError(
            "encoder name", f"{config.name!r} is not a registered backbone"
        )
    return factory(config)
