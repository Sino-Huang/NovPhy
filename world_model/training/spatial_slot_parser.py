"""Spatial slot parser: the issue-70 parser interface with localized slot attention.

The frozen issue-70 parser pools a 96x64 image to one global vector and adds a
learned per-slot offset, so it cannot localize small objects (on the N1 families
it reads the pig as absent in every decision frame, issue #99). This parser keeps
the exact output interface consumed by ``TemporalVisualCarrierAdapter.parse_batch``
(presence/center/kind logits per vocabulary slot, pairwise relation logits, macro
logits) but keeps a spatial feature map and lets each slot query attend over it.
A slot's center is the attention-weighted mean of the feature-cell coordinates of
its last attention layer, so localization is grounded in where the slot looks.
"""
from __future__ import annotations

import torch
from torch import nn

from world_model.training.cohort_v2_visual_parser import (
    ENTITY_KINDS,
    MACRO_PREDICATES,
    RELATION_PREDICATES,
    CohortV2VisualParserConfig,
)


def _block(inputs, outputs, kernel, stride):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=kernel // 2),
                         nn.GroupNorm(8, outputs), nn.LeakyReLU(0.1))


class SpatialSlotParser(nn.Module):
    architecture_identity = "issue-99-spatial-slot-parser-v1"
    maximum_object_slots = None

    def __init__(self, config: CohortV2VisualParserConfig, object_vocabulary: tuple[str, ...],
                 layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        if config.image_height % 8 or config.image_width % 8:
            raise ValueError("spatial slot parser needs image sides divisible by 8")
        self.config = config
        self.object_vocabulary = tuple(object_vocabulary)
        width = config.hidden_dim
        self.backbone = nn.Sequential(_block(3, 32, 5, 2), _block(32, 64, 3, 2),
                                      _block(64, 96, 3, 2), _block(96, width, 3, 1))
        rows, columns = config.image_height // 8, config.image_width // 8
        y, x = torch.meshgrid((torch.arange(rows) + 0.5) / rows, (torch.arange(columns) + 0.5) / columns,
                              indexing="ij")
        self.register_buffer("coordinates", torch.stack((x, y), -1).reshape(-1, 2), persistent=False)
        self.position = nn.Linear(2, width)
        self.queries = nn.Parameter(torch.randn(len(self.object_vocabulary), width) * 0.02)
        self.attention = nn.ModuleList(nn.MultiheadAttention(width, heads, batch_first=True)
                                       for _ in range(layers))
        self.attention_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.feedforward = nn.ModuleList(nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                                       nn.Linear(2 * width, width))
                                         for _ in range(layers))
        self.feedforward_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.presence_head = nn.Linear(width, 1)
        self.center_offset = nn.Linear(width, 2)
        self.kind_head = nn.Linear(width, len(ENTITY_KINDS))
        self.relation_head = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(),
                                           nn.Linear(width, len(RELATION_PREDICATES)))
        self.macro_head = nn.Linear(width, len(MACRO_PREDICATES))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images.float() / 127.5 - 1.0).flatten(2).transpose(1, 2)
        features = features + self.position(self.coordinates)[None]
        slots = self.queries[None].expand(len(features), -1, -1)
        weights = None
        for attend, norm, feedforward, norm2 in zip(self.attention, self.attention_norms,
                                                    self.feedforward, self.feedforward_norms):
            update, weights = attend(norm(slots), features, features, need_weights=True,
                                     average_attn_weights=True)
            slots = slots + update
            slots = slots + feedforward(norm2(slots))
        attended = weights @ self.coordinates
        centers = (attended + 0.1 * torch.tanh(self.center_offset(slots))).clamp(0.0, 1.0)
        count = len(self.object_vocabulary)
        first = slots.unsqueeze(2).expand(-1, -1, count, -1)
        second = slots.unsqueeze(1).expand(-1, count, -1, -1)
        return {
            "presence_logits": self.presence_head(slots).squeeze(-1),
            "centers": centers,
            "kind_logits": self.kind_head(slots),
            "relation_logits": self.relation_head(torch.cat((first, second, first - second), -1)),
            "macro_logits": self.macro_head(features.mean(1)),
        }
