"""Deterministic sampling and padded batching for temporal windows."""
from __future__ import annotations

from collections.abc import Iterator, Sequence, Sized
from typing import TypedDict

import torch

from world_model.data.dataset import WindowSample
from world_model.data.types import ContractValueError


class TemporalWindowBatch(TypedDict):
    """A padded batch of temporal-window samples."""

    context_image: torch.Tensor
    target_images: torch.Tensor
    target_mask: torch.Tensor
    action: torch.Tensor
    prediction_steps: torch.Tensor
    shot_frame_count: torch.Tensor
    frame_indices: list[list[int]]
    horizon_frames: torch.Tensor
    stride_frames: torch.Tensor
    horizon: torch.Tensor
    stride: torch.Tensor
    provenance: list


class EpochSampler(torch.utils.data.Sampler[int]):
    """Draw a fixed, epoch-specific sequence without using global RNG state."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        draw_count: int,
    ) -> None:
        if type(seed) is not int:
            raise ContractValueError("seed", "must be an integer")
        if type(draw_count) is not int or draw_count <= 0:
            raise ContractValueError("draw_count", "must be a positive integer")
        if len(data_source) == 0:
            raise ContractValueError("data_source", "must not be empty")

        self._data_source = data_source
        self._seed = seed
        self._draw_count = draw_count
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        """Yield exactly ``draw_count`` valid indices for the configured epoch."""
        item_count = len(self._data_source)
        if item_count == 0:
            raise ContractValueError("data_source", "must not be empty")

        generator = torch.Generator()
        generator.manual_seed(self._epoch_seed())
        if self._draw_count <= item_count:
            order = torch.randperm(item_count, generator=generator)
        else:
            order = torch.randint(
                high=item_count,
                size=(self._draw_count,),
                generator=generator,
            )
        yield from order.tolist()[: self._draw_count]

    def __len__(self) -> int:
        """Return the configured, fixed sample budget."""
        return self._draw_count

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic sample order for a nonnegative epoch."""
        if type(epoch) is not int or epoch < 0:
            raise ContractValueError("epoch", "must be a nonnegative integer")
        self._epoch = epoch

    def _epoch_seed(self) -> int:
        """Derive a valid local Torch seed from the configured seed and epoch."""
        modulus = (1 << 63) - 1
        return (self._seed + self._epoch) % modulus


class TemporalWindowCollator:
    """Stack temporal windows and pad their target-image sequences only."""

    def __call__(self, samples: Sequence[WindowSample]) -> TemporalWindowBatch:
        """Collate samples with matching RGB geometry and fixed five-value actions."""
        if not samples:
            raise ContractValueError("samples", "must not be empty")

        first_context = samples[0]["context_image"]
        self._validate_rgb_image(first_context, "context_image")
        target_counts: list[int] = []

        for sample in samples:
            context_image = sample["context_image"]
            self._validate_matching_image(context_image, first_context, "context_image")
            self._validate_action(sample["action"])

            targets = sample["target_images"]
            if not targets:
                raise ContractValueError("target_images", "must not be empty")
            if sample["prediction_steps"] != len(targets):
                raise ContractValueError(
                    "prediction_steps",
                    "must equal the number of target images",
                )
            for target in targets:
                self._validate_matching_image(target, first_context, "target_images")
            target_counts.append(len(targets))

        batch_size = len(samples)
        maximum_targets = max(target_counts)
        target_images = torch.zeros(
            (batch_size, maximum_targets, *first_context.shape),
            dtype=first_context.dtype,
            device=first_context.device,
        )
        target_mask = torch.zeros(
            (batch_size, maximum_targets),
            dtype=torch.bool,
            device=first_context.device,
        )
        for row, sample in enumerate(samples):
            target_count = target_counts[row]
            target_images[row, :target_count] = torch.stack(sample["target_images"])
            target_mask[row, :target_count] = True

        return TemporalWindowBatch(
            context_image=torch.stack([sample["context_image"] for sample in samples]),
            target_images=target_images,
            target_mask=target_mask,
            action=torch.stack([sample["action"] for sample in samples]),
            prediction_steps=torch.tensor(
                [sample["prediction_steps"] for sample in samples],
                dtype=torch.long,
                device=first_context.device,
            ),
            shot_frame_count=torch.tensor(
                [
                    sample.get(
                        "shot_frame_count",
                        sample["provenance"].get("shot_frame_count", max(sample["frame_indices"]) + 1),
                    )
                    for sample in samples
                ],
                dtype=torch.long,
                device=first_context.device,
            ),
            frame_indices=[list(sample["frame_indices"]) for sample in samples],
            horizon_frames=torch.tensor(
                [sample["horizon_frames"] for sample in samples],
                dtype=torch.long,
                device=first_context.device,
            ),
            stride_frames=torch.tensor(
                [sample["stride_frames"] for sample in samples],
                dtype=torch.long,
                device=first_context.device,
            ),
            horizon=torch.tensor(
                [sample["horizon_frames"] for sample in samples],
                dtype=torch.long,
                device=first_context.device,
            ),
            stride=torch.tensor(
                [sample["stride_frames"] for sample in samples],
                dtype=torch.long,
                device=first_context.device,
            ),
            provenance=[sample["provenance"] for sample in samples],
        )

    @staticmethod
    def _validate_rgb_image(image: torch.Tensor, field: str) -> None:
        """Validate one channel-first RGB image tensor."""
        if not isinstance(image, torch.Tensor):
            raise ContractValueError(field, "must be a torch tensor")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ContractValueError(field, "must have shape [3, H, W]")

    @classmethod
    def _validate_matching_image(
        cls,
        image: torch.Tensor,
        reference: torch.Tensor,
        field: str,
    ) -> None:
        """Validate RGB image geometry, dtype, and device against the batch reference."""
        cls._validate_rgb_image(image, field)
        if image.shape != reference.shape:
            raise ContractValueError(field, "must match the context image geometry")
        if image.dtype != reference.dtype:
            raise ContractValueError(field, "must match the context image dtype")
        if image.device != reference.device:
            raise ContractValueError(field, "must match the context image device")

    @staticmethod
    def _validate_action(action: torch.Tensor) -> None:
        """Validate the fixed, per-window five-value action representation."""
        if not isinstance(action, torch.Tensor):
            raise ContractValueError("action", "must be a torch tensor")
        if action.shape != torch.Size([5]):
            raise ContractValueError("action", "must have shape [5]")
