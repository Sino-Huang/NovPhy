from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt


CORE_DATASETS = ("pixels", "action", "reward", "ep_len", "ep_offset")
REQUIRED_METADATA_DATASETS = (
    "episode_id",
    "step_id",
    "task_id",
    "scenario_id",
    "novelty_level",
    "seed",
    "score",
    "terminated",
    "truncated",
    "transition_reason",
    "action_coordinate_convention",
    "static_wait_steps",
)


class SchemaValidationError(ValueError):
    """Raised when a NovPhy HDF5 dataset violates the agreed MVP schema."""


@dataclass(frozen=True)
class DatasetSummary:
    transition_count: int
    episode_count: int
    min_episode_length: int
    action_dim: int


def _read_array(handle: h5py.File, name: str) -> npt.NDArray[np.generic]:
    if name not in handle:
        raise SchemaValidationError(f"Missing required dataset '{name}'.")
    dataset = handle[name]
    if not isinstance(dataset, h5py.Dataset):
        raise SchemaValidationError(f"'{name}' must be an HDF5 dataset.")
    return cast(npt.NDArray[np.generic], np.asarray(dataset[...]))


def _ensure_lengths(ep_len: npt.NDArray[np.int_], ep_offset: npt.NDArray[np.int_], transition_count: int) -> None:
    if ep_len.ndim != 1 or ep_offset.ndim != 1:
        raise SchemaValidationError("'ep_len' and 'ep_offset' must be 1D arrays.")
    if len(ep_len) != len(ep_offset):
        raise SchemaValidationError("'ep_len' and 'ep_offset' must have the same length.")
    if len(ep_len) == 0:
        raise SchemaValidationError("At least one episode is required.")
    if not np.issubdtype(ep_len.dtype, np.integer) or not np.issubdtype(ep_offset.dtype, np.integer):
        raise SchemaValidationError("'ep_len' and 'ep_offset' must be integer arrays.")
    if np.any(ep_len <= 0):
        raise SchemaValidationError("All values in 'ep_len' must be positive.")
    if ep_offset[0] != 0:
        raise SchemaValidationError("'ep_offset' must start at 0.")
    if np.any(np.diff(ep_offset) < 0):
        raise SchemaValidationError("'ep_offset' must be monotonic non-decreasing.")
    expected_offsets = np.concatenate(([0], np.cumsum(ep_len[:-1], dtype=np.int64)))
    if not np.array_equal(ep_offset, expected_offsets):
        raise SchemaValidationError("'ep_offset' must describe contiguous, non-overlapping flattened episodes.")
    if np.any(ep_offset + ep_len > transition_count):
        raise SchemaValidationError("Episode offsets/lengths exceed the number of transitions.")


def _ensure_metadata_lengths(handle: h5py.File, transition_count: int, episode_count: int) -> None:
    for dataset_name in REQUIRED_METADATA_DATASETS:
        values = _read_array(handle, dataset_name)
        if values.ndim == 0:
            raise SchemaValidationError(f"Metadata dataset '{dataset_name}' must not be scalar.")
        length = len(values)
        if length not in (transition_count, episode_count):
            raise SchemaValidationError(
                f"Metadata dataset '{dataset_name}' must have length {transition_count} or {episode_count}; got {length}."
            )


def validate_hdf5_dataset(
    path: str | Path,
    *,
    min_sequence_length: int = 2,
    require_metadata: bool = True,
) -> DatasetSummary:
    file_path = Path(path)
    if not file_path.exists():
        raise SchemaValidationError(f"Dataset file '{file_path}' does not exist.")

    with h5py.File(file_path, "r") as handle:
        pixels = _read_array(handle, "pixels")
        action = _read_array(handle, "action")
        reward = _read_array(handle, "reward")
        ep_len = cast(npt.NDArray[np.int_], _read_array(handle, "ep_len"))
        ep_offset = cast(npt.NDArray[np.int_], _read_array(handle, "ep_offset"))

        if pixels.ndim < 2:
            raise SchemaValidationError("'pixels' must have at least two dimensions with transitions along axis 0.")
        if action.ndim != 2:
            raise SchemaValidationError(f"'action' must have shape (N, 2), got {action.shape}.")
        if action.shape[1] != 2:
            raise SchemaValidationError(f"'action' must have last dimension 2, got {action.shape[1]}.")
        if reward.ndim not in (1, 2):
            raise SchemaValidationError(f"'reward' must be 1D or 2D, got shape {reward.shape}.")

        transition_count = int(action.shape[0])
        if pixels.shape[0] != transition_count:
            raise SchemaValidationError("'pixels' and 'action' must have the same number of transitions.")
        if reward.shape[0] != transition_count:
            raise SchemaValidationError("'reward' and 'action' must have the same number of transitions.")
        if not np.isfinite(action).all():
            raise SchemaValidationError("'action' contains non-finite values.")
        if not np.isfinite(reward).all():
            raise SchemaValidationError("'reward' contains non-finite values.")

        _ensure_lengths(ep_len, ep_offset, transition_count)
        min_episode_length = int(np.min(ep_len))
        if min_episode_length < min_sequence_length:
            raise SchemaValidationError(
                f"Minimum episode length {min_episode_length} is shorter than required sequence length {min_sequence_length}."
            )

        episode_count = int(len(ep_len))
        if require_metadata:
            _ensure_metadata_lengths(handle, transition_count, episode_count)

        return DatasetSummary(
            transition_count=transition_count,
            episode_count=episode_count,
            min_episode_length=min_episode_length,
            action_dim=int(action.shape[1]),
        )
