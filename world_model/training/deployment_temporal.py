"""Atomic world-model training and validation over deployment trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model.data.deployment_temporal import (
    DeploymentTemporalError,
    DeploymentTrajectoryReader,
    TemporalVisualCarrierAdapter,
    TransitionCarriers,
    build_transition_carriers,
)


@dataclass(frozen=True, slots=True)
class DeploymentTrajectoryCarrierExample:
    trajectory_identity: str
    scenario_lineage_identity: str
    exposure_role: str
    transitions: tuple[TransitionCarriers, ...]


class DeploymentTemporalTrainingData(torch.utils.data.Dataset):
    """Expose one complete trajectory, never one independently sampled decision."""

    def __init__(
        self,
        reader: DeploymentTrajectoryReader,
        adapter: TemporalVisualCarrierAdapter,
    ) -> None:
        if not isinstance(reader, DeploymentTrajectoryReader) or not isinstance(
            adapter, TemporalVisualCarrierAdapter
        ):
            raise DeploymentTemporalError(
                "deployment training data requires a trajectory reader and adapter"
            )
        self.reader = reader
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.reader.trajectories)

    def __getitem__(self, index: int) -> DeploymentTrajectoryCarrierExample:
        trajectory = self.reader.trajectories[index]
        return DeploymentTrajectoryCarrierExample(
            trajectory_identity=trajectory.identity,
            scenario_lineage_identity=trajectory.scenario_lineage_identity,
            exposure_role=trajectory.exposure_role,
            transitions=tuple(
                build_transition_carriers(transition, self.adapter)
                for transition in trajectory.transitions
            ),
        )


def validate_deployment_temporal_training_data(
    data: DeploymentTemporalTrainingData,
) -> dict[str, int | str]:
    """Exercise the shared adapter while preserving trajectory-level accounting."""
    if not isinstance(data, DeploymentTemporalTrainingData) or not len(data):
        raise DeploymentTemporalError("deployment training validation data is empty")
    transition_count = 0
    for index in range(len(data)):
        example = data[index]
        if (
            example.exposure_role != data.reader.exposure_role
            or not example.transitions
            or any(
                item.context.tensor.shape != item.target.tensor.shape
                for item in example.transitions
            )
        ):
            raise DeploymentTemporalError(
                "deployment training example differs from its trajectory contract"
            )
        transition_count += len(example.transitions)
    return {
        "exposure_role": data.reader.exposure_role,
        "trajectory_count": len(data),
        "transition_count": transition_count,
        "lineage_manifest_identity": data.reader.lineage_manifest.identity,
    }


__all__ = [
    "DeploymentTemporalTrainingData",
    "DeploymentTrajectoryCarrierExample",
    "validate_deployment_temporal_training_data",
]
