import tempfile
import unittest
from pathlib import Path

import torch

from world_model.model import JepaBackbone
from world_model.training import (
    GridRunError,
    PhaseAConfig,
    TeacherForcedTrainer,
    fixture_jepa_config,
    load_checkpoint,
)


class _MaliciousCheckpointPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker,)


class CheckpointSecurityTests(unittest.TestCase):
    def test_checkpoint_rejects_pickle_reducer_without_executing_it(self) -> None:
        # Given
        model_config = fixture_jepa_config()
        phase = PhaseAConfig(steps=1, batch_size=2, device="cpu")
        trainer = TeacherForcedTrainer(
            JepaBackbone(model_config), phase.training_config(device="cpu")
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "malicious.pt"
            marker = Path(directory) / "reducer-ran"
            torch.save({"payload": _MaliciousCheckpointPayload(marker)}, checkpoint)

            # When
            with self.assertRaises(GridRunError) as captured:
                load_checkpoint(
                    checkpoint,
                    trainer,
                    config_digest=phase.identity,
                    grid_digest=phase.grid_digest,
                )

            # Then
            self.assertFalse(marker.exists())
            self.assertEqual(str(captured.exception), "checkpoint payload is invalid")
