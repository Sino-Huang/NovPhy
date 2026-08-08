import tempfile
import unittest
from pathlib import Path

import torch

from world_model.model import JepaBackbone
from world_model.training import (
    GridRunError,
    PhaseAConfig,
    TeacherForcedTrainer,
    fixture_batch,
    fixture_jepa_config,
    load_checkpoint,
    save_checkpoint,
    score_checkpoint,
)


class GridRunTests(unittest.TestCase):
    def test_phase_a_defaults_pin_primary_contract(self) -> None:
        config = PhaseAConfig()
        self.assertEqual((config.seed, config.steps, config.batch_size), (20260807, 3600, 64))
        self.assertEqual(config.grid_digest, PhaseAConfig().grid_digest)

    def test_checkpoint_round_trip_restores_exact_step_and_rejects_digest(self) -> None:
        model_config = fixture_jepa_config()
        phase = PhaseAConfig(steps=2, batch_size=2, device="cpu")
        first = TeacherForcedTrainer(JepaBackbone(model_config), phase.training_config(device="cpu"))
        first.train_step(fixture_batch(model_config, seed=phase.seed, batch_size=2, step=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            saved = save_checkpoint(path, first, config_digest=phase.identity, grid_digest=phase.grid_digest)
            second = TeacherForcedTrainer(JepaBackbone(model_config), phase.training_config(device="cpu"))
            loaded = load_checkpoint(path, second, config_digest=phase.identity, grid_digest=phase.grid_digest)
            self.assertEqual((saved.digest, loaded.digest, loaded.step), (saved.digest, saved.digest, 1))
            with self.assertRaises(GridRunError):
                load_checkpoint(path, second, config_digest="0" * 64, grid_digest=phase.grid_digest)

    def test_score_is_frozen_and_gradient_free(self) -> None:
        model_config = fixture_jepa_config()
        phase = PhaseAConfig(steps=1, batch_size=2, device="cpu")
        trainer = TeacherForcedTrainer(JepaBackbone(model_config), phase.training_config(device="cpu"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, trainer, config_digest=phase.identity, grid_digest=phase.grid_digest)
            result = score_checkpoint(
                checkpoint,
                phase_config=phase,
                model_config=model_config,
                batches=tuple(fixture_batch(model_config, seed=phase.seed, batch_size=2, step=i) for i in range(9)),
            )
            self.assertEqual((result.step, result.count), (0, 18))
            self.assertTrue(torch.isfinite(torch.tensor(result.mean_loss)))


if __name__ == "__main__":
    unittest.main()
