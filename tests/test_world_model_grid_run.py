import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from scripts import run_jepa_pair_grid
import world_model.training as world_model_training
from world_model.data import LEGACY_RGB_V1, EpisodeCatalog, catalog_digest
from world_model.data.types import EpisodeRecord, FrameRecord, ShotAction, ShotRecord
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
    def test_real_phase_data_decodes_scheduled_windows_and_indexes_all_states(self) -> None:
        # Given
        self.assertTrue(hasattr(world_model_training, "RealPhaseData"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = None
            partitions = None
            for episode_count in range(8, 41):
                episodes = tuple(
                    self._real_episode(index, frame_count=17)
                    for index in range(episode_count)
                )
                candidate = EpisodeCatalog(
                    root=root,
                    split="dev",
                    capture_contract=LEGACY_RGB_V1,
                    required_capabilities=(),
                    plan_path=None,
                    episodes=episodes,
                    rejection_count=0,
                    rejection_code_counts={},
                    provenance_available=False,
                )
                candidate_partitions = world_model_training.partition_episodes(
                    candidate, seed=20260807
                )
                if (
                    len(candidate_partitions.controller_train) >= 3
                    and candidate_partitions.calibration
                    and candidate_partitions.evaluation
                ):
                    catalog = candidate
                    partitions = candidate_partitions
                    break
            self.assertIsNotNone(catalog)
            self.assertIsNotNone(partitions)
            assert catalog is not None and partitions is not None
            controller = {
                episode.relative_path: (1, 4, 10)[index % 3]
                for index, episode in enumerate(partitions.controller_train)
            }
            calibration = {episode.relative_path: 5 for episode in partitions.calibration}
            for episode in catalog.episodes:
                scale = controller.get(episode.relative_path, calibration.get(episode.relative_path, 5))
                for frame in episode.shots[0].frames:
                    path = root / frame.relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value = min(255, frame.index * scale)
                    Image.new("RGB", (8, 6), (value, value, value)).save(path)
            phase = PhaseAConfig(steps=9, batch_size=2, device="cpu")
            model_config = fixture_jepa_config()

            # When
            with mock.patch(
                "world_model.training.real_data.EpisodeCatalog.build", return_value=catalog
            ):
                data = world_model_training.RealPhaseData.build(root, phase, model_config)
            trainer = TeacherForcedTrainer(
                JepaBackbone(model_config), phase.training_config(device="cpu")
            )
            batches = tuple(
                data.training_batch(*trainer.schedule_at(step), phase.batch_size, step)
                for step in range(9)
            )

            # Then
            self.assertEqual(data.catalog_digest, catalog_digest(catalog))
            self.assertEqual(len(data.examples), len(catalog.episodes) * 16)
            self.assertEqual(
                {
                    (batch["prediction_pair"].delta, batch["motion_regime"])
                    for batch in batches
                },
                {
                    (delta, regime)
                    for delta in (1, 5, 15)
                    for regime in world_model_training.MotionRegime
                },
            )
            self.assertTrue(all(batch["context_image"].shape == (2, 3, 16, 16) for batch in batches))
            self.assertTrue(
                all(
                    target - context == batch["prediction_pair"].delta
                    for batch in batches
                    for context, target in batch["frame_indices"]
                )
            )

    @staticmethod
    def _real_episode(index: int, frame_count: int) -> EpisodeRecord:
        name = f"episode_{index:03d}"
        shot_path = f"dev/{name}/shot_001"
        return EpisodeRecord(
            name=name,
            split="dev",
            relative_path=f"dev/{name}",
            shots=(
                ShotRecord(
                    name="shot_001",
                    relative_path=shot_path,
                    action=ShotAction((300.0, 220.0, -80.0, 20.0, 120.0)),
                    frames=tuple(
                        FrameRecord(
                            frame_index,
                            f"{shot_path}/frames/frame_{frame_index:06d}.png",
                        )
                        for frame_index in range(frame_count)
                    ),
                ),
            ),
            capture_contract=LEGACY_RGB_V1,
        )

    def test_real_cli_defaults_to_the_protected_legacy_dataset(self) -> None:
        # Given / When
        args = run_jepa_pair_grid._parse_args(["train"])

        # Then
        self.assertEqual(
            args.dataset_root,
            Path(
                "/mnt/array/sukaih/Project/NovPhy/data/"
                "novphy_rollouts_dataset_20260708_171531"
            ),
        )

    def test_real_cli_rejects_output_inside_the_protected_dataset(self) -> None:
        # Given
        with tempfile.TemporaryDirectory(prefix="novphy_rollouts_dataset_") as directory:
            dataset_root = Path(directory)

            # When / Then
            with self.assertRaisesRegex(GridRunError, "protected dataset"):
                run_jepa_pair_grid.main(
                    [
                        "train",
                        "--dataset-root",
                        str(dataset_root),
                        "--output-dir",
                        str(dataset_root / "run"),
                        "--device",
                        "cpu",
                        "--steps",
                        "1",
                    ]
                )

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

    def test_checkpoint_rejects_a_stale_catalog_digest(self) -> None:
        # Given
        model_config = fixture_jepa_config()
        phase = PhaseAConfig(steps=1, batch_size=2, device="cpu")
        first = TeacherForcedTrainer(
            JepaBackbone(model_config), phase.training_config(device="cpu")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                first,
                config_digest=phase.identity,
                grid_digest=phase.grid_digest,
                catalog_digest="a" * 64,
                run_identity="b" * 64,
            )
            second = TeacherForcedTrainer(
                JepaBackbone(model_config), phase.training_config(device="cpu")
            )

            # When / Then
            with self.assertRaisesRegex(GridRunError, "catalog digest mismatch"):
                load_checkpoint(
                    path,
                    second,
                    config_digest=phase.identity,
                    grid_digest=phase.grid_digest,
                    expected_catalog_digest="c" * 64,
                )

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
