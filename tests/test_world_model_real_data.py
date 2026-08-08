import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from world_model.data import LEGACY_RGB_V1, EpisodeCatalog
from world_model.data.types import EpisodeRecord, FrameRecord, ShotAction, ShotRecord
from world_model.training import (
    PhaseAConfig,
    calibrate_motion_regimes,
    diagnostic_motion_score,
    enumerate_scoring_states,
    fixture_jepa_config,
    partition_episodes,
)
from world_model.training.real_data import RealPhaseData


class RealDataAdapterTests(unittest.TestCase):
    def test_build_reads_each_shot_frame_once_for_motion_calibration(self) -> None:
        # Given
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = tuple(self._episode(index) for index in range(40))
            catalog = EpisodeCatalog(
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
            phase = PhaseAConfig(steps=1, batch_size=1)
            partitions = partition_episodes(catalog, seed=phase.seed)
            controller_scales = {
                episode.relative_path: (1, 4, 10)[index % 3]
                for index, episode in enumerate(partitions.controller_train)
            }
            for episode in episodes:
                scale = controller_scales.get(episode.relative_path, 5)
                for frame in episode.shots[0].frames:
                    path = root / frame.relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value = min(255, frame.index * scale)
                    Image.new("RGB", (8, 6), (value,) * 3).save(path)

            states = enumerate_scoring_states(catalog)
            shots = {
                (episode.relative_path, shot.relative_path): shot
                for episode in episodes
                for shot in episode.shots
            }
            expected_scores = tuple(
                diagnostic_motion_score(
                    root
                    / shots[(state.episode_relative_path, state.shot_relative_path)]
                    .frames[state.context_position]
                    .relative_path,
                    root
                    / shots[(state.episode_relative_path, state.shot_relative_path)]
                    .frames[state.targets[-1].frame_position]
                    .relative_path,
                )
                for state in states
            )
            calibration_paths = {
                episode.relative_path for episode in partitions.calibration
            }
            expected_calibration = calibrate_motion_regimes(
                tuple(
                    score
                    for state, score in zip(states, expected_scores, strict=True)
                    if state.episode_relative_path in calibration_paths
                )
            )
            expected_opens = sum(len(shot.frames) for episode in episodes for shot in episode.shots)

            # When
            with mock.patch(
                "world_model.training.real_data.EpisodeCatalog.build", return_value=catalog
            ), mock.patch(
                "world_model.training.real_data.Image.open",
                wraps=Image.open,
            ) as image_open:
                data = RealPhaseData.build(root, phase, fixture_jepa_config())

            # Then
            self.assertEqual(image_open.call_count, expected_opens)
            self.assertEqual(data.calibration, expected_calibration)
            self.assertEqual(
                tuple(example.motion_regime for example in data.examples),
                tuple(expected_calibration.classify(score) for score in expected_scores),
            )

    @staticmethod
    def _episode(index: int) -> EpisodeRecord:
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
                        FrameRecord(index, f"{shot_path}/frames/frame_{index:06d}.png")
                        for index in range(17)
                    ),
                ),
            ),
            capture_contract=LEGACY_RGB_V1,
        )
