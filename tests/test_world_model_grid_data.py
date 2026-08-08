import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from world_model.data.types import (
    CaptureContractDescriptor,
    EpisodeRecord,
    FrameRecord,
    ShotAction,
    ShotRecord,
)
from world_model.training import grid_data


def _episode(name: str, frame_count: int = 5) -> EpisodeRecord:
    frames = tuple(
        FrameRecord(index=index, relative_path=f"{name}/shot_001/frame_{index:06d}.png")
        for index in range(frame_count)
    )
    return EpisodeRecord(
        name=name,
        split="dev",
        relative_path=f"dev/{name}",
        shots=(
            ShotRecord(
                name="shot_001",
                relative_path=f"dev/{name}/shot_001",
                action=ShotAction((1.0, 2.0, 3.0, 4.0, 5.0)),
                frames=frames,
            ),
        ),
        capture_contract=CaptureContractDescriptor(
            contract_name="legacy_rgb_v1",
            contract_version="1",
            artifact_layout_version="collector_v1",
            player_provenance=None,
            protocol_provenance="desktop-imagegrab",
        ),
    )


class GridDataTests(unittest.TestCase):
    def test_five_frame_shot_enumerates_four_states_and_three_targets(self):
        states = grid_data.enumerate_scoring_states(
            (_episode("episode_001"),),
            catalog_digest="a" * 64,
            split="dev",
        )
        self.assertEqual(len(states), 4)
        self.assertTrue(all(len(state.targets) == 3 for state in states))
        self.assertEqual(
            tuple(target.frame_position for target in states[-1].targets), (4, 4, 4)
        )
        self.assertEqual(tuple(target.effective_delta for target in states[-1].targets), (1, 1, 1))

    def test_one_frame_shot_fails_closed(self):
        with self.assertRaises(grid_data.GridDataContractError):
            grid_data.enumerate_scoring_states(
                (_episode("episode_001", frame_count=1),),
                catalog_digest="a" * 64,
                split="dev",
            )

    def test_partition_is_deterministic_disjoint_and_complete(self):
        episodes = tuple(_episode(f"episode_{index:03d}") for index in range(50))
        first = grid_data.partition_episodes(episodes, catalog_digest="b" * 64, seed=7)
        second = grid_data.partition_episodes(episodes, catalog_digest="b" * 64, seed=7)
        self.assertEqual(first, second)
        sets = [set(first.controller_train), set(first.calibration), set(first.evaluation)]
        self.assertEqual(sum(map(len, sets)), len(episodes))
        self.assertEqual(set().union(*sets), set(episodes))
        self.assertEqual(sum(len(left & right) for left in sets for right in sets if left is not right), 0)

    def test_partition_digest_uses_contract_tuple(self):
        episode = _episode("episode_001")
        expected = hashlib.sha256(
            grid_data.canonical_partition_payload(7, "c" * 64, episode.relative_path)
        ).hexdigest()
        self.assertEqual(grid_data.partition_digest(7, "c" * 64, episode.relative_path), expected)

    def test_regime_boundary_ties_are_deterministic(self):
        calibration = grid_data.calibrate_motion_regimes((0.1, 0.2, 0.3, 0.4, 0.5))
        self.assertEqual(calibration.classify(calibration.p50), grid_data.MotionRegime.QUIESCENT)
        self.assertEqual(calibration.classify(calibration.p90), grid_data.MotionRegime.TRANSITIONAL)

    def test_motion_score_is_target_aware_and_resized(self):
        context = torch.zeros(3, 4, 5)
        target = torch.ones(3, 2, 3)
        score = grid_data.diagnostic_motion_score(context, target)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(grid_data.DIAGNOSTIC_IMAGE_SIZE, (240, 320))

    def test_collator_retains_temporal_metadata(self):
        from world_model.data.sampling import TemporalWindowCollator

        sample = {
            "context_image": torch.zeros(3, 4, 4),
            "target_images": [torch.ones(3, 4, 4)],
            "action": torch.zeros(5),
            "frame_indices": [2, 4],
            "horizon_frames": 2,
            "prediction_steps": 1,
            "stride_frames": 2,
            "shot_frame_count": 5,
            "provenance": {},
        }
        batch = TemporalWindowCollator()([sample])
        self.assertEqual(batch["shot_frame_count"].tolist(), [5])
        self.assertEqual(batch["frame_indices"], [[2, 4]])
        self.assertEqual(batch["horizon_frames"].tolist(), [2])
        self.assertEqual(batch["stride_frames"].tolist(), [2])
        self.assertEqual(batch["horizon"].tolist(), [2])
        self.assertEqual(batch["stride"].tolist(), [2])

    def test_catalog_bound_motion_rejects_state_with_stale_shot_length(self):
        from tests.test_world_model_data import _build_catalog_from_fixture

        with TemporaryDirectory() as temporary:
            catalog = _build_catalog_from_fixture(
                Path(temporary), split="dev", frame_count=5, shot_count=1
            )
            state = grid_data.enumerate_scoring_states(catalog)[0]
            stale_state = grid_data.ScoringState(
                catalog_digest=state.catalog_digest,
                split=state.split,
                episode_relative_path=state.episode_relative_path,
                shot_relative_path=state.shot_relative_path,
                context_position=0,
                shot_frame_count=6,
                targets=(
                    grid_data.ScoringTarget(1, 1, 1),
                    grid_data.ScoringTarget(5, 5, 5),
                    grid_data.ScoringTarget(15, 5, 5),
                ),
            )
            with self.assertRaises(grid_data.GridDataContractError):
                grid_data.diagnostic_motion_for_state(catalog, stale_state)


if __name__ == "__main__":
    unittest.main()
