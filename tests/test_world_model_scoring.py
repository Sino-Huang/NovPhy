from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from world_model.model import Abstraction, JepaBackbone, PredictionPair
from world_model.training.grid_run import PhaseAConfig, fixture_jepa_config, save_checkpoint
from world_model.training.loop import TeacherForcedTrainer
from world_model.training.grid_data import MotionRegime, ScoringState, ScoringTarget
from world_model.training.grid_artifacts import canonical_json_bytes
from world_model.training.scoring import (
    ExhaustiveScorer,
    Partition,
    ScoreArtifactError,
    ScoringExample,
    validate_score_artifacts,
    write_score_artifacts,
)
from world_model.training.scoring_torch import TorchCatalogPredictor, score_fixture_checkpoint


class RegimePredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, tuple[str, ...]]] = []

    def latent_mse(
        self,
        examples: tuple[ScoringExample, ...],
        requested_delta: int,
        effective_delta: int,
    ) -> tuple[float, ...]:
        self.calls.append((requested_delta, effective_delta, tuple(item.state_id for item in examples)))
        best = {
            MotionRegime.QUIESCENT: 15,
            MotionRegime.TRANSITIONAL: 5,
            MotionRegime.HIGH_MOTION: 1,
        }
        return tuple(abs(requested_delta - best[item.motion_regime]) / 10.0 for item in examples)


class _SyntheticShotData:
    def __init__(self, states: tuple[ScoringState, ...]) -> None:
        self._states = {
            ScoringExample.from_grid_state(state, Partition.CONTROLLER_TRAIN, MotionRegime.QUIESCENT).state_id: state
            for state in states
        }
        self.decode_batch_sizes: list[int] = []

    def state_record(self, state_id: str):
        return self._states[state_id]

    def shot_frame_count(self, episode_relative_path: str, shot_relative_path: str) -> int:
        del episode_relative_path, shot_relative_path
        return 4

    def shot_action(self, episode_relative_path: str, shot_relative_path: str) -> torch.Tensor:
        del episode_relative_path, shot_relative_path
        return torch.tensor((2.0,), dtype=torch.float32)

    def shot_frame_batch(
        self,
        episode_relative_path: str,
        shot_relative_path: str,
        frame_positions: tuple[int, ...],
    ) -> torch.Tensor:
        del episode_relative_path, shot_relative_path
        self.decode_batch_sizes.append(len(frame_positions))
        return torch.tensor(
            [[[[float(position)]]] for position in frame_positions], dtype=torch.float32
        )


class _InstrumentedBackbone:
    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros(1))
        self.encode_batch_sizes: list[int] = []
        self.target_encode_batch_sizes: list[int] = []
        self.predict_batch_sizes: list[int] = []

    def parameters(self):
        return iter((self._parameter,))

    def encode(self, images: torch.Tensor) -> SimpleNamespace:
        self.encode_batch_sizes.append(images.shape[0])
        return SimpleNamespace(latent=images[:, 0, 0, 0].unsqueeze(1))

    def encode_target(self, images: torch.Tensor) -> SimpleNamespace:
        self.target_encode_batch_sizes.append(images.shape[0])
        return SimpleNamespace(latent=100.0 + images[:, 0, 0, 0].unsqueeze(1))

    def predict(
        self, latent: torch.Tensor, action: torch.Tensor, pair: PredictionPair
    ) -> SimpleNamespace:
        self.predict_batch_sizes.append(latent.shape[0])
        return SimpleNamespace(carrier=latent + action + pair.delta)


def _examples() -> tuple[ScoringExample, ...]:
    rows: list[ScoringExample] = []
    for partition, prefix in (
        (Partition.CONTROLLER_TRAIN, "train"),
        (Partition.CALIBRATION, "cal"),
        (Partition.EVALUATION, "eval"),
    ):
        for index in range(6):
            rows.append(
                ScoringExample(
                    state_id=f"{prefix}-{index:02d}",
                    partition=partition,
                    motion_regime=tuple(MotionRegime)[index % 3],
                    frame_count=7,
                    context_position=index,
                )
            )
    return tuple(rows)


class ExhaustiveScoringTests(unittest.TestCase):
    def test_real_catalog_scoring_bounds_decode_and_encode_batches(self) -> None:
        # Given
        states = tuple(
            ScoringState(
                catalog_digest="a" * 64,
                split="dev",
                episode_relative_path="dev/synthetic",
                shot_relative_path="dev/synthetic/shot_001",
                context_position=position,
                shot_frame_count=4,
                targets=tuple(
                    ScoringTarget(delta, min(delta, 3 - position), min(position + delta, 3))
                    for delta in (1, 5, 15)
                ),
            )
            for position in range(3)
        )
        examples = tuple(
            ScoringExample.from_grid_state(state, partition, MotionRegime.QUIESCENT)
            for state, partition in zip(states, Partition, strict=True)
        )
        data = _SyntheticShotData(states)
        backbone = _InstrumentedBackbone()
        predictor = TorchCatalogPredictor(backbone, data, batch_size=2, examples=examples)

        # When
        result = ExhaustiveScorer(predictor).score(examples)

        # Then
        self.assertLessEqual(max(data.decode_batch_sizes), 2)
        self.assertLessEqual(max(backbone.encode_batch_sizes), 2)
        self.assertLessEqual(max(backbone.target_encode_batch_sizes), 2)
        self.assertLessEqual(max(backbone.predict_batch_sizes), 2)
        self.assertEqual(data.decode_batch_sizes, [2, 2])
        for scored in result.scored_states:
            position = scored.example.context_position
            for metric in scored.label.metrics:
                target_position = min(position + metric.requested_delta, 3)
                expected = (position + 2.0 + metric.requested_delta - (100.0 + target_position)) ** 2
                self.assertAlmostEqual(metric.latent_mse, expected)

    def test_every_state_has_three_ordered_scores_and_one_regime_dependent_label(self) -> None:
        # Given
        predictor = RegimePredictor()

        # When
        result = ExhaustiveScorer(predictor).score(_examples())

        # Then
        self.assertEqual(result.score_count, 18 * 3)
        self.assertEqual(len(result.labels), 12)
        for scored in result.scored_states:
            self.assertEqual(tuple(metric.pair.delta for metric in scored.label.metrics), (1, 5, 15))
        selected = {
            scored.example.motion_regime: scored.label.selected_pair.delta
            for scored in result.labels
        }
        self.assertEqual(
            selected,
            {
                MotionRegime.QUIESCENT: 15,
                MotionRegime.TRANSITIONAL: 5,
                MotionRegime.HIGH_MOTION: 1,
            },
        )

    def test_batches_are_keyed_by_actual_requested_and_effective_delta(self) -> None:
        # Given
        predictor = RegimePredictor()

        # When
        ExhaustiveScorer(predictor).score(_examples())

        # Then
        keys = {(requested, effective) for requested, effective, _ids in predictor.calls}
        self.assertEqual(
            keys,
            {(1, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
             (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6)},
        )

    def test_scale_uses_only_calibration_p90_and_aggregates_recompute(self) -> None:
        # Given
        examples = _examples()

        # When
        result = ExhaustiveScorer(RegimePredictor()).score(examples)

        # Then
        calibration_errors = sorted(
            metric.weighted_prediction_error
            for state in result.scored_states
            if state.example.partition is Partition.CALIBRATION
            for metric in state.label.metrics
        )
        rank = 0.9 * (len(calibration_errors) - 1)
        lower = math.floor(rank)
        expected = calibration_errors[lower] + (
            calibration_errors[math.ceil(rank)] - calibration_errors[lower]
        ) * (rank - lower)
        self.assertEqual(result.score_spec.error_scale, expected)
        self.assertEqual(sum(metric.count for metric in result.per_pair_metrics if metric.motion_regime is None), 12 * 3)
        delta_one = next(
            metric for metric in result.per_pair_metrics
            if metric.partition == Partition.EVALUATION and metric.motion_regime is None and metric.delta == 1
        )
        raw = [
            metric for state in result.labels
            if state.example.partition is Partition.EVALUATION
            for metric in state.label.metrics if metric.pair.delta == 1
        ]
        self.assertEqual(delta_one.latent_mse_mean, sum(item.latent_mse for item in raw) / len(raw))

    def test_every_emitted_aggregate_has_a_canonical_truncation_rate(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())

        # When
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            payload = json.loads((root / "per_pair_metrics.json").read_text(encoding="ascii"))

        # Then
        self.assertTrue(result.per_pair_metrics)
        for aggregate, record in zip(result.per_pair_metrics, payload, strict=True):
            self.assertGreater(aggregate.count, 0)
            self.assertTrue(math.isfinite(aggregate.truncation_rate))
            self.assertEqual(aggregate.truncation_rate, aggregate.truncation_count / aggregate.count)
            self.assertEqual(record["truncation_rate"], aggregate.truncation_rate)

    def test_empty_motion_groups_are_omitted_instead_of_fabricating_rates(self) -> None:
        # Given
        examples = tuple(replace(item, motion_regime=MotionRegime.QUIESCENT) for item in _examples())

        # When
        result = ExhaustiveScorer(RegimePredictor()).score(examples)

        # Then
        self.assertEqual(
            {metric.motion_regime for metric in result.per_pair_metrics},
            {None, MotionRegime.QUIESCENT},
        )
        self.assertTrue(all(metric.count > 0 for metric in result.per_pair_metrics))

    def test_validator_recomputes_truncation_rate_from_label_shards(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            metrics_path = root / "per_pair_metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="ascii"))
            payload[0]["truncation_rate"] = 0.5 if payload[0]["truncation_rate"] == 0.0 else 0.0
            metrics_path.write_bytes(canonical_json_bytes(payload))

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "do not recompute"):
                validate_score_artifacts(root)

    def test_temporal_oracle_and_fixed_pairs_share_identical_state_set(self) -> None:
        # Given / When
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())

        # Then
        ceiling = result.temporal_oracle_ceiling
        self.assertEqual(ceiling.state_count, 6)
        self.assertTrue(all(item.state_digest == ceiling.state_digest for item in ceiling.fixed_pairs))
        self.assertLessEqual(ceiling.oracle_primary_mean, min(item.primary_mean for item in ceiling.fixed_pairs))

    def test_artifacts_round_trip_and_resume_recompute_from_shards(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())

        # When
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            first = write_score_artifacts(root, result, checkpoint_digest="a" * 64, resume=False, shard_size=4)
            second = write_score_artifacts(root, result, checkpoint_digest="a" * 64, resume=True, shard_size=4)
            validated = validate_score_artifacts(root)

        # Then
        self.assertEqual(first, second)
        self.assertEqual(validated, first)

    def test_interleaved_partition_results_validate_in_canonical_shard_order(self) -> None:
        # Given
        examples = _examples()
        interleaved = tuple(
            examples[index]
            for row in range(6)
            for index in (row, row + 6, row + 12)
        )
        result = ExhaustiveScorer(RegimePredictor()).score(interleaved)

        # When / Then
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            self.assertEqual(validate_score_artifacts(root).state_count, len(interleaved))

    def test_publication_aborts_for_partial_mixed_nonfinite_or_changed_scale(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        first = result.scored_states[0]
        missing_label = replace(first.label, metrics=first.label.metrics[1:])
        missing = replace(result, scored_states=(replace(first, label=missing_label), *result.scored_states[1:]))
        calibration_index = next(
            index for index, item in enumerate(result.scored_states)
            if item.example.partition is Partition.CALIBRATION
        )
        calibration_state = result.scored_states[calibration_index]
        mixed_state = replace(
            calibration_state,
            example=replace(calibration_state.example, partition=Partition.EVALUATION),
        )
        mixed_states = list(result.scored_states)
        mixed_states[calibration_index] = mixed_state
        mixed = replace(result, scored_states=tuple(mixed_states))
        changed_scale = replace(
            result,
            score_spec=replace(result.score_spec, error_scale=result.score_spec.error_scale * 2.0),
        )
        empty = replace(result, scored_states=(), labels=())

        # When / Then
        invalid = (missing, mixed, changed_scale, empty)
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ScoreArtifactError):
                        write_score_artifacts(Path(directory), candidate, checkpoint_digest="a" * 64)

    def test_nonfinite_prediction_aborts_scoring(self) -> None:
        # Given
        class NonfinitePredictor(RegimePredictor):
            def latent_mse(self, examples, requested_delta, effective_delta):
                values = super().latent_mse(examples, requested_delta, effective_delta)
                return (float("nan"), *values[1:])

        # When / Then
        with self.assertRaises(ScoreArtifactError):
            ExhaustiveScorer(NonfinitePredictor()).score(_examples())

    def test_partial_predictor_batch_aborts_scoring(self) -> None:
        # Given
        class PartialPredictor(RegimePredictor):
            def latent_mse(self, examples, requested_delta, effective_delta):
                return super().latent_mse(examples, requested_delta, effective_delta)[:-1]

        # When / Then
        with self.assertRaises(ScoreArtifactError):
            ExhaustiveScorer(PartialPredictor()).score(_examples())

    def test_incomplete_train_checkpoint_aborts_exhaustive_scoring(self) -> None:
        # Given
        model_config = fixture_jepa_config()
        phase = PhaseAConfig(steps=3, batch_size=2, device="cpu")
        trainer = TeacherForcedTrainer(JepaBackbone(model_config), phase.training_config(device="cpu"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, trainer, config_digest=phase.identity, grid_digest=phase.grid_digest)

            # When / Then
            with self.assertRaisesRegex(ValueError, "completed train-mode checkpoint"):
                score_fixture_checkpoint(checkpoint, phase, model_config)

    def test_resume_rejects_changed_checkpoint_binding(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "binding mismatch"):
                write_score_artifacts(
                    root,
                    result,
                    checkpoint_digest="b" * 64,
                    resume=True,
                    shard_size=4,
                )

    def test_resume_rejects_changed_shard_size_when_payloads_match(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=7)
            manifest_path = root / "manifest.json"
            before = manifest_path.read_bytes()

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "topology"):
                write_score_artifacts(root, result, checkpoint_digest="a" * 64, resume=True, shard_size=8)
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_resume_rejects_an_unlisted_stale_shard(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            manifest_path = root / "manifest.json"
            before = manifest_path.read_bytes()
            stale = root / "label_shards" / "controller-train" / "shard-999999.jsonl"
            stale.write_bytes(b"stale\n")

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "topology"):
                write_score_artifacts(root, result, checkpoint_digest="a" * 64, resume=True, shard_size=4)
            self.assertEqual(manifest_path.read_bytes(), before)
            self.assertTrue(stale.is_file())

    def test_resume_rejects_reordered_manifest_shards(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["shards"] = list(reversed(manifest["shards"]))
            before = canonical_json_bytes(manifest)
            manifest_path.write_bytes(before)

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "topology"):
                write_score_artifacts(root, result, checkpoint_digest="a" * 64, resume=True, shard_size=4)
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_resume_rejects_incomplete_manifest_shards(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["shards"] = manifest["shards"][:-1]
            before = canonical_json_bytes(manifest)
            manifest_path.write_bytes(before)

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "topology"):
                write_score_artifacts(root, result, checkpoint_digest="a" * 64, resume=True, shard_size=4)
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_validator_rejects_a_corrupted_shard(self) -> None:
        # Given
        result = ExhaustiveScorer(RegimePredictor()).score(_examples())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scores"
            write_score_artifacts(root, result, checkpoint_digest="a" * 64, shard_size=4)
            shard = next((root / "label_shards").rglob("*.jsonl"))
            shard.write_bytes(shard.read_bytes() + b"{}\n")

            # When / Then
            with self.assertRaisesRegex(ScoreArtifactError, "digest mismatch"):
                validate_score_artifacts(root)


if __name__ == "__main__":
    unittest.main()
