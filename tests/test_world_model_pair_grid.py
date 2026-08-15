from __future__ import annotations

import math
import unittest

from world_model.model import Abstraction, PredictionPair
from world_model.training import TeacherForcedTrainer, TrainingConfig
from world_model.training import pair_grid


class LegacyContractCharacterizationTests(unittest.TestCase):
    def test_prediction_pair_identity_remains_the_legacy_tuple(self) -> None:
        # Given
        pair = PredictionPair(delta=5, abstraction=Abstraction.CONTINUOUS)

        # When
        identity = pair.identity

        # Then
        self.assertEqual(identity, (5, "continuous"))

    def test_scalar_training_imports_remain_public(self) -> None:
        # Given / When / Then
        self.assertEqual(TrainingConfig.__name__, "TrainingConfig")
        self.assertEqual(TeacherForcedTrainer.__name__, "TeacherForcedTrainer")


class PairGridConfigTests(unittest.TestCase):
    def test_default_grid_is_the_approved_continuous_temporal_grid(self) -> None:
        # Given / When
        config = pair_grid.PairGridConfig()

        # Then
        self.assertEqual(
            tuple(pair.identity for pair in config.pairs),
            ((1, "continuous"), (5, "continuous"), (15, "continuous")),
        )
        self.assertEqual(
            tuple((item.abstraction, item.reason) for item in config.exclusions),
            (
                (Abstraction.MICRO, "symbolic_supervision_unavailable"),
                (Abstraction.MACRO, "symbolic_supervision_unavailable"),
            ),
        )

    def test_grid_identity_is_stable(self) -> None:
        # Given / When
        identity = pair_grid.PairGridConfig().identity

        # Then
        self.assertEqual(
            identity,
            "0c12ab0e135a79e56da7c98e61e907f4850da057e317dbdd186b5b89f9e8b28d",
        )

    def test_invalid_grids_fail_closed(self) -> None:
        # Given
        invalid_grids = (
            (),
            (
                PredictionPair(1, Abstraction.CONTINUOUS),
                PredictionPair(1, Abstraction.CONTINUOUS),
            ),
            (PredictionPair(5, Abstraction.MICRO),),
            (PredictionPair(2, Abstraction.CONTINUOUS),),
        )

        # When / Then
        for pairs in invalid_grids:
            with self.subTest(pairs=pairs):
                with self.assertRaises(pair_grid.PairGridContractError):
                    pair_grid.PairGridConfig(pairs=pairs)


class PairMetricTests(unittest.TestCase):
    def test_temporal_extent_is_frame_count_minus_one(self) -> None:
        # Given / When / Then
        self.assertEqual(pair_grid.temporal_extent(16), 15)
        with self.assertRaises(pair_grid.PairGridContractError):
            pair_grid.temporal_extent(1)

    def test_ordinary_horizon_uses_requested_delta(self) -> None:
        # Given / When
        metric = pair_grid.build_pair_metric(
            PredictionPair(5, Abstraction.CONTINUOUS),
            frame_count=21,
            t=2,
            latent_mse=8.0,
        )

        # Then
        self.assertEqual(metric.requested_delta, 5)
        self.assertEqual(metric.effective_delta, 5)
        self.assertEqual(metric.duration_weight, 0.25)
        self.assertEqual(metric.weighted_prediction_error, 2.0)
        self.assertEqual(metric.compute_cost, 0.2)

    def test_terminal_horizon_is_clamped_for_scoring(self) -> None:
        # Given / When
        metric = pair_grid.build_pair_metric(
            PredictionPair(15, Abstraction.CONTINUOUS),
            frame_count=11,
            t=8,
            latent_mse=4.0,
        )

        # Then
        self.assertEqual(metric.effective_delta, 2)
        self.assertEqual(metric.duration_weight, 0.2)
        self.assertEqual(metric.weighted_prediction_error, 0.8)
        self.assertEqual(metric.compute_cost, 0.5)

    def test_single_transition_shot_clamps_every_pair_to_one(self) -> None:
        # Given / When
        metric = pair_grid.build_pair_metric(
            PredictionPair(15, Abstraction.CONTINUOUS),
            frame_count=2,
            t=0,
            latent_mse=3.0,
        )

        # Then
        self.assertEqual(metric.effective_delta, 1)
        self.assertEqual(metric.duration_weight, 1.0)
        self.assertEqual(metric.weighted_prediction_error, 3.0)

    def test_invalid_state_or_nonfinite_error_fails_closed(self) -> None:
        # Given
        pair = PredictionPair(1, Abstraction.CONTINUOUS)
        invalid_inputs = (
            (1, 0, 1.0),
            (3, -1, 1.0),
            (3, 2, 1.0),
            (3, 0, math.nan),
            (3, 0, math.inf),
        )

        # When / Then
        for frame_count, t, latent_mse in invalid_inputs:
            with self.subTest(frame_count=frame_count, t=t, latent_mse=latent_mse):
                with self.assertRaises(pair_grid.PairGridContractError):
                    pair_grid.build_pair_metric(pair, frame_count, t, latent_mse)

    def test_malformed_pair_boundary_raises_typed_error(self) -> None:
        # Given / When / Then
        with self.assertRaises(pair_grid.PairGridContractError):
            pair_grid.build_pair_metric(None, frame_count=3, t=0, latent_mse=1.0)


class ScoreSpecTests(unittest.TestCase):
    def test_calibration_uses_linearly_interpolated_p90(self) -> None:
        # Given / When
        spec = pair_grid.ScoreSpec.from_calibration(tuple(float(x) for x in range(11)))

        # Then
        self.assertEqual(spec.error_scale, 9.0)
        self.assertEqual(spec.lambda_cost, (0.0, 0.25, 1.0, 4.0))

    def test_calibration_scale_has_a_strict_floor(self) -> None:
        # Given / When
        spec = pair_grid.ScoreSpec.from_calibration((0.0, 0.0))

        # Then
        self.assertEqual(spec.error_scale, 1e-12)

    def test_empty_or_nonfinite_calibration_fails_closed(self) -> None:
        # Given / When / Then
        for values in ((), (math.nan,), (math.inf,)):
            with self.subTest(values=values):
                with self.assertRaises(pair_grid.PairGridContractError):
                    pair_grid.ScoreSpec.from_calibration(values)


class BestPairSelectionTests(unittest.TestCase):
    @staticmethod
    def _candidate(delta: int, weighted_error: float) -> pair_grid.PairMetric:
        effective_delta = delta
        return pair_grid.PairMetric(
            pair=PredictionPair(delta, Abstraction.CONTINUOUS),
            requested_delta=delta,
            effective_delta=effective_delta,
            duration_weight=1.0,
            latent_mse=weighted_error,
            weighted_prediction_error=weighted_error,
            compute_cost=1.0 / effective_delta,
        )

    def test_primary_and_sensitivity_scores_follow_the_frozen_objective(self) -> None:
        # Given
        candidates = (
            self._candidate(1, 0.1),
            self._candidate(5, 0.15),
            self._candidate(15, 0.3),
        )
        spec = pair_grid.ScoreSpec(error_scale=0.1)

        # When
        label = pair_grid.select_best_pair(candidates, spec)

        # Then
        self.assertEqual(label.selected_pair.identity, (5, "continuous"))
        self.assertAlmostEqual(label.primary_objective, 1.7)
        self.assertEqual(
            tuple(item.selected_pair.delta for item in label.sensitivity),
            (1, 1, 5, 5),
        )

    def test_objective_ties_inside_tolerance_use_lower_prediction_error(self) -> None:
        # Given
        candidates = (
            self._candidate(1, 0.0),
            self._candidate(5, 0.8000005),
        )

        # When
        label = pair_grid.select_best_pair(candidates, pair_grid.ScoreSpec(error_scale=1.0))

        # Then
        self.assertEqual(label.selected_pair.delta, 1)
        self.assertEqual(tuple(pair.delta for pair in label.tied_pairs), (1, 5))

    def test_objective_values_outside_tolerance_are_not_tied(self) -> None:
        # Given
        candidates = (
            self._candidate(1, 0.0),
            self._candidate(5, 0.800002),
        )

        # When
        label = pair_grid.select_best_pair(candidates, pair_grid.ScoreSpec(error_scale=1.0))

        # Then
        self.assertEqual(tuple(pair.delta for pair in label.tied_pairs), (1,))

    def test_empty_duplicate_nonfinite_and_no_valid_candidates_fail_closed(self) -> None:
        # Given
        valid = self._candidate(1, 0.1)

        # When / Then
        with self.assertRaises(pair_grid.NoValidPairError):
            pair_grid.select_best_pair((), pair_grid.ScoreSpec(error_scale=1.0))
        with self.assertRaises(pair_grid.PairGridContractError):
            pair_grid.select_best_pair((valid, valid), pair_grid.ScoreSpec(error_scale=1.0))
        with self.assertRaises(pair_grid.PairGridContractError):
            pair_grid.PairMetric(
                pair=PredictionPair(5, Abstraction.CONTINUOUS),
                requested_delta=5,
                effective_delta=5,
                duration_weight=1.0,
                latent_mse=math.nan,
                weighted_prediction_error=math.nan,
                compute_cost=0.2,
            )
        unavailable = tuple(
            pair_grid.UnavailablePairMetric(pair=pair, reason="prediction_unavailable")
            for pair in pair_grid.PairGridConfig().pairs
        )
        with self.assertRaises(pair_grid.NoValidPairError):
            pair_grid.select_best_pair(unavailable, pair_grid.ScoreSpec(error_scale=1.0))

    def test_malformed_candidate_boundary_raises_typed_error(self) -> None:
        # Given / When / Then
        with self.assertRaises(pair_grid.PairGridContractError):
            pair_grid.select_best_pair((None,), pair_grid.ScoreSpec(error_scale=1.0))

    def test_best_pair_label_rejects_inconsistent_direct_construction(self) -> None:
        # Given
        pair = PredictionPair(1, Abstraction.CONTINUOUS)

        # When / Then
        with self.assertRaises(pair_grid.PairGridContractError):
            pair_grid.BestPairLabel(
                metrics=(),
                selected_pair=pair,
                tied_pairs=(pair,),
                primary_objective=1.0,
                sensitivity=(),
            )


if __name__ == "__main__":
    unittest.main()
