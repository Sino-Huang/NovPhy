from __future__ import annotations

import unittest

from scripts.run_cohort_v2_visual_parser import (
    _metric_deltas,
    _paired_effects,
    _parser,
)


class CohortV2VisualParserRunnerTests(unittest.TestCase):
    def test_paired_stress_rule_uses_six_complete_rollouts_and_upper_bounds(self):
        candidate = [
            {
                "attempt_id": f"attempt-{index}",
                "mean_endpoint_prediction_error": 0.1,
                "mean_endpoint_violation_rate": 0.0,
            }
            for index in range(6)
        ]
        reference = [
            {
                "attempt_id": f"attempt-{index}",
                "mean_endpoint_prediction_error": 0.1,
                "mean_endpoint_violation_rate": 0.0,
            }
            for index in range(6)
        ]
        comparison = {
            "budget": 10.0,
            "endpoint_bootstrap_seed": 20261026,
            "violation_bootstrap_seed": 20261027,
            "practical_effect_threshold_absolute_endpoint_error_reduction": 0.01,
            "physical_violation_margin": 0.0,
        }

        result = _paired_effects(candidate, reference, (comparison,))[0]

        self.assertEqual(result["endpoint"]["mean"], 0.0)
        self.assertEqual(result["violation"]["mean"], 0.0)
        self.assertTrue(result["budget_rule_passed"])
        self.assertEqual(len(result["endpoint"]["paired_rollout_values"]), 6)

    def test_visual_minus_feature_deltas_keep_calibration_and_predicate_metrics(self):
        keys = {
            "agreement": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "brier_score": 0.2,
            "negative_log_likelihood": 0.3,
            "expected_calibration_error_10_bin": 0.1,
        }
        visual = {name: dict(keys) for name in (
            "contact", "supports", "steady-state", "structure-unstable"
        )}
        feature = {
            name: {key: value - 0.1 for key, value in keys.items()}
            for name in visual
        }

        result = _metric_deltas(visual, feature)

        self.assertAlmostEqual(result["contact"]["f1"], 0.1)
        self.assertAlmostEqual(
            result["structure-unstable"]["negative_log_likelihood"], 0.1
        )

    def test_cli_defaults_to_visible_progress_and_separate_aligned_release(self):
        args = _parser().parse_args([])

        self.assertGreater(args.score_log_every, 0)
        self.assertEqual(
            str(args.aligned_root),
            ".local-artifacts/issue-59-aligned-observation-release",
        )


if __name__ == "__main__":
    unittest.main()
