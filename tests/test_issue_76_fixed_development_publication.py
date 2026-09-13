import unittest

from scripts import publish_issue_76_fixed_development as publication
from tests.test_issue_76_fixed_development_scoring import selection_fixture


SPEC = {"seed": 760940001, "bootstrap_replicates": 10000, "confidence": .95}


class FixedDevelopmentPublicationTests(unittest.TestCase):
    def test_bootstrap_carries_seed_vector_and_equal_family_weights(self):
        rows = [{"base_cluster": str(i), "family": family, "differences": values, "prediction_failure": False}
                for i, (family, values) in enumerate((("A", [0., 0., 0.]), ("A", [0., 0., 0.]), ("B", [9., 10., 11.])))]
        result = publication.clustered_effect(rows, [1, 2, 3], SPEC)
        self.assertEqual(result["mean_difference"], 5.)
        self.assertEqual(result["per_seed"], {"1": 4.5, "2": 5., "3": 5.5})
        # Seed repeats are not independent bootstrap lineages.
        self.assertEqual(result["interval"], [5., 5.])
        self.assertEqual(result["assigned_lineages"], 3)
        self.assertEqual(result, publication.clustered_effect(rows, [1, 2, 3], SPEC))

    def test_unavailable_lineages_are_retained_and_missing_seeds_rejected(self):
        rows = [{"base_cluster": "a", "family": "A", "differences": [1., 1., 1.], "prediction_failure": False},
                {"base_cluster": "b", "family": "B", "differences": None, "prediction_failure": False}]
        result = publication.clustered_effect(rows, [1, 2, 3], SPEC)
        self.assertEqual(result["assigned_lineages"], 2)
        self.assertEqual(result["available_lineages"], 1)
        self.assertIsNone(result["interval"])
        rows[1]["differences"] = [2., 2.]
        with self.assertRaises(ValueError):
            publication.clustered_effect(rows, [1, 2, 3], SPEC)

    def test_contrasts_use_frozen_recipes_and_keep_reference_separate(self):
        inventory, results = selection_fixture()
        for entry in inventory["entries"]:
            entry["exposure_role"] = "model_selection"
        policy = publication.run.fit.policy_name(publication.run.fit.CONTINUOUS_PAIRS[0])
        choice = {"hybrid": {"recipe": "second", "policy": policy}, "pure": {"recipe": "first", "policy": policy}}
        for result in results:
            for row in result["rows"]:
                row["exposure_role"] = "model_selection"
                row["unchanged_carrier_reference"] = {"curves": {"11250": {"carrier_mse": 16., "failure": None}}}
        expected = {"pure": -1., "same_hybrid_continuous": 0., "unchanged": -12.}
        for name, difference in expected.items():
            rows = publication.contrast_rows(inventory, results, choice, name)
            summary = publication.clustered_effect(rows, inventory["seeds"], SPEC)
            self.assertEqual(summary["mean_difference"], difference)
        rows = [row for result in results if result["cell"]["recipe"] == "first" and not result["cell"]["pure"] for row in result["rows"]]
        summary = publication.curve_summary(rows, policy, 11250, "recursive", inventory["seeds"], ["A", "B"])
        self.assertEqual(summary["fields"]["carrier_mse"]["equal_family_seed_mean"], 5.)
        self.assertEqual(summary["fields"]["carrier_mse"]["available_metric_rows"], 9)


if __name__ == "__main__":
    unittest.main()
