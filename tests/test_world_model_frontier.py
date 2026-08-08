import unittest

from world_model.training.frontier import FrontierError, analyze_frontier, pareto_frontier


class FrontierTests(unittest.TestCase):
    def test_dominance_and_ties(self):
        self.assertEqual(pareto_frontier([
            {"delta": 1, "weighted_prediction_error": 1.0, "compute_cost": 1.0},
            {"delta": 5, "weighted_prediction_error": 2.0, "compute_cost": 2.0},
        ]), [1])
        self.assertEqual(pareto_frontier([
            {"delta": 1, "weighted_prediction_error": 1.0, "compute_cost": 1.0},
            {"delta": 5, "weighted_prediction_error": 1.0, "compute_cost": 1.0},
        ]), [1, 5])

    def test_bootstrap_is_deterministic_and_requires_states(self):
        rows = [{"regime": "calm" if i % 2 else "active", "delta": delta,
                 "weighted_prediction_error": error, "compute_cost": cost}
                for i in range(100) for delta, error, cost in ((1, 1.0, 1.0), (5, .5, .5), (15, .2, .2))]
        self.assertEqual(analyze_frontier(rows, seed=4), analyze_frontier(rows, seed=4))
        self.assertEqual(analyze_frontier(rows, seed=4)["bootstrap"]["replicates"], 1000)
        with self.assertRaises(FrontierError):
            analyze_frontier(rows[:99])

    def test_nonfinite_metric_is_rejected(self):
        rows = [{"delta": 1, "weighted_prediction_error": float("nan"), "compute_cost": 1.0}] * 100
        with self.assertRaises(FrontierError):
            analyze_frontier(rows)


if __name__ == "__main__":
    unittest.main()
