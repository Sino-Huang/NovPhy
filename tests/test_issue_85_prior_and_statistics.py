"""Focused tests for the issue-85 statistics and disposition logic.

Covers the pre-declared no-model ordinal prior fallback, the engine-truth
pilot gate, the paired descriptive bootstrap determinism and pairing
semantics, the Phase-C proxy-vs-engine cross-check, and the pre-declared
disposition mappings. No GPU work and no engine execution is performed.
"""
import unittest

import numpy as np

from world_model.training import engine_outcome_reactive_diagnostic as module


class PriorFallbackTests(unittest.TestCase):
    def test_ordinal_eight_when_admissible(self) -> None:
        self.assertEqual(module.prior_fallback_choice([0, 3, 8, 12]), 8)

    def test_nearest_with_ties_to_the_lower_ordinal(self) -> None:
        self.assertEqual(module.prior_fallback_choice([0, 1, 2, 3, 5, 7, 9, 10, 11]), 7)
        self.assertEqual(module.prior_fallback_choice([9, 10, 11]), 9)
        self.assertEqual(module.prior_fallback_choice([0, 1, 2]), 2)
        self.assertEqual(module.prior_fallback_choice([12]), 12)

    def test_never_none_on_a_nonempty_inventory(self) -> None:
        for ordinals in ([8], [0], [1, 5], list(range(13)), [7, 9]):
            self.assertIsNotNone(module.prior_fallback_choice(ordinals))

    def test_empty_or_duplicated_inventories_are_typed_errors(self) -> None:
        self.assertRaises(ValueError, module.prior_fallback_choice, [])
        self.assertRaises(ValueError, module.prior_fallback_choice, [8, 8])


def gate_row(identity, failure=None, success=None):
    return {"identity": identity, "failure": failure, "success": success}


class GateTests(unittest.TestCase):
    def test_gate_passes_at_the_floor_with_enough_valid_cells(self) -> None:
        rows = [gate_row(f"cell-{index}", success=index < 6) for index in range(48)]
        gate = module.evaluate_gate(rows)
        self.assertAlmostEqual(gate["prevalence"], 6 / 48)
        self.assertGreaterEqual(gate["prevalence"], module.PREVALENCE_FLOOR)
        self.assertTrue(gate["passed"])

    def test_gate_fails_below_the_floor(self) -> None:
        rows = [gate_row(f"cell-{index}", success=index < 1) for index in range(48)]
        gate = module.evaluate_gate(rows)
        self.assertLess(gate["prevalence"], module.PREVALENCE_FLOOR)
        self.assertFalse(gate["passed"])

    def test_typed_failures_are_excluded_and_reported(self) -> None:
        rows = [gate_row(f"cell-{index}", success=True) for index in range(20)]
        rows += [gate_row(f"cell-typed-{index}", failure="decision_failure: x")
                 for index in range(28)]
        gate = module.evaluate_gate(rows)
        self.assertEqual(gate["valid_executions"], 20)
        self.assertEqual(gate["successes"], 20)
        self.assertEqual(gate["prevalence"], 1.0)
        self.assertFalse(gate["passed"])  # below the minimum valid executions
        self.assertEqual(len(gate["typed_failures"]), 28)


class PairedIntervalTests(unittest.TestCase):
    def test_mean_and_interval_recompute_deterministically(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        interval = module.paired_interval(values)
        self.assertEqual(interval["mean"], 2.5)
        rng = np.random.default_rng(module.BOOTSTRAP_SEED)
        draws = np.asarray(values)[rng.integers(
            len(values), size=(module.BOOTSTRAP_DRAWS, len(values)))].mean(1)
        expected = np.quantile(draws, [.025, .975]).tolist()
        self.assertEqual(interval["descriptive_95_percent_interval"], expected)

    def test_empty_input_has_no_interval(self) -> None:
        self.assertIsNone(module.paired_interval([]))


def per_state(seeds, states, values):
    """values: (system, metric) -> state -> per-seed value or None."""
    return {str(seed): {system: {state: {"success": values.get(("success", system), {})
                                               .get((state, seed)),
                                         "regret": values.get(("regret", system), {})
                                               .get((state, seed))}
                                for state in states}
                       for system in module.SYSTEMS}
            for seed in seeds}


class ContrastTests(unittest.TestCase):
    SEEDS = (20260908, 20260909)
    STATES = ("s-a", "s-b")

    def test_differences_are_reference_minus_tested_averaged_over_seeds(self) -> None:
        table = per_state(self.SEEDS, self.STATES, {
            ("success", "hybrid-fixed-h1"): {("s-a", 20260908): True, ("s-a", 20260909): True,
                                             ("s-b", 20260908): False, ("s-b", 20260909): True},
            ("success", "continuous-fixed-h5"): {("s-a", 20260908): False, ("s-a", 20260909): True,
                                                 ("s-b", 20260908): False, ("s-b", 20260909): False},
        })
        contrasts = module.summarize_contrasts(table, self.SEEDS)
        success = next(contrast for contrast in contrasts
                       if contrast["reference"] == "continuous-fixed-h5"
                       and contrast["metric"] == "success")
        # positive favors hybrid: s-a: (1-0 + 1-1)/2 = +0.5 ; s-b: (0-0 + 1-0)/2
        # = +0.5 -> mean +0.5 where hybrid strictly dominates on 3 of 4 pairs.
        self.assertEqual(success["descriptive"]["mean"], 0.5)
        self.assertEqual(success["paired_states"], 2)
        self.assertEqual(success["label"], "DESCRIPTIVE")

    def test_states_with_missing_outcomes_are_dropped_and_reported(self) -> None:
        table = per_state(self.SEEDS, self.STATES, {
            ("success", "hybrid-fixed-h1"): {("s-a", 20260908): True, ("s-a", 20260909): True},
            ("success", "no-model-ordinal-prior"): {("s-a", 20260908): True,
                                                    ("s-a", 20260909): True},
        })
        contrasts = module.summarize_contrasts(table, self.SEEDS)
        prior = next(contrast for contrast in contrasts
                     if contrast["reference"] == "no-model-ordinal-prior")
        self.assertEqual(prior["paired_states"], 1)
        self.assertEqual(prior["descriptive"]["mean"], 0.0)


class DispositionTests(unittest.TestCase):
    def test_supported_requires_interval_exclusion_above_zero(self) -> None:
        supported = {"descriptive": {"mean": 0.2,
                                     "descriptive_95_percent_interval": [0.05, 0.4]}}
        self.assertEqual(module.disposition_for_primary(supported)[0], "supported")
        crossing = {"descriptive": {"mean": 0.2,
                                    "descriptive_95_percent_interval": [-0.05, 0.4]}}
        self.assertEqual(module.disposition_for_primary(crossing)[0],
                         "not_supported_by_this_experiment")
        negative = {"descriptive": {"mean": -0.1,
                                    "descriptive_95_percent_interval": [-0.3, 0.05]}}
        self.assertEqual(module.disposition_for_primary(negative)[0],
                         "not_supported_by_this_experiment")
        self.assertEqual(module.disposition_for_primary({"descriptive": None})[0],
                         "not_supported_by_this_experiment")

    def test_tokens_are_exactly_the_vocabulary(self) -> None:
        allowed = {"supported", "not_supported_by_this_experiment",
                   "readiness_or_precision_insufficient"}
        self.assertIn(module.disposition_for_primary({"descriptive": None})[0], allowed)


class ProxyCrossCheckTests(unittest.TestCase):
    def row(self, system, seed, state, success, cost):
        return {"system": system, "seed": seed, "state": state,
                "success": success, "realized_count_cost": cost,
                "identity": f"{system}--seed{seed}--{state}", "failure": None}

    def test_quadrant_counts_and_agreement(self) -> None:
        threshold = module.PROXY_THRESHOLD
        rows = [
            self.row("hybrid-fixed-h1", 1, "s-a", True, threshold - 1),   # both
            self.row("hybrid-fixed-h1", 1, "s-b", False, threshold + 1),  # both
            self.row("continuous-fixed-h1", 1, "s-a", False, threshold - 1),  # proxy only
            self.row("continuous-fixed-h5", 1, "s-a", True, threshold + 1),   # engine only
            self.row("continuous-fixed-h5", 1, "s-b", None, None),        # typed failure
        ]
        table = module.proxy_cross_check(rows)
        self.assertEqual(table["cells_compared"], 4)
        self.assertEqual(table["cells_excluded"], 1)
        self.assertEqual(table["counts"], {"both_success": 1, "proxy_only": 1,
                                           "engine_only": 1, "both_failure": 1})
        self.assertAlmostEqual(table["agreement"], 0.5)
        self.assertEqual(table["per_system"]["continuous-fixed-h5"]["cells"], 1)
        self.assertEqual(table["per_system"]["hybrid-fixed-h1"]["cells"], 2)
        self.assertEqual(table["per_system"]["hybrid-fixed-h1"]["both_success"], 1)
        self.assertEqual(table["proxy_threshold"], threshold)

    def test_no_compared_cells_yields_no_agreement(self) -> None:
        table = module.proxy_cross_check([self.row("hybrid-fixed-h1", 1, "s-a", None, None)])
        self.assertEqual(table["cells_compared"], 0)
        self.assertIsNone(table["agreement"])


if __name__ == "__main__":
    unittest.main()
