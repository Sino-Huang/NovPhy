"""Issue-109 capture smoke report: frozen statistics, guards and verdict boundaries."""
import copy
import unittest

from scripts import capture_smoke_report as report
from scripts import run_capture_reliability_smoke as runner

AUDIT = copy.deepcopy(runner.AUDIT)
BOOT = AUDIT["bootstrap"]
GUARDS = AUDIT["precision_guards"]


def contrast(difference, low, high, **extra):
    value = {"computable": True, "difference": difference, "ci_low": low, "ci_high": high,
             "n_target": 40, "members_target": 10, "usable_draws": BOOT["draws"],
             "draws": BOOT["draws"], "guards_tripped": []}
    value.update(extra)
    return value


def rows(target_members, target_per_member, reference_members=16, target_value=1.0):
    out = []
    for m in range(reference_members):
        member = f"issue-77-n1-{m + 1:03d}"
        out += [(member, False, 0.0)] * 5
        if m < target_members:
            out += [(member, True, target_value)] * target_per_member
    return out


class FailureRateGateTests(unittest.TestCase):
    def test_rate_exactly_at_gate_is_not_supported(self):
        rate, token = report.failure_rate_token(5, 100, AUDIT["failure_rate_gate"])
        self.assertEqual(rate, 0.05)
        self.assertEqual(token, "not_supported_by_this_experiment")

    def test_rate_below_gate_is_supported(self):
        self.assertEqual(report.failure_rate_token(4, 100, AUDIT["failure_rate_gate"])[1], "supported")


class PartAVerdictTests(unittest.TestCase):
    def test_bound_equal_to_margin_is_bounded(self):
        self.assertEqual(report.part_a_verdict(5 / 100, None, AUDIT), "bias_bounded_below_margin")

    def test_bound_above_margin_without_contrast_is_indeterminate(self):
        self.assertEqual(report.part_a_verdict(6 / 100, None, AUDIT), "indeterminate")

    def test_correlated_direct_contrast_overrides_small_bound(self):
        contrasts = {"pig_removed": contrast(0.2, 0.05, 0.35)}
        self.assertEqual(report.part_a_verdict(0.01, contrasts, AUDIT), "outcome_correlated")

    def test_interval_touching_zero_does_not_correlate(self):
        contrasts = {"pig_removed": contrast(0.3, 0.0, 0.6)}
        self.assertEqual(report.part_a_verdict(0.01, contrasts, AUDIT), "bias_bounded_below_margin")

    def test_tokens_follow_frozen_dispositions(self):
        disposition = AUDIT["part_a"]["disposition"]
        self.assertEqual(report.token_for("bias_bounded_below_margin", disposition), "supported")
        self.assertEqual(report.token_for("outcome_correlated", disposition),
                         "not_supported_by_this_experiment")
        self.assertEqual(report.token_for("indeterminate", disposition),
                         "readiness_or_precision_insufficient")


class PartBVerdictTests(unittest.TestCase):
    def test_interval_touching_zero_is_indeterminate(self):
        contrasts = {name: contrast(0.0, -0.05, 0.05) for name in report.OUTCOMES}
        contrasts["native_clear"] = contrast(0.2, 0.0, 0.4)
        self.assertEqual(report.part_b_verdict(contrasts, AUDIT), "indeterminate")

    def test_interval_excluding_zero_with_margin_difference_is_correlated(self):
        contrasts = {name: contrast(0.0, -0.05, 0.05) for name in report.OUTCOMES}
        contrasts["native_clear"] = contrast(0.1, 0.01, 0.2)
        self.assertEqual(report.part_b_verdict(contrasts, AUDIT), "outcome_correlated")
        self.assertEqual(report.token_for("outcome_correlated", AUDIT["part_b"]["disposition"]), "supported")

    def test_blind_requires_open_interval_inside_margin(self):
        inside = {name: contrast(0.0, -0.09, 0.09) for name in report.OUTCOMES}
        self.assertEqual(report.part_b_verdict(inside, AUDIT), "outcome_blind")
        touching = dict(inside, right_censored=contrast(0.0, -0.05, AUDIT["contrast_margin"]))
        self.assertEqual(report.part_b_verdict(touching, AUDIT), "indeterminate")

    def test_tripped_guard_blocks_both_verdicts(self):
        strong = {name: contrast(0.5, 0.3, 0.7) for name in report.OUTCOMES}
        strong["pig_removed"]["guards_tripped"] = ["min_target_slots"]
        self.assertEqual(report.part_b_verdict(strong, AUDIT), "indeterminate")


class ClusteredContrastTests(unittest.TestCase):
    def test_point_estimate_and_interval(self):
        value = report.clustered_contrast(rows(8, 3), BOOT)
        self.assertTrue(value["computable"])
        self.assertEqual((value["n_target"], value["members_target"]), (24, 8))
        self.assertAlmostEqual(value["difference"], 1.0)
        self.assertAlmostEqual(value["ci_low"], 1.0)
        self.assertEqual(value, report.clustered_contrast(rows(8, 3), BOOT))

    def test_empty_side_is_not_computable(self):
        value = report.clustered_contrast([("issue-77-n1-001", False, 1.0)], BOOT)
        self.assertFalse(value["computable"])
        self.assertEqual(report.guards_tripped(value, GUARDS), ["contrast_unavailable"])

    def test_target_slot_guard_boundary(self):
        at_floor = rows(10, 2)  # 20 target slots over 10 members
        self.assertEqual(report.guards_tripped(report.clustered_contrast(at_floor, BOOT), GUARDS), [])
        below = list(at_floor)
        below.remove(next(row for row in below if row[1]))
        self.assertEqual(report.guards_tripped(report.clustered_contrast(below, BOOT), GUARDS),
                         ["min_target_slots"])

    def test_target_member_guard_boundary(self):
        self.assertEqual(report.guards_tripped(report.clustered_contrast(rows(8, 3), BOOT), GUARDS), [])
        self.assertEqual(report.guards_tripped(report.clustered_contrast(rows(7, 3), BOOT), GUARDS),
                         ["min_target_members"])

    def test_usable_draw_guard_trips_when_target_is_rare(self):
        value = report.clustered_contrast(rows(1, 25, reference_members=30), BOOT)
        self.assertLess(value["usable_draws"] / value["draws"], GUARDS["min_usable_draws_fraction"])
        self.assertIn("min_usable_draws_fraction", report.guards_tripped(value, GUARDS))


class MechanismTests(unittest.TestCase):
    def test_same_port_partner_window(self):
        cells = {"a": (0.0, {1, 2, 3}), "b": (5.0, {3, 4, 5}), "c": (11.0, {5}), "d": (1.0, {9})}
        self.assertEqual(report._partners(cells), {"a", "b"})

    def test_member_parse_rejects_foreign_identity(self):
        self.assertEqual(report.member_of("issue-77-n1-014-a07"), "issue-77-n1-014")
        with self.assertRaises(ValueError):
            report.member_of("oracle--issue-77-n1-014-a07")


if __name__ == "__main__":
    unittest.main()
