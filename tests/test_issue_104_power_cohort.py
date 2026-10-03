"""#104 joint cohort freeze: depth rule, #113 window rule check, hard-link byte accounting."""
import os
from pathlib import Path
import tempfile
import unittest

from scripts import capture_pipeline_v2 as pipeline
from scripts import prepare_power_cohort as cohort
from scripts import run_power_cohort_smoke as smoke


def power(grid_need, grid_share, secondary):
    """Minimal power block: one contrast per inventory with a design requirement at both deltas."""
    def contrast(need_02, need_05):
        return {"J-F*": {"required_mixed_members": {"0.02": {"design": need_02}, "0.05": {"design": need_05}}}}
    contrasts = {"grid": contrast(grid_need, 1)}
    shares = {"grid": {"conservative": grid_share}}
    for inventory, (need, share) in secondary.items():
        contrasts[inventory] = contrast(1, need)
        shares[inventory] = {"conservative": share}
    return {"contrasts": contrasts, "mixed_shares": shares}


class DepthRuleTests(unittest.TestCase):
    def test_depth_divides_by_both_families_and_rounds_up_to_twenty(self):
        value = cohort.depth(power(208, 0.75, {"offset": (32, 0.5455), "angle": (55, 0.5), "power": (53, 0.4545)}))
        self.assertEqual(value["grid"]["levels_per_family"], 140)       # 208 / 1.5 = 138.7 -> 140
        self.assertEqual(value["union_levels_per_family"], 60)          # max(30, 55, 59) -> 60

    def test_depth_is_capped_at_the_roster(self):
        value = cohort.depth(power(900, 0.5, {"offset": (900, 0.5), "angle": (1, 1.0), "power": (1, 1.0)}))
        self.assertEqual(value["grid"]["levels_per_family"], 200)
        self.assertEqual(value["union_levels_per_family"], 200)


def manifest(last, terminal=None, failure=None, window=(30000, 2500)):
    value = {"first_fixed_step": 30000, "last_fixed_step": 30000 + last, "terminal_evidence": terminal,
             "failure": failure}
    if window is not None:
        value["maximum_shot_steps"], value["rest_tail_steps"] = window
    return value


def events(*pairs):
    return [(30000 + step, kind, ()) for step, kind in pairs]


class WindowRuleTests(unittest.TestCase):
    def check(self, *args, **kwargs):
        return smoke.window_check(*args, **kwargs)[0]

    def test_tail_must_end_exactly_rest_tail_steps_after_the_last_rest(self):
        rested = events((100, "bird_launched"), (9000, "stable_entered"))
        self.assertTrue(self.check(manifest(11500, {"reason": "rest_tail_complete"}), rested))
        self.assertFalse(self.check(manifest(11450, {"reason": "rest_tail_complete"}), rested))

    def test_rest_tail_uses_the_last_rest_after_an_exit(self):
        rested = events((9000, "stable_entered"), (9500, "stable_exited"), (12000, "stable_entered"))
        self.assertTrue(self.check(manifest(14500, {"reason": "rest_tail_complete"}), rested))
        self.assertFalse(self.check(manifest(11500, {"reason": "rest_tail_complete"}), rested))

    def test_censoring_is_at_the_cap_or_after_a_tail_cancelled_past_it(self):
        self.assertTrue(self.check(manifest(30000, failure="native_time_window_limit"), events()))
        cancelled = events((29000, "stable_entered"), (30120, "stable_exited"))
        self.assertTrue(self.check(manifest(30121, failure="native_time_window_limit"), cancelled))
        self.assertFalse(self.check(manifest(30121, failure="native_time_window_limit"), events()))

    def test_old_stable_terminal_and_undeclared_window_violate_the_rule(self):
        self.assertFalse(self.check(manifest(9000, {"reason": "stable_entered"}), events((9000, "stable_entered"))))
        self.assertFalse(self.check(manifest(30000, failure="native_time_window_limit", window=None), events()))

    def test_clear_or_fail_may_end_a_tail(self):
        self.assertTrue(self.check(manifest(9083, {"reason": "level_fail"}), events((9000, "stable_entered"))))


class ByteAccountingTests(unittest.TestCase):
    def test_player_hard_link_is_not_charged_to_the_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "player").mkdir()
            (root / "attempt").mkdir()
            (root / "player" / "engine.so").write_bytes(b"x" * 1000)
            os.link(root / "player" / "engine.so", root / "attempt" / "engine.so")
            (root / "attempt" / "trace.json").write_bytes(b"y" * 300)
            seen = set()
            self.assertEqual(pipeline._attempt_bytes(root / "player", seen), 500)   # size / link count
            self.assertEqual(pipeline._attempt_bytes(root / "attempt", seen), 300)


if __name__ == "__main__":
    unittest.main()
