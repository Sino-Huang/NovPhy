import unittest

from scripts.issue_76_replay_headroom import endpoint_counts, summarize


class HeadroomTests(unittest.TestCase):
    def test_zero_pigs_without_clear_is_not_a_win(self):
        sample = {"entities": [{"scenario_object_id": "pig:0000", "lifecycle": "destroyed"}]}
        for terminal in (None, {"reason": "level_fail"}):
            row = endpoint_counts(sample, terminal)
            self.assertEqual(row["ranking_cost"], 1e9)
            self.assertIsNone(row["settled_count_cost"])
            self.assertFalse(row["native_level_clear"])
        stable = endpoint_counts(sample, {"reason": "stable_entered"})
        self.assertEqual(stable["ranking_cost"], 0)
        self.assertFalse(stable["native_level_clear"])

    def test_reference_and_penalty_do_not_create_count_headroom(self):
        rows = [{"original_action_reference": i == 0, "candidate_ordinal": i,
                 "settled_count_cost": cost, "ranking_cost": 1e9 if cost is None else cost,
                 "ranking_failure": cost is None, "native_level_clear": False}
                for i, cost in enumerate((0, 1001, None, 1001))]
        result = summarize(rows)
        self.assertEqual(result["candidate_count"], 3)
        self.assertFalse(result["count_discriminating"])
        self.assertEqual(result["settled_count_range"], 0)
        self.assertEqual(result["best_action_ordinals"], [1, 3])
        self.assertFalse(result["all_ranking_costs_tied"])
