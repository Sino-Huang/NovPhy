import unittest
from scripts.publish_issue_76_replay_rankings import ranking


class RankingTests(unittest.TestCase):
    def targets(self, costs):
        return [{"ranking_cost": cost, "ranking_failure": cost == 1e9,
                 "settled_count_cost": None if cost == 1e9 else cost,
                 "candidate_ordinal": i + 1, "native_level_clear": False}
                for i, cost in enumerate(costs)]

    def test_penalty_and_count_regret_stay_separate(self):
        targets = self.targets([1005, 1006, 1e9])
        row = ranking([1., 0., 2.], targets)
        self.assertEqual(row["absolute_settled_count_regret"], 1)
        self.assertLess(row["regret"], 1e-8)
        failed = ranking([2., 1., 0.], targets)
        self.assertIsNone(failed["absolute_settled_count_regret"])
        self.assertEqual(failed["regret"], 1)

    def test_all_failed_is_not_topk_success_and_ties_keep_order(self):
        row = ranking([0., 0.], self.targets([1e9, 1e9]))
        self.assertFalse(row["top1"])
        self.assertFalse(row["top3"])
        self.assertEqual(row["selected_action_ordinal"], 1)
        self.assertTrue(row["predicted_all_tied"])
