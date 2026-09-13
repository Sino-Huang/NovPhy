import unittest

from scripts.audit_issue_76_action_support import assigned_support, observed_training_support
from world_model.data.corrective_ranking_cohort import action_bounds
from world_model.training.native_history_data import action_context


class ActionSupportTests(unittest.TestCase):
    def setUp(self):
        self.actions = [dict(drag_x=x, drag_y=y, release_time_ms=600, tap_time_ms=0)
                        for x, y in ((-80, 10), (-60, 45), (-10, 80))]
        self.members = [dict(identity="a", generator_family="A", exposure_role="training", actions=self.actions[:1]),
                        dict(identity="b", generator_family="B", exposure_role="training", actions=self.actions)]

    def test_later_shot_support_is_not_first_shot_support(self):
        result = assigned_support(self.members, action_bounds())
        self.assertEqual(result["grid_actions_seen_at_first_shot"], 0)
        self.assertEqual(result["grid_actions_seen_at_any_shot"], 1)
        self.assertEqual(result["grid"][8]["assigned_all_shots"], 1)
        self.assertEqual(result["grid"][8]["assigned_first_shots"], 0)

    def test_observed_counts_retain_unusable_and_unattempted_assignments(self):
        entries = [dict(member_identity="a", exposure_role="training", usable=False),
                   dict(member_identity="b", exposure_role="training", usable=True)]
        shard = {"segment_ranges": [{"action": action_context(action)} for action in self.actions[:2]]}
        observed = observed_training_support(self.members, entries, lambda entry: shard, lambda: None)
        self.assertEqual(observed["assigned_members"], 2)
        self.assertEqual(len(observed["member_inventory"][0]["observed_actions"]), 0)
        self.assertEqual(sum(row["count"] for row in observed["observed_all_actions"]), 2)
        entries[1]["exposure_role"] = "model_selection"
        with self.assertRaises(ValueError):
            observed_training_support(self.members, entries, lambda entry: shard, lambda: None)


if __name__ == "__main__":
    unittest.main()
